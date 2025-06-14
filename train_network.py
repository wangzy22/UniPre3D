"""
Training script for 3D Gaussian Splatting model with multi-modal fusion support.
This implementation supports both single-GPU and distributed training.

Key features:
- Multi-modal fusion between 2D and 3D features
- EMA (Exponential Moving Average) support
- Distributed training support
- Flexible validation and checkpointing
"""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import hydra
from omegaconf import DictConfig, open_dict
from ema_pytorch import EMA
import torch.distributed as dist
import pointcept.utils.comm as comm
from pointcept.engines.launch import launch
from pointcept.engines.defaults import create_ddp_model, worker_init_fn
from pointcept.datasets import point_collate_fn

from model.gaussian_predictor import GaussianSplatPredictor
from dataset.dataset_factory import get_dataset
from gaussian_renderer import render_predicted
from eval import evaluate_dataset
from utils.general_utils import safe_state, to_device
from utils.loss_utils import l1_loss, l2_loss, focal_l2_loss
import lpips as lpips_lib
from typing import Dict, Any, List, Tuple
from functools import partial
import multiprocessing

from logger import Logger


class DataManager:
    """Manages all data loading and processing operations"""

    def __init__(self, cfg: DictConfig, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.setup_dataloaders()

    def setup_dataloaders(self) -> None:
        """Initialize all data loaders"""
        self.dataset = get_dataset(self.cfg, "train", device=self.device)
        self.val_dataset = get_dataset(self.cfg, "val", device=self.device)
        self.test_dataset = get_dataset(self.cfg, "test", device=self.device)

        # Setup distributed sampling
        self.train_sampler = (
            torch.utils.data.distributed.DistributedSampler(self.dataset)
            if comm.get_world_size() > 1
            else None
        )
        self.val_sampler = (
            torch.utils.data.distributed.DistributedSampler(self.val_dataset)
            if comm.get_world_size() > 1
            else None
        )

        # Calculate batch size per GPU
        self.bs_per_gpu = (
            self.cfg.opt.batch_size
            if not self.cfg.general.multiple_gpu
            else self.cfg.opt.batch_size // len(self.cfg.general.device)
        )

        self.init_fn = self._get_worker_init_fn()
        self._create_dataloaders()

    def _get_worker_init_fn(self):
        """Get worker initialization function for data loading"""
        if self.cfg.general.random_seed is not None and self.cfg.general.multiple_gpu:
            return partial(
                worker_init_fn,
                num_workers=len(self.cfg.general.device) * 4,
                rank=comm.get_rank(),
                seed=self.cfg.general.random_seed,
            )
        return None

    def _create_dataloaders(self) -> None:
        """Create data loaders based on model type"""
        common_loader_params = {
            "num_workers": 0,
            "collate_fn": point_collate_fn if self.cfg.opt.level == "scene" else None,
        }

        if self.cfg.opt.level == "scene":
            self.train_loader = DataLoader(
                self.dataset,
                batch_size=self.bs_per_gpu,
                shuffle=(self.train_sampler is None),
                drop_last=True,
                sampler=self.train_sampler,
                **common_loader_params,
            )

            self.val_loader = DataLoader(
                self.val_dataset,
                batch_size=self.bs_per_gpu,
                shuffle=False,
                drop_last=False,
                sampler=self.val_sampler,
                worker_init_fn=self.init_fn,
                **common_loader_params,
            )
        else:
            self.train_loader = DataLoader(
                self.dataset,
                batch_size=self.cfg.opt.batch_size,
                shuffle=True,
                drop_last=True,
                **common_loader_params,
            )

            self.val_loader = DataLoader(
                self.val_dataset,
                batch_size=self.cfg.opt.batch_size,
                shuffle=True,
                **common_loader_params,
            )

        # Test loader configuration remains same for both cases
        self.test_loader = DataLoader(
            self.test_dataset, batch_size=1, shuffle=True, **common_loader_params
        )


class ModelManager:
    """Manages model creation, optimization and checkpointing"""

    def __init__(self, cfg: DictConfig, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.model, self.optimizer, self.scheduler = self._create_model_and_optimizer()
        self.model = self.model.to(device)
        self.setup_distributed()
        self.setup_ema()

    def _create_model_and_optimizer(
        self,
    ) -> Tuple[torch.nn.Module, torch.optim.Optimizer]:
        """Create and initialize model and optimizer"""

        model_class = GaussianSplatPredictor
        model = model_class(self.cfg)

        optimizer_params = self._get_optimizer_params(model)
        optimizer = torch.optim.AdamW(
            optimizer_params, lr=0.0, eps=1e-15, betas=self.cfg.opt.betas
        )
        scheduler = None
        if self.cfg.opt.step_lr != -1:
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=self.cfg.opt.step_lr, gamma=self.cfg.opt.lr_gamma
            )
        return model, optimizer, scheduler

    def _get_optimizer_params(self, model: torch.nn.Module) -> List[Dict]:
        """Get optimizer parameters based on model type"""
        base_lr = self.cfg.opt.base_lr
        params = [{"params": model.point_network.parameters(), "lr": base_lr}]

        if self.cfg.opt.use_fusion:
            fusion_params = [
                {"params": model.fusion_mlps.parameters(), "lr": base_lr},
                {"params": model.image_conv.parameters(), "lr": base_lr},
            ]
            params.extend(fusion_params)

        return params

    def setup_distributed(self) -> None:
        """Setup distributed training if needed"""
        if self.cfg.general.multiple_gpu:
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.model = create_ddp_model(
                self.model.cuda(), broadcast_buffers=False, find_unused_parameters=True
            )

    def setup_ema(self) -> None:
        """Setup EMA if enabled"""
        if self.cfg.opt.ema.use:
            self.ema = EMA(
                self.model,
                beta=self.cfg.opt.ema.beta,
                update_every=self.cfg.opt.ema.update_every,
                update_after_step=self.cfg.opt.ema.update_after_step,
            )
        else:
            self.ema = None

    def save_checkpoint(self, iteration: int, best_psnr: float, save_path: str) -> None:
        """Save model checkpoint"""
        ckpt_save_dict = {
            "iteration": iteration,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "model_state_dict": (
                self.ema.ema_model.state_dict() if self.ema else self.model.state_dict()
            ),
            "best_PSNR": best_psnr,
        }
        torch.save(ckpt_save_dict, save_path)


class ValidationManager:
    """Manages model validation and evaluation"""

    def __init__(self, cfg: DictConfig, device: torch.device, logger):
        self.cfg = cfg
        self.device = device
        self.logger = logger
        self.lpips_fn = (
            lpips_lib.LPIPS(net="vgg").to(device) if cfg.opt.lambda_lpips != 0 else None
        )

    def validate_model(
        self, model: torch.nn.Module, val_loader: DataLoader, iteration: int
    ) -> float:
        """Validate model performance"""
        torch.cuda.empty_cache()

        model.eval()
        with torch.no_grad():
            scores = evaluate_dataset(
                model,
                val_loader,
                device=self.device,
                model_cfg=self.cfg,
            )
            self.logger.log_validation_progress(
                scores,
                iteration,
                lr=(
                    self.model_manager.scheduler.get_last_lr()[0]
                    if self.model_manager.scheduler is not None
                    else self.cfg.opt.base_lr
                ),
            )

            psnr = torch.tensor(scores["PSNR_novel"]).to(self.device)
            if self.cfg.general.multiple_gpu:
                dist.all_reduce(psnr, op=dist.ReduceOp.SUM)
                psnr /= dist.get_world_size()

        return psnr.item()

    def calculate_losses(
        self,
        rendered_images: torch.Tensor,
        gt_images: torch.Tensor,
        iteration: int,
    ) -> Dict[str, torch.Tensor]:
        """Calculate all training losses"""
        losses = {}

        # Calculate reconstruction loss
        background = torch.tensor(
            [1, 1, 1] if self.cfg.data.white_background else [0, 0, 0],
            dtype=torch.float32,
            device=self.device,
        )

        if self.cfg.opt.loss == "focal_l2":
            losses["l12_loss"] = focal_l2_loss(
                rendered_images,
                gt_images,
                background,
                self.cfg.opt.non_bg_color_loss_rate,
                self.cfg.opt.bg_color_loss_rate,
            )
        else:
            loss_func = l1_loss if self.cfg.opt.loss == "l1" else l2_loss
            losses["l12_loss"] = loss_func(rendered_images, gt_images)

        # Add LPIPS loss if enabled
        if (
            self.cfg.opt.lambda_lpips != 0
            and iteration > self.cfg.opt.start_lpips_after
        ):
            losses["lpips_loss"] = torch.mean(
                self.lpips_fn(rendered_images * 2 - 1, gt_images * 2 - 1)
            )

        # Calculate total loss
        losses["total_loss"] = (
            losses["l12_loss"] + losses.get("lpips_loss", 0) * self.cfg.opt.lambda_lpips
        )

        return losses


class Trainer:
    """Main trainer class that orchestrates the training process"""

    DATASET_KEYS = [
        "coord",
        "feat",
        "segment",
        "offset",
        "grid_coord",
        "links",
        "inverse",
        "no_occlusion_links",
        "condition",
    ]

    def __init__(self, cfg: DictConfig):
        self.vis_dir = os.getcwd()
        self.device = safe_state(cfg)
        self.cfg = cfg

        # Initialize components
        self.logger = Logger(cfg, self.vis_dir)
        self.data_manager = DataManager(cfg, self.device)
        self.model_manager = ModelManager(cfg, self.device)
        self.validation_manager = ValidationManager(cfg, self.device, self.logger)

        self.best_psnr = 0.0

    def train(self) -> None:
        """Main training loop"""
        for iteration in range(1, self.cfg.opt.iterations + 1):
            if self.cfg.opt.mode != "test":
                # Training step
                loss_dict = self.train_iteration(iteration)

                # Optimizer step
                loss_dict["total_loss"].backward()
                self.model_manager.optimizer.step()
                self.model_manager.optimizer.zero_grad()

                # Step scheduler if enabled
                if self.model_manager.scheduler is not None:
                    self.model_manager.scheduler.step()

                # Update EMA if enabled
                if self.model_manager.ema:
                    self.model_manager.ema.update()

                # Logging
                if iteration % self.cfg.logging.loss_log == 0:
                    self.logger.log_training_progress(loss_dict, iteration)

                # Validation
                if iteration % self.cfg.logging.val_log == 0:
                    self.validate(iteration)

                # Generating test examples
                if iteration % self.cfg.logging.loop_log == 0 or iteration == 1:
                    self.generate_test_examples(iteration)

        self.logger.finish()

    def prepare_model_inputs(
        self, data: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare input data for model forward pass.

        Args:
            data: Dictionary containing all input data including images, point clouds,
                 and transformation matrices

        Returns:
            Dictionary containing processed model inputs with appropriate device placement
            and batch organization
        """
        point_cloud = data["point_cloud"] if "point_cloud" in data else data
        view_transforms = data["view_to_world_transforms"][
            :, : self.cfg.data.input_images, ...
        ]

        model_inputs = {
            "point_cloud": point_cloud,
            "source_cameras_view_to_world": view_transforms,
            "image": None,
            "unprojected_coords": None,
            "links": None,
        }

        if self.cfg.opt.use_fusion:
            model_inputs.update(
                {
                    "image": data["gt_images"][:, : self.cfg.data.input_images, ...],
                    "unprojected_coords": (
                        data["unprojected_coords"][:, : self.cfg.data.input_images, ...]
                        if self.cfg.opt.level == "scene"
                        else None
                    ),
                    "links": (
                        data["links"][:, :, : self.cfg.data.input_images]
                        if self.cfg.opt.level == "scene"
                        else None
                    ),
                }
            )

        return to_device(model_inputs, self.device)

    def render_validation_views(
        self, gaussian_splats: Dict[str, torch.Tensor], data: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Render validation views using the predicted Gaussian splats.

        Args:
            gaussian_splats: Dictionary containing predicted Gaussian splat parameters
            data: Dictionary containing ground truth data and camera parameters

        Returns:
            tuple: (rendered_images, gt_images)
                - rendered_images: Tensor of rendered novel views
                - gt_images: Tensor of corresponding ground truth images
        """
        rendered_images = []
        gt_images = []

        # Set background color based on configuration
        background = torch.tensor(
            [1, 1, 1] if self.cfg.data.white_background else [0, 0, 0],
            dtype=torch.float32,
            device=self.device,
        )

        # Process each batch
        for b_idx in range(data["gt_images"].shape[0]):
            # Extract gaussian parameters for current batch
            gaussian_splat_batch = {
                k: v[b_idx].contiguous()
                for k, v in gaussian_splats.items()
                if len(v.shape) > 1
            }

            # Render each validation view
            for r_idx in range(self.cfg.data.input_images, data["gt_images"].shape[1]):
                # Render the novel view
                image = render_predicted(
                    gaussian_splat_batch,
                    data["world_view_transforms"][b_idx, r_idx].to(self.device),
                    data["full_proj_transforms"][b_idx, r_idx].to(self.device),
                    data["camera_centers"][b_idx, r_idx].to(self.device),
                    background,
                    self.cfg,
                    focals_pixels=None,
                )["render"]

                gt_image = data["gt_images"][b_idx, r_idx].to(self.device)

                rendered_images.append(image)
                gt_images.append(gt_image)

        # Stack all images into tensors
        rendered_images = torch.stack(rendered_images, dim=0)
        gt_images = torch.stack(gt_images, dim=0)

        return rendered_images, gt_images

    def train_iteration(self, iteration: int) -> Dict[str, torch.Tensor]:
        """Execute one training iteration"""
        data = next(iter(self.data_manager.train_loader))

        model_inputs = self.prepare_model_inputs(data)

        gaussian_splats = self.model_manager.model(**model_inputs)
        rendered_images, gt_images = self.render_validation_views(gaussian_splats, data)

        # Log rendered images if needed

        return self.validation_manager.calculate_losses(
            rendered_images, gt_images, iteration
        )

    def validate(self, iteration: int) -> None:
        """Perform validation and generate test videos"""
        current_psnr = self.validation_manager.validate_model(
            (
                self.model_manager.model
                if not self.model_manager.ema
                else self.model_manager.ema
            ),
            self.data_manager.val_loader,
            iteration,
        )

        if current_psnr > self.best_psnr and comm.get_rank() == 0:
            self.best_psnr = current_psnr
            self.model_manager.save_checkpoint(
                iteration, self.best_psnr, os.path.join(self.vis_dir, "model_best.pth")
            )

    def generate_test_examples(self, iteration: int) -> None:
        """Generate test videos if needed"""
        # Get test data from test loader
        vis_data = next(iter(self.data_manager.test_loader))

        vis_data = to_device(vis_data, self.device)

        # Generate gaussian splats
        model_inputs = self.prepare_model_inputs(vis_data)
        gaussian_splats = self.model_manager.model(**model_inputs)

        # Generate test videos
        test_loop = []
        test_loop_gt = []
        # Set background color based on configuration
        background = torch.tensor(
            [1, 1, 1] if self.cfg.data.white_background else [0, 0, 0],
            dtype=torch.float32,
            device=self.device,
        )

        # Render each view
        for r_idx in range(vis_data["gt_images"].shape[1]):
            # Render predicted view
            test_image = render_predicted(
                {k: v[0].contiguous() for k, v in gaussian_splats.items()},
                vis_data["world_view_transforms"][:, r_idx],
                vis_data["full_proj_transforms"][:, r_idx],
                vis_data["camera_centers"][:, r_idx],
                background,
                self.cfg,
                focals_pixels=None,
            )["render"]

            test_loop.append(
                (np.clip(test_image.detach().cpu().numpy(), 0, 1) * 255).astype(
                    np.uint8
                )
            )

            # Add ground truth
            test_loop_gt.append(
                (
                    np.clip(
                        vis_data["gt_images"][0, r_idx].detach().cpu().numpy(),
                        0,
                        1,
                    )
                    * 255
                ).astype(np.uint8)
            )

        # Log videos
        self.logger.log_test_videos(
            test_loop,
            test_loop_gt,
            iteration,
            0,
        )


@hydra.main(version_base=None, config_path="configs", config_name="default_config")
def main(cfg: DictConfig):
    """Main entry point for training"""
    with open_dict(cfg):
        cfg.general.multiple_gpu = (
            isinstance(cfg.general.device, list) and len(cfg.general.device) > 1
        )

    multiprocessing.set_start_method("spawn")
    if cfg.general.multiple_gpu:
        launch(
            main_worker,
            num_gpus_per_machine=len(cfg.general.device),
            dist_url="auto",
            cfg=(cfg,),
        )
    else:
        launch(main_worker, num_gpus_per_machine=1, dist_url="auto", cfg=(cfg,))


def main_worker(cfg: DictConfig):
    """Main training worker function"""
    trainer = Trainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
