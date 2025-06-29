import wandb
import os
from omegaconf import DictConfig, OmegaConf
import torch
import numpy as np
import glob
from typing import Dict, List
import pointcept.utils.comm as comm


class Logger:
    """Manages all logging operations with fallback for offline mode"""

    def __init__(self, cfg: DictConfig, vis_dir: str):
        self.cfg = cfg
        self.vis_dir = vis_dir
        self.is_wandb_available = self.setup_wandb()
        # Create videos directory if wandb is not available
        if not self.is_wandb_available:
            self.videos_dir = os.path.join(self.vis_dir, "videos")
            os.makedirs(self.videos_dir, exist_ok=True)

    def check_wandb_connection(self) -> bool:
        """
        Check if wandb can be connected

        Returns:
            bool: True if wandb is available and can be connected, False otherwise
        """
        try:
            # Try to initialize wandb in test mode
            wandb.init(anonymous="must", project="test_connection")
            wandb.finish()
            return True
        except Exception as e:
            print(
                f"Warning: Unable to connect to wandb ({str(e)}). Running in offline mode."
            )
            return False

    def setup_wandb(self) -> bool:
        """
        Initialize W&B logging with fallback to offline mode

        Returns:
            bool: True if wandb was successfully initialized, False if running in offline mode
        """
        # Check if wandb can be connected
        if not self.check_wandb_connection():
            self.wandb_run = None
            return False

        dict_cfg = OmegaConf.to_container(self.cfg, resolve=True, throw_on_missing=True)

        try:
            if os.path.isdir(os.path.join(self.vis_dir, "wandb")):
                # Resume existing run
                run_name_path = glob.glob(
                    os.path.join(self.vis_dir, "wandb", "latest-run", "run-*")
                )[0]
                run_id = (
                    os.path.basename(run_name_path).split("run-")[1].split(".wandb")[0]
                )
                self.wandb_run = wandb.init(
                    project=self.cfg.wandb.project,
                    resume=True,
                    id=run_id,
                    config=dict_cfg,
                )
            else:
                # Start new run
                self.wandb_run = wandb.init(
                    project=self.cfg.wandb.project, reinit=True, config=dict_cfg
                )
            return True
        except Exception as e:
            print(
                f"Warning: Failed to initialize wandb ({str(e)}). Running in offline mode."
            )
            self.wandb_run = None
            return False

    def log_validation_progress(
        self, scores: Dict[str, torch.Tensor], iteration: int, lr: float = None
    ) -> None:
        """Log validation progress with fallback for offline mode"""
        if not (
            (comm.get_rank() == 0 and self.cfg.general.multiple_gpu)
            or not self.cfg.general.multiple_gpu
        ):
            return

        if self.is_wandb_available:
            wandb.log(scores, step=iteration)
        
        print(f"@ Iteration {iteration} Val:", end="")
        print(scores)
        if lr is not None:
            print(f"  Learning rate: {lr:.6f}")

    def _check_main_process(self) -> bool:
        """
        Check if the current process is the main process for logging

        Returns:
            bool: True if this is the main process, False otherwise
        """
        return (
            (comm.get_rank() == 0 and self.cfg.general.multiple_gpu)
            or not self.cfg.general.multiple_gpu
        )

    def log_training_progress(
        self, loss_dict: Dict[str, torch.Tensor], iteration: int
    ) -> None:
        """Log training progress with fallback for offline mode"""
        if not self._check_main_process():
            return

        # If wandb is not available, print to console instead
        print(f"@ Iteration {iteration}:", end="")
        print(
            f"  Training log10 loss: {np.log10(loss_dict['total_loss'].item() + 1e-8):.4f}",
            end="",
        )
        if "l12_loss" in loss_dict:
            print(
                f"  L12 log10 loss: {np.log10(loss_dict['l12_loss'].item() + 1e-8):.4f}",
                end="",
            )
        if "lpips_loss" in loss_dict:
            print(
                f"  LPIPS loss: {np.log10(loss_dict['lpips_loss'].item() + 1e-8):.4f}",
                end="",
            )
        print("")

        if self.is_wandb_available:
            # Log to wandb if available
            wandb.log(
                {"training_loss": np.log10(loss_dict["total_loss"].item() + 1e-8)},
                step=iteration,
            )

            if "l12_loss" in loss_dict:
                wandb.log(
                    {"training_l12_loss": np.log10(loss_dict["l12_loss"].item() + 1e-8)},
                    step=iteration,
                )

            if "lpips_loss" in loss_dict:
                wandb.log(
                    {
                        "training_lpips_loss": np.log10(
                            loss_dict["lpips_loss"].item() + 1e-8
                        )
                    },
                    step=iteration,
                )

    def log_test_videos(
        self,
        test_loop: List[np.ndarray],
        test_loop_gt: List[np.ndarray],
        iteration: int,
        test_generate_num: int = None,
    ) -> None:
        """
        Log test videos to wandb or save locally with fallback for offline mode.

        Args:
            test_loop: List of rendered test images for video
            test_loop_gt: List of ground truth test images for video
            iteration: Current training iteration
            test_generate_num: Test generation number for multiple test cases
        """
        if not self._check_main_process():
            return
        
        if self.is_wandb_available:
            # Log to wandb
            if test_loop is not None:
                video_name = (
                    f"rot_{test_generate_num}"
                    if test_generate_num is not None
                    else "rot"
                )
                wandb.log(
                    {
                        video_name: wandb.Video(
                            np.asarray(test_loop), fps=10, format="mp4"
                        )
                    },
                    step=iteration,
                )

            if test_loop_gt is not None:
                video_name = (
                    f"rot_gt_{test_generate_num}"
                    if test_generate_num is not None
                    else "rot_gt"
                )
                wandb.log(
                    {
                        video_name: wandb.Video(
                            np.asarray(test_loop_gt), fps=5, format="mp4"
                        )
                    },
                    step=iteration,
                )
        else:
            # Save locally using imageio
            try:
                import imageio.v3 as iio
            except ImportError:
                print("Please install imageio with: pip install imageio imageio-ffmpeg")
                return

            print(
                f"@ Iteration {iteration}: Saving videos locally to {self.videos_dir}"
            )

            def process_frames(frames):
                """Convert frames to correct format for video saving"""
                return [
                    # Convert from (C, H, W) to (H, W, C) and ensure uint8
                    (
                        (frame.transpose(1, 2, 0) if frame.shape[0] == 3 else frame)
                        if frame.dtype == np.uint8
                        else (frame.transpose(1, 2, 0) * 255).astype(np.uint8)
                    )
                    for frame in frames
                ]

            # Save predicted video
            if test_loop is not None:
                video_name = (
                    f"iter_{iteration}_rot_{test_generate_num}.mp4"
                    if test_generate_num is not None
                    else f"iter_{iteration}_rot.mp4"
                )
                video_path = os.path.join(self.videos_dir, video_name)
                iio.imwrite(
                    video_path,
                    process_frames(test_loop),
                    fps=10,
                    codec="libx264",
                    output_params=["-pix_fmt", "yuv420p"],
                )
                print(f"Saved predicted video to {video_path}")

            # Save ground truth video
            if test_loop_gt is not None:
                video_name = (
                    f"iter_{iteration}_rot_gt_{test_generate_num}.mp4"
                    if test_generate_num is not None
                    else f"iter_{iteration}_rot_gt.mp4"
                )
                video_path = os.path.join(self.videos_dir, video_name)
                iio.imwrite(
                    video_path,
                    process_frames(test_loop_gt),
                    fps=5,
                    codec="libx264",
                    output_params=["-pix_fmt", "yuv420p"],
                )
                print(f"Saved ground truth video to {video_path}")

    def finish(self):
        """Cleanup wandb run if it exists"""
        if self.wandb_run is not None:
            self.wandb_run.finish()
