import torch
import torch.nn as nn

import math
import numpy as np

import spconv.pytorch as spconv

from typing import Any, List, Optional, Tuple, Union, Dict

from model.image_predictor import ImageFeaturePredictor
from model.point_predictor import PointFeaturePredictor


class GaussianSplatPredictor(nn.Module):
    """Unified Gaussian Splat Predictor supporting both basic and fusion architectures"""

    MODEL_CONFIGS = {
        "pointmlp": {
            "feature_dim": 128,
            "fusion_dim": 128,
            "final_dim": 128,
        },
        "transformer": {
            "feature_dim": 384,
            "fusion_dim": 384,
            "final_dim": 384,
        },
        "pcm": {
            "feature_dim": 384,
            "fusion_dim": 384,
            "final_dim": 384,
        },
        "mamba3d": {
            "feature_dim": 384,
            "fusion_dim": 384,
            "final_dim": 384,
        },
        "sparseunet": {
            "feature_dim": 128,
            "fusion_dim": 96,
            "final_dim": 96,
        },
        "ptv3": {
            "feature_dim": 128,
            "fusion_dim": 128,
            "final_dim": 128,
        },
    }

    def __init__(self, cfg):
        super(GaussianSplatPredictor, self).__init__()
        self.cfg = cfg
        self.use_fusion = hasattr(cfg.opt, "use_fusion") and cfg.opt.use_fusion

        # Initialize network parameters
        split_dimensions = self._get_network_params()

        # Initialize networks
        if self.use_fusion:
            self._init_fusion_networks(split_dimensions)
        else:
            self._init_basic_network(
                split_dimensions,
            )

        # Initialize common components
        self._init_activations()

        # Initialize SH matrices if needed
        if self.cfg.model.max_sh_degree > 0:
            self._init_sh_matrices()

        # Initialize camera intrinsics for fusion mode
        if self.use_fusion and self.cfg.opt.level == "object":
            self.intrinsic = self._get_camera_intrinsics()

    def forward(
        self,
        point_cloud: torch.Tensor,
        image: Optional[torch.Tensor] = None,
        source_cameras_view_to_world: Optional[torch.Tensor] = None,
        unprojected_coords: Optional[torch.Tensor] = None,
        links: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through the network"""
        if self.use_fusion:
            return self._forward_fusion(
                point_cloud,
                image,
                source_cameras_view_to_world,
                unprojected_coords,
            )
        else:
            return self._forward_basic(point_cloud, source_cameras_view_to_world)

    def _forward_basic(self, point_cloud, source_cameras_view_to_world):
        """Forward pass for basic network"""
        B = source_cameras_view_to_world.shape[0]
        N_views = 1

        # Process network output
        point_output, center = self.point_network(point_cloud)

        # Generate final output
        network_output = point_output.split(self.split_dimensions, dim=1)
        return self._process_output(
            network_output, center, B, N_views, source_cameras_view_to_world
        )

    def _forward_fusion(
        self,
        point_cloud: torch.Tensor,
        image: torch.Tensor,
        source_cameras_view_to_world: Optional[torch.Tensor] = None,
        unprojected_coords: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for fusion network with support for both object and scene level processing

        Args:
            point_cloud: Input point cloud tensor
            image: Input image tensor
            source_cameras_view_to_world: Camera transformation matrix (for object level)
            unprojected_coords: Unprojected coordinates (for scene level)

        Returns:
            Dictionary containing processed features and parameters
        """
        B = image.shape[0]
        N_views = image.shape[1]

        # Process image features
        image = image.reshape(B * N_views, *image.shape[2:])

        image_output = self.image_network.forward(image)

        # Generate image features
        image_features = self.image_conv.forward(image_output["decoder_block_3"])

        # Process point cloud features based on optimization level
        if self.cfg.opt.level == "object":
            point_features, center = self.point_network.forward_feat_fusion(
                point_cloud,
                image_features,
                source_cameras_view_to_world,
                self.fusion_mlps,
                self.intrinsic,
            )
            network_output = point_features.split(self.split_dimensions, dim=1)
            out_dict = self._process_network_output(
                network_output, center, None, is_scene_level=False
            )
            out_dict = self._multi_view_union(out_dict, B, N_views)
            return self._make_contiguous(out_dict)

        elif self.cfg.opt.level == "scene":
            point_features, indices = self.point_network.forward_point_fusion(
                point_cloud,
                image_features,
                unprojected_coords,
                self.fusion_mlps,
            )
            network_output = point_features.split(self.split_dimensions, dim=1)
            return self._process_network_output(
                network_output, point_cloud["coord"], indices, is_scene_level=True
            )

        else:
            raise ValueError("Invalid optimization level")

    def _get_network_params(self):
        """Get network parameters for initialization"""
        split_dimensions = [3, 1, 3, 4, 3]  # xyz, opacity, scale, rotation, features_dc

        if self.cfg.model.max_sh_degree != 0:
            sh_dims = (self.cfg.model.max_sh_degree + 1) ** 2 - 1
            split_dimensions.append(sh_dims * 3)

        self.split_dimensions = split_dimensions
        return split_dimensions

    def _init_basic_network(self, split_dimensions):
        """Initialize basic network without fusion"""
        self.point_network = networkCallBack(
            self.cfg,
            self.cfg.model.backbone_type,
            split_dimensions,
            pretrained_path=self.cfg.opt.pretrained_ckpt,
        )

    def _init_fusion_networks(self, split_dimensions):
        """Initialize networks with fusion components"""
        self.image_network = networkCallBack(self.cfg, "image", [128])

        self.point_network = networkCallBack(
            self.cfg,
            self.cfg.model.backbone_type,
            split_dimensions,
            pretrained_path=self.cfg.opt.pretrained_ckpt,
        )

        model_config = self.MODEL_CONFIGS[self.cfg.model.backbone_type]

        # Initialize image convolution layers
        image_conv_in_dim = self.image_network.encoder_config["block_out_channels"][0]
        image_conv_out_dim = model_config["feature_dim"]
        fusion_dim = model_config["fusion_dim"]

        self.image_conv = (
            nn.Sequential(
                nn.GroupNorm(32, image_conv_in_dim, eps=1e-06),
                nn.Conv2d(image_conv_in_dim, image_conv_out_dim, kernel_size=1),
            )
            if self.cfg.opt.level == "object"
            else nn.Sequential(
                nn.GroupNorm(32, image_conv_in_dim, eps=1e-06),
                nn.Conv2d(image_conv_in_dim, fusion_dim, kernel_size=1),
            )
        )

        # Initialize fusion MLPs
        self.fusion_mlps = (
            (
                nn.Sequential(
                    nn.Linear(image_conv_out_dim + fusion_dim, fusion_dim),
                    nn.ReLU(),
                )
            )
            if self.cfg.opt.level == "object"
            else spconv.SparseSequential(
                spconv.SubMConv3d(fusion_dim, fusion_dim, kernel_size=3, padding=1),
                nn.BatchNorm1d(fusion_dim),
                nn.ReLU(inplace=True),
            )
        )

    def _init_activations(self):
        """Initialize activation functions"""
        self.pos_act = nn.Tanh()
        self.scaling_activation = torch.exp
        self.opacity_activation = torch.sigmoid
        self.rotation_activation = torch.nn.functional.normalize

    def _init_sh_matrices(self):
        """Initialize spherical harmonics transformation matrices"""
        v_to_sh = torch.tensor([[0, 0, -1], [-1, 0, 0], [0, 1, 0]], dtype=torch.float32)
        sh_to_v = v_to_sh.transpose(0, 1)
        self.register_buffer("sh_to_v_transform", sh_to_v.unsqueeze(0))
        self.register_buffer("v_to_sh_transform", v_to_sh.unsqueeze(0))

    def _get_camera_intrinsics(self):
        """Get camera intrinsic parameters"""
        fov = self.cfg.data.fov
        res = self.cfg.data.training_resolution

        intrinsics = np.zeros((3, 4))
        intrinsics[2, 2] = 1

        focal = (res / 2.0) / math.tan(math.radians(fov / 2.0))
        intrinsics[0, 0] = focal
        intrinsics[1, 1] = focal
        intrinsics[0, 2] = res / 2.0
        intrinsics[1, 2] = res / 2.0

        return intrinsics

    def _process_network_output(
        self,
        network_output: List[torch.Tensor],
        center: Optional[torch.Tensor] = None,
        indices: Optional[torch.Tensor] = None,
        is_scene_level: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Process network output into final format for both object and scene level

        Args:
            network_output: List of output tensors from network
            center: Center points tensor
            indices: Batch indices for scene level processing
            is_scene_level: Whether processing scene level output

        Returns:
            Dictionary containing processed features
        """
        # Extract basic features
        xyz_raw, opacity, scaling, rotation, features_dc = network_output[:5]

        # Process position
        pos = self.pos_act(xyz_raw) * self.cfg.model.offset_scale
        pos = pos.permute(0, 2, 1) + center[:, :, :3]

        # Handle isotropic scaling
        if self.cfg.model.isotropic:
            scaling = scaling[:, :1].expand(-1, 3, -1)

        if not is_scene_level:
            # Object level processing
            out_dict = {
                "xyz": pos,
                "opacity": self._flatten_vector(self.opacity_activation(opacity)),
                "scaling": self._flatten_vector(self.scaling_activation(scaling)),
                "rotation": self._flatten_vector(self.rotation_activation(rotation)),
                "features_dc": self._flatten_vector(features_dc).unsqueeze(2),
            }

            # Process SH features
            out_dict["features_rest"] = self._process_sh_features_object(
                network_output[5] if self.cfg.model.max_sh_degree > 0 else None,
                out_dict["features_dc"],
            )

            return out_dict
        else:
            # Scene level processing
            batch_size = indices[:, 0].max().item() + 1
            out_dict = {
                "xyz": [None] * batch_size,
                "opacity": [None] * batch_size,
                "scaling": [None] * batch_size,
                "rotation": [None] * batch_size,
                "features_dc": [None] * batch_size,
                "features_rest": [None] * batch_size,
            }

            # Process features for each batch
            for batch_id in range(batch_size):
                mask = indices[:, 0] == batch_id
                out_dict["xyz"][batch_id] = pos[mask]
                out_dict["opacity"][batch_id] = self.opacity_activation(opacity[mask])
                out_dict["scaling"][batch_id] = self.scaling_activation(scaling[mask])
                out_dict["rotation"][batch_id] = self.rotation_activation(
                    rotation[mask]
                )
                out_dict["features_dc"][batch_id] = features_dc[mask].unsqueeze(1)

                # Process SH features
                if self.cfg.model.max_sh_degree > 0:
                    features_rest = network_output[5][mask]
                    rest_shape = (features_rest.shape[0], -1, 3)
                    out_dict["features_rest"][batch_id] = self._process_sh_features(
                        features_rest, rest_shape
                    )
                else:
                    out_dict["features_rest"][batch_id] = self._create_zero_sh_features(
                        out_dict["features_dc"][batch_id], xyz_raw.device
                    )

            return out_dict

    def _process_sh_features_object(
        self, features_rest: Optional[torch.Tensor], features_dc: torch.Tensor
    ) -> torch.Tensor:
        """Process SH features for object level output

        Args:
            features_rest: SH features tensor or None
            features_dc: DC features tensor for shape reference

        Returns:
            Processed SH features
        """
        if features_rest is not None:
            features_rest = self._flatten_vector(features_rest)
            return features_rest.reshape(*features_rest.shape[:2], -1, 3)
        else:
            return self._create_zero_sh_features(features_dc, features_dc.device)

    def _create_zero_sh_features(
        self, reference_tensor: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        """Create zero-initialized SH features

        Args:
            reference_tensor: Tensor for shape reference
            device: Device to create tensor on

        Returns:
            Zero-initialized SH features tensor
        """
        return torch.zeros(
            (
                reference_tensor.shape[0],
                (self.cfg.model.max_sh_degree + 1) ** 2 - 1,
                3,
            ),
            dtype=reference_tensor.dtype,
            device=device,
        )

    def _flatten_vector(self, x: torch.Tensor) -> torch.Tensor:
        """Flatten image dimensions to point list"""
        return x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)

    def _make_contiguous(self, tensor_dict):
        return {k: v.contiguous() for k, v in tensor_dict.items()}

    def _multi_view_union(
        self, tensor_dict: Dict[str, torch.Tensor], B: int, N_view: int
    ) -> Dict[str, torch.Tensor]:
        """Combine multiple views into a single tensor"""
        return {
            k: v.reshape(B, N_view * v.shape[1], *v.shape[2:])
            for k, v in tensor_dict.items()
        }

    def _process_sh_features(
        self, features: torch.Tensor, shape: Tuple
    ) -> torch.Tensor:
        """Process spherical harmonics features

        Args:
            features: Input features tensor
            shape: Target shape for reshaping

        Returns:
            Processed SH features
        """
        features = self._flatten_vector(features).reshape(*shape)
        if self.cfg.model.max_sh_degree == 1:
            assert features.shape[2] == 3, "Only order 1 spherical harmonics supported"
        return features


def networkCallBack(cfg, name, out_channels, **kwargs):
    """Network factory function"""
    if name == "image":
        return ImageFeaturePredictor(cfg, out_channels, **kwargs)
    elif name in ["pointmlp", "transformer", "pcm", "mamba3d", "sparseunet", "ptv3"]:
        return PointFeaturePredictor(cfg, out_channels, **kwargs)
    else:
        raise ValueError(f"Unsupported network type: {name}")
