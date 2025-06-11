import torch
import torch.nn as nn

from openpoints.models.backbone.pointmlp import pointMLP
from openpoints.models.backbone.transformer import PointTransformerEncoder
from openpoints.models.Mamba3D.Mamba3D import Mamba3DSeg
from openpoints.models.segmentation import BaseSeg
from openpoints.models.PCM.PCM import PointMambaEncoder

from pointcept.models.sparse_unet.spconv_unet_v1m1_base import SpUNetBase
from pointcept.models.point_transformer_v3.point_transformer_v3m1_base import (
    PointTransformerV3,
)

from typing import Any, List, Optional, Tuple, Union, Dict


class PointFeaturePredictor(nn.Module):
    """Point cloud feature predictor supporting multiple backbone architectures"""

    SUPPORTED_MODELS = {
        "pointmlp": pointMLP,
        "transformer": PointTransformerEncoder,
        "pcm": PointMambaEncoder,
        "mamba3d": Mamba3DSeg,
        "sparseunet": SpUNetBase,
        "ptv3": PointTransformerV3,
    }

    def __init__(self, cfg, out_channels, pretrained_path=None):
        super(PointFeaturePredictor, self).__init__()
        self.cfg = cfg
        self.out_channels = out_channels

        # Initialize backbone
        self.encoder = self._create_encoder()

        # Initialize final layers
        self.final = self._create_final_layers()

        # Print model statistics
        self._print_model_stats()

        if pretrained_path is not None:
            self._load_pretrained_weights(pretrained_path)

    def _load_pretrained_weights(self, pretrained_path):
        """Load pretrained weights from a file"""
        state_dict = torch.load(pretrained_path)
        info = self.load_state_dict(state_dict, strict=False)
        print("Loaded pretrained weights from {}".format(pretrained_path))
        print("Missing keys: {}".format(info.missing_keys))
        print("Unexpected keys: {}".format(info.unexpected_keys))

    def _create_encoder(self):
        """Create encoder based on model configuration"""
        model_type = self.cfg.model.backbone_type.lower()
        if model_type not in self.SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model type: {model_type}")

        if model_type == "transformer":
            return self.SUPPORTED_MODELS[model_type](
                in_channels=3, num_groups=128, encoder_dims=384, depth=16
            )
        elif model_type == "sparseunet":
            return self.SUPPORTED_MODELS[model_type](
                in_channels=6, num_classes=23, cfg=self.cfg
            )
        elif model_type == "pcm":
            return BaseSeg(**self._get_mamba_config())
        elif model_type == "mamba3d":
            return Mamba3DSeg(self._get_mamba3d_config())
        else:  # pointmlp
            return self.SUPPORTED_MODELS[model_type](cfg=self.cfg)

    def _create_final_layers(self):
        """Create final layers based on model type"""
        if self.cfg.model.backbone_type.lower() in ["transformer", "mamba3d"]:
            return nn.Sequential(nn.Linear(384, 128), nn.ReLU(), nn.Linear(128, 23))
        else:
            return nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 23))

    def _print_model_stats(self):
        """Print model parameter statistics"""
        all_params = sum(
            p.numel() for p in self.encoder.parameters() if p.requires_grad
        )
        print(f"Encoder parameters: {all_params}")

    def forward(
        self, x: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass through the network"""
        x, center = self.encoder(x, None, None, None, None)
        output = self.final(x).permute(0, 2, 1)
        return (output, center)

    def forward_feat_fusion(
        self,
        x: torch.Tensor,
        image_features: torch.Tensor,
        c2w_projection_matrix: torch.Tensor,
        fusion_mlps: nn.ModuleList,
        intrinsic: torch.Tensor,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass with image fusion"""
        x, center = self.encoder.forward(
            x, image_features, c2w_projection_matrix, fusion_mlps, intrinsic
        )
        output = self.final(x).permute(0, 2, 1)
        return (output, center)

    def forward_point_fusion(
        self,
        x: torch.Tensor,
        image_features: torch.Tensor,
        links: torch.Tensor,
        unprojected_coords: torch.Tensor,
        fusion_mlps: nn.ModuleList,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass with image fusion"""
        output = self.encoder.forward(
            x, image_features, links, unprojected_coords, fusion_mlps
        )
        return (output.features, output.indices)

    def _get_mamba_config(self):
        """Get configuration for PointMambaEncoder"""
        return {
            "encoder_args": {
                "NAME": "PointMambaEncoder",
                "in_channels": 4,
                "embed_dim": 384,
                "groups": 1,
                "res_expansion": 1,
                "activation": "relu",
                "bias": False,
                "use_xyz": True,
                "normalize": "anchor",
                "dim_expansion": [1, 1, 2, 1],
                "pre_blocks": [1, 1, 1, 1],
                "mamba_blocks": [1, 2, 2, 4],
                "pos_blocks": [0, 0, 0, 0],
                "k_neighbors": [12, 12, 12, 12],
                "reducers": [2, 2, 2, 2],
                "rms_norm": True,
                "residual_in_fp32": True,
                "fused_add_norm": True,
                "bimamba_type": "v2",
                "drop_path_rate": 0.1,
                "mamba_pos": True,
                "mamba_layers_orders": [
                    "xyz",
                    "xzy",
                    "yxz",
                    "yzx",
                    "zxy",
                    "zyx",
                    "hilbert",
                    "z",
                    "z-trans",
                ],
                "use_order_prompt": True,
                "prompt_num_per_order": 6,
            },
            "decoder_args": {
                "NAME": "PointMambaDecoder",
                "encoder_channel_list": [384, 384, 384, 768, 768],
                "decoder_channel_list": [
                    768,
                    384,
                    384,
                    384,
                    # 128
                ],
                "decoder_blocks": [1, 1, 1, 1],
                "mamba_blocks": [0, 0, 0, 0],
                "mamba_layers_orders": [],
            },
            "cls_args": {
                "NAME": "SegHead",
                # "global_feat": "max,avg",
                "num_classes": 128,
                "in_channels": 384,
                # "in_channels": 128,
                "norm_args": {"norm": "bn"},
            },
        }

    def _get_mamba3d_config(self):
        """Get configuration for Mamba3DSeg"""

        class Mamba3DConfig:
            def __init__(self):
                self.NAME = "Mamba3D"
                self.trans_dim = 384
                self.depth = 16
                self.drop_path_rate = 0.1
                self.num_heads = 6
                self.group_size = 32
                self.num_group = 128
                self.encoder_dims = 384
                self.bimamba_type = "v4"
                self.center_local_k = 4
                self.ordering = False
                self.label_smooth = 0.0
                self.lr_ratio_cls = 1.0
                self.lr_ratio_lfa = 1.0
                self.fusion = True

        return Mamba3DConfig()
