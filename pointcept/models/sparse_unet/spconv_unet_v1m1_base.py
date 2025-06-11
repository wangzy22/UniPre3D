"""
SparseUNet Architecture Implementation
Based on Sparse Convolution (SpConv) - Recommended Version

This implementation provides a sparse UNet architecture using sparse convolutions,
designed for efficient 3D point cloud processing.
"""

from functools import partial
from collections import OrderedDict

import torch
import torch.nn as nn

import spconv.pytorch as spconv
from torch_geometric.utils import scatter

from timm.models.layers import trunc_normal_

from pointcept.models.utils import offset2batch

from fusion.point_fusion import PointFusion


class BasicBlock(spconv.SparseModule):
    """
    Basic residual block for sparse convolution.

    Args:
        in_channels (int): Number of input channels
        embed_channels (int): Number of embedding channels
        stride (int): Stride for convolution
        norm_fn (callable): Normalization function
        indice_key (str): Key for sparse convolution indices
        bias (bool): Whether to use bias in convolutions
    """

    expansion = 1

    def __init__(
        self,
        in_channels,
        embed_channels,
        stride=1,
        norm_fn=None,
        indice_key=None,
        bias=False,
    ):
        super().__init__()
        assert norm_fn is not None

        # Projection layer for residual connection if channels don't match
        if in_channels == embed_channels:
            self.proj = spconv.SparseSequential(nn.Identity())
        else:
            self.proj = spconv.SparseSequential(
                spconv.SubMConv3d(
                    in_channels, embed_channels, kernel_size=1, bias=False
                ),
                norm_fn(embed_channels),
            )

        # Main convolution blocks
        self.conv1 = spconv.SubMConv3d(
            in_channels,
            embed_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=bias,
            indice_key=indice_key,
        )
        self.bn1 = norm_fn(embed_channels)
        self.relu = nn.ReLU()
        self.conv2 = spconv.SubMConv3d(
            embed_channels,
            embed_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=bias,
            indice_key=indice_key,
        )
        self.bn2 = norm_fn(embed_channels)
        self.stride = stride

    def forward(self, x):
        """Forward pass with residual connection"""
        residual = x

        # First conv block
        out = self.conv1(x)
        out = out.replace_feature(self.bn1(out.features))
        out = out.replace_feature(self.relu(out.features))

        # Second conv block
        out = self.conv2(out)
        out = out.replace_feature(self.bn2(out.features))

        # Add residual and apply ReLU
        out = out.replace_feature(out.features + self.proj(residual).features)
        out = out.replace_feature(self.relu(out.features))

        return out


class SpUNetBase(nn.Module):
    """
    Base class for Sparse UNet architecture.

    Args:
        in_channels (int): Input channels
        num_classes (int): Number of output classes
        cfg (Config): Configuration object
        base_channels (int): Base number of channels
        channels (tuple): Channel numbers for each layer
        layers (tuple): Number of blocks for each layer
        cls_mode (bool): Whether to use classification mode
    """

    def __init__(
        self,
        in_channels,
        num_classes,
        cfg=None,
        base_channels=32,
        channels=(32, 64, 128, 256, 256, 128, 96, 96),
        layers=(2, 3, 4, 6, 2, 2, 2, 2),
        cls_mode=False,
    ):
        super().__init__()
        assert len(layers) % 2 == 0, "Number of layers must be even"
        assert len(layers) == len(channels), "Channels and layers must have same length"

        # Initialize configuration
        self.cfg = cfg
        self.use_fusion = hasattr(cfg.model, "use_fusion") and cfg.model.use_fusion

        # Network parameters
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.base_channels = base_channels
        self.channels = channels
        self.layers = layers
        self.num_stages = len(layers) // 2
        self.cls_mode = cls_mode

        # Define normalization function and block type
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        block = BasicBlock

        # Input convolution
        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(
                in_channels,
                base_channels,
                kernel_size=5,
                padding=1,
                bias=False,
                indice_key="stem",
            ),
            norm_fn(base_channels),
            nn.ReLU(),
        )

        # Initialize encoder and decoder parameters
        enc_channels = base_channels
        dec_channels = channels[-1]

        # Create encoder and decoder modules
        self.down = nn.ModuleList()
        self.up = nn.ModuleList()
        self.enc = nn.ModuleList()
        self.dec = nn.ModuleList() if not self.cls_mode else None

        # Build encoder and decoder architecture
        for s in range(self.num_stages):
            # Encoder stage
            self._build_encoder_stage(s, enc_channels, channels, norm_fn, block)

            # Decoder stage (if not in classification mode)
            if not self.cls_mode:
                self._build_decoder_stage(
                    s, dec_channels, enc_channels, channels, norm_fn, block
                )

            enc_channels = channels[s]
            dec_channels = channels[len(channels) - s - 2]

        # Final convolution layer
        final_in_channels = (
            channels[-1] if not self.cls_mode else channels[self.num_stages - 1]
        )
        self.final = (
            spconv.SubMConv3d(
                final_in_channels, num_classes, kernel_size=1, padding=1, bias=True
            )
            if num_classes > 0
            else spconv.Identity()
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _build_encoder_stage(self, stage, enc_channels, channels, norm_fn, block):
        """Helper method to build encoder stage"""
        self.down.append(
            spconv.SparseSequential(
                spconv.SparseConv3d(
                    enc_channels,
                    channels[stage],
                    kernel_size=2,
                    stride=2,
                    bias=False,
                    indice_key=f"spconv{stage + 1}",
                ),
                norm_fn(channels[stage]),
                nn.ReLU(),
            )
        )

        self.enc.append(
            spconv.SparseSequential(
                OrderedDict(
                    [
                        (
                            f"block{i}",
                            block(
                                channels[stage],
                                channels[stage],
                                norm_fn=norm_fn,
                                indice_key=f"subm{stage + 1}",
                            ),
                        )
                        for i in range(self.layers[stage])
                    ]
                )
            )
        )

    def _build_decoder_stage(
        self, stage, dec_channels, enc_channels, channels, norm_fn, block
    ):
        """Helper method to build decoder stage"""
        self.up.append(
            spconv.SparseSequential(
                spconv.SparseInverseConv3d(
                    channels[len(channels) - stage - 2],
                    dec_channels,
                    kernel_size=2,
                    bias=False,
                    indice_key=f"spconv{stage + 1}",
                ),
                norm_fn(dec_channels),
                nn.ReLU(),
            )
        )

        self.dec.append(
            spconv.SparseSequential(
                OrderedDict(
                    [
                        (
                            f"block{i}",
                            block(
                                dec_channels + enc_channels if i == 0 else dec_channels,
                                dec_channels,
                                norm_fn=norm_fn,
                                indice_key=f"subm{stage}",
                            ),
                        )
                        for i in range(self.layers[len(channels) - stage - 1])
                    ]
                )
            )
        )

    @staticmethod
    def _init_weights(m):
        """Initialize network weights"""
        if isinstance(m, (nn.Linear, spconv.SubMConv3d)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(
        self,
        input_dict,
        img_features,
        links=None,
        unprojected_coords=None,
        fusion_mlps=None,
    ):
        """
        Forward pass of the network

        Args:
            input_dict (dict): Input dictionary containing point cloud data
            img_features (tensor): Image features for fusion
            links (tensor): Linking information
            unprojected_coords (tensor): Unprojected coordinates for fusion
            fusion_mlps (nn.Module): MLPs for point fusion
        """
        if self.use_fusion:
            point_fusion = PointFusion(
                fusion_mlps=fusion_mlps,
                fea2d_dim=self.channels[0],
                viewNum=self.cfg.data.input_images,
            )

        grid_coord = input_dict["grid_coord"]
        feat = input_dict["feat"]
        offset = input_dict["offset"]

        raw_inverse = input_dict["inverse"]
        pts_num_list = input_dict["pts_num"]
        current_links = links

        pts_accumulated_num = 0
        for i in range(len(offset) - 1):
            pts_accumulated_num += pts_num_list[i]
            raw_inverse[
                pts_accumulated_num : pts_accumulated_num + pts_num_list[i + 1]
            ] += offset[i]
            current_links[
                pts_accumulated_num : pts_accumulated_num + pts_num_list[i + 1], 0
            ] = (i + 1)
        # Add batch_num in links
        batch = offset2batch(offset)
        sparse_shape = torch.add(torch.max(grid_coord, dim=0).values, 96).tolist()
        init_x = spconv.SparseConvTensor(
            features=feat,
            indices=torch.cat(
                [batch.unsqueeze(-1).int(), grid_coord.int()], dim=1
            ).contiguous(),
            spatial_shape=sparse_shape,
            batch_size=batch[-1].tolist() + 1,
        )
        x = self.conv_input(init_x)

        img_features.reverse()
        if self.use_fusion:
            assert unprojected_coords is not None
            point_fusion = point_fusion.to(img_features[0].device)
            x = point_fusion(
                img_features[0],
                x,
                unprojected_coords,
                input_dict,
            )

        skips = [x]
        # enc forward
        for s in range(self.num_stages):
            x = self.down[s](x)
            x = self.enc[s](x)
            skips.append(x)
        x = skips.pop(-1)

        if not self.cls_mode:
            # dec forward
            h, w = img_features[0].shape[2], img_features[0].shape[3]
            for s in reversed(range(self.num_stages)):

                x = self.up[s](x)

                skip = skips.pop(-1)
                x = x.replace_feature(torch.cat((x.features, skip.features), dim=1))
                x = self.dec[s](x)

        x = self.final(x)
        if self.cls_mode:
            x = x.replace_feature(
                scatter(x.features, x.indices[:, 0].long(), reduce="mean", dim=0)
            )

        return x
