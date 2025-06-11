"""PointMLP

Rethinking Network Design and Local Geometry in Point Cloud: A Simple Residual MLP Framework
Xu Ma and Can Qin and Haoxuan You and Haoxi Ran and Yun Fu

Reference:
https://github.com/ma-xu/pointMLP-pytorch
"""

import string
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..layers import (
    furthest_point_sample,
    random_sample,
    LocalAggregation,
    create_convblock2d,
    three_interpolate,
    three_nn,
    gather_operation,
    create_linearblock,
    create_convblock1d,
    create_grouper,
)
import logging
import copy
from ..build import MODELS
from ..layers import furthest_point_sample, fps
from ..layers.group import QueryAndGroup
import torch
import torch.nn as nn
import torch.nn.functional as F
from fusion import FeatureFusion


def get_activation(activation):
    if activation.lower() == "gelu":
        return nn.GELU()
    elif activation.lower() == "rrelu":
        return nn.RReLU(inplace=True)
    elif activation.lower() == "selu":
        return nn.SELU(inplace=True)
    elif activation.lower() == "silu":
        return nn.SiLU(inplace=True)
    elif activation.lower() == "hardswish":
        return nn.Hardswish(inplace=True)
    elif activation.lower() == "leakyrelu":
        return nn.LeakyReLU(inplace=True)
    else:
        return nn.ReLU(inplace=True)


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src**2, -1).view(B, N, 1)
    dist += torch.sum(dst**2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = (
        torch.arange(B, dtype=torch.long)
        .to(device)
        .view(view_shape)
        .repeat(repeat_shape)
    )
    new_points = points[batch_indices, idx, :]
    return new_points


def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
    return group_idx


class LocalGrouper(nn.Module):
    def __init__(
        self,
        channel,
        sample_ratio,
        kneighbors,
        use_xyz=True,
        normalize="center",
        **kwargs,
    ):
        """
        Give xyz[b,p,3] and fea[b,p,d], return new_xyz[b,g,3] and new_fea[b,g,k,d]
        :param groups: groups number
        :param kneighbors: k-nerighbors
        :param kwargs: others
        """
        super(LocalGrouper, self).__init__()
        self.sample_ratio = sample_ratio
        self.kneighbors = kneighbors
        self.use_xyz = use_xyz
        if normalize is not None:
            self.normalize = normalize.lower()
        else:
            self.normalize = None
        if self.normalize not in ["center", "anchor"]:
            print(
                f"Unrecognized normalize parameter (self.normalize), set to None. Should be one of [center, anchor]."
            )
            self.normalize = None
        if self.normalize is not None:
            add_channel = 3 if self.use_xyz else 0
            self.affine_alpha = nn.Parameter(
                torch.ones([1, 1, 1, channel + add_channel])
            )
            self.affine_beta = nn.Parameter(
                torch.zeros([1, 1, 1, channel + add_channel])
            )

    def forward(self, xyz, points):
        B, N, C = xyz.shape
        S = N // self.sample_ratio
        xyz = xyz.contiguous()  # xyz [btach, points, xyz]

        fps_idx = furthest_point_sample(xyz, S).long()  # [B, npoint]
        new_xyz = index_points(xyz, fps_idx)  # [B, npoint, 3]
        new_points = index_points(points, fps_idx)  # [B, npoint, d]

        idx = knn_point(self.kneighbors, xyz, new_xyz)
        grouped_xyz = index_points(xyz, idx)  # [B, npoint, k, 3]
        grouped_points = index_points(points, idx)  # [B, npoint, k, d]
        if self.use_xyz:
            grouped_points = torch.cat(
                [grouped_points, grouped_xyz], dim=-1
            )  # [B, npoint, k, d+3]
        if self.normalize is not None:
            if self.normalize == "center":
                mean = torch.mean(grouped_points, dim=2, keepdim=True)
            if self.normalize == "anchor":
                mean = (
                    torch.cat([new_points, new_xyz], dim=-1)
                    if self.use_xyz
                    else new_points
                )
                mean = mean.unsqueeze(dim=-2)  # [B, npoint, 1, d+3]
            std = (
                torch.std((grouped_points - mean).reshape(B, -1), dim=-1, keepdim=True)
                .unsqueeze(dim=-1)
                .unsqueeze(dim=-1)
            )
            grouped_points = (grouped_points - mean) / (std + 1e-5)
            grouped_points = self.affine_alpha * grouped_points + self.affine_beta

        new_points = torch.cat(
            [
                grouped_points,
                new_points.view(B, S, 1, -1).repeat(1, 1, self.kneighbors, 1),
            ],
            dim=-1,
        )
        return new_xyz, new_points


class ConvBNReLU1D(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size=1, bias=True, activation="relu"
    ):
        super(ConvBNReLU1D, self).__init__()
        self.act = get_activation(activation)
        self.net = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                bias=bias,
            ),
            nn.BatchNorm1d(out_channels),
            self.act,
        )

    def forward(self, x):
        return self.net(x)


class ConvBNReLURes1D(nn.Module):
    def __init__(
        self,
        channel,
        kernel_size=1,
        groups=1,
        res_expansion=1.0,
        bias=True,
        activation="relu",
    ):
        super(ConvBNReLURes1D, self).__init__()
        self.act = get_activation(activation)
        self.net1 = nn.Sequential(
            nn.Conv1d(
                in_channels=channel,
                out_channels=int(channel * res_expansion),
                kernel_size=kernel_size,
                groups=groups,
                bias=bias,
            ),
            nn.BatchNorm1d(int(channel * res_expansion)),
            self.act,
        )
        if groups > 1:
            self.net2 = nn.Sequential(
                nn.Conv1d(
                    in_channels=int(channel * res_expansion),
                    out_channels=channel,
                    kernel_size=kernel_size,
                    groups=groups,
                    bias=bias,
                ),
                nn.BatchNorm1d(channel),
                self.act,
                nn.Conv1d(
                    in_channels=channel,
                    out_channels=channel,
                    kernel_size=kernel_size,
                    bias=bias,
                ),
                nn.BatchNorm1d(channel),
            )
        else:
            self.net2 = nn.Sequential(
                nn.Conv1d(
                    in_channels=int(channel * res_expansion),
                    out_channels=channel,
                    kernel_size=kernel_size,
                    bias=bias,
                ),
                nn.BatchNorm1d(channel),
            )

    def forward(self, x):
        return self.act(self.net2(self.net1(x)) + x)


class PreExtraction(nn.Module):
    def __init__(
        self,
        channels,
        out_channels,
        blocks=1,
        groups=1,
        res_expansion=1,
        bias=True,
        activation="relu",
        use_xyz=True,
    ):
        """
        input: [b,g,k,d]: output:[b,d,g]
        :param channels:
        :param blocks:
        """
        super(PreExtraction, self).__init__()
        in_channels = 3 + 2 * channels if use_xyz else 2 * channels
        self.transfer = ConvBNReLU1D(
            in_channels, out_channels, bias=bias, activation=activation
        )
        operation = []
        for _ in range(blocks):
            operation.append(
                ConvBNReLURes1D(
                    out_channels,
                    groups=groups,
                    res_expansion=res_expansion,
                    bias=bias,
                    activation=activation,
                )
            )
        self.operation = nn.Sequential(*operation)

    def forward(self, x):
        b, n, s, d = x.size()  # torch.Size([32, 512, 32, 6])
        x = x.permute(0, 1, 3, 2)
        x = x.reshape(-1, d, s)
        x = self.transfer(x)
        batch_size, _, _ = x.size()
        x = self.operation(x)  # [b, d, k]
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x = x.reshape(b, n, -1).permute(0, 2, 1)
        return x


class PosExtraction(nn.Module):
    def __init__(
        self,
        channels,
        blocks=1,
        groups=1,
        res_expansion=1,
        bias=True,
        activation="relu",
    ):
        """
        input[b,d,g]; output[b,d,g]
        :param channels:
        :param blocks:
        """
        super(PosExtraction, self).__init__()
        operation = []
        for _ in range(blocks):
            operation.append(
                ConvBNReLURes1D(
                    channels,
                    groups=groups,
                    res_expansion=res_expansion,
                    bias=bias,
                    activation=activation,
                )
            )
        self.operation = nn.Sequential(*operation)

    def forward(self, x):  # [b, d, g]
        return self.operation(x)


class PointNetFeaturePropagation(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        blocks=1,
        groups=1,
        res_expansion=1.0,
        bias=True,
        activation="relu",
        has_MLP=True,
    ):
        super(PointNetFeaturePropagation, self).__init__()
        if has_MLP:
            self.fuse = ConvBNReLU1D(in_channel, out_channel, 1, bias=bias)
            self.extraction = PosExtraction(
                out_channel,
                blocks,
                groups=groups,
                res_expansion=res_expansion,
                bias=bias,
                activation=activation,
            )
        self.has_MLP = has_MLP

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, N, 3]
            xyz2: sampled input points position data, [B, S, 3]
            points1: input points data, [B, D', N]
            points2: input points data, [B, D'', S]
        Return:
            new_points: upsampled points data, [B, D''', N]
        """
        # xyz1 = xyz1.permute(0, 2, 1)
        # xyz2 = xyz2.permute(0, 2, 1)

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(
                index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2
            )

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        if self.has_MLP:
            new_points = self.fuse(new_points)
            new_points = self.extraction(new_points)
        return new_points


@MODELS.register_module()
class PointMLPEncoder(nn.Module):
    def __init__(
        self,
        in_channels=4,
        embed_dim=64,
        groups=1,
        res_expansion=1.0,
        activation="relu",
        bias=False,
        use_xyz=True,
        normalize="anchor",
        dim_expansion=[2, 2, 2, 2],
        pre_blocks=[2, 2, 2, 2],
        pos_blocks=[2, 2, 2, 2],
        k_neighbors=[32, 32, 32, 32],
        reducers=[2, 2, 2, 2],
        de_blocks=[2, 2, 2, 2],
        de_dims=[512, 256, 128, 128],
        **kwargs,
    ):
        """Initialize PointMLP Encoder.

        Args:
            in_channels (int): Input channel dimension
            embed_dim (int): Initial embedding dimension
            groups (int): Groups for convolution
            res_expansion (float): Expansion ratio for residual blocks
            activation (str): Activation function type
            bias (bool): Whether to use bias in layers
            use_xyz (bool): Whether to use XYZ coordinates
            normalize (str): Normalization type ('anchor' or 'center')
            dim_expansion (list): Dimension expansion ratios for each stage
            pre_blocks (list): Number of pre-extraction blocks per stage
            pos_blocks (list): Number of post-extraction blocks per stage
            k_neighbors (list): Number of neighbors for each stage
            reducers (list): Reduction ratios for each stage
            de_blocks (list): Number of decoder blocks
            de_dims (list): Decoder dimensions
        """
        super(PointMLPEncoder, self).__init__()
        self.in_channels = in_channels
        self.stages = len(pre_blocks)
        self.embedding = ConvBNReLU1D(
            in_channels, embed_dim, bias=bias, activation=activation
        )
        self.use_fusion = kwargs.get("use_fusion", True)
        # Validate input parameters
        assert (
            len(pre_blocks)
            == len(k_neighbors)
            == len(reducers)
            == len(pos_blocks)
            == len(dim_expansion)
        ), "Stage numbers must be consistent across all parameters"

        # Initialize encoder components
        self.local_grouper_list = nn.ModuleList()
        self.pre_blocks_list = nn.ModuleList()
        self.pos_blocks_list = nn.ModuleList()
        last_channel = embed_dim
        self.channels = [embed_dim]

        # Build encoder stages
        for i in range(len(pre_blocks)):
            out_channel = last_channel * dim_expansion[i]
            self.channels.append(out_channel)

            # Create local grouper
            local_grouper = LocalGrouper(
                last_channel, reducers[i], k_neighbors[i], use_xyz, normalize
            )
            self.local_grouper_list.append(local_grouper)

            # Create pre-extraction block
            pre_block_module = PreExtraction(
                last_channel,
                out_channel,
                pre_blocks[i],
                groups=groups,
                res_expansion=res_expansion,
                bias=bias,
                activation=activation,
                use_xyz=use_xyz,
            )
            self.pre_blocks_list.append(pre_block_module)

            # Create post-extraction block
            pos_block_module = PosExtraction(
                out_channel,
                pos_blocks[i],
                groups=groups,
                res_expansion=res_expansion,
                bias=bias,
                activation=activation,
            )
            self.pos_blocks_list.append(pos_block_module)

            last_channel = out_channel

        self.out_channels = last_channel

        # Build decoder
        self.decode_list = nn.ModuleList()
        self.channels.reverse()
        en_dims = self.channels
        de_dims.insert(0, en_dims[0])

        assert (
            len(en_dims) == len(de_dims) == len(de_blocks) + 1
        ), "Decoder dimensions mismatch"

        # Create decoder layers
        for i in range(len(de_dims) - 1):
            self.decode_list.append(
                PointNetFeaturePropagation(
                    de_dims[i] + en_dims[i + 1],
                    de_dims[i + 1],
                    blocks=de_blocks[i],
                    groups=groups,
                    res_expansion=res_expansion,
                    bias=bias,
                    activation=activation,
                )
            )

        self.channels.reverse()
        self.act = get_activation(activation)

    def forward(
        self, p, image_features, c2w_projection_matrix, feature_mlps, intrinsic
    ):
        """Forward pass of PointMLP Encoder.

        Args:
            p: Point cloud data (can be dict with 'pos' key or tensor)
            image_features: Features from image branch
            c2w_projection_matrix: Camera-to-world projection matrix
            feature_mlps: MLPs for feature processing
            intrinsic: Camera intrinsic parameters
            resolution: Image resolution
            x: Optional input features

        Returns:
            tuple: (processed features, final point positions)
        """
        # Handle input format
        if isinstance(p, dict):
            p, x = p["pos"], p.get("x", None)
        if x is None:
            x = p.transpose(1, 2).contiguous()

        # Initial embedding
        x = self.embedding(x)  # [B,D,N]
        p_list, x_list = [p], [x]

        # Initialize feature fusion module
        if self.use_fusion:
            feature_fusion = FeatureFusion(feature_mlps)

        # Encoder forward pass
        for i in range(self.stages):
            p, x = self.local_grouper_list[i](p, x.permute(0, 2, 1))
            x = self.pre_blocks_list[i](x)
            x = self.pos_blocks_list[i](x)
            p_list.append(p)
            x_list.append(x)

        # Reverse lists for decoder
        p_list.reverse()
        x_list.reverse()
        x = x_list[0]

        # Decoder forward pass with feature fusion
        if self.use_fusion:
            fusion_layer = [len(self.decode_list) - 1]
        else:
            fusion_layer = []

        for i in range(len(self.decode_list)):
            x = self.decode_list[i](p_list[i + 1], p_list[i], x_list[i + 1], x)

            # Apply feature fusion at specified layers
            if i in fusion_layer:
                x = feature_fusion(
                    x.transpose(2, 1),
                    p_list[i + 1][..., :3],
                    image_features,
                    c2w_projection_matrix,
                    intrinsic,
                )

        return x, p_list[-1]


# -------- There is Point Mlp Original Model Config
def pointMLP(num_classes=40, cfg=None, **kwargs) -> PointMLPEncoder:
    return PointMLPEncoder(
        in_channels=cfg.model.in_channels,
        num_classes=num_classes,
        embed_dim=64,
        groups=1,
        res_expansion=1.0,
        activation="relu",
        bias=False,
        use_xyz=False,
        normalize="anchor",
        dim_expansion=[2, 2, 2, 2],
        pre_blocks=[2, 2, 2, 2],
        pos_blocks=[2, 2, 2, 2],
        k_neighbors=[24, 24, 24, 24],
        reducers=[2, 2, 2, 2],
        de_dims=[512, 256, 128, 128],
        **kwargs,
    )