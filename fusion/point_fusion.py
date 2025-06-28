#!/usr/bin/env python
#!/usr/bin/env python

import torch
from torch import nn
import spconv.pytorch as spconv
from pointcept.datasets.transform_with_extrinsic import GridSample


class PointFusion(nn.Module):
    """
    A module that links 2D and 3D features through unprojection and fusion.

    This module performs the following main operations:
    1. Projects 2D features into 3D space
    2. Fuses multiple view features
    3. Combines the projected features with existing 3D features
    """

    def __init__(self, fusion_mlp: nn.Module, fea2d_dim: int = 128, viewNum: int = 8):
        """
        Initialize the PointFusion module.

        Args:
            fea2d_dim (int): Dimension of 2D features
            fea3d_dim (int): Dimension of 3D features
            viewNum (int): Number of views to process
        """
        super(PointFusion, self).__init__()
        self.viewNum = viewNum
        self.fea2d_dim = fea2d_dim

        # Network for final 3D feature fusion
        self.fuseTo3d = fusion_mlp

    def forward(
        self,
        feat_2d_all: torch.Tensor,
        feat_3d: spconv.SparseConvTensor,
        unprojected_coord: torch.Tensor,
        init_3d_data: dict,
        grid_size: float = 0.02,
    ) -> spconv.SparseConvTensor:
        """
        Project 2D features to 3D space and fuse them with existing 3D features.

        Args:
            feat_2d_all: [V_B * C * H * W] 2D features from multiple views
            feat_3d: SparseTensor containing current 3D point cloud features
            unprojected_coord: [N, 3, H, W] Unprojected 3D coordinates
            init_3d_data: Dictionary containing initial 3D data
            grid_size: Size of the grid for sampling (default: 0.02)

        Returns:
            SparseTensor containing fused 2D and 3D features
        """
        # Initialize tensor for storing unprojected features
        feat_2d_to_3d = torch.zeros(
            [feat_3d.features.shape[0], self.viewNum, self.fea2d_dim],
            dtype=torch.float,
            device="cuda",
        )

        # Initialize grid sampler
        grid_sampler = GridSample(
            grid_size=grid_size,
            hash_type="fnv",
            mode="train",
            keys=("coord", "feat"),
            return_grid_coord=True,
            return_inverse=True,
            min_coord=init_3d_data["coord"].min(0)[0].detach().cpu().numpy(),
        )

        # Process unprojected coordinates
        unprojected_coord = unprojected_coord[0].contiguous()
        N, H, W, _ = unprojected_coord.shape
        unprojected_coord = unprojected_coord.view(-1, 4)

        # Filter valid coordinates
        valid_mask = unprojected_coord[..., 3].bool()
        valid_coords = unprojected_coord[valid_mask][:, :3]

        # Apply bounding box constraints
        bouding_mask = self._get_bounding_mask(valid_coords, init_3d_data["coord"])
        valid_coords = valid_coords[bouding_mask]

        # Process 2D features
        feat_2d_all_flat = (
            feat_2d_all.permute((0, 2, 3, 1))
            .contiguous()
            .view(-1, feat_2d_all.shape[-3])
        )
        valid_feat_2d_all = feat_2d_all_flat[valid_mask][bouding_mask]

        # Prepare data for grid sampling
        data_dict = {
            "coord": valid_coords.cpu().numpy(),
            "feat": valid_feat_2d_all,
        }

        # Handle empty case
        if len(data_dict["coord"]) == 0:
            print("Warning: No valid unprojected coordinates.")
            return feat_3d

        # Apply grid sampling
        data_dict = grid_sampler(data_dict)

        # Create sparse tensor from grid sampled data
        grid_sampled_coords = torch.tensor(data_dict["grid_coord"], device="cuda")
        indices = torch.cat(
            [
                torch.zeros(
                    grid_sampled_coords.shape[0], 1, dtype=torch.int32, device="cuda"
                ),
                grid_sampled_coords.int(),
            ],
            dim=1,
        ).contiguous()

        # Create sparse tensor for 2D features
        feat_2d_sparse = spconv.SparseConvTensor(
            data_dict["feat"],
            indices,
            feat_3d.spatial_shape,
            feat_3d.batch_size,
        )

        # Combine 2D and 3D features
        return self._fuse_features(feat_3d, feat_2d_sparse, init_3d_data, data_dict)

    def _get_bounding_mask(
        self, coords: torch.Tensor, reference_coords: torch.Tensor
    ) -> torch.Tensor:
        """
        Create a mask for coordinates within the bounding box of reference coordinates.
        """
        return (
            (coords[:, 0] >= reference_coords[:, 0].min())
            & (coords[:, 0] <= reference_coords[:, 0].max())
            & (coords[:, 1] >= reference_coords[:, 1].min())
            & (coords[:, 1] <= reference_coords[:, 1].max())
            & (coords[:, 2] >= reference_coords[:, 2].min())
            & (coords[:, 2] <= reference_coords[:, 2].max())
        )

    def _fuse_features(
        self,
        feat_3d: spconv.SparseConvTensor,
        feat_2d_to_3d: spconv.SparseConvTensor,
        init_3d_data: dict,
        data_dict: dict,
    ) -> spconv.SparseConvTensor:
        """
        Fuse 2D and 3D features and update the initial 3D data.
        """
        # Combine features and indices
        all_indices = torch.cat([feat_3d.indices, feat_2d_to_3d.indices], dim=0)
        all_features = torch.cat([feat_3d.features, feat_2d_to_3d.features], dim=0)

        # Create combined sparse tensor
        combined_tensor = spconv.SparseConvTensor(
            features=all_features,
            indices=all_indices,
            spatial_shape=feat_3d.spatial_shape,
            batch_size=feat_3d.batch_size,
        )

        # Apply final fusion
        fused_3d = self.fuseTo3d(combined_tensor)

        # Update initial 3D data
        self._update_init_3d_data(init_3d_data, data_dict, all_indices)

        return fused_3d

    def _update_init_3d_data(
        self, init_3d_data: dict, data_dict: dict, all_indices: torch.Tensor
    ):
        """
        Update the initial 3D data with new coordinates and batch information.
        """
        init_coords = init_3d_data["coord"]
        feat_2d_selected_coords = torch.from_numpy(data_dict["coord"]).to(
            init_coords.device
        )

        init_3d_data.update(
            {
                "coord": torch.cat((init_coords, feat_2d_selected_coords), dim=0),
                "batch": all_indices[:, 0],
                "grid_coord": all_indices[:, 1:],
            }
        )
