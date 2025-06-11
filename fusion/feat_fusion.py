import torch
import torch.nn as nn


class FeatureFusion:
    """
    A class for fusing 3D point features with 2D image features through projection.

    This class handles the projection of 3D points into 2D image space and
    extracts corresponding image features for each point, considering occlusion
    through depth comparison.
    """

    def __init__(self, fusion_mlp: nn.Module):
        """
        Initialize the FeatureFusion module.

        Args:
            fusion_mlp (nn.Module): MLP network for processing concatenated features
        """
        self.fusion_mlp = fusion_mlp

    def project_points_to_image(
        self, center: torch.Tensor, c2w_matrix: torch.Tensor, intrinsic: torch.Tensor
    ) -> tuple:
        """
        Project 3D points to 2D image space using camera parameters.

        Args:
            center (torch.Tensor): 3D points in world space [B, N, 3]
            c2w_matrix (torch.Tensor): Camera-to-world matrix [B, 4, 4]
            intrinsic (torch.Tensor): Camera intrinsic matrix [B, 3, 3]

        Returns:
            tuple: Projected pixel coordinates and depth values
        """
        # Add homogeneous coordinate
        coords_homogeneous = torch.concatenate(
            [center, torch.ones([*center.shape[:2], 1], device=center.device)], dim=2
        )
        # Transform points from world to camera space
        w2c_matrix = torch.linalg.inv(c2w_matrix.permute(0, 2, 1))
        camera_points = torch.matmul(
            w2c_matrix, coords_homogeneous.transpose(1, 2)
        ).transpose(1, 2)

        # Perspective projection
        pixel_coords = camera_points.clone()
        pixel_coords[..., 0] = (
            camera_points[..., 0] * intrinsic[0][0]
        ) / camera_points[..., 2] + intrinsic[0][2]
        pixel_coords[..., 1] = (
            camera_points[..., 1] * intrinsic[1][1]
        ) / camera_points[..., 2] + intrinsic[1][2]

        return torch.round(pixel_coords[..., :2]).long(), camera_points[..., 2]

    def __call__(
        self,
        x: torch.Tensor,
        center: torch.Tensor,
        image_features: torch.Tensor,
        c2w_projection_matrix: torch.Tensor,
        intrinsic: torch.Tensor,
    ) -> torch.Tensor:
        """
        Perform feature fusion between 3D points and 2D image features.

        Args:
            x (torch.Tensor): Point features [B, N, C]
            center (torch.Tensor): 3D points in world space [B, N, 3]
            image_features (torch.Tensor): Image features from UNet decoder [B, C, H, W]
            c2w_projection_matrix (torch.Tensor): Camera-to-world matrix [B, 4, 4]
            intrinsic (torch.Tensor): Camera intrinsic matrix [B, 3, 3]

        Returns:
            torch.Tensor: Fused features [B, C', N]
        """
        B, N = center.shape[:2]
        C, H, W = image_features.shape[1:]

        if c2w_projection_matrix.dim() == 4:
            c2w_projection_matrix = c2w_projection_matrix[:, 0]

        # Project 3D points to 2D image space
        pi_xy, p_depth = self.project_points_to_image(
            center, c2w_projection_matrix, intrinsic
        )

        # Check which points project inside the image bounds
        inside_mask = (
            (pi_xy[..., 0] >= 0)
            & (pi_xy[..., 1] >= 0)
            & (pi_xy[..., 0] < H)
            & (pi_xy[..., 1] < W)
            & (p_depth >= 0)
        )

        # Get valid point indices
        valid_indices = torch.nonzero(inside_mask)

        batch_indices, point_indices = valid_indices[:, 0], valid_indices[:, 1]
        pixel_x = pi_xy[batch_indices, point_indices, 0]
        pixel_y = pi_xy[batch_indices, point_indices, 1]
        valid_depth = p_depth[batch_indices, point_indices]

        # Handle occlusions using depth comparison
        unique_ids = batch_indices * (H * W) + pixel_y * H + pixel_x
        max_id = unique_ids.max().item() + 1
        min_depths = torch.full(
            (max_id,), float("inf"), device=valid_depth.device, dtype=valid_depth.dtype
        )
        min_depths.scatter_reduce_(
            0, unique_ids, valid_depth, reduce="amin", include_self=False
        )

        # Keep only the closest points for each pixel
        min_depth_mask = valid_depth == min_depths[unique_ids]

        # Initialize output feature tensor
        mapped_features = torch.zeros((B, N, C), device=center.device)

        # Assign features from valid projections
        mapped_features[
            batch_indices[min_depth_mask], point_indices[min_depth_mask]
        ] = image_features[
            batch_indices[min_depth_mask],
            :,
            pixel_x[min_depth_mask],
            pixel_y[min_depth_mask],
        ]
        # # Concatenate original features with mapped features
        x_num = x.shape[1]
        if x_num > N:
            # Transformer CLS Token
            x_patch = torch.cat([x[:, 1:], mapped_features], dim=-1)
            CLS_token_features = torch.cat(
                [x[:, 0:1], torch.zeros((B, 1, C), device=center.device)], dim=-1
            )
            x = torch.cat([CLS_token_features, x_patch], dim=1)
        else:
            x = torch.cat([x, mapped_features], dim=-1)

        # Apply feature MLP and return
        return self.fusion_mlp(x)
