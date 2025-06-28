import glob
import os
import random
import numpy as np
import math
import torch
from torch.utils.data import Dataset

from .dataset_readers import readCamerasFromTxt
from utils.general_utils import PILtoTorch
from utils.graphics_utils import (
    getWorld2View2,
    getProjectionMatrix,
    getView2World,
)
from pointcept.datasets.transform_with_extrinsic import (
    GridSample,
    ToTensor,
    NormalizeColor,
    Collect,
    FPS,
    RandomJitter,
    ChromaticAutoContrast,
    ChromaticTranslation,
    ChromaticJitter,
    CenterShift,
    RandomRotate,
    RandomFlip,
    RandomScale,
)

# Global constants
SCANNET_PT_DATASET_ROOT = "/remote-home/share/PTScannet/"
SCANNET_COLOR_DATASET_ROOT = (
    "/remote-home/share/dataset/datasets_wzy/datasets/ScanNetV2/Segmentation/2D"
)


class ScanNetDataset(Dataset):
    """
    Dataset class for ScanNet, handling point cloud and image data.
    Supports both training and testing modes with various data augmentation options.
    """

    VALID_ASSETS = ["coord", "color", "normal", "segment20", "instance"]

    def __init__(self, cfg, dataset_name="train"):
        """
        Initialize the ScanNet dataset.

        Args:
            cfg: Configuration object containing model and training parameters
            dataset_name: Dataset split name ('train', 'val', or 'test')
        """
        super().__init__()
        self.cfg = cfg
        self.dataset_name = dataset_name
        self.use_ref_images = self.cfg.opt.use_fusion
        self.base_path = SCANNET_PT_DATASET_ROOT

        # Get metadata paths based on mode
        if self.cfg.opt.mode == "test":
            self.metadata = glob.glob(os.path.join(self.base_path, "train", "*"))
        else:
            self.metadata = glob.glob(os.path.join(self.base_path, dataset_name, "*"))

        self.dataset_len = len(self.metadata)

        # Initialize projection matrix
        self.projection_matrix = self._init_projection_matrix()
        self.imgs_per_obj = self.cfg.opt.imgs_per_obj

        # Initialize transforms based on configuration
        self.transforms = self._init_transforms()

        # Initialize link creator for point-image correspondence
        self.link_creator = LinkCreator()

    def _init_projection_matrix(self):
        """Initialize the projection matrix based on configuration parameters."""
        return getProjectionMatrix(
            znear=self.cfg.data.znear,
            zfar=self.cfg.data.zfar,
            fovX=self.cfg.data.fov * 2 * np.pi / 360,
            fovY=self.cfg.data.fov * 2 * np.pi / 360,
        ).transpose(0, 1)

    def _init_transforms(self):
        """Initialize data transforms based on configuration and dataset split."""
        if self.cfg.model.aug and self.dataset_name == "train":
            return self._get_training_transforms()
        else:
            return self._get_evaluation_transforms()

    def _get_training_transforms(self):
        """Get data augmentation transforms for training."""
        return [
            CenterShift(apply_z=True, keys=["extrinsic"]),
            RandomRotate(
                angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5, keys=["extrinsic"]
            ),
            RandomRotate(angle=[-1 / 64, 1 / 64], axis="x", p=0.5, keys=["extrinsic"]),
            RandomRotate(angle=[-1 / 64, 1 / 64], axis="y", p=0.5, keys=["extrinsic"]),
            RandomJitter(sigma=0.005, clip=0.02),
            ChromaticAutoContrast(p=0.2, blend_factor=None),
            ChromaticTranslation(p=0.95, ratio=0.05),
            ChromaticJitter(p=0.95, std=0.05),
            GridSample(
                grid_size=0.02,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
                return_inverse=True,
            ),
            CenterShift(apply_z=False, keys=["extrinsic"]),
            NormalizeColor(),
            ToTensor(),
            Collect(
                keys=("coord", "grid_coord", "segment", "inverse"),
                stack_keys=("extrinsic", "gt_images", "depth"),
                feat_keys=("normal", "color"),
            ),
        ]

    def _get_evaluation_transforms(self):
        """Get transforms for evaluation (validation/testing)."""
        if self.cfg.model.backbone_type == "sparseunet":
            return [
                GridSample(
                    grid_size=0.02,
                    hash_type="fnv",
                    mode="train",
                    return_grid_coord=True,
                    return_inverse=True,
                ),
                NormalizeColor(),
                ToTensor(),
                Collect(
                    keys=("coord", "grid_coord", "segment", "inverse"),
                    feat_keys=("normal", "color"),
                    stack_keys=("extrinsic", "gt_images", "depth"),
                ),
            ]
        elif self.cfg.model.backbone_type == "ptv3":
            return [
                GridSample(
                    grid_size=0.02,
                    hash_type="fnv",
                    mode="train",
                    return_grid_coord=True,
                    return_inverse=True,
                ),
                NormalizeColor(),
                ToTensor(),
                Collect(
                    keys=("coord", "segment", "grid_coord", "inverse"),
                    feat_keys=("color", "normal"),
                    stack_keys=("extrinsic", "gt_images", "depth"),
                ),
                FPS(max_points=80000),
            ]

    def __len__(self):
        """Return the total number of samples in the dataset."""
        return len(self.metadata)

    def load_example_id(self, example_id, metadata_path):
        """
        Load and process data for a specific example.

        Args:
            example_id: Unique identifier for the example
            metadata_path: Path to the example's metadata
        """
        # Initialize data structures if not already done
        if not hasattr(self, "all_rgbs"):
            self._initialize_data_containers()

        # Skip if example already loaded
        if example_id in self.all_rgbs:
            return

        # Get image paths
        rgb_paths, pose_paths, depth_paths = self._get_image_paths(metadata_path)

        # Initialize containers for current example
        self._initialize_example_containers(example_id)

        # Load point cloud data
        data_dict = self._load_point_cloud_data(metadata_path)

        # Process point cloud data
        moving_centers = self._process_point_cloud_data(example_id, data_dict)

        # Process camera data
        self._process_camera_data(example_id, rgb_paths, pose_paths, depth_paths, moving_centers=moving_centers)

    def get_example_id(self, index: int) -> str:
        """
        Get the example ID for a given index.

        Args:
            index: Index of the sample

        Returns:
            Unique identifier for the example
        """
        metadata_path = self.metadata[index]
        parent_dir, category_instance_dir = os.path.split(metadata_path)
        category, instance = os.path.split(parent_dir)
        return f"{instance}-{category_instance_dir}"
    
    def _initialize_data_containers(self):
        """Initialize all data containers for the dataset."""
        self.all_rgbs = {}
        self.all_w2c = {}
        self.all_world_view_transforms = {}
        self.all_view_to_world_transforms = {}
        self.all_full_proj_transforms = {}
        self.all_camera_centers = {}
        self.all_pts_coord = {}
        self.all_pts_color = {}
        self.all_pts_normal = {}
        self.all_pts_segment = {}
        self.all_pts_instance = {}
        self.all_unprojected_coords = {}
        self.all_depth = {}
        self.original_coords_len = {}

    def _get_image_paths(self, metadata_path):
        """Get sorted paths for RGB, pose, and depth images."""
        path_parts = metadata_path.split("/")
        last_path = path_parts[-1]

        def extract_number(filename):
            """Extract number from filename for sorting."""
            import re

            match = re.search(r"\d+", filename)
            return int(match.group()) if match else -1

        rgb_paths = sorted(
            glob.glob(
                os.path.join(SCANNET_COLOR_DATASET_ROOT, "color", last_path, "*.jpg")
            ),
            key=lambda x: extract_number(os.path.basename(x)),
        )
        pose_paths = sorted(
            glob.glob(
                os.path.join(SCANNET_COLOR_DATASET_ROOT, "pose", last_path, "*.txt")
            ),
            key=lambda x: extract_number(os.path.basename(x)),
        )
        depth_paths = sorted(
            glob.glob(
                os.path.join(SCANNET_COLOR_DATASET_ROOT, "depth", last_path, "*.png")
            ),
            key=lambda x: extract_number(os.path.basename(x)),
        )

        assert (
            len(rgb_paths) == len(pose_paths) == len(depth_paths)
        ), "Mismatched number of images"
        return rgb_paths, pose_paths, depth_paths

    def _initialize_example_containers(self, example_id):
        """Initialize data containers for a specific example."""
        self.all_rgbs[example_id] = []
        self.all_w2c[example_id] = []
        self.all_world_view_transforms[example_id] = []
        self.all_full_proj_transforms[example_id] = []
        self.all_camera_centers[example_id] = []
        self.all_view_to_world_transforms[example_id] = []
        self.all_pts_coord[example_id] = []
        self.all_pts_color[example_id] = []
        self.all_pts_normal[example_id] = []
        self.all_pts_segment[example_id] = []
        self.all_pts_instance[example_id] = []
        self.all_unprojected_coords[example_id] = []
        self.all_depth[example_id] = []

    def _load_point_cloud_data(self, metadata_path):
        """Load point cloud data from .npy files."""
        data_dict = {}
        for asset in os.listdir(metadata_path):
            if not asset.endswith(".npy") or asset[:-4] not in self.VALID_ASSETS:
                continue
            data_dict[asset[:-4]] = np.load(os.path.join(metadata_path, asset))
        return data_dict

    def _process_point_cloud_data(self, example_id, data_dict):
        """Process and store point cloud data for an example."""
        self.all_pts_coord[example_id] = data_dict["coord"]
        self.all_pts_color[example_id] = data_dict["color"]
        self.all_pts_normal[example_id] = data_dict["normal"]

        # Process segment data
        if "segment20" in data_dict:
            self.all_pts_segment[example_id] = (
                data_dict.pop("segment20").reshape([-1]).astype(np.int32)
            )
        elif "segment200" in data_dict:
            self.all_pts_segment[example_id] = (
                data_dict.pop("segment200").reshape([-1]).astype(np.int32)
            )
        else:
            self.all_pts_segment[example_id] = (
                np.ones(data_dict["coord"].shape[0], dtype=np.int32) * -1
            )

        # Process instance data
        instance_data = data_dict.get(
            "instance", np.ones(data_dict["coord"].shape[0], dtype=np.int32) * -1
        )
        self.all_pts_instance[example_id] = instance_data.reshape([-1])

        # Center the coordinates
        moving_centers = self.all_pts_coord[example_id].mean(axis=0)
        self.all_pts_coord[example_id] -= moving_centers
        return moving_centers

    def _process_camera_data(
        self, example_id, rgb_paths, pose_paths, depth_paths, moving_centers=None
    ):
        """Process and store camera data for an example."""
        cam_infos = readCamerasFromTxt(
            rgb_paths,
            pose_paths,
            list(range(len(rgb_paths))),
            fov=self.cfg.data.fov,
            moving_centers=moving_centers,
            depth_paths=depth_paths,
            dataset_type="scannet",
        )

        for cam_info in cam_infos:
            # Process image data
            resized_image = cam_info.image.resize(
                (self.cfg.data.training_width, self.cfg.data.training_height)
            )
            resized_image = np.array(resized_image).astype(np.float32)
            self.all_rgbs[example_id].append(resized_image.transpose(2, 0, 1))
            self.all_w2c[example_id].append(torch.from_numpy(cam_info.w2c))

            # Process camera transforms
            world_view_transform = torch.tensor(
                getWorld2View2(cam_info.R, cam_info.T)
            ).transpose(0, 1)
            view_world_transform = torch.tensor(
                getView2World(cam_info.R, cam_info.T)
            ).transpose(0, 1)

            # Process projection and camera center
            full_proj_transform = (
                world_view_transform.unsqueeze(0).bmm(
                    self.projection_matrix.unsqueeze(0)
                )
            ).squeeze(0)
            camera_center = world_view_transform.inverse()[3, :3]

            # Process depth and unprojection
            depth_tensor = PILtoTorch(
                cam_info.depth,
                (self.cfg.data.training_width, self.cfg.data.training_height),
                is_depth=True,
            )[0, :, :]

            if self.use_ref_images:
                unprojected_coords = self.link_creator.computeUnprojection(
                    camera_to_world=view_world_transform, depth=depth_tensor
                )
                self.all_unprojected_coords[example_id].append(unprojected_coords)
                self.all_depth[example_id].append(np.array(depth_tensor))

            # Store transforms
            self.all_world_view_transforms[example_id].append(world_view_transform)
            self.all_view_to_world_transforms[example_id].append(view_world_transform)
            self.all_full_proj_transforms[example_id].append(full_proj_transform)
            self.all_camera_centers[example_id].append(camera_center)

        # Stack all camera-related tensors
        self._stack_camera_tensors(example_id)

    def _stack_camera_tensors(self, example_id):
        """Stack all camera-related tensors for an example."""
        self.all_world_view_transforms[example_id] = torch.stack(
            self.all_world_view_transforms[example_id]
        )
        self.all_view_to_world_transforms[example_id] = torch.stack(
            self.all_view_to_world_transforms[example_id]
        )
        self.all_full_proj_transforms[example_id] = torch.stack(
            self.all_full_proj_transforms[example_id]
        )
        self.all_camera_centers[example_id] = torch.stack(
            self.all_camera_centers[example_id]
        )
        if self.use_ref_images:
            self.all_unprojected_coords[example_id] = torch.stack(
                self.all_unprojected_coords[example_id]
            )
        self.all_rgbs[example_id] = np.stack(self.all_rgbs[example_id])
        self.all_w2c[example_id] = torch.stack(self.all_w2c[example_id])
        if hasattr(self, "all_depth") and example_id in self.all_depth and len(self.all_depth[example_id]) > 0:
            self.all_depth[example_id] = np.stack(self.all_depth[example_id])

    def __getitem__(self, index):
        """
        Get a data sample by index.

        Args:
            index: Index of the sample to get

        Returns:
            Dictionary containing processed point cloud and image data
        """
        metadata_path = self.metadata[index]
        parent_dir, category_instance_dir = os.path.split(metadata_path)
        category, instance = os.path.split(parent_dir)
        example_id = f"{instance}-{category_instance_dir}"

        # Load example data if not already loaded
        self.load_example_id(example_id, metadata_path)

        # Get number of available images
        num_images = len(self.all_rgbs[example_id])

        # Check if we have enough images
        min_required_images = 2 * self.cfg.data.input_images
        if num_images < min_required_images:
            return self.__getitem__(random.randint(0, len(self.metadata) - 1))

        # Select frames based on dataset mode
        frame_idxs = self._select_frames(num_images)

        # Process point cloud data
        pts_data = self._process_points_for_sample(example_id, frame_idxs)

        # Process camera and image data
        camera_data = self._process_cameras_for_sample(example_id, frame_idxs)

        # Combine and return all data
        return {**pts_data, **camera_data}

    def _select_frames(self, num_images):
        """
        Select frames for training or evaluation based on configuration.

        Args:
            num_images: Total number of available images

        Returns:
            List of selected frame indices
        """
        if (
            self.dataset_name == "train"
            or self.dataset_name == "val"
        ):
            return self._select_training_frames(num_images)
        else:
            return self._select_evaluation_frames(num_images)

    def _select_training_frames(self, num_images):
        """Select frames for training with various sampling strategies."""
        # Calculate number of GT images needed
        gt_images_num = int(self.cfg.data.input_images)

        # Divide images into subsequences and select GT frames
        sub_seq_len = num_images // gt_images_num
        remainder = num_images % gt_images_num

        # Create subsequences
        sub_sequences = []
        start = 0
        for i in range(gt_images_num):
            end = start + sub_seq_len + (1 if i < remainder else 0)
            sub_sequences.append(list(range(start, end)))
            start = end

        # Select GT frames from each subsequence
        gt_images_idxs = [random.choice(sub_seq) for sub_seq in sub_sequences]

        if self.use_ref_images:
            ref_images_idxs = self._select_reference_frames(num_images, gt_images_idxs)
            return ref_images_idxs + gt_images_idxs
        return gt_images_idxs

    def _select_reference_frames(self, num_images, gt_images_idxs):
        """Select reference frames based on configuration strategy."""
        if self.cfg.data.use_neighbor_imgs:
            return self._select_neighbor_frames(num_images, gt_images_idxs)
        else:
            return self._select_random_frames(num_images, gt_images_idxs)

    def _select_neighbor_frames(self, num_images, gt_images_idxs):
        """Select frames from neighboring frames of GT images."""
        if self.cfg.data.use_self_supervise:
            return gt_images_idxs

        ref_images_idxs = []
        for idx in gt_images_idxs:
            start = max(0, idx - self.cfg.data.supervised_max_distance)
            end = min(num_images, idx + self.cfg.data.supervised_max_distance + 1)
            possible_idxs = [i for i in range(start, end) if i != idx]

            if possible_idxs:
                ref_images_idxs.append(random.choice(possible_idxs))
            else:
                ref_images_idxs.append(None)

        return ref_images_idxs

    def _select_random_frames(self, num_images, gt_images_idxs):
        """Select frames randomly from remaining frames."""
        if self.cfg.data.use_self_supervise:
            return gt_images_idxs

        remain_indices = [i for i in range(num_images) if i not in gt_images_idxs]
        ref_images_num = self.cfg.data.input_images

        # Divide remaining images into subsequences
        sub_seq_len = len(remain_indices) // ref_images_num
        remainder = len(remain_indices) % ref_images_num

        sub_sequences = []
        start = 0
        for i in range(ref_images_num):
            end = start + sub_seq_len + (1 if i < remainder else 0)
            sub_sequences.append(remain_indices[start:end])
            start = end

        return [random.choice(sub_seq) for sub_seq in sub_sequences]

    def _select_evaluation_frames(self, num_images):
        """Select frames for evaluation."""
        frame_idxs = list(range(num_images))

        if self.use_ref_images:
            # Select reference frames evenly across the sequence
            sub_seq_len = num_images // self.cfg.data.input_images
            remainder = num_images % self.cfg.data.input_images

            sub_sequences = []
            start = 0
            for i in range(self.cfg.data.input_images):
                end = start + sub_seq_len + (1 if i < remainder else 0)
                sub_sequences.append(list(range(start, end)))
                start = end

            ref_images_idxs = [random.choice(sub_seq) for sub_seq in sub_sequences]
            return ref_images_idxs + frame_idxs

        return frame_idxs

    def _process_points_for_sample(self, example_id, frame_idxs):
        """Process point cloud data for a sample."""
        pts_data = {
            "coord": self.all_pts_coord[example_id].copy(),
            "color": self.all_pts_color[example_id].copy(),
            "normal": self.all_pts_normal[example_id].copy(),
            "segment": self.all_pts_segment[example_id].copy(),
            "instance": self.all_pts_instance[example_id].copy(),
            "extrinsic": self.all_w2c[example_id][frame_idxs].clone(),
            "gt_images": self.all_rgbs[example_id][frame_idxs].copy(),
            "depth": self.all_depth[example_id][frame_idxs].copy() if self.use_ref_images else torch.zeros(
                (len(frame_idxs), self.cfg.data.training_height, self.cfg.data.training_width))
        }

        for transform in self.transforms:
            pts_data = transform(pts_data)
        pts_data["gt_images"] = pts_data["gt_images"][0]

        return pts_data

    def _process_cameras_for_sample(self, example_id, frame_idxs):
        """Process camera and image data for a sample."""
        camera_data = {
            "world_view_transforms": self.all_world_view_transforms[example_id][
                frame_idxs
            ],
            "view_to_world_transforms": self.all_view_to_world_transforms[example_id][
                frame_idxs
            ],
            "full_proj_transforms": self.all_full_proj_transforms[example_id][
                frame_idxs
            ],
            "camera_centers": self.all_camera_centers[example_id][frame_idxs],
            "unprojected_coords": self.all_unprojected_coords[example_id][frame_idxs],
        }
        if self.use_ref_images:
            camera_data["unprojected_coords"] = self.all_unprojected_coords[example_id][
                frame_idxs
            ]
        return camera_data


class LinkCreator:
    """Handle point-image correspondence computation."""

    def __init__(
        self,
        fx=144.46765125,
        fy=144.46765125,
        mx=79.5,
        my=59.5,
        image_dim=(160, 120),
        voxel_size=0.02,
    ):
        """Initialize LinkCreator with camera parameters."""
        self.intrinsic = self._make_and_adjust_intrinsic(fx, fy, mx, my, image_dim)
        self.image_dim = image_dim
        self.voxel_size = voxel_size

    def _make_and_adjust_intrinsic(self, fx, fy, mx, my, image_dim):
        """Create and adjust camera intrinsic matrix."""
        intrinsic = np.eye(4)
        intrinsic[0][0] = fx
        intrinsic[1][1] = fy
        intrinsic[0][2] = mx
        intrinsic[1][2] = my
        return self._adjust_intrinsic(intrinsic, image_dim, (160, 120))

    def _adjust_intrinsic(self, intrinsic, intrinsic_image_dim, image_dim):
        """Adjust intrinsic matrix for image resizing."""
        if intrinsic_image_dim == image_dim:
            return intrinsic

        resize_width = int(
            math.floor(
                image_dim[1]
                * float(intrinsic_image_dim[0])
                / float(intrinsic_image_dim[1])
            )
        )

        intrinsic[0, 0] *= float(resize_width) / float(intrinsic_image_dim[0])
        intrinsic[1, 1] *= float(image_dim[1]) / float(intrinsic_image_dim[1])
        intrinsic[0, 2] *= float(image_dim[0] - 1) / float(intrinsic_image_dim[0] - 1)
        intrinsic[1, 2] *= float(image_dim[1] - 1) / float(intrinsic_image_dim[1] - 1)

        return intrinsic

    def computeUnprojection(self, camera_to_world, depth):
        """
        Compute 3D world coordinates for each pixel in the depth image.

        Args:
            camera_to_world: 4x4 camera-to-world transformation matrix
            depth: HxW depth image

        Returns:
            HxWx4 tensor of world coordinates and validity mask
        """
        H, W = depth.shape
        u = torch.arange(W, device=depth.device).view(1, -1).repeat(H, 1)
        v = torch.arange(H, device=depth.device).view(-1, 1).repeat(1, W)
        z = depth

        # Compute normalized camera coordinates
        x = (u - self.intrinsic[0, 2]) * z / self.intrinsic[0, 0]
        y = (v - self.intrinsic[1, 2]) * z / self.intrinsic[1, 1]

        # Create homogeneous coordinates
        camera_coords = torch.stack([x, y, z, torch.ones_like(z)], axis=-1)
        camera_coords_flat = camera_coords.reshape(-1, 4).T

        # Transform to world coordinates
        world_coords_flat = torch.matmul(camera_to_world.T, camera_coords_flat)
        valid_mask = camera_coords_flat[2] > 5e-2

        # Combine coordinates with validity mask
        world_coords_flat = torch.cat(
            [world_coords_flat[:3, :], valid_mask.unsqueeze(0)], dim=0
        ).T
        return world_coords_flat.reshape(H, W, 4)
