import glob
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import random
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

from .dataset_readers import readCamerasFromTxt
from utils.general_utils import PILtoTorch, matrix_to_quaternion
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, getView2World
from openpoints.dataset.data_util import IO
from openpoints.models.layers.subsample import furthest_point_sample
from openpoints.transforms.point_transformer_gpu import (
    PointCloudRotationWithExtrinsic,
    PointCloudScaleAndJitterWithExtrinsic,
    PointCloudCenterAndNormalizeWithExtrinsic,
    PointCloudJitter,
)

# Dataset root path configuration
SHAPENET_DATASET_ROOT = "/data1/datasets/center_renders/image"
FILE_TITLE = "easy"

assert (
    SHAPENET_DATASET_ROOT is not None
), "Please update the location of the SRN Shapenet Dataset"


class ShapeNetDataset(Dataset):
    """
    ShapeNet dataset loader class
    Handles loading and processing of ShapeNet data for training and evaluation
    """

    # Class constants
    TRAIN_SPLIT_RATIO = 0.75
    VAL_SPLIT_RATIO = 0.2
    CAMERA_DISTANCE = 1.75
    GRAVITY_DIM = 2

    def __init__(self, cfg: Any, dataset_name: str = "train"):
        """
        Initialize ShapeNet dataset

        Args:
            cfg: Configuration object containing dataset parameters
            dataset_name: Name of the dataset split ("train", "val", or "test")
        """
        super().__init__()
        self.cfg = cfg
        self.dataset_name = dataset_name

        # Initialize dataset paths and metadata
        self.base_path = SHAPENET_DATASET_ROOT
        self.metadata = self._initialize_metadata()

        # Split dataset based on dataset_name
        self._split_dataset()

        # Initialize pose matrices and projection matrix
        self.continuous_pose_matrix = self._generate_continuous_pose(num=200)
        self.projection_matrix = self._initialize_projection_matrix()

        # Set number of images per object and testing image indices
        self.imgs_per_obj = self.cfg.opt.imgs_per_obj
        self.test_input_idxs = self._get_test_input_indices()

        # Initialize image recording flags and transforms
        self.record_img = cfg.opt.record_img
        self.now_rgbs = []
        self.transforms = [PointCloudRotationWithExtrinsic()]

    def _initialize_metadata(self) -> List[str]:
        """
        Initialize dataset metadata by scanning directory structure

        Returns:
            List of metadata paths
        """
        metadata = []
        subfolders_level1 = [f.path for f in os.scandir(self.base_path) if f.is_dir()]
        self.classes = [
            os.path.basename(class_path) for class_path in subfolders_level1
        ]

        # Two-level directory traversal for standard configuration
        for subfolder_level1 in subfolders_level1:
            subfolders_level2 = [
                f.path for f in os.scandir(subfolder_level1) if f.is_dir()
            ]
            metadata.extend(subfolders_level2)

        return sorted(metadata)

    def _split_dataset(self) -> None:
        """Split dataset into train/val/test sets"""
        # Shuffle metadata with fixed seed for reproducibility
        random.seed(self.cfg.general.random_seed + 1)
        random.shuffle(self.metadata)

        # Calculate split lengths
        train_length = int(len(self.metadata) * self.TRAIN_SPLIT_RATIO)
        val_length = int(len(self.metadata) * self.VAL_SPLIT_RATIO)

        # Split dataset based on mode
        if self.dataset_name == "train":
            self.metadata = self.metadata[:train_length]
        elif self.dataset_name == "val":
            self.metadata = self.metadata[train_length : (val_length + train_length)]
        else:  # test
            self.metadata = self.metadata[(val_length + train_length) :]

        print(f"Dataset {self.dataset_name} size: {len(self.metadata)}")

    def _initialize_projection_matrix(self) -> torch.Tensor:
        """Initialize projection matrix based on configuration"""
        return getProjectionMatrix(
            znear=self.cfg.data.znear,
            zfar=self.cfg.data.zfar,
            fovX=self.cfg.data.fov * 2 * np.pi / 360,
            fovY=self.cfg.data.fov * 2 * np.pi / 360,
        ).transpose(0, 1)

    def _get_test_input_indices(self) -> List[int]:
        """Get input indices for testing based on configuration"""
        if self.cfg.data.input_images == 1:
            return [0]
        else:
            raise NotImplementedError("Only 1 input images supported")

    def load_example_id(
        self,
        example_id: str,
        metadata_path: str,
        trans: np.ndarray = np.array([0.0, 0.0, 0.0]),
        scale: float = 1.0,
    ) -> None:
        """
        Load example data by ID

        Args:
            example_id: Unique identifier for the example
            metadata_path: Path to metadata file
            trans: Translation vector
            scale: Scale factor
        """
        # Get file paths
        rgb_paths = sorted(glob.glob(os.path.join(metadata_path, FILE_TITLE, "*.png")))
        pose_paths = sorted(
            glob.glob(os.path.join(metadata_path, FILE_TITLE, "[0-9]*.txt"))
        )
        metadata_paths = sorted(
            glob.glob(os.path.join(metadata_path, FILE_TITLE, "rendering_metadata.txt"))
        )
        pts_paths = sorted(glob.glob(os.path.join(metadata_path, "pts", "*")))

        if len(rgb_paths) == 0:
            return None

        assert len(rgb_paths) == len(pose_paths), "Mismatch between RGB and pose files"

        # Initialize storage if needed
        self._initialize_storage_if_needed()

        # Load camera information
        if example_id not in self.all_rgbs.keys(): 
            self._load_camera_info(
                example_id, rgb_paths, pose_paths, metadata_paths, trans, scale
            )

            # Load and process point cloud data
            self._load_point_cloud_data(example_id, pts_paths)

    def _initialize_storage_if_needed(self) -> None:
        """Initialize storage dictionaries if not already initialized"""
        if not hasattr(self, "all_rgbs"):
            self.all_rgbs = {}
            self.all_world_view_transforms = {}
            self.all_view_to_world_transforms = {}
            self.all_full_proj_transforms = {}
            self.all_camera_centers = {}
            self.all_w2c = {}
            self.all_pts = {}
            if self.dataset_name == "test":
                self.all_rolls = {}
                self.all_pitches = {}

    def _load_camera_info(
        self,
        example_id: str,
        rgb_paths: List[str],
        pose_paths: List[str],
        metadata_paths: List[str],
        trans: np.ndarray,
        scale: float,
    ) -> None:
        """
        Load camera information and images

        Args:
            example_id: Unique identifier for the example
            rgb_paths: List of paths to RGB images
            pose_paths: List of paths to pose files
            metadata_paths: List of paths to metadata files
            trans: Translation vector
            scale: Scale factor
        """
        if not self.record_img:
            self.now_rgbs = []
        else:
            self.all_rgbs[example_id] = []

        # Initialize transform lists
        self.all_world_view_transforms[example_id] = []
        self.all_full_proj_transforms[example_id] = []
        self.all_camera_centers[example_id] = []
        self.all_view_to_world_transforms[example_id] = []
        self.all_w2c[example_id] = []

        # Handle test dataset specific data
        if self.dataset_name == "test":
            cam_infos = self._load_test_dataset_info(example_id, rgb_paths, metadata_paths)
        else:
            cam_infos = readCamerasFromTxt(
                rgb_paths, pose_paths, list(range(len(rgb_paths))),fov = self.cfg.data.fov,dataset_type="shapenet"
            )

        # Process camera information
        self._process_camera_info(example_id, cam_infos, trans, scale)

    def _load_test_dataset_info(
        self, example_id: str, rgb_paths: List[str], metadata_paths: List[str]
    ) -> None:
        """
        Load test dataset specific information

        Args:
            example_id: Unique identifier for the example
            rgb_paths: List of paths to RGB images
            metadata_paths: List of paths to metadata files
        """
        self.all_rolls[example_id] = []
        self.all_pitches[example_id] = []

        # Calculate rate for pose matrix replication
        len_matrix = len(self.continuous_pose_matrix)
        len_rgb_path = len(rgb_paths)
        rate = len_matrix // len_rgb_path + 1

        # Read camera information
        cam_infos = readCamerasFromTxt(
            rgb_paths * rate,
            self.continuous_pose_matrix,
            list(range(len(self.continuous_pose_matrix))),
            fov = self.cfg.data.fov,
            is_path=False,
            dataset_type="shapenet"
        )

        # Load metadata
        self._load_metadata_angles(example_id, metadata_paths[0])
        return cam_infos

    def _load_metadata_angles(self, example_id: str, metadata_path: str) -> None:
        """
        Load roll and pitch angles from metadata file

        Args:
            example_id: Unique identifier for the example
            metadata_path: Path to metadata file
        """
        with open(metadata_path, "r") as file:
            for line in file:
                data = line.strip().split()
                if line[0] == "[":
                    data = eval(line)[0]
                self.all_rolls[example_id].append(torch.tensor(float(data[1])))
                self.all_pitches[example_id].append(torch.tensor(float(data[0])))

    def _process_camera_info(
        self, example_id: str, cam_infos: List[Any], trans: np.ndarray, scale: float
    ) -> None:
        """
        Process camera information and store transforms

        Args:
            example_id: Unique identifier for the example
            cam_infos: List of camera information objects
            trans: Translation vector
            scale: Scale factor
        """
        for cam_info in cam_infos:
            # Get rotation and translation
            R = cam_info.R
            T = cam_info.T

            # Calculate transforms
            world_view_transform = torch.tensor(
                getWorld2View2(R, T, trans, scale)
            ).transpose(0, 1)
            view_world_transform = torch.tensor(
                getView2World(R, T, trans, scale)
            ).transpose(0, 1)

            # Calculate projection transforms
            full_proj_transform = (
                world_view_transform.unsqueeze(0).bmm(
                    self.projection_matrix.unsqueeze(0)
                )
            ).squeeze(0)
            camera_center = world_view_transform.inverse()[3, :3]

            # Store image data
            if self.record_img:
                self.all_rgbs[example_id].append(self._process_image(cam_info.image))
            else:
                self.now_rgbs.append(self._process_image(cam_info.image))

            # Store transforms
            self.all_w2c[example_id].append(torch.from_numpy(cam_info.w2c).float())
            self.all_world_view_transforms[example_id].append(world_view_transform)
            self.all_view_to_world_transforms[example_id].append(view_world_transform)
            self.all_full_proj_transforms[example_id].append(full_proj_transform)
            self.all_camera_centers[example_id].append(camera_center)

    def _process_image(self, image: Any) -> torch.Tensor:
        """
        Process image data

        Args:
            image: Input image

        Returns:
            Processed image tensor
        """
        return PILtoTorch(
            image,
            (self.cfg.data.training_resolution, self.cfg.data.training_resolution),
        ).clamp(0.0, 1.0)[:3, :, :]

    def _load_point_cloud_data(self, example_id: str, pts_paths: List[str]) -> None:
        """
        Load and process point cloud data

        Args:
            example_id: Unique identifier for the example
            pts_paths: List of paths to point cloud files
        """
        try:
            data = IO.get(pts_paths[0])
        except Exception as e:
            print(f"Error loading data from {pts_paths}: {e}")
            return

        # Convert to tensor and center the point cloud
        tensor_data = (
            torch.tensor(self.center_point_cloud(data), dtype=torch.float32)
            .unsqueeze(0)
            .to(f"cuda:{self.cfg.general.device}")
        )

        # Sample points using furthest point sampling
        idx = furthest_point_sample(
            tensor_data[:3], 1024
        ).long()

        # Process point cloud based on input channel configuration
        new_data = self._process_point_cloud(tensor_data, idx)

        # Store processed data
        self._store_processed_data(example_id, new_data)

    def _process_point_cloud(
        self, tensor_data: torch.Tensor, idx: torch.Tensor
    ) -> torch.Tensor:
        """
        Process point cloud data based on input channel configuration

        Args:
            tensor_data: Input point cloud tensor
            idx: Sampling indices

        Returns:
            Processed point cloud tensor
        """
        new_data = torch.gather(
            tensor_data, 1, idx.unsqueeze(-1).expand(-1, -1, 3)
        ).squeeze(0)

        # Transform coordinates for standard configuration
        new_data = self._transform_point_data(new_data)

        return new_data

    def _transform_point_data(self, data: torch.Tensor) -> torch.Tensor:
        """
        Transform standard (3/4 channel) point cloud data

        Args:
            data: Input point cloud tensor

        Returns:
            Transformed point cloud tensor
        """
        data = data[:, (2, 0, 1)]
        data[:, 0] *= -1
        data[:, 2] *= -1

        return data

    def _add_gravity_dimension(self, data: torch.Tensor) -> torch.Tensor:
        """
        Add gravity dimension to point cloud data

        Args:
            data: Input point cloud tensor

        Returns:
            Point cloud tensor with gravity dimension
        """
        if self.cfg.model.in_channels == 3:
            return data
        else:
            gravity_dim = data[:, self.GRAVITY_DIM : self.GRAVITY_DIM + 1]
            gravity_feature = gravity_dim - gravity_dim.min()
            return torch.cat((data, gravity_feature), dim=1)

    def _store_processed_data(self, example_id: str, data: torch.Tensor) -> None:
        """
        Store processed point cloud and camera data

        Args:
            example_id: Unique identifier for the example
            data: Processed point cloud data
        """
        self.all_pts[example_id] = torch.stack(tuple(data))
        self.all_w2c[example_id] = torch.stack(self.all_w2c[example_id])
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

        if self.record_img:
            self.all_rgbs[example_id] = torch.stack(self.all_rgbs[example_id])
        else:
            self.now_rgbs = torch.stack(self.now_rgbs)

        if self.dataset_name == "test":
            self.all_rolls[example_id] = torch.stack(self.all_rolls[example_id])
            self.all_pitches[example_id] = torch.stack(self.all_pitches[example_id])

    def __len__(self) -> int:
        """Return the length of the dataset"""
        return len(self.metadata)

    def get_example_id(self, index: int) -> str:
        """
        Get example ID from index

        Args:
            index: Dataset index

        Returns:
            Example identifier string
        """
        metadata_path = self.metadata[index]
        return os.path.basename(os.path.dirname(metadata_path))

    def center_point_cloud(self, pc: np.ndarray) -> np.ndarray:
        """
        Center point cloud by subtracting mean

        Args:
            pc: Input point cloud (Nx3 or NxC)

        Returns:
            Centered point cloud
        """
        centroid = np.mean(pc[:, 0:3], axis=0)
        pc[:, 0:3] = pc[:, 0:3] - centroid
        return pc

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """
        Get dataset item by index

        Args:
            index: Dataset index

        Returns:
            Dictionary containing processed data
        """
        # Get metadata and example ID
        metadata_path = self.metadata[index]
        parent_dir, category_instance_dir = os.path.split(metadata_path)
        category, instance = os.path.split(parent_dir)
        example_id = f"{instance}-{category_instance_dir}"

        # Load example data
        self.load_example_id(example_id, metadata_path)
        if self.record_img and example_id not in self.all_rgbs:
            print(f"Warning: {example_id} RGB info not found")
            return self.__getitem__(random.randint(0, len(self.metadata) - 1))

        # Get frame indices based on dataset type
        frame_indices = self._get_frame_indices(example_id)

        # Prepare base output dictionary
        output_dict = self._prepare_base_output(example_id, frame_indices)

        pts_to_be_transformed = {
            "pos": self.all_pts[example_id],
            "extrinsic": self.all_w2c[example_id][frame_indices].clone().cuda(),
        }
        
        # Apply augmentation if needed
        if self.cfg.model.aug and self.dataset_name == "train":
            output_dict = self._apply_augmentation(
                pts_to_be_transformed, frame_indices, output_dict
            )
        output_dict["point_cloud"] = pts_to_be_transformed
        output_dict["point_cloud"]["pos"] = self._add_gravity_dimension(
            output_dict["point_cloud"]["pos"]
        )

        return output_dict

    def _apply_augmentation(self,pts_to_be_transformed, frame_indices, output_dict):
        """
        Apply data augmentation to point cloud and camera poses.

        Args:
            pts_to_be_transformed (dict)
            frame_indices (torch.Tensor): Indices of selected frames
            output_dict (dict): Dictionary containing data to be augmented

        Returns:
            dict: Augmented data dictionary containing transformed point cloud and camera poses
        """
        # Prepare point cloud data for transformation

        # Apply each transformation in sequence
        for transform in self.transforms:
            pts_to_be_transformed = transform(pts_to_be_transformed)

        # Recalculate camera poses after point cloud transformation
        now_world_view_transform = []
        now_view_world_transform = []
        now_full_proj_transform = []
        now_camera_centers = []

        # Process each camera pose
        for w2c in pts_to_be_transformed["extrinsic"]:
            current_w2c = np.array(w2c.detach().cpu())

            # Extract rotation and translation from world-to-camera transform
            R = np.transpose(
                current_w2c[:3, :3]
            )  # R is stored transposed due to 'glm' in CUDA code
            T = current_w2c[:3, 3]

            # Calculate new transformation matrices
            world_view_transform = torch.tensor(getWorld2View2(R, T)).transpose(0, 1)
            view_world_transform = torch.tensor(getView2World(R, T)).transpose(0, 1)

            # Calculate projection and camera center
            full_proj_transform = (
                world_view_transform.unsqueeze(0).bmm(
                    self.projection_matrix.unsqueeze(0)
                )
            ).squeeze(0)
            camera_center = world_view_transform.inverse()[3, :3]

            # Store transformed matrices
            now_world_view_transform.append(world_view_transform)
            now_view_world_transform.append(view_world_transform)
            now_full_proj_transform.append(full_proj_transform)
            now_camera_centers.append(camera_center)

        # Update output dictionary with transformed data
        output_dict.update(
            {
                "point_cloud": pts_to_be_transformed,
                "world_view_transforms": torch.stack(now_world_view_transform),
                "view_to_world_transforms": torch.stack(now_view_world_transform),
                "full_proj_transforms": torch.stack(now_full_proj_transform),
                "camera_centers": torch.stack(now_camera_centers),
            }
        )

        return output_dict

    def _get_frame_indices(self, example_id: str) -> torch.Tensor:
        """Get frame indices based on dataset type"""
        if self.dataset_name == "train":
            total_frames = (
                len(self.all_rgbs[example_id])
                if self.record_img
                else len(self.now_rgbs)
            )
            frame_indices = torch.randperm(total_frames)[: self.imgs_per_obj]
            return torch.cat(
                [frame_indices[: self.cfg.data.input_images], frame_indices]
            )

        elif self.dataset_name in ["val"]:
            non_input_indices = [i for i in range(24) if i not in self.test_input_idxs]
            return torch.cat(
                [torch.tensor(self.test_input_idxs), torch.tensor(non_input_indices)]
            )

        else:  # test
            non_input_indices = [
                i
                for i in range(len(self.continuous_pose_matrix))
                if i not in self.test_input_idxs
            ]
            return torch.cat(
                [torch.tensor(self.test_input_idxs), torch.tensor(non_input_indices)]
            )

    def _prepare_base_output(
        self, example_id: str, frame_indices: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Prepare base output dictionary"""
        return {
            "gt_images": (
                self.all_rgbs[example_id][frame_indices].clone()
                if self.record_img
                else self.now_rgbs[frame_indices].clone()
            ),
            "pitches": (
                self.all_pitches[example_id]
                if self.dataset_name == "test"
                else torch.Tensor(0)
            ),
            "rolls": (
                self.all_rolls[example_id]
                if self.dataset_name == "test"
                else torch.Tensor(0)
            ),
            "world_view_transforms": self.all_world_view_transforms[example_id][
                frame_indices
            ],
            "view_to_world_transforms": self.all_view_to_world_transforms[example_id][
                frame_indices
            ],
            "full_proj_transforms": self.all_full_proj_transforms[example_id][
                frame_indices
            ],
            "camera_centers": self.all_camera_centers[example_id][frame_indices],
            "point_cloud": self.all_pts[example_id],
        }

    def get_source_cw2wT(
        self, source_cameras_view_to_world: torch.Tensor
    ) -> torch.Tensor:
        """Convert camera transforms to quaternions"""
        return torch.stack(
            [
                matrix_to_quaternion(cam[:3, :3].transpose(0, 1))
                for cam in source_cameras_view_to_world
            ]
        )

    def _calculate_transformation_matrix(
        self, roll: float, pitch: float, distance_ratio: float
    ) -> np.ndarray:
        """
        Calculate camera transformation matrix

        Args:
            roll: Camera roll angle in degrees
            pitch: Camera pitch angle in degrees
            distance_ratio: Distance ratio for camera position

        Returns:
            4x4 transformation matrix
        """
        DISTANCE = 1.75
        azimuth_rad = math.radians(-roll)
        elevation_rad = math.radians(pitch - 90)
        in_plane_rotation_rad = 0

        # Calculate rotation matrix
        rotation_matrix = np.array(
            [
                [
                    math.cos(in_plane_rotation_rad) * math.cos(azimuth_rad)
                    - math.sin(in_plane_rotation_rad)
                    * math.cos(elevation_rad)
                    * math.sin(azimuth_rad),
                    math.sin(in_plane_rotation_rad) * math.cos(azimuth_rad)
                    + math.cos(in_plane_rotation_rad)
                    * math.cos(elevation_rad)
                    * math.sin(azimuth_rad),
                    math.sin(elevation_rad) * math.sin(azimuth_rad),
                ],
                [
                    -math.cos(in_plane_rotation_rad) * math.sin(azimuth_rad)
                    - math.sin(in_plane_rotation_rad)
                    * math.cos(elevation_rad)
                    * math.cos(azimuth_rad),
                    -math.sin(in_plane_rotation_rad) * math.sin(azimuth_rad)
                    + math.cos(in_plane_rotation_rad)
                    * math.cos(elevation_rad)
                    * math.cos(azimuth_rad),
                    math.sin(elevation_rad) * math.cos(azimuth_rad),
                ],
                [
                    math.sin(in_plane_rotation_rad) * math.sin(elevation_rad),
                    -math.cos(in_plane_rotation_rad) * math.sin(elevation_rad),
                    math.cos(elevation_rad),
                ],
            ]
        )

        # Calculate translation vector
        translation_vector = np.array(
            [
                -self.CAMERA_DISTANCE
                * math.sin(elevation_rad)
                * math.sin(azimuth_rad)
                * distance_ratio,
                -self.CAMERA_DISTANCE
                * math.sin(elevation_rad)
                * math.cos(azimuth_rad)
                * distance_ratio,
                -self.CAMERA_DISTANCE * math.cos(elevation_rad) * distance_ratio,
            ]
        )

        # Construct transformation matrix
        transform = np.eye(4)
        transform[:3, :3] = rotation_matrix
        transform[:3, 3] = translation_vector
        return transform

    def _generate_continuous_pose(self, num: int = 160) -> List[np.ndarray]:
        """
        Generate continuous camera poses

        Args:
            num: Number of poses to generate

        Returns:
            List of transformation matrices
        """
        rolls = np.linspace(-180, 180, num)
        pitches_1 = np.linspace(0, 20, num)
        pitches_2 = np.linspace(20, 90, num)

        return [
            self._calculate_transformation_matrix(roll, pitch, 1.0)
            for roll, pitch in list(zip(rolls, pitches_1)) + list(zip(rolls, pitches_2))
        ]
