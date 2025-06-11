import os
from PIL import Image
from typing import NamedTuple, List, Union, Optional, Any, Tuple
from utils.graphics_utils import focal2fov, fov2focal
import numpy as np
from pathlib import Path
import json
import torch
import math


class CameraInfo(NamedTuple):
    """Camera information container for both ShapeNet and ScanNet datasets"""

    uid: int  # Unique identifier
    R: np.array  # Rotation matrix (transposed for CUDA code)
    T: np.array  # Translation vector
    c2w: np.array  # Camera to world transform matrix
    w2c: np.array  # World to camera transform matrix
    FovY: np.array  # Vertical field of view in radians
    FovX: np.array  # Horizontal field of view in radians
    image: np.array  # Image data
    image_path: str  # Path to image file
    image_name: str  # Image name
    width: int  # Image width
    height: int  # Image height
    depth: np.array = None  # Depth map data
    intrinsic: np.array = None  # Camera intrinsic matrix


def readCamerasFromTxt(
    rgb_paths: List[str],
    pose_paths_or_matrix: Union[List[str], np.ndarray],
    idxs: List[int],
    fov: float,
    is_path: bool = True,
    moving_centers: Optional[np.ndarray] = None,
    depth_paths: Optional[List[str]] = None,
    dataset_type: str = "shapenet",
) -> List[CameraInfo]:
    """
    Read camera information from text files for both ShapeNet and ScanNet datasets

    Args:
        rgb_paths: List of paths to RGB images
        pose_paths_or_matrix: Camera pose information (paths or matrices)
        idxs: Indices of frames to process
        is_path: Whether pose_paths_or_matrix contains paths or matrices
        moving_centers: Centers for ScanNet camera adjustment
        depth_paths: List of paths to depth maps
        dataset_type: Type of dataset ("ShapeNet" or "ScanNet")
    """
    cam_infos = []

    # Set default FOV based on dataset type
    fovx = fov

    for idx in idxs:
        # Get camera pose matrix
        if is_path:
            cam_name = (
                pose_paths_or_matrix[0]
                if len(pose_paths_or_matrix) == 1
                else pose_paths_or_matrix[idx]
            )

            if cam_name.endswith("metadata.txt"):
                continue

            c2w = _read_camera_matrix(cam_name)

            # Apply ScanNet-specific transformations
            if dataset_type == "scannet" and moving_centers is not None:
                c2w[:3, 3] -= moving_centers
        else:
            c2w = pose_paths_or_matrix[idx]

        # Calculate world to camera transform
        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3, :3])
        T = w2c[:3, 3]

        # Handle image and depth data
        image_path = rgb_paths[idx]
        image_name = Path(cam_name).stem if is_path else image_path
        image = Image.open(image_path)

        depth = None
        if depth_paths is not None:
            depth = (
                Image.open(depth_paths[idx])
                if dataset_type == "scannet"
                else depth_paths[idx]
            )

        # Calculate field of view
        fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])

        cam_infos.append(
            CameraInfo(
                uid=idx,
                R=R,
                T=T,
                c2w=c2w,
                w2c=w2c,
                FovY=fovy,
                FovX=fovx,
                image=image,
                depth=depth,
                image_path=image_path,
                image_name=image_name,
                width=image.size[0],
                height=image.size[1],
            )
        )

    return cam_infos


def _read_camera_matrix(cam_name: str) -> np.ndarray:
    """Helper function to read camera matrix from file"""
    if cam_name.endswith(".json"):
        with open(cam_name, "r") as f:
            json_data = json.load(f)
            if "camera_angle_x" in json_data:
                fovx = json_data["camera_angle_x"]
            return np.array(json_data["frames"][0]["transform_matrix"])
    elif cam_name.endswith(".txt"):
        with open(cam_name, "r") as f:
            lines = f.readlines()
        return np.array([float(num) for line in lines for num in line.split()]).reshape(
            4, 4
        )


def adjust_intrinsic(
    intrinsic: np.ndarray,
    intrinsic_image_dim: Tuple[int, int],
    image_dim: Tuple[int, int],
) -> np.ndarray:
    """
    Adjust camera intrinsic matrix for image resizing

    Args:
        intrinsic: Original intrinsic matrix
        intrinsic_image_dim: Original image dimensions
        image_dim: Target image dimensions
    """
    if intrinsic_image_dim == image_dim:
        return intrinsic

    resize_width = int(
        math.floor(
            image_dim[1] * float(intrinsic_image_dim[0]) / float(intrinsic_image_dim[1])
        )
    )

    intrinsic = intrinsic.copy()
    intrinsic[0, 0] *= float(resize_width) / float(intrinsic_image_dim[0])
    intrinsic[1, 1] *= float(image_dim[1]) / float(intrinsic_image_dim[1])
    intrinsic[0, 2] *= float(image_dim[0] - 1) / float(intrinsic_image_dim[0] - 1)
    intrinsic[1, 2] *= float(image_dim[1] - 1) / float(intrinsic_image_dim[1] - 1)

    return intrinsic
