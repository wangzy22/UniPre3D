#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import sys
from datetime import datetime
import numpy as np
import random
from PIL import Image
from typing import Dict, Any

def inverse_sigmoid(x):
    return torch.log(x / (1 - x))


def PILtoTorch(pil_image, resolution, is_depth=False, resize=True):
    if is_depth:
        resized_image_PIL = pil_image.resize(resolution, resample=Image.NEAREST)
    else:
        resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL).astype(np.float32))
    if resize:
        if not is_depth:
            resized_image /= 255.0
        else:
            resized_image /= 1000.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)


def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper


def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty


def strip_symmetric(sym):
    return strip_lowerdiag(sym)


def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    """
    From Pytorch3d
    Convert a unit quaternion to a standard form: one in which the real
    part is non negative.

    Args:
        quaternions: Quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Standardized quaternions as tensor of shape (..., 4).
    """
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)


def quaternion_raw_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    From Pytorch3d
    Multiply two quaternions.
    Usual torch rules for broadcasting apply.

    Args:
        a: Quaternions as tensor of shape (..., 4), real part first.
        b: Quaternions as tensor of shape (..., 4), real part first.

    Returns:
        The product of a and b, a tensor of quaternions shape (..., 4).
    """
    aw, ax, ay, az = torch.unbind(a, -1)
    bw, bx, by, bz = torch.unbind(b, -1)
    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    return torch.stack((ow, ox, oy, oz), -1)


# Matrix to quaternion does not come under NVIDIA Copyright
# Written by Stan Szymanowicz 2023
def matrix_to_quaternion(M: torch.Tensor) -> torch.Tensor:
    """
    Matrix-to-quaternion conversion method. Equation taken from
    https://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/index.htm
    Args:
        M: rotation matrices, (3 x 3)
    Returns:
        q: quaternion of shape (4)
    """
    tr = 1 + M[0, 0] + M[1, 1] + M[2, 2]

    if tr > 0:
        r = torch.sqrt(tr) / 2.0
        x = (M[2, 1] - M[1, 2]) / (4 * r)
        y = (M[0, 2] - M[2, 0]) / (4 * r)
        z = (M[1, 0] - M[0, 1]) / (4 * r)
    elif (M[0, 0] > M[1, 1]) and (M[0, 0] > M[2, 2]):
        S = torch.sqrt(1.0 + M[0, 0] - M[1, 1] - M[2, 2]) * 2  # S=4*qx
        r = (M[2, 1] - M[1, 2]) / S
        x = 0.25 * S
        y = (M[0, 1] + M[1, 0]) / S
        z = (M[0, 2] + M[2, 0]) / S
    elif M[1, 1] > M[2, 2]:
        S = torch.sqrt(1.0 + M[1, 1] - M[0, 0] - M[2, 2]) * 2  # S=4*qy
        r = (M[0, 2] - M[2, 0]) / S
        x = (M[0, 1] + M[1, 0]) / S
        y = 0.25 * S
        z = (M[1, 2] + M[2, 1]) / S
    else:
        S = torch.sqrt(1.0 + M[2, 2] - M[0, 0] - M[1, 1]) * 2  # S=4*qz
        r = (M[1, 0] - M[0, 1]) / S
        x = (M[0, 2] + M[2, 0]) / S
        y = (M[1, 2] + M[2, 1]) / S
        z = 0.25 * S

    return torch.stack([r, x, y, z], dim=-1)


def build_rotation(r):
    norm = torch.sqrt(
        r[:, 0] * r[:, 0] + r[:, 1] * r[:, 1] + r[:, 2] * r[:, 2] + r[:, 3] * r[:, 3]
    )

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device="cuda")

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - r * z)
    R[:, 0, 2] = 2 * (x * z + r * y)
    R[:, 1, 0] = 2 * (x * y + r * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - r * x)
    R[:, 2, 0] = 2 * (x * z - r * y)
    R[:, 2, 1] = 2 * (y * z + r * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:, 0, 0] = s[:, 0]
    L[:, 1, 1] = s[:, 1]
    L[:, 2, 2] = s[:, 2]

    L = R @ L
    return L


def safe_state(cfg, silent=False):
    old_f = sys.stdout

    class F:
        def __init__(self, silent):
            self.silent = silent

        def write(self, x):
            if not self.silent:
                if x.endswith("\n"):
                    old_f.write(
                        x.replace(
                            "\n",
                            " [{}]\n".format(
                                str(datetime.now().strftime("%d/%m %H:%M:%S"))
                            ),
                        )
                    )
                else:
                    old_f.write(x)

        def flush(self):
            old_f.flush()

    sys.stdout = F(silent)

    random.seed(cfg.general.random_seed)
    np.random.seed(cfg.general.random_seed)
    torch.manual_seed(cfg.general.random_seed)
    torch.cuda.manual_seed_all(cfg.general.random_seed)

    if hasattr(cfg.general.device, '__len__') and not isinstance(cfg.general.device, str):
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
    else:
        device_str = "cuda:{}".format(cfg.general.device) if cfg.general.cuda else "cpu"
        device = torch.device(device_str)
    if cfg.general.cuda:
        torch.cuda.set_device(device)

    return device


def prepare_model_inputs(
        data: Dict[str, torch.Tensor], cfg: Dict[str, Any], bs_per_gpu: int, device: torch.device
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare input data for model forward pass.

        Args:
            data: Dictionary containing all input data including images, point clouds,
                 and transformation matrices

        Returns:
            Dictionary containing processed model inputs with appropriate device placement
            and batch organization
        """
        if cfg.data.category == "scannet":
            reshape_scannet_data(data, bs_per_gpu)
            
        
        point_cloud = data["point_cloud"] if "point_cloud" in data else data
        view_transforms = data["view_to_world_transforms"][
            :, : cfg.data.input_images, ...
        ]

        model_inputs = {
            "point_cloud": point_cloud,
            "source_cameras_view_to_world": view_transforms,
            "image": None,
            "unprojected_coords": None,
        }

        if cfg.opt.use_fusion:
            model_inputs.update(
                {
                    "image": data["gt_images"][:, : cfg.data.input_images, ...],
                    "unprojected_coords": (
                        data["unprojected_coords"][:, : cfg.data.input_images, ...]
                        if cfg.opt.level == "scene"
                        else None
                    ),
                }
            )

        return to_device(model_inputs, device)


def to_device(data, device):
    """
    Recursively moves data to specified device.

    Args:
        data: Input data, can be:
            - torch.Tensor
            - Dictionary containing tensors/nested dictionaries
            - List/Tuple containing tensors/nested dictionaries
            - None
        device: Target device (e.g., 'cuda', 'cpu', torch.device object)

    Returns:
        Same data structure with all tensors moved to specified device
    """
    if data is None:
        return None

    # Handle torch.Tensor
    if torch.is_tensor(data):
        return data.to(device)

    # Handle dictionary
    if isinstance(data, dict):
        return {key: to_device(value, device) for key, value in data.items()}

    # Handle list/tuple
    if isinstance(data, (list, tuple)):
        return type(data)(to_device(x, device) for x in data)

    # Return unchanged if not tensor/dict/list/tuple
    return data


def reshape_scannet_data(data: dict, bs_per_gpu: int):
    """
    Reshape the data dictionary to have a batch size of bs_per_gpu.
    Args:
        data (dict): The input data dictionary containing tensors.
        bs_per_gpu (int): The desired batch size per GPU.
    """
    
    DATASET_KEYS = [
        "coord",
        "feat",
        "segment",
        "offset",
        "grid_coord",
        "links",
        "inverse",
        "condition",
    ]
    for key in data:
        if key not in DATASET_KEYS and torch.is_tensor(data[key]):
            original_shape = data[key].shape
            if original_shape[0] // bs_per_gpu > 0:
                new_shape = (
                    bs_per_gpu,
                    original_shape[0] // bs_per_gpu,
                ) + original_shape[1:]
                data[key] = data[key].reshape(new_shape)
            else:
                data[key] = data[key].unsqueeze(0)
