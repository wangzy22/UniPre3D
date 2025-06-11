from .shapenet import ShapeNetDataset
from .scannet import ScanNetDataset


def get_dataset(cfg, name, device=None):
    if cfg.data.category == "shapenet":
        return ShapeNetDataset(cfg, name)
    elif cfg.data.category == "scannet":
        return ScanNetDataset(cfg, name)
