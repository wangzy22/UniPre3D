# Data Preparation 🗂️

This document provides detailed instructions for preparing the datasets used in our project. We use two main datasets: ShapeNet-Multiview and ScanNet v2.

## Table of Contents
- [ShapeNet-Multiview Dataset](#shapenet-multiview-dataset)
  - [Option 1: Download from Baidu Cloud](#option-1-download-from-baidu-cloud)
  - [Option 2: Download from HuggingFace](#option-2-download-from-huggingface)
  - [Dataset Structure](#shapenet-dataset-structure)
- [ScanNet v2](#scannet-v2)
  - [Download and Preparation](#download-and-preparation)
  - [Dataset Structure](#scannet-dataset-structure)
- [Acknowledgements](#acknowledgements)

## ShapeNet-Multiview Dataset 🖼️ (For Object-level Pretraining) <a id="shapenet-multiview-dataset"></a>

ShapeNet-Multiview dataset is a collection of multi-view renderings of 3D models in the ShapeNet dataset.

Our ShapeNet-Multiview dataset is based on the work of [ShapenetRender_more_variation](https://github.com/Xharlie/ShapenetRender_more_variation). We configured different parameters specifically for compatibility with our framework. It provides high-quality multi-view renderings of 3D models with the following features:

- 24 views per model
- Resolution: 128 x 128
- Modalities: RGB and point cloud
- Camera parameters and metadata included

### Dataset Download

**Option 1: Download from [Baidu Cloud](https://pan.baidu.com/s/1XIuxMYMhXeIhd9Bf8XZuoQ?pwd=ve54)**
```bash
Link: https://pan.baidu.com/s/1XIuxMYMhXeIhd9Bf8XZuoQ?pwd=ve54
Extraction Code: ve54
```

**Option 2: Download from [HuggingFace](https://huggingface.co/datasets/Yanran21/Shapenet_multiview)**
```bash
https://huggingface.co/datasets/Yanran21/Shapenet_multiview
```

### Dataset Structure

The dataset is split into multiple zip files, which need to be downloaded, merged, and extracted. The commands for merging and extracting are as follows:
```bash
zip -s 0 shapenet_dataset.zip --out shapenet_dataset_merged.zip
unzip shapenet_dataset_merged.zip
```

After downloading and extracting, your dataset should follow this structure:
```
shapenet_dataset_merged/
├── image/
│   ├── 02691156/   
│   │   ├── 10155655850468db78d106ce0a280f87/
│   │   │   ├── easy/
│   │   │   │   ├── 00.png
│   │   │   │   ├── 00.txt
│   │   │   │   ├── 01.png
│   │   │   │   ├── 01.txt
│   │   │   │   ├── ...
│   │   │   │   ├── rendering_metadata.txt
│   │   │   ├── pts/
│   │   │   │   ├── 02691156-10155655850468db78d106ce0a280f87.npy
│   │   └── ...
│   └── 02747177/
│       ├── ...
```

where `rendering_metadata.txt` contains the camera parameters and metadata for each view. `pts` contains the point cloud data. `image` contains the rendered images.

For using the dataset, you need to set the `data.dataset_root` in the `configs/dataset/shapenet.yaml` file.

## ScanNet v2 🏠 (For Scene-level Pretraining) <a id="scannet-v2"></a>

[ScanNet v2](http://www.scan-net.org/) is a large-scale indoor scene dataset with RGB-D scans and 3D reconstructions.

### Download and Preparation

We use the processed ScanNet dataset provided by [Pointcept](https://github.com/Pointcept/Pointcept?tab=readme-ov-file#scannet-v2).

### ScanNet Dataset Structure
After processing, your ScanNet dataset should look like:
```
scannet/
├── train/
│   ├── scene0000_00/
│   │   ├── color.npy
│   │   ├── coord.npy
│   │   ├── normal.npy
│   │   ├── instance.npy
│   │   ├── segment20.npy
│   │   └── segment200.npy
│   └── ...
├── val/
│   ├── scene0000_00/
│   │   ├── color.npy
│   │   └── ...
│   └── ...
├── test/
│   ├── scene0000_00/
│   │   ├── color.npy
│   │   └── ...
│   └── ...
```

For using the dataset, you need to set the `data.dataset_root` in the `configs/dataset/scannet.yaml` file.