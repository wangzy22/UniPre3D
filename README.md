# [CVPR 2025] UniPre3D: Unified Pre-training of 3D Point Cloud Models with Cross-Modal Gaussian Splatting


Created by [Ziyi Wang*](https://wangzy22.github.io/), [Yanran Zhang*](https://github.com/Zhangyr2022), [Jie Zhou](https://scholar.google.com/citations?user=6a79aPwAAAAJ&hl=en&authuser=1), [Jiwen Lu](https://scholar.google.com/citations?user=TN8uDQoAAAAJ&hl=zh-CN) (* indicates equal contribution)

This repository is an official implementation of **UniPre3D (CVPR 2025)**.

**[Paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_UniPre3D_Unified_Pre-training_of_3D_Point_Cloud_Models_with_Cross-Modal_CVPR_2025_paper.pdf)** | **[arXiv](https://arxiv.org/pdf/2506.09952v1)** | **[Project Page](https://ivg-yanranzhang.github.io/UniPre3D/)**

<div align=center>
<img src='assets\teaser.png' width=350 height=300>
</div>

UniPre3D is the first unified pre-training method for 3D point clouds that effectively handles both object- and scene-level data through cross-modal Gaussian splatting.

<div align=center>
<img src='assets\pipeline.png' width=450 height=260>
</div>

Our proposed pre-training task involves predicting Gaussian parameters from the input point cloud. The 3D backbone network is expected to extract representative features, and 3D Gaussian splatting is implemented to render images for direct supervision. To incorporate additional texture information and adjust task complexity, we introduce a pre-trained image model and propose a scale-adaptive fusion block to accommodate varying data scales.

## News 🔥

- [2025-07-02] Our scene-level pretraining code is released.
- [2025-06-12] Our arXiv paper is released.
- [2025-06-11] Our object-level pretraining code is released.
- [2025-02-27] Our paper is accepted by CVPR 2025.

## TODO (In Progress) ⭐

- [x] Release datasets
- [x] Release object-level pretraining code.
- [x] Release object-level logs and checkpoints.
- [x] Add more details about diverse downstream tasks.
- [x] Release scene-level pretraining code.
- [x] Release scene-level logs and checkpoints.

## Visualization Results 📷

Below is visualization of UniPre3D pre-training outputs. The first row presents the input point clouds, followed by the reference view images in the second row. The third row displays the rendered images, which are supervised by the ground truth images shown in the fourth row. In the rightmost column, we illustrate a schematic diagram of the view selection principle for both object- and scene-level samples.

<div align=center>
<img src='assets\visualization.png' width=850 height=280>
</div>

# Getting Started 🚀
## Table of Contents 📖

- [\[CVPR 2025\] UniPre3D: Unified Pre-training of 3D Point Cloud Models with Cross-Modal Gaussian Splatting](#cvpr-2025-unipre3d-unified-pre-training-of-3d-point-cloud-models-with-cross-modal-gaussian-splatting)
  - [News 🔥](#news-)
  - [TODO (In Progress) ⭐](#todo-in-progress-)
  - [Visualization Results 📷](#visualization-results-)
- [Getting Started 🚀](#getting-started-)
  - [Table of Contents 📖](#table-of-contents-)
  - [Environment Setup 🔧 ](#environment-setup--)
    - [Recommended Environment](#recommended-environment)
    - [Hardware Requirements](#hardware-requirements)
    - [Data Preparation](#data-preparation)
  - [Object-level Pre-training 🪑 ](#object-level-pre-training--)
    - [Usage](#usage)
  - [Finetune on Object-level Downstream Tasks 🎯 ](#finetune-on-object-level-downstream-tasks--)
    - [Model Zoo (Pretrained Checkpoints)](#model-zoo-pretrained-checkpoints)
  - [Scene-level Pretraining 🏠 ](#scene-level-pretraining--)
    - [Usage](#usage-1)
  - [Finetune on Scene-level Downstream Tasks 🎯 ](#finetune-on-scene-level-downstream-tasks--)
    - [Model Zoo (Pretrained Checkpoints)](#model-zoo-pretrained-checkpoints-1)
  - [Acknowledgements 🙏 ](#acknowledgements--)
  - [Citation 📚 ](#citation--)

## Environment Setup 🔧 <a id="environment-setup"></a>

### Recommended Environment

- Python 3.11
- PyTorch 2.2
- CUDA 12.0 or higher
- Linux or Windows operating system

Please follow **[docs/INSTALLATION.md](docs/INSTALLATION.md)** for detailed installation instructions.

### Hardware Requirements

- CUDA-capable GPU with compute capability 6.0 or higher
- Minimum 8GB GPU memory (16GB+ recommended for large-scale experiments)
- 16GB+ RAM

### Data Preparation

Please follow **[docs/DATA_PREPARATION.md](docs/DATA_PREPARATION.md)** for detailed data preparation instructions.

## Object-level Pre-training 🪑 <a id="object-pretraining"></a>

Object-level pre-training is a technique where we train a 3D model on a large collection of individual 3D objects before fine-tuning it for specific downstream tasks. This approach helps the model learn fundamental geometric patterns and structural representations that can be transferred to various 3D understanding tasks.

**Key Characteristics:**
- Focuses on learning from individual objects (e.g., chairs, airplanes, cars)
- Captures fine-grained local geometric structures
- Enables knowledge transfer to tasks like object classification and part segmentation

<div align=center>
<img src='assets\obj-vis.png' width="75%">
</div>

### Usage

PointMLP pretraining:
```bash
CUDA_VISIBLE_DEVICES=<GPUs> python train_network.py --config-name pointmlp_pretraining
```

Standard Transformer pretraining:
```bash
CUDA_VISIBLE_DEVICES=<GPUs> python train_network.py --config-name transformer_pretraining
```

Mamba3D pretraining:
```bash
CUDA_VISIBLE_DEVICES=<GPUs> python train_network.py --config-name mamba3d_pretraining
```

Point Cloud Mamba pretraining:
```bash
CUDA_VISIBLE_DEVICES=<GPUs> python train_network.py --config-name pcm_pretraining
```

> We cache dataset images in memory to accelerate data loading.  If you encounter memory constraints: Disable this feature by setting `opt.record_img` to `false` in `configs/settings.yaml`

## Finetune on Object-level Downstream Tasks 🎯 <a id="finetune-object"></a>

We evaluate the effectiveness of UniPre3D on various object-level downstream tasks, including:
- Object Classification
- Part Segmentation
- Object Detection

### Model Zoo (Pretrained Checkpoints)
We provide pretrained models and checkpoints for object-level tasks in the following table:
| Model | Pretrained Checkpoint | Downstream Task | Performance | Finetuning Logs |
|-------|------------------------|------------------|---------------|-----------|
| Standard Transformer | [Baidu Disk](https://pan.baidu.com/s/1jtepDicFhptP3VDsd00T0g?pwd=gmaw)<br>[Google Drive](https://drive.google.com/drive/folders/1MIHC1oMtcjeBPUaOwtF08t0P0ZcpH-zQ?usp=drive_link) | Classification | 87.93% Acc<br>(+10.69%) | [Logs](https://drive.google.com/drive/folders/1Tzd6pvZ-ADwctMg6MwrJJMbwFDlwXfb-?usp=drive_link) |
| PointMLP | [Baidu Disk](https://pan.baidu.com/s/1jtepDicFhptP3VDsd00T0g?pwd=gmaw)<br>[Google Drive](https://drive.google.com/drive/folders/1MIHC1oMtcjeBPUaOwtF08t0P0ZcpH-zQ?usp=drive_link)  | Classification | 89.5% Acc<br>(+2.1%) | [Logs](https://drive.google.com/drive/folders/1Tzd6pvZ-ADwctMg6MwrJJMbwFDlwXfb-?usp=drive_link) |
| Point Cloud Mamba| [Baidu Disk](https://pan.baidu.com/s/1jtepDicFhptP3VDsd00T0g?pwd=gmaw)<br>[Google Drive](https://drive.google.com/drive/folders/1MIHC1oMtcjeBPUaOwtF08t0P0ZcpH-zQ?usp=drive_link) | Classification | 89.0% Acc<br>(+0.9%) | [Logs](https://drive.google.com/drive/folders/1Tzd6pvZ-ADwctMg6MwrJJMbwFDlwXfb-?usp=drive_link) |
| Mamba3D  | [Baidu Disk](https://pan.baidu.com/s/1jtepDicFhptP3VDsd00T0g?pwd=gmaw)<br>[Google Drive](https://drive.google.com/drive/folders/1MIHC1oMtcjeBPUaOwtF08t0P0ZcpH-zQ?usp=drive_link) | Classification | 93.4% Acc<br>(+0.8%) | [Logs](https://drive.google.com/drive/folders/1Tzd6pvZ-ADwctMg6MwrJJMbwFDlwXfb-?usp=drive_link) |
| PointMLP | [Baidu Disk](https://pan.baidu.com/s/1jtepDicFhptP3VDsd00T0g?pwd=gmaw)<br>[Google Drive](https://drive.google.com/drive/folders/1MIHC1oMtcjeBPUaOwtF08t0P0ZcpH-zQ?usp=drive_link) | Part Segmentation | 85.5% $\text{mIoU}_C$<br>(+0.9%)| [Logs](https://drive.google.com/drive/folders/1Tzd6pvZ-ADwctMg6MwrJJMbwFDlwXfb-?usp=drive_link) |

For more details on the usage of downstream tasks, please refer to the **[docs/OBJECT_LEVEL_DOWNSTREAM_TASKS.md](docs/OBJECT_LEVEL_DOWNSTREAM_TASKS.md)** file.


## Scene-level Pretraining 🏠 <a id="scene-level-pretraining-setup"></a>

Scene-level pretraining focuses on learning representations from complex 3D environments containing multiple objects and spatial relationships. This approach helps models understand large-scale geometric structures and spatial contexts that are crucial for scene understanding tasks.

**Key Characteristics:**
- Processes complete indoor/outdoor scenes rather than individual objects
- Captures long-range spatial relationships and contextual information
- Optimized for tasks like semantic segmentation and instance segmentation

<div align=center>
<img src='assets\scene-vis.png' width="75%">
</div>


### Usage


Sparse Unet pretraining:
```bash
CUDA_VISIBLE_DEVICES=<GPUs> python train_network.py --config-name sparseunet_pretraining
```

PTv3 pretraining:
```bash
CUDA_VISIBLE_DEVICES=<GPUs> python train_network.py --config-name ptv3_pretraining
```

> We cache dataset images in memory to accelerate data loading.  If you encounter memory constraints: Disable this feature by setting `opt.record_img` to `false` in `configs/settings.yaml`

## Finetune on Scene-level Downstream Tasks 🎯 <a id="finetune-scene"></a>

We evaluate the effectiveness of UniPre3D on various object-level downstream tasks, including:
- Semantic Segmentation
- Instance Segmentation
- 3D Object Detection

### Model Zoo (Pretrained Checkpoints)

We provide pretrained models and checkpoints for scene-level tasks in the following table:
| Model | Pretrained Checkpoint | Downstream Task | Dataset | Performance | Finetuning Logs |
|------|----------------------|-----------------|--------------|-------|-----------|
| Sparse Unet | [Baidu Disk](https://pan.baidu.com/s/1dq0JE4eiQPl2g85VUXOcJg?pwd=vv5f) [Google Drive](https://drive.google.com/file/d/1PrXLXpJ0d0tYkUKBozVT4UOPRtIl8XjU/view?usp=drive_link) | Semantic Segmentation | ScanNet20 | 75.8% mIoU<br>(+3.6%) | [Logs](https://drive.google.com/file/d/1-OxryEUVjdYwm7thX_q_EkCX4ESVCmQi/view?usp=drive_link) |
| Sparse Unet | [Baidu Disk](https://pan.baidu.com/s/1dq0JE4eiQPl2g85VUXOcJg?pwd=vv5f) [Google Drive](https://drive.google.com/file/d/1PrXLXpJ0d0tYkUKBozVT4UOPRtIl8XjU/view?usp=drive_link) | Semantic Segmentation | ScanNet200 | 33.0% mIoU<br>(+8.0%) | [Logs](https://drive.google.com/file/d/1eGAK3LjQ7s9nqERXMw14kFgYsvSmhV5r/view?usp=drive_link) |
| Sparse Unet | [Baidu Disk](https://pan.baidu.com/s/1dq0JE4eiQPl2g85VUXOcJg?pwd=vv5f) [Google Drive](https://drive.google.com/file/d/1PrXLXpJ0d0tYkUKBozVT4UOPRtIl8XjU/view?usp=drive_link) | Semantic Segmentation | S3DIS | 71.5% mIoU<br>(+6.1%) | [Logs](https://drive.google.com/file/d/1L-d0aCCxiIDQNwTZKSZ_02OIFZOhjOg3/view?usp=drive_link) |
| Sparse Unet | [Baidu Disk](https://pan.baidu.com/s/1dq0JE4eiQPl2g85VUXOcJg?pwd=vv5f) [Google Drive](https://drive.google.com/file/d/1PrXLXpJ0d0tYkUKBozVT4UOPRtIl8XjU/view?usp=drive_link)  | Instance Segmentation  | ScanNet20 |75.9% mAP@25<br>(+1.2%) | [Logs](https://drive.google.com/file/d/1MbPD4ODXAOqjHrbnENv1NUySo1EvMlwi/view?usp=drive_link) |
| Sparse Unet | [Baidu Disk](https://pan.baidu.com/s/1dq0JE4eiQPl2g85VUXOcJg?pwd=vv5f) [Google Drive](https://drive.google.com/file/d/1PrXLXpJ0d0tYkUKBozVT4UOPRtIl8XjU/view?usp=drive_link) | Instance Segmentation  | ScanNet200 | 37.1%  mAP@25<br>(+2.8%) | [Logs](https://drive.google.com/file/d/1yiuW_hQs1f885Qj7ruHbsURhAxlG02fj/view?usp=drive_link) |
| Point Transformer v3 | [Baidu Disk](https://pan.baidu.com/s/1_FtuKqEhxzcyg7W7YHpFBQ?pwd=kwn2) [Google Drive](https://drive.google.com/file/d/1Q0V5fpq7aWGRkMgYijnD0r-Aw2-z62Zj/view?usp=drive_link) | Semantic Segmentation | ScanNet20 | 76.6% mIoU<br>(+0.1%)| [Logs](https://drive.google.com/file/d/1_MZlklx7SZaPow82tm3PX3MpJQeZpZ4I/view?usp=drive_link) |
| Point Transformer v3 | [Baidu Disk](https://pan.baidu.com/s/1_FtuKqEhxzcyg7W7YHpFBQ?pwd=kwn2) [Google Drive](https://drive.google.com/file/d/1Q0V5fpq7aWGRkMgYijnD0r-Aw2-z62Zj/view?usp=drive_link) | Semantic Segmentation | ScanNet200 |  36.0% mIoU<br>(+0.8%)| [Logs](https://drive.google.com/file/d/1kTbqTjEfK-wKQmBTuhnd3EWlaH7Bkjyj/view?usp=drive_link) |
<!-- | Standard Transformer | [Baidu Disk](https://example.com/pcm_pretrained.pth) [Google Drive]() | Semantic Segmentation | S3DIS | 62.0% mIoU<br>(+2.0%) | [Logs](https://example.com/pcm_logs.txt) | -->

For more details on the usage of downstream tasks, please refer to the **[docs/SCENE_LEVEL_DOWNSTREAM_TASKS.md](docs/SCENE_LEVEL_DOWNSTREAM_TASKS.md)** file.

## Acknowledgements 🙏 <a id="acknowledgements"></a>

We would like to express our gratitude to
- [Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting)
- [Openpoints](https://github.com/guochenqian/openpoints)
- [Pointcept](https://github.com/Pointcept/Pointcept)
- [ShapenetRender_more_variation](https://github.com/Xharlie/ShapenetRender_more_variation)
- [Splatter Image](https://github.com/szymanowiczs/splatter-image)
- [ShapeNet](https://shapenet.org/)
- [ScanNet](http://www.scan-net.org/)
- [PointCloudMamba](https://github.com/SkyworkAI/PointCloudMamba)
- [Mamba3D](https://github.com/xhanxu/Mamba3D)

For any questions about data preparation, please feel free to open an issue in our repository or send email to 1302821779@qq.com

## Citation 📚 <a id="cite"></a>

If you find this work useful in your research, please consider citing:

```bibtex
@inproceedings{wang2025unipre3d,
  title={UniPre3D: Unified Pre-training of 3D Point Cloud Models with Cross-Modal Gaussian Splatting},
  author={Wang, Ziyi and Zhang, Yanran and Zhou, Jie and Lu, Jiwen},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={1319--1329},
  year={2025}
}
```

