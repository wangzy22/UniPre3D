# [CVPR 2025] UniPre3D: Unified Pre-training of 3D Point Cloud Models with Cross-Modal Gaussian Splatting


Created by [Ziyi Wang*](https://wangzy22.github.io/), [Yanran Zhang*](https://github.com/Zhangyr2022), [Jie Zhou](https://scholar.google.com/citations?user=6a79aPwAAAAJ&hl=en&authuser=1), [Jiwen Lu](https://scholar.google.com/citations?user=TN8uDQoAAAAJ&hl=zh-CN) (* indicates equal contribution)

This repository is an official implementation of **UniPre3D (CVPR 2025)**.

**[[Paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_UniPre3D_Unified_Pre-training_of_3D_Point_Cloud_Models_with_Cross-Modal_CVPR_2025_paper.pdf)]** | **[[Project Page](https://ivg-yanranzhang.github.io/UniPre3D/)]**

<div align=center>
<img src='assets\teaser.png' width=350 height=300>
</div>

UniPre3D is the first unified pre-training method for 3D point clouds that effectively handles both object- and scene-level data through cross-modal Gaussian splatting.

<div align=center>
<img src='assets\pipeline.png' width=450 height=260>
</div>

Our proposed pre-training task involves predicting Gaussian parameters from the input point cloud. The 3D backbone network is expected to extract representative features, and 3D Gaussian splatting is implemented to render images for direct supervision. To incorporate additional texture information and adjust task complexity, we introduce a pre-trained image model and propose a scale-adaptive fusion block to accommodate varying data scales.

## News 🔥

- [2025-06-12] Our arXiv paper is released.
- [2025-06-11] Our pretraining code is released.
- [2025-02-27] Our paper is accepted by CVPR 2025.

## TODO (In Progress) ⭐

- [x] Release datasets
- [x] Release object-level pretraining code.
- [x] Release object-level logs and checkpoints.
- [x] Add more details about diverse downstream tasks.
- [ ] Release scene-level pretraining code.
- [ ] Release scene-level logs and checkpoints.

## Visualization Results 📷

Below is visualization of UniPre3D pre-training outputs. The first row presents the input point clouds, followed by the reference view images in the second row. The third row displays the rendered images, which are supervised by the ground truth images shown in the fourth row. In the rightmost column, we illustrate a schematic diagram of the view selection principle for both object- and scene-level samples.

<div align=center>
<img src='assets\visualization.png' width=850 height=280>
</div>

# Getting Started 🚀
## Table of Contents 📖

1. [Environment Setup 🔧](#environment-setup)
    - [Recommended Environment](#recommended-environment)
    - [Hardware Requirements](#hardware-requirements)
    - [Data Preparation](#data-preparation)
2. [Object-level Pretraining 🪑](#object-pretraining)
    - [Usage](#usage)
    - [Finetune on Object-level Downstream Tasks 🎯](#finetune-object)
    - [Model Zoo (Pretrained Checkpoints)](#model-zoo-pretrained-checkpoints)
3. [Scene-level Pretraining 🏠](#scene-level-pretraining-setup)
    - [Usage](#usage-1)
    - [Finetune on Scene-level Downstream Tasks 🎯(Coming soon...)](#finetune-scene)
    - [Model Zoo (Pretrained Checkpoints)(Coming soon...)](#model-zoo-pretrained-checkpoints-1)
4. [Acknowledgements 🙏](#acknowledgements)
5. [Citation 📚](#cite)

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
| Standard Transformer | [Baidu Disk](https://pan.baidu.com/s/1jtepDicFhptP3VDsd00T0g?pwd=gmaw)<br>[Google Drive](https://drive.google.com/drive/folders/1MIHC1oMtcjeBPUaOwtF08t0P0ZcpH-zQ?usp=drive_link
) | Classification | 87.93% Acc<br>(+10.69%) | [Logs](https://drive.google.com/drive/folders/1Tzd6pvZ-ADwctMg6MwrJJMbwFDlwXfb-?usp=drive_link) |
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

Coming soon...

## Finetune on Scene-level Downstream Tasks 🎯 <a id="finetune-scene"></a>

We evaluate the effectiveness of UniPre3D on various object-level downstream tasks, including:
- Semantic Segmentation
- Instance Segmentation

### Model Zoo (Pretrained Checkpoints)

Coming soon...

## Acknowledgements 🙏 <a id="acknowledgements"></a>

We would like to express our gratitude to
- [Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting)
- [Openpoints](https://github.com/guochenqian/openpoints)
- [Pointcept](https://github.com/Pointcept/Pointcept)
- [ShapenetRender_more_variation](https://github.com/Xharlie/ShapenetRender_more_variation)
- [Splatter Image](https://github.com/szymanowiczs/splatter-image)
- [ShapeNet](https://shapenet.org/)
- [ScanNet](http://www.scan-net.org/)

For any questions about data preparation, please feel free to open an issue in our repository.

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

