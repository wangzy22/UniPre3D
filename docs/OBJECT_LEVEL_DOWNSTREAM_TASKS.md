# Object-Level Downstream Tasks 🪑


## ScanObjectNN Classification
To evaluate the object classification performance on the **[ScanObjectNN](https://hkust-vgd.github.io/scanobjectnn/)**, clone the model repository, follow their installation instructions, and put the pretrained model in the specified path.

**PointMLP:**
```bash
git clone https://github.com/guochengqian/openpoints.git
# ... Follow the installation instructions in the openpoints repository.
CUDA_VISIBLE_DEVICES=$GPUs python examples/classification/main.py \
    --cfg ./cfgs/scanobjectnn/pointmlp.yaml \
    --pretrained_path /path/to/pointmlp_pretrained.pth \
```

**Standard Transformer:**

For fair comparison, we use the **[ACT](https://github.com/RunpeiDong/ACT)** repository to run the Standard Transformer on ScanObjectNN.
```bash
git clone https://github.com/RunpeiDong/ACT.git
# ... Follow the installation instructions in the ACT repository.
CUDA_VISIBLE_DEVICES=<GPUs> python main.py --config cfgs/finetune_classification/full/finetune_scan_hardest.yaml \
--finetune_model --exp_name <output_file_name> --ckpts <path/to/pre-trained/model>
```

**Mamba3D:**
```bash
git clone https://github.com/xhanxu/Mamba3D.git
# ... Follow the installation instructions in the Mamba3D repository.
bash script/run_scratch.sh
bash script/run_vote.sh
```

**Point Cloud Mamba:**
```bash
git clone https://github.com/SkyworkAI/PointCloudMamba.git
# ... Follow the installation instructions in the Point Cloud Mamba repository.
# train
CUDA_VISIBLE_DEVICES=0 python examples/classification/main.py --cfg cfgs/scanobjectnn/PCM.yaml
# test
CUDA_VISIBLE_DEVICES=0 python examples/classification/main.py --cfg cfgs/scanobjectnn/PCM.yaml  mode=test --pretrained_path /path/to/PCM.pth
```

## Shapenet Part Segmentation
Below are instructions to valuate the part segmentation performance on the **[Shapenet](https://shapenet.org/)**.

**PointMLP:**
```bash
git clone https://github.com/wangzy22/TAP.git 
# ... Follow the installation instructions in the openpoints repository.
CUDA_VISIBLE_DEVICES=$GPUs python examples/shapenetpart/main.py \
    --cfg ./cfgs/shapenetpart/pointmlp_finetune.yaml \
    --pretrained_path /path/to/pointmlp_pretrained.pth \
```

## Note

When loading the model, some repositories may use strict loading mode. Please modify the torch.load parameter to `strict=False` to ensure successful loading.
