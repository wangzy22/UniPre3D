# Installation Guide 🔧

1. **Create and activate conda environment**
```bash
conda create -n UniPre3D python=3.11
conda activate UniPre3D
```

2. **Install PyTorch and dependencies**
```bash
# Install PyTorch with CUDA support
pip install torch==2.2.2 torchvision==0.17.2

# Install project dependencies
pip install -r requirements.txt

# Install flash-attn for efficient attention mechanisms
pip install flash-attn --no-build-isolation
```

3. **Install C++ extensions**
```bash
# Install PointNet++ modules
cd openpoints/cpp/pointnet2_batch
python setup.py install
cd ../

# Install Chamfer Distance and emd modules
cd chamfer_dist
python setup.py install --user
cd ../emd
python setup.py install --user
cd ../../../
```

4. **Install Mamba3D dependencies**
```bash
# Install PointNet2 operations library
pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"

# Install GPU KNN
pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl

# Install Mamba SSM dependencies
pip install causal-conv1d==1.2.2.post1
pip install mamba-ssm==1.2.2
```

causal-conv1d and mamba-ssm are required for the Mamba3D model, you should select the version that matches your CUDA and pytorch version.

5. **Install Gaussian Splatting Renderer**

The Gaussian Splatting renderer is required for rendering Gaussian Point clouds to images.

```bash
# Clone the repository
git clone https://github.com/graphdeco-inria/gaussian-splatting.git --recursive
cd gaussian-splatting

# Install the renderer
pip install submodules/diff-gaussian-rasterization
```

6. **Download pre-trained image feature extractor**

Please download the pre-trained image feature extractor `diffusion_pytorch_model.bin` from [here](https://huggingface.co/stabilityai/sd-vae-ft-mse/tree/main) and put it in the `weights` folder.

### Troubleshooting

- If you encounter issues installing PointNet2 operations, please refer to [this solution](https://github.com/erikwijmans/Pointnet2_PyTorch/issues/174#issuecomment-2232300080) for manual installation steps.
- For Gaussian Splatting, ensure your system meets the [hardware requirements](https://github.com/graphdeco-inria/gaussian-splatting/blob/main/README.md#hardware-requirements).