# Scene-Level Downstream Tasks 🏠

For various downstream tasks of **Sparse Unet** and **Point Transformer v3**, please follow the instructions in the [Pointcept](https://github.com/Pointcept/Pointcept) repository. 

For the **Standard Transformer** on the S3DIS Semantic Segmentation task, use the following command:

```bash
git clone https://github.com/RunpeiDong/ACT.git
# ... Follow the installation instructions in the ACT repository.
cd semantic_segmentation
python main.py --ckpts <path/to/pre-trained/model> --root path/to/data --learning_rate 0.0002 --epoch 60
```
