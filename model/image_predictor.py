import torch
import torch.nn as nn
import os
import pickle
from diffusers import AutoencoderKL

from typing import Any, List, Optional, Tuple, Union, Dict


class ImageFeaturePredictor(nn.Module):
    """Image feature extractor using AutoencoderKL"""

    def __init__(
        self,
        cfg: Any,
        out_channels: int,
        pretrained_path: Optional[str] = None,
    ) -> None:
        super(ImageFeaturePredictor, self).__init__()
        self.out_channels = out_channels
        self.cfg = cfg

        # Initialize encoder
        # self.encoder = AutoencoderKL.from_single_file("https://huggingface.co/stabilityai/sd-vae-ft-mse-original/blob/main/vae-ft-mse-840000-ema-pruned.safetensors",resume_download=True)
        weights_path = os.path.abspath(
            os.path.normpath(os.path.join(os.path.dirname(__file__), "../weights/"))
        )
        self.encoder = AutoencoderKL.from_pretrained(
            weights_path, local_files_only=True, use_safetensors=False
        )
        print("Load ImageFeaturePredictor successfully!")
        self.encoder.eval()
        self.encoder.to(torch.cuda.current_device())
        
        # Set the model not to require gradients
        for param in self.encoder.parameters():
            param.requires_grad = False
            
        # Load encoder config
        self.encoder_config = self.encoder.config

    def _load_pretrained_weights(self, pretrained_path):
        """Load pretrained weights from file"""
        if pretrained_path.endswith(".pth"):
            pretrained_ckpt = torch.load(pretrained_path)
        elif pretrained_path.endswith(".pkl"):
            with open(pretrained_path, "rb") as f:
                pretrained_ckpt = pickle.load(f)

        incompatible = self.encoder.load_state_dict(
            pretrained_ckpt["model"], strict=False
        )
        if incompatible:
            print(f"Incompatible keys: {incompatible}")

    def forward(
        self,
        x: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Extract features from input images"""
        features = {}
        hooks = []

        # Register hooks for each decoder block
        for i, block in enumerate(self.encoder.decoder.up_blocks):

            def hook_factory(block_idx):
                def hook(_, __, output):
                    features[f"decoder_block_{block_idx}"] = output.clone()

                return hook

            hooks.append(block.register_forward_hook(hook_factory(i)))

        with torch.no_grad():
            self.encoder(x)

        for h in hooks:
            h.remove()

        return features
