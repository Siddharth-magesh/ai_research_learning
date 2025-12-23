"""
Configuration file for contrastive language-image prtraining pipeline.
All hyperparameters and settings are defined here.
"""

import torch
from dataclasses import dataclass
from typing import Tuple

@dataclass
class Config:
    # Vision Transformer settings
    img_size: int = 224
    patch_size: int = 16
    in_channels: int = 3
    embed_dim: int = 768
    depth: int = 12
    num_heads: int = 12
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    output_dim: int = 512
    vision_dropout: float = 0.1

    # Text Transformer settings
    vocab_size: int = 49408
    text_embed_dim: int = 512
    max_len: int = 77
    text_num_heads: int = 8
    text_depth: int = 8
    text_mlp_ratio: float = 4.0
    text_dropout: float = 0.1
    text_output_dim: int = 512

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 64
    num_epochs: int = 20
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    temperature: float = 0.07
    num_workers: int = 4 if torch.cuda.is_available() else 2
    pin_memory: bool = True if torch.cuda.is_available() else False
    save_dir: str = "./contrastive-language-image-pretraining/checkpoints"

    def __post_init__(self):
        assert self.img_size % self.patch_size == 0, "Image size must be divisible by patch size."
        assert self.max_len > 0, "Maximum text length must be positive."
        assert self.batch_size > 0, "Batch size must be positive."
        assert self.num_epochs > 0, "Number of epochs must be positive."
        assert self.learning_rate > 0, "Learning rate must be positive."
        assert self.embed_dim > 0, "Embedding dimension must be positive."

    def display(self):
        print("Configuration:")
        for field in self.__dataclass_fields__:
            print(f"  {field}: {getattr(self, field)}")