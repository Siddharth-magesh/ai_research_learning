
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from vision_transformer import VisionTransformer
from text_transformer import TextTransformer
from typing import Tuple

class CLIP(nn.Module):
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        vision_embed_dim: int = 768,
        vision_depth: int = 12,
        vision_heads: int = 12,
        vocab_size: int = 49408,
        text_embed_dim: int = 512,
        max_len: int = 77,
        text_heads: int = 8,
        text_depth: int = 12,
        output_dim: int = 512,
        temperature: float = 0.07
    ):
        super(CLIP, self).__init__()
        self.visual = VisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=vision_embed_dim,
            depth=vision_depth,
            num_heads=vision_heads,
            output_dim=output_dim
        )
        self.text = TextTransformer(
            vocab_size=vocab_size,
            embed_dim=text_embed_dim,
            max_len=max_len,
            num_heads=text_heads,
            depth=text_depth,
            output_dim=output_dim
        )
        self.temperature = nn.Parameter(torch.ones([]) * np.log(1 / temperature))

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        image_features = self.visual(image)
        image_features = F.normalize(image_features, dim=-1)
        return image_features

    def encode_text(self, text: torch.Tensor) -> torch.Tensor:
        text_features = self.text(text)
        text_features = F.normalize(text_features, dim=-1)
        return text_features

    def forward(self, image: torch.Tensor, text: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)
        logits = image_features @ text_features.T * torch.exp(self.temperature)
        return logits, image_features, text_features

class CLIPLoss(nn.Module):
    def __init__(self):
        super(CLIPLoss, self).__init__()
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        batch_size = logits.shape[0]
        labels = torch.arange(batch_size, device=logits.device)
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.T, labels)
        loss = (loss_i2t + loss_t2i) / 2
        return loss