
import torch
import torch.nn as nn
from modules.patch_embedding import PatchEmbedding
from modules.transformer import TransformerBlock

class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        output_dim: int = 512
    ):
        super(VisionTransformer, self).__init__()
        self.patch_embed = PatchEmbedding(img_size=img_size, patch_size=patch_size, in_channels=in_channels, embed_dim=embed_dim)
        n_patches = self.patch_embed.n_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.proj = nn.Linear(in_features=embed_dim, out_features=output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.patch_embed(x) # x : (B, N, E)
        cls_token = self.cls_token.expand(B, -1, -1) # cls_token : (B, 1, E)
        x = torch.cat([cls_token, x], dim=1)  # x : (B, N+1, E)
        x = x + self.pos_embed  # x : (B, N+1, E)
        x = self.dropout(x)  # x : (B, N+1, E)
        for block in self.blocks:
            x = block(x) # x : (B, N+1, E)
        x = self.norm(x)
        x = x[:, 0]  # x : (B, E)
        x = self.proj(x)  # x : (B, output_dim)
        return x