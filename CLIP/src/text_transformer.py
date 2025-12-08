
import torch
import torch.nn as nn
from modules.transformer import TransformerBlock

class TextTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int = 49408,
        embed_dim: int = 512,
        max_len: int = 77,
        num_heads: int = 8,
        depth: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        output_dim: int = 512
    ):
        super(TextTransformer, self).__init__()
        self.embed_dim = embed_dim
        self.max_len = max_len
        self.token_embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, embed_dim))
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.proj = nn.Linear(in_features=embed_dim, out_features=output_dim)

    def forward(self, text: torch.Tensor) -> torch.Tensor:
        x = self.token_embed(text)  # x : (B, L, E)
        x = x + self.pos_embed[:, :text.shape[1], :]  # x : (B, L, E)
        x = self.dropout(x)  # x : (B, L, E)
        for block in self.blocks:
            x = block(x)  # x : (B, L, E)
        x = self.norm(x)    
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)]  # x : (B, E)
        x = self.proj(x)
        return x
