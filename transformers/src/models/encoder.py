import torch
import torch.nn as nn
from typing import Optional
from ..modules.attention import MultiHeadAttention
from ..modules.feed_forward import FeedForward


class EncoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_hidden_dim: int,
        dropout: float = 0.1
    ):
        super(EncoderLayer, self).__init__()
        
        self.self_attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        
        self.ffn = FeedForward(embed_dim, ffn_hidden_dim, dropout)
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        attn_out, _ = self.self_attn(self.norm1(x), mask)
        x = x + self.dropout(attn_out)
        ffn_out = self.ffn(self.norm2(x))
        x = x + self.dropout(ffn_out)
        
        return x


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        embed_dim: int,
        num_heads: int,
        ffn_hidden_dim: int,
        dropout: float = 0.1
    ):
        super(TransformerEncoder, self).__init__()
        
        self.layers = nn.ModuleList([
            EncoderLayer(embed_dim, num_heads, ffn_hidden_dim, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
