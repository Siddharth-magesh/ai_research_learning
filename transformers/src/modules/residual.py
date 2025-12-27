import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(
        self,
        sublayer: nn.Module,
        embed_dim: int,
        dropout: float = 0.1
    ) -> None:
        super(ResidualBlock, self).__init__()
        self.sublayer = sublayer
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        out = self.sublayer(self.norm(x), *args, **kwargs)
        out = self.dropout(out)
        return x + out
