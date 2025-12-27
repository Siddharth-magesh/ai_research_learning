import torch
import torch.nn as nn
from typing import Optional

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.embed_dim = embed_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x) * (self.embed_dim ** 0.5)

class TransformerEmbedding(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        max_seq_len: int = 512,
        dropout: float = 0.1
    ):
        super(TransformerEmbedding, self).__init__()
        self.token_embedding = TokenEmbedding(vocab_size, embed_dim)
        self.positional_encoding = self._create_positional_encoding(max_seq_len, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.embed_dim = embed_dim
    
    def _create_positional_encoding(self, max_seq_len: int, embed_dim: int) -> torch.Tensor:
        import math
        pe = torch.zeros(max_seq_len, embed_dim)
        position = torch.arange(0, max_seq_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        return pe
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        token_emb = self.token_embedding(x)
        pos_enc = self.positional_encoding[:, :seq_len, :].to(x.device)
        x = token_emb + pos_enc
        
        return self.dropout(x)
