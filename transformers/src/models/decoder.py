import torch
import torch.nn as nn
from typing import Optional
from ..modules.attention import MultiHeadAttention
from ..modules.feed_forward import FeedForward


class DecoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_hidden_dim: int,
        dropout: float = 0.1
    ):
        super(DecoderLayer, self).__init__()
        
        self.self_attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        
        self.cross_attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        
        self.ffn = FeedForward(embed_dim, ffn_hidden_dim, dropout)
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        self_attn_mask: Optional[torch.Tensor] = None,
        cross_attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        self_attn_out, _ = self.self_attn(self.norm1(x), self_attn_mask)
        x = x + self.dropout(self_attn_out)
        cross_attn_out, _ = self._cross_attention(
            self.norm2(x), encoder_output, cross_attn_mask
        )
        x = x + self.dropout(cross_attn_out)
        
        ffn_out = self.ffn(self.norm3(x))
        x = x + self.dropout(ffn_out)
        
        return x
    
    def _cross_attention(
        self,
        query: torch.Tensor,
        encoder_output: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> tuple:
        B, T_q, D = query.size()
        T_k = encoder_output.size(1)
        Q = self.cross_attn.q_linear(query)
        K = self.cross_attn.k_linear(encoder_output)
        V = self.cross_attn.v_linear(encoder_output)
        
        Q = Q.view(B, T_q, self.cross_attn.num_heads, self.cross_attn.head_dim).transpose(1, 2)
        K = K.view(B, T_k, self.cross_attn.num_heads, self.cross_attn.head_dim).transpose(1, 2)
        V = V.view(B, T_k, self.cross_attn.num_heads, self.cross_attn.head_dim).transpose(1, 2)
        
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
        
        context, attn_weights = self.cross_attn.attention(Q, K, V, mask)
        
        context = context.transpose(1, 2).contiguous().view(B, T_q, D)
        output = self.cross_attn.out_linear(context)
        
        return output, attn_weights


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        embed_dim: int,
        num_heads: int,
        ffn_hidden_dim: int,
        dropout: float = 0.1
    ):
        super(TransformerDecoder, self).__init__()
        
        self.layers = nn.ModuleList([
            DecoderLayer(embed_dim, num_heads, ffn_hidden_dim, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        self_attn_mask: Optional[torch.Tensor] = None,
        cross_attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, encoder_output, self_attn_mask, cross_attn_mask)
        return self.norm(x)
