import torch
import pytest
from ..modules.attention import ScaledDotProductAttention, MultiHeadAttention

def test_scaled_dot_product_attention():
    batch_size = 2
    seq_len = 10
    d_k = 64
    attention = ScaledDotProductAttention(dropout=0.1)
    
    Q = torch.randn(batch_size, seq_len, d_k)
    K = torch.randn(batch_size, seq_len, d_k)
    V = torch.randn(batch_size, seq_len, d_k)
    
    output, attn_weights = attention(Q, K, V)
    
    assert output.shape == (batch_size, seq_len, d_k)
    assert attn_weights.shape == (batch_size, seq_len, seq_len)
    
    assert torch.allclose(
        attn_weights.sum(dim=-1),
        torch.ones(batch_size, seq_len),
        atol=1e-6
    )


def test_multi_head_attention():
    batch_size = 2
    seq_len = 10
    embed_dim = 512
    num_heads = 8
    mha = MultiHeadAttention(embed_dim, num_heads, dropout=0.1)
    
    x = torch.randn(batch_size, seq_len, embed_dim)
    
    output, attn_weights = mha(x)
    
    assert output.shape == (batch_size, seq_len, embed_dim)
    assert attn_weights.shape == (batch_size, num_heads, seq_len, seq_len)


def test_multi_head_attention_with_mask():
    batch_size = 2
    seq_len = 10
    embed_dim = 512
    num_heads = 8
    mha = MultiHeadAttention(embed_dim, num_heads, dropout=0.1)
    
    x = torch.randn(batch_size, seq_len, embed_dim)
    mask = torch.ones(batch_size, seq_len)
    mask[:, -2:] = 0
    
    output, attn_weights = mha(x, mask)
    
    assert output.shape == (batch_size, seq_len, embed_dim)


if __name__ == '__main__':
    test_scaled_dot_product_attention()
    test_multi_head_attention()
    test_multi_head_attention_with_mask()
    print("All attention tests passed!")
