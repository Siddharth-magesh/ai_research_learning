import torch
import pytest
from ..models.encoder import EncoderLayer, TransformerEncoder

def test_encoder_layer():
    batch_size = 2
    seq_len = 10
    embed_dim = 512
    num_heads = 8
    ffn_hidden_dim = 2048
    encoder_layer = EncoderLayer(embed_dim, num_heads, ffn_hidden_dim, dropout=0.1)
    
    x = torch.randn(batch_size, seq_len, embed_dim)
    
    output = encoder_layer(x)
    
    assert output.shape == (batch_size, seq_len, embed_dim)


def test_encoder_layer_with_mask():
    batch_size = 2
    seq_len = 10
    embed_dim = 512
    num_heads = 8
    ffn_hidden_dim = 2048
    encoder_layer = EncoderLayer(embed_dim, num_heads, ffn_hidden_dim, dropout=0.1)
    
    x = torch.randn(batch_size, seq_len, embed_dim)
    mask = torch.ones(batch_size, seq_len)
    mask[:, -3:] = 0
    
    output = encoder_layer(x, mask)
    
    assert output.shape == (batch_size, seq_len, embed_dim)


def test_transformer_encoder():
    batch_size = 2
    seq_len = 10
    embed_dim = 512
    num_layers = 6
    num_heads = 8
    ffn_hidden_dim = 2048
    encoder = TransformerEncoder(
        num_layers, embed_dim, num_heads, ffn_hidden_dim, dropout=0.1
    )
    
    x = torch.randn(batch_size, seq_len, embed_dim)
    
    output = encoder(x)
    
    assert output.shape == (batch_size, seq_len, embed_dim)


def test_encoder_gradients():
    batch_size = 2
    seq_len = 10
    embed_dim = 512
    num_heads = 8
    ffn_hidden_dim = 2048
    encoder_layer = EncoderLayer(embed_dim, num_heads, ffn_hidden_dim, dropout=0.1)
    
    x = torch.randn(batch_size, seq_len, embed_dim, requires_grad=True)
    
    output = encoder_layer(x)
    loss = output.sum()
    loss.backward()
    
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()


if __name__ == '__main__':
    test_encoder_layer()
    test_encoder_layer_with_mask()
    test_transformer_encoder()
    test_encoder_gradients()
    print("All encoder tests passed!")
