import torch
import pytest
from ..models.transformer import Transformer, DecoderOnlyTransformer
from ..models.embeddings import TransformerEmbedding
from ..models.encoder import TransformerEncoder
from ..models.decoder import TransformerDecoder

def test_embedding_shapes():
    batch_size = 4
    seq_len = 20
    vocab_size = 10000
    embed_dim = 512
    max_seq_len = 512
    embedding = TransformerEmbedding(vocab_size, embed_dim, max_seq_len)
    
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    output = embedding(input_ids)
    
    assert output.shape == (batch_size, seq_len, embed_dim)


def test_encoder_shapes():
    batch_size = 4
    seq_len = 20
    embed_dim = 512
    num_layers = 6
    num_heads = 8
    ffn_hidden_dim = 2048
    encoder = TransformerEncoder(num_layers, embed_dim, num_heads, ffn_hidden_dim)
    
    x = torch.randn(batch_size, seq_len, embed_dim)
    output = encoder(x)
    
    assert output.shape == (batch_size, seq_len, embed_dim)


def test_decoder_shapes():
    batch_size = 4
    tgt_seq_len = 15
    src_seq_len = 20
    embed_dim = 512
    num_layers = 6
    num_heads = 8
    ffn_hidden_dim = 2048
    decoder = TransformerDecoder(num_layers, embed_dim, num_heads, ffn_hidden_dim)
    
    tgt = torch.randn(batch_size, tgt_seq_len, embed_dim)
    encoder_output = torch.randn(batch_size, src_seq_len, embed_dim)
    
    output = decoder(tgt, encoder_output)
    
    assert output.shape == (batch_size, tgt_seq_len, embed_dim)


def test_transformer_shapes():
    batch_size = 4
    src_seq_len = 20
    tgt_seq_len = 15
    vocab_size = 10000
    embed_dim = 512
    num_layers = 6
    num_heads = 8
    ffn_hidden_dim = 2048
    max_seq_len = 512
    model = Transformer(
        vocab_size,
        embed_dim,
        num_layers,
        num_heads,
        ffn_hidden_dim,
        max_seq_len
    )
    
    src = torch.randint(0, vocab_size, (batch_size, src_seq_len))
    tgt = torch.randint(0, vocab_size, (batch_size, tgt_seq_len))
    
    output = model(src, tgt)
    
    assert output.shape == (batch_size, tgt_seq_len, vocab_size)


def test_decoder_only_transformer_shapes():
    batch_size = 4
    seq_len = 20
    vocab_size = 10000
    embed_dim = 512
    num_layers = 6
    num_heads = 8
    ffn_hidden_dim = 2048
    max_seq_len = 512
    model = DecoderOnlyTransformer(
        vocab_size,
        embed_dim,
        num_layers,
        num_heads,
        ffn_hidden_dim,
        max_seq_len
    )
    
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    output = model(input_ids)
    
    assert output.shape == (batch_size, seq_len, vocab_size)


def test_different_batch_sizes():
    vocab_size = 10000
    embed_dim = 256
    num_layers = 4
    num_heads = 4
    ffn_hidden_dim = 1024
    max_seq_len = 512
    model = DecoderOnlyTransformer(
        vocab_size,
        embed_dim,
        num_layers,
        num_heads,
        ffn_hidden_dim,
        max_seq_len
    )
    
    for batch_size in [1, 2, 4, 8, 16]:
        seq_len = 10
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        output = model(input_ids)
        assert output.shape == (batch_size, seq_len, vocab_size)


if __name__ == '__main__':
    test_embedding_shapes()
    test_encoder_shapes()
    test_decoder_shapes()
    test_transformer_shapes()
    test_decoder_only_transformer_shapes()
    test_different_batch_sizes()
    print("All shape tests passed!")
