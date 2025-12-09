import torch
from transformer import Transformer

def main():
    src_vocab_size = 10000
    tgt_vocab_size = 10000
    batch_size = 32
    src_seq_len = 20
    tgt_seq_len = 15

    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=512,
        n_heads=8,
        n_encoder_layers=6,
        n_decoder_layers=6,
        d_ff=2048,
        dropout=0.1
    )

    src = torch.randint(0, src_vocab_size, (batch_size, src_seq_len))
    tgt = torch.randint(0, tgt_vocab_size, (batch_size, tgt_seq_len))

    tgt_mask = Transformer.create_causual_mask(tgt_seq_len).unsqueeze(0).unsqueeze(0)

    output = model(src, tgt, tgt_mask=tgt_mask)

    print(f"Input shape: {src.shape}")
    print(f"Target shape: {tgt.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    return