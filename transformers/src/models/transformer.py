import torch
import torch.nn as nn
from typing import Optional
from .embeddings import TransformerEmbedding
from .encoder import TransformerEncoder
from .decoder import TransformerDecoder
from ..modules.masking import create_causal_mask, create_padding_mask


class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        ffn_hidden_dim: int = 2048,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        weight_tying: bool = True
    ):
        super(Transformer, self).__init__()
        
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        
        self.encoder_embedding = TransformerEmbedding(
            vocab_size, embed_dim, max_seq_len, dropout
        )
        self.decoder_embedding = TransformerEmbedding(
            vocab_size, embed_dim, max_seq_len, dropout
        )
        
        self.encoder = TransformerEncoder(
            num_layers, embed_dim, num_heads, ffn_hidden_dim, dropout
        )
        self.decoder = TransformerDecoder(
            num_layers, embed_dim, num_heads, ffn_hidden_dim, dropout
        )
        
        self.output_projection = nn.Linear(embed_dim, vocab_size)
        
        if weight_tying:
            self.output_projection.weight = self.decoder_embedding.token_embedding.embedding.weight
        
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        tgt_seq_len = tgt.size(1)
        causal_mask = create_causal_mask(tgt_seq_len, tgt.device)
        if tgt_mask is not None:
            tgt_padding_mask = create_padding_mask(tgt_mask)
            combined_tgt_mask = causal_mask & tgt_padding_mask
        else:
            combined_tgt_mask = causal_mask
        
        src_embedded = self.encoder_embedding(src)
        encoder_output = self.encoder(src_embedded, src_mask)
        
        tgt_embedded = self.decoder_embedding(tgt)
        decoder_output = self.decoder(
            tgt_embedded,
            encoder_output,
            combined_tgt_mask,
            src_mask
        )
        
        logits = self.output_projection(decoder_output)
        
        return logits
    
    def encode(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        src_embedded = self.encoder_embedding(src)
        return self.encoder(src_embedded, src_mask)
    def decode(
        self,
        tgt: torch.Tensor,
        encoder_output: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        src_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        tgt_seq_len = tgt.size(1)
        causal_mask = create_causal_mask(tgt_seq_len, tgt.device)
        if tgt_mask is not None:
            tgt_padding_mask = create_padding_mask(tgt_mask)
            combined_tgt_mask = causal_mask & tgt_padding_mask
        else:
            combined_tgt_mask = causal_mask
        
        tgt_embedded = self.decoder_embedding(tgt)
        decoder_output = self.decoder(
            tgt_embedded,
            encoder_output,
            combined_tgt_mask,
            src_mask
        )
        
        logits = self.output_projection(decoder_output)
        return logits


class DecoderOnlyTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        ffn_hidden_dim: int = 2048,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        weight_tying: bool = True
    ):
        super(DecoderOnlyTransformer, self).__init__()
        
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        
        self.embedding = TransformerEmbedding(
            vocab_size, embed_dim, max_seq_len, dropout
        )
        
        self.decoder = TransformerDecoder(
            num_layers, embed_dim, num_heads, ffn_hidden_dim, dropout
        )
        
        self.output_projection = nn.Linear(embed_dim, vocab_size)
        
        if weight_tying:
            self.output_projection.weight = self.embedding.token_embedding.embedding.weight
        
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        seq_len = input_ids.size(1)
        causal_mask = create_causal_mask(seq_len, input_ids.device)
        
        if attention_mask is not None:
            padding_mask = create_padding_mask(attention_mask)
            combined_mask = causal_mask & padding_mask
        else:
            combined_mask = causal_mask
        
        x = self.embedding(input_ids)
        
        output = self._decoder_forward_no_cross_attn(x, combined_mask)
        
        logits = self.output_projection(output)
        
        return logits
    
    def _decoder_forward_no_cross_attn(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        for layer in self.decoder.layers:
            self_attn_out, _ = layer.self_attn(layer.norm1(x), mask)
            x = x + layer.dropout(self_attn_out)
            ffn_out = layer.ffn(layer.norm3(x))
            x = x + layer.dropout(ffn_out)
        
        return self.decoder.norm(x)
