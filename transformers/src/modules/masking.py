import torch

def create_padding_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    return attention_mask.unsqueeze(1).unsqueeze(2)

def create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return mask.unsqueeze(0).unsqueeze(0)
