import torch
import torch.nn as nn
from torch.optim import Adam, AdamW, SGD
from typing import Dict, Any

def create_optimizer(
    model: nn.Module,
    config: Dict[str, Any]
) -> torch.optim.Optimizer:
    optimizer_type = config.get('type', 'adamw').lower()
    lr = config.get('lr', 3e-4)
    weight_decay = config.get('weight_decay', 0.01)
    betas = tuple(config.get('betas', [0.9, 0.999]))
    eps = config.get('eps', 1e-8)
    params = model.parameters()
    
    if optimizer_type == 'adam':
        optimizer = Adam(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay
        )
    elif optimizer_type == 'adamw':
        optimizer = AdamW(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay
        )
    elif optimizer_type == 'sgd':
        momentum = config.get('momentum', 0.9)
        optimizer = SGD(
            params,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    return optimizer


def get_parameter_groups(
    model: nn.Module,
    weight_decay: float = 0.01,
    no_decay_bias: bool = True
) -> list:
    if not no_decay_bias:
        return [{'params': model.parameters(), 'weight_decay': weight_decay}]
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        if 'bias' in name or 'norm' in name or 'ln' in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    
    return [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ]
