import torch
from torch.optim.lr_scheduler import (
    LambdaLR,
    CosineAnnealingLR,
    StepLR,
    ReduceLROnPlateau
)
from typing import Dict, Any
import math

def create_scheduler(
    optimizer: torch.optim.Optimizer,
    config: Dict[str, Any],
    num_training_steps: int = None
):
    scheduler_type = config.get('type', 'linear_warmup').lower()
    warmup_steps = config.get('warmup_steps', 1000)
    if scheduler_type == 'linear_warmup':
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps or 100000
        )
    elif scheduler_type == 'cosine':
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps or 100000
        )
    elif scheduler_type == 'step':
        step_size = config.get('step_size', 1000)
        gamma = config.get('gamma', 0.1)
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_type == 'plateau':
        factor = config.get('factor', 0.1)
        patience = config.get('patience', 10)
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=factor,
            patience=patience
        )
    elif scheduler_type == 'transformer':
        d_model = config.get('d_model', 512)
        scheduler = get_transformer_schedule(
            optimizer,
            d_model=d_model,
            warmup_steps=warmup_steps
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    return scheduler


def get_linear_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    last_epoch: int = -1
):
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0,
            float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1
):
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * num_cycles * 2.0 * progress)))
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_transformer_schedule(
    optimizer: torch.optim.Optimizer,
    d_model: int,
    warmup_steps: int = 4000,
    last_epoch: int = -1
):
    def lr_lambda(current_step: int):
        current_step = max(1, current_step)
        arg1 = current_step ** (-0.5)
        arg2 = current_step * (warmup_steps ** (-1.5))
        return (d_model ** (-0.5)) * min(arg1, arg2)
    return LambdaLR(optimizer, lr_lambda, last_epoch)
