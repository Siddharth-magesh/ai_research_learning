import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import math

def compute_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100
) -> float:
    predictions = logits.argmax(dim=-1)
    mask = (targets != ignore_index)
    
    correct = (predictions == targets) & mask
    accuracy = correct.sum().item() / mask.sum().item() if mask.sum() > 0 else 0.0
    
    return accuracy


def compute_perplexity(
    loss: float
) -> float:
    return math.exp(loss)

def compute_cross_entropy_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100
) -> torch.Tensor:
    batch_size, seq_len, vocab_size = logits.shape
    logits = logits.reshape(-1, vocab_size)
    targets = targets.reshape(-1)
    loss = F.cross_entropy(logits, targets, ignore_index=ignore_index)
    
    return loss


class MetricsTracker:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.loss = 0.0
        self.accuracy = 0.0
        self.count = 0
    def update(self, loss: float, accuracy: float):
        self.loss += loss
        self.accuracy += accuracy
        self.count += 1
    def get_average(self) -> Tuple[float, float]:
        if self.count == 0:
            return 0.0, 0.0
        return self.loss / self.count, self.accuracy / self.count
    def get_perplexity(self) -> float:
        avg_loss, _ = self.get_average()
        return compute_perplexity(avg_loss)
