import torch
import os
from dataclasses import dataclass

@dataclass
class BaseConfig:
    project_name: str = "transformers-from-scratch"
    experiment_name: str = "baseline"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    num_workers: int = os.cpu.count()
    log_interval: int = 100
    save_interval: int = 1
