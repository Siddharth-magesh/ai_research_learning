"""
Configuration file for Siamese Network training pipeline.
All hyperparameters and settings are defined here.
"""

import torch
from dataclasses import dataclass
from typing import Tuple


@dataclass
class Config:
    """
    Configuration class for training and evaluation.
    
    IMPORTANT: Current values are set for lightweight training on laptop.
    For optimal results on powerful machine, use the commented values.
    """
    
    # Dataset Configuration
    triplets_per_user: int = 20  # Lightweight: 20, Optimal: 100-200
    train_split: float = 0.8
    
    # Image Configuration
    image_size: Tuple[int, int] = (128, 128)  # Lightweight: (128,128), Optimal: (224,224)
    image_mean: Tuple[float, float, float] = (0.861, 0.861, 0.861)
    image_std: Tuple[float, float, float] = (0.274, 0.274, 0.274)
    
    # Model Architecture
    embedding_dim: int = 64  # Lightweight: 64, Optimal: 128-256
    
    # Training Hyperparameters
    batch_size: int = 8  # Lightweight: 8, Optimal: 32-64
    num_epochs: int = 3  # Lightweight: 3, Optimal: 20-50
    learning_rate: float = 1e-3  # Optimal: 1e-3 to 1e-4
    weight_decay: float = 1e-4  # L2 regularization
    
    # Loss Function
    triplet_margin: float = 1.0  # Optimal: 1.0-2.0
    triplet_p: int = 2  # L2 distance
    
    # Learning Rate Scheduler
    scheduler_step_size: int = 2  # Lightweight: 2, Optimal: 5-10
    scheduler_gamma: float = 0.5  # Optimal: 0.5-0.7
    
    # Evaluation
    threshold_distance: float = 0.8  # Distance threshold for verification
    
    # Data Loading
    num_workers: int = 2  # Lightweight: 2, Optimal: 4-8
    pin_memory: bool = True if torch.cuda.is_available() else False
    
    # Checkpointing
    save_dir: str = "./checkpoints"
    save_best_only: bool = True
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Data Augmentation (Training)
    random_affine_degrees: int = 0
    random_affine_shear: int = 10  # Optimal: 10-15
    random_affine_translate: Tuple[float, float] = (0.1, 0.1)
    random_perspective_distortion: float = 0.1  # Optimal: 0.1-0.2
    random_perspective_prob: float = 0.5
    
    # Logging
    print_every: int = 10  # Print loss every N batches
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        assert 0.0 < self.train_split < 1.0, "train_split must be between 0 and 1"
        assert self.batch_size > 0, "batch_size must be positive"
        assert self.num_epochs > 0, "num_epochs must be positive"
        assert self.learning_rate > 0, "learning_rate must be positive"
        assert self.embedding_dim > 0, "embedding_dim must be positive"
        
    def display(self):
        """Display current configuration."""
        print("=" * 60)
        print("CONFIGURATION SETTINGS")
        print("=" * 60)
        print(f"Device: {self.device}")
        print(f"Image Size: {self.image_size}")
        print(f"Embedding Dimension: {self.embedding_dim}")
        print(f"Batch Size: {self.batch_size}")
        print(f"Number of Epochs: {self.num_epochs}")
        print(f"Learning Rate: {self.learning_rate}")
        print(f"Triplets per User: {self.triplets_per_user}")
        print(f"Train/Val Split: {self.train_split:.1%} / {1-self.train_split:.1%}")
        print(f"Triplet Margin: {self.triplet_margin}")
        print(f"Threshold Distance: {self.threshold_distance}")
        print("=" * 60)


# Create default config instance
config = Config()
