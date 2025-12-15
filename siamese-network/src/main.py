"""
Main training script for Siamese Network.
Run this after optimizing hyperparameters with optimize.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary

from config import Config
from data_loader import download_and_setup_data, create_data_loaders
from modules.embedding_network import SimpleEmbeddingNetwork
from siamese_network import SiameseNetwork
from train import Trainer
from evaluate import Evaluator


def main():
    """
    Main training function.
    """
    # Initialize configuration
    config = Config()
    config.display()
    
    # Set device
    device = torch.device(config.device)
    print(f"\n✓ Using device: {device}")
    
    # Set random seed
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Download and setup data
    print("\n" + "="*60)
    print("DATA LOADING")
    print("="*60)
    signature_data_dir = download_and_setup_data(config)
    train_loader, val_loader, train_dataset, val_dataset = create_data_loaders(
        signature_data_dir, config
    )
    
    # Initialize model
    print("\n" + "="*60)
    print("MODEL INITIALIZATION")
    print("="*60)
    
    embedding_net = SimpleEmbeddingNetwork(
        embedding_dim=config.embedding_dim,
        input_size=config.image_size
    )
    siamese_model = SiameseNetwork(embedding_network=embedding_net).to(device)
    
    # Print model summary using torchinfo
    print("\nModel Architecture Summary:")
    print("-" * 60)
    summary(embedding_net, input_size=(1, 3, config.image_size[0], config.image_size[1]),
            col_names=["input_size", "output_size", "num_params", "trainable"],
            depth=4, device=device)
    
    # Setup training
    print("\n" + "="*60)
    print("TRAINING SETUP")
    print("="*60)
    
    triplet_loss = nn.TripletMarginLoss(
        margin=config.triplet_margin,
        p=config.triplet_p
    )
    
    optimizer = optim.Adam(
        siamese_model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.scheduler_step_size,
        gamma=config.scheduler_gamma
    )
    
    print(f"Loss: Triplet Margin Loss (margin={config.triplet_margin})")
    print(f"Optimizer: Adam (lr={config.learning_rate})")
    print(f"Scheduler: StepLR (step={config.scheduler_step_size}, gamma={config.scheduler_gamma})")
    
    # Train model
    print("\n" + "="*60)
    print("TRAINING")
    print("="*60)
    
    trainer = Trainer(
        model=siamese_model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=triplet_loss,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        device=device
    )
    
    trained_model, history = trainer.train()
    
    # Evaluate model
    print("\n" + "="*60)
    print("EVALUATION")
    print("="*60)
    
    evaluator = Evaluator(
        model=trained_model,
        val_loader=val_loader,
        config=config,
        device=device
    )
    
    metrics = evaluator.evaluate()
    
    # Find optimal threshold
    optimal_threshold, best_accuracy, _, _ = evaluator.find_optimal_threshold()
    
    # Final summary
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"✓ Best Val Accuracy: {trainer.best_val_acc:.4f}")
    print(f"✓ Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"✓ Test AUC-ROC: {metrics['auc']:.4f}")
    print(f"✓ Optimal Threshold: {optimal_threshold:.4f}")
    print(f"✓ Model saved: {config.save_dir}/best_model.pth")
    print(f"✓ TensorBoard logs: runs/siamese_network")
    print("\nView training progress:")
    print("  tensorboard --logdir=runs")
    print("="*60)


if __name__ == "__main__":
    main()