"""
Visualization script for Siamese Network results.
Run this separately after training to generate plots.
"""

import os
import pickle
import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from config import Config
from modules.embedding_network import SimpleEmbeddingNetwork
from siamese_network import SiameseNetwork
from data_loader import download_and_setup_data, create_data_loaders
from evaluate import Evaluator


def load_checkpoint(checkpoint_path='checkpoints/best_model.pth'):
    """Load trained model checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    return checkpoint


def plot_training_curves(checkpoint_path='checkpoints/best_model.pth', 
                        save_dir='visualizations'):
    """
    Plot training and validation curves from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    checkpoint = load_checkpoint(checkpoint_path)
    history = checkpoint.get('history', {})
    
    if not history:
        print("No history found in checkpoint")
        return
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Loss plot
    axes[0].plot(epochs, history['train_loss'], 'b-o', label='Train', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-s', label='Validation', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training & Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axes[1].plot(epochs, history['val_accuracy'], 'g-^', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Validation Accuracy')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 1])
    
    # Learning rate plot
    axes[2].plot(epochs, history['learning_rate'], 'm-d', linewidth=2)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Learning Rate')
    axes[2].set_title('Learning Rate Schedule')
    axes[2].set_yscale('log')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'training_curves.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.show()


def plot_evaluation_results(config=None, checkpoint_path='checkpoints/best_model.pth',
                           save_dir='visualizations'):
    """
    Generate evaluation plots.
    
    Args:
        config: Configuration object
        checkpoint_path: Path to model checkpoint
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    if config is None:
        config = Config()
    
    device = torch.device(config.device)
    
    # Load model
    checkpoint = load_checkpoint(checkpoint_path)
    embedding_net = SimpleEmbeddingNetwork(
        embedding_dim=config.embedding_dim,
        input_size=config.image_size
    )
    model = SiameseNetwork(embedding_network=embedding_net).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load data
    signature_data_dir = download_and_setup_data(config)
    _, val_loader, _, _ = create_data_loaders(signature_data_dir, config)
    
    # Evaluate
    evaluator = Evaluator(model, val_loader, config, device)
    metrics = evaluator.evaluate()
    
    # Plot distance distributions
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    distances_genuine = metrics['distances_genuine']
    distances_fake = metrics['distances_fake']
    
    # Histogram
    axes[0].hist(distances_genuine, bins=50, alpha=0.6, label='Genuine', 
                color='green', edgecolor='black')
    axes[0].hist(distances_fake, bins=50, alpha=0.6, label='Forged', 
                color='red', edgecolor='black')
    axes[0].axvline(config.threshold_distance, color='blue', linestyle='--', 
                   linewidth=2, label='Threshold')
    axes[0].set_xlabel('Distance')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distance Distribution')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Box plot
    bp = axes[1].boxplot([distances_genuine, distances_fake], 
                         labels=['Genuine', 'Forged'], patch_artist=True)
    bp['boxes'][0].set_facecolor('green')
    bp['boxes'][1].set_facecolor('red')
    axes[1].axhline(config.threshold_distance, color='blue', linestyle='--', 
                   linewidth=2, label='Threshold')
    axes[1].set_ylabel('Distance')
    axes[1].set_title('Distance Box Plot')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'distance_distributions.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.show()
    
    # Plot confusion matrix
    cm = metrics['confusion_matrix']
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=['Forged', 'Genuine'],
           yticklabels=['Forged', 'Genuine'],
           title='Confusion Matrix',
           ylabel='True Label',
           xlabel='Predicted Label')
    
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black",
                   fontsize=20)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'confusion_matrix.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.show()
    
    # Plot ROC curve
    if metrics['roc_curve'] is not None:
        fpr, tpr, _ = metrics['roc_curve']
        auc = metrics['auc']
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        save_path = os.path.join(save_dir, 'roc_curve.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
        plt.show()


def visualize_all(config=None, checkpoint_path='checkpoints/best_model.pth'):
    """
    Generate all visualizations.
    
    Args:
        config: Configuration object
        checkpoint_path: Path to model checkpoint
    """
    print("=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)
    
    save_dir = 'visualizations'
    os.makedirs(save_dir, exist_ok=True)
    
    # Training curves
    print("\nPlotting training curves...")
    plot_training_curves(checkpoint_path, save_dir)
    
    # Evaluation results
    print("\nPlotting evaluation results...")
    plot_evaluation_results(config, checkpoint_path, save_dir)
    
    print("\n" + "=" * 60)
    print(f"All visualizations saved to: {save_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    # Run visualization
    visualize_all()