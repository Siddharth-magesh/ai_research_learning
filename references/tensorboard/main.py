"""TensorBoard Comprehensive Reference - All Logging Features

This module demonstrates all major TensorBoard logging capabilities:
- Scalars (loss, metrics, learning rate)
- Images and figures
- Model graphs and architecture
- Histograms and distributions
- Embeddings and projections
- Hyperparameters
- Text and markdown
- Custom visualizations
- PR curves and confusion matrices

Usage:
    python main.py
    tensorboard --logdir runs
"""

from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools


def log_scalars_demo(writer, num_epochs=50):
    """Demonstrate scalar logging - most common use case."""
    print("\n=== Logging Scalars ===")
    
    for epoch in range(num_epochs):
        # Simulate decreasing loss
        train_loss = 2.0 * np.exp(-epoch / 10) + 0.1 * np.random.rand()
        val_loss = 2.2 * np.exp(-epoch / 10) + 0.15 * np.random.rand()
        
        # Simulate increasing accuracy
        train_acc = 1 - np.exp(-epoch / 15) + 0.05 * np.random.rand()
        val_acc = 0.95 - np.exp(-epoch / 15) + 0.05 * np.random.rand()
        
        # Individual scalars
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/validation', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/validation', val_acc, epoch)
        
        # Learning rate schedule (cosine annealing)
        lr = 1e-3 * (0.5 + 0.5 * np.cos(np.pi * epoch / num_epochs))
        writer.add_scalar('Hyperparameters/learning_rate', lr, epoch)
        
        # Multiple scalars in one chart
        writer.add_scalars('Losses_Combined', {
            'train': train_loss,
            'validation': val_loss
        }, epoch)
        
        writer.add_scalars('Accuracies_Combined', {
            'train': train_acc,
            'validation': val_acc
        }, epoch)
        
        # Per-batch logging (simulated)
        for batch in range(10):
            step = epoch * 10 + batch
            batch_loss = train_loss + 0.1 * np.random.randn()
            writer.add_scalar('Batch/loss', batch_loss, step)
    
    print(f"✓ Logged {num_epochs} epochs of scalar data")


def log_images_demo(writer):
    """Demonstrate image logging."""
    print("\n=== Logging Images ===")
    
    # Single image
    img = torch.rand(3, 224, 224)
    writer.add_image('Sample/random_image', img, 0)
    
    # Batch of images (creates grid)
    img_batch = torch.rand(16, 3, 64, 64)
    img_grid = torchvision.utils.make_grid(img_batch, nrow=4)
    writer.add_image('Sample/image_grid', img_grid, 0)
    
    # Multiple images over training
    for epoch in [0, 10, 20, 30, 40]:
        # Simulate improving reconstructions
        noise_level = max(0.1, 1.0 - epoch / 50)
        reconstructed = torch.rand(8, 3, 64, 64) * noise_level
        grid = torchvision.utils.make_grid(reconstructed, nrow=4)
        writer.add_image('Training/reconstructions', grid, epoch)
    
    # Matplotlib figure
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        data = np.random.randn(100, 100)
        im = ax.imshow(data, cmap='viridis')
        ax.set_title(f'Feature Map {i+1}')
        plt.colorbar(im, ax=ax)
    
    writer.add_figure('Visualization/feature_maps', fig, 0)
    plt.close(fig)
    
    print("✓ Logged various image types")


def log_model_graph_demo(writer):
    """Demonstrate model graph visualization."""
    print("\n=== Logging Model Graph ===")
    
    # Simple CNN
    class SimpleCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(64 * 16 * 16, 128)
            self.fc2 = nn.Linear(128, 10)
        
        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(x.size(0), -1)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x
    
    model = SimpleCNN()
    dummy_input = torch.randn(1, 3, 64, 64)
    writer.add_graph(model, dummy_input)
    
    # Pre-trained model
    resnet = torchvision.models.resnet18(pretrained=False)
    dummy_input_resnet = torch.randn(1, 3, 224, 224)
    writer.add_graph(resnet, dummy_input_resnet)
    
    print("✓ Logged model architectures")


def log_histograms_demo(writer):
    """Demonstrate histogram logging for weights and activations."""
    print("\n=== Logging Histograms ===")
    
    model = torchvision.models.resnet18(pretrained=False)
    
    # Log initial weights
    for name, param in model.named_parameters():
        writer.add_histogram(f'Initial_Weights/{name}', param, 0)
    
    # Simulate training - log weight updates
    for epoch in [0, 10, 25, 50]:
        for name, param in model.named_parameters():
            # Simulate weight updates
            updated_param = param + torch.randn_like(param) * 0.01 * (epoch + 1)
            writer.add_histogram(f'Weights/{name}', updated_param, epoch)
    
    # Log activations (simulated)
    for epoch in range(0, 50, 10):
        activation = torch.randn(64, 512) * (1 + epoch / 50)
        writer.add_histogram('Activations/fc_layer', activation, epoch)
    
    # Gradient histograms
    for epoch in range(0, 50, 10):
        gradients = torch.randn(1000) * np.exp(-epoch / 20)
        writer.add_histogram('Gradients/all_params', gradients, epoch)
    
    print("✓ Logged weight and activation histograms")


def log_embeddings_demo(writer):
    """Demonstrate embedding visualization (t-SNE, PCA)."""
    print("\n=== Logging Embeddings ===")
    
    # Generate sample embeddings (e.g., from a model's final layer)
    num_samples = 500
    embedding_dim = 128
    num_classes = 10
    
    # Create clustered embeddings
    embeddings = []
    labels = []
    label_names = [f'Class_{i}' for i in range(num_classes)]
    
    for class_id in range(num_classes):
        # Create cluster center
        center = torch.randn(embedding_dim) * 5
        # Generate points around center
        class_embeddings = center + torch.randn(num_samples // num_classes, embedding_dim) * 0.5
        embeddings.append(class_embeddings)
        labels.extend([label_names[class_id]] * (num_samples // num_classes))
    
    embeddings = torch.cat(embeddings, dim=0)
    
    # Log embeddings with labels
    writer.add_embedding(
        embeddings,
        metadata=labels,
        tag='Sample_Embeddings',
        global_step=0
    )
    
    # With images (create small sprite image)
    sprite_images = torch.rand(num_samples, 3, 32, 32)
    writer.add_embedding(
        embeddings,
        metadata=labels,
        label_img=sprite_images,
        tag='Embeddings_with_Images',
        global_step=0
    )
    
    print("✓ Logged embeddings for visualization")


def log_hyperparameters_demo(writer):
    """Demonstrate hyperparameter logging."""
    print("\n=== Logging Hyperparameters ===")
    
    # Simulate multiple experiments with different hyperparameters
    experiments = [
        {'lr': 1e-3, 'batch_size': 32, 'dropout': 0.2, 'optimizer': 'Adam'},
        {'lr': 1e-4, 'batch_size': 64, 'dropout': 0.3, 'optimizer': 'Adam'},
        {'lr': 5e-4, 'batch_size': 128, 'dropout': 0.1, 'optimizer': 'SGD'},
    ]
    
    for idx, hparams in enumerate(experiments):
        # Simulate final metrics
        val_acc = 0.85 + np.random.rand() * 0.1
        val_loss = 0.5 + np.random.rand() * 0.3
        
        # Log hyperparameters and metrics
        writer.add_hparams(
            hparams,
            {
                'hparam/accuracy': val_acc,
                'hparam/loss': val_loss,
                'hparam/f1_score': val_acc * 0.95
            },
            run_name=f'experiment_{idx}'
        )
    
    print("✓ Logged hyperparameter experiments")


def log_text_demo(writer):
    """Demonstrate text and markdown logging."""
    print("\n=== Logging Text ===")
    
    # Experiment configuration
    config_text = """
    # Experiment Configuration
    
    ## Model
    - Architecture: ResNet18
    - Input Size: 224x224
    - Output Classes: 10
    
    ## Training
    - Optimizer: Adam
    - Learning Rate: 1e-3
    - Batch Size: 64
    - Epochs: 50
    
    ## Data Augmentation
    - Random Crop
    - Random Horizontal Flip
    - Color Jitter
    """
    writer.add_text('Experiment/config', config_text, 0)
    
    # Training log messages
    for epoch in [0, 10, 25, 50]:
        log_msg = f"Epoch {epoch}: Training completed. Loss={np.random.rand():.4f}"
        writer.add_text('Training/logs', log_msg, epoch)
    
    # Model summary
    model_summary = """
    | Layer | Type | Output Shape | Params |
    |-------|------|--------------|--------|
    | conv1 | Conv2d | [32, 64, 64] | 896 |
    | conv2 | Conv2d | [64, 32, 32] | 18,496 |
    | fc1   | Linear | [128] | 131,200 |
    | fc2   | Linear | [10] | 1,290 |
    
    **Total Parameters**: 151,882
    """
    writer.add_text('Model/summary', model_summary, 0)
    
    print("✓ Logged text and markdown")


def log_pr_curve_demo(writer):
    """Demonstrate Precision-Recall curve logging."""
    print("\n=== Logging PR Curves ===")
    
    # Simulate predictions and labels
    num_samples = 1000
    num_classes = 5
    
    for class_id in range(num_classes):
        # Ground truth (binary for this class)
        labels = torch.randint(0, 2, (num_samples,))
        # Predictions (probabilities)
        predictions = torch.rand(num_samples)
        
        writer.add_pr_curve(
            f'PR_Curve/class_{class_id}',
            labels,
            predictions,
            global_step=0
        )
    
    print("✓ Logged PR curves")


def log_confusion_matrix_demo(writer):
    """Demonstrate confusion matrix visualization."""
    print("\n=== Logging Confusion Matrix ===")
    
    num_classes = 5
    class_names = [f'Class_{i}' for i in range(num_classes)]
    
    # Simulate predictions
    y_true = np.random.randint(0, num_classes, 500)
    y_pred = np.random.randint(0, num_classes, 500)
    
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names,
           yticklabels=class_names,
           title='Confusion Matrix',
           ylabel='True label',
           xlabel='Predicted label')
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")
    
    fig.tight_layout()
    writer.add_figure('Evaluation/confusion_matrix', fig, 0)
    plt.close(fig)
    
    print("✓ Logged confusion matrix")


def log_custom_plots_demo(writer):
    """Demonstrate custom matplotlib plots."""
    print("\n=== Logging Custom Plots ===")
    
    # Loss landscape
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.linspace(0, 10, 100)
    y1 = np.exp(-x/2) * np.sin(2*x) + 1
    y2 = np.exp(-x/3) * np.cos(x) + 1
    
    ax.plot(x, y1, label='Configuration A', linewidth=2)
    ax.plot(x, y2, label='Configuration B', linewidth=2)
    ax.set_xlabel('Epochs', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training Loss Comparison', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    writer.add_figure('Analysis/loss_comparison', fig, 0)
    plt.close(fig)
    
    # Learning rate schedule
    fig, ax = plt.subplots(figsize=(10, 6))
    epochs = np.arange(100)
    lr_cosine = 1e-3 * (0.5 + 0.5 * np.cos(np.pi * epochs / 100))
    lr_step = 1e-3 * (0.1 ** (epochs // 30))
    
    ax.plot(epochs, lr_cosine, label='Cosine Annealing', linewidth=2)
    ax.plot(epochs, lr_step, label='Step Decay', linewidth=2)
    ax.set_xlabel('Epochs', fontsize=12)
    ax.set_ylabel('Learning Rate', fontsize=12)
    ax.set_title('Learning Rate Schedules', fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    writer.add_figure('Analysis/lr_schedules', fig, 0)
    plt.close(fig)
    
    print("✓ Logged custom visualizations")


def main():
    """Run all TensorBoard logging demonstrations."""
    print("\n" + "="*60)
    print("TensorBoard Comprehensive Reference")
    print("="*60)
    
    # Create writer
    writer = SummaryWriter(log_dir='runs/comprehensive_demo')
    
    try:
        # Run all demos
        log_scalars_demo(writer, num_epochs=50)
        log_images_demo(writer)
        log_model_graph_demo(writer)
        log_histograms_demo(writer)
        log_embeddings_demo(writer)
        log_hyperparameters_demo(writer)
        log_text_demo(writer)
        log_pr_curve_demo(writer)
        log_confusion_matrix_demo(writer)
        log_custom_plots_demo(writer)
        
        print("\n" + "="*60)
        print("✅ All demonstrations completed successfully!")
        print("="*60)
        print("\nTo view results:")
        print("  tensorboard --logdir runs")
        print("  Then open: http://localhost:6006")
        print("\n")
        
    finally:
        writer.close()


if __name__ == '__main__':
    main()