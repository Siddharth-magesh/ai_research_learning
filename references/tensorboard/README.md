# TensorBoard Reference Guide

Complete guide for using TensorBoard with PyTorch for experiment tracking and visualization.

## 📋 Table of Contents

1. [Quick Start](#quick-start)
2. [Installation](#installation)
3. [Basic Usage](#basic-usage)
4. [Feature Reference](#feature-reference)
5. [Best Practices](#best-practices)
6. [Common Patterns](#common-patterns)
7. [Troubleshooting](#troubleshooting)

## 🚀 Quick Start

```bash
# Run the demo
python main.py

# Start TensorBoard
tensorboard --logdir runs

# Open browser
# Navigate to http://localhost:6006
```

## 📦 Installation

```bash
# Install TensorBoard with PyTorch
pip install torch torchvision tensorboard

# Additional dependencies for full demo
pip install matplotlib numpy scikit-learn
```

## 🎯 Basic Usage

### Minimal Example

```python
from torch.utils.tensorboard import SummaryWriter

# Create writer
writer = SummaryWriter('runs/experiment_1')

# Log scalars
for epoch in range(100):
    loss = compute_loss()
    writer.add_scalar('Loss/train', loss, epoch)

# Close writer
writer.close()
```

### Context Manager (Recommended)

```python
from torch.utils.tensorboard import SummaryWriter

with SummaryWriter('runs/experiment_1') as writer:
    for epoch in range(100):
        loss = compute_loss()
        writer.add_scalar('Loss/train', loss, epoch)
```

## 📊 Feature Reference

### 1. Scalars

Track single values over time (loss, accuracy, learning rate).

```python
# Single scalar
writer.add_scalar('Loss/train', loss_value, epoch)

# Multiple scalars in one chart
writer.add_scalars('Losses', {
    'train': train_loss,
    'validation': val_loss,
    'test': test_loss
}, epoch)

# Hierarchical organization
writer.add_scalar('Metrics/Accuracy/train', acc, epoch)
writer.add_scalar('Metrics/Accuracy/val', val_acc, epoch)
writer.add_scalar('Hyperparameters/learning_rate', lr, epoch)
```

**Use Cases:**
- Training/validation loss
- Metrics (accuracy, F1, etc.)
- Learning rate schedules
- Per-batch metrics

### 2. Images

Visualize images, feature maps, and reconstructions.

```python
# Single image (C, H, W) or (H, W)
img = torch.rand(3, 224, 224)
writer.add_image('Sample/image', img, epoch)

# Image grid (batch)
img_batch = torch.rand(16, 3, 64, 64)
img_grid = torchvision.utils.make_grid(img_batch, nrow=4)
writer.add_image('Sample/grid', img_grid, epoch)

# Matplotlib figure
fig, ax = plt.subplots()
ax.plot([1, 2, 3], [4, 5, 6])
writer.add_figure('Plot/custom', fig, epoch)
plt.close(fig)
```

**Use Cases:**
- Input data visualization
- Model outputs (segmentation, detection)
- Attention maps
- Feature visualizations
- GAN generated images

### 3. Model Graphs

Visualize model architecture.

```python
model = MyModel()
dummy_input = torch.randn(1, 3, 224, 224)
writer.add_graph(model, dummy_input)
```

**Tips:**
- Run once at the beginning
- Use representative input shape
- Model must be in eval mode or have dropout disabled for clean graphs

### 4. Histograms

Track distribution of tensors over time.

```python
# Weights
for name, param in model.named_parameters():
    writer.add_histogram(f'Weights/{name}', param, epoch)

# Gradients
for name, param in model.named_parameters():
    if param.grad is not None:
        writer.add_histogram(f'Gradients/{name}', param.grad, epoch)

# Activations
def hook_fn(module, input, output):
    writer.add_histogram('Activations/layer1', output, epoch)

model.layer1.register_forward_hook(hook_fn)
```

**Use Cases:**
- Weight distribution evolution
- Gradient flow monitoring
- Activation statistics
- Detecting vanishing/exploding gradients

### 5. Embeddings

Visualize high-dimensional data with t-SNE/PCA.

```python
# Extract embeddings from model
embeddings = model.encode(data)  # Shape: (N, embedding_dim)
labels = ['class_0', 'class_1', ...]  # Length: N

# Log embeddings
writer.add_embedding(
    embeddings,
    metadata=labels,
    tag='MNIST_Embeddings',
    global_step=epoch
)

# With thumbnail images
writer.add_embedding(
    embeddings,
    metadata=labels,
    label_img=images,  # Shape: (N, C, H, W)
    tag='Embeddings_with_Images'
)
```

**Use Cases:**
- Feature space visualization
- Cluster analysis
- Model representation quality
- Transfer learning evaluation

### 6. Hyperparameters

Track different hyperparameter configurations and their results.

```python
hparams = {
    'lr': 0.001,
    'batch_size': 64,
    'optimizer': 'Adam',
    'dropout': 0.3
}

metrics = {
    'hparam/accuracy': final_acc,
    'hparam/loss': final_loss,
    'hparam/f1': final_f1
}

writer.add_hparams(hparams, metrics)
```

**Use Cases:**
- Comparing multiple experiments
- Hyperparameter tuning visualization
- Ablation studies
- Model selection

### 7. Text

Log text, markdown, and experiment notes.

```python
# Simple text
writer.add_text('Notes', 'Started training with new augmentation', epoch)

# Markdown
markdown_text = """
# Experiment Results
- Best accuracy: 95.3%
- Training time: 2 hours
- **Key finding**: Dropout helps!
"""
writer.add_text('Results/summary', markdown_text, epoch)

# Structured data
config = f"""
Model: {model_name}
Optimizer: {optimizer_type}
Learning Rate: {lr}
"""
writer.add_text('Config', config, 0)
```

**Use Cases:**
- Experiment notes
- Configuration logging
- Qualitative observations
- Error messages/warnings

### 8. PR Curves

Precision-Recall curves for classification.

```python
# Binary classification
labels = torch.tensor([0, 1, 1, 0, 1])  # Ground truth
predictions = torch.tensor([0.1, 0.9, 0.8, 0.3, 0.7])  # Predicted probs

writer.add_pr_curve('PR_Curve', labels, predictions, epoch)

# Multi-class (one curve per class)
for class_id in range(num_classes):
    binary_labels = (labels == class_id).float()
    class_probs = predictions[:, class_id]
    writer.add_pr_curve(f'PR/class_{class_id}', binary_labels, class_probs, epoch)
```

**Use Cases:**
- Binary classification evaluation
- Threshold selection
- Multi-class performance
- Model comparison

### 9. Custom Plots

Any Matplotlib figure can be logged.

```python
# Loss curve
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(epochs, train_losses, label='Train')
ax.plot(epochs, val_losses, label='Validation')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.legend()
ax.grid(True)

writer.add_figure('Training/loss_curves', fig, final_epoch)
plt.close(fig)

# Confusion matrix
import seaborn as sns
fig, ax = plt.subplots()
sns.heatmap(confusion_matrix, annot=True, fmt='d', ax=ax)
writer.add_figure('Evaluation/confusion_matrix', fig, epoch)
plt.close(fig)
```

## 🎨 Best Practices

### 1. Organize with Hierarchies

```python
# Good - organized structure
writer.add_scalar('Train/Loss', loss, step)
writer.add_scalar('Train/Accuracy', acc, step)
writer.add_scalar('Validation/Loss', val_loss, step)
writer.add_scalar('Validation/Accuracy', val_acc, step)

# Bad - flat structure
writer.add_scalar('train_loss', loss, step)
writer.add_scalar('train_acc', acc, step)
```

### 2. Use Meaningful Names

```python
# Good
writer.add_scalar('Metrics/F1_Score/macro', f1_macro, epoch)
writer.add_scalar('Metrics/F1_Score/weighted', f1_weighted, epoch)

# Bad
writer.add_scalar('f1_1', f1_macro, epoch)
writer.add_scalar('f1_2', f1_weighted, epoch)
```

### 3. Log at Appropriate Frequency

```python
# Per epoch - for expensive metrics
for epoch in range(epochs):
    train(model)
    val_loss = validate(model)
    writer.add_scalar('Loss/validation', val_loss, epoch)

# Per batch - for training monitoring
for batch_idx, (data, target) in enumerate(train_loader):
    loss = train_step(data, target)
    global_step = epoch * len(train_loader) + batch_idx
    writer.add_scalar('Batch/loss', loss, global_step)
```

### 4. Flush Regularly

```python
# Ensure data is written to disk
for epoch in range(epochs):
    # ... training ...
    writer.add_scalar('Loss', loss, epoch)
    writer.flush()  # Write to disk immediately
```

### 5. Use Unique Run Names

```python
from datetime import datetime

# Timestamp-based runs
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter(f'runs/experiment_{timestamp}')

# Descriptive names
config_name = f'lr{lr}_bs{batch_size}_do{dropout}'
writer = SummaryWriter(f'runs/{config_name}')
```

## 🔧 Common Patterns

### Training Loop Integration

```python
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

def train_model(model, train_loader, val_loader, epochs=50):
    writer = SummaryWriter('runs/my_experiment')
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Log per-batch
            step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('Batch/loss', loss.item(), step)
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in val_loader:
                output = model(data)
                val_loss += criterion(output, target).item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        accuracy = correct / len(val_loader.dataset)
        
        # Log per-epoch
        writer.add_scalar('Epoch/train_loss', avg_train_loss, epoch)
        writer.add_scalar('Epoch/val_loss', avg_val_loss, epoch)
        writer.add_scalar('Epoch/accuracy', accuracy, epoch)
        
        # Log weights histogram every 10 epochs
        if epoch % 10 == 0:
            for name, param in model.named_parameters():
                writer.add_histogram(f'Weights/{name}', param, epoch)
        
        print(f'Epoch {epoch}: Train Loss={avg_train_loss:.4f}, '
              f'Val Loss={avg_val_loss:.4f}, Acc={accuracy:.4f}')
    
    writer.close()
```

### Hyperparameter Tuning

```python
def run_experiment(config):
    writer = SummaryWriter(f'runs/exp_lr{config["lr"]}_bs{config["bs"]}')
    
    # ... training ...
    
    final_metrics = {
        'hparam/accuracy': final_acc,
        'hparam/loss': final_loss
    }
    
    writer.add_hparams(config, final_metrics)
    writer.close()
    
    return final_acc

# Grid search
configs = [
    {'lr': 1e-3, 'bs': 32, 'dropout': 0.2},
    {'lr': 1e-4, 'bs': 64, 'dropout': 0.3},
    {'lr': 5e-4, 'bs': 128, 'dropout': 0.1},
]

for config in configs:
    run_experiment(config)
```

### Multi-Run Comparison

```python
# Run 1
writer1 = SummaryWriter('runs/baseline')
# ... train baseline model ...
writer1.close()

# Run 2
writer2 = SummaryWriter('runs/with_augmentation')
# ... train with augmentation ...
writer2.close()

# Run 3
writer3 = SummaryWriter('runs/larger_model')
# ... train larger model ...
writer3.close()

# View all runs together
# tensorboard --logdir runs
```

## 🐛 Troubleshooting

### Issue: TensorBoard shows "No dashboards are active"

**Solution:** Ensure you've logged data and refresh the browser.

```python
writer.add_scalar('test', 1, 0)
writer.flush()  # Force write to disk
writer.close()
```

### Issue: Images not displaying correctly

**Solution:** Check tensor shape and range.

```python
# Correct format: (C, H, W) with values in [0, 1]
img = torch.rand(3, 224, 224)  # ✓
writer.add_image('img', img, 0)

# Or (H, W) for grayscale
gray = torch.rand(224, 224)  # ✓
writer.add_image('gray', gray, 0)

# Normalize if needed
img = (img - img.min()) / (img.max() - img.min())
```

### Issue: Multiple runs overlapping confusingly

**Solution:** Use clear naming and separate directories.

```python
# Good
writer = SummaryWriter('runs/2025-01-15_ResNet_lr0.001')

# Better
from datetime import datetime
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter(f'runs/{timestamp}_ResNet_lr0.001')
```

### Issue: Graph not showing model architecture

**Solution:** Ensure model is in eval mode and input is correct.

```python
model.eval()  # Important!
dummy_input = torch.randn(1, 3, 224, 224)  # Correct shape
writer.add_graph(model, dummy_input)
```

### Issue: Embeddings not visualizing

**Solution:** Check data format and dimensions.

```python
# Embeddings: (N, D) where N = samples, D = dimensions
embeddings = torch.randn(1000, 128)  # ✓

# Labels: list of length N
labels = ['class_0'] * 500 + ['class_1'] * 500  # ✓

writer.add_embedding(embeddings, metadata=labels)
```

## 🔗 Advanced Tips

### Custom Scalars Layout

Create custom dashboard layouts:

```python
layout = {
    "Training": {
        "loss": ["Multiline", ["Loss/train", "Loss/val"]],
        "accuracy": ["Multiline", ["Accuracy/train", "Accuracy/val"]],
    },
}

writer.add_custom_scalars(layout)
```

### Video Logging

```python
# Video tensor: (N, T, C, H, W)
# N=batch, T=time, C=channels, H=height, W=width
video = torch.rand(4, 20, 3, 64, 64)  # 4 videos, 20 frames each
writer.add_video('Video/samples', video, fps=10)
```

### Audio Logging

```python
# Audio tensor: (num_channels, num_frames)
audio = torch.rand(1, 16000)  # 1 second at 16kHz
writer.add_audio('Audio/sample', audio, sample_rate=16000)
```

## 📚 Additional Resources

- [TensorBoard Tutorial](https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html)
- [PyTorch SummaryWriter Docs](https://pytorch.org/docs/stable/tensorboard.html)
- [TensorBoard GitHub](https://github.com/tensorflow/tensorboard)

## 🎯 Quick Reference

| Feature | Method | Use Case |
|---------|--------|----------|
| Scalars | `add_scalar()` | Loss, metrics, LR |
| Multiple scalars | `add_scalars()` | Compare train/val |
| Images | `add_image()` | Visualize inputs/outputs |
| Model graph | `add_graph()` | Architecture |
| Histograms | `add_histogram()` | Weights, gradients |
| Embeddings | `add_embedding()` | Feature visualization |
| Hyperparams | `add_hparams()` | Experiment comparison |
| Text | `add_text()` | Notes, configs |
| PR curves | `add_pr_curve()` | Classification eval |
| Figures | `add_figure()` | Custom plots |

---

**Happy Tracking! 📊✨**
