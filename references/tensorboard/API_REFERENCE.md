# TensorBoard API Quick Reference

Fast lookup for all TensorBoard logging methods.

## Core Writer Methods

### Initialization

```python
from torch.utils.tensorboard import SummaryWriter

# Basic
writer = SummaryWriter()  # Default: runs/{current_time}

# Custom directory
writer = SummaryWriter('runs/experiment_1')
writer = SummaryWriter(log_dir='path/to/logs')

# With comment
writer = SummaryWriter(comment='_learning_rate_1e-3')

# Close when done
writer.close()

# Or use context manager
with SummaryWriter('runs/exp') as writer:
    # ... logging ...
    pass  # Automatically closes
```

## Scalar Data

### add_scalar()
```python
writer.add_scalar(tag, scalar_value, global_step=None, walltime=None)
```

**Parameters:**
- `tag` (str): Identifier (e.g., 'Loss/train')
- `scalar_value` (float): Value to log
- `global_step` (int): X-axis value (epoch/iteration)
- `walltime` (float): Optional timestamp

**Example:**
```python
writer.add_scalar('Loss/train', 0.5, epoch)
writer.add_scalar('Metrics/accuracy', 0.95, step)
```

### add_scalars()
```python
writer.add_scalars(main_tag, tag_scalar_dict, global_step=None, walltime=None)
```

**Parameters:**
- `main_tag` (str): Parent tag
- `tag_scalar_dict` (dict): {sub_tag: value}

**Example:**
```python
writer.add_scalars('Losses', {
    'train': 0.5,
    'validation': 0.6,
    'test': 0.55
}, epoch)
```

## Images

### add_image()
```python
writer.add_image(tag, img_tensor, global_step=None, walltime=None, dataformats='CHW')
```

**Parameters:**
- `tag` (str): Identifier
- `img_tensor` (torch.Tensor): Shape (C,H,W) or (H,W)
- `dataformats` (str): 'CHW', 'HWC', 'HW'

**Example:**
```python
img = torch.rand(3, 224, 224)  # RGB image
writer.add_image('samples/image', img, 0)

# With normalization
img = (img - img.min()) / (img.max() - img.min())
writer.add_image('normalized', img, 0)

# Grayscale
gray = torch.rand(224, 224)
writer.add_image('gray', gray, 0)
```

### add_images()
```python
writer.add_images(tag, img_tensor, global_step=None, walltime=None, dataformats='NCHW')
```

**Parameters:**
- `img_tensor` (torch.Tensor): Shape (N,C,H,W)

**Example:**
```python
batch = torch.rand(16, 3, 64, 64)
writer.add_images('batch', batch, 0)
```

### add_figure()
```python
writer.add_figure(tag, figure, global_step=None, close=True, walltime=None)
```

**Parameters:**
- `figure` (matplotlib.figure.Figure): Matplotlib figure

**Example:**
```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot([1, 2, 3], [4, 5, 6])
writer.add_figure('custom_plot', fig, 0)
plt.close(fig)
```

## Model Architecture

### add_graph()
```python
writer.add_graph(model, input_to_model, verbose=False)
```

**Parameters:**
- `model` (torch.nn.Module): Model to visualize
- `input_to_model` (torch.Tensor): Example input

**Example:**
```python
model = MyModel()
dummy_input = torch.randn(1, 3, 224, 224)
model.eval()  # Important!
writer.add_graph(model, dummy_input)
```

## Distributions

### add_histogram()
```python
writer.add_histogram(tag, values, global_step=None, bins='tensorflow', walltime=None)
```

**Parameters:**
- `tag` (str): Identifier
- `values` (torch.Tensor or numpy.array): Data to histogram
- `bins` (str or int): Binning method

**Example:**
```python
# Weights
for name, param in model.named_parameters():
    writer.add_histogram(f'weights/{name}', param, epoch)

# Gradients
for name, param in model.named_parameters():
    if param.grad is not None:
        writer.add_histogram(f'grads/{name}', param.grad, epoch)

# Activations
activations = torch.randn(64, 512)
writer.add_histogram('activations/fc1', activations, epoch)
```

## Embeddings

### add_embedding()
```python
writer.add_embedding(
    mat,
    metadata=None,
    label_img=None,
    global_step=None,
    tag='default',
    metadata_header=None
)
```

**Parameters:**
- `mat` (torch.Tensor): Shape (N, D) - N samples, D dimensions
- `metadata` (list): Labels for each sample (length N)
- `label_img` (torch.Tensor): Shape (N, C, H, W) - thumbnail images
- `tag` (str): Identifier

**Example:**
```python
# Basic
embeddings = torch.randn(1000, 128)
labels = ['class_0'] * 500 + ['class_1'] * 500
writer.add_embedding(embeddings, metadata=labels, tag='features')

# With images
images = torch.rand(1000, 3, 32, 32)
writer.add_embedding(
    embeddings,
    metadata=labels,
    label_img=images,
    tag='embeddings_with_images'
)
```

## Hyperparameters

### add_hparams()
```python
writer.add_hparams(
    hparam_dict,
    metric_dict,
    hparam_domain_discrete=None,
    run_name=None
)
```

**Parameters:**
- `hparam_dict` (dict): Hyperparameters {name: value}
- `metric_dict` (dict): Metrics {name: value}
- `run_name` (str): Experiment identifier

**Example:**
```python
hparams = {
    'lr': 0.001,
    'batch_size': 64,
    'optimizer': 'Adam',
    'dropout': 0.3
}

metrics = {
    'hparam/accuracy': 0.95,
    'hparam/loss': 0.25,
    'hparam/f1': 0.93
}

writer.add_hparams(hparams, metrics, run_name='exp_1')
```

## Text & Markdown

### add_text()
```python
writer.add_text(tag, text_string, global_step=None, walltime=None)
```

**Parameters:**
- `tag` (str): Identifier
- `text_string` (str): Text content (supports Markdown)

**Example:**
```python
# Simple text
writer.add_text('notes', 'Experiment started', 0)

# Markdown
markdown = """
# Results
- Accuracy: 95%
- **Best epoch**: 42
"""
writer.add_text('results', markdown, 50)

# Configuration
config = f"LR: {lr}, BS: {batch_size}"
writer.add_text('config', config, 0)
```

## Precision-Recall Curves

### add_pr_curve()
```python
writer.add_pr_curve(
    tag,
    labels,
    predictions,
    global_step=None,
    num_thresholds=127,
    weights=None,
    walltime=None
)
```

**Parameters:**
- `labels` (torch.Tensor): Binary labels (0 or 1)
- `predictions` (torch.Tensor): Predicted probabilities [0, 1]

**Example:**
```python
# Binary classification
labels = torch.tensor([0, 1, 1, 0, 1])
predictions = torch.tensor([0.1, 0.9, 0.8, 0.3, 0.7])
writer.add_pr_curve('pr_curve', labels, predictions, 0)

# Multi-class (one per class)
for class_id in range(num_classes):
    binary_labels = (all_labels == class_id).float()
    class_probs = all_predictions[:, class_id]
    writer.add_pr_curve(f'pr/class_{class_id}', binary_labels, class_probs, epoch)
```

## Media

### add_video()
```python
writer.add_video(tag, vid_tensor, global_step=None, fps=4, walltime=None)
```

**Parameters:**
- `vid_tensor` (torch.Tensor): Shape (N, T, C, H, W)
  - N: batch size
  - T: number of frames
  - C: channels
  - H, W: height, width
- `fps` (int): Frames per second

**Example:**
```python
# 4 videos, 20 frames each, RGB, 64x64
video = torch.rand(4, 20, 3, 64, 64)
writer.add_video('samples', video, fps=10)
```

### add_audio()
```python
writer.add_audio(tag, snd_tensor, global_step=None, sample_rate=44100, walltime=None)
```

**Parameters:**
- `snd_tensor` (torch.Tensor): Shape (1, L) - mono or (2, L) - stereo
- `sample_rate` (int): Sampling rate

**Example:**
```python
# 1 second of mono audio at 16kHz
audio = torch.rand(1, 16000)
writer.add_audio('audio', audio, sample_rate=16000)
```

## Mesh & 3D

### add_mesh()
```python
writer.add_mesh(
    tag,
    vertices,
    colors=None,
    faces=None,
    config_dict=None,
    global_step=None,
    walltime=None
)
```

**Parameters:**
- `vertices` (torch.Tensor): Shape (N, 3) - vertex positions
- `colors` (torch.Tensor): Shape (N, 3) - RGB colors
- `faces` (torch.Tensor): Shape (M, 3) - triangle indices

**Example:**
```python
vertices = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
colors = torch.tensor([[255, 0, 0], [0, 255, 0], [0, 0, 255]])
faces = torch.tensor([[0, 1, 2]])
writer.add_mesh('mesh', vertices, colors, faces)
```

## Utilities

### flush()
```python
writer.flush()
```

Force write all pending data to disk.

**Example:**
```python
for epoch in range(100):
    # ... training ...
    writer.add_scalar('loss', loss, epoch)
    writer.flush()  # Ensure immediate write
```

### close()
```python
writer.close()
```

Close the writer and flush remaining data.

**Example:**
```python
writer = SummaryWriter()
# ... logging ...
writer.close()  # Always close when done
```

## Advanced

### add_custom_scalars()
```python
writer.add_custom_scalars(layout)
```

Define custom dashboard layouts.

**Example:**
```python
layout = {
    "Training": {
        "loss": ["Multiline", ["Loss/train", "Loss/val"]],
        "accuracy": ["Multiline", ["Acc/train", "Acc/val"]],
    },
    "Metrics": {
        "f1": ["Margin", ["F1/train", "F1/val"]],
    }
}
writer.add_custom_scalars(layout)
```

## Common Naming Conventions

### Hierarchical Organization

```python
# Recommended structure
'Train/Loss'
'Train/Accuracy'
'Validation/Loss'
'Validation/Accuracy'
'Hyperparameters/learning_rate'
'Hyperparameters/weight_decay'
'Weights/conv1.weight'
'Weights/fc.bias'
'Gradients/conv1.weight'
'Activations/layer1'
```

### Grouping with add_scalars()

```python
# All related metrics in one call
writer.add_scalars('Performance', {
    'train_acc': train_acc,
    'val_acc': val_acc,
    'test_acc': test_acc
}, epoch)
```

## Data Format Specifications

### Image Tensors

```python
# Formats
(C, H, W)  # Default - Channels, Height, Width
(H, W, C)  # Alternative - specify dataformats='HWC'
(H, W)     # Grayscale
(N, C, H, W)  # Batch - for add_images()

# Value range
[0, 1]  # Preferred - will be displayed correctly
[0, 255]  # Will be normalized automatically
```

### Histogram Data

```python
# Any 1D or flattened tensor
weights = model.layer.weight.data  # Any shape
writer.add_histogram('weights', weights.flatten(), epoch)
```

### Embedding Matrices

```python
# Shape: (num_samples, embedding_dim)
embeddings = torch.randn(1000, 128)  # 1000 samples, 128-D
metadata = ['label'] * 1000  # List of length 1000
```

## Performance Tips

1. **Batch logging**: Don't log every single step
```python
if batch_idx % 100 == 0:
    writer.add_scalar('batch_loss', loss, step)
```

2. **Conditional histogram logging**:
```python
if epoch % 10 == 0:  # Only every 10 epochs
    for name, param in model.named_parameters():
        writer.add_histogram(name, param, epoch)
```

3. **Use flush() for critical data**:
```python
writer.add_scalar('final_accuracy', final_acc, epochs)
writer.flush()  # Ensure it's saved
```

4. **Close writers properly**:
```python
try:
    # ... training ...
    writer.add_scalar('metric', value, step)
finally:
    writer.close()  # Always close
```

## Typical Training Integration

```python
def train_epoch(model, loader, optimizer, criterion, writer, epoch):
    model.train()
    total_loss = 0
    
    for batch_idx, (data, target) in enumerate(loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Log per-batch
        step = epoch * len(loader) + batch_idx
        if batch_idx % 10 == 0:
            writer.add_scalar('Train/batch_loss', loss.item(), step)
    
    # Log per-epoch
    avg_loss = total_loss / len(loader)
    writer.add_scalar('Train/epoch_loss', avg_loss, epoch)
    
    # Log weights (occasionally)
    if epoch % 5 == 0:
        for name, param in model.named_parameters():
            writer.add_histogram(f'Weights/{name}', param, epoch)
            if param.grad is not None:
                writer.add_histogram(f'Grads/{name}', param.grad, epoch)
    
    return avg_loss
```

---

**Complete API Documentation**: [PyTorch TensorBoard Docs](https://pytorch.org/docs/stable/tensorboard.html)
