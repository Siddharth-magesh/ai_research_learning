# Vision Transformer Architecture

## Overview

Vision Transformer (ViT) applies the Transformer architecture to computer vision tasks by treating images as sequences of patches.

## Architecture Components

### Patch Embedding

Convert image to sequence of patch embeddings:

```
Image: (H, W, C)
Patch size: (P, P)
Number of patches: N = (H/P) × (W/P)
Patch embedding dimension: D
```

Operation:
1. Divide image into non-overlapping patches
2. Flatten each patch: (P, P, C) → (P²×C)
3. Linear projection: (P²×C) → D

Implementation:
```python
class PatchEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, in_channels, embed_dim):
        super().__init__()
        self.num_patches = (image_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_channels, 
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
    
    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x
```

### Class Token

Prependable learnable classification token:

```
CLS token: (1, D)
Patch embeddings: (N, D)
Concatenated: (N+1, D)
```

Used for classification:
```python
self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

def forward(self, x):
    B = x.shape[0]
    x = self.patch_embed(x)
    cls_tokens = self.cls_token.expand(B, -1, -1)
    x = torch.cat([cls_tokens, x], dim=1)
    return x
```

### Positional Embedding

Learned 1D positional embeddings:

```
Position embedding: (N+1, D)
```

Added to patch embeddings:
```python
self.pos_embed = nn.Parameter(
    torch.zeros(1, num_patches + 1, embed_dim)
)

def forward(self, x):
    x = x + self.pos_embed
    return x
```

### Transformer Encoder

Standard Transformer encoder:
- Multi-head self-attention
- Layer normalization (Pre-LN)
- MLP with GELU
- Residual connections

```python
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, int(embed_dim * mlp_ratio))
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
```

### Classification Head

Linear projection from CLS token:

```
CLS output: (B, D)
Classifier: (D, num_classes)
Logits: (B, num_classes)
```

Implementation:
```python
class ClassificationHead(nn.Module):
    def __init__(self, embed_dim, num_classes):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes)
    
    def forward(self, x):
        x = x[:, 0]
        x = self.norm(x)
        x = self.fc(x)
        return x
```

## Complete Architecture

```python
class VisionTransformer(nn.Module):
    def __init__(
        self,
        image_size=224,
        patch_size=16,
        in_channels=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0
    ):
        super().__init__()
        
        self.patch_embed = PatchEmbedding(
            image_size, patch_size, in_channels, embed_dim
        )
        num_patches = (image_size // patch_size) ** 2
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim)
        )
        
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio)
            for _ in range(depth)
        ])
        
        self.head = ClassificationHead(embed_dim, num_classes)
    
    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        
        for block in self.blocks:
            x = block(x)
        
        return self.head(x)
```

## Model Variants

### ViT-Base
```
Patch size: 16×16
Embed dim: 768
Depth: 12
Heads: 12
MLP ratio: 4
Parameters: 86M
```

### ViT-Large
```
Patch size: 16×16
Embed dim: 1024
Depth: 24
Heads: 16
MLP ratio: 4
Parameters: 307M
```

### ViT-Huge
```
Patch size: 14×14
Embed dim: 1280
Depth: 32
Heads: 16
MLP ratio: 4
Parameters: 632M
```

## Shape Flow

### Input Processing

```
Image:          (B, 3, 224, 224)
Patches:        (B, 196, 768)
With CLS:       (B, 197, 768)
With pos_embed: (B, 197, 768)
```

### Through Encoder

```
Input:  (B, 197, 768)
Block1: (B, 197, 768)
Block2: (B, 197, 768)
...
Block12: (B, 197, 768)
```

### Classification

```
Encoder output: (B, 197, 768)
CLS token:      (B, 768)
Logits:         (B, 1000)
```

## Training Strategy

### Pre-training

Large-scale supervised pre-training:
- Dataset: ImageNet-21k (14M images)
- Resolution: 224×224
- Augmentation: RandAugment
- Epochs: 300
- Optimizer: AdamW
- Learning rate: 1e-3
- Weight decay: 0.1
- Warmup: 10k steps

### Fine-tuning

Transfer to downstream tasks:
- Higher resolution: 384×384
- Lower learning rate: 1e-5
- Shorter training: 10-100 epochs
- Remove and reinitialize head

### Positional Encoding Interpolation

When changing resolution:
```python
def interpolate_pos_embed(pos_embed, new_size):
    N = pos_embed.shape[1] - 1
    pos_embed_token = pos_embed[:, 0]
    pos_embed_grid = pos_embed[:, 1:]
    
    pos_embed_grid = pos_embed_grid.reshape(1, H, W, D)
    pos_embed_grid = F.interpolate(
        pos_embed_grid.permute(0, 3, 1, 2),
        size=new_size,
        mode='bicubic'
    )
    pos_embed_grid = pos_embed_grid.permute(0, 2, 3, 1).reshape(1, -1, D)
    
    return torch.cat([pos_embed_token, pos_embed_grid], dim=1)
```

## Computational Complexity

### Self-Attention

```
Complexity: O(N² × D)
N = (H/P)² number of patches
D = embedding dimension
```

For 224×224 image with 16×16 patches:
```
N = 196
Complexity per layer: O(196² × 768) ≈ 30M operations
```

### MLP

```
Complexity: O(N × D²)
```

For standard config:
```
Complexity: O(196 × 768²) ≈ 115M operations
```

### Total

```
Total = Depth × (Attention + MLP)
     = 12 × (30M + 115M)
     = 1.74B operations per forward pass
```

## Advantages

1. **Scalability**: Performance improves with data and model size
2. **Transfer Learning**: Pre-training on large datasets transfers well
3. **Flexibility**: Can process variable resolution images
4. **Interpretability**: Attention maps show what model focuses on
5. **Long-range Dependencies**: Global receptive field from layer 1

## Limitations

1. **Data Hungry**: Requires large pre-training datasets
2. **Computational Cost**: Quadratic complexity in number of patches
3. **Inductive Bias**: Lacks CNN's translation equivariance
4. **Small Data**: Underperforms CNNs on small datasets

## Variants and Extensions

### DeiT (Data-efficient ViT)

- Distillation token
- Knowledge distillation from CNN
- Better performance with less data

### Swin Transformer

- Hierarchical architecture
- Shifted window attention
- Linear complexity in image size

### Pyramid Vision Transformer (PVT)

- Progressive shrinking pyramid
- Spatial reduction attention
- Multi-scale features

### ViT with Hybrid Architecture

- CNN stem for initial features
- Transformer on CNN feature maps
- Better inductive bias

## Applications

1. Image Classification
2. Object Detection (DETR)
3. Semantic Segmentation (SETR)
4. Video Understanding (VideoMAE)
5. Multi-modal Learning (CLIP)

## Implementation Notes

### Memory Optimization

```python
import torch.utils.checkpoint as checkpoint

def forward(self, x):
    for block in self.blocks:
        x = checkpoint.checkpoint(block, x)
    return x
```

### Mixed Precision

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    logits = model(images)
    loss = criterion(logits, labels)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### Efficient Inference

```python
@torch.jit.script
def forward_jit(x):
    return model(x)

model_scripted = torch.jit.script(model)
```
