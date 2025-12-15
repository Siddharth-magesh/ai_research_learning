# Quick Reference Guide - Siamese Network

## Quick Start Commands

### Train the Model
```bash
cd /home/siddharth/ai_research_learning
uv run python siamese-network/src/main.py
```

### Quick Test (Minimal Config)
```bash
cd /home/siddharth/ai_research_learning
uv run python siamese-network/src/test_pipeline.py
```

## Configuration Cheat Sheet

File: `siamese-network/src/config.py`

### Key Parameters to Adjust

| Parameter | Lightweight (Laptop) | Optimal (Powerful) |
|-----------|---------------------|-------------------|
| `image_size` | (128, 128) | (224, 224) |
| `embedding_dim` | 64 | 128-256 |
| `batch_size` | 8 | 32-64 |
| `num_epochs` | 3 | 20-50 |
| `triplets_per_user` | 20 | 100-200 |
| `num_workers` | 2 | 4-8 |

### Quick Config Change

```python
# Edit config.py
config.num_epochs = 10
config.batch_size = 16
config.embedding_dim = 128
```

## File Locations

### Source Code
- Main Pipeline: `siamese-network/src/main.py`
- Configuration: `siamese-network/src/config.py`
- Training: `siamese-network/src/train.py`
- Evaluation: `siamese-network/src/evaluate.py`

### Output
- Checkpoints: `siamese-network/checkpoints/`
- Visualizations: `siamese-network/visualizations/`
- Best Model: `siamese-network/checkpoints/best_model.pth`

### Documentation
- README: `siamese-network/README.md`
- Summary: `siamese-network/IMPLEMENTATION_SUMMARY.md`
- This Guide: `siamese-network/QUICK_REFERENCE.md`

## Expected Output

### Training Output
```
[Step 1] Data downloaded from Kaggle
[Step 2] Train: 800 samples, Val: 220 samples
[Step 3] Model: 8.9M parameters
[Step 4] Loss, optimizer, scheduler ready
[Step 5] Training with progress bars
[Step 6] Evaluation metrics computed
[Step 7] Visualizations saved
```

### Checkpoints Created
- `best_model.pth` - Best validation accuracy
- `checkpoint_epoch_X.pth` - Per-epoch saves

### Visualizations Created
- `triplet_samples.png`
- `training_history.png`
- `distance_distributions.png`
- `confusion_matrix.png`
- `roc_curve.png`
- `threshold_analysis.png`
- `predictions.png`

## Metrics Explained

### During Training
- **Train Loss**: Triplet margin loss on training data
- **Val Loss**: Triplet margin loss on validation data
- **Val Accuracy**: Percentage of correct classifications
- **Mean Distance (Genuine)**: Average distance for same-person signatures
- **Mean Distance (Fake)**: Average distance for different-person signatures

### After Training
- **Accuracy**: Overall classification accuracy
- **Precision**: Correct positive predictions / Total positive predictions
- **Recall**: Correct positive predictions / Total actual positives
- **F1-Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area under ROC curve (higher is better)

## Common Issues & Solutions

### Out of Memory
**Solution**: Reduce `batch_size` or `image_size` in config.py

### Slow Training
**Solution**: Increase `num_workers` or use GPU

### Poor Accuracy
**Solution**: 
- Increase `num_epochs`
- Increase `triplets_per_user`
- Try different `learning_rate`

## Using Trained Model

```python
import torch
from siamese_network import SiameseNetwork
from modules.embedding_network import SimpleEmbeddingNetwork

# Load model
checkpoint = torch.load('checkpoints/best_model.pth')
embedding_net = SimpleEmbeddingNetwork(embedding_dim=64)
model = SiameseNetwork(embedding_network=embedding_net)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Make predictions
with torch.no_grad():
    embedding1 = model.get_embedding(image1)
    embedding2 = model.get_embedding(image2)
    distance = model.compute_distance(embedding1, embedding2)
    is_same = distance < threshold
```

## Pipeline Architecture

```
Data (Kaggle) → DataLoader → Siamese Network → Triplet Loss
                                    ↓
                           Embeddings (64-256 dim)
                                    ↓
                           Distance Computation
                                    ↓
                          Classification (Same/Different)
```

## Model Architecture

```
Input Image (3×H×W)
    ↓
Conv Block 1 (3→32)
    ↓
Conv Block 2 (32→64)
    ↓
Conv Block 3 (64→128)
    ↓
Conv Block 4 (128→256)
    ↓
Flatten
    ↓
FC Layer (→512)
    ↓
FC Layer (→256)
    ↓
FC Layer (→embedding_dim)
    ↓
L2 Normalization
    ↓
Embedding Vector
```

## Monitoring Training

### Good Signs
- Train loss decreasing
- Val loss stable or decreasing
- Val accuracy increasing
- Separation between genuine and fake distances increasing

### Warning Signs
- Val loss increasing while train loss decreasing (overfitting)
- Very low validation accuracy (<50%)
- No separation in distance distributions

## Performance Expectations

### Lightweight Config (Laptop)
- Training Time: ~5-10 minutes per epoch
- Expected Accuracy: 60-75%
- Memory Usage: ~2-4 GB

### Optimal Config (Powerful Machine)
- Training Time: ~10-30 minutes per epoch
- Expected Accuracy: 80-95%
- Memory Usage: 4-8 GB

## Directory Structure

```
siamese-network/
├── src/                    # Source code
├── checkpoints/            # Saved models
├── visualizations/         # Generated plots
├── README.md              # Full documentation
├── IMPLEMENTATION_SUMMARY.md  # What was built
└── QUICK_REFERENCE.md     # This file
```

## Important Notes

1. **Current Config**: Set for lightweight laptop training
2. **Dataset**: Auto-downloads from Kaggle (requires internet)
3. **Device**: Auto-detects GPU, falls back to CPU
4. **Checkpoints**: Only best model saved by default
5. **Visualizations**: Generated at end of training

## Contact & Support

For issues or questions:
1. Check README.md for detailed documentation
2. Check IMPLEMENTATION_SUMMARY.md for component details
3. Review config.py for parameter descriptions

## Version Info

- Python: 3.13+
- PyTorch: 2.5.0+
- Main Dependencies: torch, torchvision, scikit-learn, matplotlib, tqdm

Last Updated: December 2025
