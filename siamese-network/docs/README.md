# Siamese Network for Signature Verification

A complete implementation of Siamese Network for signature verification using triplet loss, with TensorBoard logging, Optuna hyperparameter optimization, and comprehensive evaluation.

## Quick Start

### 1. Optimize Hyperparameters (Optional)
```bash
cd /home/siddharth/ai_research_learning
uv run python siamese-network/src/optimize.py
```

This runs Optuna to find the best hyperparameters and saves results to `optuna_results.csv` and `best_hyperparameters.txt`.

### 2. Train the Model
```bash
uv run python siamese-network/src/main.py
```

Trains the model with TensorBoard logging. Monitor training in real-time:
```bash
tensorboard --logdir=runs
```

### 3. Visualize Results
```bash
uv run python siamese-network/src/visual.py
```

Generates plots for training curves, confusion matrix, ROC curve, and distance distributions.

## Project Structure

```
siamese-network/
├── src/
│   ├── config.py                   # Configuration settings
│   ├── optimize.py                 # Optuna hyperparameter optimization
│   ├── main.py                     # Training script
│   ├── visual.py                   # Visualization script
│   ├── train.py                    # Trainer with TensorBoard
│   ├── evaluate.py                 # Evaluation metrics
│   ├── data_loader.py              # Data loading utilities
│   ├── siamese_network.py          # Siamese Network model
│   └── modules/
│       ├── embedding_network.py    # CNN embedding network
│       ├── signature_triplet_dataset.py
│       └── transformation.py
├── checkpoints/                    # Saved models
├── visualizations/                 # Generated plots
├── runs/                          # TensorBoard logs
└── README.md
```

## Model Architecture

## Model Architecture

### Embedding Network (CNN)

4-block convolutional network with batch normalization:

```
Input (3×H×W) → Conv1 (32) → Conv2 (64) → Conv3 (128) → Conv4 (256)
→ FC (512) → FC (256) → FC (embedding_dim) → L2 Normalize
```

Each conv block: Conv2d → BatchNorm → ReLU → MaxPool → Dropout

### Siamese Network

Shared-weight architecture for triplet learning:
```
Anchor ─┐
        ├─→ Embedding Net → Triplet Loss
Positive│
        │
Negative┘
```

## Workflow

### Step 1: Hyperparameter Optimization (Optional but Recommended)

Run Optuna to find optimal hyperparameters:
```bash
uv run python siamese-network/src/optimize.py
```

**Output:**
- `optuna_results.csv` - All trial results
- `best_hyperparameters.txt` - Best parameters found

**What it optimizes:**
- Image size (64, 96, 128)
- Embedding dimension (32, 64, 128)
- Batch size (4, 8, 16)
- Learning rate (1e-4 to 1e-2)
- Weight decay (1e-5 to 1e-3)
- Triplet margin (0.5 to 2.0)
- Scheduler gamma (0.3 to 0.7)

**Update config.py** with the best parameters before training.

### Step 2: Training

Train the model with current configuration:
```bash
uv run python siamese-network/src/main.py
```

**Features:**
- Automatic data download from Kaggle
- TensorBoard logging (runs/siamese_network)
- Model summary with torchinfo
- Progress bars with tqdm
- Best model checkpointing
- Comprehensive evaluation

**Monitor training:**
```bash
tensorboard --logdir=runs
```
Then open http://localhost:6006 in your browser.

**TensorBoard metrics:**
- Training and validation loss
- Validation accuracy (overall, genuine, fake)
- Distance statistics (genuine vs fake)
- Learning rate schedule

### Step 3: Visualization

Generate evaluation plots:
```bash
uv run python siamese-network/src/visual.py
```

**Generated plots:**
- `training_curves.png` - Loss, accuracy, learning rate
- `distance_distributions.png` - Histogram and box plot
- `confusion_matrix.png` - Classification matrix
- `roc_curve.png` - ROC curve with AUC

## Configuration

Edit `src/config.py` to adjust hyperparameters:

| Parameter | Lightweight (Laptop) | Optimal (GPU) |
|-----------|---------------------|---------------|
| image_size | (128, 128) | (224, 224) |
| embedding_dim | 64 | 128-256 |
| batch_size | 8 | 32-64 |
| num_epochs | 3 | 20-50 |
| triplets_per_user | 20 | 100-200 |
| learning_rate | 1e-3 | 1e-3 to 1e-4 |

## Model Summary

View model architecture with torchinfo (shown during training):
- Layer-by-layer structure
- Input/output sizes
- Parameter counts
- Trainable parameters

## Dependencies

```
torch>=2.5.0
torchvision>=0.20.0
tensorboard>=2.18.0
optuna>=4.1.0
torchinfo>=1.8.0
scikit-learn>=1.6.1
matplotlib>=3.10.7
tqdm>=4.67.1
kagglehub>=0.3.13
```

Install with:
```bash
cd /home/siddharth/ai_research_learning
uv sync
```

## Output Files

### After Optimization
- `optuna_results.csv` - All hyperparameter trials
- `best_hyperparameters.txt` - Best configuration

### After Training
- `checkpoints/best_model.pth` - Best model weights
- `runs/siamese_network/` - TensorBoard logs

### After Visualization
- `visualizations/training_curves.png`
- `visualizations/distance_distributions.png`
- `visualizations/confusion_matrix.png`
- `visualizations/roc_curve.png`

## License

See LICENSE file in project root.
