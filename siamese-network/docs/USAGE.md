# Siamese Network - Usage Guide

## Three-Step Workflow

### Step 1: Find Best Hyperparameters
```bash
cd /home/siddharth/ai_research_learning
uv run python siamese-network/src/optimize.py
```

**What it does:**
- Runs 20 Optuna trials to find best hyperparameters
- Tests different combinations of image size, embedding dim, batch size, learning rate, etc.
- Saves results to `optuna_results.csv`
- Saves best config to `best_hyperparameters.txt`

**After running:**
- Open `best_hyperparameters.txt`
- Update values in `siamese-network/src/config.py`

---

### Step 2: Train the Model
```bash
uv run python siamese-network/src/main.py
```

**What it does:**
- Downloads signature dataset from Kaggle
- Shows model architecture with torchinfo
- Trains with TensorBoard logging
- Saves best model to `checkpoints/best_model.pth`
- Evaluates and prints metrics

**Monitor training in real-time:**
```bash
tensorboard --logdir=runs
```
Open http://localhost:6006 in your browser

**TensorBoard shows:**
- Loss curves (train & validation)
- Accuracy curves
- Distance metrics
- Learning rate schedule

---

### Step 3: Generate Visualizations
```bash
uv run python siamese-network/src/visual.py
```

**What it does:**
- Loads trained model from checkpoint
- Generates 4 plots:
  1. Training curves (loss, accuracy, learning rate)
  2. Distance distributions (genuine vs forged)
  3. Confusion matrix
  4. ROC curve
- Saves plots to `visualizations/` folder

---

## Quick Configuration

Edit `siamese-network/src/config.py`:

```python
# For laptop (fast testing)
image_size = (128, 128)
embedding_dim = 64
batch_size = 8
num_epochs = 3
triplets_per_user = 20

# For powerful machine (best results)
image_size = (224, 224)  # Better quality
embedding_dim = 128      # More capacity
batch_size = 32          # Faster training
num_epochs = 20          # Better convergence
triplets_per_user = 100  # More data
```

---

## File Outputs

### After optimize.py
- `optuna_results.csv` - All trial results
- `best_hyperparameters.txt` - Best config

### After main.py
- `checkpoints/best_model.pth` - Trained model
- `runs/siamese_network/` - TensorBoard logs

### After visual.py
- `visualizations/training_curves.png`
- `visualizations/distance_distributions.png`
- `visualizations/confusion_matrix.png`
- `visualizations/roc_curve.png`

---

## Tips

**First time:**
1. Run optimize.py (takes ~30-60 min)
2. Update config.py with best params
3. Run main.py
4. Run visual.py

**For experiments:**
1. Edit config.py directly
2. Run main.py
3. Check TensorBoard
4. Run visual.py

**GPU vs CPU:**
- Config automatically detects GPU
- CPU: Set small batch_size (4-8)
- GPU: Set larger batch_size (32-64)

---

## Common Commands

```bash
# Navigate to project
cd /home/siddharth/ai_research_learning

# Install dependencies
uv sync

# Optimize
uv run python siamese-network/src/optimize.py

# Train
uv run python siamese-network/src/main.py

# Visualize
uv run python siamese-network/src/visual.py

# TensorBoard
tensorboard --logdir=runs
```
