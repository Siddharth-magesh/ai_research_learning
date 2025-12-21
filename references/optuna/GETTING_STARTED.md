# Getting Started Guide

## Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (optional, but recommended)

## Installation

### 1. Clone or navigate to the project

```bash
cd d:\ai_research_learning\references\optuna
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install torch torchvision optuna plotly kaleido pandas numpy
```

### 3. Verify installation

```bash
python -c "import optuna; print(f'Optuna version: {optuna.__version__}')"
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Quick Start

### Option 1: Run Hyperparameter Optimization

This will search for the best hyperparameters:

```bash
cd src
python optimize.py
```

**What happens:**
- Runs 30 optimization trials
- Tests different combinations of:
  - Number of layers (2-4)
  - Channels (16-64)
  - Learning rate (1e-4 to 1e-1)
  - Optimizer (Adam or SGD)
  - And more...
- Saves results to `../results/`
- Generates visualization plots

**Expected output:**
```
Trial 0 finished with value: 0.6234 and parameters: {...}
Trial 1 finished with value: 0.6523 and parameters: {...}
...
Trial 29 finished with value: 0.7123 and parameters: {...}

Best accuracy: 0.7234
Best hyperparameters:
  n_layers: 3
  channels: 48
  dropout: 0.03
  activation: relu
  lr: 0.00123
  ...

Results saved to: ../results
```

**Time estimate:** 1-3 hours depending on your hardware

### Option 2: Train with Best Parameters

If you already ran optimization or want to use default best parameters:

```bash
cd src
python main.py
```

**What happens:**
- Uses predefined best hyperparameters
- Trains for specified epochs
- Shows training progress
- Saves final model

**Expected output:**
```
Training on cuda
Model parameters: 1,234,567

Epoch 1/15: Accuracy = 0.5234
Epoch 2/15: Accuracy = 0.6123
...
Epoch 15/15: Accuracy = 0.7234

Model saved to: ../results/best_model.pth
```

**Time estimate:** 30-60 minutes

## Understanding the Results

After running `optimize.py`, check the `results/` folder:

### 1. best_results.json

```json
{
    "best_accuracy": 0.7234,
    "best_hyperparameters": {
        "n_layers": 3,
        "channels": 48,
        "dropout": 0.03,
        "activation": "relu",
        "lr": 0.00123,
        "optimizer": "Adam",
        "batch_size": 64,
        "epochs": 15
    },
    "n_trials": 30
}
```

### 2. Visualization Images

- **optimization_history.png**: Shows how accuracy improved over trials
- **param_importances.png**: Shows which hyperparameters matter most
- **parallel_coordinate.png**: Shows relationships between parameters

### 3. Database (cnn_optuna.db)

SQLite database containing all trial information. Can be loaded for further analysis:

```python
import optuna
study = optuna.load_study(
    study_name='cifar10_cnn',
    storage='sqlite:///results/cnn_optuna.db'
)
print(study.best_params)
```

## Learning Resources

### Read in Order:

1. **[Quick Reference](docs/00_QUICK_REFERENCE.md)** - 10 min read
   - Quick lookup for common operations
   - Start here if you're in a hurry

2. **[Optuna Overview](docs/01_OPTUNA_OVERVIEW.md)** - 20 min read
   - What is Optuna and how it works
   - Core concepts explained
   - When to use it

3. **[Hyperparameter Tuning](docs/02_HYPERPARAMETER_TUNING.md)** - 30 min read
   - How to suggest different types of hyperparameters
   - Search space design
   - Optimization strategies

4. **[Methods Reference](docs/03_METHODS_REFERENCE.md)** - 30 min read
   - Complete API reference
   - All methods explained with examples
   - Quick decision tables

5. **[Pruning and Samplers](docs/04_PRUNING_SAMPLERS.md)** - 40 min read
   - Deep dive into pruning algorithms
   - Deep dive into sampling algorithms
   - Choosing the right combination

6. **[Visualization Guide](docs/05_VISUALIZATION.md)** - 30 min read
   - All visualization types
   - How to interpret plots
   - Saving and customization

**Total reading time:** ~2.5 hours

### Quick Reference While Coding:

Keep [00_QUICK_REFERENCE.md](docs/00_QUICK_REFERENCE.md) open - it has:
- Code snippets for common operations
- Cheat sheet table
- Complete working examples

## Common Workflows

### 1. First Time User

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Read Quick Reference (10 min)
# Open docs/00_QUICK_REFERENCE.md

# 3. Run a quick optimization (modify n_trials to 5 for testing)
# Edit src/optimize.py: change n_trials=30 to n_trials=5
cd src
python optimize.py

# 4. Check results
# Look at results/best_results.json
# View results/*.png images

# 5. Read full documentation
# Go through docs/01-05 in order
```

### 2. Researcher/Student

```bash
# 1. Read all documentation thoroughly
# Start with docs/01_OPTUNA_OVERVIEW.md

# 2. Understand the code
# Read through src/*.py files
# Understand each component

# 3. Modify search space
# Edit src/optimize.py
# Add/remove hyperparameters

# 4. Run optimization
python src/optimize.py

# 5. Analyze results
# Check all visualizations
# Read documentation on interpretation
```

### 3. Experimenting with Different Models

```bash
# 1. Add your model to src/convolution_network.py
class YourModel(nn.Module):
    # ... your model code ...

# 2. Modify src/optimize.py objective function
def objective(trial):
    # ... suggest hyperparameters ...
    model = YourModel(...)  # Use your model
    # ... rest of code ...

# 3. Run optimization
python optimize.py

# 4. Train best model
# Modify src/main.py to use your model
python main.py
```

## Customization Examples

### Change Number of Trials

Edit `src/optimize.py`:

```python
# Line ~79
study.optimize(objective, n_trials=100)  # Instead of 30
```

### Narrow Search Space

Edit `src/optimize.py` in the `objective` function:

```python
# Focus on fewer layers
n_layers = trial.suggest_int('n_layers', 2, 3)  # Instead of 2-4

# Smaller learning rate range
lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)  # Instead of 1e-4 to 1e-1
```

### Add New Hyperparameter

Edit `src/optimize.py`:

```python
def objective(trial):
    # ... existing hyperparameters ...
    
    # Add new one
    use_batch_norm = trial.suggest_categorical('use_batch_norm', [True, False])
    
    # Use it in model
    model = FlexibleCNN(..., use_batch_norm=use_batch_norm)
```

### Use Different Dataset

Edit `src/data_loader.py`:

```python
# Replace CIFAR10 with your dataset
dataset = datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)
```

## Troubleshooting

### Issue: CUDA Out of Memory

**Solution:**
```python
# Edit src/optimize.py
# Reduce batch size options
batch_size = trial.suggest_categorical('batch_size', [16, 32])  # Instead of [32, 64, 128]
```

### Issue: Optimization Not Improving

**Solution:**
1. Check visualization plots
2. Increase n_trials
3. Verify search space ranges
4. Read [04_PRUNING_SAMPLERS.md](docs/04_PRUNING_SAMPLERS.md)

### Issue: Import Errors

**Solution:**
```bash
pip install --upgrade torch torchvision optuna plotly kaleido
```

### Issue: Plots Not Saving

**Solution:**
```bash
pip install kaleido
```

### Issue: Too Slow

**Solutions:**
1. Reduce epochs in search space
2. Use smaller dataset
3. Enable pruning (already enabled)
4. Run on GPU if available

## Next Steps

1. ✅ Install dependencies
2. ✅ Run quick test (5 trials)
3. ✅ Read Quick Reference
4. ✅ Run full optimization (30+ trials)
5. ✅ Read all documentation
6. ✅ Experiment with different models
7. ✅ Share your findings!

## Getting Help

1. Check [Troubleshooting](#troubleshooting) section
2. Read relevant documentation in `docs/`
3. Check [Official Optuna Docs](https://optuna.readthedocs.io/)
4. Review code comments in `src/`

## Tips for Success

1. **Start Small**: Run 5-10 trials first to debug
2. **Use GPU**: Makes training much faster
3. **Monitor Progress**: Check visualizations after each run
4. **Read Docs**: Comprehensive guides in `docs/` folder
5. **Experiment**: Try different search spaces and models
6. **Save Everything**: Results automatically saved to `results/`
7. **Be Patient**: Good optimization takes time

## Success Checklist

- [ ] Installed all dependencies
- [ ] Verified CUDA availability (if using GPU)
- [ ] Read Quick Reference
- [ ] Ran test optimization (5 trials)
- [ ] Examined results in `results/` folder
- [ ] Read full documentation
- [ ] Ran full optimization (30+ trials)
- [ ] Interpreted visualizations
- [ ] Customized search space
- [ ] Trained final model with best parameters

Happy optimizing! 🚀
