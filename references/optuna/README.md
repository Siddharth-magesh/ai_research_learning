# Optuna Hyperparameter Optimization Project

A comprehensive implementation of hyperparameter optimization using Optuna for CNN training on CIFAR-10 dataset.

> **🚀 Quick Start**: Read [GETTING_STARTED.md](GETTING_STARTED.md) for step-by-step instructions

## Project Structure

```
references/optuna/
├── src/
│   ├── convolution_network.py    # CNN model architectures
│   ├── data_loader.py            # CIFAR-10 data loading
│   ├── trainer.py                # Training and evaluation logic
│   ├── optimize.py               # Optuna optimization script
│   └── main.py                   # Train with best hyperparameters
├── docs/
│   ├── 01_OPTUNA_OVERVIEW.md     # Comprehensive Optuna introduction
│   ├── 02_HYPERPARAMETER_TUNING.md  # Hyperparameter tuning guide
│   ├── 03_METHODS_REFERENCE.md   # Detailed methods documentation
│   ├── 04_PRUNING_SAMPLERS.md    # Pruning and samplers deep dive
│   └── 05_VISUALIZATION.md       # Visualization guide
└── results/                      # Output directory (created automatically)
    ├── cnn_optuna.db            # SQLite database with all trials
    ├── best_results.json        # Best hyperparameters and accuracy
    ├── optimization_history.png # Optimization progress plot
    ├── param_importances.png    # Parameter importance plot
    ├── parallel_coordinate.png  # Parallel coordinate plot
    └── best_model.pth           # Trained model weights
```

## Quick Start

### Installation

```bash
pip install torch torchvision optuna plotly kaleido
```

### Run Hyperparameter Optimization

```bash
cd references/optuna/src
python optimize.py
```

This will:
- Optimize hyperparameters for 30 trials
- Save results to `../results/`
- Generate visualization plots
- Store study in SQLite database

### Train with Best Hyperparameters

```bash
cd references/optuna/src
python main.py
```

This will train a model using the best found hyperparameters.

## Features

### Models

1. **SimpleCNN**: Basic 2-layer CNN
   - Fixed architecture
   - Good baseline model

2. **FlexibleCNN**: Configurable CNN
   - Variable number of layers
   - Configurable channels
   - Multiple activation functions
   - Dropout regularization

### Hyperparameters Optimized

- **Architecture**: Number of layers, channels per layer
- **Regularization**: Dropout rate, weight decay
- **Optimization**: Learning rate, optimizer type (Adam/SGD), momentum
- **Scheduler**: Learning rate schedules (StepLR, ExponentialLR, CosineAnnealingLR)
- **Training**: Batch size, number of epochs
- **Activation**: ReLU, LeakyReLU, ELU

### Optuna Configuration

- **Sampler**: TPESampler (default, best for most cases)
- **Pruner**: MedianPruner (stops unpromising trials early)
- **Storage**: SQLite database for persistence
- **Trials**: 30 by default (configurable)

## Documentation

Comprehensive documentation is available in the `docs/` folder:

### 1. [Optuna Overview](docs/01_OPTUNA_OVERVIEW.md)
- What is Optuna?
- Core concepts (Study, Trial, Objective, Samplers, Pruners)
- Why use Optuna?
- When to use and when not to use
- Basic workflow
- Best practices

### 2. [Hyperparameter Tuning Guide](docs/02_HYPERPARAMETER_TUNING.md)
- Types of hyperparameters
- Suggesting hyperparameters (int, float, categorical)
- Conditional hyperparameters
- Search space design
- Optimization strategies
- Validation strategy
- Tips and best practices

### 3. [Methods Reference](docs/03_METHODS_REFERENCE.md)
- Study methods (create_study, optimize, etc.)
- Trial methods (suggest_*, report, should_prune, etc.)
- Sampler methods (TPE, Random, Grid, CmaEs, QMC)
- Pruner methods (Median, Percentile, SuccessiveHalving, Hyperband)
- Quick decision guide

### 4. [Pruning and Samplers Deep Dive](docs/04_PRUNING_SAMPLERS.md)
- How pruning works
- All pruning algorithms explained
- All sampling algorithms explained
- Choosing the right pruner
- Choosing the right sampler
- Combining pruner and sampler
- Performance comparisons

### 5. [Visualization Guide](docs/05_VISUALIZATION.md)
- All visualization types
- Interpretation guide
- Saving visualizations
- Complete workflow examples
- Advanced customization
- Troubleshooting

## Code Overview

### convolution_network.py

Two CNN architectures:

```python
# Simple baseline
model = SimpleCNN(channels=32, dropout=0.2)

# Flexible for optimization
model = FlexibleCNN(
    n_layers=3,
    channels=32,
    dropout=0.3,
    activation='relu'
)
```

### data_loader.py

CIFAR-10 data loading with proper normalization:

```python
train_loader, val_loader = get_data_loader(batch_size=64)
```

### trainer.py

Training and evaluation with scheduler support:

```python
trainer = Trainer(
    model=model,
    device=device,
    train_dataloader=train_loader,
    val_dataloader=val_loader,
    optimizer=optimizer,
    scheduler=scheduler  # Optional
)

trainer.train()  # One epoch
accuracy = trainer.evaluate()
```

### optimize.py

Optuna optimization:

```python
def objective(trial):
    # Suggest hyperparameters
    n_layers = trial.suggest_int('n_layers', 2, 4)
    lr = trial.suggest_float('lr', 1e-4, 1e-1, log=True)
    # ... more hyperparameters
    
    # Build and train model
    model = FlexibleCNN(n_layers, ...)
    accuracy = train_and_evaluate(model)
    
    return accuracy

# Run optimization
study.optimize(objective, n_trials=30)
```

### main.py

Train final model with best hyperparameters:

```python
python main.py
```

## Results

After running optimization, check:

1. **best_results.json**: Best hyperparameters and accuracy
2. **Visualizations**: Understanding the optimization process
3. **Database**: All trial history for further analysis

Example output:
```
Best accuracy: 0.7234
Best hyperparameters:
  n_layers: 3
  channels: 48
  dropout: 0.03
  activation: relu
  lr: 0.00123
  optimizer: Adam
  batch_size: 64
  epochs: 15
```

## Customization

### Change Number of Trials

Edit `optimize.py`:
```python
study.optimize(objective, n_trials=100)  # Instead of 30
```

### Modify Search Space

Edit the `objective` function in `optimize.py`:
```python
# Narrower range
n_layers = trial.suggest_int('n_layers', 2, 3)

# Different optimizer options
optimizer_name = trial.suggest_categorical('optimizer', 
    ['Adam', 'SGD', 'AdamW', 'RMSprop'])
```

### Different Dataset

Modify `data_loader.py`:
```python
dataset = datasets.MNIST(...)  # Instead of CIFAR10
```

### Custom Model

Add your model to `convolution_network.py` and use in `objective`.

## Advanced Usage

### Resume Optimization

```python
# Automatically resumes if study exists
study = optuna.create_study(
    storage='sqlite:///results/optuna.db',
    study_name='cnn_optimization',
    load_if_exists=True  # Key parameter
)
study.optimize(objective, n_trials=50)  # Add 50 more trials
```

### Parallel Optimization

Run multiple processes:

```bash
# Terminal 1
python optimize.py

# Terminal 2 (simultaneously)
python optimize.py
```

Both will share the same database and coordinate trials.

### Custom Callback

```python
def callback(study, trial):
    if trial.number % 10 == 0:
        print(f"Best so far: {study.best_value}")

study.optimize(objective, n_trials=100, callbacks=[callback])
```

### Export Results

```python
import optuna
import pandas as pd

study = optuna.load_study(
    study_name='cnn_optimization',
    storage='sqlite:///results/optuna.db'
)

# Export to CSV
df = study.trials_dataframe()
df.to_csv('all_trials.csv', index=False)

# Export best parameters
import json
with open('best_params.json', 'w') as f:
    json.dump(study.best_params, f, indent=4)
```

## Troubleshooting

### Import Errors

```bash
pip install torch torchvision optuna plotly kaleido
```

### CUDA Out of Memory

Reduce batch size in search space:
```python
batch_size = trial.suggest_categorical('batch_size', [16, 32])
```

### Optimization Not Improving

1. Check search space ranges
2. Increase number of trials
3. Try different sampler
4. Verify objective function is correct

### Pruning Too Aggressive

Adjust pruner:
```python
pruner = optuna.pruners.MedianPruner(n_startup_trials=10)
```

### Visualizations Not Saving

Install kaleido:
```bash
pip install kaleido
```

## Performance Tips

1. **Use pruning** to save time on bad trials
2. **Start with fewer trials** (10-20) to debug
3. **Use GPU** if available (`cuda` device)
4. **Adjust n_startup_trials** based on search space size
5. **Save study to database** to avoid losing progress
6. **Run parallel trials** for faster optimization

## Best Practices

1. **Always set a seed** for reproducibility:
   ```python
   sampler = optuna.samplers.TPESampler(seed=42)
   ```

2. **Use log scale for learning rates**:
   ```python
   lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
   ```

3. **Save intermediate results**:
   ```python
   trial.set_user_attr('best_epoch', best_epoch)
   ```

4. **Validate on separate data**:
   - Use train/val split during optimization
   - Test on separate test set after optimization

5. **Document your search space**:
   - Comment why ranges are chosen
   - Based on literature or previous experiments

## Learn More

- [Official Optuna Documentation](https://optuna.readthedocs.io/)
- [Optuna GitHub](https://github.com/optuna/optuna)
- [Optuna Examples](https://github.com/optuna/optuna-examples)

## License

This project is for educational purposes.

## Contributing

Feel free to:
- Experiment with different models
- Try different datasets
- Modify hyperparameter ranges
- Share findings and improvements
