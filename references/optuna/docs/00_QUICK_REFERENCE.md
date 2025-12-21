# Optuna Quick Reference

Quick lookup guide for common Optuna operations.

## Study Creation

```python
import optuna

# Basic
study = optuna.create_study(direction='maximize')

# With persistence
study = optuna.create_study(
    storage='sqlite:///optuna.db',
    study_name='my_study',
    load_if_exists=True,
    direction='maximize'
)

# With sampler and pruner
study = optuna.create_study(
    direction='maximize',
    sampler=optuna.samplers.TPESampler(seed=42),
    pruner=optuna.pruners.MedianPruner()
)
```

## Suggesting Hyperparameters

```python
def objective(trial):
    # Integer
    n_layers = trial.suggest_int('n_layers', 1, 10)
    batch_size = trial.suggest_int('batch_size', 16, 128, step=16)
    
    # Float (linear)
    dropout = trial.suggest_float('dropout', 0.0, 0.5)
    
    # Float (logarithmic) - USE THIS FOR LEARNING RATES
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    
    # Categorical
    optimizer = trial.suggest_categorical('optimizer', ['Adam', 'SGD'])
    
    return objective_value
```

## Running Optimization

```python
# Fixed number of trials
study.optimize(objective, n_trials=100)

# Time limit
study.optimize(objective, timeout=3600)  # 1 hour

# Parallel (requires shared storage)
study.optimize(objective, n_trials=100, n_jobs=4)

# With callback
def callback(study, trial):
    print(f"Trial {trial.number}: {trial.value}")

study.optimize(objective, n_trials=100, callbacks=[callback])
```

## Accessing Results

```python
# Best value
print(study.best_value)

# Best parameters
print(study.best_params)  # Dict

# Best trial
trial = study.best_trial
print(trial.value)
print(trial.params)
print(trial.number)

# All trials
for trial in study.trials:
    print(trial.number, trial.value)

# As DataFrame
df = study.trials_dataframe()
```

## Pruning

```python
def objective(trial):
    model = build_model(trial)
    
    for epoch in range(10):
        accuracy = train_epoch(model)
        
        # Report intermediate value
        trial.report(accuracy, epoch)
        
        # Prune if necessary
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    return accuracy
```

## Samplers

```python
# TPE (default, recommended)
sampler = optuna.samplers.TPESampler(
    n_startup_trials=10,
    seed=42
)

# Random
sampler = optuna.samplers.RandomSampler(seed=42)

# Grid
search_space = {
    'lr': [0.001, 0.01, 0.1],
    'n_layers': [2, 3, 4]
}
sampler = optuna.samplers.GridSampler(search_space)

# CmaEs
sampler = optuna.samplers.CmaEsSampler(seed=42)
```

## Pruners

```python
# Median (default, recommended)
pruner = optuna.pruners.MedianPruner(
    n_startup_trials=5,
    n_warmup_steps=0
)

# Percentile (more aggressive)
pruner = optuna.pruners.PercentilePruner(
    percentile=25.0,
    n_startup_trials=5
)

# Successive Halving
pruner = optuna.pruners.SuccessiveHalvingPruner()

# Hyperband
pruner = optuna.pruners.HyperbandPruner()

# No pruning
pruner = optuna.pruners.NopPruner()
```

## Visualization

```python
import optuna.visualization as vis

# Optimization history
fig = vis.plot_optimization_history(study)
fig.show()
fig.write_image('history.png')

# Parameter importance
fig = vis.plot_param_importances(study)
fig.show()

# Parallel coordinate
fig = vis.plot_parallel_coordinate(study)
fig.show()

# Slice plot
fig = vis.plot_slice(study)
fig.show()

# Contour plot
fig = vis.plot_contour(study, params=['lr', 'n_layers'])
fig.show()

# EDF (compare studies)
fig = vis.plot_edf([study1, study2])
fig.show()

# Intermediate values
fig = vis.plot_intermediate_values(study)
fig.show()
```

## Loading Existing Study

```python
study = optuna.load_study(
    study_name='my_study',
    storage='sqlite:///optuna.db'
)

# Continue optimization
study.optimize(objective, n_trials=50)
```

## Complete Example

```python
import optuna
import torch
from torch import nn

def objective(trial):
    # Suggest hyperparameters
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    n_layers = trial.suggest_int('n_layers', 1, 5)
    dropout = trial.suggest_float('dropout', 0.0, 0.5)
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'SGD'])
    
    # Build model
    model = build_model(n_layers, dropout)
    
    # Create optimizer
    if optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    
    # Train and evaluate
    for epoch in range(10):
        train(model, optimizer)
        accuracy = evaluate(model)
        
        # For pruning
        trial.report(accuracy, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    return accuracy

# Create study
study = optuna.create_study(
    direction='maximize',
    storage='sqlite:///optuna.db',
    study_name='my_study',
    load_if_exists=True,
    sampler=optuna.samplers.TPESampler(seed=42),
    pruner=optuna.pruners.MedianPruner()
)

# Optimize
study.optimize(objective, n_trials=100)

# Results
print(f"Best value: {study.best_value}")
print(f"Best params: {study.best_params}")

# Visualize
fig = optuna.visualization.plot_optimization_history(study)
fig.write_image('optimization_history.png')

fig = optuna.visualization.plot_param_importances(study)
fig.write_image('param_importances.png')
```

## Common Patterns

### Multi-stage Optimization

```python
# Stage 1: Architecture
study1 = optuna.create_study()
study1.optimize(objective_architecture, n_trials=50)

# Stage 2: Training hyperparameters
best_arch = study1.best_params
study2 = optuna.create_study()
study2.optimize(lambda t: objective_training(t, best_arch), n_trials=50)
```

### Conditional Hyperparameters

```python
def objective(trial):
    optimizer = trial.suggest_categorical('optimizer', ['Adam', 'SGD'])
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    
    if optimizer == 'SGD':
        momentum = trial.suggest_float('momentum', 0.5, 0.99)
    else:
        momentum = 0  # Not used for Adam
    
    # Use optimizer and momentum...
```

### Save Custom Attributes

```python
def objective(trial):
    # ... train model ...
    
    trial.set_user_attr('model_size', count_parameters(model))
    trial.set_user_attr('best_epoch', best_epoch)
    
    return accuracy

# Later, access custom attributes
for trial in study.trials:
    print(trial.user_attrs)
```

### Export Results

```python
# To JSON
import json

results = {
    'best_value': study.best_value,
    'best_params': study.best_params
}
with open('results.json', 'w') as f:
    json.dump(results, f, indent=4)

# To CSV
df = study.trials_dataframe()
df.to_csv('trials.csv', index=False)
```

## Cheat Sheet

| Task | Code |
|------|------|
| Create study | `study = optuna.create_study(direction='maximize')` |
| Int parameter | `trial.suggest_int('n', 1, 10)` |
| Float parameter | `trial.suggest_float('x', 0.0, 1.0)` |
| Log-scale float | `trial.suggest_float('lr', 1e-5, 1e-1, log=True)` |
| Categorical | `trial.suggest_categorical('opt', ['A', 'B'])` |
| Run optimization | `study.optimize(objective, n_trials=100)` |
| Best value | `study.best_value` |
| Best params | `study.best_params` |
| Report intermediate | `trial.report(value, step)` |
| Check pruning | `if trial.should_prune(): raise optuna.TrialPruned()` |
| Save to DB | `storage='sqlite:///optuna.db'` |
| Load study | `optuna.load_study(study_name='...', storage='...')` |
| Plot history | `optuna.visualization.plot_optimization_history(study)` |
| Plot importance | `optuna.visualization.plot_param_importances(study)` |

## Scale Guidelines

| Hyperparameter Type | Method |
|-------------------|--------|
| Learning rate | `suggest_float(..., log=True)` |
| Weight decay | `suggest_float(..., log=True)` |
| Dropout | `suggest_float(...)` |
| Momentum | `suggest_float(...)` |
| Batch size | `suggest_categorical([...])` or `suggest_int(..., step=...)` |
| Number of layers | `suggest_int(...)` |
| Number of units | `suggest_int(...)` or `suggest_categorical([...])` |

## Installation

```bash
# Basic
pip install optuna

# With visualization
pip install optuna plotly kaleido

# With dashboard
pip install optuna optuna-dashboard
```

## Resources

- Docs: https://optuna.readthedocs.io/
- GitHub: https://github.com/optuna/optuna
- Examples: https://github.com/optuna/optuna-examples
