# Optuna Methods Reference

## Study Methods

### Creating a Study

#### `optuna.create_study()`

Creates a new study or loads an existing one.

```python
study = optuna.create_study(
    study_name='my_study',
    direction='maximize',
    sampler=optuna.samplers.TPESampler(),
    pruner=optuna.pruners.MedianPruner(),
    storage='sqlite:///optuna.db',
    load_if_exists=True
)
```

**Parameters:**

- `study_name` (str, optional): Name of the study
- `direction` (str): `"maximize"` or `"minimize"`
- `sampler` (BaseSampler, optional): Sampling algorithm
- `pruner` (BasePruner, optional): Pruning algorithm
- `storage` (str, optional): Database URL for persistence
- `load_if_exists` (bool): Load existing study if True

**When to use:**
- At the start of every optimization session
- Use `storage` for resumable optimization
- Use `load_if_exists=True` to continue previous optimization

**Example:**
```python
# New study in memory
study = optuna.create_study(direction='maximize')

# Persistent study with SQLite
study = optuna.create_study(
    storage='sqlite:///results/optuna.db',
    study_name='cnn_optimization',
    load_if_exists=True,
    direction='maximize'
)
```

### Running Optimization

#### `study.optimize()`

Runs the optimization process.

```python
study.optimize(
    func=objective,
    n_trials=100,
    timeout=3600,
    n_jobs=1,
    show_progress_bar=True,
    callbacks=[callback_function]
)
```

**Parameters:**

- `func` (callable): Objective function to optimize
- `n_trials` (int, optional): Number of trials to run
- `timeout` (float, optional): Time limit in seconds
- `n_jobs` (int): Number of parallel jobs (-1 for all CPUs)
- `show_progress_bar` (bool): Display progress bar
- `callbacks` (list, optional): List of callback functions

**When to use:**
- Always after creating a study
- Use `n_trials` for fixed number of evaluations
- Use `timeout` for time-limited optimization
- Use `n_jobs > 1` for parallel optimization (requires shared storage)

**Example:**
```python
# Basic optimization
study.optimize(objective, n_trials=100)

# Time-limited optimization
study.optimize(objective, timeout=3600)  # 1 hour

# Parallel optimization
study.optimize(objective, n_trials=100, n_jobs=4)
```

### Accessing Results

#### `study.best_trial`

Returns the best trial object.

```python
best_trial = study.best_trial
print(f"Value: {best_trial.value}")
print(f"Params: {best_trial.params}")
```

**Attributes:**
- `value`: Objective value
- `params`: Dictionary of hyperparameters
- `number`: Trial number
- `datetime_start`: Start time
- `datetime_complete`: End time

#### `study.best_params`

Returns dictionary of best hyperparameters.

```python
best_params = study.best_params
# {'lr': 0.001, 'n_layers': 3, 'dropout': 0.2}
```

#### `study.best_value`

Returns the best objective value.

```python
best_accuracy = study.best_value
print(f"Best accuracy: {best_accuracy:.4f}")
```

#### `study.trials`

Returns list of all trials.

```python
for trial in study.trials:
    print(f"Trial {trial.number}: {trial.value}")
```

**Use cases:**
- Analyzing all trials
- Custom plotting
- Exporting results

### Study Analysis

#### `study.trials_dataframe()`

Converts trials to pandas DataFrame.

```python
df = study.trials_dataframe()
df.to_csv('results/trials.csv')
```

**Columns:**
- `number`: Trial number
- `value`: Objective value
- `datetime_start`: Start timestamp
- `datetime_complete`: End timestamp
- `duration`: Trial duration
- `params_*`: Hyperparameter values
- `state`: Trial state (COMPLETE, PRUNED, FAIL)

#### `study.get_trials()`

Gets trials with filtering.

```python
# Get only completed trials
completed = study.get_trials(states=[optuna.trial.TrialState.COMPLETE])

# Get only pruned trials
pruned = study.get_trials(states=[optuna.trial.TrialState.PRUNED])
```

## Trial Methods

### Suggesting Hyperparameters

#### `trial.suggest_int()`

Suggests an integer value.

```python
n_layers = trial.suggest_int('n_layers', 1, 10)
batch_size = trial.suggest_int('batch_size', 16, 128, step=16)
```

**Parameters:**
- `name` (str): Hyperparameter name
- `low` (int): Lower bound (inclusive)
- `high` (int): Upper bound (inclusive)
- `step` (int, optional): Step size
- `log` (bool): Use log scale

**When to use:**
- Number of layers/units
- Batch size (with step)
- Discrete integer values

#### `trial.suggest_float()`

Suggests a float value.

```python
lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
dropout = trial.suggest_float('dropout', 0.0, 0.5)
momentum = trial.suggest_float('momentum', 0.5, 0.99, step=0.01)
```

**Parameters:**
- `name` (str): Hyperparameter name
- `low` (float): Lower bound
- `high` (float): Upper bound
- `step` (float, optional): Step size
- `log` (bool): Use log scale

**When to use:**
- Learning rate (with `log=True`)
- Dropout rate
- Regularization parameters
- Momentum

**Important:** Use `log=True` for exponential-scale parameters like learning rates.

#### `trial.suggest_categorical()`

Suggests from discrete choices.

```python
optimizer = trial.suggest_categorical('optimizer', ['Adam', 'SGD', 'RMSprop'])
activation = trial.suggest_categorical('activation', ['relu', 'tanh', 'elu'])
```

**Parameters:**
- `name` (str): Hyperparameter name
- `choices` (list): List of possible values

**When to use:**
- Optimizer selection
- Activation functions
- Architecture choices
- Any discrete non-numeric values

#### `trial.suggest_loguniform()`

Suggests float from log-uniform distribution (deprecated, use `suggest_float` with `log=True`).

```python
# Old way
lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)

# New way (preferred)
lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
```

### Trial Information

#### `trial.number`

Returns the trial number (0-indexed).

```python
def objective(trial):
    print(f"Running trial {trial.number}")
    # ...
```

#### `trial.params`

Returns dictionary of suggested parameters so far.

```python
def objective(trial):
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    print(trial.params)  # {'lr': 0.001}
    
    n_layers = trial.suggest_int('n_layers', 1, 5)
    print(trial.params)  # {'lr': 0.001, 'n_layers': 3}
```

### Pruning

#### `trial.report()`

Reports an intermediate value for pruning.

```python
def objective(trial):
    model = build_model(trial)
    
    for epoch in range(10):
        accuracy = train_epoch(model)
        
        # Report intermediate value
        trial.report(accuracy, epoch)
        
        # Check if trial should be pruned
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    return accuracy
```

**Parameters:**
- `value` (float): Intermediate objective value
- `step` (int): Step number (usually epoch)

**When to use:**
- During iterative training
- When trials can be stopped early
- To save computational resources

#### `trial.should_prune()`

Checks if trial should be pruned.

```python
if trial.should_prune():
    raise optuna.TrialPruned()
```

**Returns:** Boolean indicating whether to prune

**When to use:**
- After calling `trial.report()`
- During training loop
- With a pruner configured in the study

### Setting User Attributes

#### `trial.set_user_attr()`

Sets custom attributes for the trial.

```python
def objective(trial):
    model = build_model(trial)
    accuracy = train(model)
    
    # Store additional information
    trial.set_user_attr('model_size', count_parameters(model))
    trial.set_user_attr('best_epoch', best_epoch)
    
    return accuracy
```

**Use cases:**
- Storing model size
- Recording best epoch
- Saving any trial-specific metadata

#### `study.set_user_attr()`

Sets custom attributes for the study.

```python
study.set_user_attr('dataset', 'CIFAR10')
study.set_user_attr('framework', 'PyTorch')
```

## Sampler Methods

### TPESampler (Default)

Tree-structured Parzen Estimator - generally the best choice.

```python
sampler = optuna.samplers.TPESampler(
    n_startup_trials=10,
    n_ei_candidates=24,
    seed=42
)
study = optuna.create_study(sampler=sampler)
```

**Parameters:**
- `n_startup_trials` (int): Random trials before using TPE
- `n_ei_candidates` (int): Number of candidates for EI calculation
- `seed` (int): Random seed for reproducibility

**When to use:**
- Default choice for most problems
- Works well with 10+ hyperparameters
- Good balance of exploration/exploitation

**Don't use when:**
- Very small search space (use GridSampler)
- Need reproducible grid search

### RandomSampler

Pure random search.

```python
sampler = optuna.samplers.RandomSampler(seed=42)
study = optuna.create_study(sampler=sampler)
```

**When to use:**
- Baseline comparison
- Very simple problems
- Parallel optimization without shared state

**Don't use when:**
- You have enough budget for smarter search
- Hyperparameters have dependencies

### GridSampler

Exhaustive grid search.

```python
search_space = {
    'lr': [0.001, 0.01, 0.1],
    'n_layers': [2, 3, 4],
    'dropout': [0.1, 0.3, 0.5]
}
sampler = optuna.samplers.GridSampler(search_space)
study = optuna.create_study(sampler=sampler)
```

**When to use:**
- Small search space (< 100 combinations)
- Need to try all combinations
- Reproducible results required

**Don't use when:**
- Large search space
- Continuous hyperparameters
- Limited computational budget

### CmaEsSampler

Covariance Matrix Adaptation Evolution Strategy.

```python
sampler = optuna.samplers.CmaEsSampler(
    seed=42,
    n_startup_trials=10
)
study = optuna.create_study(sampler=sampler)
```

**When to use:**
- Continuous optimization problems
- Scientific computing applications
- Few categorical hyperparameters

**Don't use when:**
- Many categorical hyperparameters
- Discrete search spaces
- Small number of trials

## Pruner Methods

### MedianPruner

Prunes trials with intermediate values worse than median.

```python
pruner = optuna.pruners.MedianPruner(
    n_startup_trials=5,
    n_warmup_steps=0,
    interval_steps=1
)
study = optuna.create_study(pruner=pruner)
```

**Parameters:**
- `n_startup_trials` (int): Number of trials before pruning starts
- `n_warmup_steps` (int): Steps before pruning can occur
- `interval_steps` (int): Interval between pruning checks

**When to use:**
- General purpose pruning
- Balanced aggressiveness
- Iterative training (epochs)

### PercentilePruner

Prunes trials worse than n-th percentile.

```python
pruner = optuna.pruners.PercentilePruner(
    percentile=25.0,
    n_startup_trials=5,
    n_warmup_steps=0
)
study = optuna.create_study(pruner=pruner)
```

**Parameters:**
- `percentile` (float): Percentile threshold (0-100)
- Lower percentile = more aggressive pruning

**When to use:**
- More aggressive than MedianPruner
- When computational budget is tight
- Many trials expected

### SuccessiveHalvingPruner

Allocates resources using successive halving.

```python
pruner = optuna.pruners.SuccessiveHalvingPruner(
    min_resource=1,
    reduction_factor=4,
    min_early_stopping_rate=0
)
study = optuna.create_study(pruner=pruner)
```

**When to use:**
- Limited computational budget
- Many trials needed
- Tournament-style selection preferred

### HyperbandPruner

Combines successive halving with different resource allocations.

```python
pruner = optuna.pruners.HyperbandPruner(
    min_resource=1,
    max_resource=10,
    reduction_factor=3
)
study = optuna.create_study(pruner=pruner)
```

**When to use:**
- Most aggressive pruning
- Very limited budget
- Large-scale optimization

### NopPruner

No pruning (baseline).

```python
pruner = optuna.pruners.NopPruner()
study = optuna.create_study(pruner=pruner)
```

**When to use:**
- Baseline comparison
- Trials are cheap
- Want to complete all trials

## Method Comparison Table

| Method | Use Case | Complexity | Speed |
|--------|----------|------------|-------|
| `suggest_int()` | Discrete integers | Low | Fast |
| `suggest_float()` | Continuous values | Low | Fast |
| `suggest_categorical()` | Discrete choices | Low | Fast |
| `TPESampler` | General optimization | Medium | Medium |
| `RandomSampler` | Baseline/simple | Low | Fast |
| `GridSampler` | Small search space | Low | Slow |
| `MedianPruner` | General pruning | Medium | Medium |
| `PercentilePruner` | Aggressive pruning | Medium | Fast |
| `HyperbandPruner` | Resource allocation | High | Fast |

## Quick Decision Guide

**Choosing Sampler:**
- Default → TPESampler
- Small space → GridSampler
- Baseline → RandomSampler
- Continuous only → CmaEsSampler

**Choosing Pruner:**
- Default → MedianPruner
- Aggressive → PercentilePruner (15-25%)
- Very aggressive → HyperbandPruner
- No pruning → NopPruner

**Choosing Suggestion Method:**
- Integers → suggest_int()
- Floats (linear) → suggest_float()
- Floats (exponential) → suggest_float(log=True)
- Categories → suggest_categorical()
