# Optuna Overview

## What is Optuna?

Optuna is an automatic hyperparameter optimization framework designed for machine learning. It uses a "define-by-run" API that dynamically constructs the search space, making it highly flexible and efficient for hyperparameter tuning.

## Core Concepts

### 1. Study
A study is the optimization session. It manages the entire optimization process.

```python
study = optuna.create_study(direction="maximize")
```

**Direction:**
- `"maximize"`: When optimizing metrics like accuracy, F1-score
- `"minimize"`: When optimizing metrics like loss, error rate

### 2. Trial
A trial is a single execution of the objective function. Each trial suggests hyperparameters and returns an objective value.

```python
def objective(trial):
    x = trial.suggest_float('x', -10, 10)
    return (x - 2) ** 2
```

### 3. Objective Function
The function that Optuna tries to optimize. It receives a trial object and returns a score.

**Key Points:**
- Must accept a `trial` parameter
- Must return a single numeric value (float or int)
- Should not raise exceptions unless you want to prune the trial

### 4. Samplers
Algorithms that suggest hyperparameters for each trial.

**Common Samplers:**
- `TPESampler` (default): Tree-structured Parzen Estimator - works well in most cases
- `RandomSampler`: Random search - baseline approach
- `GridSampler`: Exhaustive grid search - for small search spaces
- `CmaEsSampler`: Covariance Matrix Adaptation Evolution Strategy - good for continuous spaces

### 5. Pruners
Algorithms that stop unpromising trials early.

**Common Pruners:**
- `MedianPruner`: Prunes if the trial's intermediate value is worse than the median
- `PercentilePruner`: Prunes if worse than the n-th percentile
- `SuccessiveHalvingPruner`: Tournament-style pruning
- `HyperbandPruner`: Combination of successive halving with different resource allocations

## Why Use Optuna?

### Advantages

1. **Define-by-Run API**
   - Dynamically construct search spaces
   - Conditional hyperparameters based on other hyperparameters
   - More flexible than static configuration

2. **Efficient Optimization**
   - Smart sampling algorithms (TPE by default)
   - Early stopping of unpromising trials (pruning)
   - Parallel execution support

3. **Easy Integration**
   - Works with any machine learning framework
   - Simple Python API
   - No configuration files needed

4. **Persistence**
   - Save/load studies from databases
   - Resume interrupted optimizations
   - Share studies across processes

5. **Rich Visualization**
   - Built-in plotting functions
   - Interactive dashboards
   - Parameter importance analysis

### When to Use Optuna

**Use Optuna when:**
- You have multiple hyperparameters to tune (>3)
- Manual tuning is time-consuming
- You need reproducible optimization
- You want to leverage parallel computing
- You need conditional hyperparameters

**Don't use Optuna when:**
- You have only 1-2 hyperparameters (manual tuning might be faster)
- Your objective function is extremely fast (<0.1s)
- You don't have computational resources for multiple trials
- Search space is very small (use grid search)

## Basic Workflow

```python
import optuna

# 1. Define objective function
def objective(trial):
    # Suggest hyperparameters
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    n_layers = trial.suggest_int('n_layers', 1, 5)
    
    # Train model with suggested hyperparameters
    model = build_model(n_layers)
    accuracy = train_model(model, lr)
    
    # Return metric to optimize
    return accuracy

# 2. Create study
study = optuna.create_study(direction='maximize')

# 3. Optimize
study.optimize(objective, n_trials=100)

# 4. Get results
print(f"Best accuracy: {study.best_value}")
print(f"Best params: {study.best_params}")
```

## Storage and Persistence

Optuna supports multiple storage backends:

```python
# SQLite (local file)
study = optuna.create_study(
    storage='sqlite:///optuna.db',
    study_name='my_study',
    load_if_exists=True
)

# PostgreSQL (for production)
study = optuna.create_study(
    storage='postgresql://user:pass@localhost/dbname',
    study_name='my_study'
)

# In-memory (no persistence)
study = optuna.create_study()
```

## Parallel Optimization

Run multiple trials simultaneously:

```python
# Process 1
study = optuna.create_study(
    storage='sqlite:///optuna.db',
    study_name='shared_study',
    load_if_exists=True
)
study.optimize(objective, n_trials=50)

# Process 2 (can run simultaneously)
study = optuna.create_study(
    storage='sqlite:///optuna.db',
    study_name='shared_study',
    load_if_exists=True
)
study.optimize(objective, n_trials=50)
```

## Best Practices

1. **Start with fewer trials** (10-20) to debug your objective function
2. **Use logging** to monitor progress
3. **Save your study** to a database for persistence
4. **Use pruning** to save computational resources
5. **Visualize results** to understand the search space
6. **Set appropriate ranges** for hyperparameters based on domain knowledge
7. **Use `log=True`** for learning rates and other exponential-scale parameters
8. **Handle exceptions** in the objective function properly
9. **Monitor resource usage** when running parallel trials
10. **Document your search space** for reproducibility

## Common Pitfalls

1. **Too many trials without pruning**: Wastes resources
2. **Search space too large**: Makes optimization inefficient
3. **Not using log scale for learning rates**: Poor exploration
4. **Returning None or NaN**: Causes study to fail
5. **Not persisting to storage**: Lose progress if interrupted
6. **Ignoring parameter importance plots**: Miss insights about what matters
7. **Using categorical for continuous values**: Reduces optimization efficiency
8. **Not validating on separate data**: Overfitting to validation set
