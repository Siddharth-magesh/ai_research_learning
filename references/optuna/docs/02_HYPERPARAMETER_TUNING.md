# Hyperparameter Tuning Guide

## Introduction to Hyperparameter Optimization

Hyperparameter optimization is the process of finding the best configuration of hyperparameters for your machine learning model. Unlike model parameters (which are learned during training), hyperparameters are set before training begins.

## Types of Hyperparameters

### 1. Model Architecture Hyperparameters
- Number of layers
- Number of units/channels per layer
- Kernel sizes (for CNNs)
- Attention heads (for Transformers)

### 2. Optimization Hyperparameters
- Learning rate
- Batch size
- Optimizer type (Adam, SGD, etc.)
- Momentum
- Weight decay (L2 regularization)

### 3. Regularization Hyperparameters
- Dropout rate
- L1/L2 regularization strength
- Data augmentation parameters

### 4. Training Hyperparameters
- Number of epochs
- Learning rate schedule
- Early stopping patience

## Suggesting Hyperparameters in Optuna

### Integer Hyperparameters

```python
def objective(trial):
    # Discrete integer values
    n_layers = trial.suggest_int('n_layers', 1, 5)
    # With step
    batch_size = trial.suggest_int('batch_size', 16, 128, step=16)
    # Result: 16, 32, 48, 64, 80, 96, 112, 128
```

**When to use:**
- Number of layers
- Number of units/channels
- Batch size (with step)
- Epochs

### Float Hyperparameters

```python
def objective(trial):
    # Linear scale
    dropout = trial.suggest_float('dropout', 0.0, 0.5)
    
    # Logarithmic scale (RECOMMENDED for learning rates)
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    
    # With step
    momentum = trial.suggest_float('momentum', 0.5, 0.99, step=0.01)
```

**When to use:**
- Dropout rate: linear scale
- Learning rate: logarithmic scale
- Weight decay: logarithmic scale
- Momentum: linear scale

**Why log scale?**
Learning rates often span several orders of magnitude (0.001 to 0.1). Log scale ensures even exploration across this range.

### Categorical Hyperparameters

```python
def objective(trial):
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'SGD', 'RMSprop'])
    activation = trial.suggest_categorical('activation', ['relu', 'tanh', 'elu'])
```

**When to use:**
- Optimizer selection
- Activation functions
- Loss functions
- Any discrete non-numeric choices

### Logarithmic Uniform (Alternative Syntax)

```python
def objective(trial):
    # Old style (still works)
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
    
    # New style (preferred)
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
```

## Conditional Hyperparameters

Sometimes hyperparameters depend on other hyperparameters:

```python
def objective(trial):
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'SGD'])
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    
    if optimizer_name == 'SGD':
        # Momentum only relevant for SGD
        momentum = trial.suggest_float('momentum', 0.5, 0.99)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    return train_and_evaluate(optimizer)
```

**Benefits:**
- More efficient search space
- Avoids meaningless combinations
- Better optimization results

## Search Space Design

### Principles

1. **Start Broad, Then Narrow**
   ```python
   # First study: broad search
   lr = trial.suggest_float('lr', 1e-6, 1e-1, log=True)
   
   # Second study: narrow around best value from first study
   lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
   ```

2. **Use Domain Knowledge**
   ```python
   # Good: based on common practices
   dropout = trial.suggest_float('dropout', 0.1, 0.5)
   
   # Bad: unrealistic range
   dropout = trial.suggest_float('dropout', 0.0, 0.99)
   ```

3. **Consider Computational Cost**
   ```python
   # Expensive hyperparameters: fewer options
   n_layers = trial.suggest_int('n_layers', 2, 4)
   
   # Cheap hyperparameters: more options
   activation = trial.suggest_categorical('activation', 
       ['relu', 'leaky_relu', 'elu', 'gelu', 'swish'])
   ```

### Common Search Spaces

#### CNN Optimization

```python
def objective(trial):
    # Architecture
    n_layers = trial.suggest_int('n_layers', 2, 5)
    channels = trial.suggest_int('channels', 16, 128, step=16)
    
    # Regularization
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
    
    # Optimization
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'SGD'])
    
    # Training
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    epochs = trial.suggest_int('epochs', 10, 50, step=10)
    
    # Build and train model...
```

#### Transformer Optimization

```python
def objective(trial):
    # Architecture
    n_layers = trial.suggest_int('n_layers', 4, 12)
    d_model = trial.suggest_categorical('d_model', [256, 512, 768])
    n_heads = trial.suggest_categorical('n_heads', [4, 8, 12])
    d_ff = trial.suggest_int('d_ff', 1024, 4096, step=512)
    
    # Regularization
    dropout = trial.suggest_float('dropout', 0.1, 0.3)
    
    # Optimization
    lr = trial.suggest_float('lr', 1e-5, 5e-4, log=True)
    warmup_steps = trial.suggest_int('warmup_steps', 1000, 10000, step=1000)
```

## Practical Example: Our CNN Optimizer

```python
def objective(trial):
    # Model architecture
    n_layers = trial.suggest_int('n_layers', 2, 4)
    channels = trial.suggest_int('channels', 16, 64, step=16)
    dropout = trial.suggest_int('dropout', 0, 5) / 100.0
    activation = trial.suggest_categorical('activation', ['relu', 'leaky_relu', 'elu'])
    
    # Optimizer configuration
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'SGD'])
    lr = trial.suggest_float('lr', 1e-4, 1e-1, log=True)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-5, 1e-2)
    
    # Conditional hyperparameter
    if optimizer_name == 'SGD':
        momentum = trial.suggest_float('momentum', 0.5, 0.9)
    
    # Scheduler configuration
    scheduler_name = trial.suggest_categorical('scheduler', 
        ['StepLR', 'ExponentialLR', 'CosineAnnealingLR'])
    
    # Training configuration
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    epochs = trial.suggest_int('epochs', 5, 20, step=5)
```

## Optimization Strategies

### 1. Multi-Stage Optimization

**Stage 1: Architecture Search**
```python
# Focus on model architecture only
def objective_architecture(trial):
    n_layers = trial.suggest_int('n_layers', 2, 6)
    channels = trial.suggest_int('channels', 16, 128, step=16)
    # Use fixed learning rate
    lr = 0.001
```

**Stage 2: Training Configuration**
```python
# Use best architecture, optimize training
def objective_training(trial):
    n_layers = 4  # From stage 1
    channels = 64  # From stage 1
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
```

### 2. Hierarchical Search

```python
def objective(trial):
    # High-level decision first
    model_type = trial.suggest_categorical('model_type', ['simple', 'complex'])
    
    if model_type == 'simple':
        n_layers = trial.suggest_int('n_layers', 2, 3)
        channels = trial.suggest_int('channels', 16, 32, step=16)
    else:
        n_layers = trial.suggest_int('n_layers', 4, 6)
        channels = trial.suggest_int('channels', 64, 128, step=32)
```

### 3. Budget-Aware Search

```python
def objective(trial):
    # Expensive hyperparameters: coarse search
    n_layers = trial.suggest_int('n_layers', 2, 4)  # Only 3 values
    
    # Cheap hyperparameters: fine search
    dropout = trial.suggest_float('dropout', 0.0, 0.5)  # Continuous
    activation = trial.suggest_categorical('activation', 
        ['relu', 'leaky_relu', 'elu', 'gelu'])  # 4 options
```

## Hyperparameter Importance Analysis

After optimization, analyze which hyperparameters matter most:

```python
# Run optimization
study.optimize(objective, n_trials=100)

# Analyze importance
fig = optuna.visualization.plot_param_importances(study)
fig.show()
```

**Interpretation:**
- High importance: Focus on these in future optimizations
- Low importance: Can use fixed values
- Helps reduce search space for next study

## Tips and Best Practices

### Do's

1. **Use appropriate scales**
   - Learning rate: logarithmic
   - Dropout: linear
   - Weight decay: logarithmic

2. **Start with literature values**
   - Check paper implementations
   - Use proven ranges as starting points

3. **Consider computational budget**
   - More important hyperparameters get finer granularity
   - Less important ones get coarser granularity

4. **Use pruning for expensive trials**
   - Saves time by stopping bad trials early

5. **Run multiple studies**
   - Iteratively narrow search space
   - Focus on important hyperparameters

### Don'ts

1. **Don't optimize everything at once**
   - Too large search space
   - Inefficient optimization

2. **Don't use linear scale for learning rates**
   - Poor exploration
   - Misses important regions

3. **Don't set unrealistic ranges**
   - Wastes trials on impossible values
   - Slows down convergence

4. **Don't ignore domain knowledge**
   - Use reasonable bounds
   - Exclude known-bad combinations

5. **Don't stop too early**
   - Run enough trials (50-100 minimum)
   - More trials for larger search spaces

## Choosing Number of Trials

**Rule of thumb:**
- Small search space (< 10 hyperparameters): 50-100 trials
- Medium search space (10-20 hyperparameters): 100-200 trials
- Large search space (> 20 hyperparameters): 200-500 trials

**Factors to consider:**
- Computational budget
- Trial duration
- Search space size
- Optimization algorithm (TPE needs more trials than grid search)

## Validation Strategy

```python
def objective(trial):
    # Build model with trial hyperparameters
    model = build_model(trial)
    
    # IMPORTANT: Use separate validation set
    # Don't use test set during optimization
    train_loader, val_loader = get_data_loaders()
    
    # Train on training set
    train(model, train_loader)
    
    # Evaluate on validation set
    accuracy = evaluate(model, val_loader)
    
    return accuracy

# After optimization, evaluate on test set
best_model = build_model_with_best_params()
test_accuracy = evaluate(best_model, test_loader)
```

**Why separate sets?**
- Prevents overfitting to validation set
- Ensures generalization to unseen data
- Gives honest performance estimate
