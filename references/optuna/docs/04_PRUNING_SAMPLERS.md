# Pruning and Samplers Deep Dive

## Pruning: Early Stopping for Trials

### What is Pruning?

Pruning automatically stops unpromising trials early, saving computational resources. Instead of waiting for a bad trial to complete all epochs, Optuna can detect poor performance early and terminate it.

### How Pruning Works

```python
def objective(trial):
    model = build_model(trial)
    
    for epoch in range(num_epochs):
        # Train for one epoch
        train_one_epoch(model)
        
        # Evaluate
        val_accuracy = evaluate(model, val_loader)
        
        # Report intermediate value
        trial.report(val_accuracy, epoch)
        
        # Check if should stop
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    return val_accuracy
```

**Key Components:**
1. `trial.report(value, step)`: Report intermediate results
2. `trial.should_prune()`: Check if trial should stop
3. `raise optuna.TrialPruned()`: Stop the trial

### Pruning Algorithms

## 1. MedianPruner

**Algorithm:**
- Compares trial's intermediate value to the median of all trials at the same step
- Prunes if the current trial's value is worse than the median

```python
pruner = optuna.pruners.MedianPruner(
    n_startup_trials=5,      # Don't prune first N trials
    n_warmup_steps=0,        # Don't prune first N steps
    interval_steps=1         # Check every N steps
)
```

**Example:**
```
Step 5 intermediate values:
Trial 1: 0.65
Trial 2: 0.72
Trial 3: 0.68  <- Median
Trial 4: 0.71
Trial 5: 0.63  <- Will be pruned (< median)
```

**When to use:**
- General purpose pruning
- Moderate aggressiveness
- When you want to prune obviously bad trials

**Advantages:**
- Balanced approach
- Not too aggressive
- Works well in practice

**Disadvantages:**
- Requires multiple trials to start
- May not prune enough early on

## 2. PercentilePruner

**Algorithm:**
- Prunes trials worse than the specified percentile
- More configurable than MedianPruner

```python
pruner = optuna.pruners.PercentilePruner(
    percentile=25.0,         # Prune bottom 75%
    n_startup_trials=5,
    n_warmup_steps=0,
    interval_steps=1
)
```

**Percentile Guide:**
- 50.0: Same as MedianPruner
- 25.0: Aggressive (keep top 25%)
- 10.0: Very aggressive (keep top 10%)
- 75.0: Conservative (keep top 75%)

**When to use:**
- When you want more/less aggressive pruning than median
- Limited computational budget (lower percentile)
- Want to explore more (higher percentile)

**Example:**
```python
# Very aggressive - only keep top 10%
pruner = optuna.pruners.PercentilePruner(percentile=10.0)

# Conservative - only prune bottom 50%
pruner = optuna.pruners.PercentilePruner(percentile=50.0)
```

## 3. SuccessiveHalvingPruner

**Algorithm:**
- Allocates resources in a tournament-style manner
- Progressively eliminates worst-performing trials
- Doubles resources (e.g., epochs) at each round

```python
pruner = optuna.pruners.SuccessiveHalvingPruner(
    min_resource=1,              # Minimum epochs
    reduction_factor=4,          # Keep 1/4 of trials each round
    min_early_stopping_rate=0
)
```

**How it works:**
```
Round 1: 16 trials × 1 epoch = 16 trial-epochs
Round 2: 4 trials × 4 epochs = 16 trial-epochs
Round 3: 1 trial × 16 epochs = 16 trial-epochs
Total: 48 trial-epochs (vs 256 without pruning)
```

**When to use:**
- Very limited computational budget
- Want to try many configurations
- Don't need all trials to complete

**Advantages:**
- Very efficient resource usage
- Tries many hyperparameters
- Proven theoretical guarantees

**Disadvantages:**
- Aggressive pruning
- May prune trials that would improve later
- Requires careful tuning of min_resource

## 4. HyperbandPruner

**Algorithm:**
- Extension of SuccessiveHalving
- Runs multiple SuccessiveHalving rounds with different resource allocations
- Robust to unknown optimal resource allocation

```python
pruner = optuna.pruners.HyperbandPruner(
    min_resource=1,
    max_resource=27,
    reduction_factor=3
)
```

**When to use:**
- Don't know how many epochs are needed
- Want state-of-the-art pruning
- Very large-scale optimization

**Advantages:**
- Most theoretically sound
- No need to guess resource allocation
- Very efficient

**Disadvantages:**
- Most complex
- Harder to understand/debug
- May be overkill for small problems

## 5. NopPruner

**Algorithm:**
- No pruning - all trials complete

```python
pruner = optuna.pruners.NopPruner()
```

**When to use:**
- Baseline comparison
- Trials are very cheap
- Suspicious of pruning

## Choosing a Pruner

### Decision Tree

```
Is computational budget very limited?
├─ Yes → HyperbandPruner or SuccessiveHalvingPruner
└─ No
   ├─ Want aggressive pruning? → PercentilePruner(10-25)
   ├─ Want moderate pruning? → MedianPruner
   ├─ Want conservative pruning? → PercentilePruner(50-75)
   └─ No pruning? → NopPruner
```

### Practical Recommendations

**Starting out:**
```python
pruner = optuna.pruners.MedianPruner(
    n_startup_trials=10,
    n_warmup_steps=5
)
```

**Limited budget:**
```python
pruner = optuna.prulers.PercentilePruner(
    percentile=20.0,
    n_startup_trials=5
)
```

**Very limited budget:**
```python
pruner = optuna.pruners.HyperbandPruner()
```

## Samplers: Hyperparameter Selection

### What is a Sampler?

A sampler determines which hyperparameters to try next. It's the brain of the optimization.

## 1. TPESampler (Tree-structured Parzen Estimator)

**Algorithm:**
- Models good and bad trials separately
- Samples from regions likely to produce good results
- Default and recommended for most use cases

```python
sampler = optuna.samplers.TPESampler(
    n_startup_trials=10,     # Random trials before using TPE
    n_ei_candidates=24,      # Candidates for expected improvement
    seed=42,                 # For reproducibility
    multivariate=False       # Independent or joint modeling
)
```

**How it works:**
1. First `n_startup_trials`: random exploration
2. After that: model P(good) and P(bad)
3. Sample from regions where P(good)/P(bad) is high

**When to use:**
- Default choice
- Medium to large search spaces
- Mixed hyperparameter types
- Most deep learning problems

**Parameters to tune:**
- `n_startup_trials`: Increase for larger search spaces (20-50)
- `multivariate`: Set to True if hyperparameters interact
- `seed`: Set for reproducibility

**Example:**
```python
# For large search space
sampler = optuna.samplers.TPESampler(
    n_startup_trials=20,
    multivariate=True,
    seed=42
)
```

**Advantages:**
- Works very well in practice
- Handles mixed types (int, float, categorical)
- Balances exploration/exploitation
- Proven track record

**Disadvantages:**
- Needs at least 10-20 trials to be effective
- Not optimal for very small search spaces
- Black box (hard to interpret)

## 2. RandomSampler

**Algorithm:**
- Completely random selection
- No learning from previous trials

```python
sampler = optuna.samplers.RandomSampler(seed=42)
```

**When to use:**
- Baseline comparison
- Very simple problems
- Debugging
- Parallel optimization without shared storage

**Advantages:**
- Simple and fast
- No dependencies between trials
- Easy parallel execution
- Reproducible with seed

**Disadvantages:**
- Inefficient for large search spaces
- Doesn't learn from previous trials
- Requires many trials

## 3. GridSampler

**Algorithm:**
- Exhaustive search over predefined grid
- Tries all combinations

```python
search_space = {
    'lr': [0.001, 0.01, 0.1],
    'batch_size': [32, 64, 128],
    'dropout': [0.1, 0.3, 0.5]
}
sampler = optuna.samplers.GridSampler(search_space)
```

**Number of trials:** Product of all choices (3 × 3 × 3 = 27)

**When to use:**
- Small search space (< 100 combinations)
- Need reproducible results
- Want to try all combinations
- Scientific publication

**Advantages:**
- Deterministic
- Complete coverage
- Easy to understand
- Reproducible

**Disadvantages:**
- Exponential growth with dimensions
- Inefficient for continuous variables
- Requires knowing good ranges beforehand

**Curse of dimensionality:**
```python
# 3 hyperparameters × 5 values = 125 trials
# 5 hyperparameters × 5 values = 3,125 trials
# 10 hyperparameters × 5 values = 9,765,625 trials!
```

## 4. CmaEsSampler

**Algorithm:**
- Covariance Matrix Adaptation Evolution Strategy
- Uses evolutionary algorithm
- Models covariance between parameters

```python
sampler = optuna.samplers.CmaEsSampler(
    x0=None,                 # Initial solution
    sigma0=None,             # Initial step size
    seed=42,
    n_startup_trials=1
)
```

**When to use:**
- Continuous optimization problems
- Scientific computing
- Few categorical hyperparameters
- When TPE isn't working well

**Advantages:**
- Very good for continuous spaces
- Adapts to parameter interactions
- Proven in scientific computing

**Disadvantages:**
- Poor with categorical variables
- Requires more trials initially
- More complex than TPE

## 5. QMCSampler (Quasi-Monte Carlo)

**Algorithm:**
- Uses quasi-random sequences (Sobol, Halton)
- Better coverage than pure random

```python
sampler = optuna.samplers.QMCSampler(
    qmc_type='sobol',        # 'sobol' or 'halton'
    seed=42,
    scramble=True
)
```

**When to use:**
- Want better coverage than random
- Low-dimensional spaces (< 10 dimensions)
- Quick exploration

**Advantages:**
- Better coverage than random
- Deterministic with seed
- Fast

**Disadvantages:**
- Doesn't learn from trials
- Not as good as TPE for most problems

## Choosing a Sampler

### Decision Tree

```
Search space size?
├─ Small (< 100 combinations)
│  └─ GridSampler
└─ Large
   ├─ All continuous parameters?
   │  └─ CmaEsSampler
   └─ Mixed parameters?
      ├─ Default choice → TPESampler
      ├─ Baseline → RandomSampler
      └─ Better random → QMCSampler
```

### Practical Recommendations

**Default (recommended):**
```python
sampler = optuna.samplers.TPESampler(seed=42)
```

**Large search space:**
```python
sampler = optuna.samplers.TPESampler(
    n_startup_trials=50,
    multivariate=True,
    seed=42
)
```

**Small grid search:**
```python
search_space = {
    'lr': [1e-4, 1e-3, 1e-2],
    'batch_size': [32, 64],
    'n_layers': [2, 3, 4]
}
sampler = optuna.samplers.GridSampler(search_space)
```

## Combining Pruner and Sampler

**Conservative approach:**
```python
study = optuna.create_study(
    direction='maximize',
    sampler=optuna.samplers.TPESampler(n_startup_trials=10),
    pruner=optuna.pruners.MedianPruner(n_startup_trials=5)
)
```

**Aggressive approach:**
```python
study = optuna.create_study(
    direction='maximize',
    sampler=optuna.samplers.TPESampler(n_startup_trials=20),
    pruner=optuna.pruners.PercentilePruner(percentile=15.0)
)
```

**Very aggressive (limited budget):**
```python
study = optuna.create_study(
    direction='maximize',
    sampler=optuna.samplers.TPESampler(multivariate=True),
    pruner=optuna.pruners.HyperbandPruner()
)
```

## Advanced: Multivariate TPE

Regular TPE treats hyperparameters independently. Multivariate TPE models their interactions.

```python
# Independent (default)
sampler = optuna.samplers.TPESampler(multivariate=False)

# Joint modeling
sampler = optuna.samplers.TPESampler(multivariate=True)
```

**Use multivariate=True when:**
- Hyperparameters interact (e.g., lr and batch_size)
- Large search space
- Enough trials (50+)

**Use multivariate=False when:**
- Hyperparameters are independent
- Small number of trials
- Simple problems

## Performance Comparison

### Sample Efficiency (trials to good solution)
1. GridSampler (small space): Excellent
2. TPESampler: Very Good
3. CmaEsSampler: Good
4. QMCSampler: Fair
5. RandomSampler: Poor

### Wall Clock Time (per trial)
1. RandomSampler: Fastest
2. QMCSampler: Very Fast
3. TPESampler: Fast
4. CmaEsSampler: Medium
5. GridSampler: Slow (many trials)

### Scalability (large search spaces)
1. TPESampler: Excellent
2. CmaEsSampler: Good
3. RandomSampler: Fair
4. QMCSampler: Fair
5. GridSampler: Poor

## Debugging Tips

### Pruner not working?

Check:
1. Are you calling `trial.report()`?
2. Is `n_startup_trials` too high?
3. Are all trials performing similarly?

### Sampler not improving?

Check:
1. Is `n_startup_trials` too low?
2. Is search space too large?
3. Is objective function noisy?
4. Are you running enough trials?

### Getting same results repeatedly?

Solution: Set different seeds
```python
sampler = optuna.samplers.TPESampler(seed=42)
```

## Summary Table

| Pruner | Aggressiveness | Complexity | Use Case |
|--------|---------------|------------|----------|
| MedianPruner | Medium | Low | General purpose |
| PercentilePruner | Variable | Low | Tunable aggressiveness |
| SuccessiveHalvingPruner | High | Medium | Limited budget |
| HyperbandPruner | Very High | High | Very limited budget |
| NopPruner | None | Low | Baseline/cheap trials |

| Sampler | Sample Efficiency | Speed | Use Case |
|---------|------------------|-------|----------|
| TPESampler | Very Good | Fast | General purpose (DEFAULT) |
| RandomSampler | Poor | Fastest | Baseline |
| GridSampler | Excellent* | Slow | Small space |
| CmaEsSampler | Good | Medium | Continuous only |
| QMCSampler | Fair | Very Fast | Quick exploration |

*Only if search space is small enough
