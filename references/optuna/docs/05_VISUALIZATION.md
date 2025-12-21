# Visualization Guide

Optuna provides powerful built-in visualization tools to understand your optimization results. All visualizations are interactive and can be saved as images.

## Setup

```python
import optuna
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_parallel_coordinate,
    plot_slice,
    plot_contour,
    plot_edf,
    plot_intermediate_values
)
```

## Core Visualizations

### 1. Optimization History

Shows how the best value improves over trials.

```python
fig = optuna.visualization.plot_optimization_history(study)
fig.show()
fig.write_image('optimization_history.png')
```

**What it shows:**
- X-axis: Trial number
- Y-axis: Objective value
- Each point: One trial
- Line: Best value so far

**Interpretation:**
- **Upward trend** (maximize): Optimization is working
- **Flat line**: May need more trials or different search space
- **Noisy**: High variance in results, consider more stable training

**Use cases:**
- Monitor optimization progress
- Decide when to stop optimization
- Compare different studies

**Example:**
```python
study1 = optuna.create_study()
study1.optimize(objective1, n_trials=100)

study2 = optuna.create_study()
study2.optimize(objective2, n_trials=100)

# Compare both
fig = plot_optimization_history([study1, study2])
fig.show()
```

### 2. Parameter Importances

Shows which hyperparameters matter most.

```python
fig = optuna.visualization.plot_param_importances(study)
fig.show()
fig.write_image('param_importances.png')
```

**What it shows:**
- Ranking of hyperparameters by importance
- Based on fANOVA (functional ANOVA)

**Interpretation:**
- **High importance**: Focus optimization on these
- **Low importance**: Can use fixed values
- **Zero importance**: Remove from search space

**Use cases:**
- Simplify search space
- Understand what matters
- Guide future optimizations

**Example interpretation:**
```
lr:              ████████████ 0.45
n_layers:        ████████     0.32
dropout:         ███          0.15
batch_size:      ██           0.08
```
→ Focus on learning rate and n_layers, batch_size matters less

**Advanced:**
```python
# Custom evaluator
from optuna.importance import FanovaImportanceEvaluator, MeanDecreaseImpurityImportanceEvaluator

fig = plot_param_importances(
    study,
    evaluator=MeanDecreaseImpurityImportanceEvaluator()
)
```

### 3. Parallel Coordinate Plot

Visualizes relationships between hyperparameters and objective.

```python
fig = optuna.visualization.plot_parallel_coordinate(study)
fig.show()
fig.write_image('parallel_coordinate.png')
```

**What it shows:**
- Each line: One trial
- Each vertical axis: One hyperparameter or objective
- Color: Objective value (red=good, blue=bad)

**Interpretation:**
- **Red lines cluster**: Good hyperparameter region
- **Line patterns**: Relationships between parameters
- **Wide spread**: Parameter doesn't matter much

**Use cases:**
- Find good hyperparameter combinations
- Identify parameter interactions
- Understand search space structure

**Customization:**
```python
# Only show specific parameters
fig = plot_parallel_coordinate(
    study,
    params=['lr', 'n_layers', 'dropout']
)

# Highlight top 10 trials
fig = plot_parallel_coordinate(
    study,
    target=lambda t: t.values[0],
    target_name='accuracy'
)
```

### 4. Slice Plot

Shows marginal effects of each hyperparameter.

```python
fig = optuna.visualization.plot_slice(study)
fig.show()
fig.write_image('slice_plot.png')
```

**What it shows:**
- One subplot per hyperparameter
- How objective changes with that parameter
- Other parameters marginalized out

**Interpretation:**
- **Clear trend**: Strong effect
- **Flat**: No clear effect
- **Optimal region**: Best values for parameter

**Use cases:**
- Understand individual parameter effects
- Find optimal ranges
- Validate search space bounds

**Example:**
```python
# Focus on specific parameters
fig = plot_slice(study, params=['lr', 'n_layers'])
fig.show()
```

### 5. Contour Plot

Shows 2D interactions between hyperparameters.

```python
fig = optuna.visualization.plot_contour(study)
fig.show()
fig.write_image('contour_plot.png')
```

**What it shows:**
- 2D heatmap for each pair of parameters
- Color intensity: Objective value

**Interpretation:**
- **Red region**: Best values
- **Blue region**: Worst values
- **Diagonal patterns**: Parameters interact

**Use cases:**
- Find parameter interactions
- Discover optimal combinations
- Understand 2D search space

**Example:**
```python
# Specific parameter pairs
fig = plot_contour(study, params=['lr', 'batch_size'])
fig.show()
```

### 6. Empirical Distribution Function (EDF)

Compares the distribution of objective values across studies.

```python
fig = optuna.visualization.plot_edf([study1, study2])
fig.show()
fig.write_image('edf_plot.png')
```

**What it shows:**
- Cumulative distribution of objective values
- One line per study

**Interpretation:**
- **Line to the right**: Better overall performance
- **Steeper line**: More consistent results
- **Crossing lines**: Different trade-offs

**Use cases:**
- Compare different algorithms
- Statistical comparison of studies
- Reproducibility analysis

### 7. Intermediate Values

Shows learning curves for trials.

```python
fig = optuna.visualization.plot_intermediate_values(study)
fig.show()
fig.write_image('intermediate_values.png')
```

**What it shows:**
- Training curves (e.g., accuracy over epochs)
- One line per trial
- Pruned trials marked

**Interpretation:**
- **Pruned trials**: Stop early
- **Continuing trials**: Complete all epochs
- **Convergence patterns**: Training dynamics

**Use cases:**
- Verify pruning is working
- Understand training dynamics
- Debug training issues

**Requirements:**
- Must call `trial.report()` in objective function

## Saving Visualizations

### As Images

```python
fig = plot_optimization_history(study)

# PNG (requires kaleido)
fig.write_image('optimization_history.png')

# SVG (vector graphics)
fig.write_image('optimization_history.svg')

# PDF
fig.write_image('optimization_history.pdf')
```

### As HTML

```python
fig = plot_optimization_history(study)
fig.write_html('optimization_history.html')
```

**Benefits:**
- Interactive
- Can zoom/pan
- Hover for details

### Installation for Image Export

```bash
pip install kaleido
```

## Complete Visualization Workflow

```python
import optuna
from pathlib import Path

# Run optimization
study = optuna.create_study(
    direction='maximize',
    storage='sqlite:///results/optuna.db',
    study_name='cnn_optimization',
    load_if_exists=True
)
study.optimize(objective, n_trials=100)

# Create output directory
output_dir = Path('results')
output_dir.mkdir(exist_ok=True)

# Generate all visualizations
visualizations = {
    'optimization_history': optuna.visualization.plot_optimization_history,
    'param_importances': optuna.visualization.plot_param_importances,
    'parallel_coordinate': optuna.visualization.plot_parallel_coordinate,
    'slice': optuna.visualization.plot_slice,
    'contour': optuna.visualization.plot_contour,
    'intermediate_values': optuna.visualization.plot_intermediate_values
}

for name, plot_func in visualizations.items():
    try:
        fig = plot_func(study)
        fig.write_image(str(output_dir / f'{name}.png'))
        fig.write_html(str(output_dir / f'{name}.html'))
        print(f"Saved {name}")
    except Exception as e:
        print(f"Failed to create {name}: {e}")
```

## Interpretation Guide

### Successful Optimization

**Optimization History:**
- Clear upward/downward trend
- Stabilizes after 50-100 trials
- Final values close to theoretical optimum

**Parameter Importances:**
- Clear ranking
- Top 2-3 parameters dominate
- Low-importance parameters identified

**Parallel Coordinate:**
- Red lines cluster in specific regions
- Clear patterns visible
- Good and bad regions separated

### Problematic Optimization

**Signs of Issues:**

1. **Flat optimization history**
   - Problem: Not finding better solutions
   - Solution: Adjust search space or increase trials

2. **All parameters equally important**
   - Problem: Search space may be wrong
   - Solution: Reconsider hyperparameters

3. **Random parallel coordinate**
   - Problem: No clear patterns
   - Solution: Objective may be too noisy

4. **All trials pruned**
   - Problem: Pruner too aggressive
   - Solution: Adjust pruner settings

## Advanced Customization

### Custom Plots with Plotly

```python
import plotly.graph_objects as go

# Get trial data
df = study.trials_dataframe()

# Custom plot
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=df['number'],
    y=df['value'],
    mode='markers',
    marker=dict(
        size=10,
        color=df['params_lr'],
        colorscale='Viridis',
        showscale=True,
        colorbar=dict(title='Learning Rate')
    )
))
fig.update_layout(
    title='Trials colored by Learning Rate',
    xaxis_title='Trial Number',
    yaxis_title='Accuracy'
)
fig.show()
```

### Multiple Studies Comparison

```python
# Run multiple studies
studies = []
for seed in [1, 2, 3]:
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)
    studies.append(study)

# Compare
fig = optuna.visualization.plot_edf(studies)
fig.show()
```

### Custom Parameter Selection

```python
# Only visualize important parameters
important_params = ['lr', 'n_layers', 'dropout']

fig1 = plot_parallel_coordinate(study, params=important_params)
fig2 = plot_slice(study, params=important_params)
fig3 = plot_contour(study, params=['lr', 'n_layers'])
```

## Matplotlib Alternative

For environments without Plotly:

```python
from optuna.visualization.matplotlib import (
    plot_optimization_history,
    plot_param_importances,
    plot_parallel_coordinate
)

import matplotlib.pyplot as plt

fig = plot_optimization_history(study)
plt.savefig('optimization_history.png')
plt.close()
```

## Dashboard (Optuna-Dashboard)

For real-time monitoring:

```bash
pip install optuna-dashboard
optuna-dashboard sqlite:///results/optuna.db
```

Access at `http://localhost:8080`

**Features:**
- Real-time updates
- All visualizations in one place
- Study management
- Trial inspection

## Common Visualization Workflows

### During Optimization

```python
# Monitor progress every 10 trials
def callback(study, trial):
    if trial.number % 10 == 0:
        fig = plot_optimization_history(study)
        fig.write_image(f'progress_{trial.number}.png')

study.optimize(objective, n_trials=100, callbacks=[callback])
```

### After Optimization

```python
# Comprehensive report
study = optuna.load_study(
    study_name='cnn_optimization',
    storage='sqlite:///results/optuna.db'
)

print(f"Best value: {study.best_value}")
print(f"Best params: {study.best_params}")

# Create all visualizations
for name, func in [
    ('history', plot_optimization_history),
    ('importance', plot_param_importances),
    ('parallel', plot_parallel_coordinate),
    ('slice', plot_slice)
]:
    fig = func(study)
    fig.write_image(f'results/{name}.png')
```

### Comparison Study

```python
# Compare different configurations
baseline_study = create_and_run_study('baseline')
improved_study = create_and_run_study('improved')

# Compare
fig = plot_edf([baseline_study, improved_study])
fig.show()
```

## Best Practices

1. **Always create optimization history plot**
   - First thing to check
   - Shows if optimization is working

2. **Use parameter importance early**
   - After 20-30 trials
   - Guides further optimization

3. **Save visualizations automatically**
   - Don't rely on manual saving
   - Include in your pipeline

4. **Use HTML for exploration**
   - Interactive is better for analysis
   - PNG for reports/papers

5. **Create comprehensive reports**
   - Multiple visualizations together
   - Tell the full story

6. **Monitor during long runs**
   - Use callbacks or dashboard
   - Catch issues early

7. **Compare across studies**
   - Use EDF plots
   - Statistical validation

## Troubleshooting

**Problem:** Plots not showing
```python
# Solution: Use show() or save to file
fig = plot_optimization_history(study)
fig.show()  # or
fig.write_image('plot.png')
```

**Problem:** Image export fails
```bash
# Solution: Install kaleido
pip install kaleido
```

**Problem:** Too many parameters in parallel coordinate
```python
# Solution: Select subset
fig = plot_parallel_coordinate(study, params=['lr', 'n_layers', 'dropout'])
```

**Problem:** Contour plot is empty
```python
# Solution: Ensure enough trials (20+) and correct parameter names
print(study.best_params.keys())  # Check parameter names
```

## Summary

| Plot | Purpose | Best For |
|------|---------|----------|
| Optimization History | Monitor progress | Always use |
| Parameter Importances | Identify key parameters | After 30+ trials |
| Parallel Coordinate | Find good combinations | Understanding relationships |
| Slice | Individual effects | Simple interpretation |
| Contour | 2D interactions | Parameter pairs |
| EDF | Compare studies | Statistical comparison |
| Intermediate Values | Training dynamics | With pruning |

**Recommended minimum set:**
1. Optimization History
2. Parameter Importances
3. Parallel Coordinate

These three plots cover most analysis needs.
