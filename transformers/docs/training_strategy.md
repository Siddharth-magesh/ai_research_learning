# Training Strategy

## Optimization Algorithm

### AdamW Optimizer

Weighted Adam with decoupled weight decay:

```
m_t = β_1 * m_{t-1} + (1 - β_1) * g_t
v_t = β_2 * v_{t-1} + (1 - β_2) * g_t^2
m̂_t = m_t / (1 - β_1^t)
v̂_t = v_t / (1 - β_2^t)
θ_t = θ_{t-1} - α * (m̂_t / (√v̂_t + ε) + λ * θ_{t-1})
```

Parameters:
- α = 3e-4 (learning rate)
- β_1 = 0.9 (first moment decay)
- β_2 = 0.98 (second moment decay)
- ε = 1e-8 (numerical stability)
- λ = 0.01 (weight decay)

### Learning Rate Scheduling

#### Transformer Schedule

Original schedule from Vaswani et al.:

```
lr = d_model^{-0.5} * min(step^{-0.5}, step * warmup^{-1.5})
```

Properties:
- Linear warmup for first W steps
- Inverse square root decay afterward
- Scale factor: d_model^{-0.5}

Typical warmup: 4000 steps

#### Linear Warmup with Decay

```
if step < warmup_steps:
    lr = base_lr * (step / warmup_steps)
else:
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    lr = base_lr * (1 - progress)
```

#### Cosine Annealing

```
if step < warmup_steps:
    lr = base_lr * (step / warmup_steps)
else:
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    lr = base_lr * 0.5 * (1 + cos(π * progress))
```

## Regularization

### Dropout

Applied at multiple locations:

1. Embedding dropout: 0.1
   ```
   x = dropout(embedding(tokens) + positional_encoding)
   ```

2. Attention dropout: 0.1
   ```
   attention_weights = dropout(softmax(scores))
   ```

3. Residual dropout: 0.1
   ```
   x = x + dropout(sublayer(x))
   ```

4. FFN dropout: 0.1
   ```
   hidden = dropout(activation(W_1 * x))
   output = W_2 * hidden
   ```

### Label Smoothing

Replace hard targets with smoothed distribution:

```
y_smoothed = (1 - ε) * y_hard + ε / V
```

Typical ε = 0.1

Benefits:
- Prevents overconfidence
- Improves generalization
- Reduces overfitting

### Weight Decay

Decoupled weight decay (AdamW):

```
θ_t = θ_{t-1} - λ * θ_{t-1}
```

λ = 0.01 for most parameters

Exceptions (no weight decay):
- Bias terms
- LayerNorm parameters
- Positional encodings

### Gradient Clipping

Clip gradient norm to prevent explosion:

```
if ||g|| > threshold:
    g = g * (threshold / ||g||)
```

Typical threshold: 1.0

## Initialization

### Xavier/Glorot Initialization

For linear layers:

```
W ~ U(-√(6/(n_in + n_out)), √(6/(n_in + n_out)))
```

For attention and FFN weights.

### Embedding Initialization

```
E ~ N(0, d_model^{-0.5})
```

### Bias Initialization

```
b = 0
```

### LayerNorm Initialization

```
γ = 1 (gain)
β = 0 (bias)
```

## Training Techniques

### Mixed Precision Training

Use FP16 for forward and backward:

```
with autocast():
    logits = model(inputs)
    loss = criterion(logits, targets)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

Benefits:
- 2-3x speedup
- Reduced memory usage
- Maintains FP32 master weights

### Gradient Accumulation

Simulate larger batch sizes:

```
for i, batch in enumerate(dataloader):
    loss = model(batch) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

Effective batch size = batch_size × accumulation_steps

### Gradient Checkpointing

Trade compute for memory:

```
def forward(x):
    return checkpoint(layer, x)
```

Recomputes activations during backward pass.

## Batch Construction

### Dynamic Batching

Group sequences by length:

```
batches = group_by_length(sequences, tolerance=10)
```

Minimizes padding overhead.

### Token-Based Batching

Limit by total tokens rather than sequences:

```
max_tokens_per_batch = 4096
batch_size = max_tokens_per_batch // sequence_length
```

### Padding Strategy

Pad to nearest multiple:

```
padded_length = ((length + pad_multiple - 1) // pad_multiple) * pad_multiple
```

Reduces number of unique shapes for optimization.

## Loss Computation

### Cross-Entropy Loss

```
L = -Σ_t log P(y_t | y_{<t}, x)
```

Implementation:
```
loss = F.cross_entropy(
    logits.view(-1, vocab_size),
    targets.view(-1),
    ignore_index=pad_id,
    label_smoothing=0.1
)
```

### Perplexity

```
PPL = exp(L)
```

Lower is better, measures prediction uncertainty.

## Training Monitoring

### Metrics to Track

1. Training loss (per step)
2. Validation loss (per epoch)
3. Perplexity
4. Token accuracy
5. Learning rate
6. Gradient norm
7. Parameter norm

### Early Stopping

Criteria:
```
if val_loss_patience > max_patience:
    stop_training()
```

Typical patience: 3-5 epochs

### Checkpointing

Save:
- Model state dict
- Optimizer state dict
- Scheduler state
- Epoch number
- Best validation loss
- Random state

## Training Schedule

### Warmup Phase

Steps: 0 to W (typically 4000)
- Gradual learning rate increase
- Stabilizes initial training
- Prevents early divergence

### Main Training

Steps: W to T
- Learning rate decay
- Standard optimization
- Regular validation

### Fine-tuning Phase

Optional:
- Lower learning rate (1e-5)
- Reduced dropout (0.05)
- Task-specific adaptation

## Convergence Criteria

### Loss-Based

```
if |loss_t - loss_{t-1}| < threshold:
    converged = True
```

Threshold: 1e-4

### Validation-Based

```
if val_loss increases for N epochs:
    converged = True
```

N = 5 epochs

### Time-Based

```
if steps > max_steps:
    stop_training()
```

## Hyperparameter Tuning

### Critical Hyperparameters

1. Learning rate: [1e-5, 1e-3]
2. Warmup steps: [500, 10000]
3. Dropout: [0.0, 0.3]
4. Weight decay: [0.0, 0.1]
5. Batch size: [8, 128]

### Search Strategy

Grid search over:
```
lr: [1e-4, 3e-4, 1e-3]
warmup: [2000, 4000, 8000]
dropout: [0.1, 0.2]
```

Or random search for efficiency.

## Troubleshooting

### Loss Diverges

1. Reduce learning rate
2. Increase warmup steps
3. Enable gradient clipping
4. Check for NaN/Inf values

### Slow Convergence

1. Increase learning rate
2. Reduce warmup
3. Check data preprocessing
4. Verify batch size

### Overfitting

1. Increase dropout
2. Add weight decay
3. Use label smoothing
4. Reduce model capacity
5. Augment training data

### Memory Issues

1. Reduce batch size
2. Enable gradient accumulation
3. Use mixed precision
4. Enable gradient checkpointing
5. Reduce sequence length

## Best Practices

1. Always use warmup with Transformers
2. Monitor gradient norms
3. Save checkpoints frequently
4. Validate on held-out set
5. Use learning rate finder
6. Track multiple metrics
7. Log hyperparameters
8. Maintain reproducibility
9. Profile memory and compute
10. Use distributed training for large models
