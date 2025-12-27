# Attention Mechanism Mathematics

## Scaled Dot-Product Attention

### Definition

Given queries Q, keys K, and values V:

```
Attention(Q, K, V) = softmax(QK^T / √d_k)V
```

### Mathematical Derivation

#### Step 1: Similarity Computation

Compute unnormalized attention scores between queries and keys:

```
S = QK^T
```

Where:
- Q ∈ ℝ^(n×d_k): Query matrix
- K ∈ ℝ^(m×d_k): Key matrix  
- S ∈ ℝ^(n×m): Similarity scores

#### Step 2: Scaling

Apply temperature scaling to prevent saturation of softmax:

```
S_scaled = S / √d_k
```

Rationale: When d_k is large, dot products grow large in magnitude, pushing softmax into regions with small gradients.

#### Step 3: Normalization

Apply softmax row-wise to obtain attention weights:

```
A = softmax(S_scaled) = exp(S_scaled) / Σ_j exp(S_scaled_ij)
```

Where:
- A ∈ ℝ^(n×m): Attention weight matrix
- Each row sums to 1
- A_ij represents importance of key j for query i

#### Step 4: Weighted Aggregation

Compute weighted sum of values:

```
Output = AV
```

Where:
- V ∈ ℝ^(m×d_v): Value matrix
- Output ∈ ℝ^(n×d_v): Final output

### Complexity Analysis

#### Time Complexity

- QK^T computation: O(n·m·d_k)
- Softmax: O(n·m)
- AV multiplication: O(n·m·d_v)
- Total: O(n·m·(d_k + d_v))

#### Space Complexity

- Attention matrix A: O(n·m)
- Intermediate results: O(n·d_k + m·d_k + m·d_v)
- Total: O(n·m + n·d + m·d) where d = max(d_k, d_v)

### Gradient Analysis

#### Backward Pass

Gradient with respect to values:
```
∂L/∂V = A^T · ∂L/∂Output
```

Gradient with respect to attention weights:
```
∂L/∂A = (∂L/∂Output) · V^T
```

Gradient with respect to queries and keys requires chain rule through softmax.

## Multi-Head Attention

### Definition

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O

where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
```

### Linear Projections

For each head i:
- W_i^Q ∈ ℝ^(d_model×d_k)
- W_i^K ∈ ℝ^(d_model×d_k)
- W_i^V ∈ ℝ^(d_model×d_v)

Output projection:
- W^O ∈ ℝ^(h·d_v×d_model)

### Head Dimension

Typically:
```
d_k = d_v = d_model / h
```

This ensures total computational cost is similar to single-head attention with full dimensionality.

### Computational Flow

1. Project inputs to h subspaces:
   ```
   Q_i = QW_i^Q, K_i = KW_i^K, V_i = VW_i^V
   ```

2. Apply scaled dot-product attention in parallel:
   ```
   head_i = Attention(Q_i, K_i, V_i)
   ```

3. Concatenate heads:
   ```
   Concat = [head_1; head_2; ...; head_h]
   ```

4. Final linear projection:
   ```
   Output = Concat · W^O
   ```

### Advantages

1. **Multiple Representation Subspaces**: Each head can attend to different aspects
2. **Increased Capacity**: h parallel attention mechanisms
3. **Computational Efficiency**: Similar cost to single full-dimensional attention

## Self-Attention vs Cross-Attention

### Self-Attention

Q, K, V derived from same input:
```
Q = K = V = X
```

Used in:
- Encoder layers
- Decoder self-attention (with masking)

### Cross-Attention

Q from one source, K and V from another:
```
Q = X_1
K = V = X_2
```

Used in:
- Decoder attending to encoder output

## Masking

### Padding Mask

Prevent attention to padding tokens:
```
M_pad[i,j] = 1 if j is valid token
M_pad[i,j] = 0 if j is padding
```

Apply before softmax:
```
S_masked = S + (1 - M_pad) · (-∞)
```

### Causal Mask

Prevent attention to future positions:
```
M_causal[i,j] = 1 if j ≤ i
M_causal[i,j] = 0 if j > i
```

Results in lower triangular attention matrix.

### Combined Mask

```
M_combined = M_pad ⊙ M_causal
```

Where ⊙ is element-wise AND.

## Implementation Considerations

### Numerical Stability

1. **Scaling**: Division by √d_k prevents large values
2. **Softmax Stability**: Subtract max before exponentiation
3. **Gradient Clipping**: Prevent exploding gradients

### Memory Optimization

1. **Chunked Attention**: Process in blocks for long sequences
2. **Gradient Checkpointing**: Trade compute for memory
3. **Mixed Precision**: Use FP16 for attention scores

### Parallelization

1. **Batch Parallelism**: Process multiple sequences simultaneously
2. **Head Parallelism**: Compute all heads in parallel
3. **Sequence Parallelism**: For very long sequences

## Attention Patterns

### Encoder Self-Attention

- Fully bidirectional
- Each position attends to all positions
- Mask only padding tokens

### Decoder Self-Attention

- Causal/autoregressive
- Position i attends only to positions ≤ i
- Prevents information leakage from future

### Decoder Cross-Attention

- Decoder queries attend to encoder output
- Bidirectional over encoder sequence
- Enables conditioning on source

## Theoretical Properties

### Permutation Invariance

Without positional encoding, attention is permutation-invariant:
```
Attention(P·Q, P·K, P·V) = P·Attention(Q, K, V)
```

Where P is any permutation matrix.

### Universal Approximation

Multi-head attention with sufficient heads and dimensions can approximate any continuous function on compact sets.

### Inductive Bias

Attention has minimal inductive bias:
- No assumption about locality
- No assumption about hierarchy
- Must learn structure from data

## Optimization

### Gradient Flow

Attention provides direct paths between all positions:
- Shortest path length: O(1)
- Compared to O(n) for RNNs
- Enables better gradient propagation

### Learning Dynamics

Early training:
- Attention weights often uniform
- Gradual specialization

Late training:
- Sharp attention patterns emerge
- Task-specific structures learned
