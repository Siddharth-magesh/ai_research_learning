# Tensor Shape Tracking

## Overview

This document tracks tensor dimensions through the Transformer architecture to ensure dimensional consistency and aid debugging.

## Notation

- B: Batch size
- S_src: Source sequence length
- S_tgt: Target sequence length
- V: Vocabulary size
- D: Model dimension (embedding dimension)
- H: Number of attention heads
- D_h: Head dimension (D / H)
- D_ff: Feed-forward hidden dimension (typically 4D)

## Input Processing

### Token IDs

```
Input: (B, S)
Type: torch.LongTensor
Range: [0, V)
```

Example: B=32, S=128, V=32000
```
src_ids: (32, 128)
tgt_ids: (32, 100)
```

### Token Embedding

```
Input:  (B, S)
Weight: (V, D)
Output: (B, S, D)
```

Operation:
```
embedded = embedding_table[input_ids]
embedded = embedded * sqrt(D)
```

Example: D=512
```
src_ids:      (32, 128)
src_embedded: (32, 128, 512)
```

### Positional Encoding

```
Positional Encoding Matrix: (1, S_max, D)
Broadcasted to: (B, S, D)
Added to embeddings: (B, S, D)
```

Example:
```
pe:           (1, 512, 512)
embedded:     (32, 128, 512)
pe_broadcast: (32, 128, 512)
output:       (32, 128, 512)
```

## Encoder

### Encoder Layer Input

```
Input: (B, S, D)
```

### Multi-Head Attention

#### Linear Projections

```
Input:  (B, S, D)
W_Q:    (D, D)
W_K:    (D, D)
W_V:    (D, D)
Q, K, V: (B, S, D)
```

#### Reshape for Multiple Heads

```
Q: (B, S, D) → (B, H, S, D_h)
K: (B, S, D) → (B, H, S, D_h)
V: (B, S, D) → (B, H, S, D_h)
```

Example: H=8, D_h=64
```
Q before: (32, 128, 512)
Q after:  (32, 8, 128, 64)
```

#### Attention Scores

```
Q:      (B, H, S, D_h)
K^T:    (B, H, D_h, S)
Scores: (B, H, S, S)
```

Operation:
```
scores = Q @ K.transpose(-2, -1) / sqrt(D_h)
```

Example:
```
Q:      (32, 8, 128, 64)
K^T:    (32, 8, 64, 128)
scores: (32, 8, 128, 128)
```

#### Attention Weights

```
Scores:  (B, H, S, S)
Weights: (B, H, S, S)
```

Operation:
```
weights = softmax(scores, dim=-1)
```

#### Attention Output

```
Weights: (B, H, S, S)
V:       (B, H, S, D_h)
Context: (B, H, S, D_h)
```

Operation:
```
context = weights @ V
```

#### Concatenate Heads

```
Context:     (B, H, S, D_h) → (B, S, H*D_h) → (B, S, D)
```

Example:
```
Context:     (32, 8, 128, 64)
Transpose:   (32, 128, 8, 64)
Reshape:     (32, 128, 512)
```

#### Output Projection

```
Input:  (B, S, D)
W_O:    (D, D)
Output: (B, S, D)
```

### Feed-Forward Network

```
Input:   (B, S, D)
W_1:     (D, D_ff)
Hidden:  (B, S, D_ff)
W_2:     (D_ff, D)
Output:  (B, S, D)
```

Example: D_ff=2048
```
Input:   (32, 128, 512)
Hidden:  (32, 128, 2048)
Output:  (32, 128, 512)
```

### Encoder Stack Output

```
Input:  (B, S_src, D)
Output: (B, S_src, D)
```

After N=6 layers:
```
encoder_output: (32, 128, 512)
```

## Decoder

### Decoder Layer Input

```
Decoder Input:  (B, S_tgt, D)
Encoder Output: (B, S_src, D)
```

### Masked Self-Attention

```
Input: (B, S_tgt, D)
Q, K, V from same input
Output: (B, S_tgt, D)
```

Causal mask:
```
Mask: (S_tgt, S_tgt)
```

Lower triangular:
```
[[1, 0, 0],
 [1, 1, 0],
 [1, 1, 1]]
```

### Cross-Attention

```
Query:  (B, S_tgt, D) from decoder
Key:    (B, S_src, D) from encoder
Value:  (B, S_src, D) from encoder
Output: (B, S_tgt, D)
```

Attention scores:
```
Q:      (B, H, S_tgt, D_h)
K^T:    (B, H, D_h, S_src)
Scores: (B, H, S_tgt, S_src)
```

Example:
```
Q:      (32, 8, 100, 64)
K^T:    (32, 8, 64, 128)
scores: (32, 8, 100, 128)
```

### Decoder Feed-Forward

Same as encoder:
```
(B, S_tgt, D) → (B, S_tgt, D_ff) → (B, S_tgt, D)
```

### Decoder Stack Output

```
Input:  (B, S_tgt, D)
Output: (B, S_tgt, D)
```

## Output Projection

```
Decoder Output: (B, S_tgt, D)
W_out:          (D, V)
Logits:         (B, S_tgt, V)
```

Example:
```
Decoder output: (32, 100, 512)
Logits:         (32, 100, 32000)
```

### Softmax

```
Logits:        (B, S_tgt, V)
Probabilities: (B, S_tgt, V)
```

Applied over vocabulary dimension:
```
probs = softmax(logits, dim=-1)
```

## Masking Shapes

### Padding Mask

```
Attention Mask: (B, S)
Expanded:       (B, 1, 1, S)
Broadcast to:   (B, H, S, S)
```

### Causal Mask

```
Base:      (S, S)
Expanded:  (1, 1, S, S)
Broadcast: (B, H, S, S)
```

### Combined Mask

```
Padding: (B, H, S, S)
Causal:  (B, H, S, S)
Combined: (B, H, S, S)
```

Operation:
```
combined = padding_mask & causal_mask
```

## Training Shapes

### Forward Pass

```
Input IDs:       (B, S_tgt)
Target Labels:   (B, S_tgt)
Logits:          (B, S_tgt, V)
```

### Loss Computation

```
Logits:  (B, S_tgt, V)
Targets: (B, S_tgt)
Loss:    scalar
```

Reshaping for CrossEntropyLoss:
```
logits:  (B, S_tgt, V) → (B*S_tgt, V)
targets: (B, S_tgt)    → (B*S_tgt,)
```

### Gradient Shapes

Gradients have same shape as parameters:
```
W_Q gradient:     (D, D)
W_embedding grad: (V, D)
```

## Inference Shapes

### Autoregressive Generation

Initial:
```
input_ids: (B, 1)
```

Step t:
```
input_ids: (B, t)
logits:    (B, t, V)
next_token: (B, 1)
```

Concatenation:
```
input_ids: (B, t) + (B, 1) → (B, t+1)
```

### Beam Search

```
Initial:
input_ids: (B, 1)

After beam expansion:
input_ids: (B*beam_size, 1)

Step t:
input_ids: (B*beam_size, t)
```

## Memory Analysis

### Attention Memory

Attention scores for single head:
```
Memory = B * S * S * sizeof(float)
```

Example: B=32, S=512, float32
```
Memory = 32 * 512 * 512 * 4 bytes = 33.5 MB
```

All heads:
```
Total = H * B * S * S * 4 bytes
```

### Activation Memory

Per layer activations:
```
Attention: B * S * D
FFN:       B * S * D_ff
```

Total for N layers:
```
Memory ≈ N * B * S * (D + D_ff) * 4 bytes
```

### Parameter Memory

Single attention layer:
```
W_Q, W_K, W_V, W_O: 4 * D * D
```

FFN:
```
W_1, W_2: D * D_ff + D_ff * D = 2 * D * D_ff
```

## Common Errors

### Shape Mismatch in Attention

Error:
```
RuntimeError: mat1 and mat2 shapes cannot be multiplied
```

Check:
- Q, K dimensions match for Q @ K^T
- Proper transpose of K
- Head dimension consistency

### Batch Dimension Mismatch

Error:
```
RuntimeError: The size of tensor a (32) must match the size of tensor b (16)
```

Check:
- Batch sizes consistent throughout
- Proper broadcasting

### Sequence Length Issues

Error:
```
IndexError: index out of range
```

Check:
- Positional encoding max length
- Mask dimensions
- Sequence length limits
