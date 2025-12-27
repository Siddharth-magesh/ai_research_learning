# Transformer Architecture Documentation

## Overview

This document provides a detailed explanation of the Transformer architecture implemented in this project, based on "Attention Is All You Need" (Vaswani et al., 2017).

## High-Level Architecture

```
                    ┌─────────────────────┐
                    │   Input Sequence    │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  Token Embedding    │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │ Positional Encoding │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │   Encoder Stack     │
                    │  (N=6 layers)       │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │   Decoder Stack     │
                    │  (N=6 layers)       │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  Linear + Softmax   │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  Output Probabilities│
                    └─────────────────────┘
```

## Core Components

### 1. Embeddings

#### Token Embedding
- Converts token IDs to dense vectors
- Dimension: `vocab_size → embed_dim`
- Scaled by √(embed_dim) for proper gradient flow

```python
embedding = nn.Embedding(vocab_size, embed_dim)
output = embedding(tokens) * sqrt(embed_dim)
```

#### Positional Encoding
- Adds position information to embeddings
- Uses sinusoidal functions:
  - PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
  - PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
- Allows model to learn relative positions

### 2. Multi-Head Attention

The core innovation of the Transformer architecture.

#### Scaled Dot-Product Attention

```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

**Steps:**
1. Compute attention scores: `QK^T`
2. Scale by √d_k to prevent large values
3. Apply softmax to get attention weights
4. Multiply by V to get weighted values

**Shape Flow:**
- Q, K, V: `(batch, seq_len, d_k)`
- Scores: `(batch, seq_len, seq_len)`
- Output: `(batch, seq_len, d_k)`

#### Multi-Head Attention

Allows the model to attend to different representation subspaces.

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O

where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
```

**Configuration:**
- Number of heads (h): 8
- Head dimension: embed_dim / num_heads = 64
- Total parameters: 4 × embed_dim²

**Shape Flow:**
```
Input:  (batch, seq_len, embed_dim)
        ↓ Linear projections
Q,K,V:  (batch, seq_len, embed_dim)
        ↓ Split into heads
        (batch, num_heads, seq_len, head_dim)
        ↓ Scaled dot-product attention
        (batch, num_heads, seq_len, head_dim)
        ↓ Concatenate heads
Output: (batch, seq_len, embed_dim)
```

### 3. Feed-Forward Network

Position-wise fully connected feed-forward network.

```
FFN(x) = ReLU(xW_1 + b_1)W_2 + b_2
```

**Configuration:**
- Hidden dimension: 2048 (4 × embed_dim)
- Activation: GELU or ReLU
- Dropout after each layer

**Shape Flow:**
```
Input:  (batch, seq_len, 512)
        ↓ Linear 1
        (batch, seq_len, 2048)
        ↓ ReLU + Dropout
        (batch, seq_len, 2048)
        ↓ Linear 2
Output: (batch, seq_len, 512)
```

### 4. Encoder Layer

Each encoder layer contains:
1. Multi-head self-attention
2. Feed-forward network
3. Residual connections around each
4. Layer normalization

```python
# Encoder Layer
x = x + Dropout(MultiHeadAttention(LayerNorm(x)))
x = x + Dropout(FeedForward(LayerNorm(x)))
```

**Pre-Normalization** (Pre-LN):
- More stable training
- Better gradient flow
- Faster convergence

### 5. Decoder Layer

Each decoder layer contains:
1. Masked multi-head self-attention
2. Cross-attention to encoder output
3. Feed-forward network
4. Residual connections
5. Layer normalization

```python
# Decoder Layer
x = x + Dropout(MaskedAttention(LayerNorm(x)))
x = x + Dropout(CrossAttention(LayerNorm(x), encoder_output))
x = x + Dropout(FeedForward(LayerNorm(x)))
```

**Masking:**
- Causal mask prevents attending to future positions
- Padding mask ignores pad tokens

### 6. Complete Encoder Stack

```python
for layer in encoder_layers:
    x = layer(x, attention_mask)
x = LayerNorm(x)
```

**Shape Flow:**
```
Input:  (batch, src_seq_len, embed_dim)
        ↓ Layer 1-6
        (batch, src_seq_len, embed_dim)
        ↓ Final LayerNorm
Output: (batch, src_seq_len, embed_dim)
```

### 7. Complete Decoder Stack

```python
for layer in decoder_layers:
    x = layer(x, encoder_output, self_mask, cross_mask)
x = LayerNorm(x)
```

**Shape Flow:**
```
Input:  (batch, tgt_seq_len, embed_dim)
Enc:    (batch, src_seq_len, embed_dim)
        ↓ Layer 1-6
        (batch, tgt_seq_len, embed_dim)
        ↓ Final LayerNorm
Output: (batch, tgt_seq_len, embed_dim)
```

## Model Configurations

### Encoder-Decoder Transformer

Used for sequence-to-sequence tasks (translation, summarization).

```yaml
Configuration:
- Vocab Size: 32,000
- Embedding Dim: 512
- Num Layers: 6
- Num Heads: 8
- FFN Hidden: 2048
- Dropout: 0.1
- Max Seq Len: 512
```

**Total Parameters:** ~65M

### Decoder-Only Transformer (GPT-style)

Used for language modeling and text generation.

```yaml
Configuration:
- Similar to encoder-decoder
- No cross-attention
- Only causal self-attention
```

## Training Details

### Optimization

**Optimizer:** AdamW
- Learning rate: 3e-4
- Beta1: 0.9
- Beta2: 0.98
- Epsilon: 1e-8
- Weight decay: 0.01

**Learning Rate Schedule:**
```
lr = d_model^(-0.5) × min(step^(-0.5), step × warmup^(-1.5))
```

### Regularization

1. **Dropout:** 0.1
   - After attention
   - After feed-forward
   - On embeddings

2. **Layer Normalization**
   - Pre-normalization (Pre-LN)
   - Stabilizes training

3. **Gradient Clipping:** 1.0
   - Prevents exploding gradients

### Loss Function

**Cross-Entropy Loss:**
```python
loss = CrossEntropy(logits, targets, ignore_index=-100)
```

## Attention Patterns

### Self-Attention (Encoder)
- Each position attends to all positions
- Bidirectional context

### Masked Self-Attention (Decoder)
- Each position attends only to previous positions
- Causal/autoregressive

### Cross-Attention (Decoder)
- Query from decoder
- Key and Value from encoder
- Allows decoder to attend to encoder output

## Computational Complexity

### Time Complexity
- Self-Attention: O(n² × d)
- Feed-Forward: O(n × d²)

Where:
- n = sequence length
- d = model dimension

### Space Complexity
- Attention: O(n²) for attention matrix
- Model: O(layers × d²) for parameters

## Inference

### Greedy Decoding
```python
for i in range(max_length):
    logits = model(input_ids)
    next_token = argmax(logits[:, -1, :])
    input_ids = concat(input_ids, next_token)
```

### Sampling Strategies

1. **Temperature Sampling**
   - Scale logits: `logits / temperature`
   - Higher temperature = more random

2. **Top-k Sampling**
   - Keep only top k tokens
   - Sample from filtered distribution

3. **Top-p (Nucleus) Sampling**
   - Keep tokens with cumulative probability ≥ p
   - Dynamic vocabulary size

## Key Innovations

1. **Self-Attention Mechanism**
   - Replaces recurrence
   - Parallelizable
   - Long-range dependencies

2. **Multi-Head Attention**
   - Multiple representation subspaces
   - Richer representations

3. **Positional Encoding**
   - No position information in attention
   - Sinusoidal encoding is permutation-invariant

4. **Residual Connections**
   - Easier gradient flow
   - Enables deep networks

5. **Layer Normalization**
   - Stabilizes training
   - Faster convergence

## Implementation Details

### Memory Optimization
- Gradient checkpointing for large models
- Mixed precision training (FP16)
- Gradient accumulation

### Numerical Stability
- Scaled attention (÷√d_k)
- Layer normalization
- Proper initialization

### Efficiency
- Batched matrix operations
- Optimized attention kernels
- Cached key-value pairs (inference)

## References

1. Vaswani et al., "Attention Is All You Need" (2017)
2. Radford et al., "Language Models are Unsupervised Multitask Learners" (GPT-2, 2019)
3. Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers" (2019)

## Visualization

See the notebooks folder for:
- Attention weight visualizations
- Token embedding analysis
- Shape debugging utilities
