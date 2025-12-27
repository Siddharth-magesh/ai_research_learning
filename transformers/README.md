# Transformer Implementation

Production-grade implementation of the Transformer architecture from "Attention Is All You Need" (Vaswani et al., 2017), built from scratch using PyTorch.

## Features

### Complete Transformer Implementation
- Encoder-decoder architecture
- Decoder-only architecture (GPT-style)
- Multi-head self-attention mechanism
- Sinusoidal positional encoding
- Position-wise feed-forward networks

### Training Infrastructure
- Mixed precision training (AMP)
- Gradient accumulation
- Learning rate scheduling (warmup, cosine, linear)
- Model checkpointing and resumption
- Comprehensive logging and monitoring

### Evaluation and Inference
- Model evaluation with metrics (loss, accuracy, perplexity)
- Text generation with multiple sampling strategies
- Beam search for sequence-to-sequence tasks
- Top-k and nucleus (top-p) sampling

### Code Quality
- Modular architecture
- Type annotations
- Unit test coverage
- Production-ready codebase

## Project Structure

```
transformers/
├── src/
│   ├── config/              # Configuration files
│   ├── data/                # Dataset loading and preprocessing
│   ├── models/              # Transformer models
│   ├── modules/             # Building blocks (attention, FFN, etc.)
│   ├── optim/               # Optimizers and schedulers
│   ├── utils/               # Utility functions
│   ├── tests/               # Unit tests
│   ├── main.py              # Main entry point
│   ├── train.py             # Training loop
│   ├── evaluate.py          # Evaluation logic
│   └── inference.py         # Text generation
├── docs/                    # Documentation
├── experiments/             # Experiment configurations
├── requirements.txt         # Dependencies
└── README.md
```

## Installation

```bash
# Clone the repository
cd transformers

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Training

```bash
python -m src.main \
    --mode train \
    --transformer-config src/config/transformer.yaml \
    --dataset-config src/config/dataset.yaml \
    --train-config src/config/train.yaml
```

### Evaluation

```bash
python -m src.main \
    --mode eval \
    --checkpoint ./checkpoints/best_model.pth \
    --transformer-config src/config/transformer.yaml \
    --dataset-config src/config/dataset.yaml \
    --train-config src/config/train.yaml
```

### Text Generation

```bash
python -m src.main \
    --mode generate \
    --checkpoint ./checkpoints/best_model.pth \
    --prompt "Once upon a time" \
    --max-new-tokens 50 \
    --temperature 0.8 \
    --do-sample \
    --transformer-config src/config/transformer.yaml \
    --dataset-config src/config/dataset.yaml
```

## Configuration

### Model Configuration (`transformer.yaml`)

```yaml
model:
  vocab_size: 32000
  embedding_dim: 512
  num_layers: 6
  num_heads: 8
  max_seq_len: 512
  dropout: 0.1
```

### Training Configuration (`train.yaml`)

```yaml
training:
  epochs: 10
  batch_size: 32
  optimizer:
    type: adamw
    lr: 3e-4
  scheduler:
    type: linear_warmup
    warmup_steps: 4000
```

## Architecture

### Transformer Encoder

- Multi-head self-attention
- Position-wise feed-forward networks
- Layer normalization
- Residual connections

### Transformer Decoder

- Masked multi-head self-attention
- Cross-attention to encoder outputs
- Position-wise feed-forward networks
- Layer normalization
- Residual connections

### Key Components

1. **Multi-Head Attention**: Scaled dot-product attention with multiple heads
2. **Positional Encoding**: Sinusoidal position embeddings
3. **Feed-Forward Network**: Two-layer MLP with GELU activation
4. **Layer Normalization**: Pre-normalization for stable training

## Testing

Run unit tests to verify implementation:

```bash
# Test attention mechanisms
python -m src.tests.test_attention

# Test encoder
python -m src.tests.test_encoder

# Test shape consistency
python -m src.tests.test_shapes
```

## Advanced Features

### Mixed Precision Training

Enable mixed precision for faster training:

```yaml
mixed_precision: true
```

### Gradient Accumulation

Simulate larger batch sizes:

```yaml
gradient_accumulation_steps: 4
```

### Learning Rate Scheduling

Choose from multiple schedulers:
- `linear_warmup`: Linear warmup + linear decay
- `cosine`: Cosine annealing with warmup
- `transformer`: Original Transformer schedule

## Metrics

The implementation tracks:
- Cross-entropy loss
- Token-level accuracy
- Perplexity

## Citation

If you use this implementation, please cite the original paper:

```bibtex
@article{vaswani2017attention,
  title={Attention is all you need},
  author={Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N and Kaiser, {\L}ukasz and Polosukhin, Illia},
  journal={Advances in neural information processing systems},
  volume={30},
  year={2017}
}
```

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- Based on "Attention Is All You Need" (Vaswani et al., 2017)
- Inspired by various Transformer implementations in the community
