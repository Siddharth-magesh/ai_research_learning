# TRANSFORMER PROJECT — FOLDER STRUCTURE

```
transformers/
│
├── src/
│   ├── main.py                  # Entry point: training / evaluation trigger
│   ├── train.py                 # Training loop logic
│   ├── evaluate.py              # Validation / testing logic
│   ├── inference.py             # Inference / generation
│   │
│   ├── config/
│   │   ├── __init__.py
│   │   ├── base_config.py       # Common hyperparameters
│   │   ├── transformer.yaml     # Model config (layers, heads, dim, etc.)
│   │   ├── dataset.yaml         # Dataset paths & params
│   │   └── train.yaml           # Optimizer, scheduler, epochs
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py           # HF / Kaggle dataset loading
│   │   ├── tokenizer.py         # Tokenizer logic
│   │   └── collate.py           # Padding, masking, batching
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── transformer.py       # Full Transformer model
│   │   ├── encoder.py           # Encoder stack
│   │   ├── decoder.py           # Decoder stack
│   │   └── embeddings.py        # Token + positional embeddings
│   │
│   ├── modules/
│   │   ├── __init__.py
│   │   ├── attention.py         # Scaled dot-product + multi-head attention
│   │   ├── feed_forward.py      # MLP / FFN block
│   │   ├── layer_norm.py        # Custom LayerNorm (optional)
│   │   ├── residual.py          # Residual + norm wrapper
│   │   ├── positional_encoding.py
│   │   └── masking.py           # Causal & padding masks
│   │
│   ├── optim/
│   │   ├── __init__.py
│   │   ├── optimizer.py         # Adam / AdamW wrappers
│   │   └── scheduler.py         # LR schedulers (warmup, cosine)
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── logging.py           # Logging utilities
│   │   ├── checkpoint.py        # Save / load models
│   │   ├── metrics.py           # Loss, accuracy, perplexity
│   │   └── seed.py              # Reproducibility
│   │
│   └── tests/
│       ├── test_attention.py
│       ├── test_encoder.py
│       └── test_shapes.py       # Shape consistency tests
│
├── experiments/
│   ├── exp_001_baseline/
│   │   ├── config.yaml
│   │   └── notes.md
│   └── exp_002_scaling/
│       ├── config.yaml
│       └── notes.md
│
├── docs/
│   ├── architecture.md          # Transformer architecture explanation
│   ├── attention_math.md        # Full math derivation
│   ├── shape_tracking.md        # Tensor shape flow diagrams
│   ├── training_strategy.md
│   └── vision_transformer.md
│
├── notebooks/
│   ├── attention_visualization.ipynb
│   ├── token_embeddings.ipynb
│   └── debugging_shapes.ipynb
│
├── requirements.txt
├── README.md
└── .gitignore
```