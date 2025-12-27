import argparse
import yaml
import torch
from pathlib import Path
from .config.base_config import BaseConfig
from .models.transformer import Transformer, DecoderOnlyTransformer
from .data.datamodule import Datamodule
from .train import Trainer
from .evaluate import Evaluator
from .inference import TextGenerator
from .optim.optimizer import create_optimizer
from .optim.scheduler import create_scheduler
from .utils.seed import set_seed
from .utils.logging import setup_logger
from .utils.checkpoint import load_checkpoint

import logging


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def create_model(model_config: dict, device: str) -> torch.nn.Module:
    model_type = model_config.get('type', 'encoder_decoder')
    if model_type == 'encoder_decoder':
        model = Transformer(
            vocab_size=model_config['vocab_size'],
            embed_dim=model_config['embedding_dim'],
            num_layers=model_config['num_layers'],
            num_heads=model_config['num_heads'],
            ffn_hidden_dim=model_config['ffn']['hidden_dim'],
            max_seq_len=model_config['max_seq_len'],
            dropout=model_config['dropout'],
            weight_tying=model_config.get('weight_tying', True)
        )
    elif model_type == 'decoder_only':
        model = DecoderOnlyTransformer(
            vocab_size=model_config['vocab_size'],
            embed_dim=model_config['embedding_dim'],
            num_layers=model_config['num_layers'],
            num_heads=model_config['num_heads'],
            ffn_hidden_dim=model_config['ffn']['hidden_dim'],
            max_seq_len=model_config['max_seq_len'],
            dropout=model_config['dropout'],
            weight_tying=model_config.get('weight_tying', True)
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model


def train(args):
    base_config = BaseConfig()
    transformer_config = load_config(args.transformer_config)
    dataset_config = load_config(args.dataset_config)
    train_config = load_config(args.train_config)
    logger = setup_logger(
        'transformer',
        log_file=f'{base_config.project_name}/logs/train.log'
    )
    
    set_seed(base_config.seed)
    logger.info(f"Seed set to {base_config.seed}")
    
    device = base_config.device
    logger.info(f"Using device: {device}")
    
    logger.info("Setting up data module...")
    datamodule = Datamodule(dataset_config['dataset'], train_config['training'])
    datamodule.setup()
    
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    logger.info(f"Train samples: {len(train_loader.dataset)}")
