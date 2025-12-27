import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any
from tqdm import tqdm
import logging
from .utils.metrics import compute_cross_entropy_loss, compute_accuracy, MetricsTracker
from .utils.checkpoint import save_checkpoint, save_best_model

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        config: Dict[str, Any],
        device: str = 'cuda'
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = device
        
        self.epochs = config.get('epochs', 10)
        self.gradient_accumulation_steps = config.get('gradient_accumulation_steps', 1)
        self.gradient_clipping = config.get('gradient_clipping', 1.0)
        self.log_interval = config.get('log_interval', 100)
        self.save_dir = config.get('checkpointing', {}).get('save_dir', './checkpoints')
        self.ignore_index = config.get('loss', {}).get('ignore_index', -100)
        
        self.use_amp = config.get('mixed_precision', False)
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        
        self.best_val_loss = float('inf')
        
        self.model.to(self.device)
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        metrics = MetricsTracker()
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        
        for batch_idx, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            is_decoder_only = 'decoder_input_ids' not in batch
            
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    if is_decoder_only:
                        logits = self.model(input_ids, attention_mask)
                    else:
                        decoder_input_ids = batch['decoder_input_ids'].to(self.device)
                        logits = self.model(input_ids, decoder_input_ids, attention_mask)
                    
                    loss = compute_cross_entropy_loss(logits, labels, self.ignore_index)
                    loss = loss / self.gradient_accumulation_steps
            else:
                if is_decoder_only:
                    logits = self.model(input_ids, attention_mask)
                else:
                    decoder_input_ids = batch['decoder_input_ids'].to(self.device)
                    logits = self.model(input_ids, decoder_input_ids, attention_mask)
                
                loss = compute_cross_entropy_loss(logits, labels, self.ignore_index)
                loss = loss / self.gradient_accumulation_steps
            
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                if self.gradient_clipping > 0:
                    if self.use_amp:
                        self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clipping)
                
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                if self.scheduler is not None:
                    self.scheduler.step()
                
                self.optimizer.zero_grad()
            
            with torch.no_grad():
                accuracy = compute_accuracy(logits, labels, self.ignore_index)
            
            metrics.update(loss.item() * self.gradient_accumulation_steps, accuracy)
            
            if (batch_idx + 1) % self.log_interval == 0:
                avg_loss, avg_acc = metrics.get_average()
                lr = self.optimizer.param_groups[0]['lr']
                progress_bar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'acc': f'{avg_acc:.4f}',
                    'lr': f'{lr:.6f}'
                })
        
        avg_loss, avg_acc = metrics.get_average()
        return {'loss': avg_loss, 'accuracy': avg_acc}
    
    def validate(self) -> Dict[str, float]:
        self.model.eval()
        metrics = MetricsTracker()
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                is_decoder_only = 'decoder_input_ids' not in batch
                
                if is_decoder_only:
                    logits = self.model(input_ids, attention_mask)
                else:
                    decoder_input_ids = batch['decoder_input_ids'].to(self.device)
                    logits = self.model(input_ids, decoder_input_ids, attention_mask)
                
                loss = compute_cross_entropy_loss(logits, labels, self.ignore_index)
                accuracy = compute_accuracy(logits, labels, self.ignore_index)
                
                metrics.update(loss.item(), accuracy)
        
        avg_loss, avg_acc = metrics.get_average()
        perplexity = metrics.get_perplexity()
        
        return {
            'loss': avg_loss,
            'accuracy': avg_acc,
            'perplexity': perplexity
        }
    
    def train(self) -> None:
        logger.info("Starting training...")
        logger.info(f"Device: {self.device}")
        logger.info(f"Epochs: {self.epochs}")
        logger.info(f"Gradient accumulation steps: {self.gradient_accumulation_steps}")
        for epoch in range(1, self.epochs + 1):
            train_metrics = self.train_epoch(epoch)
            logger.info(f"Epoch {epoch} - Train Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
            
            val_metrics = self.validate()
            logger.info(f"Epoch {epoch} - Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, PPL: {val_metrics['perplexity']:.2f}")
            
            save_checkpoint(
                self.model,
                self.optimizer,
                epoch,
                val_metrics['loss'],
                f"{self.save_dir}/checkpoint_epoch_{epoch}.pth",
                self.scheduler,
                val_loss=val_metrics['loss'],
                val_accuracy=val_metrics['accuracy']
            )
            
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                save_best_model(
                    self.model,
                    self.save_dir,
                    val_metrics['loss'],
                    epoch,
                    'loss'
                )
                logger.info(f"New best model saved with loss: {val_metrics['loss']:.4f}")
        
        logger.info("Training completed!")