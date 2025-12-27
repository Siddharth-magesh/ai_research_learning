import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any
from tqdm import tqdm
import logging
from .utils.metrics import compute_cross_entropy_loss, compute_accuracy, compute_perplexity

logger = logging.getLogger(__name__)


class Evaluator:
    def __init__(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        device: str = 'cuda',
        ignore_index: int = -100
    ):
        self.model = model
        self.test_loader = test_loader
        self.device = device
        self.ignore_index = ignore_index
        
        self.model.to(self.device)
        self.model.eval()
    
    def evaluate(self) -> Dict[str, float]:
        logger.info("Starting evaluation...")
        total_loss = 0.0
        total_accuracy = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc='Evaluating'):
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
                
                batch_size = input_ids.size(0)
                total_loss += loss.item() * batch_size
                total_accuracy += accuracy * batch_size
                total_samples += batch_size
        
        avg_loss = total_loss / total_samples
        avg_accuracy = total_accuracy / total_samples
        perplexity = compute_perplexity(avg_loss)
        
        results = {
            'loss': avg_loss,
            'accuracy': avg_accuracy,
            'perplexity': perplexity
        }
        
        logger.info(f"Evaluation Results:")
        logger.info(f"  Loss: {avg_loss:.4f}")
        logger.info(f"  Accuracy: {avg_accuracy:.4f}")
        logger.info(f"  Perplexity: {perplexity:.2f}")
        
        return results
    
    def evaluate_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        self.model.eval()
        with torch.no_grad():
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
        
        return {
            'loss': loss.item(),
            'accuracy': accuracy
        }
