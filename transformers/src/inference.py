import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Any
import logging
logger = logging.getLogger(__name__)


class TextGenerator:
    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        device: str = 'cuda',
        max_length: int = 512
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length
        
        self.model.to(self.device)
        self.model.eval()
    
    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = True,
        num_return_sequences: int = 1
    ) -> List[str]:
        encoded = self.tokenizer.encode(prompt, max_length=self.max_length)
        input_ids = torch.tensor([encoded['input_ids']] * num_return_sequences).to(self.device)
        generated_sequences = []
        
        for _ in range(max_new_tokens):
            logits = self.model(input_ids)
            
            next_token_logits = logits[:, -1, :] / temperature
            
            if top_k is not None:
                next_token_logits = self._top_k_filtering(next_token_logits, top_k)
            
            if top_p is not None:
                next_token_logits = self._top_p_filtering(next_token_logits, top_p)
            
            if do_sample:
                probs = F.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1)
            else:
                next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            input_ids = torch.cat([input_ids, next_tokens], dim=1)
            
            if input_ids.size(1) >= self.max_length:
                break
            
            if hasattr(self.tokenizer.tokenizer, 'eos_token_id'):
                eos_token_id = self.tokenizer.tokenizer.eos_token_id
                if (next_tokens == eos_token_id).all():
                    break
        
        for seq in input_ids:
            text = self.tokenizer.decode(seq.cpu().tolist())
            generated_sequences.append(text)
        
        return generated_sequences
    
    def _top_k_filtering(self, logits: torch.Tensor, top_k: int) -> torch.Tensor:
        top_k = min(top_k, logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = float('-inf')
        return logits
    def _top_p_filtering(self, logits: torch.Tensor, top_p: float) -> torch.Tensor:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove
        )
        logits[indices_to_remove] = float('-inf')
        return logits
    
    @torch.no_grad()
    def translate(
        self,
        source_text: str,
        max_length: int = 100,
        beam_size: int = 4
    ) -> str:
        encoded = self.tokenizer.encode(source_text, max_length=self.max_length)
        src_input_ids = torch.tensor([encoded['input_ids']]).to(self.device)
        src_attention_mask = torch.tensor([encoded['attention_mask']]).to(self.device)
        encoder_output = self.model.encode(src_input_ids, src_attention_mask)
        
        bos_token_id = self.tokenizer.tokenizer.bos_token_id if hasattr(
            self.tokenizer.tokenizer, 'bos_token_id'
        ) else 0
        decoder_input_ids = torch.tensor([[bos_token_id]]).to(self.device)
        
        for _ in range(max_length):
            logits = self.model.decode(
                decoder_input_ids,
                encoder_output,
                None,
                src_attention_mask
            )
            
            next_token_logits = logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            decoder_input_ids = torch.cat([decoder_input_ids, next_token], dim=1)
            
            if hasattr(self.tokenizer.tokenizer, 'eos_token_id'):
                if next_token.item() == self.tokenizer.tokenizer.eos_token_id:
                    break
        
        translated_text = self.tokenizer.decode(decoder_input_ids[0].cpu().tolist())
        
        return translated_text
