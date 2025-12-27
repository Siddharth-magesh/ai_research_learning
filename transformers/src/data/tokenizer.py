from transformers import AutoTokenizer
from typing import List

class Tokenizer:
    def __init__(self, config: dict) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            config["tokenizer_name"],
            use_fast=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def encode(self, text: str, max_length: int, truncation: bool =True) -> dict:
        return self.tokenizer(
            text,
            max_length=max_length,
            truncation=truncation,
            padding=False,
            return_attention_mask=True
        )
    
    def decode(self, token_ids: List[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)
