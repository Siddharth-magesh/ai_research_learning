import torch
from torch.utils.data import DataLoader

from .dataset import TextDataset
from .collate import collate_fn
from .preprocessing import preprocess_text
from .tokenizer import Tokenizer

class Datamodule():
    def __init__(self, dataset_cfg: dict, train_cfg: dict) -> None:
        self.dataset_cfg = dataset_cfg
        self.train_cfg = train_cfg
        self.tokenizer = Tokenizer(train_cfg["tokenizer"])

    def setup(self) -> None:
        self.train_dataset = TextDataset(self.dataset_cfg, split="train")
        self.val_dataset = TextDataset(self.dataset_cfg, split="validation")

    def _tokenize_sample(self, sample: dict) -> dict:
        text = preprocess_text(
            sample["text"],
            self.dataset_cfg.get("preprocessing",{})
        )
        encoded = self.tokenizer.encode(
            text,
            max_length=self.dataset_cfg["sequence_length"]
        )
        return {
            "input_ids": torch.tensor(encoded["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(encoded["attention_mask"], dtype=torch.long)
        }
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_cfg["batch_size"],
            shuffle=True,
            num_workers=self.train_cfg.get("num_workers", 0),
            collate_fn=lambda batch: collate_fn(
                [self._tokenize_sample(x) for x in batch],
                self.tokenizer,
                self.dataset_cfg["sequence_length"],
                is_decoder_only=self.train_cfg.get("decoder_only", False)
            )
        )
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.train_cfg["batch_size"],
            shuffle=False,
            num_workers=self.train_cfg.get("num_workers",0),
            collate_fn=lambda batch: collate_fn(
                [self._tokenize_sample(x) for x in batch],
                self.tokenizer,
                self.dataset_cfg["sequence_length"],
                is_decoder_only=self.train_cfg.get("decoder_only", False)
            )
        )
