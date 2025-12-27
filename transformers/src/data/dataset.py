from datasets import load_dataset, Dataset

class TextDataset:
    def __init__(self, config: dict, split: str = "train") -> None:
        self.config = config
        self.split = split
        self.dataset = self._load_dataset()

    def _load_dataset(self) -> Dataset:
        return load_dataset(
            self.config["name"],
            self.config.get("subset", None),
            split = self.split
        )
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> dict:
        sample = self.dataset[idx]
        return {
            "text": sample[self.config["text_field"]]
        }
