import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

class Trainer():
    def __init__(
        self, 
        model: nn.Module, 
        device: torch.device, 
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        optimizer: Optimizer,
        scheduler: LRScheduler = None
    ) -> None:
        self.model = model
        self.device = device
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler

    def train(self) -> None:
        self.model.train()
        for x, y in self.train_dataloader:
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            loss = F.cross_entropy(self.model(x), y)
            loss.backward()
            self.optimizer.step()
        
        if self.scheduler is not None:
            self.scheduler.step()
            

    def evaluate(self) -> float:
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in self.val_dataloader:
                x, y = x.to(self.device), y.to(self.device)
                preds = self.model(x).argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        return correct / total