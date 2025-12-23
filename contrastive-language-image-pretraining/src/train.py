import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
from clip import CLIP, CLIPLoss
from data_loader import get_data_loader
from config import Config
import time

class Trainer:
    """Trainer class for CLIP model training."""
    
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(config.device)
        self.best_loss = float('inf')
        
        # Setup checkpoint directory
        self.checkpoint_dir = Path(config.save_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize model, loss, and optimizer
        self.model = None
        self.loss_fn = None
        self.optimizer = None
        self.train_loader = None
        self.scheduler = None
        
    def setup_paths(self):
        """Setup data paths for Flickr30k dataset."""
        home_dir = Path.home()
        kaggle_cache = home_dir / '.cache' / 'kagglehub' / 'datasets' / 'hsankesara' / 'flickr-image-dataset' / 'versions' / '1'
        
        if kaggle_cache.exists():
            images_dir = kaggle_cache / 'flickr30k_images'
            captions_file = kaggle_cache / 'results.csv'
            
            if images_dir.exists() and captions_file.exists():
                return str(images_dir), str(captions_file)
        
        raise FileNotFoundError("Flickr30k dataset not found. Please ensure it's downloaded via kagglehub.")
    
    def load_data(self, max_samples=None):
        """Load dataset and create data loader."""
        images_dir, captions_file = self.setup_paths()
        print(f"Images directory: {images_dir}")
        print(f"Captions file: {captions_file}")
        
        print("\nLoading dataset...")
        self.train_loader = get_data_loader(
            images_dir, 
            captions_file, 
            batch_size=self.config.batch_size, 
            num_workers=self.config.num_workers,
            max_samples=max_samples
        )
        print(f"Loaded {len(self.train_loader)} batches")
        
    def build_model(self):
        """Initialize model, loss function, and optimizer."""
        print("\nInitializing CLIP model...")
        self.model = CLIP().to(self.device)
        self.loss_fn = CLIPLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        # Initialize learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=self.config.num_epochs,
            eta_min=1e-6
        )
        print(f"Model moved to {self.device}")
        
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        current_lr = self.optimizer.param_groups[0]['lr']
        
        pbar = tqdm(self.train_loader, desc="Training", leave=True)
        for images, captions in pbar:
            images = images.to(self.device)
            captions = captions.to(self.device)
            
            self.optimizer.zero_grad()
            logits, image_features, text_features = self.model(images, captions)
            loss = self.loss_fn(logits)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{current_lr:.6f}"
            })
        
        avg_loss = total_loss / len(self.train_loader)
        # Step the scheduler after each epoch
        if self.scheduler is not None:
            self.scheduler.step()
        
        return avg_loss
    
    def save_checkpoint(self, epoch, avg_loss):
        """Save model checkpoint."""
        if (epoch + 1) % 5 == 0 or avg_loss < self.best_loss:
            checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pth"
            checkpoint_dict = {
                'epoch': epoch + 1,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': avg_loss,
                'config': self.config
            }
            if self.scheduler is not None:
                checkpoint_dict['scheduler_state_dict'] = self.scheduler.state_dict()
            
            torch.save(checkpoint_dict, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")
            
            if avg_loss < self.best_loss:
                self.best_loss = avg_loss
                best_path = self.checkpoint_dir / "best_model.pth"
                torch.save(self.model.state_dict(), best_path)
                print(f"Best model updated with loss: {self.best_loss:.4f}")
    
    def train(self, epochs=None):
        """Main training loop."""
        num_epochs = epochs if epochs is not None else self.config.num_epochs
        print(f"\nStarting training for {num_epochs} epochs...\n")
        
        for epoch in range(self.config.num_epochs):
            epoch_start = time.time()
            avg_loss = self.train_epoch()
            epoch_time = time.time() - epoch_start
            
            print(f"Epoch {epoch+1}/{self.config.num_epochs} | Loss: {avg_loss:.4f} | Time: {epoch_time:.2f}s")
            
            self.save_checkpoint(epoch, avg_loss)
            print()
        
        print("Training complete!")
        return self.model
    
    def load_checkpoint(self, checkpoint_path):
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print(f"Loaded checkpoint from epoch {epoch} with loss {loss:.4f}")
        return epoch