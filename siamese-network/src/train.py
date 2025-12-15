"""
Training module for Siamese Network with TensorBoard logging.
"""

import os
import time
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


class Trainer:
    """
    Trainer class for Siamese Network with comprehensive training loop.
    """
    
    def __init__(self, model, train_loader, val_loader, loss_fn, optimizer, 
                 scheduler, config, device):
        """
        Initialize the trainer.
        
        Args:
            model: Siamese Network model
            train_loader: Training data loader
            val_loader: Validation data loader
            loss_fn: Loss function
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            config: Configuration object
            device: Device to train on
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = device
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'learning_rate': []
        }
        
        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.best_state_dict = None
        
        # Create checkpoint directory
        os.makedirs(config.save_dir, exist_ok=True)
        
        # TensorBoard writer
        self.writer = SummaryWriter(log_dir='runs/siamese_network')
        print(f"✓ TensorBoard logging to: runs/siamese_network")
    
    def train_one_epoch(self, epoch):
        """
        Train for one epoch.
        
        Args:
            epoch: Current epoch number
        
        Returns:
            Average training loss for the epoch
        """
        self.model.train()
        running_loss = 0.0
        num_batches = len(self.train_loader)
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.num_epochs}")
        
        for batch_idx, batch in enumerate(progress_bar):
            anchor, positive, negative = batch
            anchor = anchor.to(self.device)
            positive = positive.to(self.device)
            negative = negative.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            z_a, z_p, z_n = self.model(anchor, positive, negative, triplet_bool=True)
            
            # Compute loss
            loss = self.loss_fn(z_a, z_p, z_n)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Update metrics
            batch_loss = loss.item()
            running_loss += batch_loss
            
            # Update progress bar
            if (batch_idx + 1) % self.config.print_every == 0:
                progress_bar.set_postfix({'loss': f'{batch_loss:.4f}'})
        
        avg_train_loss = running_loss / num_batches
        return avg_train_loss
    
    def validate(self, epoch):
        """
        Validate the model.
        
        Args:
            epoch: Current epoch number
        
        Returns:
            Tuple of (avg_val_loss, val_accuracy, metrics_dict)
        """
        self.model.eval()
        val_loss = 0.0
        
        # Metrics
        genuine_correct = 0
        fake_correct = 0
        total = 0
        
        all_distances_ap = []
        all_distances_an = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                anchor, positive, negative = batch
                anchor = anchor.to(self.device)
                positive = positive.to(self.device)
                negative = negative.to(self.device)
                
                # Forward pass
                z_a, z_p, z_n = self.model(anchor, positive, negative, triplet_bool=True)
                
                # Compute loss
                loss = self.loss_fn(z_a, z_p, z_n)
                val_loss += loss.item()
                
                # Compute distances
                d_ap = F.pairwise_distance(z_a, z_p)
                d_an = F.pairwise_distance(z_a, z_n)
                
                # Store distances for analysis
                all_distances_ap.extend(d_ap.cpu().tolist())
                all_distances_an.extend(d_an.cpu().tolist())
                
                # Accuracy computation
                genuine_correct += (d_ap < self.config.threshold_distance).sum().item()
                fake_correct += (d_an >= self.config.threshold_distance).sum().item()
                total += anchor.size(0)
        
        avg_val_loss = val_loss / len(self.val_loader)
        
        # Calculate accuracy
        genuine_acc = genuine_correct / total if total > 0 else 0.0
        fake_acc = fake_correct / total if total > 0 else 0.0
        overall_acc = (genuine_correct + fake_correct) / (2 * total) if total > 0 else 0.0
        
        # Distance statistics
        import numpy as np
        metrics = {
            'genuine_accuracy': genuine_acc,
            'fake_accuracy': fake_acc,
            'overall_accuracy': overall_acc,
            'mean_dist_genuine': np.mean(all_distances_ap),
            'mean_dist_fake': np.mean(all_distances_an),
            'std_dist_genuine': np.std(all_distances_ap),
            'std_dist_fake': np.std(all_distances_an)
        }
        
        return avg_val_loss, overall_acc, metrics
    
    def save_checkpoint(self, epoch, metrics):
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch
            metrics: Dictionary of metrics
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config,
            'history': self.history
        }
        
        checkpoint_path = os.path.join(self.config.save_dir, f'checkpoint_epoch_{epoch+1}.pth')
        torch.save(checkpoint, checkpoint_path)
        print(f"✓ Checkpoint saved: {checkpoint_path}")
    
    def save_best_model(self):
        """Save the best model."""
        if self.best_state_dict is not None:
            best_model_path = os.path.join(self.config.save_dir, 'best_model.pth')
            torch.save({
                'epoch': self.best_epoch,
                'model_state_dict': self.best_state_dict,
                'best_val_acc': self.best_val_acc,
                'config': self.config
            }, best_model_path)
            print(f"✓ Best model saved: {best_model_path}")
    
    def train(self):
        """
        Main training loop.
        
        Returns:
            Trained model and training history
        """
        print("=" * 60)
        print("STARTING TRAINING")
        print("=" * 60)
        
        start_time = time.time()
        
        for epoch in range(self.config.num_epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{self.config.num_epochs}")
            print(f"{'='*60}")
            
            # Train
            train_loss = self.train_one_epoch(epoch)
            
            # Validate
            val_loss, val_acc, metrics = self.validate(epoch)
            
            # Update learning rate
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_accuracy'].append(val_acc)
            self.history['learning_rate'].append(current_lr)
            
            # TensorBoard logging
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('Accuracy/val', val_acc, epoch)
            self.writer.add_scalar('Accuracy/genuine', metrics['genuine_accuracy'], epoch)
            self.writer.add_scalar('Accuracy/fake', metrics['fake_accuracy'], epoch)
            self.writer.add_scalar('Distance/genuine_mean', metrics['mean_dist_genuine'], epoch)
            self.writer.add_scalar('Distance/fake_mean', metrics['mean_dist_fake'], epoch)
            self.writer.add_scalar('Learning_Rate', current_lr, epoch)
            
            # Print epoch summary
            print(f"\nEpoch {epoch+1} Summary:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Val Accuracy: {val_acc:.4f} (Genuine: {metrics['genuine_accuracy']:.4f}, Fake: {metrics['fake_accuracy']:.4f})")
            print(f"  Mean Distance (Genuine): {metrics['mean_dist_genuine']:.4f} ± {metrics['std_dist_genuine']:.4f}")
            print(f"  Mean Distance (Fake): {metrics['mean_dist_fake']:.4f} ± {metrics['std_dist_fake']:.4f}")
            print(f"  Learning Rate: {current_lr:.6f}")
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                self.best_state_dict = self.model.state_dict().copy()
                print(f"  ★ New best validation accuracy: {self.best_val_acc:.4f}")
            
            # Save checkpoint (only best if configured)
            if not self.config.save_best_only or val_acc == self.best_val_acc:
                self.save_checkpoint(epoch, metrics)
        
        # Load best model
        if self.best_state_dict is not None:
            self.model.load_state_dict(self.best_state_dict)
            print(f"\n✓ Loaded best model from epoch {self.best_epoch+1}")
        
        # Save best model separately
        self.save_best_model()
        
        # Close TensorBoard writer
        self.writer.close()
        
        # Training complete
        elapsed_time = time.time() - start_time
        print("=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)
        print(f"Total training time: {elapsed_time/60:.2f} minutes")
        print(f"Best validation accuracy: {self.best_val_acc:.4f} (Epoch {self.best_epoch+1})")
        print("=" * 60)
        
        return self.model, self.history