# TensorBoard Examples Collection

Practical, copy-paste ready examples for common use cases.

## Table of Contents

1. [Basic Training Loop](#basic-training-loop)
2. [Image Classification](#image-classification)
3. [Image Generation (GAN/VAE)](#image-generation)
4. [Object Detection](#object-detection)
5. [Semantic Segmentation](#semantic-segmentation)
6. [Natural Language Processing](#natural-language-processing)
7. [Reinforcement Learning](#reinforcement-learning)
8. [Hyperparameter Tuning](#hyperparameter-tuning)
9. [Transfer Learning](#transfer-learning)
10. [Model Debugging](#model-debugging)

---

## Basic Training Loop

```python
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
from tqdm import tqdm

def train(model, train_loader, val_loader, epochs=50):
    writer = SummaryWriter('runs/basic_training')
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    # Log model graph
    dummy_input = torch.randn(1, 3, 224, 224)
    writer.add_graph(model, dummy_input)
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for batch_idx, (data, target) in enumerate(pbar):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pred = output.argmax(dim=1)
            train_correct += pred.eq(target).sum().item()
            train_total += target.size(0)
            
            # Log batch-level metrics
            if batch_idx % 10 == 0:
                step = epoch * len(train_loader) + batch_idx
                writer.add_scalar('Batch/loss', loss.item(), step)
                pbar.set_postfix({'loss': loss.item()})
        
        # Compute epoch metrics
        avg_train_loss = train_loss / len(train_loader)
        train_acc = train_correct / train_total
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                output = model(data)
                loss = criterion(output, target)
                val_loss += loss.item()
                pred = output.argmax(dim=1)
                val_correct += pred.eq(target).sum().item()
                val_total += target.size(0)
        
        avg_val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total
        
        # Log epoch-level metrics
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Loss/validation', avg_val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/validation', val_acc, epoch)
        writer.add_scalar('Learning_Rate', scheduler.get_last_lr()[0], epoch)
        
        # Combined view
        writer.add_scalars('Losses', {
            'train': avg_train_loss,
            'validation': avg_val_loss
        }, epoch)
        
        # Log weights histogram every 5 epochs
        if epoch % 5 == 0:
            for name, param in model.named_parameters():
                writer.add_histogram(f'Weights/{name}', param, epoch)
                if param.grad is not None:
                    writer.add_histogram(f'Gradients/{name}', param.grad, epoch)
        
        scheduler.step()
        
        print(f'Epoch {epoch}: Train Loss={avg_train_loss:.4f}, '
              f'Val Loss={avg_val_loss:.4f}, Val Acc={val_acc:.4f}')
    
    writer.close()
    print(\"Training complete! Run: tensorboard --logdir runs\")
```

---

## Image Classification

```python
from torch.utils.tensorboard import SummaryWriter
import torch
import torchvision
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class ImageClassifier:
    def __init__(self, model, num_classes, class_names):
        self.model = model
        self.num_classes = num_classes
        self.class_names = class_names
        self.writer = SummaryWriter('runs/image_classification')
    
    def log_predictions(self, images, labels, predictions, epoch):
        \"\"\"Log sample predictions with images.\"\"\"
        # Create grid of images with predictions
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        for idx, ax in enumerate(axes.flat):
            if idx < len(images):
                # Denormalize image
                img = images[idx].cpu()
                img = img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                img = img + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                img = torch.clamp(img, 0, 1)
                
                ax.imshow(img.permute(1, 2, 0))
                true_label = self.class_names[labels[idx]]
                pred_label = self.class_names[predictions[idx]]
                color = 'green' if labels[idx] == predictions[idx] else 'red'
                ax.set_title(f'True: {true_label}\\nPred: {pred_label}', 
                           color=color, fontsize=8)
                ax.axis('off')
        
        self.writer.add_figure('Predictions/samples', fig, epoch)
        plt.close(fig)
    
    def log_confusion_matrix(self, all_labels, all_predictions, epoch):
        \"\"\"Log confusion matrix.\"\"\"
        cm = confusion_matrix(all_labels, all_predictions)
        
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names,
                   ax=ax)
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        ax.set_title(f'Confusion Matrix - Epoch {epoch}')
        
        self.writer.add_figure('Evaluation/confusion_matrix', fig, epoch)
        plt.close(fig)
    
    def log_per_class_metrics(self, all_labels, all_predictions, epoch):
        \"\"\"Log per-class precision, recall, F1.\"\"\"
        report = classification_report(
            all_labels, all_predictions,
            target_names=self.class_names,
            output_dict=True,
            zero_division=0
        )
        
        for class_name in self.class_names:
            if class_name in report:
                metrics = report[class_name]
                self.writer.add_scalar(
                    f'PerClass/Precision/{class_name}',
                    metrics['precision'], epoch
                )
                self.writer.add_scalar(
                    f'PerClass/Recall/{class_name}',
                    metrics['recall'], epoch
                )
                self.writer.add_scalar(
                    f'PerClass/F1/{class_name}',
                    metrics['f1-score'], epoch
                )
    
    def evaluate(self, test_loader, epoch):
        \"\"\"Full evaluation with TensorBoard logging.\"\"\"
        self.model.eval()
        all_labels = []
        all_predictions = []
        all_probs = []
        sample_images = []
        sample_labels = []
        sample_preds = []
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                output = self.model(data)
                probs = torch.softmax(output, dim=1)
                pred = output.argmax(dim=1)
                
                all_labels.extend(target.cpu().numpy())
                all_predictions.extend(pred.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                
                # Collect samples for visualization
                if batch_idx == 0:
                    sample_images = data[:16]
                    sample_labels = target[:16].cpu()
                    sample_preds = pred[:16].cpu()
        
        # Log visualizations
        self.log_predictions(sample_images, sample_labels, sample_preds, epoch)
        self.log_confusion_matrix(all_labels, all_predictions, epoch)
        self.log_per_class_metrics(all_labels, all_predictions, epoch)
        
        # Log PR curves for each class
        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)
        for class_id in range(self.num_classes):
            binary_labels = (all_labels == class_id).astype(float)
            class_probs = all_probs[:, class_id]
            self.writer.add_pr_curve(
                f'PR_Curve/{self.class_names[class_id]}',
                torch.tensor(binary_labels),
                torch.tensor(class_probs),
                epoch
            )
        
        # Overall accuracy
        accuracy = (np.array(all_labels) == np.array(all_predictions)).mean()
        self.writer.add_scalar('Test/accuracy', accuracy, epoch)
        
        return accuracy
```

---

## Image Generation

### GAN Training

```python
from torch.utils.tensorboard import SummaryWriter
import torch
import torchvision

class GANTrainer:
    def __init__(self, generator, discriminator):
        self.G = generator
        self.D = discriminator
        self.writer = SummaryWriter('runs/gan_training')
        
        # Fixed noise for consistent generation visualization
        self.fixed_noise = torch.randn(64, 100, 1, 1)
    
    def train_step(self, real_images, epoch, batch_idx):
        batch_size = real_images.size(0)
        
        # Train Discriminator
        self.D.zero_grad()
        real_labels = torch.ones(batch_size, 1)
        fake_labels = torch.zeros(batch_size, 1)
        
        # Real images
        real_output = self.D(real_images)
        d_loss_real = nn.BCELoss()(real_output, real_labels)
        
        # Fake images
        noise = torch.randn(batch_size, 100, 1, 1)
        fake_images = self.G(noise)
        fake_output = self.D(fake_images.detach())
        d_loss_fake = nn.BCELoss()(fake_output, fake_labels)
        
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        self.d_optimizer.step()
        
        # Train Generator
        self.G.zero_grad()
        fake_output = self.D(fake_images)
        g_loss = nn.BCELoss()(fake_output, real_labels)
        g_loss.backward()
        self.g_optimizer.step()
        
        # Logging
        step = epoch * len(self.train_loader) + batch_idx
        
        if batch_idx % 50 == 0:
            self.writer.add_scalar('Loss/Discriminator', d_loss.item(), step)
            self.writer.add_scalar('Loss/Generator', g_loss.item(), step)
            self.writer.add_scalar('D(x)', real_output.mean().item(), step)
            self.writer.add_scalar('D(G(z))', fake_output.mean().item(), step)
        
        return d_loss.item(), g_loss.item()
    
    def log_generated_images(self, epoch):
        \"\"\"Log generated images from fixed noise.\"\"\"
        self.G.eval()
        with torch.no_grad():
            fake_images = self.G(self.fixed_noise)
            grid = torchvision.utils.make_grid(
                fake_images[:64], nrow=8, normalize=True
            )
            self.writer.add_image('Generated/samples', grid, epoch)
        self.G.train()
```

### VAE Training

```python
class VAETrainer:
    def __init__(self, model):
        self.model = model
        self.writer = SummaryWriter('runs/vae_training')
    
    def vae_loss(self, recon_x, x, mu, logvar):
        # Reconstruction loss
        recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
        
        # KL divergence
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        return recon_loss + kl_div, recon_loss, kl_div
    
    def train_epoch(self, train_loader, epoch):
        self.model.train()
        total_loss = 0
        total_recon = 0
        total_kl = 0
        
        for batch_idx, (data, _) in enumerate(train_loader):
            self.optimizer.zero_grad()
            
            recon_batch, mu, logvar = self.model(data)
            loss, recon, kl = self.vae_loss(recon_batch, data, mu, logvar)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            total_recon += recon.item()
            total_kl += kl.item()
            
            # Log batch metrics
            if batch_idx % 100 == 0:
                step = epoch * len(train_loader) + batch_idx
                self.writer.add_scalar('Batch/total_loss', loss.item(), step)
                self.writer.add_scalar('Batch/reconstruction', recon.item(), step)
                self.writer.add_scalar('Batch/kl_divergence', kl.item(), step)
        
        # Log epoch metrics
        n_batches = len(train_loader)
        self.writer.add_scalar('Epoch/total_loss', total_loss/n_batches, epoch)
        self.writer.add_scalar('Epoch/reconstruction', total_recon/n_batches, epoch)
        self.writer.add_scalar('Epoch/kl_divergence', total_kl/n_batches, epoch)
        
        # Log reconstructions
        self.log_reconstructions(data[:8], recon_batch[:8], epoch)
    
    def log_reconstructions(self, original, reconstructed, epoch):
        \"\"\"Compare original vs reconstructed images.\"\"\"
        comparison = torch.cat([original, reconstructed])
        grid = torchvision.utils.make_grid(comparison, nrow=8, normalize=True)
        self.writer.add_image('Reconstruction/comparison', grid, epoch)
```

---

## Hyperparameter Tuning

```python
from torch.utils.tensorboard import SummaryWriter
import itertools

def hyperparameter_search():
    # Define hyperparameter grid
    learning_rates = [1e-4, 1e-3, 1e-2]
    batch_sizes = [32, 64, 128]
    dropouts = [0.1, 0.3, 0.5]
    optimizers = ['Adam', 'SGD']
    
    results = []
    
    for lr, bs, dropout, opt_name in itertools.product(
        learning_rates, batch_sizes, dropouts, optimizers
    ):
        # Create unique run name
        run_name = f'lr{lr}_bs{bs}_do{dropout}_{opt_name}'
        writer = SummaryWriter(f'runs/hparam_search/{run_name}')
        
        print(f\"\\nTraining: {run_name}\")
        
        # Train model with these hyperparameters
        model = create_model(dropout=dropout)
        
        if opt_name == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        
        # Training loop
        best_acc = 0
        for epoch in range(20):  # Quick search
            train_loss, train_acc = train_epoch(model, train_loader, optimizer)
            val_loss, val_acc = validate(model, val_loader)
            
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Accuracy/val', val_acc, epoch)
            
            best_acc = max(best_acc, val_acc)
        
        # Log hyperparameters and final metrics
        hparams = {
            'lr': lr,
            'batch_size': bs,
            'dropout': dropout,
            'optimizer': opt_name
        }
        
        metrics = {
            'hparam/best_accuracy': best_acc,
            'hparam/final_loss': val_loss
        }
        
        writer.add_hparams(hparams, metrics)
        writer.close()
        
        results.append({
            **hparams,
            'best_accuracy': best_acc
        })
    
    # Print best configuration
    best = max(results, key=lambda x: x['best_accuracy'])
    print(f\"\\nBest configuration: {best}\")
    
    return results
```

---

## Model Debugging

```python
from torch.utils.tensorboard import SummaryWriter
import torch

class ModelDebugger:
    def __init__(self, model):
        self.model = model
        self.writer = SummaryWriter('runs/debugging')
        self.activation_hooks = {}
        self.gradient_hooks = {}
        
    def register_hooks(self):
        \"\"\"Register hooks to capture activations and gradients.\"\"\"
        
        def forward_hook(name):
            def hook(module, input, output):
                self.activation_hooks[name] = output.detach()
            return hook
        
        def backward_hook(name):
            def hook(module, grad_input, grad_output):
                self.gradient_hooks[name] = grad_output[0].detach()
            return hook
        
        # Register for each layer
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                module.register_forward_hook(forward_hook(name))
                module.register_backward_hook(backward_hook(name))
    
    def log_statistics(self, epoch):
        \"\"\"Log activation and gradient statistics.\"\"\"
        
        # Activations
        for name, activation in self.activation_hooks.items():
            self.writer.add_histogram(f'Activations/{name}', activation, epoch)
            self.writer.add_scalar(
                f'Activation_Stats/{name}/mean',
                activation.mean().item(), epoch
            )
            self.writer.add_scalar(
                f'Activation_Stats/{name}/std',
                activation.std().item(), epoch
            )
            self.writer.add_scalar(
                f'Activation_Stats/{name}/max',
                activation.max().item(), epoch
            )
        
        # Gradients
        for name, gradient in self.gradient_hooks.items():
            self.writer.add_histogram(f'Gradients/{name}', gradient, epoch)
            grad_norm = gradient.norm().item()
            self.writer.add_scalar(f'Gradient_Norm/{name}', grad_norm, epoch)
            
            # Detect gradient issues
            if grad_norm < 1e-7:
                print(f\"⚠️  Vanishing gradient in {name}: {grad_norm}\")
            elif grad_norm > 1e3:
                print(f\"⚠️  Exploding gradient in {name}: {grad_norm}\")
        
        # Parameter statistics
        for name, param in self.model.named_parameters():
            self.writer.add_histogram(f'Parameters/{name}', param, epoch)
            if param.grad is not None:
                grad_to_param_ratio = (param.grad.norm() / param.norm()).item()
                self.writer.add_scalar(
                    f'GradToParam_Ratio/{name}',
                    grad_to_param_ratio, epoch
                )
        
        self.activation_hooks.clear()
        self.gradient_hooks.clear()
    
    def check_dead_neurons(self, epoch):
        \"\"\"Detect dead ReLU neurons.\"\"\"
        for name, activation in self.activation_hooks.items():
            if 'relu' in name.lower():
                dead_neurons = (activation == 0).float().mean().item()
                self.writer.add_scalar(
                    f'DeadNeurons/{name}',
                    dead_neurons, epoch
                )
                if dead_neurons > 0.5:
                    print(f\"⚠️  {dead_neurons*100:.1f}% dead neurons in {name}\")

# Usage
debugger = ModelDebugger(model)
debugger.register_hooks()

for epoch in range(epochs):
    # Training step
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    
    # Log debugging info
    debugger.log_statistics(epoch)
    debugger.check_dead_neurons(epoch)
    
    optimizer.step()
```

---

**More examples available in the reference code!**

Run `python main.py` to see all demonstrations in action.
