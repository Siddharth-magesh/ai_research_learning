import torch
from pathlib import Path
from convolution_network import FlexibleCNN
from data_loader import get_data_loader
from trainer import Trainer

def train_best_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(__file__).parent.parent / 'results'
    
    best_params = {
        'n_layers': 3,
        'channels': 32,
        'dropout': 0.3,
        'activation': 'relu',
        'lr': 0.001,
        'batch_size': 64,
        'epochs': 15
    }
    
    model = FlexibleCNN(
        n_layers=best_params['n_layers'],
        channels=best_params['channels'],
        dropout=best_params['dropout'],
        activation=best_params['activation']
    ).to(device)
    
    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=best_params['lr']
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer,
        T_max=best_params['epochs']
    )
    
    train_loader, val_loader = get_data_loader(best_params['batch_size'])
    
    trainer = Trainer(
        model=model,
        device=device,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler
    )
    
    print(f"Training on {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    for epoch in range(best_params['epochs']):
        trainer.train()
        accuracy = trainer.evaluate()
        print(f"Epoch {epoch+1}/{best_params['epochs']}: Accuracy = {accuracy:.4f}")
    
    output_dir.mkdir(exist_ok=True)
    torch.save(model.state_dict(), output_dir / 'best_model.pth')
    print(f"\nModel saved to: {output_dir / 'best_model.pth'}")

if __name__ == '__main__':
    train_best_model()
