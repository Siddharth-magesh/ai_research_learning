import optuna
import torch
import os
import json
from pathlib import Path
from convolution_network import FlexibleCNN
from data_loader import get_data_loader
from trainer import Trainer

def objective(trial: optuna.Trial) -> float:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_layers = trial.suggest_int('n_layers', 2, 4)
    channels = trial.suggest_int('channels', 16, 64, step=16)
    dropout = trial.suggest_int('dropout', 0, 5) / 100.0
    activation = trial.suggest_categorical('activation', ['relu', 'leaky_relu', 'elu'])
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-5, 1e-2)
    lr = trial.suggest_float('lr', 1e-4, 1e-1, log=True)
    momentum = trial.suggest_float('momentum', 0.5, 0.9)
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'SGD'])
    scheduler_name = trial.suggest_categorical('scheduler', ['StepLR', 'ExponentialLR', 'CosineAnnealingLR'])
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    epochs = trial.suggest_int('epochs', 5, 20, step=5)

    model = FlexibleCNN(
        n_layers=n_layers,
        channels=channels,
        dropout=dropout,
        activation=activation
    ).to(device)
    
    if optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(
            params=model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    else:
        optimizer = torch.optim.SGD(
            params=model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay
        )
    
    if scheduler_name == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=5, gamma=0.5)
    elif scheduler_name == 'ExponentialLR':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.9)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=epochs)

    train_loader, val_loader = get_data_loader(batch_size)
    trainer = Trainer(
        model=model,
        device=device,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler
    )

    for _ in range(epochs):
        trainer.train()
    accuracy = trainer.evaluate()
    return accuracy

def main():
    output_dir = Path(__file__).parent.parent / 'results'
    output_dir.mkdir(exist_ok=True)
    
    db_path = output_dir / 'cnn_optuna.db'
    
    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(),
        storage=f"sqlite:///{db_path}",
        study_name="cifar10_cnn",
        load_if_exists=True
    )
    study.optimize(objective, n_trials=30)
    
    fig = optuna.visualization.plot_optimization_history(study)
    fig.write_image(str(output_dir / 'optimization_history.png'))
    
    fig = optuna.visualization.plot_param_importances(study)
    fig.write_image(str(output_dir / 'param_importances.png'))
    
    fig = optuna.visualization.plot_parallel_coordinate(study)
    fig.write_image(str(output_dir / 'parallel_coordinate.png'))
    
    results = {
        'best_accuracy': float(study.best_value),
        'best_hyperparameters': study.best_params,
        'n_trials': len(study.trials)
    }
    
    with open(output_dir / 'best_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"Best accuracy: {study.best_value:.4f}")
    print("\nBest hyperparameters:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
    print(f"\nResults saved to: {output_dir}")

if __name__ == '__main__':
    main()