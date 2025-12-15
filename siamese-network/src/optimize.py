"""
Hyperparameter optimization using Optuna.
Run this first to find optimal hyperparameters.
"""

import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
from pathlib import Path

from data_loader import download_and_setup_data, create_data_loaders
from modules.embedding_network import SimpleEmbeddingNetwork
from siamese_network import SiameseNetwork
from modules.signature_triplet_dataset import SignatureTripletDataset, create_signature_datasets_splits
from modules.transformation import get_train_transform, get_val_transform
from config import Config


def objective(trial):
    """
    Optuna objective function for hyperparameter optimization.
    
    Args:
        trial: Optuna trial object
    
    Returns:
        Validation accuracy
    """
    # Suggest hyperparameters
    config = Config()
    
    # Image and model params
    image_size_options = [64, 96, 128]
    image_size = trial.suggest_categorical('image_size', image_size_options)
    config.image_size = (image_size, image_size)
    
    config.embedding_dim = trial.suggest_categorical('embedding_dim', [32, 64, 128])
    config.batch_size = trial.suggest_categorical('batch_size', [4, 8, 16])
    
    # Training params
    config.learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    config.weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)
    config.triplet_margin = trial.suggest_float('triplet_margin', 0.5, 2.0)
    
    # Scheduler params
    config.scheduler_gamma = trial.suggest_float('scheduler_gamma', 0.3, 0.7)
    
    # Fixed for optuna (lightweight)
    config.num_epochs = 2
    config.triplets_per_user = 10
    
    device = torch.device(config.device)
    
    # Load data
    signature_data_dir = download_and_setup_data(config)
    train_loader, val_loader, _, _ = create_data_loaders(signature_data_dir, config)
    
    # Build model
    embedding_net = SimpleEmbeddingNetwork(
        embedding_dim=config.embedding_dim,
        input_size=config.image_size
    )
    model = SiameseNetwork(embedding_network=embedding_net).to(device)
    
    # Setup training
    criterion = nn.TripletMarginLoss(margin=config.triplet_margin, p=2)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, 
                          weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, 
                                         gamma=config.scheduler_gamma)
    
    # Training loop
    for epoch in range(config.num_epochs):
        model.train()
        for batch in train_loader:
            anchor, positive, negative = batch
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
            
            optimizer.zero_grad()
            z_a, z_p, z_n = model(anchor, positive, negative, triplet_bool=True)
            loss = criterion(z_a, z_p, z_n)
            loss.backward()
            optimizer.step()
        
        scheduler.step()
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                anchor, positive, negative = batch
                anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
                
                z_a, z_p, z_n = model(anchor, positive, negative, triplet_bool=True)
                
                d_ap = torch.nn.functional.pairwise_distance(z_a, z_p)
                d_an = torch.nn.functional.pairwise_distance(z_a, z_n)
                
                genuine_correct = (d_ap < config.threshold_distance).sum().item()
                fake_correct = (d_an >= config.threshold_distance).sum().item()
                correct += (genuine_correct + fake_correct)
                total += anchor.size(0) * 2
        
        accuracy = correct / total if total > 0 else 0.0
        
        # Report intermediate value
        trial.report(accuracy, epoch)
        
        # Prune trial if needed
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    return accuracy


def run_optuna_study(n_trials=20, study_name='siamese_optimization'):
    """
    Run Optuna hyperparameter optimization study.
    
    Args:
        n_trials: Number of trials to run
        study_name: Name of the study
    """
    print("=" * 60)
    print("OPTUNA HYPERPARAMETER OPTIMIZATION")
    print("=" * 60)
    print(f"Running {n_trials} trials...")
    print("This may take a while...\n")
    
    # Create study
    study = optuna.create_study(
        direction='maximize',
        study_name=study_name,
        pruner=optuna.pruners.MedianPruner()
    )
    
    # Optimize
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    # Results
    print("\n" + "=" * 60)
    print("OPTIMIZATION COMPLETE")
    print("=" * 60)
    
    print(f"\nBest Trial: {study.best_trial.number}")
    print(f"Best Validation Accuracy: {study.best_value:.4f}")
    print("\nBest Hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # Save results to CSV
    df = study.trials_dataframe()
    output_path = Path('optuna_results.csv')
    df.to_csv(output_path, index=False)
    print(f"\n✓ Results saved to: {output_path}")
    
    # Save best params to text file
    best_params_path = Path('best_hyperparameters.txt')
    with open(best_params_path, 'w') as f:
        f.write("Best Hyperparameters from Optuna Study\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Best Validation Accuracy: {study.best_value:.4f}\n\n")
        for key, value in study.best_params.items():
            f.write(f"{key} = {value}\n")
    
    print(f"✓ Best parameters saved to: {best_params_path}")
    
    # Print top 5 trials
    print("\n" + "=" * 60)
    print("TOP 5 TRIALS")
    print("=" * 60)
    df_sorted = df.sort_values('value', ascending=False).head(5)
    print(df_sorted[['number', 'value', 'params_embedding_dim', 'params_batch_size', 
                     'params_learning_rate', 'params_triplet_margin']].to_string(index=False))
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    # Run optimization with 20 trials (adjust as needed)
    run_optuna_study(n_trials=20)
