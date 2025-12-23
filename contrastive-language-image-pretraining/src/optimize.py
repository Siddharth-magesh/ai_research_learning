import optuna
import torch
import json
from pathlib import Path
from config import Config
from train import Trainer

def objective(trial: optuna.Trial) -> float:
    config = Config()
    
    # HIGH PRIORITY
    config.learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-3)
    config.temperature = trial.suggest_float("temperature", 0.01, 0.1)
    config.batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    config.weight_decay = trial.suggest_loguniform("weight_decay", 1e-6, 1e-3)

    # MEDIUM PRIORITY
    config.text_depth = trial.suggest_categorical("text_depth", [6, 8, 12])
    config.depth = trial.suggest_categorical("depth", [10, 12, 14])
    config.embed_dim = trial.suggest_categorical("embed_dim", [512, 768, 1024])
    config.text_embed_dim = trial.suggest_categorical("text_embed_dim", [384, 512, 768])
    config.dropout = trial.suggest_categorical("vision_dropout", [0.05, 0.1, 0.2])
    config.text_dropout = trial.suggest_categorical("text_dropout", [0.05, 0.1, 0.2])

    # LOW PRIORITY
    config.num_heads = trial.suggest_categorical("num_heads", [8, 12, 16])
    config.text_num_heads = trial.suggest_categorical("text_num_heads", [6, 8, 10])
    config.mlp_ratio = trial.suggest_categorical("mlp_ratio", [2.0, 4.0, 6.0])
    config.text_mlp_ratio = trial.suggest_categorical("text_mlp_ratio", [2.0, 4.0, 6.0])

    config.num_epochs = 5
    config.save_dir = f"./contrastive-language-image-pretraining/optuna_trials/trial_{trial.number}"
    trainer = Trainer(config)
    
    try:
        trainer.load_data(max_samples=5000)
        trainer.build_model()
        trainer.train()
        final_loss = trainer.best_loss
        
        print(f"\nTrial {trial.number} finished with loss: {final_loss:.4f}")
        
        # Return negative loss since Optuna study is set to maximize
        return -final_loss
        
    except Exception as e:
        print(f"\nTrial {trial.number} failed with error: {e}")
        return float('-inf')

def main():
    print("="*60)
    print("CLIP Hyperparameter Optimization with Optuna")
    print("="*60)
    
    output_dir = Path(__file__).parent.parent / 'optuna-results'
    output_dir.mkdir(exist_ok=True, parents=True)

    db_path = output_dir / 'optuna_study.db'
    study = optuna.create_study(
        direction='maximize',  # Maximize negative loss (minimize loss)
        pruner=optuna.pruners.MedianPruner(),
        storage=f"sqlite:///{db_path}",
        study_name="clip_hyperparameter_optimization",
        load_if_exists=True
    )
    
    print(f"\nStarting optimization with {10} trials...")
    print(f"Database: {db_path}")
    print(f"Results will be saved to: {output_dir}\n")
    
    study.optimize(objective, n_trials=10)
    
    print("\n" + "="*60)
    print("Optimization Complete!")
    print("="*60)
    
    print("\nGenerating visualizations...")
    try:
        fig = optuna.visualization.plot_optimization_history(study)
        fig.write_image(str(output_dir / 'optimization_history.png'))
        print(f"✓ Saved: optimization_history.png")
    except Exception as e:
        print(f"✗ Failed to save optimization_history.png: {e}")
    
    try:
        fig = optuna.visualization.plot_param_importances(study)
        fig.write_image(str(output_dir / 'param_importances.png'))
        print(f"✓ Saved: param_importances.png")
    except Exception as e:
        print(f"✗ Failed to save param_importances.png: {e}")
    
    try:
        fig = optuna.visualization.plot_parallel_coordinate(study)
        fig.write_image(str(output_dir / 'parallel_coordinate.png'))
        print(f"✓ Saved: parallel_coordinate.png")
    except Exception as e:
        print(f"✗ Failed to save parallel_coordinate.png: {e}")
    
    results = {
        'best_loss': -float(study.best_value),  # Convert back to positive loss
        'best_hyperparameters': study.best_params,
        'n_trials': len(study.trials),
        'study_name': study.study_name
    }
    
    results_file = output_dir / 'best_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"✓ Saved: best_results.json")
    
    # Save hyperparameters to text file
    hyperparam_file = output_dir / 'best_hyperparameters.txt'
    with open(hyperparam_file, 'w') as f:
        f.write("="*60 + "\n")
        f.write("Best Hyperparameters for CLIP Model\n")
        f.write("="*60 + "\n\n")
        f.write(f"Best Loss: {-study.best_value:.6f}\n")
        f.write(f"Number of Trials: {len(study.trials)}\n\n")
        f.write("Hyperparameters:\n")
        f.write("-"*60 + "\n")
        for k, v in study.best_params.items():
            f.write(f"{k:25s} : {v}\n")
    print(f"✓ Saved: best_hyperparameters.txt")
    
    print("\n" + "="*60)
    print("OPTIMIZATION SUMMARY")
    print("="*60)
    print(f"Best Loss: {-study.best_value:.6f}")
    print(f"Number of Trials: {len(study.trials)}")
    print("\nBest Hyperparameters:")
    print("-"*60)
    for k, v in study.best_params.items():
        print(f"  {k:23s} : {v}")
    print("\n" + "="*60)
    print(f"All results saved to: {output_dir}")
    print("="*60)

if __name__ == '__main__':
    main()