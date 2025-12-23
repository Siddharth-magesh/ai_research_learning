"""
Main script to train CLIP model on Flickr30k dataset.
"""

from config import Config
from train import Trainer

def main():
    # Load configuration
    config = Config()
    config.display()
    
    print(f"\nUsing device: {config.device}")
    
    # Initialize trainer
    trainer = Trainer(config)
    
    try:
        # Load dataset
        trainer.load_data(max_samples=None)  # Set max_samples for debugging
        
        # Build model
        trainer.build_model()
        
        # Train model
        model = trainer.train()
        
        print(f"\nTraining completed successfully!")
        print(f"Best loss: {trainer.best_loss:.4f}")
        print(f"Checkpoints saved in: {trainer.checkpoint_dir}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    except Exception as e:
        print(f"An error occurred during training: {e}")
        raise

if __name__ == "__main__":
    main()
