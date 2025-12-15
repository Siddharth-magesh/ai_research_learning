"""
Data loading and preprocessing utilities for Siamese Network.
"""

import os
import kagglehub
from torch.utils.data import DataLoader
from modules.signature_triplet_dataset import SignatureTripletDataset, create_signature_datasets_splits
from modules.transformation import get_train_transform, get_val_transform


def download_and_setup_data(config):
    """
    Download signature dataset from Kaggle and setup for training.
    
    Args:
        config: Configuration object
    
    Returns:
        Path to the downloaded dataset
    """
    print("=" * 60)
    print("DOWNLOADING DATASET FROM KAGGLE")
    print("=" * 60)
    
    try:
        signature_data_dir = kagglehub.dataset_download("siddharthmagesh/signature-verfication")
        print(f"Dataset downloaded to: {signature_data_dir}")
        
        # Handle nested directory structure (Kaggle sometimes adds an extra folder)
        if not os.path.exists(os.path.join(signature_data_dir, "Real")):
            # Check if there's a nested folder
            subdirs = [d for d in os.listdir(signature_data_dir) if os.path.isdir(os.path.join(signature_data_dir, d))]
            if len(subdirs) == 1:
                signature_data_dir = os.path.join(signature_data_dir, subdirs[0])
                print(f"Found nested directory structure, using: {signature_data_dir}")
        
        # Verify directory structure
        real_dir = os.path.join(signature_data_dir, "Real")
        fake_dir = os.path.join(signature_data_dir, "Fake")
        
        if not os.path.exists(real_dir):
            raise FileNotFoundError(f"Real directory not found: {real_dir}")
        if not os.path.exists(fake_dir):
            raise FileNotFoundError(f"Fake directory not found: {fake_dir}")
        
        print(f"✓ Real signatures directory: {real_dir}")
        print(f"✓ Fake signatures directory: {fake_dir}")
        print("=" * 60)
        
        return signature_data_dir
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        raise


def create_data_loaders(signature_data_dir, config):
    """
    Create training and validation data loaders.
    
    Args:
        signature_data_dir: Path to signature dataset
        config: Configuration object
    
    Returns:
        Tuple of (train_loader, val_loader, train_dataset, val_dataset)
    """
    print("=" * 60)
    print("CREATING DATASETS AND DATA LOADERS")
    print("=" * 60)
    
    # Create full dataset without transforms to build signature map
    print("Building signature map from dataset...")
    full_dataset = SignatureTripletDataset(
        base_data_dir=signature_data_dir,
        triplets_per_user=config.triplets_per_user,
        transform=None
    )
    
    print(f"Total users found: {len(full_dataset.user_ids)}")
    print(f"Total potential triplets: {len(full_dataset)}")
    
    # Create transforms
    train_transform = get_train_transform(
        config.image_size,
        config.image_mean,
        config.image_std,
        config
    )
    
    val_transform = get_val_transform(
        config.image_size,
        config.image_mean,
        config.image_std
    )
    
    # Split into train and validation
    print(f"\nSplitting dataset: {config.train_split:.0%} train, {1-config.train_split:.0%} val")
    train_dataset, val_dataset = create_signature_datasets_splits(
        full_dataset=full_dataset,
        train_split=config.train_split,
        train_transform=train_transform,
        val_transform=val_transform
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Training users: {len(train_dataset.user_ids)}")
    print(f"Validation users: {len(val_dataset.user_ids)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=False
    )
    
    print(f"\nTrain batches per epoch: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print("=" * 60)
    
    return train_loader, val_loader, train_dataset, val_dataset


def print_data_sample(train_loader, config):
    """
    Print information about a sample batch from the data loader.
    
    Args:
        train_loader: Training data loader
        config: Configuration object
    """
    print("=" * 60)
    print("DATA STRUCTURE SAMPLE")
    print("=" * 60)
    
    # Get one batch
    batch = next(iter(train_loader))
    anchor, positive, negative = batch
    
    print(f"Batch size: {anchor.shape[0]}")
    print(f"Anchor shape: {anchor.shape}")
    print(f"Positive shape: {positive.shape}")
    print(f"Negative shape: {negative.shape}")
    print(f"Data type: {anchor.dtype}")
    print(f"Value range: [{anchor.min():.3f}, {anchor.max():.3f}]")
    print(f"Device: {anchor.device}")
    
    # Print statistics
    print(f"\nAnchor statistics:")
    print(f"  Mean: {anchor.mean():.3f}")
    print(f"  Std: {anchor.std():.3f}")
    
    print("=" * 60)
