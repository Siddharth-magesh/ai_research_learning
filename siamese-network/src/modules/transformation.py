"""
Image transformation pipelines for training and validation.
"""

from torchvision import transforms


def get_train_transform(image_size, mean, std, config):
    """
    Get training data augmentation pipeline.
    
    Args:
        image_size: Tuple of (height, width)
        mean: Normalization mean values
        std: Normalization std values
        config: Configuration object with augmentation parameters
    
    Returns:
        Composed transform for training data
    """
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomAffine(
            degrees=config.random_affine_degrees,
            shear=config.random_affine_shear,
            translate=config.random_affine_translate
        ),
        transforms.RandomPerspective(
            distortion_scale=config.random_perspective_distortion,
            p=config.random_perspective_prob
        ),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])


def get_val_transform(image_size, mean, std):
    """
    Get validation/test data pipeline (no augmentation).
    
    Args:
        image_size: Tuple of (height, width)
        mean: Normalization mean values
        std: Normalization std values
    
    Returns:
        Composed transform for validation/test data
    """
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])