"""
Siamese Network implementation for signature verification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SiameseNetwork(nn.Module):
    """
    Siamese Network that uses shared weights to embed multiple inputs.
    
    Can operate in two modes:
    - Triplet mode: Takes anchor, positive, and negative samples
    - Pair mode: Takes two images for comparison
    """
    
    def __init__(self, embedding_network):
        """
        Initialize the Siamese Network.
        
        Args:
            embedding_network: Neural network for generating embeddings
        """
        super(SiameseNetwork, self).__init__()
        self.embedding_network = embedding_network
        
    def forward(self, *inputs, triplet_bool=True):
        """
        Forward pass through the Siamese Network.
        
        Args:
            *inputs: Either (anchor, positive, negative) or (img1, img2)
            triplet_bool: If True, expects triplet input; if False, expects pair input
        
        Returns:
            Tuple of embeddings for each input
        """
        if triplet_bool:
            if len(inputs) != 3:
                raise ValueError("Triplet mode requires 3 inputs: anchor, positive, negative")
            anchor, positive, negative = inputs
            z_a = self.embedding_network(anchor)
            z_p = self.embedding_network(positive)
            z_n = self.embedding_network(negative)
            return z_a, z_p, z_n
        else:
            if len(inputs) != 2:
                raise ValueError("Pair mode requires 2 inputs: img1, img2")
            img1, img2 = inputs
            z1 = self.embedding_network(img1)
            z2 = self.embedding_network(img2)
            return z1, z2
    
    def get_embedding(self, image):
        """
        Get embedding for a single image or batch of images.
        
        Args:
            image: Input image tensor
        
        Returns:
            Embedding vector(s)
        """
        return self.embedding_network(image)
    
    def compute_distance(self, embedding1, embedding2, metric='euclidean'):
        """
        Compute distance between two embeddings.
        
        Args:
            embedding1: First embedding tensor
            embedding2: Second embedding tensor
            metric: Distance metric ('euclidean' or 'cosine')
        
        Returns:
            Distance tensor
        """
        if metric == 'euclidean':
            return F.pairwise_distance(embedding1, embedding2, p=2)
        elif metric == 'cosine':
            return 1 - F.cosine_similarity(embedding1, embedding2)
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    def predict_similarity(self, img1, img2, threshold=0.5):
        """
        Predict if two signatures are from the same person.
        
        Args:
            img1: First image tensor
            img2: Second image tensor
            threshold: Distance threshold for classification
        
        Returns:
            Tuple of (prediction, distance) where prediction is True if similar
        """
        self.eval()
        with torch.no_grad():
            z1, z2 = self.forward(img1, img2, triplet_bool=False)
            distance = self.compute_distance(z1, z2)
            prediction = distance < threshold
        return prediction, distance