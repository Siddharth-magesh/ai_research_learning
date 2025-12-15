"""
Embedding network architecture for signature feature extraction.
"""

import torch.nn as nn
import torch.nn.functional as F


class SimpleEmbeddingNetwork(nn.Module):
    """
    CNN-based embedding network for extracting signature features.
    
    Architecture:
    - 4 convolutional blocks with batch normalization
    - Progressive feature map increase: 3 -> 32 -> 64 -> 128 -> 256
    - Fully connected layers for embedding generation
    - L2 normalization on output embeddings
    """
    
    def __init__(self, embedding_dim=128, input_size=(128, 128)):
        """
        Initialize the embedding network.
        
        Args:
            embedding_dim: Dimension of the output embedding vector
            input_size: Expected input image size (height, width)
        """
        super(SimpleEmbeddingNetwork, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.input_size = input_size
        
        # Convolutional blocks
        self.conv1 = self._conv_block(3, 32, kernel_size=5, padding=2)
        self.conv2 = self._conv_block(32, 64, kernel_size=5, padding=2)
        self.conv3 = self._conv_block(64, 128, kernel_size=3, padding=1)
        self.conv4 = self._conv_block(128, 256, kernel_size=3, padding=1)
        
        # Calculate the size after convolutions
        # Each maxpool reduces size by 2
        final_size = input_size[0] // 16  # 4 maxpool layers (2^4 = 16)
        flattened_size = 256 * final_size * final_size
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(flattened_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(256, embedding_dim)
        )
    
    def _conv_block(self, in_channels, out_channels, kernel_size, padding):
        """
        Create a convolutional block with Conv2d, BatchNorm, ReLU, and MaxPool.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Convolution kernel size
            padding: Padding size
        
        Returns:
            Sequential module containing the conv block
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25)
        )
    
    def forward(self, x):
        """
        Forward pass through the embedding network.
        
        Args:
            x: Input images tensor of shape (batch_size, 3, H, W)
        
        Returns:
            L2-normalized embedding vectors of shape (batch_size, embedding_dim)
        """
        # Convolutional feature extraction
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected embedding
        x = self.fc(x)
        
        # L2 normalization for stable distance computation
        x = F.normalize(x, p=2, dim=1)
        
        return x
    
    def get_embedding_dim(self):
        """Return the embedding dimension."""
        return self.embedding_dim