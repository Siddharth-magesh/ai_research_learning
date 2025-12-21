import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, channels: int, classes: int = 10, dropout: float = 0.1 ) -> None:
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=channels, out_channels=channels * 2, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(in_features = (channels * 2) * 8 * 8, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x , 1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)
    

class FlexibleCNN(nn.Module):
    def __init__(self, n_layers: int, channels: int, dropout: float, activation: str, classes: int = 10) -> None:
        super(FlexibleCNN, self).__init__()
        activation_map = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(),
            'elu': nn.ELU()
        }
        self.activation = activation_map.get(activation, nn.ReLU())
        layers = []
        in_channels = 3
        out_channels = channels

        for _ in range(n_layers):
            layers.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1))
            layers.append(activation_map.get(activation, nn.ReLU()))
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = out_channels
            out_channels *= 2

        self.conv = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(
            in_features=in_channels * (32 // (2 ** n_layers))**2, # in_features = input_channel (3) * (image_size // (2^n_layers))^2 (Height and Width)
            out_features=128
        )
        self.fc2 = nn.Linear(in_features=128, out_features=classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = self.activation(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)