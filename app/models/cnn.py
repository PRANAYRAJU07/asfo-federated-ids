import torch
import torch.nn as nn
from app.models.base import BaseIDSModel
from app.models.registry import ModelRegistry

@ModelRegistry.register("cnn")
class CNN(BaseIDSModel):
    def __init__(self, input_dim: int, output_dim: int, channels: list[int] = [32, 64], kernel_size: int = 3, dropout: float = 0.2):
        super().__init__()
        self.input_dim = input_dim
        layers = []
        in_channels = 1
        
        feature_size = input_dim
        for out_channels in channels:
            layers.append(nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2))
            layers.append(nn.BatchNorm1d(out_channels))
            layers.append(nn.ReLU())
            
            if feature_size >= 2:
                layers.append(nn.MaxPool1d(2))
                feature_size = feature_size // 2
                
            in_channels = out_channels
            
        self.features = nn.Sequential(*layers)
        
        flat_dim = channels[-1] * feature_size if feature_size > 0 else channels[-1]
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is (batch_size, features). Conv1d expects (batch_size, channels, seq_len)
        x = x.unsqueeze(1)
        x = self.features(x)
        x = self.classifier(x)
        return x
