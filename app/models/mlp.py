import torch
import torch.nn as nn
from app.models.base import BaseIDSModel
from app.models.registry import ModelRegistry

@ModelRegistry.register("mlp")
class MLP(BaseIDSModel):
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: list[int] = [128, 64], dropout: float = 0.2):
        super().__init__()
        layers = []
        in_dim = input_dim
        
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = h_dim
            
        layers.append(nn.Linear(in_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
