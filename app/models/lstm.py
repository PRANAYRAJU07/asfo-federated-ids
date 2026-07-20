import torch
import torch.nn as nn
from app.models.base import BaseIDSModel
from app.models.registry import ModelRegistry


@ModelRegistry.register("lstm")
class LSTM(BaseIDSModel):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is (batch, features). LSTM expects (batch, seq, features)
        x = x.unsqueeze(1)
        out, _ = self.lstm(x)
        # Take the output of the last time step
        out = out[:, -1, :]
        return self.classifier(out)
