import torch
import torch.nn as nn
from app.models.base import BaseIDSModel
from app.models.registry import ModelRegistry

@ModelRegistry.register("transformer")
class Transformer(BaseIDSModel):
    def __init__(self, input_dim: int, output_dim: int, d_model: int = 64, nhead: int = 4, num_layers: int = 2, dim_feedforward: int = 128, dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is (batch, features). Transformer expects (batch, seq, features) if batch_first=True
        x = x.unsqueeze(1)
        x = self.embedding(x)
        out = self.transformer_encoder(x)
        # Take last time step
        out = out[:, -1, :]
        return self.classifier(out)
