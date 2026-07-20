from abc import ABC, abstractmethod
import torch.nn as nn
import torch


class BaseIDSModel(nn.Module, ABC):
    """
    Base class for all Intrusion Detection System models.
    Defines the standard interface (just initialization and forward pass).
    Training and evaluation logic is kept separate in Trainer.
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass
