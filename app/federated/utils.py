import torch
import numpy as np
from collections import OrderedDict
from typing import List


def get_parameters(model: torch.nn.Module) -> List[np.ndarray]:
    """Extract model parameters to a list of NumPy arrays."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_parameters(model: torch.nn.Module, parameters: List[np.ndarray]) -> None:
    """Set model parameters from a list of NumPy arrays."""
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


def compute_byte_size(parameters: List[np.ndarray]) -> int:
    """Computes the exact byte size of the parameters list."""
    return sum(p.nbytes for p in parameters)
