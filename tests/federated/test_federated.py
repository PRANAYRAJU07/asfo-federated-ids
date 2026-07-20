import pytest
import numpy as np
import torch
import os
from tempfile import TemporaryDirectory

from app.federated.utils import compute_byte_size, get_parameters, set_parameters
from app.federated.strategy.base import BaseStrategy
from app.federated.strategy.fedavg import FedAvgStrategy
from app.models.mlp import MLP

def test_compute_byte_size():
    arr1 = np.ones((10, 10), dtype=np.float32)
    arr2 = np.ones((5,), dtype=np.float32)
    # 100 * 4 = 400 bytes, 5 * 4 = 20 bytes -> 420
    assert compute_byte_size([arr1, arr2]) == 420

def test_get_set_parameters():
    model = MLP(input_dim=10, output_dim=2)
    params = [p.copy() for p in get_parameters(model)]
    
    assert len(params) > 0
    
    # Change params
    new_params = [p + 1.0 for p in params]
    set_parameters(model, new_params)
    
    params_after = get_parameters(model)
    assert np.allclose(params_after[0], params[0] + 1.0)
    
def test_strategy_initialization():
    model = MLP(input_dim=10, output_dim=2)
    with TemporaryDirectory() as tmp_dir:
        strategy = FedAvgStrategy(model=model, checkpoint_dir=tmp_dir)
        assert strategy.global_model is not None
        assert os.path.exists(tmp_dir)
