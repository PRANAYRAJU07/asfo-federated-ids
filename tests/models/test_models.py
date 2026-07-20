import pytest
import torch
import os
from tempfile import TemporaryDirectory
from app.models.registry import ModelRegistry
from app.models.mlp import MLP
from app.models.cnn import CNN
from app.models.lstm import LSTM
from app.models.transformer import Transformer
from app.core.utils import set_seed

_ = [MLP, CNN, LSTM, Transformer]


@pytest.fixture(params=["mlp", "cnn", "lstm", "transformer"])
def model_name(request):
    return request.param


def test_model_serialization(model_name):
    input_dim = 20
    output_dim = 2

    # Initialize Model
    model1 = ModelRegistry.get_model(
        model_name, input_dim=input_dim, output_dim=output_dim
    )
    model1.eval()

    # Create Dummy Data
    x = torch.randn(5, input_dim)

    # Get predictions
    with torch.no_grad():
        pred1 = model1(x)

    # Save Model
    with TemporaryDirectory() as tmp_dir:
        path = os.path.join(tmp_dir, "model.pt")
        torch.save(model1.state_dict(), path)

        # Load Model
        model2 = ModelRegistry.get_model(
            model_name, input_dim=input_dim, output_dim=output_dim
        )
        model2.load_state_dict(torch.load(path))
        model2.eval()

        # Get predictions from loaded model
        with torch.no_grad():
            pred2 = model2(x)

        assert torch.allclose(
            pred1, pred2
        ), f"Model {model_name} failed Save->Load->Predict consistency"


def test_model_determinism():
    input_dim = 20
    output_dim = 2

    # Set Seed 1
    set_seed(42)
    model1 = MLP(input_dim=input_dim, output_dim=output_dim)
    x1 = torch.randn(5, input_dim)
    out1 = model1(x1)

    # Set Seed 1 Again
    set_seed(42)
    model2 = MLP(input_dim=input_dim, output_dim=output_dim)
    x2 = torch.randn(5, input_dim)
    out2 = model2(x2)

    assert torch.allclose(x1, x2), "Random seed did not produce identical inputs"
    assert torch.allclose(
        out1, out2
    ), "Random seed did not produce identical model weights/outputs"
