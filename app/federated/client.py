import flwr as fl
import torch
import numpy as np
from loguru import logger
from typing import Dict, List, Tuple

from app.federated.utils import get_parameters, set_parameters, compute_byte_size
from app.training.trainer import Trainer


class IDSClient(fl.client.NumPyClient):
    def __init__(
        self,
        cid: str,
        model: torch.nn.Module,
        train_loader,
        val_loader,
        optimizer: torch.optim.Optimizer,
        criterion,
        device,
        epochs: int,
        metadata: dict = None,
    ):
        self.cid = cid
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs

        self.trainer = Trainer(
            model=self.model,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            callbacks=[],
            log_dir=f"logs/client_{cid}",
        )

        self.metadata = metadata or {}
        self.metadata["cid"] = str(self.cid)
        self.metadata["train_samples"] = len(self.train_loader.dataset)
        self.metadata["val_samples"] = len(self.val_loader.dataset)

    def get_metadata(self):
        return self.metadata

    def get_parameters(self, config: Dict[str, fl.common.Scalar]) -> List[np.ndarray]:
        return get_parameters(self.model)

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        set_parameters(self.model, parameters)

    def fit(
        self, parameters: List[np.ndarray], config: Dict[str, fl.common.Scalar]
    ) -> Tuple[List[np.ndarray], int, Dict]:
        logger.info(f"Client {self.cid}: fit starting")
        self.set_parameters(parameters)

        proximal_mu = config.get("proximal_mu", 0.0)
        global_params = None
        if proximal_mu > 0.0:
            global_params = [
                torch.tensor(p).to(self.trainer.device) for p in parameters
            ]

        self.trainer.fit(
            self.train_loader,
            self.val_loader,
            epochs=self.epochs,
            proximal_mu=proximal_mu,
            global_params=global_params,
        )

        new_params = self.get_parameters(config={})
        local_steps = self.epochs * len(self.train_loader)

        # We record the bytes we are about to upload.
        # Download bytes are recorded on the server side or implicitly via parameters arg size.
        metrics = {
            "bytes_uploaded": compute_byte_size(new_params),
            "local_steps": local_steps,
            **self.get_metadata(),
        }
        return new_params, len(self.train_loader.dataset), metrics

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict[str, fl.common.Scalar]
    ) -> Tuple[float, int, Dict]:
        logger.info(f"Client {self.cid}: evaluate starting")
        self.set_parameters(parameters)

        val_metrics = self.trainer._validate_epoch(self.val_loader, 0)

        loss = val_metrics.get("val_loss", 0.0)
        accuracy = val_metrics.get("val_accuracy", 0.0)

        return (
            float(loss),
            len(self.val_loader.dataset),
            {"accuracy": float(accuracy), **val_metrics},
        )
