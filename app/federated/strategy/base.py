import time
import mlflow
import numpy as np
from pathlib import Path
import torch
from loguru import logger
from typing import Callable, Dict, List, Optional, Tuple, Union

from flwr.common import FitRes, Parameters, Scalar, EvaluateRes, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from app.federated.utils import compute_byte_size, set_parameters

def weighted_average_metrics(metrics: List[Tuple[int, Dict[str, Scalar]]]) -> Dict[str, Scalar]:
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}

class BaseStrategy(FedAvg):
    def __init__(self, model: torch.nn.Module, checkpoint_dir: str, **kwargs):
        if "evaluate_metrics_aggregation_fn" not in kwargs:
            kwargs["evaluate_metrics_aggregation_fn"] = weighted_average_metrics
            
        super().__init__(**kwargs)
        self.global_model = model
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.round_start_time = 0.0
        self.last_bytes_total = 1.0  # Avoid div by 0
        self.best_accuracy = 0.0

    def configure_fit(self, server_round: int, parameters: Parameters, client_manager):
        self.round_start_time = time.time()
        return super().configure_fit(server_round, parameters, client_manager)

    def aggregate_fit(self, server_round: int, results: List[Tuple[ClientProxy, FitRes]], failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]]) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        bytes_uploaded = 0
        for _, fit_res in results:
            bytes_uploaded += fit_res.metrics.get("bytes_uploaded", 0)
            
        agg_start = time.time()
        aggregated_parameters, metrics_aggregated = super().aggregate_fit(server_round, results, failures)
        agg_end = time.time()
        
        if aggregated_parameters is not None:
            self._log_and_checkpoint(server_round, aggregated_parameters, results, bytes_uploaded, agg_start)
            
        return aggregated_parameters, metrics_aggregated

    def _log_and_checkpoint(self, server_round, aggregated_parameters, results, bytes_uploaded, agg_start):
        agg_end = time.time()
        ndarrays = parameters_to_ndarrays(aggregated_parameters)
        bytes_downloaded = compute_byte_size(ndarrays) * len(results)
        bytes_total = bytes_uploaded + bytes_downloaded
        self.last_bytes_total = bytes_total
        
        set_parameters(self.global_model, ndarrays)
        
        round_path = self.checkpoint_dir / f"round_{server_round:03d}.pt"
        torch.save(self.global_model.state_dict(), round_path)
        torch.save(self.global_model.state_dict(), self.checkpoint_dir / "last.pt")
        
        agg_time = agg_end - agg_start
        round_wall_time = agg_end - self.round_start_time
        
        mlflow.log_metrics({
            "bytes_uploaded": bytes_uploaded,
            "bytes_downloaded": bytes_downloaded,
            "bytes_total": bytes_total,
            "aggregation_time": agg_time,
            "round_wall_time": round_wall_time
        }, step=server_round)
        
        logger.info(f"Round {server_round} | Wall time: {round_wall_time:.2f}s | Agg time: {agg_time:.2f}s | Comm: {bytes_total / 1024 / 1024:.2f} MB")

    def aggregate_evaluate(self, server_round: int, results: List[Tuple[ClientProxy, EvaluateRes]], failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]]) -> Tuple[Optional[float], Dict[str, Scalar]]:
        loss, metrics = super().aggregate_evaluate(server_round, results, failures)
        if loss is not None and metrics:
            acc = metrics.get("accuracy", 0.0)
            logger.info(f"Round {server_round} Evaluation | Loss: {loss:.4f} | Accuracy: {acc:.4f}")
            
            accuracy_per_mb = acc / max(1e-6, self.last_bytes_total / (1024 * 1024))
            
            mlflow.log_metrics({
                "global_accuracy": acc,
                "global_loss": loss,
                "accuracy_per_mb": accuracy_per_mb
            }, step=server_round)
            
            if acc > self.best_accuracy:
                self.best_accuracy = acc
                torch.save(self.global_model.state_dict(), self.checkpoint_dir / "best.pt")
                
        return loss, metrics
