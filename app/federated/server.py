from omegaconf import DictConfig
import torch
from flwr.server.strategy import Strategy

from app.federated.strategy.fedavg import FedAvgStrategy
from app.federated.strategy.fedprox import FedProxStrategy
from app.federated.strategy.fednova import FedNovaStrategy
from app.federated.asfo import ASFOStrategy


def get_strategy(
    cfg: DictConfig, model: torch.nn.Module, checkpoint_dir: str, **kwargs
) -> Strategy:
    """Factory to create the specified Federated Learning strategy."""

    strategy_name = cfg.strategy.name.lower()

    if strategy_name.startswith("asfo"):
        return ASFOStrategy(
            server_learning_rate=cfg.strategy.get("server_learning_rate", 1.0),
            lambda_max=cfg.strategy.get("lambda_max", 0.5),
            temperature=cfg.strategy.get("temperature", 1.0),
            epsilon=cfg.strategy.get("epsilon", 1e-12),
            static_lambda=cfg.strategy.get("static_lambda", None),
            use_knowledge_graph=cfg.strategy.get("use_knowledge_graph", True),
            use_rarity=cfg.strategy.get("use_rarity", True),
            fraction_fit=cfg.federated.fraction_fit,
            fraction_evaluate=cfg.federated.fraction_evaluate,
            min_fit_clients=cfg.federated.min_fit_clients,
            min_evaluate_clients=cfg.federated.min_evaluate_clients,
            min_available_clients=cfg.federated.min_available_clients,
            evaluate_fn=kwargs.get("evaluate_fn"),
        )

    fraction_fit = cfg.federated.get("fraction_fit", 1.0)
    fraction_evaluate = cfg.federated.get("fraction_evaluate", 1.0)
    min_fit_clients = cfg.federated.get("min_fit_clients", 2)
    min_evaluate_clients = cfg.federated.get("min_evaluate_clients", 2)
    min_available_clients = cfg.federated.get("min_available_clients", 2)

    common_kwargs = {
        "model": model,
        "checkpoint_dir": checkpoint_dir,
        "fraction_fit": fraction_fit,
        "fraction_evaluate": fraction_evaluate,
        "min_fit_clients": min_fit_clients,
        "min_evaluate_clients": min_evaluate_clients,
        "min_available_clients": min_available_clients,
        "evaluate_fn": kwargs.get("evaluate_fn"),
    }

    if strategy_name == "fedavg":
        return FedAvgStrategy(**common_kwargs)
    elif strategy_name == "fedprox":
        proximal_mu = cfg.strategy.get("proximal_mu", 0.1)
        return FedProxStrategy(proximal_mu=proximal_mu, **common_kwargs)
    elif strategy_name == "fednova":
        return FedNovaStrategy(**common_kwargs)
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")
