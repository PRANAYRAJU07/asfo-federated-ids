from app.federated.strategy.base import BaseStrategy


class FedAvgStrategy(BaseStrategy):
    """
    Standard FedAvg, logic is completely handled by BaseStrategy which
    inherits from flwr.server.strategy.FedAvg.
    """

    pass
