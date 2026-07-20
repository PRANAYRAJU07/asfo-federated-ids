from app.federated.strategy.base import BaseStrategy
from flwr.common import Parameters


class FedProxStrategy(BaseStrategy):
    """
    FedProx Strategy.
    Injects proximal_mu into client configuration.
    The actual proximal loss term will be handled by the client Trainer.
    """

    def __init__(self, proximal_mu: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.proximal_mu = proximal_mu

    def configure_fit(self, server_round: int, parameters: Parameters, client_manager):
        client_instructions = super().configure_fit(
            server_round, parameters, client_manager
        )

        # Inject proximal_mu into every client's fit config
        for client_proxy, fit_ins in client_instructions:
            fit_ins.config["proximal_mu"] = self.proximal_mu

        return client_instructions
