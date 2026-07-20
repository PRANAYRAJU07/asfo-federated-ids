import time
import numpy as np
from typing import Callable, Dict, List, Optional, Tuple, Union

from flwr.common import FitRes, Parameters, Scalar, parameters_to_ndarrays, ndarrays_to_parameters
from flwr.server.client_proxy import ClientProxy

from app.federated.strategy.base import BaseStrategy

class FedNovaStrategy(BaseStrategy):
    def aggregate_fit(self, server_round: int, results: List[Tuple[ClientProxy, FitRes]], failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]]) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            return None, {}
            
        bytes_uploaded = sum([fit_res.metrics.get("bytes_uploaded", 0) for _, fit_res in results])
        agg_start = time.time()
        
        global_params = [p.cpu().numpy() for p in self.global_model.parameters()]
        total_samples = sum([fit_res.num_examples for _, fit_res in results])
        
        accumulated_update = [np.zeros_like(p) for p in global_params]
        total_tau = 0.0
        
        client_updates = []
        client_taus = []
        client_weights = []
        
        for _, fit_res in results:
            client_params = parameters_to_ndarrays(fit_res.parameters)
            # Default to num_examples if local_steps isn't passed
            tau_i = fit_res.metrics.get("local_steps", fit_res.num_examples) 
            
            # delta_i = global - client
            delta_i = [g - c for g, c in zip(global_params, client_params)]
            
            p_i = fit_res.num_examples / total_samples
            
            client_updates.append(delta_i)
            client_taus.append(tau_i)
            client_weights.append(p_i)
            
            total_tau += p_i * tau_i
            
        # FedNova scaling
        for i in range(len(global_params)):
            for j in range(len(results)):
                accumulated_update[i] += (client_weights[j] / max(1e-6, client_taus[j])) * client_updates[j][i]
                
            global_params[i] -= total_tau * accumulated_update[i]
            
        aggregated_parameters = ndarrays_to_parameters(global_params)
        
        # Log and checkpoint
        self._log_and_checkpoint(server_round, aggregated_parameters, results, bytes_uploaded, agg_start)
        
        # We don't have a specific aggregated metrics dictionary from custom FedNova unless we build it
        return aggregated_parameters, {}
