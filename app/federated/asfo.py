import numpy as np
from typing import Callable, Dict, List, Optional, Tuple, Union
import flwr as fl
from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from loguru import logger
import mlflow
import json
import time

from app.federated.asfo_utils import deserialize_distribution
from app.knowledge_graph.base import ImpactProvider
from app.knowledge_graph.json_provider import JSONProvider

class ASFOStrategy(fl.server.strategy.FedAvg):
    """Adaptive Semantic Federated Optimization (ASFO) Strategy.
    Implements Phase C: Dynamic Lambda, Knowledge Graph Impact, and Global Rarity.
    """
    def __init__(
        self,
        server_learning_rate: float = 1.0,
        lambda_max: float = 0.5,
        temperature: float = 1.0,
        epsilon: float = 1e-12,
        static_lambda: Optional[float] = None,
        use_knowledge_graph: bool = True,
        use_rarity: bool = True,
        impact_provider: Optional[ImpactProvider] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.server_learning_rate = server_learning_rate
        self.lambda_max = lambda_max
        self.temperature = temperature
        self.epsilon = epsilon
        self.static_lambda = static_lambda
        self.use_knowledge_graph = use_knowledge_graph
        self.use_rarity = use_rarity
        
        # Knowledge Graph Integration
        self.impact_provider = impact_provider if impact_provider else JSONProvider()
        
        # State tracking for global updates
        self.global_parameters_cache: Optional[List[np.ndarray]] = None
        self.global_rarity_counts: Dict[str, float] = {}

    def initialize_parameters(
        self, client_manager: fl.server.client_manager.ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""
        initial_parameters = super().initialize_parameters(client_manager)
        if initial_parameters is not None:
            self.global_parameters_cache = parameters_to_ndarrays(initial_parameters)
        return initial_parameters

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using ASFO equations."""
        agg_start_time = time.time()
        
        if not results:
            return None, {}
            
        if self.global_parameters_cache is None:
            logger.warning("Global parameters cache is empty. Using absolute weights for first round.")
            
        client_deltas = []
        client_sizes = []
        client_distributions = []
        
        total_samples = sum([fit_res.num_examples for _, fit_res in results])
        all_classes_in_round = set()

        # Extract Deltas and Distributions
        for _, fit_res in results:
            w_k = parameters_to_ndarrays(fit_res.parameters)
            if self.global_parameters_cache is not None:
                delta_k = [w_layer - g_layer for w_layer, g_layer in zip(w_k, self.global_parameters_cache)]
            else:
                delta_k = w_k 
                
            client_deltas.append(delta_k)
            client_sizes.append(fit_res.num_examples)
            
            dist_str = fit_res.metrics.get("class_distribution", "{}")
            dist = deserialize_distribution(dist_str)
            client_distributions.append(dist)
            
            # Update global rarity state
            for cls, count in dist.items():
                self.global_rarity_counts[cls] = self.global_rarity_counts.get(cls, 0) + count
                all_classes_in_round.add(cls)
                
        # Phase C: Semantic Risk Score Calculation
        
        # 1. Retrieve Impact Scores (I)
        if self.use_knowledge_graph:
            impact_scores = self.impact_provider.get_impact_scores(list(all_classes_in_round))
        else:
            impact_scores = {cls: 1.0 for cls in all_classes_in_round}
            
        # 2. Compute Rarity Vector (R)
        rarity_vector = {}
        for cls in all_classes_in_round:
            if self.use_rarity:
                rarity_vector[cls] = 1.0 / (self.epsilon + self.global_rarity_counts.get(cls, 0))
            else:
                rarity_vector[cls] = 1.0
                
        # 3. Calculate Semantic Risk Score (SRS) for each client
        srs_scores = []
        for dist in client_distributions:
            total_counts = sum(dist.values())
            if total_counts == 0:
                srs_scores.append(0.0)
            else:
                # Normalized frequencies: f_hat_k
                f_hat = {k: v / total_counts for k, v in dist.items()}
                
                # SRS_k = f_hat_k * (I * R)
                srs_k = sum([f_hat.get(cls, 0) * impact_scores.get(cls, 1.0) * rarity_vector.get(cls, 1.0) 
                             for cls in all_classes_in_round])
                srs_scores.append(srs_k)
                
        # Apply Softmax with temperature (beta)
        srs_tensors = np.array(srs_scores) * self.temperature
        exp_srs = np.exp(srs_tensors - np.max(srs_tensors)) # numerical stability
        s_k = exp_srs / np.sum(exp_srs)
        
        # Calculate Semantic Pseudo-Gradient (v_sem)
        v_sem = self._compute_weighted_sum(client_deltas, s_k)
        
        # Phase B: Adaptive Lambda
        flat_vol = np.concatenate([v.flatten() for v in v_vol])
        flat_sem = np.concatenate([v.flatten() for v in v_sem])
        norm_vol = np.linalg.norm(flat_vol)
        norm_sem = np.linalg.norm(flat_sem)
        
        if self.static_lambda is not None:
            lam = self.static_lambda
            cos_sim = 0.0 # Not calculated dynamically
            if norm_vol > 0 and norm_sem > 0:
                 cos_sim = np.dot(flat_vol, flat_sem) / ((norm_vol + self.epsilon) * (norm_sem + self.epsilon))
        else:
            dot_product = np.dot(flat_vol, flat_sem)
            cos_sim = dot_product / ((norm_vol + self.epsilon) * (norm_sem + self.epsilon))
            lam = self.lambda_max * (1.0 - max(0.0, float(cos_sim)))
            
        # Global Update: w_t = w_{t-1} + n * [ (1-lam)*v_vol + lam*v_sem ]
        final_delta = [
            self.server_learning_rate * ((1.0 - lam) * layer_vol + lam * layer_sem)
            for layer_vol, layer_sem in zip(v_vol, v_sem)
        ]
        
        if self.global_parameters_cache is not None:
            new_global_weights = [
                g_layer + d_layer for g_layer, d_layer in zip(self.global_parameters_cache, final_delta)
            ]
        else:
            new_global_weights = final_delta 
            
        self.global_parameters_cache = new_global_weights
        
        mlflow.log_metric("asfo_lambda", float(lam), step=server_round)
        entropy = float(-np.sum(s_k * np.log(s_k + self.epsilon)))
        mlflow.log_metric("semantic_weight_entropy", entropy, step=server_round)
        
        # Extended telemetry
        if len(srs_scores) > 0:
            mlflow.log_metric("mean_srs", float(np.mean(srs_scores)), step=server_round)
            mlflow.log_metric("max_srs", float(np.max(srs_scores)), step=server_round)
            
        rarity_vals = list(rarity_vector.values()) if rarity_vector else [0]
        mlflow.log_metric("rarity_mean", float(np.mean(rarity_vals)), step=server_round)
        mlflow.log_metric("rarity_std", float(np.std(rarity_vals)), step=server_round)
        
        mlflow.log_metric("cosine_similarity", float(cos_sim), step=server_round)
        mlflow.log_metric("gradient_norm_vol", float(norm_vol), step=server_round)
        mlflow.log_metric("gradient_norm_sem", float(norm_sem), step=server_round)
        
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            metrics_aggregated = self.fit_metrics_aggregation_fn(
                [(res.num_examples, res.metrics) for _, res in results]
            )
            
        agg_time = time.time() - agg_start_time
        mlflow.log_metric("aggregation_time", float(agg_time), step=server_round)
        metrics_aggregated["aggregation_time"] = agg_time
            
        return ndarrays_to_parameters(new_global_weights), metrics_aggregated

    def _compute_weighted_sum(self, client_deltas: List[List[np.ndarray]], weights: List[float]) -> List[np.ndarray]:
        """Compute weighted sum of model parameters."""
        weighted_deltas = [
            [layer * w for layer in delta] for delta, w in zip(client_deltas, weights)
        ]
        
        summed_deltas = []
        num_layers = len(client_deltas[0])
        for i in range(num_layers):
            layer_sum = sum(wd[i] for wd in weighted_deltas)
            summed_deltas.append(layer_sum)
            
        return summed_deltas
