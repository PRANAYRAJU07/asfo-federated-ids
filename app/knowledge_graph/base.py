from abc import ABC, abstractmethod
from typing import Dict, List

class ImpactProvider(ABC):
    """Abstract base class for providing semantic impact scores from a Knowledge Graph."""
    
    @abstractmethod
    def get_impact_scores(self, attack_classes: List[str]) -> Dict[str, float]:
        """
        Retrieve impact scores for the given attack classes.
        
        Args:
            attack_classes: List of attack class names (e.g., ["DDoS", "MITM", "0"]).
            
        Returns:
            Dictionary mapping class names to their impact scores (e.g., 0.0 to 10.0).
            If a class is not found, it should return a default score (e.g., 1.0).
        """
        pass
