import json
from typing import Dict, Any


def serialize_distribution(dist: Dict[Any, float]) -> str:
    """Serialize a class distribution dictionary to a JSON string for Flower metrics."""
    # Convert keys to strings to ensure JSON serialization compatibility
    str_dist = {str(k): float(v) for k, v in dist.items()}
    return json.dumps(str_dist)


def deserialize_distribution(dist_str: str) -> Dict[str, float]:
    """Deserialize a class distribution JSON string back to a dictionary."""
    if not dist_str:
        return {}
    return json.loads(dist_str)
