import json
import os
from typing import Dict, List
from loguru import logger

from app.knowledge_graph.base import ImpactProvider


class JSONProvider(ImpactProvider):
    """Retrieves impact scores from a static JSON knowledge graph file."""

    def __init__(self, filepath: str = "knowledge_graph.json"):
        self.filepath = filepath
        self.kg: Dict[str, float] = {}
        self._load_kg()

    def _load_kg(self):
        if os.path.exists(self.filepath):
            with open(self.filepath, "r") as f:
                self.kg = json.load(f)
            logger.info(
                f"Loaded Knowledge Graph with {len(self.kg)} entries from {self.filepath}"
            )
        else:
            logger.warning(
                f"Knowledge Graph file {self.filepath} not found. Using default scores."
            )

    def get_impact_scores(self, attack_classes: List[str]) -> Dict[str, float]:
        """Retrieve scores, defaulting to 1.0 for unknown classes to maintain mathematical stability."""
        scores = {}
        for cls in attack_classes:
            scores[str(cls)] = float(self.kg.get(str(cls), 1.0))
        return scores
