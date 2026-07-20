from typing import Dict, Type
import torch.nn as nn
from loguru import logger


class ModelRegistry:
    _registry: Dict[str, Type[nn.Module]] = {}

    @classmethod
    def register(cls, name: str):
        def inner_wrapper(wrapped_class: Type[nn.Module]):
            if name in cls._registry:
                logger.warning(f"Model {name} already registered. Overwriting.")
            cls._registry[name] = wrapped_class
            return wrapped_class

        return inner_wrapper

    @classmethod
    def get_model(cls, name: str, **kwargs) -> nn.Module:
        if name not in cls._registry:
            raise ValueError(f"Model {name} not found in registry: {name}")
        return cls._registry[name](**kwargs)
