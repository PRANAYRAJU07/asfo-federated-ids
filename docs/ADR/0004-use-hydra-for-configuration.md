# 4. Use Hydra for Configuration

Date: 2026-07-20

## Status
Accepted

## Context
Research projects in deep learning and FL require managing massive amounts of hyperparameters (datasets, partitions, models, FL strategies). Flat config files quickly become unmanageable.

## Decision
We decided to use **Hydra** (along with Pydantic for validation) for configuration management.

## Consequences
- **Pros:** Hierarchical configurations, easy command-line overrides, composition of configs (e.g., swapping out datasets or partition strategies dynamically), and automatic logging of run configurations.
- **Cons:** Slight learning curve, introduces a dependency on OmegaConf structure.
