# 1. Use Flower for Federated Learning

Date: 2026-07-20

## Status
Accepted

## Context
We need a robust, production-grade federated learning framework that supports extreme statistical heterogeneity (Non-IID), various cross-domain environments, and multiple underlying ML frameworks if needed.

## Decision
We decided to use **Flower (flwr)** as the core federated learning framework.

## Consequences
- **Pros:** Agnostic to the ML framework (works with PyTorch seamlessly), highly scalable, straightforward server-client architecture, excellent community support, and easy to orchestrate with Docker.
- **Cons:** Requires managing our own server-client infrastructure and communication over network, which we are handling via Docker Compose.
