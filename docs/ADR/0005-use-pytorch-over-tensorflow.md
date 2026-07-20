# 5. Use PyTorch over TensorFlow

Date: 2026-07-20

## Status
Accepted

## Context
We need a deep learning framework for baseline IDS models (CNN, LSTM, Transformers) and integrating with the federated learning strategy.

## Decision
We decided to use **PyTorch**.

## Consequences
- **Pros:** De facto standard for AI research, dynamic computation graph makes debugging and custom FL gradient operations (divergence, drift) much easier, excellent support for SHAP and Flower.
- **Cons:** Slightly steeper learning curve for deployment compared to TFLite, but research flexibility is paramount for the ASFO algorithm.
