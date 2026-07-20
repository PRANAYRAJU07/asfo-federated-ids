# 3. Use Ollama for Local LLM Inference

Date: 2026-07-20

## Status
Accepted

## Context
We need to extract entities and relationships from threat descriptions to populate our Knowledge Graph. We require a robust, reproducible local setup for LLM inference to maintain privacy and eliminate cloud dependencies.

## Decision
We decided to use **Ollama** for running local LLMs.

## Consequences
- **Pros:** Completely local and private, simple Docker integration, easy switching between open-source models (Llama 3, Mistral, etc.), and no API costs.
- **Cons:** Heavy resource consumption on the local machine (requires sufficient RAM/VRAM).
