# Contributing to ASFO

Thank you for your interest in contributing to ASFO!

## Development Setup

1. Fork and clone the repository.
2. Install `uv` and run `uv sync --dev`.
3. Set up pre-commit hooks: `uv run pre-commit install`.
4. Ensure tests pass: `uv run pytest`.

## Coding Standards
- Follow PEP 8 (enforced by `black` and `ruff`).
- Write type hints (enforced by `mypy`).
- Add tests for new features.
- Provide comprehensive docstrings.
