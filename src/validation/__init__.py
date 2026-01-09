"""Validation module for comparing model predictions with external data sources."""

from .immobiliare_validator import (
    ImmobiliareValidator,
    compare_with_model,
    compare_with_omi,
)

__all__ = [
    "ImmobiliareValidator",
    "compare_with_omi",
    "compare_with_model",
]
