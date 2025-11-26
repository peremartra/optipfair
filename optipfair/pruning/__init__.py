"""
Pruning package for OptiPFair.

This package provides various pruning methods for transformer-based models.
"""

from .depth import analyze_layer_importance, prune_model_depth
from .mlp_glu import prune_model_mlp_glu
from .utils import (
    count_parameters,
    get_model_layers,
    get_pruning_statistics,
    validate_model_for_glu_pruning,
)

__all__ = [
    "prune_model_mlp_glu",
    "prune_model_depth",
    "analyze_layer_importance",
    "validate_model_for_glu_pruning",
    "get_model_layers",
    "count_parameters",
    "get_pruning_statistics",
]
