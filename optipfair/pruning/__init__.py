"""
Pruning package for OptiPFair.

This package provides various pruning methods for transformer-based models.
"""

from .mlp_glu import prune_model_mlp_glu, zero_neurons_mlp
from .depth import prune_model_depth, analyze_layer_importance
from .vocab import (
    prune_model_vocab,
    save_vocab_pruned,
    analyse_vocab_usage,
    build_keep_set,
    score_tokens,
    VocabScoringContext,
    TOKEN_SCORING_METHODS,
)
from .utils import (
    validate_model_for_glu_pruning,
    get_model_layers,
    count_parameters,
    get_pruning_statistics,
    get_vocab_pruning_statistics,
)

__all__ = [
    "prune_model_mlp_glu",
    "zero_neurons_mlp",
    "prune_model_depth",
    "analyze_layer_importance",
    "validate_model_for_glu_pruning",
    "get_model_layers",
    "count_parameters",
    "get_pruning_statistics",
    "get_vocab_pruning_statistics",
    "prune_model_vocab",
    "save_vocab_pruned",
    "analyse_vocab_usage",
    "build_keep_set",
    "score_tokens",
    "VocabScoringContext",
    "TOKEN_SCORING_METHODS",
]