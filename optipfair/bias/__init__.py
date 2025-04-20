"""
Bias visualization module for OptiPFair.

This module provides tools for visualizing and analyzing how transformer models
process information differently based on protected attributes (e.g., race, gender).
It enables detailed analysis of activation patterns to identify potential bias.
"""

from .visualization import (
    visualize_bias,
    visualize_mean_differences,
    visualize_heatmap,
    visualize_pca,
)
from .metrics import calculate_bias_metrics

__all__ = [
    "visualize_bias",
    "visualize_mean_differences",
    "visualize_heatmap",
    "visualize_pca",
    "calculate_bias_metrics",
]