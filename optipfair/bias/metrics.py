"""
Metrics for quantifying bias in model activations.

This module provides functionality to calculate quantitative metrics
of bias by analyzing activation differences between pairs of prompts
that differ only in protected attributes (e.g., race, gender).
"""

import torch
from typing import Dict, List, Any, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)

def calculate_activation_differences(act1: Dict[str, torch.Tensor], act2: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Calculate differences between two sets of activations.
    
    Args:
        act1: Dictionary of activations from first prompt
        act2: Dictionary of activations from second prompt
        
    Returns:
        Dictionary mapping layer names to activation difference tensors
    """
    differences = {}
    
    for key in act1.keys():
        if key in act2:
            # Ensure shapes are compatible
            if act1[key].shape == act2[key].shape:
                # Calculate absolute difference
                diff = torch.abs(act1[key] - act2[key])
                
                # Mean over all dimensions except the first (batch dimension)
                # This preserves token-level differences
                diff_per_token = diff.mean(dim=tuple(range(1, diff.dim())))
                differences[key] = diff_per_token
            else:
                logger.warning(f"Shape mismatch for {key}: {act1[key].shape} vs {act2[key].shape}")
    
    return differences

def calculate_bias_metrics(act1: Dict[str, torch.Tensor], act2: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    """
    Calculate quantitative metrics of bias from activation differences.
    
    Args:
        act1: Dictionary of activations from first prompt
        act2: Dictionary of activations from second prompt
        
    Returns:
        Dictionary of bias metrics
    """
    # Get differences between activations
    differences = calculate_activation_differences(act1, act2)
    
    if not differences:
        logger.warning("No activation differences could be calculated")
        return {}
    
    # Initialize metrics dictionary
    metrics = {
        "layer_metrics": {},
        "overall_metrics": {},
        "component_metrics": {}
    }
    
    # Group layers by component type
    component_groups = {}
    for key in differences.keys():
        # Extract component type from key (e.g., "mlp_output" from "mlp_output_layer_3")
        parts = key.split('_layer_')
        if len(parts) == 2:
            component_type = parts[0]
            if component_type not in component_groups:
                component_groups[component_type] = []
            component_groups[component_type].append(key)
    
    # Calculate metrics for each layer
    for layer_key, diff in differences.items():
        layer_metrics = {
            "mean_difference": float(diff.mean().item()),
            "max_difference": float(diff.max().item()),
            "min_difference": float(diff.min().item()),
            "std_difference": float(diff.std().item()),
            "l1_norm": float(diff.norm(p=1).item()),
            "l2_norm": float(diff.norm(p=2).item())
        }
        metrics["layer_metrics"][layer_key] = layer_metrics
    
    # Calculate metrics for each component type
    for component_type, layer_keys in component_groups.items():
        # Concatenate differences for all layers of this component type
        all_diffs = torch.cat([differences[k] for k in layer_keys])
        
        component_metrics = {
            "mean_difference": float(all_diffs.mean().item()),
            "max_difference": float(all_diffs.max().item()),
            "min_difference": float(all_diffs.min().item()),
            "std_difference": float(all_diffs.std().item()),
            "l1_norm": float(all_diffs.norm(p=1).item()),
            "l2_norm": float(all_diffs.norm(p=2).item()),
            "num_layers": len(layer_keys)
        }
        metrics["component_metrics"][component_type] = component_metrics
    
    # Calculate overall metrics across all layers
    all_diffs = torch.cat([diff for diff in differences.values()])
    
    metrics["overall_metrics"] = {
        "mean_difference": float(all_diffs.mean().item()),
        "max_difference": float(all_diffs.max().item()),
        "min_difference": float(all_diffs.min().item()),
        "std_difference": float(all_diffs.std().item()),
        "l1_norm": float(all_diffs.norm(p=1).item()),
        "l2_norm": float(all_diffs.norm(p=2).item()),
        "total_layers": len(differences)
    }
    
    # Calculate progression metrics (how bias changes across layers)
    for component_type, layer_keys in component_groups.items():
        # Sort keys by layer number
        sorted_keys = sorted(layer_keys, key=lambda k: int(k.split('_')[-1]))
        
        if len(sorted_keys) < 2:
            continue
        
        # Get mean difference for each layer
        mean_diffs = [metrics["layer_metrics"][k]["mean_difference"] for k in sorted_keys]
        
        # Calculate trend metrics
        first_layer = mean_diffs[0]
        last_layer = mean_diffs[-1]
        max_layer = max(mean_diffs)
        
        progression_metrics = {
            "first_to_last_ratio": last_layer / first_layer if first_layer != 0 else float('inf'),
            "max_to_first_ratio": max_layer / first_layer if first_layer != 0 else float('inf'),
            "is_increasing": last_layer > first_layer,
            "layer_with_max_diff": sorted_keys[mean_diffs.index(max_layer)],
            "mean_diffs_by_layer": dict(zip(sorted_keys, mean_diffs))
        }
        
        if "progression_metrics" not in metrics["component_metrics"][component_type]:
            metrics["component_metrics"][component_type]["progression_metrics"] = progression_metrics
    
    return metrics