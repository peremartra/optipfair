"""
MLPGLUPruning - Module for pruning MLP layers with GLU architecture in transformer models.

This module provides functionality to prune neurons in MLP layers that follow the
Gated Linear Unit (GLU) architecture, as used in models like LLaMA. The pruning
is structured to maintain the paired nature of gate_proj and up_proj layers.
"""

import torch
from torch import nn
import logging
from typing import Tuple, Dict, List, Optional, Callable, Union, Any
from tqdm import tqdm
from transformers import PreTrainedModel

from .utils import validate_model_for_glu_pruning, get_model_layers

logger = logging.getLogger(__name__)

def compute_neuron_pair_importance_maw(gate_weight: torch.Tensor, up_weight: torch.Tensor) -> torch.Tensor:
    """
    Compute neuron pair importance scores using Maximum Absolute Weight method.
    
    Args:
        gate_weight: Weight matrix from the gate_proj layer
        up_weight: Weight matrix from the up_proj layer
        
    Returns:
        importance_scores: Importance scores for each neuron pair
    """
    gate_max_abs = torch.max(gate_weight, dim=1).values + torch.abs(torch.min(gate_weight, dim=1).values)
    up_max_abs = torch.max(up_weight, dim=1).values + torch.abs(torch.min(up_weight, dim=1).values)
    importance_scores = gate_max_abs + up_max_abs
    return importance_scores

def compute_neuron_pair_importance_vow(gate_weight: torch.Tensor, up_weight: torch.Tensor) -> torch.Tensor:
    """
    Compute neuron pair importance scores using Variance of Weights method.
    
    Args:
        gate_weight: Weight matrix from the gate_proj layer
        up_weight: Weight matrix from the up_proj layer
        
    Returns:
        importance_scores: Importance scores for each neuron pair
    """
    gate_variance = torch.var(gate_weight, dim=1)
    up_variance = torch.var(up_weight, dim=1)
    importance_scores = gate_variance + up_variance
    return importance_scores

def compute_neuron_pair_importance_pon(gate_weight: torch.Tensor, up_weight: torch.Tensor) -> torch.Tensor:
    """
    Compute neuron pair importance scores using Product of Norms method.
    
    Args:
        gate_weight: Weight matrix from the gate_proj layer
        up_weight: Weight matrix from the up_proj layer
        
    Returns:
        importance_scores: Importance scores for each neuron pair
    """
    gate_norms = torch.norm(gate_weight, p=1, dim=1)
    up_norms = torch.norm(up_weight, p=1, dim=1)
    importance_scores = gate_norms * up_norms
    return importance_scores

# Dictionary mapping method names to their respective functions
IMPORTANCE_FUNCTIONS = {
    "MAW": compute_neuron_pair_importance_maw,
    "VOW": compute_neuron_pair_importance_vow,
    "PON": compute_neuron_pair_importance_pon,
}

def prune_neuron_pairs(
    mlp: nn.Module,
    prune_percentage: float,
    importance_fn: Callable = compute_neuron_pair_importance_maw
) -> Tuple[nn.Linear, nn.Linear, nn.Linear, int]:
    """
    Prune a specific percentage of neurons from the MLP layers (GLU architecture).
    
    Args:
        mlp: MLP module containing gate_proj, up_proj, and down_proj layers
        prune_percentage: Percentage of neurons to prune (0-100)
        importance_fn: Function to compute neuron pair importance
        
    Returns:
        new_gate_proj: Pruned gate_proj layer
        new_up_proj: Pruned up_proj layer
        new_down_proj: Pruned down_proj layer
        k: New intermediate size after pruning
    """
    # Extract the weights from the MLP layers
    gate_weight = mlp.gate_proj.weight.data.float()
    up_weight = mlp.up_proj.weight.data.float()
    
    # Compute importance scores for neuron pairs
    importance_scores = importance_fn(gate_weight, up_weight)
    
    # Determine the new intermediate size
    original_intermediate_size = gate_weight.size(0)
    num_neuron_pairs_to_prune = min(int(prune_percentage / 100 * original_intermediate_size), original_intermediate_size - 1)
    k = original_intermediate_size - num_neuron_pairs_to_prune
    
    # Validate the new size
    if k <= 0:
        raise ValueError(f"Invalid number of neuron pairs to keep: {k}. Reduce pruning percentage.")
    
    # Select the neurons to keep based on importance scores
    _, indices_to_keep = torch.topk(importance_scores, k, largest=True)
    indices_to_keep = indices_to_keep.sort().values
    
    # Create new layers with reduced dimensions
    new_gate_proj = nn.Linear(mlp.gate_proj.in_features, k, bias=mlp.gate_proj.bias is not None)
    new_up_proj = nn.Linear(mlp.up_proj.in_features, k, bias=mlp.up_proj.bias is not None)
    new_down_proj = nn.Linear(k, mlp.down_proj.out_features, bias=mlp.down_proj.bias is not None)
    
    # Copy selected weights to the new layers
    new_gate_proj.weight.data = gate_weight[indices_to_keep, :]
    if mlp.gate_proj.bias is not None:
        new_gate_proj.bias.data = mlp.gate_proj.bias.data[indices_to_keep]
    
    new_up_proj.weight.data = up_weight[indices_to_keep, :]
    if mlp.up_proj.bias is not None:
        new_up_proj.bias.data = mlp.up_proj.bias.data[indices_to_keep]
    
    new_down_proj.weight.data = mlp.down_proj.weight.data[:, indices_to_keep]
    if mlp.down_proj.bias is not None:
        new_down_proj.bias.data = mlp.down_proj.bias.data.clone()
    
    return new_gate_proj, new_up_proj, new_down_proj, k

def calculate_pruning_percentage_from_expansion_rate(
    current_intermediate_size: int,
    current_hidden_size: int,
    target_expansion_rate: float
) -> float:
    """
    Calculate the pruning percentage needed to achieve a target expansion rate.
    
    Args:
        current_intermediate_size: Current size of the intermediate layer
        current_hidden_size: Current size of the hidden layer
        target_expansion_rate: Target expansion rate in percentage (e.g., 140 for 140%)
        
    Returns:
        pruning_percentage: Percentage of neurons to prune
    """
    current_expansion_rate = (current_intermediate_size / current_hidden_size) * 100
    target_intermediate_size = (target_expansion_rate / 100) * current_hidden_size
    
    if target_intermediate_size >= current_intermediate_size:
        raise ValueError(
            f"Target expansion rate ({target_expansion_rate}%) would increase the model size. "
            f"Current expansion rate is {current_expansion_rate:.2f}%."
        )
    
    pruning_percentage = (1 - (target_intermediate_size / current_intermediate_size)) * 100
    return pruning_percentage

def prune_model_mlp_glu(
    model: PreTrainedModel,
    neuron_selection_method: str = "MAW",
    pruning_percentage: Optional[float] = 10,
    expansion_rate: Optional[float] = None,
    show_progress: bool = True,
) -> PreTrainedModel:
    """
    Prune the MLP layers in a model with GLU architecture.
    
    Args:
        model: Pre-trained model to prune
        neuron_selection_method: Method to use for calculating neuron importance ("MAW", "VOW", or "PON")
        pruning_percentage: Percentage of neurons to prune (0-100)
        expansion_rate: Target expansion rate in percentage (mutually exclusive with pruning_percentage)
        show_progress: Whether to show progress during pruning
        
    Returns:
        model: Pruned model
    """
    # Validate the model for GLU pruning
    if not validate_model_for_glu_pruning(model):
        raise ValueError("Model is not compatible with GLU pruning. It must have gate_proj, up_proj, and down_proj layers.")
    
    # Select the appropriate importance function
    if neuron_selection_method not in IMPORTANCE_FUNCTIONS:
        raise ValueError(f"Invalid neuron selection method: {neuron_selection_method}. "
                         f"Choose from {list(IMPORTANCE_FUNCTIONS.keys())}.")
    
    importance_fn = IMPORTANCE_FUNCTIONS[neuron_selection_method]
    
    # Handle mutually exclusive parameters
    if pruning_percentage is not None and expansion_rate is not None:
        raise ValueError("pruning_percentage and expansion_rate are mutually exclusive. Provide only one.")
    
    if expansion_rate is not None:
        # Get the first MLP layer to calculate current expansion rate
        layers = get_model_layers(model)
        if not layers:
            raise ValueError("Could not find MLP layers in the model.")
        
        first_mlp = layers[0].mlp
        current_intermediate_size = first_mlp.gate_proj.out_features
        current_hidden_size = first_mlp.gate_proj.in_features
        
        pruning_percentage = calculate_pruning_percentage_from_expansion_rate(
            current_intermediate_size, current_hidden_size, expansion_rate
        )
        
        logger.info(f"Calculated pruning percentage: {pruning_percentage:.2f}% to achieve "
                   f"expansion rate of {expansion_rate}%")
    
    # Ensure pruning_percentage is within valid range
    if not 0 <= pruning_percentage <= 100:
        raise ValueError(f"pruning_percentage must be between 0 and 100, got {pruning_percentage}")
    
    # Get all layers to prune
    layers = get_model_layers(model)
    if not layers:
        raise ValueError("Could not find MLP layers in the model.")
    
    new_intermediate_size = None
    
    # Wrap with tqdm if show_progress is True
    layer_iterator = tqdm(enumerate(layers), total=len(layers), desc="Pruning layers") if show_progress else enumerate(layers)
    
    # Iterate through layers and apply pruning
    for idx, layer in layer_iterator:
        mlp = layer.mlp
        
        # Prune neuron pairs in the MLP layer
        new_gate_proj, new_up_proj, new_down_proj, new_size = prune_neuron_pairs(
            mlp, pruning_percentage, importance_fn
        )
        
        # Replace original layers with pruned layers
        mlp.gate_proj = new_gate_proj
        mlp.up_proj = new_up_proj
        mlp.down_proj = new_down_proj
        
        # Store the new intermediate size (should be the same for all layers)
        if new_intermediate_size is None:
            new_intermediate_size = new_size
    
    # Update model configuration
    if hasattr(model, "config") and hasattr(model.config, "intermediate_size"):
        model.config.intermediate_size = new_intermediate_size
        logger.info(f"Updated model config: intermediate_size = {new_intermediate_size}")
    
    return model