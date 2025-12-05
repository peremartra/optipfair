import torch
from typing import Dict, Any
from core.compression.pruning.pruning_tools.count_parameters import count_parameters
from core.compression.pruning.pruning_tools.get_model_layers import get_model_layers


def get_pruning_statistics(
    original_model: torch.nn.Module,
    pruned_model: torch.nn.Module,
) -> Dict[str, Any]:
    """
    Calculate statistics about the pruning operation.

    Args:
        original_model: Original model before pruning
        pruned_model: Model after pruning

    Returns:
        Dictionary containing pruning statistics
    """
    original_params = count_parameters(original_model)
    pruned_params = count_parameters(pruned_model)

    reduction = original_params - pruned_params
    percentage_reduction = (reduction / original_params) * 100

    # Get expansion rate if possible
    expansion_rate = None
    try:
        layers = get_model_layers(pruned_model)
        if layers:
            first_mlp = layers[0].mlp
            intermediate_size = first_mlp.gate_proj.out_features
            hidden_size = first_mlp.gate_proj.in_features
            expansion_rate = (intermediate_size / hidden_size) * 100
    except Exception:
        pass

    return {
        "original_parameters": original_params,
        "pruned_parameters": pruned_params,
        "reduction": reduction,
        "percentage_reduction": percentage_reduction,
        "expansion_rate": expansion_rate,
    }
