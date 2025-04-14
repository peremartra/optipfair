"""
OptiPFair: A library for structured pruning of large language models.

This library implements various pruning techniques for transformer-based language models,
with a focus on maintaining model performance while reducing parameter count.
"""

import logging
from typing import Optional, Union, Dict, Any
from transformers import PreTrainedModel

from .pruning.mlp_glu import prune_model_mlp_glu
from .pruning.utils import get_pruning_statistics

__version__ = "0.1.2"

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def prune_model(
    model: PreTrainedModel,
    pruning_type: str = "MLP_GLU",
    neuron_selection_method: str = "MAW",
    pruning_percentage: Optional[float] = 10,
    expansion_rate: Optional[float] = None,
    show_progress: bool = True,
    return_stats: bool = False,
) -> Union[PreTrainedModel, Dict[str, Any]]:
    """
    Prune a pre-trained language model using the specified pruning method.
    
    Args:
        model: Pre-trained model to prune
        pruning_type: Type of pruning to apply (currently only "MLP_GLU" is supported)
        neuron_selection_method: Method to calculate neuron importance ("MAW", "VOW", or "PON")
        pruning_percentage: Percentage of neurons to prune (0-100)
        expansion_rate: Target expansion rate in percentage (mutually exclusive with pruning_percentage)
        show_progress: Whether to show progress during pruning
        return_stats: Whether to return pruning statistics along with the model
        
    Returns:
        Pruned model or tuple of (pruned_model, statistics) if return_stats is True
    """
    # Keep a copy of the original model parameters for statistics
    original_param_count = None
    if return_stats:
        from copy import deepcopy
        original_model = deepcopy(model)
    
    # Apply the requested pruning method
    if pruning_type == "MLP_GLU":
        pruned_model = prune_model_mlp_glu(
            model=model,
            neuron_selection_method=neuron_selection_method,
            pruning_percentage=pruning_percentage,
            expansion_rate=expansion_rate,
            show_progress=show_progress,
        )
    else:
        supported_types = ["MLP_GLU"]
        raise ValueError(f"Unsupported pruning type: {pruning_type}. Choose from {supported_types}.")
    
    # Return statistics if requested
    if return_stats:
        stats = get_pruning_statistics(original_model, pruned_model)
        return pruned_model, stats
    
    return pruned_model