"""
OptiPFair: A library for structured pruning of large language models.

This library implements various pruning techniques for transformer-based language models,
with a focus on maintaining model performance while reducing parameter count.
"""

import logging
from typing import Optional, Union, Dict, Any, Tuple
from transformers import PreTrainedModel

from .pruning.mlp_glu import prune_model_mlp_glu, zero_neurons_mlp
from .pruning.depth import prune_model_depth, analyze_layer_importance
from .pruning.vocab import (
    prune_model_vocab,
    save_vocab_pruned,
    analyse_vocab_usage,
    build_keep_set,
    score_tokens,
    VocabScoringContext,
    TOKEN_SCORING_METHODS,
)
from .distillation import distill_model
from .distillation.mapping import (
    MAPPING_UNIFORM,
    MAPPING_LAST,
)

from .pruning.utils import (
    get_pruning_statistics,
    get_depth_pruning_statistics,
    get_vocab_pruning_statistics,
    count_parameters,
    get_model_layers,
)

__version__ = "0.5.1.dev0"

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
    expansion_divisor: Optional[int] = None,
    dataloader: Optional[Any] = None,
    show_progress: bool = True,
    return_stats: bool = False,
    # Depth pruning parameters
    num_layers_to_remove: Optional[int] = None,
    layer_indices: Optional[list] = None,
    depth_pruning_percentage: Optional[float] = None,
    layer_selection_method: str = "last",
    # Vocabulary pruning parameters
    tokenizer: Optional[Any] = None,
    calibration_dataset: Optional[Any] = None,
    keep_top_k: Optional[int] = None,
    token_selection_method: Any = "frequency",
) -> Union[PreTrainedModel, Tuple[Any, ...]]:
    """
    Prune a pre-trained language model using the specified pruning method.
    
    Args:
        model: Pre-trained model to prune
        pruning_type: Type of pruning to apply ("MLP_GLU", "DEPTH" or "VOCAB")
        neuron_selection_method: Method to calculate neuron importance ("MAW", "VOW", "PON", or "L2") - for MLP_GLU only
        pruning_percentage: Percentage of neurons to prune (0-100) - for MLP_GLU only
        expansion_rate: Target expansion rate in percentage (mutually exclusive with pruning_percentage) - for MLP_GLU only
        expansion_divisor: Optional divisor to round the intermediate layer size (32, 64, 128, 256, or None).
            When specified, the intermediate size will be rounded down to the largest multiple of this value
            after applying pruning_percentage or expansion_rate. Cannot be used alone - requires either
            pruning_percentage or expansion_rate. Only for MLP_GLU pruning.
        dataloader: Optional PyTorch DataLoader for data-driven pruning (MLP_GLU only).
            When provided with neuron_selection_method='MAW', enables hybrid pruning that
            combines weight magnitudes with activation statistics from calibration data.
            Only compatible with 'MAW' method. If None, traditional static pruning is used.
        show_progress: Whether to show progress during pruning
        return_stats: Whether to return pruning statistics along with the model
        num_layers_to_remove: Number of layers to remove - for DEPTH only
        layer_indices: Specific layer indices to operate on. For DEPTH pruning: layers to remove.
            For MLP_GLU pruning: must be None or cover every layer, since all MLPs must keep the same
            intermediate size (selective width pruning raises ValueError). If None, all layers are affected.
        depth_pruning_percentage: Percentage of layers to remove - for DEPTH only
        layer_selection_method: Method for selecting layers ("last", "first", "custom") - for DEPTH only
        tokenizer: Fast BPE tokenizer of the model - required for VOCAB
        calibration_dataset: Representative domain texts (iterable of str, dict rows, or a
            datasets.Dataset) used to measure token usage - required for VOCAB
        keep_top_k: Number of highest-scoring tokens to keep in addition to specials and byte
            tokens - required for VOCAB
        token_selection_method: How tokens are ranked for VOCAB pruning: a name from
            TOKEN_SCORING_METHODS ("frequency", "document_frequency") or a callable
            (VocabScoringContext) -> Dict[int, float] - for VOCAB only

    Returns:
        Pruned model or tuple of (pruned_model, statistics) if return_stats is True.
        For VOCAB the token ids change, so the new tokenizer is always returned too:
        (pruned_model, new_tokenizer) or (pruned_model, new_tokenizer, statistics).
        The pruned model must be used with new_tokenizer.
    """
    # Apply the requested pruning method
    if pruning_type == "MLP_GLU":
        if expansion_rate is not None and pruning_percentage == 10:
            pruning_percentage = 0

        # For MLP_GLU, capture original model via deepcopy
        if return_stats:
            from copy import deepcopy
            original_model = deepcopy(model)
        
        pruned_model = prune_model_mlp_glu(
            model=model,
            neuron_selection_method=neuron_selection_method,
            pruning_percentage=pruning_percentage,
            expansion_rate=expansion_rate,
            expansion_divisor=expansion_divisor,
            dataloader=dataloader,
            layer_indices=layer_indices,
            show_progress=show_progress,
        )
        
        # Return statistics if requested
        if return_stats:
            stats = get_pruning_statistics(original_model, pruned_model)
            return pruned_model, stats
        
        return pruned_model
        
    elif pruning_type == "DEPTH":
        # For DEPTH pruning, capture stats BEFORE pruning to avoid deepcopy issues
        if return_stats:
            original_params = count_parameters(model)
            original_layers = get_model_layers(model)
            original_layer_count = len(original_layers)
        
        pruned_model = prune_model_depth(
            model=model,
            num_layers_to_remove=num_layers_to_remove,
            layer_indices=layer_indices,
            depth_pruning_percentage=depth_pruning_percentage,
            layer_selection_method=layer_selection_method,
            show_progress=show_progress,
        )
        
        # Return statistics if requested
        if return_stats:
            # Calculate layers removed
            layers_removed = original_layer_count - len(get_model_layers(pruned_model))
            stats = get_depth_pruning_statistics(
                original_params=original_params,
                original_layer_count=original_layer_count,
                pruned_model=pruned_model,
                layers_removed=layers_removed,
            )
            return pruned_model, stats
        
        return pruned_model
        
    elif pruning_type == "VOCAB":
        if tokenizer is None or calibration_dataset is None or keep_top_k is None:
            raise ValueError(
                "VOCAB pruning requires tokenizer, calibration_dataset and keep_top_k."
            )
        original_params = count_parameters(model)
        original_vocab_size = model.get_input_embeddings().weight.shape[0]

        pruned_model, new_tokenizer, _ = prune_model_vocab(
            model=model,
            tokenizer=tokenizer,
            calibration_dataset=calibration_dataset,
            keep_top_k=keep_top_k,
            token_selection_method=token_selection_method,
            show_progress=show_progress,
        )

        if return_stats:
            stats = get_vocab_pruning_statistics(
                original_params=original_params,
                original_vocab_size=original_vocab_size,
                pruned_model=pruned_model,
            )
            return pruned_model, new_tokenizer, stats

        return pruned_model, new_tokenizer

    else:
        supported_types = ["MLP_GLU", "DEPTH", "VOCAB"]
        raise ValueError(f"Unsupported pruning type: {pruning_type}. Choose from {supported_types}.")
