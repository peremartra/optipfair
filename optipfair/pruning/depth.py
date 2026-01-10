"""
Depth Pruning - Module for removing entire transformer layers from models.

This module provides functionality to prune complete transformer layers,
which is more aggressive than neuron-level pruning but can lead to significant
efficiency gains with proper fine-tuning.
"""

import torch
from torch import nn
import logging
import torch.nn.functional as F
from typing import List, Optional, Union, Tuple, Dict, Any
from tqdm import tqdm
from transformers import PreTrainedModel

from .utils import get_model_layers, count_parameters, _prepare_batch_inputs

logger = logging.getLogger(__name__)


def validate_layer_removal_params(
    model: PreTrainedModel,
    num_layers_to_remove: Optional[int] = None,
    layer_indices: Optional[List[int]] = None,
    depth_pruning_percentage: Optional[float] = None,
    layer_selection_method: str = "last"
) -> Dict[str, Any]:
    """
    Validate parameters for layer removal and return validated configuration.
    
    This function ensures that the layer removal parameters are valid and
    mutually exclusive where appropriate. It follows the same validation
    pattern as the existing MLP pruning functions.
    
    Args:
        model: Pre-trained model to validate
        num_layers_to_remove: Number of layers to remove
        layer_indices: Specific layer indices to remove
        depth_pruning_percentage: Percentage of layers to remove
        layer_selection_method: Method for selecting layers ("last", "first", "custom")
        
    Returns:
        Dictionary with validated parameters and model info
        
    Raises:
        ValueError: If parameters are invalid or mutually exclusive
    """
    # Get model layers using existing utility
    layers = get_model_layers(model)
    if not layers:
        raise ValueError("Could not find transformer layers in the model.")
    
    total_layers = len(layers)
    
    # Count non-None parameters to ensure mutual exclusivity
    param_count = sum(1 for p in [num_layers_to_remove, layer_indices, depth_pruning_percentage] if p is not None)
    
    if param_count == 0:
        raise ValueError("Must specify one of: num_layers_to_remove, layer_indices, or depth_pruning_percentage")
    
    if param_count > 1:
        raise ValueError("Parameters num_layers_to_remove, layer_indices, and depth_pruning_percentage are mutually exclusive")
    
    # Validate layer_selection_method
    valid_methods = ["last", "first", "custom"]
    if layer_selection_method not in valid_methods:
        raise ValueError(f"layer_selection_method must be one of {valid_methods}, got {layer_selection_method}")
    
    # Validate specific parameters
    if num_layers_to_remove is not None:
        if not isinstance(num_layers_to_remove, int) or num_layers_to_remove <= 0:
            raise ValueError("num_layers_to_remove must be a positive integer")
        if num_layers_to_remove >= total_layers:
            raise ValueError(f"Cannot remove {num_layers_to_remove} layers from model with {total_layers} layers")
    
    if depth_pruning_percentage is not None:
        if not 0 < depth_pruning_percentage < 100:
            raise ValueError("depth_pruning_percentage must be between 0 and 100")
        num_layers_to_remove = int(total_layers * depth_pruning_percentage / 100)
        if num_layers_to_remove >= total_layers:
            raise ValueError(f"depth_pruning_percentage {depth_pruning_percentage}% would remove all layers")
        if num_layers_to_remove == 0:
            raise ValueError(f"depth_pruning_percentage {depth_pruning_percentage}% would remove 0 layers")
    
    if layer_indices is not None:
        if not isinstance(layer_indices, list) or not layer_indices:
            raise ValueError("layer_indices must be a non-empty list")
        if not all(isinstance(idx, int) for idx in layer_indices):
            raise ValueError("All layer_indices must be integers")
        if not all(0 <= idx < total_layers for idx in layer_indices):
            raise ValueError(f"All layer_indices must be between 0 and {total_layers-1}")
        if len(set(layer_indices)) != len(layer_indices):
            raise ValueError("layer_indices must not contain duplicates")
        if len(layer_indices) >= total_layers:
            raise ValueError(f"Cannot remove {len(layer_indices)} layers from model with {total_layers} layers")
        
        # For custom indices, override selection method
        layer_selection_method = "custom"
        num_layers_to_remove = len(layer_indices)
    
    return {
        "total_layers": total_layers,
        "num_layers_to_remove": num_layers_to_remove,
        "layer_indices": layer_indices,
        "layer_selection_method": layer_selection_method,
        "layers": layers
    }


def select_layers_to_remove(
    total_layers: int,
    num_layers_to_remove: int,
    layer_selection_method: str,
    custom_indices: Optional[List[int]] = None
) -> List[int]:
    """
    Select which layer indices to remove based on the specified method.
    
    This function implements different strategies for selecting layers,
    similar to how neuron selection methods work in MLP pruning.
    
    Args:
        total_layers: Total number of layers in the model
        num_layers_to_remove: Number of layers to remove
        layer_selection_method: Method for selection ("last", "first", "custom")
        custom_indices: Specific indices when method is "custom"
        
    Returns:
        List of layer indices to remove (sorted)
        
    Raises:
        ValueError: If method is invalid or parameters don't match
    """
    if layer_selection_method == "last":
        # Remove the last N layers (typically best for maintaining model performance)
        return list(range(total_layers - num_layers_to_remove, total_layers))
    
    elif layer_selection_method == "first":
        # Remove the first N layers
        return list(range(0, num_layers_to_remove))
    
    elif layer_selection_method == "custom":
        if custom_indices is None:
            raise ValueError("custom_indices must be provided when layer_selection_method is 'custom'")
        return sorted(custom_indices)
    
    else:
        raise ValueError(f"Unknown layer_selection_method: {layer_selection_method}")


def remove_layers_from_model(
    model: PreTrainedModel,
    layer_indices_to_remove: List[int],
    show_progress: bool = True
) -> PreTrainedModel:
    """
    Remove specified layers from the model.
    
    This function performs the actual layer removal, similar to how
    prune_neuron_pairs works for MLP pruning. It modifies the model
    in-place for memory efficiency.
    
    Args:
        model: Model to modify
        layer_indices_to_remove: Sorted list of layer indices to remove
        show_progress: Whether to show progress bar
        
    Returns:
        Modified model with layers removed
    """
    layers = get_model_layers(model)
    original_layer_count = len(layers)
    
    # Create set for O(1) lookup
    indices_to_remove_set = set(layer_indices_to_remove)
    
    # Build new layer list excluding the layers to remove
    new_layers = []
    
    layer_iterator = tqdm(enumerate(layers), total=len(layers), desc="Removing layers") if show_progress else enumerate(layers)
    
    for idx, layer in layer_iterator:
        if idx not in indices_to_remove_set:
            new_layers.append(layer)
    
    # Replace the model's layers using the same logic as get_model_layers
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        # LLaMA-style models
        model.model.layers = nn.ModuleList(new_layers)
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        # GPT-style models
        model.transformer.h = nn.ModuleList(new_layers)
    elif hasattr(model, 'encoder') and hasattr(model.encoder, 'layer'):
        # BERT-style models
        model.encoder.layer = nn.ModuleList(new_layers)
    elif hasattr(model, 'layers'):
        # Direct layers attribute
        model.layers = nn.ModuleList(new_layers)
    else:
        raise ValueError("Could not determine model architecture for layer replacement")
    
    # Update model configuration
    if hasattr(model, 'config') and hasattr(model.config, 'num_hidden_layers'):
        model.config.num_hidden_layers = len(new_layers)
        logger.info(f"Updated model config: num_hidden_layers = {len(new_layers)}")
    
    logger.info(f"Removed {len(layer_indices_to_remove)} layers. Model now has {len(new_layers)} layers.")
    
    return model

def _infer_layers_path(model):
    """
    Automatically infer the path to transformer layers for different model architectures.
    
    Args:
        model: Pre-trained transformer model
        
    Returns:
        str or None: Path to the transformer layers (e.g., 'model.layers') or None if not found
        
    Examples:
        >>> # For LLaMA/Qwen models: returns 'model.layers'
        >>> # For GPT-2 models: returns 'transformer.h' 
        >>> # For T5 models: returns 'encoder.block' or 'decoder.block'
    """
    # Known patterns for different architectures
    # Order matters: more specific patterns first
    LAYER_PATH_PATTERNS = [
        # Modern architectures (LLaMA family, Qwen, Mistral, Gemma)
        'model.layers',
        
        # GPT-2 family (including DistilGPT2)
        'transformer.h',
        'gpt_neox.layers',
                
        # BERT family
        'encoder.layer',
        'bert.encoder.layer',

        # T5 family
        'encoder.block',
        'decoder.block',
        
        # Other architectures
        'transformer.layers',
        'model.decoder.layers',
        'model.encoder.layers',
        'blocks',
        'layers',
        'h',  # Some older models use just 'h'
    ]
    
    # Try each pattern
    for pattern in LAYER_PATH_PATTERNS:
        try:
            # Navigate through the model using the pattern
            current = model
            parts = pattern.split('.')
            
            # Traverse the path
            for part in parts:
                if hasattr(current, part):
                    current = getattr(current, part)
                else:
                    break
            else:
                # If we successfully traversed all parts, check if it's a layer container
                if _is_valid_layer_container(current):
                    return pattern
                    
        except (AttributeError, TypeError):
            # Continue to next pattern if this one fails
            continue
    
    # If no pattern worked, try to find layers by inspection
    return _find_layers_by_inspection(model)


def _is_valid_layer_container(container):
    """
    Check if a container object holds transformer layers.
    
    Args:
        container: Object that potentially contains transformer layers
        
    Returns:
        bool: True if container has transformer-like layers
    """
    try:
        # Check if it's a container with elements
        if not hasattr(container, '__len__'):
            return False
            
        container_len = len(container)
        if container_len == 0:
            return False
            
        # Get the first element safely
        first_layer = None
        
        # Try different ways to access the first element
        if hasattr(container, '__getitem__'):
            try:
                first_layer = container[0]
            except (KeyError, IndexError, TypeError):
                # Try with different indices or methods
                try:
                    # Some containers might use string keys
                    if hasattr(container, 'keys'):
                        first_key = next(iter(container.keys()))
                        first_layer = container[first_key]
                    else:
                        # Try iterating
                        first_layer = next(iter(container))
                except (StopIteration, KeyError, TypeError):
                    return False
        else:
            try:
                first_layer = next(iter(container))
            except (StopIteration, TypeError):
                return False
                
        if first_layer is None:
            return False
        
        # Look for common transformer layer components
        layer_indicators = [
            # Attention mechanisms
            'self_attn', 'self_attention', 'attention', 'attn',
            # Feed-forward networks  
            'mlp', 'feed_forward', 'ffn',
            # Normalization layers
            'layernorm', 'layer_norm', 'norm', 'ln_1', 'ln_2',
            'input_layernorm', 'post_attention_layernorm'
        ]
        
        # Get attributes safely
        try:
            layer_attrs = [attr.lower() for attr in dir(first_layer)]
        except (AttributeError, TypeError):
            return False
        
        # Check if at least 2 indicators are present (attention + something else)
        indicators_found = sum(1 for indicator in layer_indicators 
                             if any(indicator in attr for attr in layer_attrs))
        
        return indicators_found >= 2
        
    except Exception:
        # Catch any unexpected errors and return False
        return False

def _find_layers_by_inspection(model):
    """
    Fallback method: inspect the model structure to find transformer layers.
    
    Args:
        model: Pre-trained transformer model
        
    Returns:
        str or None: Inferred path to transformer layers or None
    """
    def _inspect_object(obj, current_path="", max_depth=3):
        """Recursively inspect object attributes to find layer containers."""
        if max_depth <= 0:
            return None
            
        try:
            for attr_name in dir(obj):
                # Skip private/special attributes and methods
                if attr_name.startswith('_') or callable(getattr(obj, attr_name, None)):
                    continue
                    
                try:
                    attr_value = getattr(obj, attr_name)
                    new_path = f"{current_path}.{attr_name}" if current_path else attr_name
                    
                    # Check if this attribute is a valid layer container
                    if _is_valid_layer_container(attr_value):
                        return new_path
                        
                    # Recursively inspect this attribute
                    result = _inspect_object(attr_value, new_path, max_depth - 1)
                    if result:
                        return result
                        
                except (AttributeError, TypeError):
                    continue
                    
        except (AttributeError, TypeError):
            pass
            
        return None
    
    # Start inspection from the model root
    return _inspect_object(model)


def _calculate_cosine_importance(input_tensor, output_tensor, layer_idx, is_first_batch=False):
    """
    Calculate importance score using cosine similarity between input and output tensors.
    
    Args:
        input_tensor: Input tensor to the layer
        output_tensor: Output tensor from the layer  
        layer_idx: Layer index (for debugging)
        is_first_batch: Whether this is the first batch (for debugging output)
    
    Returns:
        float: Importance score (0.0 to 1.0), where higher values indicate more importance
    """
    # Validate tensor dimensions
    if input_tensor.numel() == 0 or output_tensor.numel() == 0:
        return 0.0
    
    try:
        # Flatten tensors: [batch_size, features]  
        input_flat = input_tensor.view(input_tensor.size(0), -1)
        output_flat = output_tensor.view(output_tensor.size(0), -1)
        
        # Filter out non-finite values
        input_valid_mask = torch.all(torch.isfinite(input_flat), dim=1)
        output_valid_mask = torch.all(torch.isfinite(output_flat), dim=1)
        valid_mask = input_valid_mask & output_valid_mask
        
        if not valid_mask.any():
            if is_first_batch:
                print(f"Warning: Layer {layer_idx} has all inf/nan samples")
            return 0.0
        
        # Use only valid samples
        input_valid = input_flat[valid_mask]
        output_valid = output_flat[valid_mask]
        
        # Calculate cosine similarity
        similarity = F.cosine_similarity(input_valid, output_valid, dim=1)
        
        # Filter finite similarities and calculate importance
        finite_similarities = similarity[torch.isfinite(similarity)]
        if len(finite_similarities) == 0:
            return 0.0
        
        importance = 1 - finite_similarities.mean().item()
        
        return importance
    
    except Exception as e:
        if is_first_batch:
            print(f"Error in layer {layer_idx}: {e}")
        return 0.0


def _aggregate_importance_scores(layer_scores):
    """
    Aggregate importance scores across all batches.
    
    Args:
        layer_scores: Dict with {layer_idx: [scores_list]}
    
    Returns:
        dict: Dictionary with final averaged scores per layer {layer_idx: avg_score}
    """
    final_scores = {}
    for layer_idx, scores in layer_scores.items():
        if scores:
            # Filter out invalid scores (nan, inf)
            import numpy as np
            valid_scores = [s for s in scores if not (np.isnan(s) or np.isinf(s))]
            final_scores[layer_idx] = np.mean(valid_scores) if valid_scores else 0.0
        else:
            final_scores[layer_idx] = 0.0
    
    return final_scores

def _setup_layer_hooks(model, layers_path):
    """
    Register hooks to capture input/output of each transformer layer.
    
    Args:
        model: Pre-trained transformer model
        layers_path: Path to transformer layers (e.g., 'model.layers', 'transformer.h')
        
    Returns:
        tuple: (hooks, layer_inputs, layer_outputs, num_layers)
    """
    # Get the layers container using the path
    layers_container = model
    for part in layers_path.split('.'):
        layers_container = getattr(layers_container, part)
    
    num_layers = len(layers_container)
    layer_inputs = {}
    layer_outputs = {}
    hooks = []

    def create_input_hook(layer_idx):
        def hook(module, input):
            if isinstance(input, tuple) and len(input) > 0:
                layer_inputs[layer_idx] = input[0].detach()
        return hook

    def create_output_hook(layer_idx):
        def hook(module, input, output):
            if isinstance(output, tuple) and len(output) > 0:
                layer_outputs[layer_idx] = output[0].detach()
            else:
                layer_outputs[layer_idx] = output.detach()
        return hook

    # Register hooks for each layer
    for i, layer in enumerate(layers_container):
        hooks.append(layer.register_forward_pre_hook(create_input_hook(i)))
        hooks.append(layer.register_forward_hook(create_output_hook(i)))

    return hooks, layer_inputs, layer_outputs, num_layers

def analyze_layer_importance(model, dataloader, layers_path=None, show_progress=True):
    """
    Analyze transformer layer importance using cosine similarity between input/output representations.
    
    Args:
        model: Pre-trained transformer model
        dataloader: DataLoader with tokenized text data (prepared by user)
        layers_path: Optional path to transformer blocks (e.g., 'model.layers'). 
                    If None, will try to infer automatically.
        show_progress: Show progress bar during processing (default: True)
    
    Returns:
        dict: Layer importance scores {layer_index: cosine_distance} sorted by layer_index
        
    Examples:
        >>> importance_scores = analyze_layer_importance(model, dataloader)
        >>> print(importance_scores)
        {0: 0.890395, 1: 0.307580, 2: 0.771541, ...}
        
        >>> # Manual layers path
        >>> scores = analyze_layer_importance(model, dataloader, layers_path='transformer.h')
    """
    # Infer device from model
    device = next(model.parameters()).device
    
    # Step 1: Determine layers path
    if layers_path is None:
        layers_path = _infer_layers_path(model)
        if layers_path is None:
            raise ValueError(
                "Could not automatically detect transformer layers. "
                "Please specify layers_path manually (e.g., 'model.layers', 'transformer.h')"
            )
    
    # Step 2: Setup hooks and storage
    hooks, layer_inputs, layer_outputs, num_layers = _setup_layer_hooks(model, layers_path)
    layer_importance_scores = {i: [] for i in range(num_layers)}
    
    try:
        # Step 3: Process all batches with progress tracking
        iterator = tqdm(dataloader, desc="Processing batches") if show_progress else dataloader
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(iterator):
                # Normalize batch format (supports dict, tuple/list, or single tensor)
                inputs = _prepare_batch_inputs(batch, device)
                
                # Forward pass to trigger hooks
                model(**inputs)
                
                # Calculate importance for each layer
                for layer_idx in range(num_layers):
                    if layer_idx not in layer_inputs or layer_idx not in layer_outputs:
                        layer_importance_scores[layer_idx].append(0.0)
                        continue
                    
                    input_tensor = layer_inputs[layer_idx]
                    output_tensor = layer_outputs[layer_idx]
                    
                    importance = _calculate_cosine_importance(
                        input_tensor, output_tensor, layer_idx,
                        is_first_batch=(batch_idx == 0)
                    )
                    
                    layer_importance_scores[layer_idx].append(importance)
                
                # Clear storage for next batch
                layer_inputs.clear()
                layer_outputs.clear()
    
    finally:
        # Step 4: Cleanup hooks (always executed, even if there's an error)
        for hook in hooks:
            hook.remove()
    
    # Step 5: Aggregate final scores
    final_scores = _aggregate_importance_scores(layer_importance_scores)
    
    # Step 6: Return sorted by layer index
    return dict(sorted(final_scores.items()))

def prune_model_depth(
    model: PreTrainedModel,
    num_layers_to_remove: Optional[int] = None,
    layer_indices: Optional[List[int]] = None,
    depth_pruning_percentage: Optional[float] = None,
    layer_selection_method: str = "last",
    show_progress: bool = True,
) -> PreTrainedModel:
    """
    Prune complete transformer layers from a model.
    
    This function removes entire transformer layers, which is more aggressive
    than neuron-level pruning but can lead to significant efficiency gains.
    The function follows the same patterns as prune_model_mlp_glu.
    
    Args:
        model: Pre-trained model to prune
        num_layers_to_remove: Number of layers to remove
        layer_indices: Specific layer indices to remove (mutually exclusive with other options)
        depth_pruning_percentage: Percentage of layers to remove (mutually exclusive with other options)
        layer_selection_method: Method for selecting layers ("last", "custom")
        show_progress: Whether to show progress during pruning
        
    Returns:
        Model with layers removed
        
    Raises:
        ValueError: If parameters are invalid or model is incompatible
    """
    # Validate all parameters
    config = validate_layer_removal_params(
        model=model,
        num_layers_to_remove=num_layers_to_remove,
        layer_indices=layer_indices,
        depth_pruning_percentage=depth_pruning_percentage,
        layer_selection_method=layer_selection_method
    )
    
    # Extract validated parameters
    total_layers = config["total_layers"]
    num_layers_to_remove = config["num_layers_to_remove"]
    layer_indices = config["layer_indices"]
    layer_selection_method = config["layer_selection_method"]
    
    logger.info(f"Starting depth pruning: removing {num_layers_to_remove} layers from {total_layers} total layers")
    
    # Select which layers to remove
    if layer_selection_method == "custom":
        layers_to_remove = layer_indices
    else:
        layers_to_remove = select_layers_to_remove(
            total_layers=total_layers,
            num_layers_to_remove=num_layers_to_remove,
            layer_selection_method=layer_selection_method
        )
    
    logger.info(f"Removing layers: {layers_to_remove} using method '{layer_selection_method}'")
    
    # Perform the actual layer removal
    model = remove_layers_from_model(
        model=model,
        layer_indices_to_remove=layers_to_remove,
        show_progress=show_progress
    )
    
    return model