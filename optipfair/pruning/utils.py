"""
Utility functions for the OptiPFair library.

This module provides helper functions for model compatibility checking,
layer extraction, and other common tasks needed across different pruning methods.
"""

import torch
from typing import List, Optional, Union, Any, Dict
import logging
from transformers import PreTrainedModel

logger = logging.getLogger(__name__)

def validate_model_for_glu_pruning(model: PreTrainedModel) -> bool:
    """
    Validate that a model is compatible with GLU pruning.
    
    Args:
        model: Model to validate
        
    Returns:
        bool: True if the model is compatible, False otherwise
    """
    # Check if the model has the expected structure
    try:
        layers = get_model_layers(model)
        if not layers:
            logger.warning("Could not find decoder layers in the model")
            return False
        
        # Check the first layer for GLU components
        first_layer = layers[0]
        if not hasattr(first_layer, 'mlp'):
            logger.warning("Model layers do not have 'mlp' attribute")
            return False
        
        mlp = first_layer.mlp
        required_attributes = ['gate_proj', 'up_proj', 'down_proj']
        for attr in required_attributes:
            if not hasattr(mlp, attr):
                logger.warning(f"MLP does not have required attribute: {attr}")
                return False
            
            # Verify these are linear layers
            layer = getattr(mlp, attr)
            if not isinstance(layer, torch.nn.Linear):
                logger.warning(f"{attr} is not a Linear layer")
                return False
        
        # Verify gate_proj and up_proj have the same dimensions
        if mlp.gate_proj.in_features != mlp.up_proj.in_features:
            logger.warning("gate_proj and up_proj have different input dimensions")
            return False
            
        if mlp.gate_proj.out_features != mlp.up_proj.out_features:
            logger.warning("gate_proj and up_proj have different output dimensions")
            return False
            
        if mlp.down_proj.in_features != mlp.gate_proj.out_features:
            logger.warning("down_proj input dimensions don't match gate_proj output dimensions")
            return False
            
        return True
    
    except Exception as e:
        logger.warning(f"Error validating model for GLU pruning: {str(e)}")
        return False

def get_model_layers(model: PreTrainedModel) -> List[Any]:
    """
    Extract transformer layers from a pre-trained model.
    Currently supports LLaMA, Mistral, and similar model architectures.
    
    Args:
        model: Pre-trained model
        
    Returns:
        List of decoder layers that contain MLP blocks
    """
    # Try different attribute paths based on common model architectures
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        # LLaMA, Mistral, and similar architectures
        return list(model.model.layers)
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        # GPT-2 and similar architectures
        return list(model.transformer.h)
    elif hasattr(model, 'encoder') and hasattr(model.encoder, 'layer'):
        # BERT and similar architectures
        return list(model.encoder.layer)
    elif hasattr(model, 'layers'):
        # Direct layers attribute
        return list(model.layers)
        
    logger.warning("Could not find layers in the model")
    return []

def count_parameters(model: torch.nn.Module) -> int:
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_pruning_statistics(
    original_model: torch.nn.Module,
    pruned_model: torch.nn.Module,
) -> Dict[str, Any]:
    """
    Calculate statistics about the pruning operation for width pruning (MLP_GLU).
    
    This function is designed for width pruning operations where neurons/channels
    are pruned from existing layers. For depth pruning (layer removal), use
    get_depth_pruning_statistics() instead, which avoids deepcopy limitations.
    
    Args:
        original_model: Original model before pruning (typically a deepcopy)
        pruned_model: Model after pruning
        
    Returns:
        Dictionary containing pruning statistics with keys:
            - original_parameters: Parameter count before pruning
            - pruned_parameters: Parameter count after pruning
            - reduction: Absolute reduction in parameters
            - percentage_reduction: Percentage reduction in parameters
            - expansion_rate: Expansion rate percentage (for MLP_GLU)
            - pruned_layers: Number of layers where pruning was applied (optional)
            - total_layers: Total number of layers (optional)
            
    Note:
        This function requires a deepcopy of the original model, which can be
        problematic for certain architectures. For depth pruning operations,
        use get_depth_pruning_statistics() instead.
    """
    original_params = count_parameters(original_model)
    pruned_params = count_parameters(pruned_model)
    
    reduction = original_params - pruned_params
    percentage_reduction = (reduction / original_params) * 100
    
    # Get expansion rate and layer information if possible
    expansion_rate = None
    pruned_layer_count = None
    total_layer_count = None
    
    try:
        layers = get_model_layers(pruned_model)
        if layers:
            total_layer_count = len(layers)
            
            # Check for MLP structure
            first_mlp = layers[0].mlp
            intermediate_size = first_mlp.gate_proj.out_features
            hidden_size = first_mlp.gate_proj.in_features
            expansion_rate = (intermediate_size / hidden_size) * 100
            
            # Check if selective pruning was applied (different intermediate sizes)
            intermediate_sizes = set()
            for layer in layers:
                try:
                    intermediate_sizes.add(layer.mlp.gate_proj.out_features)
                except:
                    pass
            
            # If we have multiple intermediate sizes, count how many were pruned
            if len(intermediate_sizes) > 1:
                original_layers = get_model_layers(original_model)
                if original_layers:
                    original_intermediate_size = original_layers[0].mlp.gate_proj.out_features
                    pruned_layer_count = sum(
                        1 for layer in layers
                        if layer.mlp.gate_proj.out_features < original_intermediate_size
                    )
    except Exception:
        pass
    
    stats = {
        "original_parameters": original_params,
        "pruned_parameters": pruned_params,
        "reduction": reduction,
        "percentage_reduction": percentage_reduction,
        "expansion_rate": expansion_rate
    }
    
    # Add selective pruning info if available
    if pruned_layer_count is not None and total_layer_count is not None:
        stats["pruned_layers"] = pruned_layer_count
        stats["total_layers"] = total_layer_count
    
    return stats


def get_depth_pruning_statistics(
    original_params: int,
    original_layer_count: int,
    pruned_model: torch.nn.Module,
    layers_removed: int,
) -> Dict[str, Any]:
    """
    Calculate statistics for depth pruning operations.
    
    This function is specifically designed for depth pruning, where entire layers
    are removed from the model. Unlike get_pruning_statistics(), this function
    does not require the original model to be passed (which avoids deepcopy issues
    with PyTorch ModuleLists in certain architectures).
    
    Args:
        original_params: Parameter count before pruning
        original_layer_count: Number of layers before pruning
        pruned_model: Model after depth pruning
        layers_removed: Number of layers that were removed
        
    Returns:
        Dictionary containing depth pruning statistics with keys:
            - original_parameters: Parameter count before pruning
            - pruned_parameters: Parameter count after pruning
            - reduction: Absolute reduction in parameters
            - percentage_reduction: Percentage reduction in parameters
            - original_layer_count: Number of layers before pruning
            - final_layer_count: Number of layers after pruning
            - layers_removed: Number of layers removed
            - layer_reduction_percentage: Percentage of layers removed
    """
    pruned_params = count_parameters(pruned_model)
    pruned_layers = get_model_layers(pruned_model)
    final_layer_count = len(pruned_layers) if pruned_layers else 0
    
    # Calculate reduction statistics
    reduction = original_params - pruned_params
    percentage_reduction = (reduction / original_params) * 100 if original_params > 0 else 0.0
    layer_reduction_percentage = (layers_removed / original_layer_count) * 100 if original_layer_count > 0 else 0.0
    
    stats = {
        "original_parameters": original_params,
        "pruned_parameters": pruned_params,
        "reduction": reduction,
        "percentage_reduction": percentage_reduction,
        "original_layer_count": original_layer_count,
        "final_layer_count": final_layer_count,
        "layers_removed": layers_removed,
        "layer_reduction_percentage": layer_reduction_percentage,
    }
    
    return stats


def _prepare_batch_inputs(batch: Any, device: torch.device) -> Dict[str, torch.Tensor]:
    """
    Normalize batch data from various DataLoader formats to a unified dict format.
    
    This internal utility supports multiple input formats commonly used with
    transformer models, converting them to a consistent dictionary format
    suitable for model forward passes.
    
    Supported formats:
        - torch.Tensor: Treated as input_ids only
        - dict: Keys extracted directly (e.g., {'input_ids': ..., 'attention_mask': ...})
        - list/tuple: Positional mapping following transformer convention:
            [0] -> input_ids
            [1] -> attention_mask  
            [2] -> token_type_ids
            [3] -> position_ids
            [4] -> head_mask
            [5] -> inputs_embeds
    
    Args:
        batch: Batch data from a DataLoader. Can be a tensor, dict, list, or tuple.
        device: Target device to move tensors to.
        
    Returns:
        Dict[str, torch.Tensor]: Normalized inputs ready for model(**inputs).
        
    Raises:
        ValueError: If batch format is not supported.
        
    Note:
        This is an internal utility function (prefixed with _) and not part
        of the public API. It may change without notice.
    """
    # Standard transformer argument order for positional mapping
    POSITIONAL_KEYS = [
        'input_ids',
        'attention_mask', 
        'token_type_ids',
        'position_ids',
        'head_mask',
        'inputs_embeds',
    ]
    
    inputs: Dict[str, torch.Tensor] = {}
    
    # Case 1: Single tensor - treat as input_ids
    if isinstance(batch, torch.Tensor):
        logger.debug("Single tensor batch detected, treating as input_ids")
        inputs['input_ids'] = batch.to(device)
        return inputs
    
    # Case 2: Dictionary - extract keys directly
    if isinstance(batch, dict):
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(device)
            elif v is not None:
                inputs[k] = v
        return inputs
    
    # Case 3: List or tuple - positional mapping
    if isinstance(batch, (list, tuple)):
        mapped_keys = []
        for idx, value in enumerate(batch):
            if idx >= len(POSITIONAL_KEYS):
                logger.debug(
                    f"Batch has more elements ({len(batch)}) than standard keys "
                    f"({len(POSITIONAL_KEYS)}), ignoring extra elements"
                )
                break
            
            if value is None:
                continue
                
            key = POSITIONAL_KEYS[idx]
            if isinstance(value, torch.Tensor):
                inputs[key] = value.to(device)
                mapped_keys.append(key)
            else:
                inputs[key] = value
                mapped_keys.append(key)
        
        logger.debug(f"Positional mapping applied: {mapped_keys}")
        return inputs
    
    # Unsupported format
    raise ValueError(
        f"Unsupported batch format: {type(batch).__name__}. "
        f"Expected torch.Tensor, dict, list, or tuple."
    )