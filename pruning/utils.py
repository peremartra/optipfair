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
        "expansion_rate": expansion_rate
    }