"""
Activation capture for bias visualization.

This module provides functionality to capture activations from transformer models
using hooks, which can then be used for bias analysis and visualization.
"""

import torch
import logging
from typing import Dict, List, Any, Callable, Optional, Union

logger = logging.getLogger(__name__)

def register_hooks(model) -> List[Any]:
    """
    Register hooks in the model to capture activations from various components.
    
    This function registers forward hooks on attention mechanisms, MLP blocks,
    and GLU components throughout the model. These hooks capture the activations
    when the model processes input.
    
    Args:
        model: A Hugging Face transformer model
        
    Returns:
        List of hook handles that can be used to remove the hooks later
    """
    # Dictionary to store activations - this will be populated during forward pass
    activations = {}
    model._optipfair_activations = activations
    
    # List to keep track of hook handles
    handles = []
    
    # Function to create hooks for capturing outputs
    def hook_fn(name):
        def hook(module, input, output):
            # Handle self-attention which may return tuple of (attn_output, attn_weights)
            if isinstance(output, tuple):
                # Store just the attention output (first element)
                activations[name] = output[0].detach().cpu()
            else:
                # For regular tensor outputs (MLP components)
                activations[name] = output.detach().cpu()
        return hook
    
    # Try to detect if this is a supported model by checking for common patterns
    try:
        # For LlamaForCausalLM style models (access layers via model.model.layers)
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            layers = model.model.layers
            for i, layer in enumerate(layers):
                # Register hooks for various components
                
                # 1. Hook self-attention output
                if hasattr(layer, "self_attn"):
                    handles.append(layer.self_attn.register_forward_hook(
                        hook_fn(f"attention_output_layer_{i}")
                    ))
                
                # 2. Hook MLP output
                if hasattr(layer, "mlp"):
                    handles.append(layer.mlp.register_forward_hook(
                        hook_fn(f"mlp_output_layer_{i}")
                    ))
                    
                    # 3. Hook GLU components for detailed analysis
                    if hasattr(layer.mlp, "gate_proj"):
                        handles.append(layer.mlp.gate_proj.register_forward_hook(
                            hook_fn(f"gate_proj_layer_{i}")
                        ))
                    
                    if hasattr(layer.mlp, "up_proj"):
                        handles.append(layer.mlp.up_proj.register_forward_hook(
                            hook_fn(f"up_proj_layer_{i}")
                        ))
                    
                    if hasattr(layer.mlp, "down_proj"):
                        handles.append(layer.mlp.down_proj.register_forward_hook(
                            hook_fn(f"down_proj_layer_{i}")
                        ))
                
                # 4. Hook layer norms if present
                if hasattr(layer, "input_layernorm"):
                    handles.append(layer.input_layernorm.register_forward_hook(
                        hook_fn(f"input_norm_layer_{i}")
                    ))
        
        # Add support for other model architectures here
        # e.g., GPT-style models with model.transformer.h
        elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            layers = model.transformer.h
            # Similar hook registration pattern...
            
        # BERT-style models with model.encoder.layer
        elif hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
            layers = model.encoder.layer
            # Similar hook registration pattern...
            
        else:
            logger.warning("Model architecture not fully recognized. Attempting to locate layers...")
            
    except Exception as e:
        logger.warning(f"Error setting up hooks: {e}")
        
    if not handles:
        logger.warning("No hooks were registered. The model architecture may not be supported.")
    
    return handles

def remove_hooks(handles: List[Any]) -> None:
    """
    Remove hooks from a model to stop capturing activations.
    
    Args:
        handles: List of hook handles returned by register_hooks()
    """
    for handle in handles:
        handle.remove()
    logger.info(f"Removed {len(handles)} hooks")

def process_prompt(model: Any, tokenizer: Any, prompt: str) -> Dict[str, torch.Tensor]:
    """
    Process a prompt through the model and capture activations.
    
    Args:
        model: A Hugging Face transformer model
        tokenizer: Matching tokenizer for the model
        prompt: The text prompt to process
        
    Returns:
        Dictionary mapping layer names to activation tensors
    """
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    # print(f"Tokenized '{prompt}' to {inputs['input_ids'].shape[1]} tokens")
    
    # Register activation hooks
    hooks = register_hooks(model)
    
    # Pass through the model without generating additional text
    try:
        with torch.no_grad():
            _ = model(**inputs)
        
        # Copy the activations to prevent overwriting
        result = {k: v.clone() for k, v in model._optipfair_activations.items()}
        # print(f"Captured {len(result)} activation layers")
        
    except Exception as e:
        logger.error(f"Error processing prompt: {e}")
        result = {}  # Return empty dict on error
    finally:
        # Always remove hooks, even if there was an error
        remove_hooks(hooks)
        # Clean up the activations dictionary
        if hasattr(model, "_optipfair_activations"):
            delattr(model, "_optipfair_activations")
    
    return result

def get_activation_pairs(model: Any, tokenizer: Any, prompt1: str, prompt2: str) -> tuple:
    """
    Get activations for a pair of prompts.
    
    Args:
        model: A Hugging Face transformer model
        tokenizer: Matching tokenizer for the model
        prompt1: First prompt text
        prompt2: Second prompt text
        
    Returns:
        Tuple of (activations1, activations2) dictionaries
    """
    # Process each prompt
    activations1 = process_prompt(model, tokenizer, prompt1)
    activations2 = process_prompt(model, tokenizer, prompt2)
    
    return activations1, activations2

def get_layer_names(activations: Dict[str, torch.Tensor], layer_type: Optional[str] = None) -> List[str]:
    """
    Get sorted layer names from activations, optionally filtered by type.
    
    Args:
        activations: Dictionary of activations
        layer_type: Optional type to filter (e.g., "mlp", "attention", "gate_proj")
        
    Returns:
        List of layer names sorted by layer number
    """
    if not activations:
        return []
    
    # Filter by layer type if specified
    if layer_type:
        layer_keys = [k for k in activations.keys() if layer_type in k]
    else:
        layer_keys = list(activations.keys())
    
    if not layer_keys:
        return []
    
    # Extract layer numbers and sort
    layer_dict = {}
    for k in layer_keys:
        parts = k.split('_')
        try:
            layer_num = int(parts[-1])
            layer_dict[k] = layer_num
        except (ValueError, IndexError):
            # Skip keys that don't have a layer number
            continue
    
    # Sort by layer number
    return sorted(layer_dict.keys(), key=lambda k: layer_dict[k])

def select_layers(layer_names: List[str], selection: Union[str, List[int]] = "first_middle_last") -> List[str]:
    """
    Select layers based on the specified selection strategy.
    
    Args:
        layer_names: List of all layer names
        selection: Selection strategy ("first_middle_last", "all", or list of indices)
        
    Returns:
        List of selected layer names
    """
    if not layer_names:
        return []
    
    if selection == "all":
        return layer_names
    
    if selection == "first_middle_last":
        if len(layer_names) < 3:
            return layer_names
        
        # Get first, middle, and last layers
        first = layer_names[0]
        middle = layer_names[len(layer_names) // 2]
        last = layer_names[-1]
        return [first, middle, last]
    
    # If selection is a list of indices
    if isinstance(selection, list):
        selected = []
        for idx in selection:
            if 0 <= idx < len(layer_names):
                selected.append(layer_names[idx])
        return selected
    
    # Default: return first layer only
    return [layer_names[0]] if layer_names else []