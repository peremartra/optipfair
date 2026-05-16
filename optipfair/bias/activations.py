"""
Activation capture for bias visualization.

This module provides functionality to capture activations from transformer models
using hooks, which can then be used for bias analysis and visualization.
"""

import torch
import logging
from typing import Dict, List, Any, Callable, Optional, Union, Tuple
from tqdm import tqdm

from optipfair.pruning.utils import get_model_layers
from optipfair.pruning.mlp_glu import compute_neuron_pair_importance_maw

logger = logging.getLogger(__name__)

# Allowed target layers to prevent accidental string matching.
# Includes "input_norm" for backward compatibility with the current hook
# that captures input_layernorm activations.
ALLOWED_TARGET_LAYERS = frozenset({
    "gate_proj", "up_proj", "down_proj", "down_proj_input", "mlp_output", "attention", "input_norm"
})

# Layers that must be explicitly requested and are never captured by default
# (i.e. when target_layers=None). Add new opt-in-only layers here.
_OPT_IN_ONLY_LAYERS = frozenset({"down_proj_input"})


def _should_register(hook_prefix: str, target_layers: Optional[List[str]]) -> bool:
    """Check if a hook with the given prefix should be registered."""
    if target_layers is None:
        return hook_prefix not in _OPT_IN_ONLY_LAYERS
    return hook_prefix in target_layers


def register_hooks(model, target_layers: Optional[List[str]] = None) -> List[Any]:
    """
    Register hooks in the model to capture activations from various components.

    This function registers forward hooks on attention mechanisms, MLP blocks,
    and GLU components throughout the model. These hooks capture the activations
    when the model processes input.

    Args:
        model: A Hugging Face transformer model
        target_layers: Optional list of layer type prefixes to capture.
            Valid values: "gate_proj", "up_proj", "down_proj", "down_proj_input",
            "mlp_output", "attention", "input_norm". If None, all legacy
            supported layers are captured.

    Returns:
        List of hook handles that can be used to remove the hooks later

    Raises:
        ValueError: If target_layers contains invalid layer type names.
    """
    # Validate target_layers
    if target_layers is not None:
        invalid = set(target_layers) - ALLOWED_TARGET_LAYERS
        if invalid:
            raise ValueError(
                f"Invalid target_layers: {sorted(invalid)}. "
                f"Valid options are: {sorted(ALLOWED_TARGET_LAYERS)}"
            )

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

    # Function to create pre-hooks for capturing module inputs
    def hook_fn_input(name):
        def hook(module, inputs):
            activations[name] = inputs[0].detach().cpu()
        return hook

    # Use get_model_layers for multi-architecture support
    layers = get_model_layers(model)

    if not layers:
        logger.warning(
            "No hooks were registered. The model architecture may not be supported."
        )
        return handles

    for i, layer in enumerate(layers):
        # 1. Hook self-attention output
        if _should_register("attention", target_layers):
            if hasattr(layer, "self_attn"):
                handles.append(
                    layer.self_attn.register_forward_hook(
                        hook_fn(f"attention_output_layer_{i}")
                    )
                )

        # 2. Hook MLP output
        if hasattr(layer, "mlp"):
            if _should_register("mlp_output", target_layers):
                handles.append(
                    layer.mlp.register_forward_hook(
                        hook_fn(f"mlp_output_layer_{i}")
                    )
                )

            # 3. Hook GLU components for detailed analysis
            if _should_register("gate_proj", target_layers):
                if hasattr(layer.mlp, "gate_proj"):
                    handles.append(
                        layer.mlp.gate_proj.register_forward_hook(
                            hook_fn(f"gate_proj_layer_{i}")
                        )
                    )

            if _should_register("up_proj", target_layers):
                if hasattr(layer.mlp, "up_proj"):
                    handles.append(
                        layer.mlp.up_proj.register_forward_hook(
                            hook_fn(f"up_proj_layer_{i}")
                        )
                    )

            if _should_register("down_proj", target_layers):
                if hasattr(layer.mlp, "down_proj"):
                    handles.append(
                        layer.mlp.down_proj.register_forward_hook(
                            hook_fn(f"down_proj_layer_{i}")
                        )
                    )

            if _should_register("down_proj_input", target_layers):
                if hasattr(layer.mlp, "down_proj"):
                    handles.append(
                        layer.mlp.down_proj.register_forward_pre_hook(
                            hook_fn_input(f"down_proj_input_layer_{i}")
                        )
                    )

        # 4. Hook layer norms if present
        if _should_register("input_norm", target_layers):
            if hasattr(layer, "input_layernorm"):
                handles.append(
                    layer.input_layernorm.register_forward_hook(
                        hook_fn(f"input_norm_layer_{i}")
                    )
                )

    if not handles:
        logger.warning(
            "No hooks were registered. The model architecture may not be supported."
        )

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

def process_prompt(
    model: Any,
    tokenizer: Any,
    prompt: str,
    target_layers: Optional[List[str]] = None,
) -> Dict[str, torch.Tensor]:
    """
    Process a prompt through the model and capture activations.

    Args:
        model: A Hugging Face transformer model
        tokenizer: Matching tokenizer for the model
        prompt: The text prompt to process
        target_layers: Optional list of layer type prefixes to capture.
            If None, captures all supported layers.

    Returns:
        Dictionary mapping layer names to activation tensors
    """
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Register activation hooks
    hooks = register_hooks(model, target_layers=target_layers)

    # Pass through the model without generating additional text
    try:
        with torch.no_grad():
            _ = model(**inputs)

        # Copy the activations to prevent overwriting
        result = {k: v.clone() for k, v in model._optipfair_activations.items()}

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


def get_activation_pairs(
    model: Any,
    tokenizer: Any,
    prompt1: str,
    prompt2: str,
    target_layers: Optional[List[str]] = None,
) -> tuple:
    """
    Get activations for a pair of prompts.

    Args:
        model: A Hugging Face transformer model
        tokenizer: Matching tokenizer for the model
        prompt1: First prompt text
        prompt2: Second prompt text
        target_layers: Optional list of layer type prefixes to capture.
            If None, captures all supported layers.

    Returns:
        Tuple of (activations1, activations2) dictionaries
    """
    # Process each prompt
    activations1 = process_prompt(model, tokenizer, prompt1, target_layers=target_layers)
    activations2 = process_prompt(model, tokenizer, prompt2, target_layers=target_layers)

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


# ==============================================================================
# NEURON BIAS ANALYSIS & FAIRNESS PRUNING SCORES
# ==============================================================================


def analyze_neuron_bias(
    model: Any,
    tokenizer: Any,
    prompt_pairs: List[Tuple[str, str]],
    target_layers: Optional[List[str]] = None,
    aggregation: str = "mean",
    show_progress: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Analyze per-neuron bias across multiple prompt pairs.

    Processes a batch of prompt pairs that differ only in a demographic attribute,
    computes activation differences per neuron, and aggregates them into a single
    BiasScore per neuron per layer.

    This function robustly handles varying sequence lengths between prompt pairs
    by applying mean pooling over all dimensions except the neuron dimension
    before computing differences.

    Args:
        model: A Hugging Face CausalLM model.
        tokenizer: Matching tokenizer for the model.
        prompt_pairs: List of (prompt_1, prompt_2) tuples differing only in
            demographic attribute.
        target_layers: Optional list of layer type prefixes to capture.
            If None, defaults to ["gate_proj", "up_proj"] (the layers relevant
            for MLP GLU fairness pruning). Note: this differs from register_hooks
            where None means "all layers".
        aggregation: How to aggregate across pairs: "mean" or "max".
        show_progress: Whether to show a tqdm progress bar.

    Returns:
        Dictionary mapping layer keys (e.g. "gate_proj_layer_0") to per-neuron
        aggregated BiasScore tensors of shape [intermediate_size]. All tensors
        are on CPU.

    Raises:
        ValueError: If prompt_pairs is empty or aggregation is invalid.
    """
    # Resolve default target_layers for fairness pruning use case
    if target_layers is None:
        target_layers = ["gate_proj", "up_proj"]

    # Validate aggregation
    if aggregation not in ("mean", "max"):
        raise ValueError(
            f"aggregation must be 'mean' or 'max', got '{aggregation}'"
        )

    # Validate prompt pairs
    if not prompt_pairs:
        raise ValueError("prompt_pairs cannot be empty")

    pairs = prompt_pairs

    # Initialize accumulator: layer_key -> List[Tensor of shape [intermediate_size]]
    accumulated_diffs: Dict[str, List[torch.Tensor]] = {}

    for prompt_1, prompt_2 in tqdm(
        pairs,
        disable=not show_progress,
        desc="Analyzing bias across prompt pairs",
    ):
        act1, act2 = get_activation_pairs(
            model, tokenizer, prompt_1, prompt_2, target_layers=target_layers
        )

        for layer_key in act1:
            if layer_key not in act2:
                continue

            # STRATEGY FOR VARYING SEQUENCE LENGTHS & BATCH SIZE > 1:
            # We cannot do act1 - act2 directly because token lengths might differ.
            # Step 1: Mean pooling over all dimensions except the last one
            # (supports [B,S,I] and other shapes where the last dim is neuron index).
            act1_tensor = act1[layer_key].float()
            act2_tensor = act2[layer_key].float()

            reduce_dims_1 = tuple(range(act1_tensor.ndim - 1))
            reduce_dims_2 = tuple(range(act2_tensor.ndim - 1))

            if not reduce_dims_1 or not reduce_dims_2:
                raise ValueError(
                    f"Activation tensor for {layer_key} must have at least "
                    f"2 dimensions"
                )

            act1_pooled = act1_tensor.mean(dim=reduce_dims_1)
            act2_pooled = act2_tensor.mean(dim=reduce_dims_2)

            # Step 2: Compute absolute difference on the pooled representations
            neuron_diff = torch.abs(act1_pooled - act2_pooled)

            if layer_key not in accumulated_diffs:
                accumulated_diffs[layer_key] = []

            accumulated_diffs[layer_key].append(
                neuron_diff.cpu()
            )  # Move to CPU to save VRAM

    # Aggregate accumulated diffs across all prompt pairs
    result = {}
    for layer_key, diffs_list in accumulated_diffs.items():
        stacked = torch.stack(diffs_list)  # [num_pairs, intermediate_size]
        if aggregation == "mean":
            result[layer_key] = stacked.mean(dim=0)
        elif aggregation == "max":
            result[layer_key] = stacked.max(dim=0).values

    return result


def _normalize(t: torch.Tensor) -> torch.Tensor:
    """
    Min-max normalization to [0, 1].

    If max == min (e.g., all values are identical), returns a tensor of zeros
    to prevent division by zero, representing no relative differences.

    Args:
        t: Input tensor.

    Returns:
        Normalized tensor in [0, 1].
    """
    t_min = t.min()
    t_max = t.max()
    if torch.isclose(t_max, t_min):
        return torch.zeros_like(t)
    return (t - t_min) / (t_max - t_min + 1e-8)


def compute_fairness_pruning_scores(
    model: Any,
    bias_scores: Dict[str, torch.Tensor],
    bias_weight: float = 0.8,
) -> Dict[int, torch.Tensor]:
    """
    Combine BiasScore with ImportanceScore to produce FairnessPruningScores.

    For each layer, computes:
        FairnessPruningScore_i = bias_weight * BiasScore_norm_i
                               + (1 - bias_weight) * (1 - ImportanceScore_norm_i)

    A high FairnessPruningScore means the neuron is a strong pruning candidate:
    high bias sensitivity AND/OR low structural importance.

    Args:
        model: A Hugging Face CausalLM model with GLU MLP layers.
        bias_scores: Dictionary mapping layer keys (e.g. "gate_proj_layer_0")
            to per-neuron BiasScore tensors, as returned by analyze_neuron_bias().
        bias_weight: Weight for the bias component in [0.0, 1.0].
            1.0 = pure bias, 0.0 = pure importance.

    Returns:
        Dictionary mapping layer indices (int) to FairnessPruningScore tensors
        of shape [intermediate_size] on CPU.

    Raises:
        ValueError: If bias_weight is out of range or bias_scores is empty.
    """
    # Validate parameters
    if not 0.0 <= bias_weight <= 1.0:
        raise ValueError(
            f"bias_weight must be in [0.0, 1.0], got {bias_weight}"
        )

    if not bias_scores:
        raise ValueError("bias_scores dictionary is empty")

    result = {}

    # Multi-architecture support
    layers = get_model_layers(model)

    for i, layer in enumerate(layers):
        gate_bias_key = f"gate_proj_layer_{i}"
        up_bias_key = f"up_proj_layer_{i}"

        available = [
            bias_scores[k]
            for k in (gate_bias_key, up_bias_key)
            if k in bias_scores
        ]

        if not available:
            continue

        bias = torch.stack(available).mean(dim=0)

        # Static ImportanceScore from weights (Ensure layer is GLU)
        if (
            not hasattr(layer, "mlp")
            or not hasattr(layer.mlp, "gate_proj")
            or not hasattr(layer.mlp, "up_proj")
        ):
            continue

        gate_weight = layer.mlp.gate_proj.weight.data.float()
        up_weight = layer.mlp.up_proj.weight.data.float()
        importance = compute_neuron_pair_importance_maw(
            gate_weight, up_weight
        ).cpu()

        bias_norm = _normalize(bias)
        importance_norm = _normalize(importance)

        fairness_score = bias_weight * bias_norm + (1 - bias_weight) * (
            1 - importance_norm
        )

        result[i] = fairness_score

    return result