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
import gc

logger = logging.getLogger(__name__)

# ==============================================================================
# DATA-DRIVEN PRUNING: Activation Capture via Hooks
# ==============================================================================

# Global variable to accumulate activation norms during calibration
_accumulated_act_norms = {}


def setup_mlp_hooks_for_importance(
    model: PreTrainedModel,
    device: torch.device,
    layer_indices: Optional[List[int]] = None
) -> List:
    """
    Register forward hooks on down_proj layers to capture input activations (X_d).
    
    Implements the activation capture mechanism from CFSP paper (Equation 8).
    Computes L2 norm of each neuron's activations: ||X_d^i|| = sqrt(sum_{b,s} X_d[b,s,i]²)
    
    The hooks accumulate norms across multiple batches during calibration, storing
    results on CPU to minimize VRAM usage.
    
    Args:
        model: Pre-trained model with transformer layers
        device: Device where the model is located
        layer_indices: Optional list of layer indices to register hooks on.
            If None, registers hooks on all layers.
        
    Returns:
        handles: List of hook handles (must be removed after calibration)
        
    Example:
        >>> handles = setup_mlp_hooks_for_importance(model, device)
        >>> # ... run forward passes ...
        >>> for handle in handles:
        >>>     handle.remove()
    """
    global _accumulated_act_norms
    _accumulated_act_norms.clear()
    
    # Free memory before starting calibration
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    handles = []
    
    # Get model layers (supports LLaMA, Mistral, etc.)
    layers = get_model_layers(model)
    if not layers:
        raise ValueError("Could not find transformer layers in model")
    
    # Filter layers if layer_indices specified
    if layer_indices is not None:
        layers_to_hook = [(idx, layers[idx]) for idx in layer_indices]
    else:
        layers_to_hook = list(enumerate(layers))
    
    # Initialize storage on CPU to save VRAM (only for selected layers)
    for idx, layer in layers_to_hook:
        intermediate_size = layer.mlp.down_proj.in_features
        _accumulated_act_norms[idx] = torch.zeros(
            intermediate_size,
            dtype=torch.float32,
            device='cpu'
        )
    
    def make_hook(layer_idx: int):
        """Factory function to create hook with layer index in closure"""
        def hook(module, input, output):
            """
            Hook function to capture X_d (input to down_proj) and compute L2 norm.
            
            X_d shape: [batch_size, seq_len, intermediate_size]
            Output: [intermediate_size] with ||X_d^i|| for each neuron i
            """
            X_d = input[0].detach()  # [B, S, I]
            
            # Compute L2 norm (CFSP Equation 8):
            # torch.norm with p=2 and dim=(0,1) computes:
            # ||X_d^i|| = sqrt(sum_{b,s} X_d[b,s,i]²)
            act_norms_L2 = torch.norm(
                X_d.to(torch.float32),  # Ensure precision
                p=2,
                dim=(0, 1)  # Sum over batch and sequence dimensions
            )  # Result: [intermediate_size]
            
            # Accumulate on CPU to save VRAM
            _accumulated_act_norms[layer_idx] += act_norms_L2.cpu()
        
        return hook
    
    # Register hooks on selected down_proj layers
    for idx, layer in layers_to_hook:
        handle = layer.mlp.down_proj.register_forward_hook(make_hook(idx))
        handles.append(handle)
    
    if layer_indices is not None:
        logger.info(f"Registered {len(handles)} hooks on down_proj layers {layer_indices} for activation capture")
    else:
        logger.info(f"Registered {len(handles)} hooks on all down_proj layers for activation capture")
    
    return handles


def get_activation_norms() -> Dict[int, torch.Tensor]:
    """
    Retrieve accumulated L2 norms from calibration.
    
    Returns a dictionary mapping layer indices to their accumulated activation norms.
    The returned tensors are clones to prevent accidental modifications.
    
    Returns:
        Dict mapping layer_idx -> activation_norms tensor [intermediate_size]
        
    Example:
        >>> activation_norms = get_activation_norms()
        >>> print(activation_norms[0].shape)  # torch.Size([8192]) for standard LLaMA
    """
    return {
        layer_idx: norms.clone()  # Clone to prevent external modifications
        for layer_idx, norms in _accumulated_act_norms.items()
    }


def run_calibration_forward_passes(
    model: PreTrainedModel,
    dataloader: Any,
    device: torch.device,
    show_progress: bool = True
) -> None:
    """
    Run forward passes over dataloader to collect activation statistics.
    
    This function puts the model in eval mode and runs inference on the provided
    dataloader while hooks capture activations. Memory is periodically cleared
    to prevent OOM errors.
    
    Args:
        model: Model with registered hooks
        dataloader: DataLoader providing calibration data
        device: Device where model is located
        show_progress: Whether to show progress bar
        
    Note:
        Hooks must be registered before calling this function using
        setup_mlp_hooks_for_importance()
    """
    model.eval()
    
    iterator = tqdm(dataloader, desc="Calibration") if show_progress else dataloader
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(iterator):
            # Handle different dataloader formats
            if isinstance(batch, dict):
                inputs = {
                    'input_ids': batch['input_ids'].to(device),
                    'attention_mask': batch.get('attention_mask', None)
                }
                if inputs['attention_mask'] is not None:
                    inputs['attention_mask'] = inputs['attention_mask'].to(device)
            elif isinstance(batch, (list, tuple)):
                # Assume (input_ids, attention_mask) format
                inputs = {
                    'input_ids': batch[0].to(device),
                    'attention_mask': batch[1].to(device) if len(batch) > 1 else None
                }
            else:
                raise ValueError(
                    f"Unsupported batch format: {type(batch)}. "
                    f"Expected dict or tuple of tensors."
                )
            
            # Forward pass (hooks are triggered automatically)
            _ = model(**inputs)
            
            # Periodic memory cleanup to avoid OOM
            if (batch_idx + 1) % 10 == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    
    logger.info(f"Completed calibration over {len(dataloader)} batches")

# ==============================================================================
# CLASSIC STATIC PRUNING. Neuron Pair Importance Computation
# ==============================================================================

def compute_neuron_pair_importance_maw(gate_weight: torch.Tensor, up_weight: torch.Tensor) -> torch.Tensor:
    """
    Compute neuron pair importance scores using Peak-to-Peak Magnitude (PPM) method.
    
    This method calculates importance as the sum of the peak-to-peak magnitude
    (max + |min|) of weights for each neuron in both gate_proj and up_proj layers.
    
    Reference: Martra, P. (2025). Fragile Knowledge, Robust Instruction-Following: 
    The Width Pruning Dichotomy in Llama-3.2. ArXiv. https://arxiv.org/abs/2512.22671
    
    Note: For backward compatibility, this function is also accessible via the
    "MAW" parameter name in pruning functions.
    
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
    importance_scores = gate_norms + up_norms
    return importance_scores

def compute_neuron_pair_importance_l2(gate_weight: torch.Tensor, up_weight: torch.Tensor) -> torch.Tensor:
    """
    Compute neuron pair importance scores using L2 norm method.

    Args:
        gate_weight: Weight matrix from the gate_proj layer
        up_weight: Weight matrix from the up_proj layer
        
    Returns:
        importance_scores: Importance scores for each neuron pair
    """
    gate_norms = torch.norm(gate_weight, p=2, dim=1)
    up_norms = torch.norm(up_weight, p=2, dim=1)
    importance_scores = gate_norms + up_norms
    return importance_scores

def compute_neuron_pair_importance_maw_hybrid(
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    X_d_norm: torch.Tensor
) -> torch.Tensor:
    """
    Compute neuron pair importance using a hybrid PPM + activations method.

    This implementation combines:
    - Structural importance from weights (PPM-style range: max + |min|)
      for gate_proj, up_proj, and down_proj.
    - Dynamic importance from activation norms X_d_norm collected during calibration.

    Reference: Martra, P. (2025). Fragile Knowledge, Robust Instruction-Following:
    The Width Pruning Dichotomy in Llama-3.2. ArXiv. https://arxiv.org/abs/2512.22671

    Important: This function ALWAYS uses PPM internally for computing structural importance,
    regardless of which neuron_selection_method is used in the pruning call. The hybrid
    calculation is exclusively available for data-driven pruning and combines PPM-based
    weight analysis with activation statistics. Other methods (VOW, PON) are only used
    in static (weight-only) pruning and do NOT support data-driven mode.

    The final score for neuron i is:
        importance_i = (PPM_gate_i_norm + PPM_up_i_norm + PPM_down_i_norm) * X_d_norm_i

    Args:
        gate_weight: Weight matrix from gate_proj [intermediate_size, hidden_size]
        up_weight: Weight matrix from up_proj [intermediate_size, hidden_size]
        down_weight: Weight matrix from down_proj [hidden_size, intermediate_size]
        X_d_norm: Accumulated L2 norms from calibration [intermediate_size]

    Returns:
        importance_scores: Importance score per neuron pair [intermediate_size]
    """
    # Ensure all tensors are on the same device and in float32 for stability
    gate_weight = gate_weight.float()
    up_weight = up_weight.float()
    down_weight = down_weight.float()
    X_d_norm = X_d_norm.float().to(gate_weight.device)

    # -------------------------------------------------------------------------
    # STATIC COMPONENT: PPM-style range (max + |min|) for each neuron
    # -------------------------------------------------------------------------
    # gate_proj and up_proj have shape [intermediate_size, hidden_size]
    # We compute a single scalar score per neuron (row) in each matrix.

    gate_score = (
        torch.max(gate_weight, dim=1).values +
        torch.abs(torch.min(gate_weight, dim=1).values)
    )  # [intermediate_size]

    up_score = (
        torch.max(up_weight, dim=1).values +
        torch.abs(torch.min(up_weight, dim=1).values)
    )  # [intermediate_size]

    # down_proj has shape [hidden_size, intermediate_size]
    # Each column corresponds to one intermediate neuron, so we reduce over dim=0.
    down_score = (
        torch.max(down_weight, dim=0).values +
        torch.abs(torch.min(down_weight, dim=0).values)
    )  # [intermediate_size]

    # -------------------------------------------------------------------------
    # NORMALIZATION: scale each component to [0, 1] to make them comparable
    # -------------------------------------------------------------------------
    gate_norm = gate_score / (gate_score.max() + 1e-8)
    up_norm = up_score / (up_score.max() + 1e-8)
    down_norm = down_score / (down_score.max() + 1e-8)

    # -------------------------------------------------------------------------
    # COMBINATION: structural (weights) + dynamic (activations)
    # -------------------------------------------------------------------------
    # structural_score aggregates the normalized structural importance from
    # gate, up, and down projections for each neuron.
    structural_score = gate_norm + up_norm + down_norm  # [intermediate_size]

    # Finally, modulate structural importance by the activation norms X_d_norm.
    importance_scores = structural_score * X_d_norm      # [intermediate_size]

    return importance_scores


# Dictionary mapping method names to their respective functions
IMPORTANCE_FUNCTIONS = {
    "PPM": compute_neuron_pair_importance_maw,
    "MAW": compute_neuron_pair_importance_maw,
    "VOW": compute_neuron_pair_importance_vow,
    "PON": compute_neuron_pair_importance_pon,
    "L2": compute_neuron_pair_importance_l2,
}

def round_to_divisor(value: int, divisor: int) -> int:
    """
    Round value to the nearest multiple of divisor.
    
    Args:
        value: Value to round
        divisor: Divisor to round to
        
    Returns:
        Rounded value (nearest multiple of divisor)
        
    Example:
        >>> round_to_divisor(8100, 128)
        8064
        >>> round_to_divisor(8200, 128)
        8192
    """
    return round(value / divisor) * divisor

def prune_neuron_pairs(
    mlp: nn.Module,
    prune_percentage: float,
    importance_fn: Callable = compute_neuron_pair_importance_maw,
    activation_norms: Optional[torch.Tensor] = None,
    layer_idx: Optional[int] = None,
    expansion_divisor: Optional[int] = None,
) -> Tuple[nn.Linear, nn.Linear, nn.Linear, int]:
    """
    Prune a specific percentage of neurons from the MLP layers (GLU architecture).
    
    Supports both static (weight-only) and hybrid (weight + activation) pruning.
    
    Args:
        mlp: MLP module containing gate_proj, up_proj, and down_proj layers
        prune_percentage: Percentage of neurons to prune (0-100)
        importance_fn: Function to compute neuron pair importance (static methods)
        activation_norms: Optional activation norms from calibration [intermediate_size].
            When provided, uses hybrid importance calculation.
        layer_idx: Layer index (used for logging when activation_norms provided)
        expansion_divisor: Optional divisor to round the intermediate size to nearest multiple
        
    Returns:
        new_gate_proj: Pruned gate_proj layer
        new_up_proj: Pruned up_proj layer
        new_down_proj: Pruned down_proj layer
        k: New intermediate size after pruning
    """
    # Store original dtype for later use
    original_dtype = mlp.gate_proj.weight.dtype
    
    # Extract the weights from the MLP layers and convert to float for calculations
    gate_weight = mlp.gate_proj.weight.data.float()
    up_weight = mlp.up_proj.weight.data.float()
    down_weight = mlp.down_proj.weight.data.float()
    
    # Compute importance scores (HYBRID or STATIC)
    if activation_norms is not None:
        # DATA-DRIVEN: Use hybrid importance calculation
        importance_scores = compute_neuron_pair_importance_maw_hybrid(
            gate_weight=gate_weight,
            up_weight=up_weight,
            down_weight=down_weight,
            X_d_norm=activation_norms
        )
    else:
        # STATIC: Use traditional weight-based importance
        importance_scores = importance_fn(gate_weight, up_weight)
    
    # Determine the new intermediate size
    original_intermediate_size = gate_weight.size(0)
    num_neuron_pairs_to_prune = min(int(prune_percentage / 100 * original_intermediate_size), original_intermediate_size - 1)
    k = original_intermediate_size - num_neuron_pairs_to_prune
    
    # Apply expansion_divisor rounding if specified
    if expansion_divisor is not None:
        k_rounded = round_to_divisor(k, expansion_divisor)
        # Ensure we don't exceed original size or go below 1
        k_rounded = min(k_rounded, original_intermediate_size)
        k_rounded = max(k_rounded, 1)
        
        if k_rounded != k:
            logger.debug(
                f"Layer {layer_idx}: Adjusted intermediate size from {k} to {k_rounded} "
                f"(divisible by {expansion_divisor})"
            )
        k = k_rounded
    
    # Validate the new size
    if k <= 0:
        raise ValueError(f"Invalid number of neuron pairs to keep: {k}. Reduce pruning percentage.")
    # Select the neurons to keep based on importance scores
    _, indices_to_keep = torch.topk(importance_scores, k, largest=True)
    indices_to_keep = indices_to_keep.sort().values
    
    # Create new layers with reduced dimensions
    device = next(mlp.parameters()).device
    new_gate_proj = nn.Linear(mlp.gate_proj.in_features, k, bias=mlp.gate_proj.bias is not None).to(device)
    new_up_proj = nn.Linear(mlp.up_proj.in_features, k, bias=mlp.up_proj.bias is not None).to(device)
    new_down_proj = nn.Linear(k, mlp.down_proj.out_features, bias=mlp.down_proj.bias is not None).to(device)
    
    # Copy selected weights to the new layers and convert back to original dtype
    new_gate_proj.weight.data = gate_weight[indices_to_keep, :].to(original_dtype)
    if mlp.gate_proj.bias is not None:
        new_gate_proj.bias.data = mlp.gate_proj.bias.data[indices_to_keep].to(original_dtype)
    
    new_up_proj.weight.data = up_weight[indices_to_keep, :].to(original_dtype)
    if mlp.up_proj.bias is not None:
        new_up_proj.bias.data = mlp.up_proj.bias.data[indices_to_keep].to(original_dtype)
    
    new_down_proj.weight.data = mlp.down_proj.weight.data[:, indices_to_keep].to(original_dtype)
    if mlp.down_proj.bias is not None:
        new_down_proj.bias.data = mlp.down_proj.bias.data.clone().to(original_dtype)
    
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
    neuron_selection_method: str = "PPM",
    pruning_percentage: Optional[float] = 10,
    expansion_rate: Optional[float] = None,
    expansion_divisor: Optional[int] = None,
    dataloader: Optional[Any] = None,
    layer_indices: Optional[List[int]] = None,
    show_progress: bool = True,
) -> PreTrainedModel:
    """
    Prune the MLP layers in a model with GLU architecture using PPM (Peak-to-Peak Magnitude) method.
    
    The default neuron selection method PPM calculates importance based on the full dynamic
    range of weights (max + |min|). For backward compatibility, the parameter value "MAW"
    is accepted and maps to PPM.
    
    Reference: Martra, P. (2025). Fragile Knowledge, Robust Instruction-Following:
    The Width Pruning Dichotomy in Llama-3.2. ArXiv. https://arxiv.org/abs/2512.22671
    
    Args:
        model: Pre-trained model to prune
        neuron_selection_method: Method to use for calculating neuron importance ("MAW"/PPM, "VOW", "PON", or "L2")
        pruning_percentage: Percentage of neurons to prune (0-100)
        expansion_rate: Target expansion rate in percentage (mutually exclusive with pruning_percentage)
        expansion_divisor: Optional divisor (32, 64, 128, 256, or None) to round intermediate layer size.
            When specified, the intermediate size will be rounded to the nearest multiple after applying
            pruning. Cannot be used alone - requires either pruning_percentage or expansion_rate.
        dataloader: Optional DataLoader for data-driven pruning. When provided with
            neuron_selection_method='MAW' (PPM method), enables hybrid importance calculation using
            both weight magnitudes and activation statistics. Only compatible with PPM/'MAW'.
        layer_indices: Optional list of layer indices to prune. If None, all layers are pruned.
            When specified, only the listed layers will have their neurons pruned; other layers remain unchanged.
        show_progress: Whether to show progress during pruning
        
    Returns:
        model: Pruned model
    """
    # Validate the model for GLU pruning
    if not validate_model_for_glu_pruning(model):
        raise ValueError("Model is not compatible with GLU pruning. It must have gate_proj, up_proj, and down_proj layers.")
    
    # Validate layer_indices if provided
    layers = get_model_layers(model)
    if not layers:
        raise ValueError("Could not find MLP layers in the model.")
    
    if layer_indices is not None:
        if not isinstance(layer_indices, list):
            raise TypeError(f"layer_indices must be a list, got {type(layer_indices)}")
        
        if not all(isinstance(idx, int) for idx in layer_indices):
            raise TypeError("All elements in layer_indices must be integers")
        
        if not layer_indices:
            raise ValueError("layer_indices cannot be an empty list")
        
        invalid_indices = [idx for idx in layer_indices if idx < 0 or idx >= len(layers)]
        if invalid_indices:
            raise ValueError(
                f"Invalid layer indices: {invalid_indices}. "
                f"Model has {len(layers)} layers (valid indices: 0-{len(layers)-1})"
            )
        
        if len(layer_indices) != len(set(layer_indices)):
            raise ValueError("layer_indices contains duplicate values")
        
        logger.info(f"Selective pruning: will prune {len(layer_indices)} of {len(layers)} layers: {sorted(layer_indices)}")
    
    # Validate expansion_divisor
    if expansion_divisor is not None:
        valid_divisors = [32, 64, 128, 256]
        if expansion_divisor not in valid_divisors:
            raise ValueError(
                f"expansion_divisor must be one of {valid_divisors} or None. "
                f"Got: {expansion_divisor}"
            )
        
        # expansion_divisor requires pruning_percentage or expansion_rate
        if pruning_percentage is None and expansion_rate is None:
            raise ValueError(
                "expansion_divisor cannot be used alone. "
                "Please provide either pruning_percentage or expansion_rate."
            )
    
    # Validate dataloader compatibility 
    if dataloader is not None and neuron_selection_method != "MAW":
        raise ValueError(
            f"Data-driven pruning with dataloader is only supported for PPM method (parameter 'MAW'). "
            f"Got neuron_selection_method='{neuron_selection_method}'. "
            f"Please use neuron_selection_method='MAW' or remove the dataloader parameter."
        )    

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
    
    # =============================================================================
    # DATA-DRIVEN CALIBRATION (if dataloader provided)
    # ==============================================================================
    activation_norms = None
    
    if dataloader is not None:
        logger.info("Starting data-driven calibration with provided dataloader")
        
        device = next(model.parameters()).device
        
        # Step 1: Register hooks to capture activations (only on selected layers)
        handles = setup_mlp_hooks_for_importance(model, device, layer_indices=layer_indices)
        
        try:
            # Step 2: Run forward passes to collect statistics
            run_calibration_forward_passes(model, dataloader, device, show_progress)
            
            # Step 3: Extract accumulated norms
            activation_norms = get_activation_norms()
            
            # Verify we collected norms for selected layers
            expected_layers = len(layer_indices) if layer_indices is not None else len(get_model_layers(model))
            if len(activation_norms) != expected_layers:
                raise RuntimeError(
                    f"Calibration failed: expected norms for {expected_layers} layers, "
                    f"got {len(activation_norms)}"
                )
            
            logger.info(f"Calibration complete: collected activation norms for {expected_layers} layers")
            
        finally:
            # Step 4: Always clean up hooks (even if error occurs)
            for handle in handles:
                handle.remove()
            logger.info("Removed activation capture hooks")
    
    # ==============================================================================
    # PRUNING
    # ==============================================================================

    # Filter layers to prune if layer_indices specified
    if layer_indices is not None:
        layers_to_prune = [(idx, layers[idx]) for idx in sorted(layer_indices)]
        if show_progress:
            iterator = tqdm(layers_to_prune, desc=f"Pruning {len(layers_to_prune)} selected layers")
        else:
            iterator = layers_to_prune
    else:
        layers_to_prune = list(enumerate(layers))
        if show_progress:
            iterator = tqdm(layers_to_prune, desc="Pruning all layers")
        else:
            iterator = layers_to_prune
    
    new_intermediate_size = None
    
    # Prune each selected layer
    for idx, layer in iterator:
        mlp = layer.mlp
        
        # Store original size
        original_intermediate_size = mlp.gate_proj.out_features
        
        # Get activation norms for this layer (if available)
        layer_activation_norms = None
        if activation_norms is not None:
            if idx not in activation_norms:
                raise KeyError(
                    f"No activation norms found for layer {idx}. "
                    f"Available layers: {list(activation_norms.keys())}"
                )
            layer_activation_norms = activation_norms[idx]
        
        # Prune the neuron pairs (HYBRID if activation_norms provided, STATIC otherwise)
        new_gate_proj, new_up_proj, new_down_proj, new_intermediate_size = prune_neuron_pairs(
            mlp=mlp,
            prune_percentage=pruning_percentage,
            importance_fn=importance_fn,
            activation_norms=layer_activation_norms,
            layer_idx=idx,
            expansion_divisor=expansion_divisor,
        )
        
        # Replace original layers with pruned layers
        mlp.gate_proj = new_gate_proj
        mlp.up_proj = new_up_proj
        mlp.down_proj = new_down_proj
    
    # Update model configuration
    if hasattr(model, "config") and hasattr(model.config, "intermediate_size"):
        model.config.intermediate_size = new_intermediate_size
        logger.info(f"Updated model config: intermediate_size = {new_intermediate_size}")
    
    return model