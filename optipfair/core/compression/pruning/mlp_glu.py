import gc
from typing import Dict, List, Optional, Tuple, Literal
from loguru import logger
import torch
from torch import Tensor, nn
from tqdm import tqdm
from transformers import PreTrainedModel
from core.compression.pruning.pruning_tools import (
    get_model_layers,
)
from core.compression.pruning.pruning_tools.round_to_divisor import round_to_divisor
from core.compression.pruning.pruning_tools.neuron_importance.factory import (
    factory as neuron_importance_fn_factory,
)
from core.compression.pruning.base import BasePruner
from core.compression.pruning.factory import register_pruner
from torch.utils.data import DataLoader
from core.compression.pruning.types.mlp_glu.kwargs import MlpGluPrunerKwargs
from core.compression.pruning.pruning_tools.calculate_pruning_percentage_from_expansion_rate import (
    calculate_pruning_percentage_from_expansion_rate,
)
from transformers import PreTrainedTokenizerBase


@register_pruner("mlp_glu")
class MlpGluPruner(BasePruner):
    """
    MlpGluPruner - class for pruning MLP layers with GLU architecture in transformer models.

    This class provides functionality to prune neurons in MLP layers that follow the
    Gated Linear Unit (GLU) architecture, as used in models like LLaMA. The pruning
    is structured to maintain the paired nature of gate_proj and up_proj layers.
    """

    def __init__(self):
        self._accumulated_act_norms: Dict[int, Tensor] = dict()

    def _setup_mlp_hooks_for_importance(
        self, model: PreTrainedModel, device: torch.device
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

        Returns:
            handles: List of hook handles (must be removed after calibration)

        Example:
            >>> handles = setup_mlp_hooks_for_importance(model, device)
            >>> # ... run forward passes ...
            >>> for handle in handles:
            >>>     handle.remove()
        """
        self._accumulated_act_norms.clear()

        # Free memory before starting calibration
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        handles = []

        # Get model layers (supports LLaMA, Mistral, etc.)
        layers = get_model_layers(model)
        if not layers:
            raise ValueError("Could not find transformer layers in model")

        # Initialize storage on CPU to save VRAM
        for idx, layer in enumerate(layers):
            intermediate_size = layer.mlp.down_proj.in_features
            self._accumulated_act_norms[idx] = torch.zeros(
                intermediate_size, dtype=torch.float32, device="cpu"
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
                    dim=(0, 1),  # Sum over batch and sequence dimensions
                )  # Result: [intermediate_size]

                # Accumulate on CPU to save VRAM
                self._accumulated_act_norms[layer_idx] += act_norms_L2.cpu()

            return hook

        # Register hooks on all down_proj layers
        for idx, layer in enumerate(layers):
            handle = layer.mlp.down_proj.register_forward_hook(make_hook(idx))
            handles.append(handle)

        logger.info(
            f"Registered {len(handles)} hooks on down_proj layers for activation capture"
        )

        return handles

    def _run_calibration_forward_passes(
        self,
        model: PreTrainedModel,
        dataloader: DataLoader,
        device: torch.device,
        show_progress: bool = True,
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
                        "input_ids": batch["input_ids"].to(device),
                        "attention_mask": batch.get("attention_mask", None),
                    }
                    if inputs["attention_mask"] is not None:
                        inputs["attention_mask"] = inputs["attention_mask"].to(device)
                elif isinstance(batch, (list, tuple)):
                    # Assume (input_ids, attention_mask) format
                    inputs = {
                        "input_ids": batch[0].to(device),
                        "attention_mask": batch[1].to(device)
                        if len(batch) > 1
                        else None,
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

    def _get_activation_norms(self) -> Dict[int, torch.Tensor]:
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
            for layer_idx, norms in self._accumulated_act_norms.items()
        }

    def _prune_neuron_pairs(
        self,
        mlp: nn.Module,
        prune_percentage: float,
        importance_fn: Literal["maw", "maw_hybrid", "von", "pon"] = "maw",
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

        concrete_importance_fn = neuron_importance_fn_factory(importance_fn)

        importance_fn_kwargs = {
            "gate_weight": gate_weight,
            "up_weight": up_weight,
            "down_weight": down_weight,
            "X_d_norm": activation_norms,
        }

        importance_scores = concrete_importance_fn(**importance_fn_kwargs)

        # Determine the new intermediate size
        original_intermediate_size = gate_weight.size(0)
        num_neuron_pairs_to_prune = min(
            int(prune_percentage / 100 * original_intermediate_size),
            original_intermediate_size - 1,
        )
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
            raise ValueError(
                f"Invalid number of neuron pairs to keep: {k}. Reduce pruning percentage."
            )
        # Select the neurons to keep based on importance scores
        _, indices_to_keep = torch.topk(importance_scores, k, largest=True)
        indices_to_keep = indices_to_keep.sort().values

        # Create new layers with reduced dimensions
        device = next(mlp.parameters()).device
        new_gate_proj = nn.Linear(
            mlp.gate_proj.in_features, k, bias=mlp.gate_proj.bias is not None
        ).to(device)
        new_up_proj = nn.Linear(
            mlp.up_proj.in_features, k, bias=mlp.up_proj.bias is not None
        ).to(device)
        new_down_proj = nn.Linear(
            k, mlp.down_proj.out_features, bias=mlp.down_proj.bias is not None
        ).to(device)

        # Copy selected weights to the new layers and convert back to original dtype
        new_gate_proj.weight.data = gate_weight[indices_to_keep, :].to(original_dtype)
        if mlp.gate_proj.bias is not None:
            new_gate_proj.bias.data = mlp.gate_proj.bias.data[indices_to_keep].to(
                original_dtype
            )

        new_up_proj.weight.data = up_weight[indices_to_keep, :].to(original_dtype)
        if mlp.up_proj.bias is not None:
            new_up_proj.bias.data = mlp.up_proj.bias.data[indices_to_keep].to(
                original_dtype
            )

        new_down_proj.weight.data = mlp.down_proj.weight.data[:, indices_to_keep].to(
            original_dtype
        )
        if mlp.down_proj.bias is not None:
            new_down_proj.bias.data = mlp.down_proj.bias.data.clone().to(original_dtype)

        return new_gate_proj, new_up_proj, new_down_proj, k

    def prune(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        *args,
        **kwargs,
    ) -> PreTrainedModel:
        """
        Prune the MLP layers in a model with GLU architecture.

        Args:
            model: Pre-trained model to prune
            neuron_selection_method: Method to use for calculating neuron importance ("MAW", "VOW", or "PON")
            pruning_percentage: Percentage of neurons to prune (0-100)
            expansion_rate: Target expansion rate in percentage (mutually exclusive with pruning_percentage)
            expansion_divisor: Optional divisor (32, 64, 128, 256, or None) to round intermediate layer size.
                When specified, the intermediate size will be rounded to the nearest multiple after applying
                pruning. Cannot be used alone - requires either pruning_percentage or expansion_rate.
            dataloader: Optional DataLoader for data-driven pruning. When provided with
                neuron_selection_method='MAW', enables hybrid importance calculation using
                both weight magnitudes and activation statistics. Only compatible with 'MAW'.
            show_progress: Whether to show progress during pruning

        Returns:
            model: Pruned model
        """
        parsed_kwargs = MlpGluPrunerKwargs.model_validate(kwargs)

        if parsed_kwargs.expansion_rate is not None:
            parsed_kwargs.pruning_percentage = (
                calculate_pruning_percentage_from_expansion_rate(
                    parsed_kwargs.expansion_rate, model
                )
            )
            parsed_kwargs.expansion_rate = None
        # =============================================================================
        # DATA-DRIVEN CALIBRATION (if dataloader provided)
        # ==============================================================================
        activation_norms = None

        if parsed_kwargs.dataloader is not None:
            logger.info("Starting data-driven calibration with provided dataloader")

            device = next(model.parameters()).device

            # Step 1: Register hooks to capture activations
            handles = self._setup_mlp_hooks_for_importance(model, device)

            try:
                # Step 2: Run forward passes to collect statistics
                self._run_calibration_forward_passes(
                    model,
                    parsed_kwargs.dataloader,
                    device,
                    parsed_kwargs.show_progress,
                )

                # Step 3: Extract accumulated norms
                activation_norms = self._get_activation_norms()

                # Verify we collected norms for all layers
                num_layers = len(get_model_layers(model))
                if len(activation_norms) != num_layers:
                    raise RuntimeError(
                        f"Calibration failed: expected norms for {num_layers} layers, "
                        f"got {len(activation_norms)}"
                    )

                logger.info(
                    f"Calibration complete: collected activation norms for {num_layers} layers"
                )

            finally:
                # Step 4: Always clean up hooks (even if error occurs)
                for handle in handles:
                    handle.remove()
                logger.info("Removed activation capture hooks")

        # ==============================================================================
        # PRUNING
        # ==============================================================================

        # Get all layers to prune
        layers = get_model_layers(model)
        if not layers:
            raise ValueError("Could not find MLP layers in the model.")

        new_intermediate_size = None

        # Prune each layer
        iterator = (
            tqdm(layers, desc="Pruning layers")
            if parsed_kwargs.show_progress
            else layers
        )

        for idx, layer in enumerate(iterator):
            mlp = layer.mlp

            # Store original size

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
            new_gate_proj, new_up_proj, new_down_proj, new_intermediate_size = (
                self._prune_neuron_pairs(
                    mlp=mlp,
                    prune_percentage=parsed_kwargs.pruning_percentage,
                    importance_fn=parsed_kwargs.importance_fn,
                    activation_norms=layer_activation_norms,
                    layer_idx=idx,
                    expansion_divisor=parsed_kwargs.expansion_divisor,
                )
            )

            # Replace original layers with pruned layers
            mlp.gate_proj = new_gate_proj
            mlp.up_proj = new_up_proj
            mlp.down_proj = new_down_proj


        if hasattr(model, "config") and hasattr(
            model.config, "intermediate_size"
        ):
            model.config.intermediate_size = new_intermediate_size
            logger.info(
                f"Updated model config: intermediate_size = {new_intermediate_size}"
            )

        return model
