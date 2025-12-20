import torch
from core.compression.pruning.base import BasePruner
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from torch import nn
from core.compression.pruning.types.mlp_w_aligment.kwargs import (
    MLPAlignmentPrunerKwargs,
)
from loguru import logger
from core.compression.pruning.pruning_tools.neuron_importance.factory import (
    factory as neuron_importance_fn_factory,
)
from core.compression.pruning.factory import register_pruner


@register_pruner("mlp_alignment")
class MLPAlignmentPruner(BasePruner):
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def prune_neuron_pairs(self, mlp, prune_percent: float, alignment: int = 128):
        """
        Reduces dimensions ensuring the resulting size is a multiple of 'alignment'.
        """
        gate_weight = mlp.gate_proj.weight.data.float()
        up_weight = mlp.up_proj.weight.data.float()

        # TODO: make the calculation of neuron importance function dynamic based on user choice
        neuron_importance_function = neuron_importance_fn_factory("max_abs_weight")
        importance_scores = neuron_importance_function(
            gate_weight=gate_weight, up_weight=up_weight
        )

        original_intermediate_size = gate_weight.size(0)

        # 1. Calculate raw target size
        raw_target_k = int(original_intermediate_size * (1.0 - prune_percent))

        # 2. Enforce Alignment (Round down to nearest multiple of alignment)
        # We use max(alignment, ...) to ensure we don't prune to 0
        k = max(alignment, (raw_target_k // alignment) * alignment)

        # Check if we accidentally kept everything (optional logic)
        if k > original_intermediate_size:
            k = (
                original_intermediate_size - alignment
            )  # Force at least one block reduction if desired

        # Get indices
        _, indices_to_keep = torch.topk(importance_scores, k, largest=True, sorted=True)
        indices_to_keep = indices_to_keep.sort().values

        # Create new layers
        new_gate_proj = nn.Linear(mlp.gate_proj.in_features, k, bias=False).to(
            self.device
        )
        new_up_proj = nn.Linear(mlp.up_proj.in_features, k, bias=False).to(self.device)
        new_down_proj = nn.Linear(k, mlp.down_proj.out_features, bias=False).to(
            self.device
        )

        # Assign weights
        new_gate_proj.weight.data = mlp.gate_proj.weight.data[indices_to_keep, :]
        new_up_proj.weight.data = mlp.up_proj.weight.data[indices_to_keep, :]
        new_down_proj.weight.data = mlp.down_proj.weight.data[:, indices_to_keep]

        return new_gate_proj, new_up_proj, new_down_proj, k

    def prune(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        *args,
        **kwargs,
    ) -> PreTrainedModel:
        parsed_kwargs = MLPAlignmentPrunerKwargs.model_validate(kwargs)
        new_intermediate_size = None

        logger.info(f"Pruning MLP with target alignment: {parsed_kwargs.alignment}")

        for layer in model.model.layers:
            mlp = layer.mlp

            # Pass the alignment parameter
            new_gate_proj, new_up_proj, new_down_proj, new_size = (
                self.prune_neuron_pairs(
                    mlp, parsed_kwargs.prune_percent, alignment=parsed_kwargs.alignment
                )
            )

            mlp.gate_proj = new_gate_proj
            mlp.up_proj = new_up_proj
            mlp.down_proj = new_down_proj

            if new_intermediate_size is None:
                new_intermediate_size = new_size

        # Update config
        model.config.intermediate_size = new_intermediate_size
        logger.info(
            f"New intermediate size: {new_intermediate_size} (Multiple of {parsed_kwargs.alignment}: {new_intermediate_size % parsed_kwargs.alignment == 0})"
        )

        return model
