import torch
from pydantic import BaseModel, ConfigDict
from core.compression.pruning.pruning_tools.neuron_importance.factory import (
    register_neuron_importance_function,
)


class ComputeNeuronPairImportanceMaxAbsWeightKwargs(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="ignore")

    gate_weight: torch.Tensor
    up_weight: torch.Tensor


@register_neuron_importance_function("max_abs_weight")
def compute_neuron_pair_importance_max_abs_weight(*args, **kwargs) -> torch.Tensor:
    """
    compute neuron pair importance scores (Maximum Absolute Weight)

    Args:
    - gate_weight: Weight matrix from the gate_proj layer.
    - up_weight: Weight matrix from the up_weight layer.

    Returns:
    - importance_scores: Importance scores for each neuron pair.
    """
    parsed_kwargs = ComputeNeuronPairImportanceMaxAbsWeightKwargs.model_validate(kwargs)
    gate_max_abs = torch.max(parsed_kwargs.gate_weight, dim=1).values + torch.abs(
        torch.min(parsed_kwargs.gate_weight, dim=1).values
    )
    up_max_abs = torch.max(parsed_kwargs.up_weight, dim=1).values + torch.abs(
        torch.min(parsed_kwargs.up_weight, dim=1).values
    )
    importance_scores = gate_max_abs + up_max_abs
    return importance_scores
