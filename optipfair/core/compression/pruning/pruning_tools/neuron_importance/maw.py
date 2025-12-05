import torch
from core.compression.pruning.pruning_tools.neuron_importance.factory import (
    register_neuron_importance_function,
)
from pydantic import BaseModel, ConfigDict, model_validator


class ComputeNeuronPairImportanceMawKwargs(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="ignore")

    gate_weight: torch.Tensor
    up_weight: torch.Tensor

    @model_validator(mode="after")
    def validate_compute_neuron_pair_importance_maw__kwargs(
        self,
    ) -> "ComputeNeuronPairImportanceMawKwargs":
        if not isinstance(self.gate_weight, torch.Tensor):
            raise ValueError("gate_weight is not instance of torch.Tensor")
        if not isinstance(self.up_weight, torch.Tensor):
            raise ValueError("up_weight is not instance of torch.Tensor")
        return self


@register_neuron_importance_function("maw")
def compute_neuron_pair_importance_maw(*args, **kwargs) -> torch.Tensor:
    """
    Compute neuron pair importance scores using Maximum Absolute Weight method.

    Args:
        gate_weight: Weight matrix from the gate_proj layer
        up_weight: Weight matrix from the up_proj layer

    Returns:
        importance_scores: Importance scores for each neuron pair
    """
    parsed_kwargs = ComputeNeuronPairImportanceMawKwargs.model_validate(**kwargs)

    gate_max_abs = torch.max(parsed_kwargs.gate_weight, dim=1).values + torch.abs(
        torch.min(parsed_kwargs.gate_weight, dim=1).values
    )
    up_max_abs = torch.max(parsed_kwargs.up_weight, dim=1).values + torch.abs(
        torch.min(parsed_kwargs.up_weight, dim=1).values
    )
    importance_scores = gate_max_abs + up_max_abs
    return importance_scores
