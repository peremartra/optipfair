import torch
from pydantic import BaseModel, ConfigDict, model_validator
from core.compression.pruning.pruning_tools.neuron_importance.factory import (
    register_neuron_importance_function,
)


class ComputeNeuronPairImportanceVowKwargs(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="ignore")

    gate_weight: torch.Tensor
    up_weight: torch.Tensor

    @model_validator(mode="after")
    def validate_compute_neuron_pair_importance_vow_kwargs(
        self,
    ) -> "ComputeNeuronPairImportanceVowKwargs":
        if not isinstance(self.gate_weight, torch.Tensor):
            raise ValueError("gate_weight is not instance of torch.Tensor")
        if not isinstance(self.up_weight, torch.Tensor):
            raise ValueError("up_weight is not instance of torch.Tensor")
        return self


@register_neuron_importance_function("vow")
def compute_neuron_pair_importance_vow(*args, **kwargs) -> torch.Tensor:
    """
    Compute neuron pair importance scores using Variance of Weights method.

    Args:
        gate_weight: Weight matrix from the gate_proj layer
        up_weight: Weight matrix from the up_proj layer

    Returns:
        importance_scores: Importance scores for each neuron pair
    """
    parsed_kwargs = ComputeNeuronPairImportanceVowKwargs.model_validate(kwargs)
    gate_variance = torch.var(parsed_kwargs.gate_weight, dim=1)
    up_variance = torch.var(parsed_kwargs.up_weight, dim=1)
    importance_scores = gate_variance + up_variance
    return importance_scores
