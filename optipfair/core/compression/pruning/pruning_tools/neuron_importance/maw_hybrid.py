import torch
from core.compression.pruning.pruning_tools.neuron_importance.factory import (
    register_neuron_importance_function,
)
from pydantic import BaseModel, ConfigDict, model_validator


class ComputeNeuronPairImportanceMawHybridKwargs(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="ignore")

    gate_weight: torch.Tensor
    up_weight: torch.Tensor
    down_weight: torch.Tensor
    X_d_norm: torch.Tensor

    @model_validator(mode="after")
    def validate_compute_neuron_pair_importance_maw_hybrid_kwargs(
        self,
    ) -> "ComputeNeuronPairImportanceMawHybridKwargs":
        if not isinstance(self.gate_weight, torch.Tensor):
            raise ValueError("gate_weight is not instance of torch.Tensor")
        if not isinstance(self.up_weight, torch.Tensor):
            raise ValueError("up_weight is not instance of torch.Tensor")
        if not isinstance(self.down_weight, torch.Tensor):
            raise ValueError("down_weight is not instance of torch.Tensor")
        if not isinstance(self.X_d_norm, torch.Tensor):
            raise ValueError("X_d_norm is not instance of torch.Tensor")

        return self


@register_neuron_importance_function("maw_hybrid")
def compute_neuron_pair_importance_maw_hybrid(*args, **kwargs) -> torch.Tensor:
    """
    Compute neuron pair importance using hybrid data-driven method (MAW + Activations).

    Implements CFSP methodology (arXiv:2409.13199v2, Equation 8):

    F_i^l = Σ_j [ |W_d^ij · ||X_d^i|| / (||W_d^*j|| · ||X_d^*||) ] +
            Σ_j [ |W_u^ij| / ||W_u^i*|| ] +
            Σ_j [ |W_g^ij| / ||W_g^i*|| ]

    Where:
    - Component 1 (down_proj): Weights weighted by activations (DATA-DRIVEN)
    - Component 2 (up_proj): Static weight-based importance
    - Component 3 (gate_proj): Static weight-based importance

    Args:
        gate_weight: Weight matrix from gate_proj [intermediate_size, hidden_size]
        up_weight: Weight matrix from up_proj [intermediate_size, hidden_size]
        down_weight: Weight matrix from down_proj [hidden_size, intermediate_size]
        X_d_norm: Accumulated L2 norms from calibration [intermediate_size]

    Returns:
        importance_scores: Importance score per neuron pair [intermediate_size]
    """

    parsed_kwargs = ComputeNeuronPairImportanceMawHybridKwargs.model_validate(**kwargs)

    device = parsed_kwargs.gate_weight.device

    # Move X_d_norm to device and ensure float32 for numerical stability
    X_d_norm = parsed_kwargs.X_d_norm.to(device).to(torch.float32)

    # Convert all weights to float32
    gate_weight = parsed_kwargs.gate_weight.float()
    up_weight = parsed_kwargs.up_weight.float()
    down_weight = parsed_kwargs.down_weight.float()

    # ==========================================================================
    # COMPONENT 1: down_proj with activations (DATA-DRIVEN)
    # Term: |W_d^ij · ||X_d^i|| / (||W_d^*j|| · ||X_d^*||)
    # ==========================================================================

    # Transpose down_weight: [hidden_size, intermediate_size] -> [intermediate_size, hidden_size]
    W_d_t = down_weight.t()  # [intermediate_size, hidden_size]
    W_d_abs = torch.abs(W_d_t)

    # NUMERATOR: |W_d^ij| * ||X_d^i||
    numerator = W_d_abs * X_d_norm.unsqueeze(1)  # [intermediate_size, hidden_size]

    # DENOMINATOR: (Σ_i |W_d^ij|) * (Σ_i ||X_d^i||)
    W_d_column_sums = W_d_abs.sum(dim=0, keepdim=True)  # [1, hidden_size]
    X_d_total_norm = X_d_norm.sum()  # Scalar
    denominator = W_d_column_sums * X_d_total_norm  # [1, hidden_size]

    # Normalized term: sum over output dimension j
    normalized_down = (numerator / (denominator + 1e-8)).sum(
        dim=1
    )  # [intermediate_size]

    # ==========================================================================
    # COMPONENT 2: up_proj weights only (STATIC)
    # Term: |W_u^ij| / ||W_u^i*||
    # ==========================================================================

    up_abs = torch.abs(up_weight)  # [intermediate_size, hidden_size]
    row_sums_up = up_abs.sum(dim=1, keepdim=True)  # [intermediate_size, 1]
    normalized_up = (up_abs / (row_sums_up + 1e-8)).sum(dim=1)  # [intermediate_size]

    # ==========================================================================
    # COMPONENT 3: gate_proj weights only (STATIC)
    # Term: |W_g^ij| / ||W_g^i*||
    # ==========================================================================

    gate_abs = torch.abs(gate_weight)  # [intermediate_size, hidden_size]
    row_sums_gate = gate_abs.sum(dim=1, keepdim=True)  # [intermediate_size, 1]
    normalized_gate = (gate_abs / (row_sums_gate + 1e-8)).sum(
        dim=1
    )  # [intermediate_size]

    # ==========================================================================
    # FINAL IMPORTANCE SCORE (Equation 8)
    # ==========================================================================

    importance_scores = normalized_down + normalized_up + normalized_gate

    return importance_scores
