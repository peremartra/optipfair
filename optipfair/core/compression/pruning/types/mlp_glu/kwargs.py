from pydantic import BaseModel, ConfigDict, model_validator
from typing import Literal, Optional
from torch.utils.data import DataLoader
from core.compression.pruning.types.neuron_selection_method import neuron_importance_calculation_methods


class MlpGluPrunerKwargs(BaseModel):
    """
    Pydantic model for validating input arguments for the MLP GLU Pruner.

    This model consolidates all input validation logic, ensuring that
    parameters for GLU pruning are correctly specified and compatible
    before the pruning process begins. It uses Pydantic v2 validators
    for robust and declarative validation.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    neuron_selection_method: neuron_importance_calculation_methods = "MAW"
    pruning_percentage: Optional[float] = None
    expansion_rate: Optional[float] = None
    expansion_divisor: Optional[Literal[32, 64, 128, 256]] = None
    dataloader: Optional[DataLoader] = None
    show_progress: bool = True

    @model_validator(mode="after")
    def validate_pruning_params_and_dataloader_usage(self) -> "MlpGluPrunerKwargs":
        """
        Performs comprehensive validation for pruning parameters ('pruning_percentage',
        'expansion_rate') and 'dataloader' usage. This validator runs after all
        individual field validators.

        It ensures:
        1. 'pruning_percentage' and 'expansion_rate' are mutually exclusive.
        2. At least one of 'pruning_percentage' or 'expansion_rate' is provided.
        3. If 'expansion_rate' is provided, it's converted to 'pruning_percentage'
           using `calculate_pruning_percentage_from_expansion_rate`.
        4. The final 'pruning_percentage' (whether direct or calculated) is between 0 and 100.
        5. 'dataloader' is only used when 'neuron_selection_method' is 'MAW'.
        6. 'expansion_divisor' is only provided if 'pruning_percentage' (or 'expansion_rate')
           is also provided.
        """
        if self.pruning_percentage is not None and self.expansion_rate is not None:
            raise ValueError(
                "Cannot provide both 'pruning_percentage' and 'expansion_rate'. "
                "Please choose one."
            )

        if self.pruning_percentage is None and self.expansion_rate is None:
            raise ValueError(
                "Either 'pruning_percentage' or 'expansion_rate' must be provided."
            )

        if self.pruning_percentage is not None and not (
            0 <= self.pruning_percentage <= 100
        ):
            raise ValueError(
                f"pruning_percentage must be between 0 and 100, but got {self.pruning_percentage}."
            )

        if self.dataloader is not None and self.neuron_selection_method != "MAW":
            raise ValueError(
                "dataloader can only be used with 'MAW' neuron_selection_method."
            )

        if self.expansion_divisor is not None and self.pruning_percentage is None:
            raise ValueError(
                "expansion_divisor requires either 'pruning_percentage' or 'expansion_rate' to be provided."
            )

        return self
