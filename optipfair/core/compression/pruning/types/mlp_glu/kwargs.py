from pydantic import BaseModel, ConfigDict, field_validator, model_validator
from core.compression.pruning.pruning_tools.calculate_pruning_percentage_from_expansion_rate import (
    calculate_pruning_percentage_from_expansion_rate,
)
from transformers import PreTrainedModel
from typing import Literal, Optional
from core.compression.pruning.pruning_tools import (
    validate_model_for_glu_pruning,
)
from torch.utils.data import DataLoader

class MlpGluPrunerKwargs(BaseModel):
    """
    Pydantic model for validating input arguments for the MLP GLU Pruner.

    This model consolidates all input validation logic, ensuring that
    parameters for GLU pruning are correctly specified and compatible
    before the pruning process begins. It uses Pydantic v2 validators
    for robust and declarative validation.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: PreTrainedModel
    neuron_selection_method: Literal["MAW", "VOW", "PON"] = "MAW"
    pruning_percentage: Optional[float] = None
    expansion_rate: Optional[float] = None
    expansion_divisor: Optional[Literal[32, 64, 128, 256]] = None
    dataloader: Optional[DataLoader] = None
    show_progress: bool = True

    @field_validator("model", mode="after")
    @classmethod
    def validate_model_is_pretrained_and_glu_compatible(
        cls, v: PreTrainedModel
    ) -> PreTrainedModel:
        """
        Validates that the provided 'model' is an instance of PreTrainedModel
        and is compatible with GLU pruning. This check delegates to an external
        helper function `validate_model_for_glu_pruning`.
        """
        validate_model_for_glu_pruning(v)
        return v

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

        if self.expansion_rate is not None:
            self.pruning_percentage = calculate_pruning_percentage_from_expansion_rate(
                self.expansion_rate, self.model
            )
            self.expansion_rate = None

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