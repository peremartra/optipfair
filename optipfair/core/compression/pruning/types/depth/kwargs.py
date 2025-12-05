from pydantic import BaseModel, field_validator, ConfigDict
from typing import List, Optional, Literal
from transformers import PreTrainedModel


class DepthPrunerKwargs(BaseModel):
    """
    Pydantic model for validating input arguments for the Depth Pruner.

    This model consolidates all input validation logic, ensuring that
    parameters for depth pruning are correctly specified and compatible
    before the pruning process begins. It uses Pydantic v2 validators
    for robust and declarative validation.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: PreTrainedModel
    num_layers_to_remove: Optional[int] = None
    layer_indices: Optional[List[int]] = None
    depth_pruning_percentage: Optional[float] = None
    layer_selection_method: Literal["last", "custom"] = "last"
    show_progress: bool = True

    @field_validator("model")
    @classmethod
    def validate_model(cls, v):
        if not isinstance(v, PreTrainedModel):
            raise ValueError(
                f"model must be an instance of PreTrainedModel, got {type(v).__name__}"
            )
        return v
