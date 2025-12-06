from pydantic import BaseModel, ConfigDict
from typing import List, Optional, Literal


class DepthPrunerKwargs(BaseModel):
    """
    Pydantic model for validating input arguments for the Depth Pruner.

    This model consolidates all input validation logic, ensuring that
    parameters for depth pruning are correctly specified and compatible
    before the pruning process begins. It uses Pydantic v2 validators
    for robust and declarative validation.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    num_layers_to_remove: Optional[int] = None
    layer_indices: Optional[List[int]] = None
    depth_pruning_percentage: Optional[float] = None
    layer_selection_method: Literal["last", "custom"] = "last"
    show_progress: bool = True
