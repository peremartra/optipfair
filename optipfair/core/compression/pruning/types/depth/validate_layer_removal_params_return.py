from pydantic import BaseModel
from typing import Optional, List, Any


class ValidateLayerRemovalParamsReturn(BaseModel):
    total_layers: int
    num_layers_to_remove: Optional[int]
    layer_indices: Optional[List[str]]
    layer_selection_method: str
    layers: List[Any]