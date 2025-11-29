from pydantic import BaseModel
from typing import List, Optional


class LayerConnectionInfo(BaseModel):
    """Information about layer connections in the model."""

    name: str
    type: str
    parameters: int
    depth: int
    input_layers: List[str]
    output_layers: List[str]
    input_shape: Optional[List[int]] = None
    output_shape: Optional[List[int]] = None
    input_size: Optional[int] = None
    output_size: Optional[int] = None
    vocab_size: Optional[int] = None
    embedding_dim: Optional[int] = None
