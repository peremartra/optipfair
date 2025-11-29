from pydantic import BaseModel
from typing import List, Any


class AttentionLayerAnalysisInfo(BaseModel):
    num_attention_layers: int
    attention_layers: List[Any]
