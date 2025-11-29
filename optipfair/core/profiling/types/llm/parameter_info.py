from pydantic import BaseModel
from typing import Dict, Any


class ParameterInfo(BaseModel):
    total: int
    trainable: int
    non_trainable: int
    total_millions: float
    total_billions: float
    by_dtype_counts: Dict[str, Any]
    by_dtype_bytes: Dict[str, Any]
