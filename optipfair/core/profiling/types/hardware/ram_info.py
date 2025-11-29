from pydantic import BaseModel
from typing import Optional


class RAMInfo(BaseModel):
    total_memory_gb: float
    available_memory_gb: float
    free_memory_gb: float
    swap_total_gb: Optional[float] = None
    swap_free_gb: Optional[float] = None
