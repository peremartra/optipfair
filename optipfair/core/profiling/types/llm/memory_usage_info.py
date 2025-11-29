from pydantic import BaseModel


class MemoryUsageInfo(BaseModel):
    allocated_memory_mb: float
    cached_memory_mb: float
    max_memory_mb: float
