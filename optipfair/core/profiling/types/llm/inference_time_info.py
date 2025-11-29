from pydantic import BaseModel


class InferenceTimeInfo(BaseModel):
    mean_time_ms: float
    std_time_ms: float
    min_time_ms: float
    max_time_ms: float
    median_time_ms: float
    runs: int
