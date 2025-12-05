from pydantic import BaseModel


class InferencePerformanceInfo(BaseModel):
    avg_time: float
    min_time: float
    max_time: float
    tokens_per_second: float
    num_runs: float
    generated_tokens: float
