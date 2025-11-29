from pydantic import BaseModel
from typing import Optional, Callable, Any


class MeasureInferenceTime(BaseModel):
    input_sample: Optional[Callable[[], Any]]
    num_runs: int
    warmup_runs: int
    tokenizer_max_length: int
