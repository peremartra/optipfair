from pydantic import BaseModel
from typing import List


class ComputePerplexityForBatchReturn(BaseModel):
    perplexities: List[float]
    mean_perplexity: float