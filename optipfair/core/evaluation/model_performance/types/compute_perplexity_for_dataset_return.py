from pydantic import BaseModel
from typing import List


class ComputePerplexityForDatasetReturn(BaseModel):
    all_perplexities: List[float]
    mean_perplexity: float