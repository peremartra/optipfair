from pydantic import BaseModel
from typing import Literal
from core.evaluation.model_performance.types.compute_perplexity_for_dataset_return import ComputePerplexityForDatasetReturn


class PerplexityTestResult(BaseModel):
    test_name: Literal['lambada']
    result: ComputePerplexityForDatasetReturn