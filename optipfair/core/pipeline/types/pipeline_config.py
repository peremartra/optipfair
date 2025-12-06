from pydantic import BaseModel
from typing import List
from core.pipeline.types.prune_config import PruneConfig
from core.profiling.types.llm import AnalyzeConnections, EstimateMemory
from core.evaluation.model_performance.model_performance import (
    PERPLEXITY_TESTS,
    ACCURACY_TESTS,
)
from typing import Literal, Set

class ProfileLLM(BaseModel):
    profile: bool = True
    analyze_connections: AnalyzeConnections = AnalyzeConnections(input_shape=(1,512))
    estimate_memory: EstimateMemory = EstimateMemory(batch_size=4, sequence_length=512)
    verbose: bool = True

class BenchmarkModelPerformance(BaseModel):
    benchmark: bool = True
    batch_size: int = 4
    tests: Set[Literal[ACCURACY_TESTS, PERPLEXITY_TESTS]] = {'lambada'}


class BenchmarkInferencePerformance(BaseModel):
    benchmark: bool = True
    prompt: str = "Explain the theory of relativity in simple terms"
    max_new_tokens: int = 100
    num_runs: int = 20
    warmup_runs: int = 2

class PipelineConfig(BaseModel):
    prune_configs: List[PruneConfig]
    benchmark_model_performance: BenchmarkModelPerformance
    benchmark_inference_performance: BenchmarkInferencePerformance
    profile_llm: ProfileLLM
    profile_hardware: bool = True
