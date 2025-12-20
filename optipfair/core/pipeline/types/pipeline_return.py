from pydantic import BaseModel
from typing import Optional, List
from core.profiling.types.hardware import HardwareProfile
from core.profiling.types.llm import LLMInfo
from core.evaluation.types.inference_performance_info import InferencePerformanceInfo
from core.evaluation.model_performance.types.perplexity_test_result import (
    PerplexityTestResult,
)
from core.evaluation.types.compare_benchmark import CompareBenchmark


class LLMProfile(BaseModel):
    pre_compression: Optional[LLMInfo]
    post_compression: Optional[LLMInfo]


class InferencePerformance(BaseModel):
    pre_compression: Optional[InferencePerformanceInfo]
    post_compression: Optional[InferencePerformanceInfo]
    compared_results: Optional[CompareBenchmark]


class ModelPerformance(BaseModel):
    pre_compression: Optional[List[PerplexityTestResult]]
    post_compression: Optional[List[PerplexityTestResult]]


class Performance(BaseModel):
    inference_performance: Optional[InferencePerformance]
    model_performance: Optional[ModelPerformance]


class PipelineReturn(BaseModel):
    hardware_profile: Optional[HardwareProfile]
    llm_profile: Optional[LLMProfile]
    performance: Optional[Performance]
