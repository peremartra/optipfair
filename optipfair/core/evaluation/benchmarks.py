"""
Benchmarking utilities for evaluating pruned models.

This module provides tools for evaluating the performance of pruned models
on standard benchmarks like LAMBADA, BoolQ, etc.
"""

from transformers import AutoTokenizer, PreTrainedModel
from core.evaluation.inference_performance.inference_performance import (
    InferencePerformanceBenchmarker,
)
from core.evaluation.model_performance.model_performance import (
    ModelPerformanceBenchmarker,
)
from core.evaluation.types.compare_benchmark import CompareBenchmark
from core.evaluation.types.inference_performance_info import InferencePerformanceInfo


class ModelBenchmarker:
    def __init__(self):
        self.inference_performance_benchmarker = InferencePerformanceBenchmarker()
        self.model_performance_benchmarker = ModelPerformanceBenchmarker()

    def compare_models_inference(
        self,
        original_benchmark: InferencePerformanceInfo,
        compressed_benchmark: InferencePerformanceInfo,
    ) -> CompareBenchmark:
        speedup = (
            original_benchmark / compressed_benchmark.avg_time
            if compressed_benchmark.avg_time > 0
            else float("inf")
        )
        tps_improvement = (
            (
                compressed_benchmark.tokens_per_second
                / original_benchmark.tokens_per_second
                - 1
            )
            * 100
            if original_benchmark.tokens_per_second > 0
            else float("inf")
        )

        return CompareBenchmark(
            speedup=speedup, tps_improvement_percent=tps_improvement
        )

    def benchmark(self, model: PreTrainedModel, tokenizer: AutoTokenizer):
        pass
