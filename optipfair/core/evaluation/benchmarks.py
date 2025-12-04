"""
Benchmarking utilities for evaluating pruned models.

This module provides tools for evaluating the performance of pruned models
on standard benchmarks like LAMBADA, BoolQ, etc.
"""

from loguru import logger
import time
from typing import List, Optional

import torch
from transformers import AutoTokenizer, PreTrainedModel
from core.evaluation.inference_performance.inference_performance import InferencePerformanceBenchmarker
from core.evaluation.model_performance.model_performance import ModelPerformanceBenchmarker


class ModelBenchmarker:

    def __init__(self, inference_performance_benchmarker: InferencePerformanceBenchmarker, model_performance_benchmarker: ModelPerformanceBenchmarker):
        self.inference_performance_benchmarker = inference_performance_benchmarker
        self.model_performance_benchmarker = model_performance_benchmarker


    def benchmark(self, model: PreTrainedModel, tokenizer: AutoTokenizer):
        pass


