
import time

import torch
from transformers import AutoTokenizer, PreTrainedModel
from core.evaluation.types.inference_performance_info import InferencePerformanceInfo
from core.evaluation.types.compare_benchmark import CompareBenchmark
from typing import Optional


class InferencePerformanceBenchmarker:

    def __init__(self):
        self.original_model_bechmark: Optional[InferencePerformanceInfo] = None
        self.compressed_model_bechmark: Optional[InferencePerformanceInfo] = None

    def time_inference(
        self,
        model: PreTrainedModel,
        tokenizer: AutoTokenizer,
        prompt: str,
        max_new_tokens: int = 100,
        num_runs: int = 5,
        warmup_runs: int = 2,
        is_uncompressed_model: bool = False
    ) -> InferencePerformanceInfo:
        """
        Measure inference time for a model.

        Args:
            model: Model to evaluate
            tokenizer: Tokenizer to use
            prompt: Input prompt for generation
            max_new_tokens: Maximum number of tokens to generate
            num_runs: Number of inference runs to average over
            warmup_runs: Number of initial runs to discard (for warm-up)

        Returns:
            Dictionary containing timing results
        """
        # Prepare input
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # Warmup runs
        for _ in range(warmup_runs):
            with torch.no_grad():
                _ = model.generate(
                    inputs.input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )

        # Timed runs
        times = []
        for _ in range(num_runs):
            start_time = time.time()
            with torch.no_grad():
                output = model.generate(
                    inputs.input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )
            end_time = time.time()
            times.append(end_time - start_time)

        generated_tokens = output.size(1) - inputs.input_ids.size(1)

        inference_performance_info = InferencePerformanceInfo(
            avg_time=sum(times) / len(times),
            min_time=min(times),
            max_time=max(times),
            tokens_per_second=generated_tokens / (sum(times) / len(times)),
            num_runs=num_runs,
            generated_tokens=generated_tokens,
        )

        if is_uncompressed_model:
            self.original_model_bechmark = inference_performance_info
        if not is_uncompressed_model:
            self.compressed_model_bechmark = inference_performance_info

        return inference_performance_info
 

    def compare_models_inference(self) -> Optional[CompareBenchmark]:

        if not self.original_model_bechmark or not self.compressed_model_bechmark:
            raise ValueError("You need to run the benchmark for the compressed and the uncompressed model before comparing them.")

        if self.original_model_bechmark and self.compressed_model_bechmark:        

            speedup = (
                self.original_model_bechmark.avg_time / self.compressed_model_bechmark.avg_time if self.compressed_model_bechmark.avg_time > 0 else float("inf")
            )
            tps_improvement = (
                (self.compressed_model_bechmark.tokens_per_second / self.original_model_bechmark.tokens_per_second - 1) * 100
                if self.original_model_bechmark.tokens_per_second > 0
                else float("inf")
            )

            return CompareBenchmark(
                speedup=speedup,
                tps_improvement_percent=tps_improvement
            )