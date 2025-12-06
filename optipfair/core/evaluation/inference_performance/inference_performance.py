import time
import torch
from transformers import AutoTokenizer, PreTrainedModel
from core.evaluation.types.inference_performance_info import InferencePerformanceInfo
from core.evaluation.types.compare_benchmark import CompareBenchmark


class InferencePerformanceBenchmarker:
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

    def time_inference(
        self,
        model: PreTrainedModel,
        tokenizer: AutoTokenizer,
        prompt: str,
        max_new_tokens: int = 100,
        num_runs: int = 5,
        warmup_runs: int = 2,
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

        return inference_performance_info
