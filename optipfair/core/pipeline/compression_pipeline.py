from transformers import PreTrainedModel, PreTrainedTokenizerBase
from core.pipeline.types.prune_config import PruneConfig
from typing import List
from core.profiling.hardware_profiler import HardwareProfiler
from core.profiling.types.hardware import HardwareProfile
from core.profiling.llm_profiler import LLMProfiler
from core.profiling.types.llm import LLMInfo, AnalyzeConnections, EstimateMemory
from core.evaluation.model_performance.model_performance import (
    ModelPerformanceBenchmarker,
)
from core.evaluation.model_performance.types.perplexity_test_result import (
    PerplexityTestResult,
)
from core.evaluation.inference_performance.inference_performance import (
    InferencePerformanceBenchmarker,
)
from core.evaluation.types.inference_performance_info import InferencePerformanceInfo
from core.compression.pruning.factory import factory as prune_factory
from core.pipeline.types.pipeline_config import PipelineConfig
from core.pipeline.types.pipeline_return import (
    PipelineReturn,
    LLMProfile,
    Performance,
    InferencePerformance,
    ModelPerformance,
)
from core.evaluation.types.compare_benchmark import CompareBenchmark
from loguru import logger


class CompressionPipeline:
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.hardware_profiler = HardwareProfiler()
        self.model_performance_benchmarker = ModelPerformanceBenchmarker()
        self.inference_performance_benchmarker = InferencePerformanceBenchmarker()

    def profile_model(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        analyze_connections: AnalyzeConnections,
        estimate_memory: EstimateMemory,
        verbose: bool = True,
    ) -> LLMInfo:
        profiler = LLMProfiler(model, tokenizer, verbose)
        return profiler.profile_complete(
            analyze_connections=analyze_connections, estimate_memory=estimate_memory
        )

    def prune_model(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        prune_config: PruneConfig,
    ) -> PreTrainedModel:
        prune_impl = prune_factory(prune_config.prune_technique)
        return prune_impl.prune(
            model=model, tokenizer=tokenizer, **prune_config.prune_technique_kwargs.model_dump(mode='python')
        )
    
    def get_model(self) -> PreTrainedModel:
        return self.model

    def run(self, pipeline_config: PipelineConfig) -> PipelineReturn:

        hardware_profile: HardwareProfile | None = None

        pre_compression_model_profile: LLMInfo | None = None
        pre_compression_inference_performance: InferencePerformanceInfo | None = None
        pre_compression_model_performance: List[PerplexityTestResult] | None = None

        post_compression_model_profile: LLMInfo | None = None
        post_compression_inference_performance: InferencePerformanceInfo | None = None
        post_compression_model_performance: List[PerplexityTestResult] | None = None

        if pipeline_config.profile_hardware:
            hardware_profile = self.hardware_profiler.retrive_hardware_information()

        # Pré compression
        logger.info("profiling model")
        if pipeline_config.profile_llm.profile:
            pre_compression_model_profile = self.profile_model(
                self.model,
                self.tokenizer,
                pipeline_config.profile_llm.analyze_connections,
                pipeline_config.profile_llm.estimate_memory,
                pipeline_config.profile_llm.verbose,
            )
        logger.info("benchmarking inference time for uncompressed model")
        if pipeline_config.benchmark_inference_performance.benchmark:
            pre_compression_inference_performance = (
                self.inference_performance_benchmarker.time_inference(
                    self.model,
                    self.tokenizer,
                    pipeline_config.benchmark_inference_performance.prompt,
                    pipeline_config.benchmark_inference_performance.max_new_tokens,
                    pipeline_config.benchmark_inference_performance.num_runs,
                    pipeline_config.benchmark_inference_performance.warmup_runs,
                )
            )
            logger.debug(f"pre compression inference performance {pre_compression_inference_performance}")

        logger.info("benchmarking model performance for uncompressed model")
        if pipeline_config.benchmark_model_performance.benchmark:
            pre_compression_model_performance = (
                self.model_performance_benchmarker.benchmark(
                    self.model,
                    self.tokenizer,
                    pipeline_config.benchmark_model_performance.tests,
                    pipeline_config.benchmark_model_performance.batch_size,
                )
            )
            logger.debug(f"pre compression model performance {pre_compression_model_performance}")

        # Compression
        logger.info("compressing model")
        for prune_config in pipeline_config.prune_configs:
            logger.debug(prune_config)
            logger.debug(self.get_model())
            self.model = self.prune_model(self.get_model(), self.tokenizer, prune_config)

        # Post compression
        logger.info("profiling model after compression")
        if pipeline_config.profile_llm.profile:
            post_compression_model_profile = self.profile_model(
                self.model,
                self.tokenizer,
                pipeline_config.profile_llm.analyze_connections,
                pipeline_config.profile_llm.estimate_memory,
                pipeline_config.profile_llm.verbose,
            )
        logger.info("benchmarking inference time for compressed model")
        if pipeline_config.benchmark_inference_performance.benchmark:
            post_compression_inference_performance = (
                self.inference_performance_benchmarker.time_inference(
                    self.model,
                    self.tokenizer,
                    pipeline_config.benchmark_inference_performance.prompt,
                    pipeline_config.benchmark_inference_performance.max_new_tokens,
                    pipeline_config.benchmark_inference_performance.num_runs,
                    pipeline_config.benchmark_inference_performance.warmup_runs,
                )
            )
            logger.debug(f"post compression inference performance {post_compression_inference_performance}")

        logger.info("benchmarking model performance for compressed model")
        if pipeline_config.benchmark_model_performance.benchmark:
            post_compression_model_performance = (
                self.model_performance_benchmarker.benchmark(
                    self.model,
                    self.tokenizer,
                    pipeline_config.benchmark_model_performance.tests,
                    pipeline_config.benchmark_model_performance.batch_size,
                )
            )
            logger.debug(f"post compression model performance {post_compression_model_performance}")

        return PipelineReturn(
            hardware_profile=hardware_profile,
            llm_profile=LLMProfile(
                pre_compression=pre_compression_model_profile,
                post_compression=post_compression_model_profile,
            ),
            performance=Performance(
                inference_performance=InferencePerformance(
                    pre_compression=pre_compression_inference_performance,
                    post_compression=post_compression_inference_performance,
                    compared_results=CompareBenchmark(
                        speedup=pre_compression_inference_performance.avg_time / post_compression_inference_performance.avg_time,
                        tps_improvement=post_compression_inference_performance.tokens_per_second - pre_compression_inference_performance.tokens_per_second,
                    )
                ),
                model_performance=ModelPerformance(
                    pre_compression=pre_compression_model_performance,
                    post_compression=post_compression_model_performance,
                ),
            ),
        )
