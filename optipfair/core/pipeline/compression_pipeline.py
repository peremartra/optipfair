from transformers import PreTrainedModel, PreTrainedTokenizerBase
from core.pipeline.types.prune_config import PruneConfig
from typing import List
from core.profiling.hardware_profiler import HardwareProfiler
from core.profiling.types.hardware import HardwareProfile
from core.profiling.llm_profiler import LLMProfiler
from core.profiling.types.llm import LLMInfo, AnalyzeConnections, EstimateMemory
from core.evaluation.benchmarks import ModelBenchmarker


class CompressionPipeline:
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, prune_techniques: List[PruneConfig]):
        self.prune_techniques = prune_techniques
        self.model = model
        self.tokenizer = tokenizer


    def profile_hardware(self) -> HardwareProfile:
        profiler = HardwareProfiler()
        return profiler.retrive_hardware_information()
    
    def profile_model(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, model_name: str, analyze_connections: AnalyzeConnections, estimate_memory: EstimateMemory, verbose: bool = True) -> LLMInfo:
        profiler = LLMProfiler(model, tokenizer, model_name, verbose)
        return profiler.profile_complete(analyze_connections=analyze_connections, estimate_memory=estimate_memory)
    
    def benchmark_model_performance(self):
        pass

    def run():
        pass

