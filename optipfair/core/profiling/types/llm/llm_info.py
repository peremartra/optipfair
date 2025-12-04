from pydantic import BaseModel
from core.profiling.types.llm import (
    ArchitectureInfo,
    MemoryUsageInfo,
    ParameterInfo,
    ModelSummary,
    AttentionLayerAnalysisInfo,
    MemoryEstimationInfo,
)


class LLMInfo(BaseModel):
    parameters: ParameterInfo
    architecture: ArchitectureInfo
    memory_usage: MemoryUsageInfo | None = None
    memory_estimation: MemoryEstimationInfo
    summary: ModelSummary
    attention_layers: AttentionLayerAnalysisInfo
