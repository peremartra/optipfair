from pydantic import BaseModel
from core.analysis.profiling.models.llm import (
    MeasureInferenceTime,
    AnalyzeConnections,
    EstimateMemory,
)


class LLMProfilerOptions(BaseModel):
    measure_inference_time: MeasureInferenceTime
    analyze_connections: AnalyzeConnections
    estimate_memory: EstimateMemory
