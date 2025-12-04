from pydantic import BaseModel, Field
from typing import Dict, Optional

class MetricResult(BaseModel):
    value: float = Field(..., description="Metric value")
    stderr: float = Field(..., description="Standard error of the metric")


class TaskMetrics(BaseModel):
    alias: str = Field(..., description="Task alias/name")
    perplexity: Optional[MetricResult] = Field(None, description="Perplexity metric")
    perplexity_stderr: Optional[MetricResult] = Field(None, description="Perplexity stderr")
    acc: Optional[MetricResult] = Field(None, description="Accuracy metric")
    acc_stderr: Optional[MetricResult] = Field(None, description="Accuracy stderr")

    class Config:
        extra = "forbid"


class ModelPerformanceInfo(BaseModel):
    """Complete evaluation results from evaluator.simple_evaluate()"""
    results: Dict[str, TaskMetrics]

    class Config:
        extra = "forbid"