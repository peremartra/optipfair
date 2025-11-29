from pydantic import BaseModel
from typing import Dict
from core.analysis.profiling.models.llm import LayerConnectionInfo


class ConnectionAnalysisInfo(BaseModel):
    """Complete connection analysis information."""

    total_connections: int
    connection_graph: Dict[str, LayerConnectionInfo]
    analysis_method: str
    has_skip_connections: bool
    max_fan_in: int
    max_fan_out: int
