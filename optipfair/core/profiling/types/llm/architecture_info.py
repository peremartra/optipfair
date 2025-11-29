from pydantic import BaseModel
from typing import Dict, Any, List, Optional
from core.analysis.profiling.models.llm import ConnectionAnalysisInfo


class ArchitectureInfo(BaseModel):
    total_layers: int
    layer_types_count: Dict[str, Any]
    layer_details: List[Dict[str, Any]]
    max_depth: Any | int
    connections: Optional[ConnectionAnalysisInfo] = None
