from pydantic import BaseModel
from typing import Optional, Any, Tuple, Callable


class AnalyzeConnections(BaseModel):
    sample_input: Optional[Callable[[], Any]] = None
    input_shape: Optional[Tuple[int, ...]]
