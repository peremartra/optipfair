from pydantic import BaseModel, ConfigDict
from typing import Literal


class AttentionPrunerKwargs(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    head_importance_method: Literal["ATTENTION_WEIGHTS"] = "ATTENTION_WEIGHTS"
    prune_percentage: float = 10
