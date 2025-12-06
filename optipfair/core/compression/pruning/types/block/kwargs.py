from pydantic import BaseModel, ConfigDict
from typing import Literal


class BlockPrunerKwargs(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    block_importance_method: Literal["PERPLEXITY"] = "PERPLEXITY"
    prune_percentage: float = 10
