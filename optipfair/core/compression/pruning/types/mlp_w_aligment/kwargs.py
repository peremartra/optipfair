from pydantic import BaseModel, ConfigDict


class MLPAlignmentPrunerKwargs(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    alignment: int
    prune_percent: float
