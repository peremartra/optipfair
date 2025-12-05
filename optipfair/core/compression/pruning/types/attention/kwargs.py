from pydantic import BaseModel, ConfigDict, field_validator
from typing import Literal
from transformers import PreTrainedModel


class AttentionPrunerKwargs(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)


    model: PreTrainedModel
    head_importance_method: Literal["ATTENTION_WEIGHTS"] = "ATTENTION_WEIGHTS"
    prune_percentage: float = 10

    @field_validator("model")
    @classmethod
    def validate_model(cls, v):
        if not isinstance(v, PreTrainedModel):
            raise ValueError(
                f"model must be an instance of PreTrainedModel, got {type(v).__name__}"
            )
        return v
