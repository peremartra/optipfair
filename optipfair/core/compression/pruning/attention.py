from core.compression.pruning.base import BasePruner
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


class AttentionPruner(BasePruner):
    def prune(self, *args, **kwargs):
        """
        Placeholder for future implementation of attention head pruning.

        Args:
            model: Model to prune
            head_importance_method: Method to calculate head importance
            prune_percentage: Percentage of heads to prune

        Returns:
            model: Pruned model
        """
        # validate kwargs
        # parsed_kwargs = AttentionPrunerKwargs.model_validate(**kwargs)
        raise NotImplementedError("Attention pruning is not yet implemented.")
