from core.compression.pruning.base import BasePruner
from transformers import PreTrainedModel
from pydantic import BaseModel, ConfigDict, field_validator
from typing import Literal



class BlockPrunerKwargs(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: PreTrainedModel
    block_importance_method: Literal["PERPLEXITY"] = "PERPLEXITY"
    prune_percentage: float = 10

    @field_validator("model")
    @classmethod
    def validate_model(cls, v):
        if not isinstance(v, PreTrainedModel):
            raise ValueError(
                f"model must be an instance of PreTrainedModel, got {type(v).__name__}"
            )
        return v

class BlockPruner(BasePruner):
    
    def prune(self, *args, **kwargs) -> PreTrainedModel:
        """
        Placeholder for future implementation of transformer block pruning.

        Args:
            model: Model to prune
            block_importance_method: Method to calculate block importance
            prune_percentage: Percentage of blocks to prune

        Returns:
            model: Pruned model
        """
        # validate kwargs
        # parsed_kwargs = BlockPrunerKwargs.model_validate(**kwargs)
        raise NotImplementedError("Block pruner yet to be implemented")
