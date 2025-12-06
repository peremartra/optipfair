from pydantic import BaseModel, model_validator
from typing import Literal, Union
from core.compression.pruning.types.depth.kwargs import DepthPrunerKwargs
from core.compression.pruning.types.mlp_glu.kwargs import MlpGluPrunerKwargs
from core.compression.pruning.types.block.kwargs import BlockPrunerKwargs
from core.compression.pruning.types.attention.kwargs import AttentionPrunerKwargs


class PruneConfig(BaseModel):
    prune_technique: Literal["block", "depth", "mlp_glu", "attention"]
    prune_technique_kwargs: Union[
        DepthPrunerKwargs, MlpGluPrunerKwargs, BlockPrunerKwargs, AttentionPrunerKwargs
    ]

    @model_validator(mode='after')
    def validate_prune_technique_kwargs(self):
        if self.prune_technique == "block" and not isinstance(self.prune_technique_kwargs, BlockPrunerKwargs):
            raise ValueError(
                f"trying to prune using {self.prune_technique} prune with wrong kwargs class"
            )
        if self.prune_technique == "depth" and not isinstance(self.prune_technique_kwargs, DepthPrunerKwargs):
            raise ValueError(
                f"trying to prune using {self.prune_technique} prune with wrong kwargs class"
            )
        if self.prune_technique == "mlp_glu" and not isinstance(self.prune_technique_kwargs, MlpGluPrunerKwargs):
            raise ValueError(
                f"trying to prune using {self.prune_technique} prune with wrong kwargs class"
            )
        if self.prune_technique == "attention" and not isinstance(
            self.prune_technique_kwargs, AttentionPrunerKwargs
        ):
            raise ValueError(
                f"trying to prune using {self.prune_technique} prune with wrong kwargs class"
            )
        return self
