from pydantic import BaseModel, field_validator
from typing import Literal, Union
from core.compression.pruning.types.depth.kwargs import DepthPrunerKwargs
from core.compression.pruning.types.mlp_glu.kwargs import MlpGluPrunerKwargs
from core.compression.pruning.types.block.kwargs import BlockPrunerKwargs
from core.compression.pruning.types.attention.kwargs import AttentionPrunerKwargs



class PruneConfig(BaseModel):
    prune_technique: Literal['block', 'depth', 'mlp_glu', 'attention']
    prune_technique_kwargs: Union[DepthPrunerKwargs, MlpGluPrunerKwargs, BlockPrunerKwargs, AttentionPrunerKwargs]

    @field_validator('prune_technique_kwargs')
    @classmethod
    def validate_prune_technique_kwargs(cls, v):
        if cls.prune_technique == 'block' and not isinstance(v, BlockPrunerKwargs):
            raise ValueError(f"trying to prune using {cls.prune_technique} prune with wrong kwargs class")
        if cls.prune_technique == 'depth' and not isinstance(v, DepthPrunerKwargs):
            raise ValueError(f"trying to prune using {cls.prune_technique} prune with wrong kwargs class")
        if cls.prune_technique == 'mlp_glu' and not isinstance(v, MlpGluPrunerKwargs):
            raise ValueError(f"trying to prune using {cls.prune_technique} prune with wrong kwargs class")
        if cls.prune_technique == 'attention' and not isinstance(v, AttentionPrunerKwargs):
            raise ValueError(f"trying to prune using {cls.prune_technique} prune with wrong kwargs class")
        
        return v
