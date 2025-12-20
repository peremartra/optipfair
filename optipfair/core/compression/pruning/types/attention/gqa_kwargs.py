from pydantic import BaseModel, ConfigDict


class GroupedQueryAttentionPrunerKwargs(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    num_kv_heads_to_keep: int
