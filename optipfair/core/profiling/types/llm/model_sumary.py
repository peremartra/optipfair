from pydantic import BaseModel


class ModelSummary(BaseModel):
    device: str
    model_class: str
    pytorch_version: str
