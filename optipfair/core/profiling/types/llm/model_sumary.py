from pydantic import BaseModel


class ModelSummary(BaseModel):
    model_name: str
    device: str
    model_class: str
    pytorch_version: str
