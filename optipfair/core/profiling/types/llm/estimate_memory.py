from pydantic import BaseModel


class EstimateMemory(BaseModel):
    sequence_length: int
    batch_size: int
