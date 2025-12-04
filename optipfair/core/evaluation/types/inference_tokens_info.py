from pydantic import BaseModel

class InferenceTokensInfo(BaseModel):
    tokens_per_second: float
    total_input_tokens: int
    total_output_tokens: int