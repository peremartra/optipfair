from pydantic import BaseModel, Field
from typing import Literal


class CPUInfo(BaseModel):
    name: str = Field(
        ...,
        description="The name of the CPU, eg: AMD Ryzen 7 5700G with Radeon Graphics",
    )
    architecture: Literal["x86_64", "arm64"]
    platform: str = Field(
        ...,
        description="OS and version, eg: Linux-6.8.0-65-generic-x86_64-with-glibc2.35",
    )
    max_freq: float
    physical_cores: int
    total_cores: int
