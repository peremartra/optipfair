from pydantic import BaseModel
from core.analysis.profiling.models.hardware import HardwareProfile
from typing import Optional
from core.analysis.profiling.models.llm import LLMInfo


class CompleteProfileOutput(BaseModel):
    hardware_profile: HardwareProfile
    llm_profile: Optional[LLMInfo]
