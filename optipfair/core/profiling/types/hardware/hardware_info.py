from pydantic import BaseModel, Field
from core.analysis.profiling.models.hardware import CPUInfo, RAMInfo, SystemGPUInfo
from datetime import datetime


class HardwareProfile(BaseModel):
    cpu: CPUInfo = Field(..., description="Information about the CPU")
    ram: RAMInfo = Field(..., description="Information about the RAM")
    gpu: SystemGPUInfo = Field(..., description="Information about the GPU(s)")
    timestamp: datetime = Field(
        default_factory=datetime.now, description="When the info was collected"
    )
