from pydantic import BaseModel


from pydantic import Field
from typing import List, Optional


class GPUMemoryInfo(BaseModel):
    total: int = Field(..., description="Total GPU memory in bytes")
    allocated: int = Field(..., description="Currently allocated GPU memory in bytes")
    cached: int = Field(..., description="Cached GPU memory in bytes")
    reserved: int = Field(..., description="Reserved GPU memory in bytes")
    free: int = Field(..., description="Free GPU memory in bytes")

    total_gb: float = Field(..., description="Total GPU memory in GB")
    allocated_gb: float = Field(..., description="Currently allocated GPU memory in GB")
    cached_gb: float = Field(..., description="Cached GPU memory in GB")
    reserved_gb: float = Field(..., description="Reserved GPU memory in GB")
    free_gb: float = Field(..., description="Free GPU memory in GB")


class GPUProperties(BaseModel):
    name: str = Field(..., description="GPU device name")
    major: int = Field(..., description="CUDA capability major version")
    minor: int = Field(..., description="CUDA capability minor version")
    total_memory: int = Field(..., description="Total GPU memory in bytes")
    multi_processor_count: int = Field(..., description="Number of multiprocessors")
    max_threads_per_multi_processor: int = Field(
        ..., description="Max threads per multiprocessor"
    )
    max_threads_per_block: int = Field(..., description="Max threads per block")
    max_block_dim: List[int] = Field(
        ..., description="Maximum block dimensions [x, y, z]"
    )
    max_grid_dim: List[int] = Field(
        ..., description="Maximum grid dimensions [x, y, z]"
    )
    warp_size: int = Field(..., description="Warp size")


class GPUInfo(BaseModel):
    device_id: int = Field(..., description="GPU device ID")
    device_name: str = Field(..., description="GPU device name")
    is_available: bool = Field(..., description="Whether the GPU is available")
    memory_info: GPUMemoryInfo = Field(..., description="GPU memory information")
    properties: GPUProperties = Field(..., description="GPU device properties")
    temperature: Optional[float] = Field(
        None, description="GPU temperature in Celsius (if available)"
    )
    utilization: Optional[float] = Field(
        None, description="GPU utilization percentage (if available)"
    )


class SystemGPUInfo(BaseModel):
    cuda_available: bool = Field(..., description="Whether CUDA is available")
    cuda_version: Optional[str] = Field(None, description="CUDA version")
    cudnn_version: Optional[str] = Field(None, description="cuDNN version")
    device_count: int = Field(..., description="Number of available GPU devices")
    current_device: Optional[int] = Field(
        None, description="Currently selected GPU device"
    )
    gpus: List[GPUInfo] = Field(
        default_factory=list, description="List of GPU information"
    )
