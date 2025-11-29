from pydantic import BaseModel, Field, field_validator, ValidationInfo
from typing import Dict, Optional, Union
from core.analysis.profiling.models.llm import PrecisionType


class MemoryEstimate(BaseModel):
    """Memory estimation for a specific precision."""

    precision: str = Field(..., description="Precision type used for estimation")
    bytes_per_parameter: float = Field(
        ..., gt=0, description="Bytes per parameter for this precision"
    )
    total_memory_mb: float = Field(..., ge=0, description="Total memory in megabytes")
    total_memory_gb: float = Field(..., ge=0, description="Total memory in gigabytes")
    model_weights_mb: float = Field(
        ..., ge=0, description="Model weights memory in megabytes"
    )
    kv_cache_mb: Optional[float] = Field(
        None, ge=0, description="KV cache memory in megabytes"
    )
    activation_memory_mb: Optional[float] = Field(
        None, ge=0, description="Activation memory in megabytes"
    )
    total_inference_memory_mb: Optional[float] = Field(
        None, ge=0, description="Total inference memory in megabytes"
    )

    gradient_memory_mb: Optional[float] = None
    optimizer_memory_mb: Optional[float] = None
    training_memory_mb: Optional[float] = None
    total_training_memory_mb: Optional[float] = None

    @field_validator("precision")
    def validate_precision(cls, v):
        """Validate that precision is one of the supported types."""
        valid_precisions = [precision.value for precision in PrecisionType]
        if v not in valid_precisions:
            raise ValueError(f"Precision must be one of {valid_precisions}, got {v}")
        return v

    @field_validator("total_memory_gb")
    def validate_memory_consistency(cls, v: float, values: ValidationInfo):
        """Validate that GB and MB values are consistent."""
        if "total_memory_mb" in values.data.keys():
            expected_gb = values.data["total_memory_mb"] / 1024
            if abs(v - expected_gb) > 0.001:
                raise ValueError(
                    f"total_memory_gb ({v}) is inconsistent with total_memory_mb ({values['total_memory_mb']})"
                )
        return v

    class Config:
        """Pydantic configuration."""

        validate_assignment = True
        extra = "forbid"


class MemoryEstimationInfo(BaseModel):
    """Complete memory estimation across different precisions."""

    estimates: Dict[str, MemoryEstimate] = Field(
        ..., description="Memory estimates for different precisions"
    )
    base_parameters: int = Field(
        ..., gt=0, description="Base number of parameters in the model"
    )

    @field_validator("estimates")
    def validate_estimates_keys(cls, v):
        """Validate that all estimate keys are valid precision types."""
        valid_precisions = [precision.value for precision in PrecisionType]
        for precision in v.keys():
            if precision not in valid_precisions:
                raise ValueError(
                    f"Invalid precision key '{precision}'. Must be one of {valid_precisions}"
                )
        return v

    @field_validator("estimates")
    def validate_estimates_consistency(cls, v, values):
        """Validate that estimates are consistent with their keys."""
        for precision_key, estimate in v.items():
            if estimate.precision != precision_key:
                raise ValueError(
                    f"Estimate precision '{estimate.precision}' doesn't match key '{precision_key}'"
                )
        return v

    def get_estimate(
        self, precision: Union[PrecisionType, str]
    ) -> Optional[MemoryEstimate]:
        """Get memory estimate for a specific precision."""
        if isinstance(precision, PrecisionType):
            precision = precision.value
        return self.estimates.get(precision)

    def add_estimate(self, estimate: MemoryEstimate) -> None:
        """Add a new memory estimate."""
        self.estimates[estimate.precision] = estimate

    def get_all_precisions(self) -> list[str]:
        """Get all available precision types."""
        return list(self.estimates.keys())

    def get_lowest_memory_estimate(self) -> Optional[MemoryEstimate]:
        """Get the estimate with the lowest memory usage."""
        if not self.estimates:
            return None
        return min(self.estimates.values(), key=lambda x: x.total_memory_mb)

    def get_highest_memory_estimate(self) -> Optional[MemoryEstimate]:
        """Get the estimate with the highest memory usage."""
        if not self.estimates:
            return None
        return max(self.estimates.values(), key=lambda x: x.total_memory_mb)

    class Config:
        """Pydantic configuration."""

        validate_assignment = True
        extra = "forbid"
