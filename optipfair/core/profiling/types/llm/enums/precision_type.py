from enum import Enum


class PrecisionType(str, Enum):
    """Supported precision types for memory estimation."""

    FP32 = "float32"
    FP16 = "float16"
    BFLOAT16 = "bfloat16"
    INT8 = "int8"
    INT4 = "int4"
