"""
Distillation package for OptiPFair.

This package provides knowledge distillation utilities for recovering
performance after pruning.
"""

from .trainer import distill_model
from .mapping import MAPPING_UNIFORM, MAPPING_LAST

__all__ = [
	"distill_model",
	"MAPPING_UNIFORM",
	"MAPPING_LAST",
]
