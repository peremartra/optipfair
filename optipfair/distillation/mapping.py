"""
Layer mapping strategies for knowledge distillation.

These utilities compute which teacher transformer block each student block
should be aligned with during feature alignment training.
"""

from typing import List

from transformers import PreTrainedModel

MAPPING_UNIFORM = "uniform"
MAPPING_LAST = "last"


def create_layer_map_uniform(n_student: int, n_teacher: int) -> List[int]:
    """
    Uniform Transformer Block mapping: Distribute student blocks proportionally
    across teacher blocks.

    Args:
        n_student: Number of student Transformer Blocks
        n_teacher: Number of teacher Transformer Blocks

    Returns:
        List of teacher Transformer Block indices for each student block
    """
    teacher_indices = []
    for i in range(n_student):
        teacher_idx = int(i * n_teacher / n_student)
        teacher_indices.append(teacher_idx)
    return teacher_indices


def create_layer_map_last(n_student: int, n_teacher: int) -> List[int]:
    """
    Last-Transformer-Block alignment: Map student blocks to the deepest
    teacher blocks.

    Args:
        n_student: Number of student Transformer Blocks
        n_teacher: Number of teacher Transformer Blocks

    Returns:
        List of teacher Transformer Block indices for each student block
    """
    offset = n_teacher - n_student
    return [i + offset for i in range(n_student)]


def resolve_layer_map(
    student_model: PreTrainedModel,
    teacher_model: PreTrainedModel,
    strategy: str,
) -> List[int]:
    """
    Resolve the layer map between student and teacher models using the
    specified strategy.

    Args:
        student_model: The pruned student model
        teacher_model: The original teacher model
        strategy: One of MAPPING_UNIFORM or MAPPING_LAST

    Returns:
        List of teacher block indices for each student block

    Raises:
        ValueError: If strategy is not supported
    """
    n_student = student_model.config.num_hidden_layers
    n_teacher = teacher_model.config.num_hidden_layers

    if strategy == MAPPING_UNIFORM:
        return create_layer_map_uniform(n_student, n_teacher)
    elif strategy == MAPPING_LAST:
        return create_layer_map_last(n_student, n_teacher)
    else:
        raise ValueError(
            f"Unsupported layer_mapping_strategy: '{strategy}'. "
            f"Choose from: '{MAPPING_UNIFORM}', '{MAPPING_LAST}'."
        )
