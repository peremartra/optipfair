"""
Loss utilities for knowledge distillation.

This module provides the compound distillation loss used to combine hard-label
supervision, teacher logits alignment, hidden-state trajectory alignment, and
feature-dynamics alignment.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .mapping import create_layer_map_uniform


def compute_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    student_hiddens: Optional[List[torch.Tensor]],
    teacher_hiddens: Optional[List[torch.Tensor]],
    labels: torch.Tensor,
    layer_map: Optional[List[int]] = None,
    alpha: float = 0.6,
    beta: float = 0.4,
    gamma: float = 0.0,
    delta: float = 0.0,
    temperature: float = 2.0,
    skew_alpha: float = 0.4,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute the compound knowledge distillation loss for a student model.

    The total loss can combine up to four components:
    1. Task loss with hard labels.
    2. Skewed KLD logits alignment inspired by DistiLLM-2 (2024), interpolating
       between student and teacher target distributions.
    3. Trajectory loss using cosine similarity between aligned hidden states.
    4. FDD derivative loss based on cosine similarity between consecutive hidden
       state deltas, following the feature dynamics distillation idea described
       in the ACL 2025 FDD work.

    Args:
        student_logits: Student logits with shape [batch, seq_len, vocab_size].
        teacher_logits: Teacher logits with shape [batch, seq_len, vocab_size].
        student_hiddens: Student hidden states, one tensor per aligned layer.
        teacher_hiddens: Teacher hidden states, one tensor per aligned layer.
        labels: Target token ids with shape [batch, seq_len].
            Positions with value -100 are ignored as invalid tokens.
        layer_map: Teacher layer indices aligned to each student layer.
        alpha: Hard labels weight: learns from ground truth (dataset).
        beta: Soft labels weight: learns from teacher's output distribution.
        gamma: Feature alignment weight: matches hidden state representations
            layer-by-layer.
        delta: Feature dynamics weight: matches how representations change
            between layers.
        temperature: Temperature used for logits distillation.
        skew_alpha: Interpolation factor for the skewed target distribution.

    Returns:
        Tuple containing the total loss tensor and a dictionary with the float
        values for total, task, logits, trajectory, and derivative losses.
    """
    device = student_logits.device

    if (
        (gamma > 0 or delta > 0)
        and layer_map is None
        and student_hiddens is not None
        and teacher_hiddens is not None
    ):
        layer_map = create_layer_map_uniform(len(student_hiddens), len(teacher_hiddens))

    shift_logits = student_logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss_task = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )
    valid_mask = shift_labels != -100
    num_valid = valid_mask.sum().clamp(min=1)

    with torch.no_grad():
        student_probs = F.softmax(student_logits[..., :-1, :] / temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits[..., :-1, :] / temperature, dim=-1)
        mixed_probs = skew_alpha * student_probs + (1 - skew_alpha) * teacher_probs

    student_log_probs = F.log_softmax(student_logits[..., :-1, :] / temperature, dim=-1)
    kl_elementwise = student_probs * (student_log_probs - torch.log(mixed_probs + 1e-9))
    kl_per_token = kl_elementwise.sum(dim=-1)
    loss_logits = (kl_per_token * valid_mask).sum() / num_valid
    loss_logits = loss_logits * (temperature ** 2)

    if gamma > 0 and student_hiddens is not None and teacher_hiddens is not None:
        loss_trajectory = 0.0
        flat_mask = valid_mask.reshape(-1)
        for student_idx, teacher_idx in enumerate(layer_map):
            student_h = student_hiddens[student_idx][:, :-1, :].contiguous()
            teacher_h = teacher_hiddens[teacher_idx][:, :-1, :].contiguous()
            student_flat = student_h.reshape(-1, student_h.size(-1))
            teacher_flat = teacher_h.reshape(-1, teacher_h.size(-1))
            student_norm = F.normalize(student_flat, p=2, dim=1)
            teacher_norm = F.normalize(teacher_flat, p=2, dim=1)
            cos_sim_per_token = (student_norm * teacher_norm).sum(dim=1)
            cos_sim = (cos_sim_per_token * flat_mask).sum() / num_valid
            loss_trajectory += 1 - cos_sim
        loss_trajectory = loss_trajectory / len(layer_map)
    else:
        loss_trajectory = torch.tensor(0.0, device=device)

    loss_derivative = torch.tensor(0.0, device=device)
    if delta > 0 and student_hiddens is not None and teacher_hiddens is not None:
        num_derivatives = 0
        flat_mask = valid_mask.reshape(-1)
        for student_idx in range(len(layer_map) - 1):
            teacher_idx = layer_map[student_idx]
            teacher_idx_next = layer_map[student_idx + 1]
            student_delta = (
                student_hiddens[student_idx + 1][:, :-1, :]
                - student_hiddens[student_idx][:, :-1, :]
            )
            teacher_delta = (
                teacher_hiddens[teacher_idx_next][:, :-1, :]
                - teacher_hiddens[teacher_idx][:, :-1, :]
            )
            student_delta_flat = student_delta.reshape(-1, student_delta.size(-1))
            teacher_delta_flat = teacher_delta.reshape(-1, teacher_delta.size(-1))
            student_delta_norm = F.normalize(student_delta_flat, p=2, dim=1)
            teacher_delta_norm = F.normalize(teacher_delta_flat, p=2, dim=1)
            cos_sim_deriv_per_token = (student_delta_norm * teacher_delta_norm).sum(dim=1)
            cos_sim_deriv = (cos_sim_deriv_per_token * flat_mask).sum() / num_valid
            loss_derivative += 1 - cos_sim_deriv
            num_derivatives += 1
        if num_derivatives > 0:
            loss_derivative = loss_derivative / num_derivatives

    total_loss = (
        alpha * loss_task
        + beta * loss_logits
        + gamma * loss_trajectory
        + delta * loss_derivative
    )

    loss_dict = {
        "total": total_loss.item(),
        "task": loss_task.item(),
        "logits": loss_logits.item(),
        "trajectory": loss_trajectory.item(),
        "derivative": loss_derivative.item(),
    }

    return total_loss, loss_dict