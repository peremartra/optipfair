"""
Trainer utilities for knowledge distillation.

This module exposes the public distillation API used to train a student model
from a teacher model with compound distillation losses.
"""

from __future__ import annotations

import math
import time
import warnings
from contextlib import contextmanager
from functools import wraps
from statistics import mean
from typing import Any, Callable, Dict, Iterator, List, Optional, Set, Tuple, Union

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import PreTrainedModel

from ..pruning.utils import _prepare_batch_inputs, get_text_config
from .loss import compute_distillation_loss
from .mapping import MAPPING_LAST, MAPPING_UNIFORM, resolve_layer_map


@contextmanager
def _temporarily_disable_use_cache(
    student_model: PreTrainedModel, teacher_model: PreTrainedModel
) -> Iterator[None]:
    configurations: List[Tuple[Any, bool, Any]] = []
    seen_config_ids: Set[int] = set()

    try:
        for model in (student_model, teacher_model):
            for config in (getattr(model, "config", None), get_text_config(model)):
                if config is None or id(config) in seen_config_ids:
                    continue
                seen_config_ids.add(id(config))
                has_use_cache = hasattr(config, "use_cache")
                original_value = getattr(config, "use_cache", None)
                configurations.append((config, has_use_cache, original_value))
                config.use_cache = False

        yield
    finally:
        for config, has_use_cache, original_value in reversed(configurations):
            if has_use_cache:
                config.use_cache = original_value
            elif hasattr(config, "use_cache"):
                delattr(config, "use_cache")


def _disable_use_cache_during_distillation(
    function: Callable[..., Any],
) -> Callable[..., Any]:
    @wraps(function)
    def wrapped(
        student_model: PreTrainedModel,
        teacher_model: PreTrainedModel,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        with _temporarily_disable_use_cache(student_model, teacher_model):
            return function(student_model, teacher_model, *args, **kwargs)

    return wrapped


@_disable_use_cache_during_distillation
def distill_model(
    student_model: PreTrainedModel,
    teacher_model: PreTrainedModel,
    dataloader: DataLoader,
    layer_mapping_strategy: str = "uniform",
    alpha: float = 0.6,
    beta: float = 0.4,
    gamma: float = 0.0,
    delta: float = 0.0,
    temperature: float = 2.0,
    skew_alpha: float = 0.4,
    epochs: int = 3,
    learning_rate: float = 4e-5,
    scheduler: str = "cosine",
    warmup_ratio: float = 0.05,
    accumulation_steps: int = 4,
    show_progress: bool = True,
    return_stats: bool = False,
    step_callback: Optional[Callable[[Dict[str, Any]], Optional[bool]]] = None,
    callback_every_steps: Optional[int] = None,
) -> Union[PreTrainedModel, Tuple[PreTrainedModel, Dict]]:
    """
    Distill a student model from a teacher model using compound losses.

    Args:
        student_model: Student model to train.
            Set `requires_grad=False` on parameters to freeze them before
            calling; only trainable parameters are passed to AdamW.
        teacher_model: Teacher model used as distillation target.
        dataloader: Training dataloader yielding token batches. Dict batches
            may include `labels`; positions set to -100 are ignored by all
            loss components. Without `labels`, targets come from `input_ids`
            and are masked by `attention_mask` when present.
        layer_mapping_strategy: Layer mapping strategy ("uniform" or "last").
        alpha: Hard labels weight.
        beta: Soft labels weight.
        gamma: Feature alignment weight.
        delta: Feature dynamics weight.
        temperature: Distillation temperature.
        skew_alpha: Skew interpolation factor.
        epochs: Number of epochs.
        learning_rate: AdamW learning rate.
        scheduler: Learning rate scheduler strategy ("cosine" or "none").
        warmup_ratio: Proportion of total optimization steps used for warmup.
        accumulation_steps: Gradient accumulation steps.
        show_progress: Whether to show tqdm progress bars.
        return_stats: Whether to return stats with the trained model.
        step_callback: Optional callback after optimizer steps. Return True to
            stop training cleanly.
        callback_every_steps: Callback interval in optimizer steps. Defaults
            to every step when a callback is provided.

    Returns:
        Trained student model, or (trained_model, stats_dict) when
        return_stats=True.

    Raises:
        ValueError: If loss weights are all zero, models are the same object,
            accumulation_steps is invalid, inputs are malformed, or mapping
            strategy is unsupported, callback arguments are invalid, or the
            student has no trainable parameters.
    """
    total_weight = alpha + beta + gamma + delta
    if math.isclose(total_weight, 0.0):
        raise ValueError("At least one loss weight (alpha, beta, gamma, delta) must be > 0.")

    if student_model is teacher_model:
        raise ValueError("student_model and teacher_model must be different objects.")

    if layer_mapping_strategy not in (MAPPING_UNIFORM, MAPPING_LAST):
        raise ValueError(f"Unsupported layer_mapping_strategy: '{layer_mapping_strategy}'.")

    if scheduler not in ("cosine", "none"):
        raise ValueError("scheduler must be one of {'cosine', 'none'}.")

    if not 0.0 <= warmup_ratio <= 1.0:
        raise ValueError("warmup_ratio must be between 0.0 and 1.0.")

    if accumulation_steps <= 0:
        raise ValueError("accumulation_steps must be a positive integer.")

    if step_callback is not None and not callable(step_callback):
        raise ValueError("step_callback must be callable.")
    if callback_every_steps is not None:
        if step_callback is None:
            raise ValueError(
                "callback_every_steps requires step_callback to be provided."
            )
        if (
            not isinstance(callback_every_steps, int)
            or isinstance(callback_every_steps, bool)
            or callback_every_steps < 1
        ):
            raise ValueError("callback_every_steps must be a positive integer.")
    callback_interval = callback_every_steps or 1

    if (gamma > 0 or delta > 0) and not math.isclose(total_weight, 1.0):
        warnings.warn(
            "Loss weights do not sum to 1.0. This is valid but may affect training scale.",
            UserWarning,
            stacklevel=2,
        )

    device = next(student_model.parameters()).device
    teacher_device = next(teacher_model.parameters()).device
    if teacher_device != device:
        teacher_model.to(device)

    student_model.train()
    teacher_model.eval()

    request_hidden_states = gamma > 0 or delta > 0
    layer_map_used: Optional[List[int]] = None
    layer_mapping_strategy_used: Optional[str] = None

    if request_hidden_states:
        layer_map_used = resolve_layer_map(student_model, teacher_model, layer_mapping_strategy)
        layer_mapping_strategy_used = layer_mapping_strategy

    trainable_parameters = [
        parameter for parameter in student_model.parameters() if parameter.requires_grad
    ]
    if not trainable_parameters:
        raise ValueError("student_model must have at least one trainable parameter.")
    optimizer = AdamW(trainable_parameters, lr=learning_rate)
    optimizer.zero_grad()

    steps_per_epoch = (
        len(dataloader) + accumulation_steps - 1
    ) // accumulation_steps
    total_steps = steps_per_epoch * epochs
    scheduler_total_steps = len(dataloader) // accumulation_steps * epochs
    warmup_steps = int(scheduler_total_steps * warmup_ratio)
    cosine_steps = scheduler_total_steps - warmup_steps

    scheduler_instance = None
    if scheduler == "cosine" and scheduler_total_steps > 0:
        if warmup_steps <= 0:
            scheduler_instance = CosineAnnealingLR(
                optimizer,
                T_max=max(1, cosine_steps),
                eta_min=learning_rate * 0.05,
            )
        elif cosine_steps <= 0:
            scheduler_instance = LinearLR(
                optimizer,
                start_factor=1e-2,
                end_factor=1.0,
                total_iters=max(1, warmup_steps),
            )
        else:
            warmup_scheduler = LinearLR(
                optimizer,
                start_factor=1e-2,
                end_factor=1.0,
                total_iters=warmup_steps,
            )
            cosine_scheduler = CosineAnnealingLR(
                optimizer,
                T_max=cosine_steps,
                eta_min=learning_rate * 0.05,
            )
            scheduler_instance = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_steps],
            )

    loss_history = {
        "total": [],
        "task": [],
        "logits": [],
        "trajectory": [],
        "derivative": [],
    }
    epoch_times: List[float] = []
    total_start_time = time.time()
    global_step = 0
    stopped_early = False

    def finish_optimizer_step(
        epoch: int,
        accumulated_losses: Dict[str, float],
        accumulation_counter: int,
        epoch_losses: Dict[str, List[float]],
    ) -> Tuple[Dict[str, float], bool]:
        nonlocal global_step

        torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler_instance is not None:
            scheduler_instance.step()
        optimizer.zero_grad()

        avg_losses = {
            key: value / accumulation_counter
            for key, value in accumulated_losses.items()
        }
        for key, value in avg_losses.items():
            epoch_losses[key].append(value)

        global_step += 1
        callback_requested_stop = False
        if step_callback is not None and global_step % callback_interval == 0:
            callback_info = {
                "model": student_model,
                "global_step": global_step,
                "total_steps": total_steps,
                "epoch": epoch,
                "losses": avg_losses,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
            try:
                callback_requested_stop = step_callback(callback_info) is True
            finally:
                student_model.train()

        return avg_losses, callback_requested_stop

    for epoch in range(epochs):
        epoch_start_time = time.time()
        epoch_losses = {key: [] for key in loss_history}
        accumulated_losses = {key: 0.0 for key in loss_history}
        accumulation_counter = 0

        progress_iter = tqdm(
            dataloader,
            desc=f"Epoch {epoch + 1}/{epochs}",
            disable=not show_progress,
        )

        for batch_idx, batch in enumerate(progress_iter):
            inputs = _prepare_batch_inputs(batch, device)
            if "input_ids" not in inputs:
                raise ValueError("Batch inputs must contain 'input_ids' for distillation training.")

            if "labels" in inputs:
                labels = inputs["labels"].clone()
            else:
                labels = inputs["input_ids"].clone()
                if "attention_mask" in inputs:
                    labels[inputs["attention_mask"] == 0] = -100
            model_inputs = {k: v for k, v in inputs.items() if k != "labels"}

            student_outputs = student_model(
                **model_inputs,
                output_hidden_states=request_hidden_states,
            )

            with torch.no_grad():
                teacher_outputs = teacher_model(
                    **model_inputs,
                    output_hidden_states=request_hidden_states,
                )

            student_hiddens = (
                list(student_outputs.hidden_states[1:]) if request_hidden_states else None
            )
            teacher_hiddens = (
                list(teacher_outputs.hidden_states[1:]) if request_hidden_states else None
            )

            loss, loss_dict = compute_distillation_loss(
                student_logits=student_outputs.logits,
                teacher_logits=teacher_outputs.logits,
                student_hiddens=student_hiddens,
                teacher_hiddens=teacher_hiddens,
                labels=labels,
                layer_map=layer_map_used,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                delta=delta,
                temperature=temperature,
                skew_alpha=skew_alpha,
            )

            (loss / accumulation_steps).backward()

            for key in accumulated_losses:
                accumulated_losses[key] += loss_dict[key]
            accumulation_counter += 1

            if (batch_idx + 1) % accumulation_steps == 0:
                avg_losses, callback_requested_stop = finish_optimizer_step(
                    epoch, accumulated_losses, accumulation_counter, epoch_losses
                )

                postfix = {
                    "loss": f"{avg_losses['total']:.4f}",
                    "task": f"{avg_losses['task']:.4f}",
                    "logits": f"{avg_losses['logits']:.4f}",
                }
                if gamma > 0:
                    postfix["traj"] = f"{avg_losses['trajectory']:.4f}"
                if delta > 0:
                    postfix["deriv"] = f"{avg_losses['derivative']:.4f}"
                progress_iter.set_postfix(postfix)

                accumulated_losses = {key: 0.0 for key in loss_history}
                accumulation_counter = 0
                if callback_requested_stop:
                    stopped_early = True
                    break

        if accumulation_counter > 0:
            avg_losses, callback_requested_stop = finish_optimizer_step(
                epoch, accumulated_losses, accumulation_counter, epoch_losses
            )
            if callback_requested_stop:
                stopped_early = True

        for key, values in epoch_losses.items():
            if values:
                loss_history[key].append(mean(values))
            else:
                loss_history[key].append(0.0)

        epoch_times.append(time.time() - epoch_start_time)
        if stopped_early:
            break

    total_time = time.time() - total_start_time

    if not return_stats:
        return student_model

    dataloader_batch_size = getattr(dataloader, "batch_size", None)
    effective_batch_size = (
        dataloader_batch_size * accumulation_steps
        if isinstance(dataloader_batch_size, int)
        else None
    )

    stats = {
        "epochs": epochs,
        "learning_rate": learning_rate,
        "accumulation_steps": accumulation_steps,
        "effective_batch_size": effective_batch_size,
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
        "delta": delta,
        "temperature": temperature,
        "skew_alpha": skew_alpha,
        "layer_mapping_strategy": layer_mapping_strategy_used,
        "layer_map_used": layer_map_used,
        "loss_history": loss_history,
        "epoch_times_seconds": epoch_times,
        "total_time_seconds": total_time,
        "avg_time_per_epoch": (mean(epoch_times) if epoch_times else 0.0),
    }
    if step_callback is not None:
        stats["stopped_early"] = stopped_early
        stats["global_steps_done"] = global_step

    return student_model, stats
