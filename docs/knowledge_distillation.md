# Knowledge Distillation

Knowledge Distillation (KD) is available in OptiPFair from version 0.4.0 through the Python API `opf.distill_model`.

This page explains how to distill a student model from a teacher model, especially after pruning, to recover quality while keeping the efficiency gains.

## Quick Start

```python
import optipfair as opf
from transformers import AutoModelForCausalLM
from torch.utils.data import DataLoader

teacher_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-0.8B-Base")
student_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-0.8B-Base")

# Optional: prune student first
student_model = opf.prune_model(
    model=student_model,
    pruning_type="DEPTH",
    num_layers_to_remove=2,
)

# Your dataloader must provide input_ids (and optionally attention_mask)
train_dataloader = DataLoader(...)

trained_student, stats = opf.distill_model(
    student_model=student_model,
    teacher_model=teacher_model,
    dataloader=train_dataloader,
    alpha=0.6,
    beta=0.4,
    gamma=0.0,
    delta=0.0,
    temperature=2.0,
    skew_alpha=0.4,
    epochs=3,
    learning_rate=4e-5,
    scheduler="cosine",
    warmup_ratio=0.05,
    accumulation_steps=4,
    show_progress=True,
    return_stats=True,
)

print("Final total loss:", stats["loss_history"]["total"][-1])
```

## API Summary

```python
import optipfair as opf

trained_student = opf.distill_model(
    student_model=student_model,
    teacher_model=teacher_model,
    dataloader=dataloader,
    layer_mapping_strategy="uniform",
    alpha=0.6,
    beta=0.4,
    gamma=0.0,
    delta=0.0,
    temperature=2.0,
    skew_alpha=0.4,
    epochs=3,
    learning_rate=4e-5,
    scheduler="cosine",
    warmup_ratio=0.05,
    accumulation_steps=4,
    show_progress=True,
    return_stats=False,
)
```

### Key Parameters

- `student_model`: Model to train in place.
- `teacher_model`: Distillation teacher (must be a different object).
- `dataloader`: Must provide `input_ids` in each batch.
- `alpha`, `beta`, `gamma`, `delta`: Loss weights (at least one must be `> 0`).
- `temperature`: Soft logits temperature.
- `skew_alpha`: Interpolation between student/teacher soft targets.
- `layer_mapping_strategy`: `"uniform"` or `"last"` (used when `gamma > 0` or `delta > 0`).
- `scheduler`: `"cosine"` or `"none"`.
- `warmup_ratio`: Warmup fraction in `[0.0, 1.0]`.
- `accumulation_steps`: Positive integer.

## Validation Rules

`opf.distill_model` raises `ValueError` when:

- `alpha + beta + gamma + delta == 0`
- `student_model is teacher_model`
- Invalid `layer_mapping_strategy`
- Invalid `scheduler`
- `warmup_ratio` is outside `[0.0, 1.0]`
- `accumulation_steps <= 0`
- Batch does not contain `input_ids`

It emits `UserWarning` when `gamma > 0` or `delta > 0` and loss weights do not sum to `1.0`.

## Recommended Starting Configs

### Fast Baseline

```python
import optipfair as opf

trained_student = opf.distill_model(
    student_model=student_model,
    teacher_model=teacher_model,
    dataloader=dataloader,
    alpha=0.6,
    beta=0.4,
    gamma=0.0,
    delta=0.0,
    temperature=2.0,
    skew_alpha=0.4,
    scheduler="cosine",
    warmup_ratio=0.05,
    accumulation_steps=4,
)
```

### With Representation Alignment

```python
import optipfair as opf

trained_student = opf.distill_model(
    student_model=student_model,
    teacher_model=teacher_model,
    dataloader=dataloader,
    layer_mapping_strategy="last",
    alpha=0.55,
    beta=0.35,
    gamma=0.05,
    delta=0.05,
    temperature=2.0,
    skew_alpha=0.3,
    scheduler="cosine",
    warmup_ratio=0.15,
    accumulation_steps=6,
)
```

## Return Stats

With `return_stats=True`, the function returns `(trained_student, stats)`.

The `stats` dictionary includes:

- Training setup (`epochs`, `learning_rate`, `accumulation_steps`)
- Effective batch size
- Loss weights and distillation hyperparameters
- Layer mapping details
- Loss history per epoch (`total`, `task`, `logits`, `trajectory`, `derivative`)
- Timing metrics

## Notebooks

- [knowledge_distillation.ipynb](https://github.com/peremartra/optipfair/blob/main/examples/knowledge_distillation.ipynb)
- [knowledge_distillation_express.ipynb](https://github.com/peremartra/optipfair/blob/main/examples/knowledge_distillation_express.ipynb)

## See Also

- [Usage](usage.md)
- [Examples](examples.md)
- [API Reference](api.md)
