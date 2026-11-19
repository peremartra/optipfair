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
    step_callback=None,
    callback_every_steps=None,
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
- `layer_map`: Optional keyword-only teacher block index per student block. Overrides the strategy and must be strictly increasing when `delta > 0`.
- `scheduler`: `"cosine"` or `"none"`.
- `warmup_ratio`: Warmup fraction in `[0.0, 1.0]`.
- `accumulation_steps`: Positive integer.
- `step_callback`: Optional function called after optimizer steps.
- `callback_every_steps`: Positive interval in optimizer steps; defaults to
  every step when a callback is provided.

### Explicit Layer Mapping

`layer_map` maps each student block to a teacher block by index. These are
decoder-block output indices: index `t` refers to teacher block `t` in
`hidden_states[1:]`; the embedding state at index `0` is excluded. The map must
have one in-range integer (not a boolean) per student block. Repeated and
out-of-order indices are allowed with `gamma` alone. When `delta > 0`, an
explicit map must be strictly increasing so consecutive student transitions
align with forward-moving teacher transitions.

`uniform` spreads student blocks across the teacher depth, and `last` aligns
them to the teacher's final blocks. Neither strategy knows which original
blocks were removed by non-uniform depth pruning. In hybrid models, a generated
map can therefore align different block types, such as linear-attention and
full-attention blocks, or point at teacher blocks that were removed from the
student. Pass the original teacher indices explicitly to preserve those
alignments.

For example, if a 32-block teacher is pruned by removing blocks
`[1, 4, 24, 25, 26, 27, 28, 29]`, the remaining 24 blocks map as follows:

```python
layer_map = [
    0, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13,
    14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 30, 31,
]

student, stats = opf.distill_model(
    student_model=student_model,
    teacher_model=teacher_model,
    dataloader=train_dataloader,
    layer_map=layer_map,
    gamma=0.1,
    delta=0.1,
    alpha=0.4,
    beta=0.4,
    return_stats=True,
)
```

The final student block maps to teacher block 31 in this example. In
architectures where the final returned hidden state has passed through a final
normalization layer but intermediate block states have not, matching the last
block to the last block avoids that particular final-versus-intermediate
comparison. This behavior is model-dependent. When `gamma` and `delta` are both
zero, a supplied map is validated and a warning explains that it will not be
used. `stats["layer_mapping_strategy"]` reports `"explicit"` when the map is
active, and `stats["layer_map_used"]` contains the applied map.

### Labels and step callbacks

Dictionary batches may include `labels`. Positions labeled `-100` are ignored
by the task, logits, trajectory, and derivative losses, allowing completion-
only training. Without `labels`, labels are derived from `input_ids` and
padding is masked using `attention_mask`. Only dictionary batches support
explicit labels; tuple/TensorDataset batches do not map a field to `labels`.
`stats["loss_history"]` continues to contain epoch averages.

The optional callback gives access to each optimizer-step result without
changing the epoch-level history:

```python
def on_step(info):
    print(info["global_step"], info["losses"]["total"])
    info["model"].save_pretrained(f"checkpoint-{info['global_step']}")
    return False  # Return True to stop training cleanly.

trained_student, stats = opf.distill_model(
    student_model=student_model,
    teacher_model=teacher_model,
    dataloader=train_dataloader,
    step_callback=on_step,
    callback_every_steps=100,
    return_stats=True,
)
```

The callback receives `model`, `global_step` (starting at 1), `total_steps`,
zero-based `epoch`, averaged `losses`, and the current `learning_rate`. It runs
after the optimizer and scheduler steps and after gradients are cleared. It is
not wrapped in `no_grad`; the student is returned to training mode after it
returns. Callback exceptions propagate, while the distillation cache settings
are still restored. With a callback and `return_stats=True`, stats also include
`stopped_early` and `global_steps_done`. Saving model weights does not save
optimizer or scheduler state for resuming the same training run.
When no callback is supplied, the return shape and stats keys remain unchanged.

## Controlling Student Memory Use

`distill_model` trains the student parameters whose `requires_grad` flag is
`True` when the call starts. Freeze any parameters you do not want to update
before calling it; the function does not freeze model components automatically.
For example, to freeze selected components, adapt the parameter-name prefixes
to the architecture you loaded:

```python
frozen_prefixes = ("model.visual.", "model.embed_tokens.")
for name, parameter in student_model.named_parameters():
    if name.startswith(frozen_prefixes):
        parameter.requires_grad_(False)

trained_student = opf.distill_model(
    student_model=student_model,
    teacher_model=teacher_model,
    dataloader=train_dataloader,
)
```

Only parameters with `requires_grad=True` are given to AdamW. A PEFT-wrapped
student can also be passed when it preserves the model's forward interface.
The function raises `ValueError` if no student parameters are trainable.

These controls can reduce gradient and optimizer-state memory, but do not
guarantee that every model fits on a particular GPU. The teacher and student
still need to fit, and logits loss memory depends on the number of unmasked
tokens and vocabulary size. The logits loss selects valid positions before
softmax, which avoids vocabulary-sized intermediate tensors for masked tokens.
Gradient checkpointing, quantized optimizers, and automatic component freezing
are not provided by `distill_model`.

## Validation Rules

`opf.distill_model` raises `ValueError` when:

- `alpha + beta + gamma + delta == 0`
- `student_model is teacher_model`
- Invalid `layer_mapping_strategy`
- Invalid `scheduler`
- `warmup_ratio` is outside `[0.0, 1.0]`
- `accumulation_steps <= 0`
- Batch does not contain `input_ids`
- The student has no trainable parameters
- `callback_every_steps` is provided without `step_callback`
- `callback_every_steps` is not a positive integer

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
