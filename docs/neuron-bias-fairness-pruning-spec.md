# Technical Specifications: Neuron Bias Analysis & Fairness Pruning Scores
## optipfair — `optipfair/bias/` module

**Version target:** 0.2.x  
**Author:** Pere Martra  
**Status:** Ready for implementation

---

## Context

The current bias module captures activations for individual prompt pairs and returns raw tensors. To support Fairness Pruning research, we need to:

1. Allow filtering which layer types are captured during the forward pass (memory optimization for large models) using exact matching from a closed vocabulary.
2. Add `analyze_neuron_bias` — processes a batch of prompt pairs, robustly handling `batch_size > 1` and asymmetric sequence lengths, and returns an aggregated per-neuron BiasScore.
3. Add `compute_fairness_pruning_scores` — combines BiasScore with ImportanceScore (via the existing `compute_neuron_pair_importance_maw`) to produce the final FairnessPruningScore per neuron.

---

## Change 1 — Add `target_layers` parameter to `register_hooks`, `process_prompt`, and `get_activation_pairs`

### Motivation

For large models (7B+), capturing all activations simultaneously consumes significant memory. Filtering at hook registration time avoids storing tensors that will never be used. To ensure multi-architecture support, hook registration must use the existing `get_model_layers` utility.

### 1.1 `register_hooks(model, target_layers=None)`

**File:** `optipfair/bias/activations.py`

**Current signature:**
```python
def register_hooks(model) -> List[Any]:
```

**New signature:**
```python
from typing import Literal

# Allowed target layers to prevent accidental string matching.
# Includes "input_norm" for backward compatibility with the current hook
# that captures input_layernorm activations.
ALLOWED_TARGET_LAYERS = frozenset({
    "gate_proj", "up_proj", "down_proj", "down_proj_input", "mlp_output", "attention", "input_norm"
})

def register_hooks(model, target_layers=None) -> List[Any]:
```

**Behavior:**
- `target_layers=None` → register hooks on all supported layers, identical to current behavior. No breaking change.
- `target_layers=["gate_proj", "up_proj"]` → register hooks only on layers whose key matches exactly the allowed prefixes.
- `target_layers=["down_proj_input"]` → register only pre-hook captures at `down_proj` input, stored as `down_proj_input_layer_{i}`.
- `target_layers=["down_proj", "down_proj_input"]` → capture both post-projection and pre-projection activations with separate key families.
- If `target_layers` contains values outside `ALLOWED_TARGET_LAYERS`, raise `ValueError` with a clear message listing valid options. Validation uses the `frozenset` constant at runtime (not `Literal`, which is not iterable).
- **Matching logic:** Instead of a generic substring match (`if "gate" in name`), use exact prefix matching from the closed vocabulary to avoid accidental captures.
- **Implementation note:** Remove the current hardcoded `try/elif` blocks (`model.model.layers`, `model.transformer.h`, etc.). Instead, import and use `get_model_layers(model)` from `optipfair.pruning.utils` to iterate over the layers dynamically. This introduces a new cross-module dependency (`bias/` → `pruning/utils`), which is acceptable since `pruning.utils` contains architecture-agnostic utilities.
- **Backward compatibility:** The current code captures 6 hook types per layer: `attention_output`, `mlp_output`, `gate_proj`, `up_proj`, `down_proj`, and `input_norm`. All 6 must continue to be captured when `target_layers=None`.

### 1.1.1 `down_proj` vs `down_proj_input` semantics

- `down_proj` captures the output of `down_proj` and remains unchanged (shape `[B, S, hidden_size]`).
- `down_proj_input` captures the input to `down_proj` using a forward pre-hook (shape `[B, S, intermediate_size]`).
- `down_proj_input` is explicit opt-in and must not be included implicitly when `target_layers=None`.

### 1.2 `process_prompt(model, tokenizer, prompt, target_layers=None)`

**File:** `optipfair/bias/activations.py`

**Current signature:**
```python
def process_prompt(model, tokenizer, prompt) -> Dict[str, torch.Tensor]:
```

**New signature:**
```python
def process_prompt(model, tokenizer, prompt, target_layers=None) -> Dict[str, torch.Tensor]:
```

**Behavior:**
- `target_layers=None` → passes `None` to `register_hooks`, captures all layers. No breaking change.
- `target_layers=[...]` → propagated to `register_hooks`.

### 1.3 `get_activation_pairs(model, tokenizer, prompt1, prompt2, target_layers=None)`

**File:** `optipfair/bias/activations.py`

**Current signature:**
```python
def get_activation_pairs(model, tokenizer, prompt1, prompt2) -> tuple:
```

**New signature:**
```python
def get_activation_pairs(model, tokenizer, prompt1, prompt2, target_layers=None) -> tuple:
```

**Behavior:**
- `target_layers=None` → captures all layers. No breaking change.
- `target_layers=[...]` → propagated to both `process_prompt` calls.

---

## Change 2 — New function `analyze_neuron_bias`

### Signature

```python
def analyze_neuron_bias(
    model,
    tokenizer,
    prompt_pairs,
    target_layers=None,
    aggregation="mean",
    show_progress=True,
) -> Dict[str, torch.Tensor]:
```

**File:** `optipfair/bias/activations.py`

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | HF CausalLM | required | Model to analyze |
| `tokenizer` | HF Tokenizer | required | Matching tokenizer |
| `prompt_pairs` | `List[Tuple[str, str]]` | required | Prompt pairs differing only in demographic attribute |
| `target_layers` | `Optional[List[str]]` | `None` | Layer type prefixes to capture. `None` defaults internally to `["gate_proj", "up_proj"]` |
| `aggregation` | `str` | `"mean"` | Aggregation across pairs: `"mean"` or `"max"` |
| `show_progress` | `bool` | `True` | Show tqdm progress bar |

> **Note on `target_layers` default:** inside `analyze_neuron_bias`, `None` is resolved to `["gate_proj", "up_proj"]` because these are the layers relevant for MLP GLU fairness pruning. This avoids capturing unnecessary activations. **Important:** this differs from `register_hooks` where `None` means "all layers". This is intentional — `analyze_neuron_bias` is optimized for the fairness pruning use case.

> **Note on `prompt_pairs` input:** expects a plain Python list of tuples `(prompt_1, prompt_2)`.

### Return value

```python
Dict[str, torch.Tensor]
```

A dictionary mapping layer keys to per-neuron aggregated BiasScore tensors of shape `[intermediate_size]`. All returned tensors are on CPU.

### Internal logic (Robust against batch sizes and varying sequence lengths)

```python
from tqdm import tqdm

# Resolve default target_layers
if target_layers is None:
    target_layers = ["gate_proj", "up_proj"]

# Validate aggregation
if aggregation not in ("mean", "max"):
    raise ValueError(f"aggregation must be 'mean' or 'max', got '{aggregation}'")

# Validate prompt pairs
if not prompt_pairs:
    raise ValueError("prompt_pairs cannot be empty")

pairs = prompt_pairs

# Initialize accumulator: layer_key → List[Tensor of shape [intermediate_size]]
accumulated_diffs: Dict[str, List[torch.Tensor]] = {}

for prompt_1, prompt_2 in tqdm(pairs, disable=not show_progress,
                                desc="Analyzing bias across prompt pairs"):
    act1, act2 = get_activation_pairs(model, tokenizer, prompt_1, prompt_2,
                                       target_layers=target_layers)
    
    for layer_key in act1:
        if layer_key not in act2:
            continue
        
        # STRATEGY FOR VARYING SEQUENCE LENGTHS & BATCH SIZE > 1:
        # We cannot do act1 - act2 directly because token lengths might differ.
        # Step 1: Mean pooling over all dimensions except the last one
        # (supports [B,S,I] and other shapes where the last dim is neuron index).
        act1_tensor = act1[layer_key].float()
        act2_tensor = act2[layer_key].float()

        reduce_dims_1 = tuple(range(act1_tensor.ndim - 1))
        reduce_dims_2 = tuple(range(act2_tensor.ndim - 1))

        if not reduce_dims_1 or not reduce_dims_2:
            raise ValueError(f"Activation tensor for {layer_key} must have at least 2 dimensions")

        act1_pooled = act1_tensor.mean(dim=reduce_dims_1)
        act2_pooled = act2_tensor.mean(dim=reduce_dims_2)
        
        # Step 2: Compute absolute difference on the pooled representations
        neuron_diff = torch.abs(act1_pooled - act2_pooled) 
        
        if layer_key not in accumulated_diffs:
            accumulated_diffs[layer_key] = []
            
        accumulated_diffs[layer_key].append(neuron_diff.cpu())  # Move to CPU to save VRAM

# Aggregate accumulated diffs across all prompt pairs
result = {}
for layer_key, diffs_list in accumulated_diffs.items():
    stacked = torch.stack(diffs_list)  # [num_pairs, intermediate_size]
    if aggregation == "mean":
        result[layer_key] = stacked.mean(dim=0)
    elif aggregation == "max":
        result[layer_key] = stacked.max(dim=0).values

return result
```

---

## Change 3 — New function `compute_fairness_pruning_scores`

### Motivation

Combines `BiasScore_i` with `ImportanceScore_i` (via `compute_neuron_pair_importance_maw`).

**FairnessPruningScore formula:**
```
FairnessPruningScore_i = bias_weight × BiasScore_norm_i
                       + (1 - bias_weight) × (1 - ImportanceScore_norm_i)
```

A high FairnessPruningScore means the neuron is a strong pruning candidate: high bias sensitivity AND/OR low structural importance.

### Signature

```python
def compute_fairness_pruning_scores(
    model,
    bias_scores,
    bias_weight=0.5,
) -> Dict[int, torch.Tensor]:
```

**File:** `optipfair/bias/activations.py`

### Internal logic

```python
from optipfair.pruning.mlp_glu import compute_neuron_pair_importance_maw
from optipfair.pruning.utils import get_model_layers

def _normalize(t: torch.Tensor) -> torch.Tensor:
    """
    Min-max normalization to [0, 1].
    If max == min (e.g., all values are identical), returns a tensor of zeros 
    to prevent division by zero, representing no relative differences.
    """
    t_min = t.min()
    t_max = t.max()
    if torch.isclose(t_max, t_min):
        return torch.zeros_like(t)
    return (t - t_min) / (t_max - t_min + 1e-8)

def compute_fairness_pruning_scores(model, bias_scores, bias_weight=0.5):

    # Validate parameters
    if not 0.0 <= bias_weight <= 1.0:
        raise ValueError(f"bias_weight must be in [0.0, 1.0], got {bias_weight}")
    
    if not bias_scores:
        raise ValueError("bias_scores dictionary is empty")

    result = {}

    # Multi-architecture support
    layers = get_model_layers(model)

    for i, layer in enumerate(layers):

        gate_bias_key = f"gate_proj_layer_{i}"
        up_bias_key   = f"up_proj_layer_{i}"

        available = [bias_scores[k] for k in (gate_bias_key, up_bias_key)
                     if k in bias_scores]

        if not available:
            continue

        bias = torch.stack(available).mean(dim=0)

        # Static ImportanceScore from weights (Ensure layer is GLU)
        if not hasattr(layer, "mlp") or not hasattr(layer.mlp, "gate_proj") or not hasattr(layer.mlp, "up_proj"):
            continue

        gate_weight = layer.mlp.gate_proj.weight.data.float()
        up_weight   = layer.mlp.up_proj.weight.data.float()
        importance  = compute_neuron_pair_importance_maw(gate_weight, up_weight).cpu()

        bias_norm       = _normalize(bias)
        importance_norm = _normalize(importance)

        fairness_score = (bias_weight * bias_norm
                         + (1 - bias_weight) * (1 - importance_norm))

        result[i] = fairness_score

    return result
```

---

## Phase 2.2 Integration Plan — Using FairnessPruningScores in `prune_model`

### Current Limitation

Currently, `prune_model()` and `prune_model_mlp_glu()` only accept neuron importance via string enumeration (`neuron_selection_method="MAW"`, etc.), which are computed internally. The `compute_fairness_pruning_scores()` returns precomputed tensors that cannot be directly injected into the pruning pipeline.

### Proposed Solution (Phase 2.2)

To enable end-to-end fairness-aware pruning, two functions require modification:

#### 2.2.1 `prune_neuron_pairs()` — Accept pre-computed scores

**Current signature:**
```python
def prune_neuron_pairs(
    mlp: nn.Module,
    prune_percentage: float,
    importance_fn: Callable = compute_neuron_pair_importance_maw,
    activation_norms: Optional[torch.Tensor] = None,
    layer_idx: Optional[int] = None,
    expansion_divisor: Optional[int] = None,
) -> Tuple[nn.Linear, nn.Linear, nn.Linear, int]:
```

**Modified signature (Phase 2.2):**
```python
def prune_neuron_pairs(
    mlp: nn.Module,
    prune_percentage: float,
    importance_fn: Callable = compute_neuron_pair_importance_maw,
    activation_norms: Optional[torch.Tensor] = None,
    layer_idx: Optional[int] = None,
    expansion_divisor: Optional[int] = None,
    custom_importance_scores: Optional[torch.Tensor] = None,
) -> Tuple[nn.Linear, nn.Linear, nn.Linear, int]:
```

**New logic:**
```python
if custom_importance_scores is not None:
    # Use pre-computed scores (e.g., from fairness pruning)
    if custom_importance_scores.shape[0] != mlp.gate_proj.weight.size(0):
        raise ValueError(f"custom_importance_scores shape mismatch")
    importance_scores = custom_importance_scores
else:
    # Fall back to internal computation (existing behavior)
    if activation_norms is not None:
        importance_scores = compute_neuron_pair_importance_maw_hybrid(...)
    else:
        importance_scores = importance_fn(gate_weight, up_weight)
```

> **CRITICAL — Score semantics:** `prune_neuron_pairs` uses `torch.topk(importance_scores, k, largest=True)`, meaning it **keeps** neurons with the **highest** scores. However, `FairnessPruningScore` is designed so that **high = prune candidate**. Therefore, when injecting fairness scores, they must be **inverted** before being used as `custom_importance_scores`:
> ```python
> custom_importance_scores = 1.0 - fairness_scores[layer_idx]
> ```
> This inversion converts "high bias → prune" into "low importance → prune", which aligns with the existing `topk(largest=True)` logic.

#### 2.2.2 `prune_model_mlp_glu()` — Accept fairness score dict

**Current signature:**
```python
def prune_model_mlp_glu(
    model: PreTrainedModel,
    neuron_selection_method: str = "PPM",
    pruning_percentage: Optional[float] = 10,
    ...
) -> PreTrainedModel:
```

**Modified signature (Phase 2.2):**
```python
def prune_model_mlp_glu(
    model: PreTrainedModel,
    neuron_selection_method: str = "PPM",
    pruning_percentage: Optional[float] = 10,
    ...,
    fairness_scores: Optional[Dict[int, torch.Tensor]] = None,
) -> PreTrainedModel:
```

**New logic in pruning loop:**
```python
for layer_idx, layer in enumerate(layers_to_prune):
    # Select scores based on fairness or neuron_selection_method
    if fairness_scores is not None and layer_idx in fairness_scores:
        # INVERT: FairnessPruningScore high = prune candidate,
        # but topk(largest=True) keeps highest scores.
        custom_scores = 1.0 - fairness_scores[layer_idx]
    else:
        custom_scores = None
    
    # Prune with appropriate scores
    prune_neuron_pairs(
        mlp=layer.mlp,
        prune_percentage=prune_percentage,
        ...,
        custom_importance_scores=custom_scores,
    )
```

#### 2.2.3 Example usage (Phase 2.2)

```python
from optipfair import prune_model
from optipfair.bias import analyze_neuron_bias, compute_fairness_pruning_scores

# Step 1: Compute bias and fairness scores
bias_scores = analyze_neuron_bias(model, tokenizer, prompt_pairs)
fairness_scores = compute_fairness_pruning_scores(model, bias_scores, bias_weight=0.6)

# Step 2: Prune using fairness scores
pruned_model = prune_model(
    model,
    pruning_type="MLP_GLU",
    pruning_percentage=10,
    fairness_scores=fairness_scores,  # NEW parameter (Phase 2.2)
)
```

### Implementation Notes

- The `fairness_scores` parameter is **optional** to maintain backward compatibility.
- If both `neuron_selection_method` and `fairness_scores` are provided, `fairness_scores` takes precedence.
- Tensors in `fairness_scores` must be on CPU and match the intermediate size of each layer.
- This change is **non-breaking**: existing code without `fairness_scores` continues to work unchanged.

---

## Changes to `__init__.py`

**File:** `optipfair/bias/__init__.py`

```python
from .visualization import (
    visualize_bias,
    visualize_mean_differences,
    visualize_heatmap,
    visualize_pca,
)
from .metrics import calculate_bias_metrics
from .activations import (
    get_activation_pairs,
    analyze_neuron_bias,
    compute_fairness_pruning_scores,
)

__all__ = [
    "visualize_bias",
    "visualize_mean_differences",
    "visualize_heatmap",
    "visualize_pca",
    "calculate_bias_metrics",
    "get_activation_pairs",
    "analyze_neuron_bias",
    "compute_fairness_pruning_scores",
]
```

---

## Testing checklist

### Change 1 — `target_layers` propagation & Architecture
- [ ] `register_hooks(model)` uses `get_model_layers(model)` and captures all compatible layers correctly.
- [ ] **BACKWARD COMPAT:** `register_hooks(model)` with no arguments captures the same 6 hook types as before: `attention_output`, `mlp_output`, `gate_proj`, `up_proj`, `down_proj`, `input_norm`.
- [ ] `register_hooks(model, target_layers=["gate_proj"])` strictly matches `gate_proj` prefix (no accidental substring matches).
- [ ] `register_hooks(model, target_layers=["invalid"])` raises `ValueError`.
- [ ] **BACKWARD COMPAT:** `process_prompt(model, tokenizer, prompt)` (without `target_layers`) returns the same keys as before.
- [ ] **BACKWARD COMPAT:** `get_activation_pairs(model, tokenizer, p1, p2)` (without `target_layers`) returns the same keys as before.
- [ ] `process_prompt` and `get_activation_pairs` — propagation tests with explicit `target_layers`.

### `analyze_neuron_bias`
- [ ] **CRITICAL:** Tests with input pairs where `len(prompt_1) != len(prompt_2)` execute without `RuntimeError` due to shape mismatch.
- [ ] **CRITICAL:** Tests with `batch_size > 1` correctly reduce to `[intermediate_size]` without throwing dimension errors.
- [ ] Accepts plain list of tuples correctly.
- [ ] Return tensors are on CPU and shape is exactly `[intermediate_size]`.
- [ ] `target_layers=None` resolves to `["gate_proj", "up_proj"]`.
- [ ] `aggregation="max"` returns max across pairs.
- [ ] Empty `prompt_pairs` list raises clear error.
- [ ] tqdm progress bar works without raising `NameError`.

### `compute_fairness_pruning_scores`
- [ ] Compatible with non-LLaMA architectures using `get_model_layers(model)`.
- [ ] Return keys are layer indices (`int`), shape is `[intermediate_size]`, on CPU.
- [ ] `bias_weight=1.0` (pure bias) and `bias_weight=0.0` (pure importance) behave correctly.
- [ ] Out-of-bounds `bias_weight` raises `ValueError`.
- [ ] Empty `bias_scores` raises `ValueError`.
- [ ] Normalization handles `max == min` safely (returns tensor of zeros).
- [ ] Layers missing expected GLU attributes are skipped gracefully.

### Integration
- [ ] No OOM on Llama-3.2-3B with 100 pairs on Colab A100.

### Backward Compatibility (regression)
- [ ] All existing tests in `test_bias_visualization.py` pass without modification.
- [ ] `visualization.py` functions (`visualize_bias`, `visualize_mean_differences`, etc.) work without changes — they call `get_activation_pairs` without `target_layers` and must continue to receive all hook types.
- [ ] `bias/__init__.py` continues to export all existing symbols. New exports (`analyze_neuron_bias`, `compute_fairness_pruning_scores`) are additive only.
- [ ] `completebias_test.py` and example notebooks continue to work unchanged.