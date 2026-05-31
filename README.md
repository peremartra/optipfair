<div align="center">
  <img src="images/optiPfair.png" alt="optipfair Logo" width="600"/>
  <h1>optipfair</h1>
  <strong>Structured pruning and knowledge distillation for large language models.</strong>
</div>

<p align="center">
  <a href="https://pypi.org/project/optipfair/"><img alt="PyPI Version" src="https://img.shields.io/pypi/v/optipfair?color=blue"></a>
  <a href="https://pypi.org/project/optipfair/"><img alt="Downloads" src="https://img.shields.io/pypi/dm/optipfair?color=orange"></a>
  <a href="https://github.com/peremartra/optipfair/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/github/license/peremartra/optipfair?color=green"></a>
  <a href="https://github.com/peremartra/optipfair/stargazers"><img alt="GitHub Stars" src="https://img.shields.io/github/stars/peremartra/optipfair?style=social"></a>
</p>

<div align="center">
  <a href="https://peremartra.github.io/optipfair/">Documentation</a>
  ·
  <a href="https://github.com/peremartra/optipfair/issues">Report Bug</a>
  ·
  <a href="https://github.com/peremartra/optipfair/issues">Request Feature</a>
  
</div>


>**Companion Library:** OptiPFair is the official open-source implementation for the upcoming Manning book **[Rearchitecting LLMs](https://hubs.la/Q040tvsK0)**. Explore the theory, mechanics, and advanced research behind these algorithms in the [book's repository](https://github.com/peremartra/Rearchitecting-LLMs).
---

## The Pipeline

OptiPFair gives you a complete, composable workflow to compress a model and recover its performance.

```
[ Analyze ]  →  [ Prune ]  →  [ Distill ]  →  [ Deploy ]
  Which           Depth          Recover         Smaller,
  Struct          and/or         performance     faster
  matter?         Width          with KD         model
```

You can apply **depth pruning** (remove entire layers), **width pruning** (reduce neuron count per layer), or **both sequentially** on the same model. Knowledge distillation is optional but recommended after aggressive pruning.

```python
import optipfair as opf

# 1. Analyze layer importance (returns a Dict[int, float])
importance = opf.analyze_layer_importance(model, dataloader)

# 2. Dynamically identify the N least important layers
n_layers_to_remove = 5
sorted_layers = sorted(importance.items(), key=lambda x: x[1])
least_important_indices = [idx for idx, score in sorted_layers[:n_layers_to_remove]]

print(f"Targeting layers for removal: {least_important_indices}")

# 3a. Depth Pruning: Remove the identified "passive" layers entirely.
model, depth_stats = opf.prune_model(
    model=model, 
    pruning_type="DEPTH", 
    layer_indices=least_important_indices,
    return_stats=True
)

# 3b. Width Pruning: Reduce neuron count in the remaining GLU MLP layers.
model, width_stats = opf.prune_model(
    model=model, 
    pruning_type="MLP_GLU",
    pruning_percentage=20, 
    neuron_selection_method="MAW", # "MAW" is the alias for PPM[cite: 8, 53].
    return_stats=True
)

# 4. Performance Recovery: Fine-tune the student model using Knowledge Distillation.
trained_model, distill_stats = opf.distill_model(
    student_model=model,
    teacher_model=teacher,
    dataloader=train_dataloader,
    epochs=4,
    alpha=0.6,  # Weight for task-specific loss[cite: 29].
    beta=0.4,   # Weight for soft label (logits) loss[cite: 30].
    return_stats=True,
)
```

Four functions. One pipeline.

---

## Pruning Strategies

### Depth Pruning — Remove entire layers
Best for aggressive size reduction. Eliminates the least important transformer blocks entirely based on calibration data.

```python
# Analyze which layers contribute least
importance = opf.analyze_layer_importance(model, dataloader)

# Remove the bottom N layers by importance score
student = opf.prune_model_depth(model, layer_indices=[21, 20, 9, 8, 17])
```

**Results on Qwen3.5-0.8B-Base** — 10 layers removed, A100, Cosmopedia (40K samples):

| Metric          | Teacher | Student (KD) |
|-----------------|---------|--------------|
| Layers          | 24      | 14           |
| Parameters      | 752M    | **540M** |
| Reduction       | —       | **−28.2%** |
| Winogrande      | 59.4%   | 54.8%        |
| PIQA            | 71.5%   | 64.5%        |
| Avg Benchmark   | 60.8%   | 48.9%        |
| **Performance Retention** | **100%** | **80.4%** |

Also validated on **Llama-3.2-3B**.

---

### Width Pruning — Reduce neurons per layer
Best for fine-grained control. Shrinks the internal dimensions of MLP layers while preserving depth. The default method is **PPM (Peak-to-Peak Magnitude)**, formally described in:

> *Martra, P. (2025). Fragile Knowledge, Robust Instruction-Following: The Width Pruning Dichotomy in Llama-3.2. [ArXiv](https://arxiv.org/abs/2512.22671)*

```python
model = opf.prune_model(
    model=model,
    pruning_ratio=0.2,           # Remove 20% of neurons
    method="PPM",
    layers_to_prune=[10, 15, 20],
)
```

> For backward compatibility, `"MAW"` is still accepted and maps to PPM.

---

### Combining Both
Apply depth pruning first to eliminate the weakest layers, then width pruning to fine-tune the remaining ones.

```python
# First pass: remove the least important layers
model = opf.prune_model_depth(model, layer_indices=[21, 20, 9, 8, 17])

# Second pass: slim down the surviving layers
model = opf.prune_model(model, pruning_ratio=0.15, method="PPM")

# Recover with knowledge distillation
trained_model, stats = opf.distill_model(
    student_model=model,
    teacher_model=teacher,
    dataloader=train_dataloader,
    epochs=4,
    return_stats=True,
)
```

---

## Installation

```bash
pip install optipfair

# With visualization dependencies (bias analysis)
pip install optipfair[viz]
```

---

## Notebooks

| Notebook | Description | Link |
|----------|-------------|------|
| **Knowledge Distillation** | Full pipeline: prune → distill → push to HF → lm_eval benchmarks → VRAM comparison | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/peremartra/optipfair/blob/main/examples/knowledge_distillation.ipynb) |
| **Knowledge Distillation Express** | Compact distillation loop with lm_eval benchmarks | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/peremartra/optipfair/blob/main/examples/knowledge_distillation_express.ipynb) |
| **Depth Pruning** | Remove entire transformer layers from models like Llama-3 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/peremartra/optipfair/blob/main/examples/depth_pruning.ipynb) |
| **Layer Importance Analysis** | Identify which layers contribute least to model performance | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/peremartra/optipfair/blob/main/examples/layer_importance_analysis.ipynb) |

---

## Knowledge Distillation — API Reference

`opf.distill_model()` trains the pruned model to recover performance using a combined loss (cross-entropy + Skew KLD). No custom training loop required.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `alpha` | `0.6` | Weight of task loss (cross-entropy) |
| `beta` | `0.40` | Weight of logits loss (Skew KLD) |
| `gamma` / `delta` | `0.0` | Feature alignment — set to 0 for labels-only distillation |
| `temperature` | `2` | Softens teacher distribution for transfer |
| `scheduler` | `"cosine"` | LR scheduler (`"cosine"` or `"linear"`) |
| `warmup_ratio` | `0.15` | Fraction of steps for LR warmup |
| `return_stats` | `False` | Returns `loss_history` dict for plotting |

> **Tip:** Use `optipfair_llm_reference_manual.txt` with any LLM assistant (ChatGPT, Claude) for guided help tuning these parameters.

---

## Bias Analysis

OptiPFair includes tools to visualize how a model processes demographic attributes — analyzing internal activations rather than outputs. Three visualization types: PCA, Mean Difference, and Layer Heatmap.

**[🚀 Try the Live Demo on HF Spaces](https://huggingface.co/spaces/oopere/optipfair-bias-analyzer)** — no setup required.

For a deployable REST API with Gradio frontend, see **[OptiPFair-API](https://github.com/peremartra/optipfair-api)**.

---

## Citation

```bibtex
@software{optipfair2025,
  author = {Martra, Pere},
  title = {OptiPFair: Structured Pruning and Knowledge Distillation for Large Language Models},
  version = {0.4.0},
  year = {2025},
  doi = {10.5281/zenodo.20473491},
  url = {https://github.com/peremartra/optipfair}
}
```

---

<div align="center">

**[⭐ Star this repo](https://github.com/peremartra/optipfair/stargazers) · [🐛 Report Bug](https://github.com/peremartra/optipfair/issues) · [📖 Documentation](https://peremartra.github.io/optipfair/)**

Made with ❤️ by [Pere Martra](https://github.com/peremartra)

</div>
