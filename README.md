# optipfair

<div align="center">

  <img src="images/optiPfair.png" alt="optipfair Logo" width="600"/>

</div>

<div align="center">
  <h1>optipfair</h1>
  <strong>The Python library for making LLMs both efficient (via pruning) and fair (via bias analysis).</strong>
</div>

<p align="center">
  <a href="https://pypi.org/project/optipfair/"><img alt="PyPI Version" src="https://img.shields.io/pypi/v/optipfair?color=blue"></a>
  <a href="https://pypi.org/project/optipfair/"><img alt="Downloads" src="https://img.shields.io/pypi/dm/optipfair?color=orange"></a>
  <a href="https://github.com/peremartra/optipfair/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/github/license/peremartra/optipfair?color=green"></a>
  <a href="https://github.com/peremartra/optipfair/stargazers"><img alt="GitHub Stars" src="https://img.shields.io/github/stars/peremartra/optipfair?style=social"></a>
</p>

<div align="center">
    <h3>
        <a href="https://peremartra.github.io/optipfair/" target="_blank">Documentation</a>
        ·
        <a href="https://github.com/peremartra/optipfair/issues" target="_blank">Report Bug</a>
        ·
        <a href="https://github.com/peremartra/optipfair/issues" target="_blank">Request Feature</a>
    </h3>
</div>

---
> **New to optipfair?** Use our [LLM Reference Manual](optipfair_llm_reference_manual.txt) - paste it into ChatGPT, Claude or your Favourite LLM for guided assistance with any optipfair task.
### 🚀 Interactive Demos: Try optipfair NOW

Experience optipfair's capabilities directly in your browser.

| Live Bias Visualization Demo |
| :--------------------------: |
| Analyze any compatible model from Hugging Face with a full UI. No setup required. |
| **[🚀 Launch the Live Demo on HF Spaces](https://huggingface.co/spaces/oopere/optipfair-bias-analyzer)** |

#### Tutorials on Google Colab

Explore optipfair’s features with these interactive notebooks.

| Tutorial | Description | Link |
| :--- | :--- | :---: |
| **Depth Pruning** | Learn how to remove entire transformer layers from models like Llama-3. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/peremartra/optipfair/blob/main/examples/depth_pruning.ipynb) |
| **Layer Importance** | Identify which transformer layers contribute the least to your model. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/peremartra/optipfair/blob/main/examples/layer_importance_analysis.ipynb) |
| **Pruning Compatibility** | Check if your model's architecture can be pruned by optipfair. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/peremartra/optipfair/blob/main/examples/pruning_compatibility_check.ipynb) |
| **Bias Compatibility** | The coder's alternative to our live demo for bias analysis. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/peremartra/optipfair/blob/main/examples/bias_compatibility_check.ipynb) |

---
### ✅ Why optipfair?

optipfair is more than just another pruning library. It's a toolkit designed for the modern AI developer who cares about both performance and responsibility.

* **Efficiency & Fairness in One Place**: Stop juggling tools. optipfair is the only library designed to integrate structured pruning with powerful, intuitive bias visualization and analysis.

* **Dual Pruning Strategies**: optipfair supports both **Width Pruning** (removing neurons from MLP layers) and **Depth Pruning** (removing entire transformer layers), giving you flexible control over the efficiency-performance trade-off.

* **Optimized for Modern Architectures**: We focus on what works now. The library is specialized for GLU-based models like LLaMA, Mistral, Gemma, and Qwen, ensuring relevant and effective pruning.

* **Go Beyond Numbers with Bias Visualization**: Don't just get a bias score. Our visualization tools (PCA, heatmaps, mean differences) help you *understand* how and where your model encodes bias, enabling more effective mitigation.

* **🤖 AI-Assisted Development**: Accelerate your workflow using the included [`LLM Reference Manual`](https://github.com/peremartra/optipfair/blob/main/optipfair_llm_reference_manual.txt). Provide it to your favorite LLM (ChatGPT, Claude) to get expert-level help and generate integration code instantly.
<p align="center">
      <img src="images/optipfair_llmflow.gif" alt="AI Pair Programming with optipfair" width="500"/>
</p>

* **🔬 Backed by Research**: Our methods aren't arbitrary. They are built upon and validated by ongoing applied research in model optimization and fairness analysis.

---
### ⚙️ Installation
Choose the installation method that best suits your needs. For bias visualization features, you'll need the [viz] extra.
**Standard Installation**
For core pruning functionality:
```python
pip install optipfair
```

**Full Installation (with Bias Visualization)**
To use the bias analysis and visualization tools, install with the [viz] extra dependencies:
```python
pip install "optipfair[viz]"
```

**Developer Installation**
To install from the source for contributing or development:
```bash
git clone https://github.com/peremartra/optipfair.git
cd optipfair
pip install -e .
```
---
## ⚡ Quick Start

See how to use optipfair's core features in just a few lines of code.

### Pruning with the Python API

Prune 20% of the MLP neurons from a model using the Maximum Absolute Weight (MAW) method.

```python
from transformers import AutoModelForCausalLM
from optipfair import prune_model

# Load a pre-trained model
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")

# Prune 20% of neurons from MLP layers
pruned_model, stats = prune_model(
    model=model,
    pruning_type="MLP_GLU",
    neuron_selection_method="MAW",
    pruning_percentage=20,
    show_progress=True,
    return_stats=True
)

# Print pruning statistics
print(f"Original parameters: {stats['original_parameters']:,}")
print(f"Pruned parameters: {stats['pruned_parameters']:,}")
print(f"Reduction: {stats['reduction']:,} parameters ({stats['percentage_reduction']:.2f}%)")

# Save the pruned model
pruned_model.save_pretrained("./pruned-llama-model")
```
The pruning process yields tangible results in model size and performance. Here's a sample comparison for **Llama-3.2-1B** after pruning 20% of its MLP neurons:

| Metric | Original Model | Pruned Model | Improvement |
| :--- | :---: | :---: | :---: |
| **Total Parameters** | 1.24B | 1.07B | **-13.03%** |
| **Inference Speed** | *Benchmark in progress* | *Benchmark in progress* | *Coming soon* |
| **MMLU Score** | *Benchmark in progress* | *Benchmark in progress* | *Minimal change expected* |

*Results based on the [MAW pruning method](#neuron-selection-methods). Full benchmark results will be published shortly.*

### Pruning Transformer Layers (Depth Pruning)

Remove entire layers from a model for significant efficiency gains. Here, we remove the last 4 layers.

```python
from transformers import AutoModelForCausalLM
from optipfair import prune_model

# Load a pre-trained model
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")

# Remove the last 4 transformer layers
pruned_model, stats = prune_model(
    model=model,
    pruning_type="DEPTH",
    num_layers_to_remove=4,
    layer_selection_method="last", # Recommended for best performance retention
    show_progress=True,
    return_stats=True
)

# Print pruning statistics
print(f"Original layers: {stats['original_layers']}")
print(f"Pruned layers: {stats['pruned_layers']}")
print(f"Reduction: {stats['reduction']:,} parameters ({stats['percentage_reduction']:.2f}%)")

# Save the pruned model
pruned_model.save_pretrained("./pruned-depth-llama-model")
```

### Analyzing Layer Importance
Before performing Depth Pruning, you can analyze which layers are the most redundant. This function measures the cosine similarity between the input and output of each transformer layer—a higher similarity (lower distance score) suggests the layer has less impact on the final representation.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader, TensorDataset
import torch
from optipfair import analyze_layer_importance

# 1. Load a model and tokenizer
model_name = "distilgpt2"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 2. Prepare a sample dataloader (this is the user's responsibility)
dummy_texts = ["This is a sample sentence." for _ in range(16)]
inputs = tokenizer(dummy_texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'])
dataloader = DataLoader(dataset, batch_size=8)

# 3. Analyze layer importance
# The function returns a dictionary {layer_index: importance_score}
# where importance is 1 - cosine_similarity
importance_scores = analyze_layer_importance(model, dataloader)

# 4. Print the results, sorted by importance (less important first)
sorted_layers = sorted(importance_scores.items(), key=lambda item: item[1])
print("Layer importance scores (lower is less important):")
for layer_idx, score in sorted_layers:
    print(f"  Layer {layer_idx}: {score:.4f}")

# The layers with the lowest scores are potential candidates for pruning.
```


### Visualizing Bias with the Python API
Generate visualizations to analyze how a model's activations differ when processing prompts with varying demographic attributes.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from optipfair.bias import visualize_bias

# Load model and tokenizer
model_name = "meta-llama/Llama-3.2-1B"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Define prompt pairs that differ in demographic attributes
prompt_pairs = [
    ("The white man walked through the neighborhood. The police officer thought he", 
     "The Black man walked through the neighborhood. The police officer thought he")
]

# Generate visualizations and get bias metrics
_, metrics = visualize_bias(
    model, 
    tokenizer,
    prompt_pairs=prompt_pairs,
    visualization_types=["mean_diff", "pca"],
    layers="first_middle_last",
    output_dir="./bias_analysis"
)

# Print overall bias metrics
overall = metrics["pair_1"]["metrics"]["overall_metrics"]
print(f"Mean activation difference: {overall['mean_difference']:.6f}")
```
The code above generates the following types of visualizations, allowing for a deep dive into model fairness:
| ![Mean Image Differences](/images/mean_image_differences.png) | ![activation_differences_layer.png](/images/activation_differences_layer.png) | ![pca_analysis.png](/images/pca_analysis.png)
| ---| --- | --- |

---
## Dive Deeper: Features & Concepts

For those who want to understand the nuts and bolts of optipfair.

### Supported Models
optipfair is designed to work with transformer-based language models that use GLU architecture in their MLP layers. This includes most modern LLMs:

* **LLaMA family** (LLaMA, LLaMA-2, LLaMA-3)
* **Mistral** models
* **Gemma** models
* **Qwen** models
* ... and other models with a similar GLU architecture.

### Pruning Strategies: Neurons vs. Layers

optipfair offers two powerful structured pruning strategies:

1.  **MLP Pruning (Width Pruning)**: Reduces the number of neurons within the MLP layers of GLU-based models. This is a fine-grained approach to improve efficiency. You can control it via `pruning_percentage` or a target `expansion_rate`. It uses several neuron importance metrics:
    * **MAW (Maximum Absolute Weight)**: Default and most effective method.
    * **VOW (Variance of Weights)**
    * **PON (Product of Norms)**

2.  **Depth Pruning (Layer Pruning)**: Removes entire transformer layers from the model. This is a more aggressive technique that can yield significant reductions in parameters and latency. It's universally compatible with most transformer architectures. You can define which layers to remove by:
    * **Number**: `num_layers_to_remove=4`
    * **Percentage**: `depth_pruning_percentage=25`
    * **Specific Indices**: `layer_indices=[12, 13, 14, 15]`
  
### Understanding Model Internals: Layer Importance Analysis
Before deciding which layers to remove with Depth Pruning, you can assess their relative importance. optipfair provides a method based on the cosine similarity between a layer's input and output embeddings.

* **How it works**: The analyze_layer_importance function passes data through the model and uses hooks to capture the input and output of each transformer layer. It then calculates a score based on 1 - cosine_similarity.
* **Interpretation**: A low score indicates that a layer alters its input representation minimally. These layers are strong candidates for removal via Depth Pruning, as their impact on the model's overall function may be less critical. This analysis provides data-driven insights to guide your pruning strategy.

---

## 🗺️ Roadmap & Community

The optipfair project is actively developed. Here's what's planned for the future.

### Future Roadmap
Our goal is to make optipfair the go-to toolkit for efficient and fair model optimization. Key upcoming features include:

* **Attention Pruning**: Implementing Attention Bypass and Adaptive Attention Bypass(AAB).
* **Advanced Benchmarks**: Integrating more comprehensive performance and evaluation benchmarks.
* **GPU Optimizations**: Creating a v2.0 with significant GPU-specific optimizations for faster execution. 
* **Large-Scale Model Support**: Adding compatibility for DeepSpeed and FSDP to handle 70B+ models efficiently. 

### 🤝 Contributing
Contributions are welcome! Whether it's bug reports, feature requests, or code contributions, please check out our [contributing guidelines](CONTRIBUTING.md) to get started.

### Citation
If you use optipfair in your research or projects, please cite the library:

```bibtex
@software{optipfair,
  author = {Pere Martra},
  title = {optipfair: A Library for Structured Pruning and Bias Visualization of Large Language Models},
  year = {2024},
  url = {https://github.com/peremartra/optipfair}
}
```
### License
This project is licensed under the Apache 2.0 License. See the LICENSE file for details.
