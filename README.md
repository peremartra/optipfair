# OptiPFair

A Python library for structured pruning of large language models, with a focus on GLU architectures.

## Overview

OptiPFair enables efficient pruning of large language models while maintaining their performance. It implements various structured pruning methods, starting with MLP pruning for GLU architectures (as used in models like LLaMA, Mistral, etc.).

Key features:
- GLU architecture-aware pruning that preserves model structure 
- Multiple neuron importance calculation methods
- Support for both pruning percentage and target expansion rate
- Simple Python API and CLI interface
- Progress tracking and detailed statistics

## Installation

```bash
# From PyPI (not yet available)
pip install optipfair

# From source
git clone https://github.com/yourusername/optipfair.git
cd optipfair
pip install -e .
```

## Quick Start

### Python API

```python
from transformers import AutoModelForCausalLM
from optipfair import prune_model

# Load a pre-trained model
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")

# Prune 20% of neurons from MLP layers using MAW method
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

# Save pruned model
pruned_model.save_pretrained("./pruned-llama-model")
```

### Command-Line Interface

```bash
# Prune a model with default settings (10% pruning, MAW method)
optipfair prune --model-path meta-llama/Llama-3.2-1B --output-path ./pruned-model

# Prune with custom settings
optipfair prune \
  --model-path meta-llama/Llama-3.2-1B \
  --pruning-type MLP_GLU \
  --method MAW \
  --pruning-percentage 20 \
  --output-path ./pruned-model

# Use expansion rate instead of pruning percentage
optipfair prune \
  --model-path meta-llama/Llama-3.2-1B \
  --expansion-rate 140 \
  --output-path ./pruned-model

# Analyze a model's architecture
optipfair analyze --model-path meta-llama/Llama-3.2-1B
```

## Neuron Selection Methods

OptiPFair supports three methods for calculating neuron importance:

1. **MAW (Maximum Absolute Weight)** - Default method that identifies influential neurons based on the magnitude of their connections. Typically provides the best pruning results.

2. **VOW (Variance of Weights)** - Identifies neurons based on the variance of their weights. May be useful for specific architectures.

3. **PON (Product of Norms)** - Uses the product of L1 norms to identify important neurons. This method may be applicable in certain contexts.

## Documentation

Complete documentation is available at [https://peremartra.github.io/optipfair/](https://peremartra.github.io/optipfair/).

## Supported Models

At his moment, OptiPFair is designed to work with transformer-based language models that use GLU architecture in their MLP layers, including:

- LLaMA family (LLaMA, LLaMA-2, LLaMA-3, )
- Mistral models, QWeN, Gemma...
- And other models with similar GLU architecture

## Expansion Rate vs Pruning Percentage

OptiPFair supports two ways to specify the pruning target:

1. **Pruning Percentage** - Directly specify what percentage of neurons to remove (e.g., 20%)

2. **Expansion Rate** - Specify the target expansion rate (ratio of intermediate size to hidden size) as a percentage (e.g., 140% instead of the default 400%)

The expansion rate approach is often more intuitive when comparing across different model scales.

## Future Roadmap

- Support for attention layer pruning
- Whole block pruning
- Integrated evaluation benchmarks
- Bias visualizations. 

## Citation

If you use OptiPFair in your research, please cite:

```
@software{optipfair2025,
  author = {Pere Martra},
  title = {OptiPFair: A Library for Structured Pruning of Large Language Models},
  year = {2025},
  url = {https://github.com/yourusername/optipfair}
}
```

## License

Apache 2.0

## Contributing

Contributions are welcome! Please check out our [contributing guidelines](CONTRIBUTING.md) for details.
