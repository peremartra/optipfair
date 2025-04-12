# OptiPFair Documentation

Welcome to the OptiPFair documentation. OptiPFair is a Python library for structured pruning of large language models, with a focus on GLU architectures.

## Why Prune Language Models?

Pruning helps to reduce the size and computational requirements of large language models, making them:

- **Faster** for inference
- **More efficient** in terms of memory usage
- **Easier to deploy** on resource-constrained devices

## Key Features

- **GLU Architecture-Aware Pruning**: Maintains the paired nature of gate_proj and up_proj layers
- **Multiple Neuron Selection Methods**: MAW, VOW, and PON for different pruning strategies
- **Flexible Pruning Targets**: Support for both pruning percentage and target expansion rate
- **Simple API and CLI**: Easy to use interfaces for Python and command line
- **Progress Tracking**: Visual progress bars and detailed statistics

## Getting Started

- [Installation](installation.md): How to install OptiPFair
- [Usage](usage.md): Basic usage examples
- [API Reference](api.md): Detailed API documentation
- [Examples](examples.md): In-depth examples and tutorials

## Supported Model Architectures

OptiPFair is designed to work with transformer-based models that use GLU architecture in their MLP components, including:

- LLaMA family (LLaMA, LLaMA-2, LLaMA-3)
- Mistral models
- And other models with similar GLU architectures

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