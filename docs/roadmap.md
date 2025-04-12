# Roadmap

This document outlines the planned features and improvements for OptiPFair.

## Mid-term Goals (0-6 months)

### Version 0.2.0
- **Attention Mechanism Pruning**: Implement pruning techniques for attention layers
- **Transformer Block Pruning**: Implement pruning techniques for entire transformer blocks
- **Bias Visualization**: Implement visualizations for Bias in pair of prompts


### Version 0.3.0
- **Comprehensive Benchmarks**: Add integration with common LLM benchmarks
- **NO GLU Models**: Implement pruning techniques for older models (no GLU)
- **Improved Documentation**: Add more examples and tutorials

### Version 0.4.0

- **Configuration Presets**: Provide optimized pruning configurations for different model families
- **Visualization Tools**: Add tools for visualizing neuron importance and pruning impact

### Version 0.5.0
- **Fairness prunning**: consider bias in pruning. 

## Long-term Goals (6+ months)

### Version 1.0.0
- **Distributed Pruning**: Support for pruning very large models across multiple GPUs
- **Dynamic Pruning**: Techniques for runtime pruning based on inference context
- **Knowledge Distillation**: Integration with knowledge distillation techniques
- **Non-transformer Models**: Extend support to other model architectures
- **Automated Pruning**: Implement algorithms to automatically determine optimal pruning parameters
- **Iterative Pruning**: Support for gradual pruning over multiple iterations
- **Fine-tuning Integration**: Direct integration with fine-tuning workflows

## Community Suggestions

We welcome community input on our roadmap! If you have suggestions for features or improvements, please submit them as issues on our [GitHub repository](https://github.com/yourusername/optipfair/issues) with the label "enhancement".