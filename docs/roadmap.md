# Roadmap

This document outlines the planned features and improvements for OptiPFair.

## Mid-term Goals (0-6 months)

### Version 0.1.3 (Released)
- **Bias Visualization**: Implemented tools for visualizing bias in transformer models ✓
  - Mean activation differences across layers
  - Heatmap visualizations for detailed pattern analysis
  - PCA analysis for dimensional reduction
  - Quantitative bias metrics

## Version 0.1.4 (Released)
- Depth pruning (Remove entire layer blocks) implementation. 

### Version 0.2.0 (Released - October 2025) ✅
- **Data-Driven Width Pruning**: Hybrid importance calculation using activation statistics
- **CFSP Integration**: Implementation based on research paper methodology
- **Extended API**: Optional dataloader parameter for calibration
- **Comprehensive Documentation**: Full guides and examples for data-driven pruning

### Version 0.3.0
- **Attention Mechanism Pruning**: Implement pruning techniques for attention layers
- **Comprehensive Benchmarks**: Add integration with common LLM benchmarks
- **NO GLU Models**: Implement pruning techniques for older models (no GLU)
- **Improved Documentation**: Add more examples and tutorials

## Long-term Goals (6+ months)

### Version 0.4.0 (Released - April 2026) ✅

- **Knowledge Distillation**: `opf.distill_model()` available — recover student quality after pruning with teacher guidance (closes #21)
- **Width Pruning Fix**: Model size can no longer increase when combining `expansion_divisor` with small pruning percentages (closes #27)
- **Depth Pruning Config Sync**: `prune_model_depth()` now correctly syncs `config.layer_types` for hybrid architectures like Qwen3.5 (closes #20)

### Version 0.4.1 (Released - May 2026) ✅

- **Status Promotion**: Project promoted from Alpha to **Beta** — core API is stable and production-ready
- **Distillation Padding Fix**: Padding tokens no longer contribute to the distillation loss; labels are masked with `-100` using `attention_mask` (closes #34)
- **Bias API Consistency**: Clarified that `analyze_neuron_bias` does not accept a `batch_size` parameter — prompt pairs are processed individually (closes #33)
- **New Activation Target `down_proj_input`**: Captures activations in the expanded MLP space (`[B, S, intermediate_size]`) before the down projection, enabling finer-grained bias analysis (closes #35)

### Version 0.5.0
- **Fairness prunning**: consider bias in pruning. 

### Version 1.0.0
- **Distributed Pruning**: Support for pruning very large models across multiple GPUs
- **Dynamic Pruning**: Techniques for runtime pruning based on inference context
- **Non-transformer Models**: Extend support to other model architectures
- **Automated Pruning**: Implement algorithms to automatically determine optimal pruning parameters
- **Iterative Pruning**: Support for gradual pruning over multiple iterations
- **Fine-tuning Integration**: Direct integration with fine-tuning workflows

## Community Suggestions

We welcome community input on our roadmap! If you have suggestions for features or improvements, please submit them as issues on our [GitHub repository](https://github.com/yourusername/optipfair/issues) with the label "enhancement".