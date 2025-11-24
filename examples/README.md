# Examples

![Optimize LLMs](/images/optiPfair.png)

> **New to OptiPFair?** Use our [LLM Reference Manual](optipfair_llm_reference_manual.txt) - paste it into ChatGPT or Claude for guided assistance with any OptiPFair task.

## Quick Start

### Check Compatibility (30 seconds)

**Is your model compatible with OptiPFair?**

- [**Pruning Compatibility Check**](pruning_compatibility_check.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/peremartra/optipfair/blob/main/examples/pruning_compatibility_check.ipynb)  
  Quickly verify if your model supports structured pruning. Checks for GLU architecture and calculates expansion ratios.

- [**Bias Analysis Compatibility Check**](bias_compatibility_check.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/peremartra/optipfair/blob/main/examples/bias_compatibility_check.ipynb)  
  Verify if your model supports bias visualization and activation capture.

### Learn by Doing (5 minutes)

**Hands-on tutorials with immediate results**

- [**Width Pruning Tutorial**](basic_pruning_mlp.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/peremartra/optipfair/blob/main/examples/basic_pruning_mlp.ipynb)  
  Interactive width pruning for modern GLU architectures (LLaMA, Qwen, Gemma). See 15-30% size reduction by removing neurons from MLP layers.

- [**Hardware-Optimized Pruning (expansion_divisor)**](expansion_divisor_example.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/peremartra/optipfair/blob/main/examples/expansion_divisor_example.ipynb)  
  Learn how to use the `expansion_divisor` parameter to optimize pruned models for specific hardware. Rounds intermediate layer sizes to multiples of 32, 64, 128, or 256 for better GPU/TPU performance.

- [**Depth Pruning Tutorial**](depth_pruning.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/peremartra/optipfair/blob/main/examples/depth_pruning.ipynb)  
  Learn how to remove entire transformer layers while maintaining model performance. Complementary to width pruning.

- [**Layer Importance Analysis**](layer_importance_analysis.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/peremartra/optipfair/blob/main/examples/layer_importance_analysis.ipynb)  
  Analyze which layers contribute most to model performance. Essential for informed pruning decisions.

### Production Ready

**Complete workflow for real projects**

- [**Complete Pruning Script**](prune_llama.py)  
  Production-ready Python script with benchmarking, text generation testing, and model saving. Everything you need for a full pruning pipeline.


## Installation

```bash
# Basic installation
pip install optipfair

# With visualization dependencies
pip install optipfair[viz]
```

## Need Help?

- **Documentation:** [https://peremartra.github.io/optipfair/](https://peremartra.github.io/optipfair/)
- **LLM Assistant:** Use `optipfair_llm_reference_manual.txt` with any AI assistant
- **GitHub Issues:** [Report problems or ask questions](https://github.com/peremartra/optipfair/issues)
