# 🚀 OptiPFair v0.2.2 - Selective Layer Width Pruning

We're excited to announce **OptiPFair v0.2.2**, bringing powerful new capabilities for fine-grained control over model pruning!

## 🎯 Headline Features

### 1️⃣ Selective Layer Width Pruning

The `layer_indices` parameter now works for **both DEPTH and MLP_GLU pruning**, giving you unprecedented control over which layers to optimize:

```python
from optipfair import prune_model

# Prune neurons ONLY in specific layers (preserve first & last)
pruned_model = prune_model(
    model=model,
    pruning_type="MLP_GLU",
    pruning_percentage=30,
    layer_indices=[5, 10, 15, 20],  # Only these layers are pruned
    show_progress=True
)
```

**Key Benefits:**
- 🛡️ **Preserve Critical Layers**: Keep embedding and output layers at full capacity
- 🎯 **Targeted Optimization**: Prune only the layers that matter
- 🔬 **Data-Driven Selection**: Combine with layer importance analysis
- ⚡ **Full Feature Support**: Works with expansion_rate, expansion_divisor, dataloader, all methods

### 2️⃣ Optimized Hybrid Importance Calculation

We've streamlined the data-driven pruning algorithm for better performance:

- **Simplified gate_proj & up_proj**: Now use the same fast MAW method as static pruning
- **Focused Complexity**: Activation-weighted calculation only where it matters (down_proj)
- **Faster Execution**: Reduced computational overhead while maintaining effectiveness
- **Consistent Methodology**: Same MAW formula across static and hybrid approaches

## 📊 What's New

### Extended API
- ✅ `layer_indices` parameter now contextual: removes layers for DEPTH, prunes neurons for MLP_GLU
- ✅ Comprehensive validation: checks for valid indices, duplicates, empty lists, type errors
- ✅ Enhanced statistics: reports `pruned_layers` and `total_layers` for selective pruning

### Improved Performance
- ⚡ Faster hybrid importance calculation
- 💾 Selective hook registration (only on specified layers)
- 🎯 More efficient calibration with layer_indices

### Better Documentation
- 📖 Complete "Selective Layer Width Pruning" guide in README
- 📝 Extended reference manual with 4+ detailed examples
- 💻 New example file with 5 practical use cases
- 🧪 12 comprehensive test cases

## 💡 Common Use Cases

### Use Case 1: Preserve Embedding Layers
```python
# Prune all middle layers, preserve first and last 5
num_layers = len(model.model.layers)
middle_layers = list(range(5, num_layers - 5))

pruned_model = prune_model(
    model=model,
    pruning_type="MLP_GLU",
    pruning_percentage=25,
    layer_indices=middle_layers
)
```

### Use Case 2: Importance-Based Pruning
```python
from optipfair import analyze_layer_importance

# Step 1: Analyze which layers are least important
importance_scores = analyze_layer_importance(model, dataloader)
sorted_layers = sorted(importance_scores.items(), key=lambda x: x[1])
least_important = [idx for idx, score in sorted_layers[:10]]

# Step 2: Prune only those layers
pruned_model = prune_model(
    model=model,
    pruning_type="MLP_GLU",
    pruning_percentage=30,
    layer_indices=least_important
)
```

### Use Case 3: Data-Driven Selective Pruning
```python
# Combine calibration data with selective pruning
pruned_model = prune_model(
    model=model,
    pruning_type="MLP_GLU",
    neuron_selection_method="MAW",
    pruning_percentage=20,
    dataloader=calibration_dataloader,  # Hybrid importance
    layer_indices=[5, 10, 15, 20],      # Only these layers
    show_progress=True
)
```

## 🔧 Technical Highlights

### Modified Core Functions
- `prune_model()`: Now passes layer_indices to MLP_GLU pruning
- `prune_model_mlp_glu()`: Full selective pruning implementation with validation
- `setup_mlp_hooks_for_importance()`: Selective hook registration
- `compute_neuron_pair_importance_maw_hybrid()`: Simplified and optimized
- `get_pruning_statistics()`: Detects and reports selective pruning

### Enhanced CLI
```bash
# CLI now supports layer_indices for both pruning types
optipfair prune \
  --model-path meta-llama/Llama-3.2-1B \
  --pruning-type MLP_GLU \
  --pruning-percentage 30 \
  --layer-indices "5,10,15,20" \
  --output-path ./pruned-model
```

## 🧪 Testing & Validation

- ✅ 12 comprehensive test cases in `tests/test_selective_layer_pruning.py`
- ✅ Tested with all neuron selection methods (MAW, VOW, PON)
- ✅ Verified compatibility with expansion_rate, expansion_divisor, dataloader
- ✅ Validated error handling and edge cases
- ✅ Confirmed backward compatibility with v0.2.1

## 📦 Installation

```bash
pip install --upgrade optipfair
```

Or with visualization support:
```bash
pip install --upgrade "optipfair[viz]"
```

## 📚 Resources

- **Documentation**: [https://peremartra.github.io/optipfair/](https://peremartra.github.io/optipfair/)
- **GitHub**: [https://github.com/peremartra/optipfair](https://github.com/peremartra/optipfair)
- **Examples**: Check out `examples/selective_layer_width_pruning.py`
- **Tests**: See `tests/test_selective_layer_pruning.py`

## 🙏 Acknowledgments

Thank you to our community for the feedback and suggestions that made this release possible!

## 📝 Full Changelog

See [CHANGELOG.md](https://github.com/peremartra/optipfair/blob/main/CHANGELOG.md) for detailed changes.

---

**Upgrade today and take control of your model optimization!** 🚀

Questions or issues? Open an issue on [GitHub](https://github.com/peremartra/optipfair/issues).
