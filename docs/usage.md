# Usage Guide

## Python API

OptiPFair provides a simple Python API for pruning models.

### Basic Usage

```python
from transformers import AutoModelForCausalLM
from optipfair import prune_model

# Load a pre-trained model
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")

# Prune the model with default settings (10% pruning, MAW method)
pruned_model = prune_model(model=model)

# Save the pruned model
pruned_model.save_pretrained("./pruned-model")
```

### Advanced Usage

```python
# Prune with custom settings
pruned_model, stats = prune_model(
    model=model,
    pruning_type="MLP_GLU",              # Type of pruning to apply
    neuron_selection_method="MAW",       # Method to calculate neuron importance
    pruning_percentage=20,               # Percentage of neurons to prune
    # expansion_rate=140,                # Alternatively, specify target expansion rate
    show_progress=True,                  # Show progress during pruning
    return_stats=True                    # Return pruning statistics
)

# Print pruning statistics
print(f"Original parameters: {stats['original_parameters']:,}")
print(f"Pruned parameters: {stats['pruned_parameters']:,}")
print(f"Reduction: {stats['reduction']:,} parameters ({stats['percentage_reduction']:.2f}%)")
```

## Command-Line Interface

OptiPFair provides a command-line interface for pruning models:

### Basic Usage

```bash
# Prune a model with default settings (10% pruning, MAW method)
optipfair prune --model-path meta-llama/Llama-3.2-1B --output-path ./pruned-model
```

### Advanced Usage

```bash
# Prune with custom settings
optipfair prune \
  --model-path meta-llama/Llama-3.2-1B \
  --pruning-type MLP_GLU \
  --method MAW \
  --pruning-percentage 20 \
  --output-path ./pruned-model \
  --device cuda \
  --dtype float16
```

### Analyzing a Model

```bash
# Analyze a model's architecture and parameter distribution
optipfair analyze --model-path meta-llama/Llama-3.2-1B
```

## Neuron Selection Methods

OptiPFair supports three methods for calculating neuron importance:

### MAW (Maximum Absolute Weight)

The MAW method identifies neurons based on the maximum absolute weight values in their connections. This is typically the most effective method for GLU architectures.

```python
pruned_model = prune_model(
    model=model,
    neuron_selection_method="MAW",
    pruning_percentage=20
)
```

### VOW (Variance of Weights)

The VOW method identifies neurons based on the variance of their weight values. This can be useful for certain specific architectures.

```python
pruned_model = prune_model(
    model=model,
    neuron_selection_method="VOW",
    pruning_percentage=20
)
```

### PON (Product of Norms)

The PON method uses the product of L1 norms to identify important neurons. This is an alternative approach that may be useful in certain contexts.

```python
pruned_model = prune_model(
    model=model,
    neuron_selection_method="PON",
    pruning_percentage=20
)
```

## Pruning Percentage vs Expansion Rate

OptiPFair supports two ways to specify the pruning target:

### Pruning Percentage

Directly specify what percentage of neurons to remove:

```python
pruned_model = prune_model(
    model=model,
    pruning_percentage=20  # Remove 20% of neurons
)
```

### Expansion Rate

Specify the target expansion rate (ratio of intermediate size to hidden size) as a percentage:

```python
pruned_model = prune_model(
    model=model,
    expansion_rate=140  # Target 140% expansion rate
)
```

This approach is often more intuitive when comparing across different model scales.

## Evaluating Pruned Models

After pruning, you can use OptiPFair's evaluation tools to assess the performance of the pruned model:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from optipfair.evaluation.benchmarks import time_inference, compare_models_inference

# Load original and pruned models
original_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
pruned_model = AutoModelForCausalLM.from_pretrained("./pruned-model")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")

# Compare inference speed
comparison = compare_models_inference(
    original_model,
    pruned_model,
    tokenizer,
    prompts=["Paris is the capital of", "The speed of light is approximately"],
    max_new_tokens=50
)

print(f"Speedup: {comparison['speedup']:.2f}x")
print(f"Tokens per second improvement: {comparison['tps_improvement_percent']:.2f}%")
```