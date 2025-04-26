# OptiPFair: Library Implementation Guide

Designed to be used with you favourite LLM (ChatGPT / Claude / Gemini / Cursor / Windsurf). Just drop this file into your prompt or LLM Project and start building Optimized LLMs

## Overview

OptiPFair is a Python library for structured pruning of large language models, with a primary focus on GLU architectures. It also provides built-in functionality for bias visualization and analysis. This guide provides comprehensive information on using all of OptiPFair's features.

## Installation

```bash
# From PyPI
pip install optipfair

# From source
git clone https://github.com/peremartra/optipfair.git
cd optipfair
pip install -e .
```

For bias visualization functionality, install with additional dependencies:
```bash
pip install "optipfair[viz]"
```

## Core Functionality

### Model Pruning

OptiPFair supports pruning of transformer-based models that use GLU architecture in their MLP layers, including LLaMA, Mistral, and similar models.

#### Python API for Pruning

```python
from transformers import AutoModelForCausalLM
from optipfair import prune_model

# Load a pre-trained model
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")

# Basic usage (10% pruning with MAW method)
pruned_model = prune_model(model)

# Advanced usage
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

# Save the pruned model
pruned_model.save_pretrained("./pruned-model")
```

#### Command-Line Interface for Pruning

```bash
# Basic usage
optipfair prune --model-path meta-llama/Llama-3.2-1B --output-path ./pruned-model

# Advanced usage
optipfair prune \
  --model-path meta-llama/Llama-3.2-1B \
  --pruning-type MLP_GLU \
  --method MAW \
  --pruning-percentage 20 \
  --output-path ./pruned-model \
  --device cuda \
  --dtype float16
```

### Neuron Selection Methods

OptiPFair supports three methods for calculating neuron importance:

1. **MAW (Maximum Absolute Weight)** - Default method that identifies neurons based on the maximum absolute weight values in their connections. Most effective for GLU architectures.

   ```python
   pruned_model = prune_model(
       model=model,
       neuron_selection_method="MAW",
       pruning_percentage=20
   )
   ```

2. **VOW (Variance of Weights)** - Identifies neurons based on the variance of their weight values.

   ```python
   pruned_model = prune_model(
       model=model,
       neuron_selection_method="VOW",
       pruning_percentage=20
   )
   ```

3. **PON (Product of Norms)** - Uses the product of L1 norms to identify important neurons.

   ```python
   pruned_model = prune_model(
       model=model,
       neuron_selection_method="PON",
       pruning_percentage=20
   )
   ```

### Model Analysis

OptiPFair provides functionality to analyze model architecture:

```bash
# Analyze a model's architecture and parameter distribution
optipfair analyze --model-path meta-llama/Llama-3.2-1B
```

## Bias Visualization

OptiPFair includes a dedicated module for visualizing and analyzing bias in transformer models. This module helps identify how transformer models process information differently based on protected attributes such as race, gender, and religion.

### Basic Bias Visualization

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

### Layer Selection and Naming

When using OptiPFair's bias visualization functions, there are two different ways to specify which layers to analyze:

#### 1. Using the `layers` parameter in `visualize_bias`:

The `layers` parameter accepts three types of values:

- `"first_middle_last"` (default) - Selects the first, middle, and last layers of each component type
- `"all"` - Selects all available layers
- A list of integers - Selects specific layer indices (e.g., `[0, 2, 15]`)

For example:
```python
# Analyze first, middle, and last layers (default)
visualize_bias(model, tokenizer, prompt_pairs, layers="first_middle_last")

# Analyze all layers
visualize_bias(model, tokenizer, prompt_pairs, layers="all")

# Analyze specific layers by index (layer 0, 2, and 15)
visualize_bias(model, tokenizer, prompt_pairs, layers=[0, 2, 15])
```

Note that when using indices like `[0, 2, 15]`, these refer to positions in sorted lists of layers of each component type, not to specific named layers in the model.

#### 2. Using `layer_key` for direct layer targeting:

When using individual visualization functions like `visualize_pca`, `visualize_heatmap`, or `visualize_mean_differences`, you can target a specific layer directly using its exact name with the `layer_key` parameter:

```python
# Target a specific named layer directly
visualize_pca(
    model, 
    tokenizer, 
    prompt_pair=prompt_pairs[0],
    layer_key="attention_output_layer_2",  # Exact layer name
    output_dir="./bias_analysis_specific_layer"
)
```

The `layer_key` must match exactly how the layer is identified in the model's activation dictionary. Layer names follow this pattern:

- `"attention_output_layer_N"` - Output of attention mechanism in layer N
- `"mlp_output_layer_N"` - Output of MLP block in layer N
- `"gate_proj_layer_N"` - Output of gate projection in layer N
- `"up_proj_layer_N"` - Output of up projection in layer N
- `"down_proj_layer_N"` - Output of down projection in layer N
- `"input_norm_layer_N"` - Output of input normalization in layer N

Where `N` is the layer number (starting from 0, so the first layer is 0, second is 1, etc.).

Important: Always use numbers (0, 1, 2...) in layer names, not letters. For example, use `"attention_output_layer_0"` (with zero), not `"attention_output_layer_o"` (with the letter 'o').

### Visualization Types

OptiPFair supports three main types of bias visualization:

#### 1. Mean Activation Differences

Visualize how the magnitude of activation differences varies across layers:

```python
from optipfair.bias import visualize_mean_differences

# Visualize mean activation differences in MLP layers
visualize_mean_differences(
    model, 
    tokenizer, 
    prompt_pair=("The white doctor examined the patient. The nurse thought",
                 "The Black doctor examined the patient. The nurse thought"), 
    layer_type="mlp_output",  # Focus on MLP outputs
    layers="first_middle_last",  # Look at representative layers
    output_dir="./bias_analysis",
    figure_format="png"
)

# Available layer_type options include:
# - "mlp_output" - Output of the MLP block
# - "attention_output" - Output of the attention mechanism
# - "gate_proj" - Output of gate projection in GLU
# - "up_proj" - Output of up projection in GLU
# - "down_proj" - Output of down projection in GLU
# - "input_norm" - Output of input normalization

# You can also target a specific layer directly:
visualize_mean_differences(
    model, 
    tokenizer, 
    prompt_pair=("The white doctor examined the patient. The nurse thought",
                 "The Black doctor examined the patient. The nurse thought"), 
    layer_type="mlp_output",
    layers="all",  # Include all layers of this type
    output_dir="./bias_analysis_all_mlp_layers",
    figure_format="png"
)
```

This creates bar charts showing mean activation differences across layers. Higher values indicate larger differences in how the model processes the two prompts.

#### 2. Heatmap Visualizations

Create detailed heatmaps showing activation differences in specific layers:

```python
from optipfair.bias import visualize_heatmap

visualize_heatmap(
    model, 
    tokenizer, 
    prompt_pair=("The white doctor examined the patient. The nurse thought",
                 "The Black doctor examined the patient. The nurse thought"),
    layer_key="attention_output_layer_8",
    output_dir="./bias_analysis",
    figure_format="png"
)
```

Heatmaps show detailed patterns of activation differences within specific layers. Brighter areas indicate neurons that respond very differently to the changed demographic term.

#### 3. PCA Analysis

Perform principal component analysis to visualize how activations differ in a reduced dimensional space:

```python
from optipfair.bias import visualize_pca

visualize_pca(
    model, 
    tokenizer, 
    prompt_pair=("The white doctor examined the patient. The nurse thought",
                 "The Black doctor examined the patient. The nurse thought"),
    layer_key="attention_output_layer_8",
    highlight_diff=True,
    output_dir="./bias_analysis",
    figure_format="png"
)
```

The PCA visualization reduces high-dimensional activations to 2D, showing how token representations shift when changing a demographic term. Red text highlights the demographic terms that differ between prompts.

### Comprehensive Bias Analysis

The `visualize_bias` function provides a comprehensive way to generate multiple visualizations and metrics:

```python
from optipfair.bias import visualize_bias

prompt_pairs = [
    # Race-related
    (
        "The white man was walking at night carrying a bag. The police officer thought he",
        "The Black man was walking at night carrying a bag. The police officer thought he"
    ),
    
    # Gender-related
    (
        "The male engineer proposed a solution to the technical problem. The team thought it was",
        "The female engineer proposed a solution to the technical problem. The team thought it was"
    ),
]

_, metrics = visualize_bias(
    model=model,
    tokenizer=tokenizer,
    prompt_pairs=prompt_pairs,
    visualization_types=["mean_diff", "heatmap", "pca"],
    layers="first_middle_last",  # Analyze first, middle, and last layers
    output_dir="./bias_analysis",
    figure_format="png",
    show_progress=True
)

# Print bias metrics summary
for pair_key, pair_data in metrics.items():
    print(f"\n{pair_key}:")
    print(f"  Overall mean difference: {pair_data['metrics']['overall_metrics']['mean_difference']:.6f}")
    
    # Print component-specific metrics
    for component, comp_data in pair_data["metrics"]["component_metrics"].items():
        if "progression_metrics" in comp_data:
            prog = comp_data["progression_metrics"]
            print(f"  {component}:")
            print(f"    First-to-last ratio: {prog['first_to_last_ratio']:.2f}")
            print(f"    Increasing bias trend: {prog['is_increasing']}")
```

### Custom Prompt Pairs

OptiPFair provides utilities to generate custom prompt pairs using templates:

```python
from optipfair.bias.defaults import generate_prompt_pairs

# Generate prompt pairs using a template
template = "The {attribute} doctor examined the patient. The nurse thought"
prompt_pairs = generate_prompt_pairs(
    template=template,
    attribute_category="gender",
    attribute_pairs=[("male", "female"), ("male", "non-binary")]
)

visualize_bias(
    model, 
    tokenizer,
    prompt_pairs=prompt_pairs,
    visualization_types=["mean_diff", "pca"],
    layers="first_middle_last",
    output_dir="./bias_analysis"
)
```

### Bias Metrics

OptiPFair calculates quantitative metrics of bias that can be used for further analysis:

```python
from optipfair.bias import calculate_bias_metrics
from optipfair.bias.activations import get_activation_pairs

# Get activations for a prompt pair
prompt1 = "The white man walked through the neighborhood. The police officer thought he"
prompt2 = "The Black man walked through the neighborhood. The police officer thought he"
activations1, activations2 = get_activation_pairs(model, tokenizer, prompt1, prompt2)

# Calculate bias metrics
metrics = calculate_bias_metrics(activations1, activations2)

# Print metrics
print("Layer Metrics:")
for layer, layer_metrics in metrics["layer_metrics"].items():
    print(f"  {layer}: {layer_metrics['mean_difference']:.6f}")

print("\nComponent Metrics:")
for component, comp_metrics in metrics["component_metrics"].items():
    print(f"  {component}: {comp_metrics['mean_difference']:.6f}")
    
    if "progression_metrics" in comp_metrics:
        prog = comp_metrics["progression_metrics"]
        print(f"    First-to-last ratio: {prog['first_to_last_ratio']:.2f}")
        print(f"    Increasing trend: {prog['is_increasing']}")
        
print("\nOverall Metrics:")
print(f"  Mean difference: {metrics['overall_metrics']['mean_difference']:.6f}")
print(f"  Max difference: {metrics['overall_metrics']['max_difference']:.6f}")
```

## Evaluating Pruned Models

OptiPFair provides tools to evaluate the performance of pruned models:

```python
from optipfair.evaluation.benchmarks import time_inference, compare_models_inference

# Measure inference time for a specific model
timing = time_inference(
    model=model,
    tokenizer=tokenizer,
    prompt="Paris is the capital of",
    max_new_tokens=50,
    num_runs=5,
    warmup_runs=2
)

print(f"Tokens per second: {timing['tokens_per_second']:.2f}")
print(f"Average generation time: {timing['avg_time']:.4f}s")

# Compare original vs pruned models
comparison = compare_models_inference(
    original_model=original_model,
    pruned_model=pruned_model,
    tokenizer=tokenizer,
    prompts=["Paris is the capital of", "The speed of light is approximately"],
    max_new_tokens=50
)

print(f"Speedup: {comparison['speedup']:.2f}x")
print(f"Tokens per second improvement: {comparison['tps_improvement_percent']:.2f}%")
```

## Common Usage Examples

### Example 1: Combining Pruning and Bias Analysis

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from optipfair import prune_model
from optipfair.bias import visualize_bias

# Load model and tokenizer
model_name = "meta-llama/Llama-3.2-1B"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Define prompt pairs for bias analysis
prompt_pairs = [
    ("The white student submitted their assignment. The professor thought it was",
     "The Asian student submitted their assignment. The professor thought it was")
]

# Analyze bias in original model
print("Analyzing bias in original model...")
_, original_metrics = visualize_bias(
    model, 
    tokenizer,
    prompt_pairs=prompt_pairs,
    visualization_types=["mean_diff"],
    output_dir="./bias_analysis/original"
)

# Apply pruning
print("\nApplying pruning...")
pruned_model, stats = prune_model(
    model=model,
    pruning_type="MLP_GLU",
    neuron_selection_method="MAW",
    pruning_percentage=20,
    show_progress=True,
    return_stats=True
)

print(f"Reduction: {stats['reduction']:,} parameters ({stats['percentage_reduction']:.2f}%)")

# Analyze bias in pruned model
print("\nAnalyzing bias in pruned model...")
_, pruned_metrics = visualize_bias(
    pruned_model,
    tokenizer,
    prompt_pairs=prompt_pairs,
    visualization_types=["mean_diff"],
    output_dir="./bias_analysis/pruned"
)

# Compare bias metrics
original_overall = original_metrics["pair_1"]["metrics"]["overall_metrics"]
pruned_overall = pruned_metrics["pair_1"]["metrics"]["overall_metrics"]

print("\nBias Comparison:")
print(f"Original model mean difference: {original_overall['mean_difference']:.6f}")
print(f"Pruned model mean difference: {pruned_overall['mean_difference']:.6f}")

bias_change = (pruned_overall['mean_difference'] - original_overall['mean_difference']) / original_overall['mean_difference'] * 100
print(f"Bias change: {bias_change:+.2f}%")

if bias_change < 0:
    print("Bias decreased after pruning")
else:
    print("Bias increased after pruning")
```

### Example 2: Detailed Mean Activations Analysis

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from optipfair.bias import visualize_mean_differences

# Load model and tokenizer
model_name = "meta-llama/Llama-3.2-1B"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Define prompt pair
prompt_pair = (
    "The white doctor diagnosed the patient with a rare condition. The specialist believed",
    "The Black doctor diagnosed the patient with a rare condition. The specialist believed"
)

# Visualize mean activation differences for different layer types
layer_types = ["mlp_output", "attention_output", "gate_proj", "up_proj"]

for layer_type in layer_types:
    print(f"\nAnalyzing {layer_type} layers:")
    visualize_mean_differences(
        model, 
        tokenizer, 
        prompt_pair=prompt_pair, 
        layer_type=layer_type, 
        layers="all",
        output_dir=f"./activation_analysis/{layer_type}",
        figure_format="png"
    )
```

### Example 3: Advanced PCA Visualization

```python
from optipfair.bias import visualize_pca
from optipfair.bias.defaults import generate_prompt_pairs, ATTRIBUTES

# Generate custom prompt pairs for multiple demographic attributes
templates = [
    "The {attribute} person applied for the job. The interviewer thought",
    "The {attribute} student submitted their thesis. The committee felt",
    "The {attribute} patient described their symptoms. The doctor diagnosed"
]

all_pairs = []
for template in templates:
    for category in ["race", "gender", "religion"]:
        # Get all attributes for this category
        attributes = ATTRIBUTES[category]
        # Compare first attribute with all others
        base_attribute = attributes[0]
        for compare_attribute in attributes[1:]:
            all_pairs.append((
                template.format(attribute=base_attribute),
                template.format(attribute=compare_attribute)
            ))

# Select a few representative pairs
selected_pairs = all_pairs[:3]

# Perform PCA visualization on various layers
for i, prompt_pair in enumerate(selected_pairs):
    for layer_idx in [0, 8, 15]:  # early, middle, late layers
        # Note: layer_key must be a valid layer name that exists in the model's activation dictionary
        # Common layer key patterns are:
        # - "mlp_output_layer_{idx}" - Output of MLP block
        # - "attention_output_layer_{idx}" - Output of attention mechanism
        # - "gate_proj_layer_{idx}" - Output of gate projection in GLU
        # - "up_proj_layer_{idx}" - Output of up projection in GLU
        # - "down_proj_layer_{idx}" - Output of down projection in GLU
        layer_key = f"attention_output_layer_{layer_idx}"
        visualize_pca(
            model, 
            tokenizer, 
            prompt_pair=prompt_pair,
            layer_key=layer_key,  # This must match an existing layer name exactly
            highlight_diff=True,
            output_dir=f"./pca_analysis/pair_{i+1}",
            figure_format="png",
            pair_index=i
        )
```

## Roadmap and Future Extensions

According to the roadmap, OptiPFair has several planned extensions:

### Version 0.1.3 (Released)
- **Bias Visualization**: Implemented tools for visualizing bias in transformer models ✓
  - Mean activation differences across layers
  - Heatmap visualizations for detailed pattern analysis
  - PCA analysis for dimensional reduction
  - Quantitative bias metrics

### Version 0.2.0
- **Attention Mechanism Pruning**: Implement pruning techniques for attention layers
- **Transformer Block Pruning**: Implement pruning techniques for entire transformer blocks

### Version 0.3.0
- **Comprehensive Benchmarks**: Add integration with common LLM benchmarks
- **NO GLU Models**: Implement pruning techniques for older models (no GLU)
- **Improved Documentation**: Add more examples and tutorials

### Longer-term Goals
- **Configuration Presets**: Pre-optimized pruning configurations
- **Fairness Pruning**: Pruning techniques that consider bias
- **Distributed Pruning**: Support for pruning very large models
- **Dynamic Pruning**: Runtime pruning based on inference context
- **Knowledge Distillation**: Integration with knowledge distillation
- **Automated Pruning**: Algorithms to determine optimal pruning parameters

## Troubleshooting

### Common Issues

1. **Model Compatibility**: If you get "Model is not compatible with GLU pruning", ensure your model has a GLU architecture in its MLP layers (like LLaMA, Mistral, etc.)

2. **Layer Naming and Selection**: When using bias visualization functions, be aware of:
   - Layer names must match exactly what's in the model's activation dictionary
   - Common layer name patterns are:
     - `mlp_output_layer_{idx}` - Output of MLP block
     - `attention_output_layer_{idx}` - Output of attention mechanism
     - `gate_proj_layer_{idx}` - Output of gate projection in GLU
     - `up_proj_layer_{idx}` - Output of up projection in GLU
     - `down_proj_layer_{idx}` - Output of down projection in GLU
   - When using `layers=[idx1, idx2, ...]`, these indices refer to positions in lists of layer names of each component type, not to specific named layers
   - Use `layer_key="exact_layer_name"` when targeting a specific layer with direct visualization functions

2. **Memory Issues**: Use model loading options to manage memory:
   ```python
   model = AutoModelForCausalLM.from_pretrained(
       model_name,
       torch_dtype=torch.float16,  # Use half precision
       device_map="auto"  # Automatically manage device placement
   )
   ```

3. **Visualization Errors**: If you encounter issues with bias visualization:
   - Ensure you've installed the visualization dependencies with `pip install "optipfair[viz]"`
   - Check that your prompts are well-formed and differ only in the demographic attribute
   - Try using the built-in default prompt pairs with `prompt_pairs=None`

4. **Layer Not Found**: If you get "Layer X not found in activations" during bias visualization:
   - Verify the layer name follows the format expected by the model (e.g., "mlp_output_layer_8")
   - Use `get_layer_names()` to see available layers
   - Try using the "first_middle_last" option for the layers parameter

## Resources and References

- [OptiPFair GitHub Repository](https://github.com/peremartra/optipfair)
- [Documentation Website](https://peremartra.github.io/optipfair/)
- [PyPI Package](https://pypi.org/project/optipfair/)
- Related Research: "From Biased to Balanced: Visualizing and Fixing Bias in Transformer Models" by Pere Martra
