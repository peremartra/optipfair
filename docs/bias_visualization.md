# Bias Visualization

This module provides tools for visualizing and analyzing how transformer models process information differently based on protected attributes (e.g., race, gender, religion).

## Overview

The bias visualization module enables detailed analysis of activation patterns in transformer models to identify potential biases. It works by:

1. Comparing activation patterns between pairs of prompts that differ only in demographic terms
2. Visualizing the differences using various techniques (mean differences, heatmaps, PCA)
3. Calculating quantitative metrics of bias that can be used for further analysis

This functionality is particularly valuable when combined with OptiPFair's pruning capabilities, as it allows users to:
- Understand how bias manifests in different components of transformer models
- Evaluate whether pruning techniques might amplify or mitigate biases
- Develop fairness-aware pruning approaches that consider both efficiency and bias impacts

## Key Features

- **Activation Capture**: Extract activations from various model components (attention, MLP, GLU)
- **Visualization Tools**: Generate visualizations that reveal bias patterns
  - Mean activation differences across layers
  - Heatmaps showing activation differences in specific layers
  - PCA analysis to visualize demographic impacts on token representations
- **Quantitative Metrics**: Calculate numeric measures of bias for consistent evaluation
- **Customizable Analysis**: Control which layers and components to analyze
- **Output Flexibility**: Display visualizations interactively or save to disk

## Basic Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from optipfair.bias import visualize_bias

# Load model and tokenizer
model_name = "meta-llama/Llama-3.2-1B"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Define prompt pairs to analyze
prompt_pairs = [
    ("The white man walked through the neighborhood. The police officer thought he", 
     "The Black man walked through the neighborhood. The police officer thought he")
]

# Generate visualizations and metrics
_, metrics = visualize_bias(
    model, 
    tokenizer,
    prompt_pairs=prompt_pairs,
    visualization_types=["mean_diff", "pca"],
    layers="first_middle_last",
    output_dir="./bias_analysis"
)

# Examine the metrics
print(metrics)
```

## API Reference

### Core Functions

#### `visualize_bias`

```python
visualize_bias(
    model, 
    tokenizer, 
    prompt_pairs=None,
    visualization_types=["mean_diff", "heatmap", "pca"],
    layers="first_middle_last",
    output_dir=None,
    figure_format="png",
    show_progress=True,
    **visualization_params
)
```

Main function that generates multiple visualization types and returns metrics.

**Parameters:**
- `model`: A HuggingFace transformer model
- `tokenizer`: Matching tokenizer for the model
- `prompt_pairs`: List of (prompt1, prompt2) tuples to compare. If None, uses default examples
- `visualization_types`: Types of visualizations to generate
- `layers`: Which layers to visualize ("first_middle_last", "all", or list of indices)
- `output_dir`: Directory to save visualizations (None = display only)
- `figure_format`: Format for saving figures (png, pdf, svg)
- `show_progress`: Whether to show progress bars
- `**visualization_params`: Additional parameters for visualization customization

**Returns:**
- Tuple of (None, metrics_json) - Visualizations are displayed/saved, metrics returned as dictionary

#### `visualize_mean_differences`

```python
visualize_mean_differences(
    model, 
    tokenizer, 
    prompt_pair, 
    layer_type="mlp_output", 
    layers="first_middle_last",
    output_dir=None,
    **params
)
```

Creates bar charts showing mean activation differences across layers.

#### `visualize_heatmap`

```python
visualize_heatmap(
    model, 
    tokenizer, 
    prompt_pair, 
    layer_key,
    output_dir=None,
    **params
)
```

Creates heatmaps visualizing activation differences in specific layers.

#### `visualize_pca`

```python
visualize_pca(
    model, 
    tokenizer, 
    prompt_pair, 
    layer_key,
    highlight_diff=True,
    output_dir=None,
    **params
)
```

Performs PCA to visualize how activations differ for identical contexts with different demographic terms.

### Metrics

The metrics returned by `visualize_bias` include:

- **Layer-specific metrics**: Detailed metrics for each individual layer
- **Component metrics**: Aggregated metrics for each component type (attention, MLP, etc.)
- **Overall metrics**: Summary metrics across all activations
- **Progression metrics**: Analysis of how bias changes across model depth

## Understanding the Visualizations

### Mean Differences

![Mean Differences Example](images/mean_diff_example.png)

This visualization shows how the magnitude of activation differences varies across layers. Higher values indicate larger differences in how the model processes the two prompts. Increasing values in deeper layers often indicate bias amplification through the network.

### Heatmaps

![Heatmap Example](images/heatmap_example.png)

Heatmaps show detailed patterns of activation differences within specific layers. Brighter areas indicate neurons that respond very differently to the changed demographic term.

### PCA Analysis

![PCA Example](images/pca_example.png)

The PCA visualization reduces high-dimensional activations to 2D, showing how token representations shift when changing a demographic term. Red text highlights the demographic terms that differ between prompts. Arrows connect corresponding tokens across the two prompts.

## Advanced Usage

### Custom Prompt Pairs

You can generate custom prompt pairs using the templates and attributes in `optipfair.bias.defaults`:

```python
from optipfair.bias.defaults import PROMPT_TEMPLATES, ATTRIBUTES, generate_prompt_pairs

# Generate prompt pairs using a template
template = "The {attribute} doctor examined the patient. The nurse thought"
prompt_pairs = generate_prompt_pairs(
    template=template,
    attribute_category="gender",
    attribute_pairs=[("male", "female"), ("male", "non-binary")]
)
```

### Saving Individual Visualizations

You can save specific visualizations for detailed analysis:

```python
from optipfair.bias import visualize_pca, visualize_heatmap

# Generate PCA for a specific layer
visualize_pca(
    model=model,
    tokenizer=tokenizer,
    prompt_pair=("The white man...", "The Black man..."),
    layer_key="attention_output_layer_8",
    output_dir="./analysis/pca",
    figure_format="pdf"
)

# Generate heatmap for the same layer
visualize_heatmap(
    model=model,
    tokenizer=tokenizer,
    prompt_pair=("The white man...", "The Black man..."),
    layer_key="attention_output_layer_8",
    output_dir="./analysis/heatmaps",
    cmap="plasma"  # Custom colormap
)
```

## Interpreting Results

When analyzing bias visualization results, consider:

1. **Layer progression**: Do differences increase in deeper layers? This suggests the model amplifies biases.
2. **Component comparison**: Do MLP layers show larger differences than attention? This helps identify which components encode more bias.
3. **Token-level patterns**: Does the model change its interpretation of neutral words based on demographic context?
4. **Magnitude**: How large are the activation differences relative to the model's overall activation range?

## Connection with Pruning

The bias visualization module works well with OptiPFair's pruning capabilities:

1. Run bias analysis on the unpruned model
2. Apply different pruning strategies
3. Run the same analysis on pruned models
4. Compare whether pruning reduces or amplifies biases

This workflow can help develop pruning techniques that optimize for both efficiency and fairness.