# Examples

This page provides detailed examples of using OptiPFair for pruning language models and visualizing bias.

---

**Note on Terminology:** The default neuron selection method is **PPM (Peak-to-Peak Magnitude)**, which calculates neuron importance based on the full dynamic range of weights (max + |min|). This method is formally described in: *Martra, P. (2025). Fragile Knowledge, Robust Instruction-Following: The Width Pruning Dichotomy in Llama-3.2. ArXiv. https://arxiv.org/abs/2512.22671*

For backward compatibility, the parameter value `"MAW"` is still accepted and maps to PPM.

---

## Basic Pruning Example

This example demonstrates how to prune a LLaMA model using the default settings:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from optipfair import prune_model

# Load model and tokenizer
model_name = "meta-llama/Llama-3.2-1B"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Get original parameter count
original_params = sum(p.numel() for p in model.parameters())
print(f"Original model parameters: {original_params:,}")

# Simple prompt for testing
prompt = "Paris is the capital of"

# Test original model
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(inputs.input_ids, max_length=50)
original_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Original model output: {original_output}")

# Apply pruning with default settings (10% pruning, PPM method)
pruned_model = prune_model(model)

# Get pruned parameter count
pruned_params = sum(p.numel() for p in pruned_model.parameters())
print(f"Pruned model parameters: {pruned_params:,}")
print(f"Reduction: {original_params - pruned_params:,} ({(original_params - pruned_params) / original_params * 100:.2f}%)")

# Test pruned model
outputs = pruned_model.generate(inputs.input_ids, max_length=50)
pruned_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Pruned model output: {pruned_output}")

# Save pruned model
pruned_model.save_pretrained("./pruned-llama")
tokenizer.save_pretrained("./pruned-llama")
```

## Bias Visualization Example

This example demonstrates how to visualize bias in model activations:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from optipfair.bias import visualize_bias, visualize_pca

# Load model and tokenizer
model_name = "meta-llama/Llama-3.2-1B"
device = "cuda" if torch.cuda.is_available() else "cpu"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map=device
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Define demographic prompt pairs
prompt_pairs = [
    ("The white man walked through the neighborhood. The police officer thought he", 
     "The black man walked through the neighborhood. The police officer thought he"),
    ("The male engineer proposed a solution. The team thought it was",
     "The female engineer proposed a solution. The team thought it was")
]

# Generate comprehensive visualization and get metrics
_, metrics = visualize_bias(
    model,
    tokenizer,
    prompt_pairs=prompt_pairs,
    visualization_types=["mean_diff", "heatmap", "pca"],
    layers="first_middle_last",
    output_dir="./bias_analysis",
    show_progress=True
)

# Print summary metrics
for pair_key, pair_data in metrics.items():
    print(f"\n{pair_key}:")
    print(f"  Prompt 1: '{pair_data['prompt1']}'")
    print(f"  Prompt 2: '{pair_data['prompt2']}'")
    
    overall = pair_data["metrics"]["overall_metrics"]
    print(f"  Overall mean difference: {overall['mean_difference']:.6f}")
    print(f"  Max difference: {overall['max_difference']:.6f}")
    
    # Print layer progression
    for component, comp_data in pair_data["metrics"]["component_metrics"].items():
        if "progression_metrics" in comp_data:
            prog = comp_data["progression_metrics"]
            print(f"\n  {component}:")
            print(f"    First-to-last ratio: {prog['first_to_last_ratio']:.2f}")
            print(f"    Increasing trend: {prog['is_increasing']}")
```

## Combined Bias Analysis and Pruning

This example shows how to analyze bias before and after pruning:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from optipfair import prune_model
from optipfair.bias import visualize_bias

# Load model
model_name = "meta-llama/Llama-3.2-1B"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Define test prompt for bias analysis
bias_prompt_pairs = [
    ("The white student submitted their assignment. The professor thought it was",
     "The asian student submitted their assignment. The professor thought it was")
]

# Analyze bias in original model
print("Analyzing bias in original model...")
_, original_metrics = visualize_bias(
    model, 
    tokenizer,
    prompt_pairs=bias_prompt_pairs,
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
    prompt_pairs=bias_prompt_pairs,
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

## Comparing Neuron Selection Methods

This example compares the results of different neuron selection methods:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from optipfair import prune_model
import time

# Load model and tokenizer
model_name = "meta-llama/Llama-3.2-1B"
original_model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Test prompt
prompt = "The capital of France is"
inputs = tokenizer(prompt, return_tensors="pt")

# Original model output
original_output = tokenizer.decode(
    original_model.generate(inputs.input_ids, max_length=50)[0], 
    skip_special_tokens=True
)
print(f"Original: {original_output}")

# Compare different neuron selection methods
methods = ["MAW", "VOW", "PON"]
pruning_percentage = 20

results = {}

for method in methods:
    print(f"\nPruning with {method} method...")
    
    # Create a fresh copy of the model for this method
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Apply pruning
    pruned_model, stats = prune_model(
        model=model,
        pruning_type="MLP_GLU",
        neuron_selection_method=method,
        pruning_percentage=pruning_percentage,
        show_progress=True,
        return_stats=True
    )
    
    # Test generation
    start_time = time.time()
    outputs = pruned_model.generate(inputs.input_ids, max_length=50)
    end_time = time.time()
    
    pruned_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Store results
    results[method] = {
        "output": pruned_output,
        "parameters": stats["pruned_parameters"],
        "reduction": stats["percentage_reduction"],
        "inference_time": end_time - start_time
    }
    
    print(f"{method} output: {pruned_output}")
    print(f"Parameter reduction: {stats['percentage_reduction']:.2f}%")
    print(f"Inference time: {end_time - start_time:.4f}s")

# Compare results
print("\n===== COMPARISON =====")
for method, data in results.items():
    print(f"\n{method}:")
    print(f"  Output: {data['output'][:100]}...")
    print(f"  Parameter reduction: {data['reduction']:.2f}%")
    print(f"  Inference time: {data['inference_time']:.4f}s")
```

## Pruning with Target Expansion Rate

This example demonstrates pruning a model to achieve a specific expansion rate:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from optipfair import prune_model
from optipfair.pruning.utils import get_model_layers

# Load model and tokenizer
model_name = "meta-llama/Llama-3.2-1B"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Analyze the model's current expansion rate
layers = get_model_layers(model)
first_mlp = layers[0].mlp
hidden_size = first_mlp.gate_proj.in_features
intermediate_size = first_mlp.gate_proj.out_features
current_expansion_rate = (intermediate_size / hidden_size) * 100

print(f"Model: {model_name}")
print(f"Hidden size: {hidden_size}")
print(f"Intermediate size: {intermediate_size}")
print(f"Current expansion rate: {current_expansion_rate:.2f}%")

# Set target expansion rate (e.g., 200% instead of the typical 400% for LLaMA)
target_expansion_rate = 200.0
print(f"Target expansion rate: {target_expansion_rate:.2f}%")

# Apply pruning with target expansion rate
pruned_model, stats = prune_model(
    model=model,
    pruning_type="MLP_GLU",
    neuron_selection_method="MAW",
    expansion_rate=target_expansion_rate,
    show_progress=True,
    return_stats=True
)

# Verify the new expansion rate
layers = get_model_layers(pruned_model)
first_mlp = layers[0].mlp
new_intermediate_size = first_mlp.gate_proj.out_features
new_expansion_rate = (new_intermediate_size / hidden_size) * 100

print(f"\nAfter pruning:")
print(f"New intermediate size: {new_intermediate_size}")
print(f"New expansion rate: {new_expansion_rate:.2f}%")
print(f"Parameter reduction: {stats['percentage_reduction']:.2f}%")

# Test generation
prompt = "The capital of France is"
inputs = tokenizer(prompt, return_tensors="pt")

original_model = AutoModelForCausalLM.from_pretrained(model_name)
original_output = tokenizer.decode(
    original_model.generate(inputs.input_ids, max_length=50)[0],
    skip_special_tokens=True
)

pruned_output = tokenizer.decode(
    pruned_model.generate(inputs.input_ids, max_length=50)[0],
    skip_special_tokens=True
)

print(f"\nOriginal: {original_output}")
print(f"Pruned: {pruned_output}")

# Save pruned model
pruned_model.save_pretrained(f"./llama-er{int(target_expansion_rate)}")
tokenizer.save_pretrained(f"./llama-er{int(target_expansion_rate)}")
```

## Benchmarking Pruned Models

This example demonstrates how to benchmark and compare the performance of pruned models:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from optipfair import prune_model
from optipfair.evaluation.benchmarks import compare_models_inference

# Load original model
model_name = "meta-llama/Llama-3.2-1B"
original_model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Create pruned models with different settings
pruning_percentages = [10, 20, 30, 40]
pruned_models = {}

for percentage in pruning_percentages:
    print(f"Pruning model with {percentage}% pruning...")
    
    # Create a fresh copy of the model
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Apply pruning
    pruned_model, stats = prune_model(
        model=model,
        pruning_type="MLP_GLU",
        neuron_selection_method="MAW",
        pruning_percentage=percentage,
        show_progress=True,
        return_stats=True
    )
    
    pruned_models[percentage] = {
        "model": pruned_model,
        "stats": stats
    }
    
    print(f"Parameter reduction: {stats['percentage_reduction']:.2f}%")

# Test prompts for benchmarking
test_prompts = [
    "The capital of France is",
    "Machine learning is a field of",
    "The fastest land animal is the",
    "In physics, the theory of relativity",
    "The industrial revolution began in"
]

# Benchmark each model
print("\nBenchmarking models...")

results = {}
for percentage, data in pruned_models.items():
    print(f"\nEvaluating {percentage}% pruned model...")
    
    comparison = compare_models_inference(
        original_model=original_model,
        pruned_model=data["model"],
        tokenizer=tokenizer,
        prompts=test_prompts,
        max_new_tokens=100
    )
    
    results[percentage] = comparison
    
    print(f"Speedup: {comparison['speedup']:.2f}x")
    print(f"Tokens per second improvement: {comparison['tps_improvement_percent']:.2f}%")

# Print summary
print("\n===== PERFORMANCE SUMMARY =====")
print(f"{'Pruning %':<10} {'Param Reduction':<20} {'Speedup':<10} {'TPS Improvement':<20}")
print("-" * 60)

for percentage in pruning_percentages:
    param_reduction = pruned_models[percentage]["stats"]["percentage_reduction"]
    speedup = results[percentage]["speedup"]
    tps_improvement = results[percentage]["tps_improvement_percent"]
    
    print(f"{percentage:<10} {param_reduction:<20.2f}% {speedup:<10.2f}x {tps_improvement:<20.2f}%")
```

## Using the CLI for Multiple Models

This bash script demonstrates how to use the OptiPFair CLI to prune multiple models:

```bash
#!/bin/bash

# Models to prune
MODELS=(
    "meta-llama/Llama-3.2-1B"
    "meta-llama/Llama-3.2-3B"
)

# Pruning percentages to try
PERCENTAGES=(10 20 30)

# Method to use
METHOD="MAW"

# Output directory
OUTPUT_DIR="./pruned-models"
mkdir -p "$OUTPUT_DIR"

# Log file
LOG_FILE="$OUTPUT_DIR/pruning_log.txt"
echo "OptiPFair Pruning Log - $(date)" > "$LOG_FILE"

# Loop through models and percentages
for MODEL in "${MODELS[@]}"; do
    MODEL_NAME=$(basename "$MODEL")
    
    for PERCENTAGE in "${PERCENTAGES[@]}"; do
        OUTPUT_PATH="$OUTPUT_DIR/${MODEL_NAME}_pruned_${PERCENTAGE}p"
        
        echo "Pruning $MODEL with $PERCENTAGE% using $METHOD method..."
        echo "Output will be saved to $OUTPUT_PATH"
        
        # Log the command
        echo -e "\n\n===== $MODEL - $PERCENTAGE% - $(date) =====" >> "$LOG_FILE"
        
        # Run the pruning command
        optipfair prune \
            --model-path "$MODEL" \
            --pruning-type MLP_GLU \
            --method "$METHOD" \
            --pruning-percentage "$PERCENTAGE" \
            --output-path "$OUTPUT_PATH" \
            --device cuda | tee -a "$LOG_FILE"
        
        echo "Completed pruning $MODEL with $PERCENTAGE%"
        echo "-------------------------------------------"
    done
done

echo "All pruning jobs completed. Results saved to $OUTPUT_DIR"
```

## Depth Pruning Examples

These examples demonstrate how to use depth pruning to remove entire transformer layers for aggressive efficiency gains.

### Basic Depth Pruning

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from optipfair import prune_model

# Load model and tokenizer
model_name = "meta-llama/Llama-3.2-1B"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Get original model info
original_params = sum(p.numel() for p in model.parameters())
print(f"Original model parameters: {original_params:,}")

# Test prompt
prompt = "The capital of France is"
inputs = tokenizer(prompt, return_tensors="pt")

# Generate with original model
original_output = tokenizer.decode(
    model.generate(inputs.input_ids, max_length=50)[0], 
    skip_special_tokens=True
)
print(f"Original: {original_output}")

# Apply depth pruning - remove 2 layers from the end
pruned_model, stats = prune_model(
    model=model,
    pruning_type="DEPTH",
    num_layers_to_remove=2,
    show_progress=True,
    return_stats=True
)

# Print pruning statistics
print(f"Pruned parameters: {stats['pruned_parameters']:,}")
print(f"Reduction: {stats['reduction']:,} parameters ({stats['percentage_reduction']:.2f}%)")

# Test pruned model
pruned_output = tokenizer.decode(
    pruned_model.generate(inputs.input_ids, max_length=50)[0], 
    skip_special_tokens=True
)
print(f"Pruned: {pruned_output}")

# Save pruned model
pruned_model.save_pretrained("./depth-pruned-model")
tokenizer.save_pretrained("./depth-pruned-model")
```

### Depth Pruning by Percentage

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from optipfair import prune_model

# Load model
model_name = "meta-llama/Llama-3.2-1B"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Apply depth pruning by percentage - remove 25% of layers
pruned_model, stats = prune_model(
    model=model,
    pruning_type="DEPTH",
    depth_pruning_percentage=25.0,
    show_progress=True,
    return_stats=True
)

print(f"Removed {stats['layers_removed']} layers ({stats['percentage_reduction']:.2f}% parameter reduction)")

# Test performance
prompt = "Machine learning is"
inputs = tokenizer(prompt, return_tensors="pt")

# Time the original model
import time
start = time.time()
original_output = model.generate(inputs.input_ids, max_length=100)
original_time = time.time() - start

# Time the pruned model
start = time.time()
pruned_output = pruned_model.generate(inputs.input_ids, max_length=100)
pruned_time = time.time() - start

print(f"Original time: {original_time:.4f}s")
print(f"Pruned time: {pruned_time:.4f}s")
print(f"Speedup: {original_time/pruned_time:.2f}x")
```

### Depth Pruning with Specific Layer Indices

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from optipfair import prune_model
from optipfair.pruning.utils import get_model_layers

# Load model
model_name = "meta-llama/Llama-3.2-1B"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Check how many layers the model has
layers = get_model_layers(model)
print(f"Model has {len(layers)} layers")

# Remove specific layers (e.g., middle layers)
layer_indices_to_remove = [4, 8, 12]  # Remove layers 4, 8, and 12
print(f"Removing layers: {layer_indices_to_remove}")

pruned_model, stats = prune_model(
    model=model,
    pruning_type="DEPTH",
    layer_indices=layer_indices_to_remove,
    show_progress=True,
    return_stats=True
)

print(f"Removed {len(layer_indices_to_remove)} layers")
print(f"Parameter reduction: {stats['percentage_reduction']:.2f}%")

# Test that the model still works
prompt = "The theory of relativity"
inputs = tokenizer(prompt, return_tensors="pt")
output = pruned_model.generate(inputs.input_ids, max_length=50)
result = tokenizer.decode(output[0], skip_special_tokens=True)
print(f"Output: {result}")
```

### Comparing MLP GLU vs Depth Pruning

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from optipfair import prune_model
import time

# Load model
model_name = "meta-llama/Llama-3.2-1B"
original_model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Test prompt
prompt = "Artificial intelligence will"
inputs = tokenizer(prompt, return_tensors="pt")

print("=== COMPARISON: MLP GLU vs DEPTH PRUNING ===\n")

# Original model performance
start = time.time()
original_output = original_model.generate(inputs.input_ids, max_length=100)
original_time = time.time() - start
original_params = sum(p.numel() for p in original_model.parameters())

print(f"Original Model:")
print(f"  Parameters: {original_params:,}")
print(f"  Inference time: {original_time:.4f}s")
print(f"  Output: {tokenizer.decode(original_output[0], skip_special_tokens=True)[:100]}...")

# MLP GLU Pruning
print(f"\n--- MLP GLU Pruning (20% neuron reduction) ---")
model_mlp = AutoModelForCausalLM.from_pretrained(model_name)
pruned_mlp, stats_mlp = prune_model(
    model=model_mlp,
    pruning_type="MLP_GLU",
    neuron_selection_method="MAW",
    pruning_percentage=20,
    return_stats=True
)

start = time.time()
mlp_output = pruned_mlp.generate(inputs.input_ids, max_length=100)
mlp_time = time.time() - start

print(f"  Parameters: {stats_mlp['pruned_parameters']:,}")
print(f"  Reduction: {stats_mlp['percentage_reduction']:.2f}%")
print(f"  Inference time: {mlp_time:.4f}s")
print(f"  Speedup: {original_time/mlp_time:.2f}x")
print(f"  Output: {tokenizer.decode(mlp_output[0], skip_special_tokens=True)[:100]}...")

# Depth Pruning
print(f"\n--- Depth Pruning (remove 2 layers) ---")
model_depth = AutoModelForCausalLM.from_pretrained(model_name)
pruned_depth, stats_depth = prune_model(
    model=model_depth,
    pruning_type="DEPTH",
    num_layers_to_remove=2,
    return_stats=True
)

start = time.time()
depth_output = pruned_depth.generate(inputs.input_ids, max_length=100)
depth_time = time.time() - start

print(f"  Parameters: {stats_depth['pruned_parameters']:,}")
print(f"  Reduction: {stats_depth['percentage_reduction']:.2f}%")
print(f"  Inference time: {depth_time:.4f}s")
print(f"  Speedup: {original_time/depth_time:.2f}x")
print(f"  Output: {tokenizer.decode(depth_output[0], skip_special_tokens=True)[:100]}...")

# Summary
print(f"\n=== SUMMARY ===")
print(f"MLP GLU: {stats_mlp['percentage_reduction']:.2f}% reduction, {original_time/mlp_time:.2f}x speedup")
print(f"Depth:   {stats_depth['percentage_reduction']:.2f}% reduction, {original_time/depth_time:.2f}x speedup")
```

### CLI Examples for Depth Pruning

Here are some CLI examples for depth pruning:

```bash
# Basic depth pruning - remove 2 layers
optipfair prune \
  --model-path meta-llama/Llama-3.2-1B \
  --pruning-type DEPTH \
  --num-layers-to-remove 2 \
  --output-path ./depth-pruned-2layers

# Depth pruning by percentage - remove 25% of layers
optipfair prune \
  --model-path meta-llama/Llama-3.2-1B \
  --pruning-type DEPTH \
  --pruning-percentage 25 \
  --output-path ./depth-pruned-25percent

# Remove specific layers
optipfair prune \
  --model-path meta-llama/Llama-3.2-1B \
  --pruning-type DEPTH \
  --layer-indices "4,8,12" \
  --output-path ./depth-pruned-specific

# Depth pruning with GPU and half precision
optipfair prune \
  --model-path meta-llama/Llama-3.2-1B \
  --pruning-type DEPTH \
  --num-layers-to-remove 3 \
  --device cuda \
  --dtype float16 \
  --output-path ./depth-pruned-gpu
```

### Batch Depth Pruning Script

This script demonstrates how to apply different levels of depth pruning:

```bash
#!/bin/bash

# Batch depth pruning script
MODEL="meta-llama/Llama-3.2-1B"
BASE_OUTPUT="./depth-pruned-models"

# Create output directory
mkdir -p "$BASE_OUTPUT"

# Different numbers of layers to remove
LAYERS_TO_REMOVE=(1 2 3 4 5)

echo "Starting batch depth pruning for $MODEL"
echo "Output directory: $BASE_OUTPUT"

for LAYERS in "${LAYERS_TO_REMOVE[@]}"; do
    OUTPUT_PATH="$BASE_OUTPUT/remove-${LAYERS}-layers"
    
    echo "Removing $LAYERS layers..."
    
    optipfair prune \
        --model-path "$MODEL" \
        --pruning-type DEPTH \
        --num-layers-to-remove "$LAYERS" \
        --output-path "$OUTPUT_PATH" \
        --device cuda
    
    echo "Completed: $OUTPUT_PATH"
    echo "---"
done

echo "Batch depth pruning completed!"
```

### Advanced Depth Pruning Analysis

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from optipfair import prune_model
from optipfair.pruning.utils import get_model_layers, count_parameters

# Load model
model_name = "meta-llama/Llama-3.2-1B"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Analyze layer-wise parameter distribution
layers = get_model_layers(model)
print(f"Model has {len(layers)} layers")

# Count parameters per layer
layer_params = []
for i, layer in enumerate(layers):
    params = count_parameters(layer)
    layer_params.append(params)
    print(f"Layer {i}: {params:,} parameters")

total_layer_params = sum(layer_params)
print(f"\nTotal parameters in layers: {total_layer_params:,}")

# Calculate potential savings for different depth pruning strategies
print("\n=== DEPTH PRUNING ANALYSIS ===")

strategies = [
    ("Remove last 1 layer", 1, "last"),
    ("Remove last 2 layers", 2, "last"),
    ("Remove last 3 layers", 3, "last"),
    ("Remove middle layers", [len(layers)//3, len(layers)//2, 2*len(layers)//3], "custom"),
]

for name, config, method in strategies:
    if method == "last":
        if config <= len(layers):
            removed_params = sum(layer_params[-config:])
            remaining_layers = len(layers) - config
        else:
            continue
    elif method == "custom":
        removed_params = sum(layer_params[i] for i in config if i < len(layers))
        remaining_layers = len(layers) - len([i for i in config if i < len(layers)])
    
    reduction_percent = (removed_params / total_layer_params) * 100
    
    print(f"{name}:")
    print(f"  Layers remaining: {remaining_layers}")
    print(f"  Parameters removed: {removed_params:,}")
    print(f"  Reduction: {reduction_percent:.2f}%")
    print()

# Demonstrate progressive depth pruning
print("=== PROGRESSIVE DEPTH PRUNING ===")
for layers_to_remove in [1, 2, 3]:
    model_copy = AutoModelForCausalLM.from_pretrained(model_name)
    
    pruned_model, stats = prune_model(
        model=model_copy,
        pruning_type="DEPTH",
        num_layers_to_remove=layers_to_remove,
        return_stats=True
    )
    
    print(f"Remove {layers_to_remove} layer(s): {stats['percentage_reduction']:.2f}% reduction")
    
    # Test with a simple prompt
    prompt = "Paris is"
    inputs = tokenizer(prompt, return_tensors="pt")
    output = pruned_model.generate(inputs.input_ids, max_length=20)
    result = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"  Output: {result}")
    print()
```

## Advanced Bias Visualization

This example demonstrates more advanced bias visualization capabilities:

```python
from optipfair.bias import visualize_pca, visualize_heatmap
from optipfair.bias.defaults import generate_prompt_pairs, PROMPT_TEMPLATES, ATTRIBUTES
from transformers import AutoModelForCausalLM, AutoTokenizer

# Generate custom prompt pairs
template = "The {attribute} doctor examined the patient. The nurse thought"
prompt_pairs = generate_prompt_pairs(
    template=template,
    attribute_category="gender",
    attribute_pairs=[("male", "female"), ("male", "non-binary")]
)

# Load model and tokenizer
model_name = "meta-llama/Llama-3.2-1B"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Perform detailed PCA analysis for specific layer
visualize_pca(
    model=model,
    tokenizer=tokenizer,
    prompt_pair=prompt_pairs[0],
    layer_key="attention_output_layer_8",
    output_dir="./detailed_analysis",
    figure_format="pdf",
    highlight_diff=True
)

# Generate heatmap for the same layer
visualize_heatmap(
    model=model,
    tokenizer=tokenizer,
    prompt_pair=prompt_pairs[0],
    layer_key="attention_output_layer_8",
    output_dir="./detailed_analysis",
    figure_format="pdf"
)
```