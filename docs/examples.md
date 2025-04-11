# Examples

This page provides detailed examples of using OptiPFair for pruning language models.

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

# Apply pruning with default settings (10% pruning, MAW method)
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