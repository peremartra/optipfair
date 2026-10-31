"""
Example script demonstrating pruning with expansion rate targets.

This script prunes a LLaMA model to different expansion rates and
evaluates the performance and quality impacts.
"""

import torch
import time
import argparse
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from optipfair import prune_model
from optipfair.pruning.utils import get_model_layers
from optipfair.evaluation.benchmarks import time_inference

# Default configuration
DEFAULT_MODEL = "meta-llama/Llama-3.2-1B"
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_EXPANSION_RATES = [140, 160, 180, 200, 250, 300, 350]
DEFAULT_OUTPUT_DIR = "./pruned-models-er"

# Test prompts for evaluation
TEST_PROMPTS = [
    "Paris is the capital of",
    "The human brain consists of",
    "Machine learning algorithms can be categorized as",
    "The theory of relativity states that",
    "In computer science, a binary tree is"
]

def create_model(model_name, device):
    """Create a model and tokenizer."""
    print(f"Loading model {model_name} to {device}...")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if "cuda" in device else torch.float32,
        device_map=device
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def get_model_info(model):
    """Get information about a model's expansion rate."""
    layers = get_model_layers(model)
    if not layers:
        raise ValueError("Could not find layers in the model")
    
    first_mlp = layers[0].mlp
    hidden_size = first_mlp.gate_proj.in_features
    intermediate_size = first_mlp.gate_proj.out_features
    expansion_rate = (intermediate_size / hidden_size) * 100
    
    return {
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "expansion_rate": expansion_rate,
        "num_layers": len(layers)
    }

def evaluate_model(model, tokenizer, prompts, max_new_tokens=50):
    """Evaluate a model on a set of prompts."""
    results = []
    
    for prompt in prompts:
        # Time inference
        timing = time_inference(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            num_runs=3,
            warmup_runs=1
        )
        
        # Generate text
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False
        )
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        results.append({
            "prompt": prompt,
            "generated_text": generated_text,
            "tokens_per_second": timing["tokens_per_second"],
            "avg_time": timing["avg_time"]
        })
    
    # Calculate average performance
    avg_tps = sum(r["tokens_per_second"] for r in results) / len(results)
    avg_time = sum(r["avg_time"] for r in results) / len(results)
    
    return {
        "results": results,
        "avg_tokens_per_second": avg_tps,
        "avg_time": avg_time
    }

def plot_results(results_data, output_path):
    """Create plots of the pruning results."""
    # Convert results to DataFrame
    df = pd.DataFrame(results_data)
    
    # Create output directory if it doesn't exist
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot 1: Parameter Reduction vs Expansion Rate
    plt.figure(figsize=(10, 6))
    plt.plot(df["expansion_rate"], df["parameter_reduction"], marker="o")
    plt.title("Parameter Reduction vs Expansion Rate")
    plt.xlabel("Expansion Rate (%)")
    plt.ylabel("Parameter Reduction (%)")
    plt.grid(True)
    plt.savefig(output_dir / "param_reduction_vs_er.png", dpi=300)
    
    # Plot 2: Performance vs Expansion Rate
    plt.figure(figsize=(10, 6))
    plt.plot(df["expansion_rate"], df["tokens_per_second"], marker="o", label="Tokens/sec")
    plt.title("Inference Performance vs Expansion Rate")
    plt.xlabel("Expansion Rate (%)")
    plt.ylabel("Tokens per Second")
    plt.grid(True)
    plt.savefig(output_dir / "performance_vs_er.png", dpi=300)
    
    # Plot 3: Parameter Reduction vs Performance
    plt.figure(figsize=(10, 6))
    plt.scatter(df["parameter_reduction"], df["tokens_per_second"], s=80)
    
    # Add expansion rate as text labels
    for i, txt in enumerate(df["expansion_rate"]):
        plt.annotate(f"{txt}%", 
                     (df["parameter_reduction"].iloc[i], df["tokens_per_second"].iloc[i]),
                     xytext=(5, 5), textcoords="offset points")
    
    plt.title("Performance vs Parameter Reduction")
    plt.xlabel("Parameter Reduction (%)")
    plt.ylabel("Tokens per Second")
    plt.grid(True)
    plt.savefig(output_dir / "performance_vs_reduction.png", dpi=300)
    
    # Export data to CSV
    df.to_csv(output_dir / "results.csv", index=False)
    
    print(f"Plots and data saved to {output_dir}")

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Prune a model to different expansion rates")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model name or path")
    parser.add_argument("--device", default=DEFAULT_DEVICE, help="Device to use")
    parser.add_argument("--expansion-rates", nargs="+", type=float, default=DEFAULT_EXPANSION_RATES,
                        help="List of expansion rates to test")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument("--save-models", action="store_true", help="Save pruned models")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load original model
    original_model, tokenizer = create_model(args.model, args.device)
    
    # Get original model info
    original_info = get_model_info(original_model)
    original_params = sum(p.numel() for p in original_model.parameters())
    
    print(f"Original model:")
    print(f"  Hidden size: {original_info['hidden_size']}")
    print(f"  Intermediate size: {original_info['intermediate_size']}")
    print(f"  Expansion rate: {original_info['expansion_rate']:.2f}%")
    print(f"  Parameters: {original_params:,}")
    
    # Evaluate original model
    print("\nEvaluating original model...")
    original_eval = evaluate_model(original_model, tokenizer, TEST_PROMPTS)
    print(f"Original tokens per second: {original_eval['avg_tokens_per_second']:.2f}")
    
    # Free up memory
    del original_model
    torch.cuda.empty_cache()
    
    # Results storage
    results = []
    
    # Process each expansion rate
    for expansion_rate in args.expansion_rates:
        print(f"\n\n{'='*50}")
        print(f"Processing expansion rate: {expansion_rate}%")
        print(f"{'='*50}")
        
        # Load a fresh model
        model, _ = create_model(args.model, args.device)
        
        # Apply pruning
        print(f"Pruning to {expansion_rate}% expansion rate...")
        pruned_model, stats = prune_model(
            model=model,
            pruning_type="MLP_GLU",
            neuron_selection_method="MAW",
            expansion_rate=expansion_rate,
            show_progress=True,
            return_stats=True
        )
        
        # Get pruned model info
        pruned_info = get_model_info(pruned_model)
        pruned_params = sum(p.numel() for p in pruned_model.parameters())
        
        print(f"\nPruned model:")
        print(f"  New intermediate size: {pruned_info['intermediate_size']}")
        print(f"  New expansion rate: {pruned_info['expansion_rate']:.2f}%")
        print(f"  Parameters: {pruned_params:,}")
        print(f"  Parameter reduction: {stats['percentage_reduction']:.2f}%")
        
        # Evaluate pruned model
        print("\nEvaluating pruned model...")
        pruned_eval = evaluate_model(pruned_model, tokenizer, TEST_PROMPTS)
        print(f"Pruned tokens per second: {pruned_eval['avg_tokens_per_second']:.2f}")
        
        # Calculate speedup
        speedup = pruned_eval['avg_tokens_per_second'] / original_eval['avg_tokens_per_second']
        print(f"Speedup: {speedup:.2f}x")
        
        # Store results
        results.append({
            "expansion_rate": expansion_rate,
            "original_intermediate_size": original_info['intermediate_size'],
            "pruned_intermediate_size": pruned_info['intermediate_size'],
            "parameter_reduction": stats['percentage_reduction'],
            "tokens_per_second": pruned_eval['avg_tokens_per_second'],
            "original_tokens_per_second": original_eval['avg_tokens_per_second'],
            "speedup": speedup
        })
        
        # Save pruned model if requested
        if args.save_models:
            model_dir = os.path.join(args.output_dir, f"er{int(expansion_rate)}")
            print(f"\nSaving model to {model_dir}...")
            pruned_model.save_pretrained(model_dir)
            tokenizer.save_pretrained(model_dir)
        
        # Free up memory
        del model
        del pruned_model
        torch.cuda.empty_cache()
    
    # Create plots
    print("\nGenerating result plots...")
    plot_results(results, args.output_dir)
    
    # Print summary
    print("\n===== SUMMARY =====")
    print(f"{'Expansion Rate':<15} {'Param Reduction':<20} {'Speedup':<10} {'Tokens/sec':<15}")
    print("-" * 60)
    
    for result in sorted(results, key=lambda x: x["expansion_rate"]):
        print(f"{result['expansion_rate']:<15.2f}% {result['parameter_reduction']:<20.2f}% "
              f"{result['speedup']:<10.2f}x {result['tokens_per_second']:<15.2f}")

if __name__ == "__main__":
    import os
    import sys
    
    # Add parent directory to path for local development
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    
    main()