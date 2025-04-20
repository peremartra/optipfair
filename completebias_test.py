"""
Comprehensive test script for OptiPFair bias visualization module.

This script tests all three main visualization types (mean differences, heatmap, and PCA)
with a real model to verify functionality.
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from optipfair.bias import (
    visualize_mean_differences,
    visualize_heatmap,
    visualize_pca,
    visualize_bias
)

# Create output directory for saved visualizations
OUTPUT_DIR = "./visualization_test_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def test_bias_visualizations():
    """Test all bias visualization functions with a real model."""
    print("Loading model and tokenizer...")
    try:
        # Load a small model for testing - use any model you have access to
        model_name = "meta-llama/Llama-3.2-1B"  # Change to any model you have access to
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,  # Use half precision for memory efficiency
            device_map="auto"  # Let the library decide the best device allocation
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Define test prompt pairs
        prompt_pairs = [
            (
                "The white doctor examined the patient. The nurse thought",
                "The Black doctor examined the patient. The nurse thought"
            )
        ]
        prompt_pair = prompt_pairs[0]  # Use the first pair for individual tests
        
        print("\n" + "="*50)
        print("TESTING INDIVIDUAL VISUALIZATION FUNCTIONS")
        print("="*50)
        
        # Test 1: Mean Differences Visualization
        print("\nTesting mean differences visualization...")
        try:
            visualize_mean_differences(
                model,
                tokenizer,
                prompt_pair,
                layer_type="mlp_output",
                layers="first_middle_last",
                output_dir=os.path.join(OUTPUT_DIR, "mean_diff"),
                figure_format="png"
            )
            print("✅ Mean differences visualization successful!")
        except Exception as e:
            print(f"❌ Mean differences visualization failed: {e}")
        
        # Test 2: Heatmap Visualization
        # For heatmap we need a specific layer, so let's get layer 8 or the middle layer
        print("\nTesting heatmap visualization...")
        try:
            visualize_heatmap(
                model,
                tokenizer,
                prompt_pair,
                layer_key="mlp_output_layer_8",  # Using middle layer - adjust if your model has fewer layers
                output_dir=os.path.join(OUTPUT_DIR, "heatmap"),
                figure_format="png"
            )
            print("✅ Heatmap visualization successful!")
        except Exception as e:
            print(f"❌ Heatmap visualization failed: {e}")
            # If the specific layer fails, try with layer 0 which should exist in any model
            print("Retrying with layer 0...")
            try:
                visualize_heatmap(
                    model,
                    tokenizer,
                    prompt_pair,
                    layer_key="mlp_output_layer_0",
                    output_dir=os.path.join(OUTPUT_DIR, "heatmap"),
                    figure_format="png"
                )
                print("✅ Heatmap visualization with layer 0 successful!")
            except Exception as e2:
                print(f"❌ Heatmap visualization with layer 0 also failed: {e2}")
        
        # Test 3: PCA Visualization
        print("\nTesting PCA visualization...")
        try:
            visualize_pca(
                model,
                tokenizer,
                prompt_pair,
                layer_key="attention_output_layer_8",  # Using middle attention layer
                highlight_diff=True,
                output_dir=os.path.join(OUTPUT_DIR, "pca"),
                figure_format="png"
            )
            print("✅ PCA visualization successful!")
        except Exception as e:
            print(f"❌ PCA visualization failed: {e}")
            # If the specific layer fails, try with layer 0
            print("Retrying with layer 0...")
            try:
                visualize_pca(
                    model,
                    tokenizer,
                    prompt_pair,
                    layer_key="attention_output_layer_0",
                    highlight_diff=True,
                    output_dir=os.path.join(OUTPUT_DIR, "pca"),
                    figure_format="png"
                )
                print("✅ PCA visualization with layer 0 successful!")
            except Exception as e2:
                print(f"❌ PCA visualization with layer 0 also failed: {e2}")
        
        # Test 4: Main visualize_bias function (combines all visualization types)
        print("\n" + "="*50)
        print("TESTING MAIN VISUALIZATION FUNCTION")
        print("="*50)
        
        print("\nTesting visualize_bias function...")
        try:
            _, metrics = visualize_bias(
                model,
                tokenizer,
                prompt_pairs=prompt_pairs,
                visualization_types=["mean_diff", "heatmap", "pca"],
                layers="first_middle_last",
                output_dir=os.path.join(OUTPUT_DIR, "combined"),
                figure_format="png",
                show_progress=True
            )
            
            print("✅ visualize_bias function successful!")
            
            # Print some metrics to verify they're being calculated correctly
            if metrics and "pair_1" in metrics:
                overall = metrics["pair_1"]["metrics"]["overall_metrics"]
                print("\nMetrics sample:")
                print(f"  Overall mean difference: {overall['mean_difference']:.6f}")
                print(f"  Max difference: {overall['max_difference']:.6f}")
                
                # Check if we have component metrics
                if "component_metrics" in metrics["pair_1"]["metrics"]:
                    comp_metrics = metrics["pair_1"]["metrics"]["component_metrics"]
                    for comp_name, comp_data in comp_metrics.items():
                        if comp_name in ["mlp_output", "attention_output"]:
                            print(f"  {comp_name} mean difference: {comp_data['mean_difference']:.6f}")
        except Exception as e:
            print(f"❌ visualize_bias function failed: {e}")
        
        print("\nTests completed. Check the output directory for visualization results:")
        print(f"  {os.path.abspath(OUTPUT_DIR)}")
        
    except Exception as e:
        print(f"Failed to load model: {e}")
        print("Please make sure you have access to the specified model and that your environment is set up correctly.")

if __name__ == "__main__":
    test_bias_visualizations()