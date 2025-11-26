"""
Example: Selective Layer Width Pruning

This example demonstrates how to use the layer_indices parameter for MLP_GLU pruning
to selectively prune neurons only in specific layers while leaving others unchanged.

Use cases:
- Preserve critical first and last layers
- Target specific layer ranges based on importance analysis
- Implement asymmetric pruning strategies
- Experiment with different layer-wise patterns
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from optipfair import prune_model, analyze_layer_importance
from torch.utils.data import DataLoader, TensorDataset


def basic_selective_pruning():
    """Basic example: Prune only middle layers."""
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic Selective Layer Pruning")
    print("="*80)
    
    # Load model
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    print(f"\nLoading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    
    # Get total number of layers
    from optipfair.pruning.utils import get_model_layers
    layers = get_model_layers(model)
    num_layers = len(layers)
    print(f"Model has {num_layers} layers")
    
    # Prune only middle layers (skip first and last)
    middle_layers = list(range(5, num_layers - 5))
    print(f"Pruning layers: {middle_layers}")
    
    pruned_model, stats = prune_model(
        model=model,
        pruning_type="MLP_GLU",
        neuron_selection_method="MAW",
        pruning_percentage=30,
        layer_indices=middle_layers,
        show_progress=True,
        return_stats=True
    )
    
    # Print results
    print(f"\nResults:")
    print(f"  Total layers: {stats.get('total_layers', num_layers)}")
    print(f"  Pruned layers: {stats.get('pruned_layers', len(middle_layers))}")
    print(f"  Parameter reduction: {stats['percentage_reduction']:.2f}%")
    print(f"  Preserved first 5 and last 5 layers at full capacity")


def importance_based_selective_pruning():
    """Advanced example: Use layer importance analysis to guide selective pruning."""
    print("\n" + "="*80)
    print("EXAMPLE 2: Importance-Based Selective Pruning")
    print("="*80)
    
    # Load model and tokenizer
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    print(f"\nLoading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Prepare sample calibration data
    print("\nPreparing calibration data...")
    sample_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming technology.",
        "Natural language processing enables computers to understand text.",
    ] * 5  # Repeat for 15 samples
    
    inputs = tokenizer(
        sample_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    )
    dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'])
    dataloader = DataLoader(dataset, batch_size=4)
    
    # Step 1: Analyze layer importance
    print("\nAnalyzing layer importance...")
    importance_scores = analyze_layer_importance(model, dataloader, show_progress=True)
    
    # Sort layers by importance (least important first)
    sorted_layers = sorted(importance_scores.items(), key=lambda x: x[1])
    
    print("\nTop 5 least important layers:")
    for idx, (layer_idx, score) in enumerate(sorted_layers[:5]):
        print(f"  {idx+1}. Layer {layer_idx}: {score:.6f}")
    
    # Step 2: Prune the 10 least important layers
    least_important_layers = [idx for idx, score in sorted_layers[:10]]
    print(f"\nPruning {len(least_important_layers)} least important layers: {least_important_layers}")
    
    pruned_model, stats = prune_model(
        model=model,
        pruning_type="MLP_GLU",
        neuron_selection_method="MAW",
        pruning_percentage=25,
        layer_indices=least_important_layers,
        show_progress=True,
        return_stats=True
    )
    
    # Print results
    print(f"\nResults:")
    print(f"  Pruned layers: {stats.get('pruned_layers', len(least_important_layers))}")
    print(f"  Parameter reduction: {stats['percentage_reduction']:.2f}%")
    print(f"  Targeted least important layers based on data analysis")


def selective_datadriven_pruning():
    """Example: Combine selective pruning with data-driven importance."""
    print("\n" + "="*80)
    print("EXAMPLE 3: Selective Data-Driven Pruning")
    print("="*80)
    
    # Load model and tokenizer
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    print(f"\nLoading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Prepare calibration data
    print("\nPreparing calibration data...")
    sample_texts = [
        "Artificial intelligence is revolutionizing industries.",
        "Deep learning models require significant computational resources.",
        "Model optimization techniques reduce inference costs.",
    ] * 3
    
    inputs = tokenizer(
        sample_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    )
    dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'])
    dataloader = DataLoader(dataset, batch_size=3)
    
    # Prune specific layers with data-driven importance
    layers_to_prune = [5, 10, 15, 20]
    print(f"Pruning layers {layers_to_prune} using data-driven importance...")
    
    pruned_model, stats = prune_model(
        model=model,
        pruning_type="MLP_GLU",
        neuron_selection_method="MAW",
        pruning_percentage=30,
        dataloader=dataloader,  # Data-driven importance
        layer_indices=layers_to_prune,  # Only these layers
        show_progress=True,
        return_stats=True
    )
    
    # Print results
    print(f"\nResults:")
    print(f"  Used calibration data for importance calculation")
    print(f"  Pruned {len(layers_to_prune)} layers: {layers_to_prune}")
    print(f"  Parameter reduction: {stats['percentage_reduction']:.2f}%")


def selective_with_expansion_rate():
    """Example: Selective pruning with expansion_rate target."""
    print("\n" + "="*80)
    print("EXAMPLE 4: Selective Pruning with Expansion Rate")
    print("="*80)
    
    # Load model
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    print(f"\nLoading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    
    # Prune alternating layers to target expansion rate
    alternating_layers = list(range(0, 22, 2))  # Every other layer
    print(f"Pruning alternating layers: {alternating_layers}")
    print(f"Target: 260% expansion rate")
    
    pruned_model, stats = prune_model(
        model=model,
        pruning_type="MLP_GLU",
        expansion_rate=260,  # Target expansion rate instead of percentage
        layer_indices=alternating_layers,
        show_progress=True,
        return_stats=True
    )
    
    # Print results
    print(f"\nResults:")
    print(f"  Pruned {len(alternating_layers)} alternating layers")
    print(f"  Final expansion rate: {stats.get('expansion_rate', 'N/A')}%")
    print(f"  Parameter reduction: {stats['percentage_reduction']:.2f}%")


def comparison_all_vs_selective():
    """Example: Compare pruning all layers vs selective pruning."""
    print("\n" + "="*80)
    print("EXAMPLE 5: Comparison - All Layers vs Selective")
    print("="*80)
    
    # Load model
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    print(f"\nLoading model: {model_name}")
    
    # Strategy 1: Prune all layers
    print("\nStrategy 1: Pruning ALL layers")
    model1 = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    
    pruned_all, stats_all = prune_model(
        model=model1,
        pruning_type="MLP_GLU",
        pruning_percentage=20,
        layer_indices=None,  # All layers
        show_progress=False,
        return_stats=True
    )
    
    print(f"  Parameter reduction: {stats_all['percentage_reduction']:.2f}%")
    
    # Strategy 2: Prune only middle layers
    print("\nStrategy 2: Pruning MIDDLE layers only")
    model2 = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    
    middle_layers = list(range(5, 17))  # Layers 5-16
    pruned_selective, stats_selective = prune_model(
        model=model2,
        pruning_type="MLP_GLU",
        pruning_percentage=20,
        layer_indices=middle_layers,
        show_progress=False,
        return_stats=True
    )
    
    print(f"  Pruned {len(middle_layers)} layers: {middle_layers}")
    print(f"  Parameter reduction: {stats_selective['percentage_reduction']:.2f}%")
    
    # Comparison
    print("\nComparison:")
    print(f"  All layers reduction: {stats_all['percentage_reduction']:.2f}%")
    print(f"  Selective reduction: {stats_selective['percentage_reduction']:.2f}%")
    print(f"  Selective preserves first/last layers at full capacity")
    print(f"  Selective may retain more model capability")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("SELECTIVE LAYER WIDTH PRUNING EXAMPLES")
    print("="*80)
    print("\nThis script demonstrates various use cases for selective layer pruning.")
    print("Each example can be run independently.\n")
    
    # Run examples
    try:
        basic_selective_pruning()
        importance_based_selective_pruning()
        selective_datadriven_pruning()
        selective_with_expansion_rate()
        comparison_all_vs_selective()
        
        print("\n" + "="*80)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY")
        print("="*80)
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()
