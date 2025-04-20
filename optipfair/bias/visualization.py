"""
Visualization functions for bias analysis.

This module provides visualization utilities for analyzing bias in transformer models
by comparing activation patterns between pairs of prompts that differ only in 
protected attributes (e.g., race, gender).
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Any
from sklearn.decomposition import PCA
import json
import logging

from .activations import get_activation_pairs, get_layer_names, select_layers
from .metrics import calculate_activation_differences, calculate_bias_metrics
from .defaults import DEFAULT_PROMPT_PAIRS
from .utils import ensure_directory

logger = logging.getLogger(__name__)

def visualize_bias(
    model: Any, 
    tokenizer: Any, 
    prompt_pairs: Optional[List[Tuple[str, str]]] = None,
    visualization_types: List[str] = ["mean_diff", "heatmap", "pca"],
    layers: Union[str, List[int]] = "first_middle_last",
    output_dir: Optional[str] = None,
    figure_format: str = "png",
    show_progress: bool = True,
    **visualization_params
) -> Tuple[None, Dict[str, Any]]:
    """
    Visualize bias in transformer model activations by comparing prompt pairs.
    
    Displays visualizations in the notebook and optionally saves to disk.
    Returns a structured JSON with quantitative metrics.
    
    Args:
        model: A HuggingFace transformer model
        tokenizer: Matching tokenizer for the model
        prompt_pairs: List of (prompt1, prompt2) tuples to compare
                      If None, uses default examples
        visualization_types: Types of visualizations to generate
        layers: Which layers to visualize ("first_middle_last", "all", or list)
        output_dir: Directory to save visualizations (None = display only)
        figure_format: Format for saving figures (png, pdf, svg)
        show_progress: Whether to show progress bars
        **visualization_params: Additional parameters for visualization customization
        
    Returns:
        tuple: (None, metrics_json) - Visualizations are displayed/saved, metrics returned
    """
    # Use default prompt pairs if none provided
    if prompt_pairs is None:
        prompt_pairs = DEFAULT_PROMPT_PAIRS
        logger.info(f"Using {len(prompt_pairs)} default prompt pairs")
    
    # Create output directory if specified
    if output_dir:
        ensure_directory(output_dir)
    
    # Initialize metrics collection
    all_metrics = {}
    
    # Process each prompt pair
    from tqdm import tqdm as tqdm_module
    prompt_iterator = tqdm_module(prompt_pairs) if show_progress else prompt_pairs
    
    for pair_idx, (prompt1, prompt2) in enumerate(prompt_iterator):
        pair_metrics = {}
        
        # Get activations for both prompts
        activations1, activations2 = get_activation_pairs(model, tokenizer, prompt1, prompt2)
        
        # Print activation metrics
        print(f"\nProcessing pair {pair_idx + 1}/{len(prompt_pairs)}:")
        print(f"Prompt 1: '{prompt1}'")
        print(f"Prompt 2: '{prompt2}'")
        
        if not activations1 or not activations2:
            logger.warning(f"Failed to capture activations for pair {pair_idx + 1}")
            continue
        
        # Calculate metrics
        pair_metrics = calculate_bias_metrics(activations1, activations2)
        
        # Generate visualizations
        if "mean_diff" in visualization_types:
            for layer_type in ["mlp_output", "attention_output", "gate_proj", "up_proj"]:
                visualize_mean_differences(
                    model, tokenizer, (prompt1, prompt2), 
                    layer_type=layer_type,
                    layers=layers,
                    output_dir=output_dir,
                    figure_format=figure_format,
                    pair_index=pair_idx,
                    **visualization_params
                )
        
        # Visualization: heatmaps
        if "heatmap" in visualization_types:
            for layer_type in ["mlp_output", "attention_output", "gate_proj", "up_proj"]:
                layer_names = get_layer_names(activations1, layer_type)
                selected_layers = select_layers(layer_names, layers)
                
                for layer_key in selected_layers:
                    if layer_key in activations1 and layer_key in activations2:
                        visualize_heatmap(
                            model, tokenizer, (prompt1, prompt2),
                            layer_key=layer_key,
                            output_dir=output_dir,
                            figure_format=figure_format,
                            pair_index=pair_idx,
                            **visualization_params
                        )
        
        # Visualization: PCA
        if "pca" in visualization_types:
            for layer_type in ["mlp_output", "attention_output"]:
                layer_names = get_layer_names(activations1, layer_type)
                selected_layers = select_layers(layer_names, layers)
                
                for layer_key in selected_layers:
                    if layer_key in activations1 and layer_key in activations2:
                        visualize_pca(
                            model, tokenizer, (prompt1, prompt2),
                            layer_key=layer_key,
                            output_dir=output_dir,
                            figure_format=figure_format,
                            pair_index=pair_idx,
                            **visualization_params
                        )
        
        # Store metrics for this pair
        all_metrics[f"pair_{pair_idx + 1}"] = {
            "prompt1": prompt1,
            "prompt2": prompt2,
            "metrics": pair_metrics
        }
    
    # Save metrics to JSON if output_dir specified
    if output_dir:
        metrics_path = os.path.join(output_dir, "bias_metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(all_metrics, f, indent=4)
        print(f"Saved metrics to {metrics_path}")
    
    return None, all_metrics

def visualize_mean_differences(
    model: Any, 
    tokenizer: Any, 
    prompt_pair: Tuple[str, str], 
    layer_type: str = "mlp_output", 
    layers: Union[str, List[int]] = "first_middle_last",
    output_dir: Optional[str] = None,
    figure_format: str = "png",
    pair_index: int = 0,
    **params
):
    """
    Visualize mean activation differences across layers for a specific component type.
    
    Args:
        model: A HuggingFace transformer model
        tokenizer: Matching tokenizer for the model
        prompt_pair: Tuple of (prompt1, prompt2) to compare
        layer_type: Type of layer to visualize (mlp_output, attention_output, etc.)
        layers: Which layers to include ("first_middle_last", "all", or list of indices)
        output_dir: Directory to save visualizations (None = display only)
        figure_format: Format for saving figures (png, pdf, svg)
        pair_index: Index of the prompt pair (for labeling)
        **params: Additional visualization parameters
    """
    prompt1, prompt2 = prompt_pair
    
    # Get activations
    activations1, activations2 = get_activation_pairs(model, tokenizer, prompt1, prompt2)
    
    if not activations1 or not activations2:
        logger.warning("Failed to capture activations")
        return
    
    # Calculate differences between activations
    differences = calculate_activation_differences(activations1, activations2)
    
    # Filter layers of the specified type
    layer_keys = get_layer_names(differences, layer_type)
    
    if not layer_keys:
        logger.warning(f"No layers of type {layer_type} found")
        return
    
    # Extract layer number from each key
    layer_nums = [int(k.split('_')[-1]) for k in layer_keys]
    
    # Create the visualization
    plt.figure(figsize=(10, 6))
    
    # Calculate values to plot
    values = [differences[k].mean().item() for k in layer_keys]
    
    # Plot
    plt.bar(range(len(values)), values)
    plt.xticks(range(len(values)), layer_nums, rotation=45)
    plt.title(f'Mean Activation Difference by Layer ({layer_type})')
    plt.xlabel('Layer Number')
    plt.ylabel('Mean Absolute Difference')
    
    # Add text about the prompt pair
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.figtext(0.5, 0.01, f'Prompt Pair: "{prompt1}" vs "{prompt2}"',
                ha="center", fontsize=9, wrap=True)
    
    plt.tight_layout()
    
    # Save if output_dir specified
    if output_dir:
        filename = f"mean_diff_{layer_type}_pair{pair_index}.{figure_format}"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved {filepath}")
    
    plt.show()

def visualize_heatmap(
    model: Any, 
    tokenizer: Any, 
    prompt_pair: Tuple[str, str], 
    layer_key: str,
    output_dir: Optional[str] = None,
    figure_format: str = "png",
    pair_index: int = 0,
    **params
):
    """
    Create a heatmap to visualize activation differences in a specific layer.
    
    Args:
        model: A HuggingFace transformer model
        tokenizer: Matching tokenizer for the model
        prompt_pair: Tuple of (prompt1, prompt2) to compare
        layer_key: Key of the layer to visualize
        output_dir: Directory to save visualizations (None = display only)
        figure_format: Format for saving figures (png, pdf, svg)
        pair_index: Index of the prompt pair (for labeling)
        **params: Additional visualization parameters
    """
    prompt1, prompt2 = prompt_pair
    
    # Get activations
    activations1, activations2 = get_activation_pairs(model, tokenizer, prompt1, prompt2)
    
    if not activations1 or not activations2:
        logger.warning("Failed to capture activations")
        return
    
    if layer_key not in activations1 or layer_key not in activations2:
        logger.warning(f"Layer {layer_key} not found in the activations")
        return
    
    # Extract activations and calculate the difference
    activation1 = activations1[layer_key]
    activation2 = activations2[layer_key]
    
    # For tensors with dimension > 2, average over all except the last two
    if activation1.dim() > 2:
        dims_to_mean = tuple(range(0, activation1.dim() - 2))
        activation1 = activation1.mean(dim=dims_to_mean)
        activation2 = activation2.mean(dim=dims_to_mean)
    
    # Ensure compatible shapes
    min_dim0 = min(activation1.shape[0], activation2.shape[0])
    min_dim1 = min(activation1.shape[1], activation2.shape[1])
    activation1 = activation1[:min_dim0, :min_dim1]
    activation2 = activation2[:min_dim0, :min_dim1]
    
    # Calculate the absolute difference
    diff = torch.abs(activation1 - activation2)
    
    # Convert to numpy for visualization
    diff_np = diff.numpy()
    
    # If the matrix is very large, take a representative subset
    max_dims = (20, 20)
    if diff_np.shape[0] > max_dims[0] or diff_np.shape[1] > max_dims[1]:
        # Take evenly spaced elements
        step_0 = max(1, diff_np.shape[0] // max_dims[0])
        step_1 = max(1, diff_np.shape[1] // max_dims[1])
        diff_np = diff_np[::step_0, ::step_1]
    
    # Create the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(diff_np, cmap="viridis", annot=False)
    plt.title(f'Activation Differences - Layer {layer_key}')
    plt.xlabel('Target Dimension')
    plt.ylabel('Source Dimension')
    
    # Add text about the prompt pair
    plt.gcf().subplots_adjust(bottom=0.25)
    plt.figtext(0.5, 0.01, f'Prompt Pair: "{prompt1}" vs "{prompt2}"',
                ha="center", fontsize=9, wrap=True)
    
    plt.tight_layout()
    
    # Save if output_dir specified
    if output_dir:
        # Extract layer type and number for the filename
        layer_parts = layer_key.split('_')
        layer_type = "_".join(layer_parts[:-1])
        layer_num = layer_parts[-1]
        
        filename = f"heatmap_{layer_type}_{layer_num}_pair{pair_index}.{figure_format}"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved {filepath}")
    
    plt.show()

def visualize_pca(
    model: Any, 
    tokenizer: Any, 
    prompt_pair: Tuple[str, str], 
    layer_key: str,
    highlight_diff: bool = True,
    output_dir: Optional[str] = None,
    figure_format: str = "png",
    pair_index: int = 0,
    **params
):
    """
    Perform PCA analysis on activations to visualize patterns.
    
    Args:
        model: A HuggingFace transformer model
        tokenizer: Matching tokenizer for the model
        prompt_pair: Tuple of (prompt1, prompt2) to compare
        layer_key: Key of the layer to visualize
        highlight_diff: Whether to highlight tokens that differ between prompts
        output_dir: Directory to save visualizations (None = display only)
        figure_format: Format for saving figures (png, pdf, svg)
        pair_index: Index of the prompt pair (for labeling)
        **params: Additional visualization parameters
    """
    prompt1, prompt2 = prompt_pair
    
    # Get activations
    activations1, activations2 = get_activation_pairs(model, tokenizer, prompt1, prompt2)
    
    if not activations1 or not activations2:
        logger.warning("Failed to capture activations")
        return
    
    if layer_key not in activations1 or layer_key not in activations2:
        logger.warning(f"Layer {layer_key} not found in the activations")
        return
    
    # Extract activations
    activation1 = activations1[layer_key].squeeze()
    activation2 = activations2[layer_key].squeeze()
    
    # Reshape for PCA if needed
    if activation1.dim() > 2:
        activation1 = activation1.view(activation1.shape[0], -1)
        activation2 = activation2.view(activation2.shape[0], -1)
    
    # Combine activations for joint PCA
    combined = torch.cat([activation1, activation2], dim=0).numpy()
    
    # Apply PCA
    pca = PCA(n_components=2)
    combined_pca = pca.fit_transform(combined)
    
    # Split back into separate results
    n = activation1.shape[0]
    pca1 = combined_pca[:n]
    pca2 = combined_pca[n:]
    
    # Get tokens and clean them
    tokens1 = [t.replace("▁", "").replace("Ġ", "") for t in tokenizer.tokenize(prompt1)]
    tokens2 = [t.replace("▁", "").replace("Ġ", "") for t in tokenizer.tokenize(prompt2)]
    
    # Get the minimum length
    min_len = min(len(pca1), len(pca2), len(tokens1), len(tokens2))
    
    # Create the visualization
    plt.figure(figsize=(12, 10))
    
    # Plot the two sets of points
    plt.scatter(pca1[:min_len, 0], pca1[:min_len, 1], label='Prompt 1', alpha=0.7)
    plt.scatter(pca2[:min_len, 0], pca2[:min_len, 1], label='Prompt 2', alpha=0.7)
    
    # Draw arrows connecting corresponding tokens
    for i in range(min_len):
        plt.arrow(
            pca1[i, 0], pca1[i, 1],
            pca2[i, 0] - pca1[i, 0],
            pca2[i, 1] - pca1[i, 1],
            color='gray', alpha=0.3, width=0.001, head_width=0.01
        )
        
        # Label the points with token text
        if highlight_diff and i < len(tokens1) and i < len(tokens2) and tokens1[i] != tokens2[i]:
            label = f"{tokens2[i]} / {tokens1[i]}"
            color = 'red'
            weight = 'bold'
        else:
            label = tokens1[i] if i < len(tokens1) else ""
            color = 'black'
            weight = 'normal'
        
        # Position the label midway between the points
        plt.text(
            (pca1[i, 0] + pca2[i, 0]) / 2,
            (pca1[i, 1] + pca2[i, 1]) / 2,
            label,
            fontsize=9, color=color, fontweight=weight, ha='center'
        )
    
    # Add title and labels
    plt.title(f'PCA Analysis of Activations - Layer {layer_key}')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} explained var.)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} explained var.)')
    plt.legend()
    
    # Add text about the prompt pair
    plt.gcf().subplots_adjust(bottom=0.25)
    plt.figtext(0.5, 0.01, f'Prompt Pair: "{prompt1}" vs "{prompt2}"',
                ha="center", fontsize=9, wrap=True)
    
    plt.tight_layout()
    
    # Save if output_dir specified
    if output_dir:
        # Extract layer type and number for the filename
        layer_parts = layer_key.split('_')
        layer_type = "_".join(layer_parts[:-1])
        layer_num = layer_parts[-1]
        
        filename = f"pca_{layer_type}_{layer_num}_pair{pair_index}.{figure_format}"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved {filepath}")
    
    plt.show()