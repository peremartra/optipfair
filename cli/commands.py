"""
Command-line interface for OptiPFair.

This module provides the CLI commands for pruning models and related operations.
"""

import click
import logging
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from .. import prune_model
from ..pruning.utils import count_parameters

logger = logging.getLogger(__name__)

@click.group()
def cli():
    """OptiPFair: A library for structured pruning of large language models."""
    pass

@cli.command()
@click.option('--model-path', required=True, help='Path or identifier of the model to prune.')
@click.option('--pruning-type', default='MLP_GLU', type=click.Choice(['MLP_GLU']), 
              help='Type of pruning to apply.')
@click.option('--method', default='MAW', type=click.Choice(['MAW', 'VOW', 'PON']), 
              help='Method to calculate neuron importance.')
@click.option('--pruning-percentage', default=None, type=float, 
              help='Percentage of neurons to prune (0-100).')
@click.option('--expansion-rate', default=None, type=float, 
              help='Target expansion rate in percentage (mutually exclusive with pruning-percentage).')
@click.option('--output-path', required=True, help='Path to save the pruned model.')
@click.option('--device', default='auto', help='Device to use for computation (auto, cpu, cuda, cuda:0, etc.)')
@click.option('--dtype', default='auto', type=click.Choice(['auto', 'float32', 'float16', 'bfloat16']), 
              help='Data type to load the model with.')
@click.option('--verbose/--quiet', default=True, help='Whether to show verbose output.')
def prune(model_path, pruning_type, method, pruning_percentage, expansion_rate, 
          output_path, device, dtype, verbose):
    """Prune a language model using the specified parameters."""
    # Configure logging based on verbosity
    log_level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(level=log_level, format='%(message)s')
    
    # Validate inputs
    if pruning_percentage is not None and expansion_rate is not None:
        raise click.UsageError("--pruning-percentage and --expansion-rate are mutually exclusive.")
    
    if pruning_percentage is None and expansion_rate is None:
        pruning_percentage = 10
        logger.info(f"No pruning target specified, defaulting to {pruning_percentage}% pruning.")
    
    # Determine device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Determine dtype
    if dtype == 'auto':
        if 'cuda' in device and torch.cuda.is_available():
            if torch.cuda.get_device_capability()[0] >= 8:  # Ampere or newer
                dtype = torch.bfloat16
            else:
                dtype = torch.float16
        else:
            dtype = torch.float32
    else:
        dtype_map = {
            'float32': torch.float32,
            'float16': torch.float16,
            'bfloat16': torch.bfloat16,
        }
        dtype = dtype_map[dtype]
    
    logger.info(f"Loading model from {model_path} to {device} with {dtype}")
    
    # Load model and tokenizer
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map=device
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise
    
    # Log original model parameters
    original_params = count_parameters(model)
    logger.info(f"Original model parameters: {original_params:,}")
    
    # Apply pruning
    logger.info(f"Pruning model with {pruning_type} pruning, {method} neuron selection method")
    if pruning_percentage is not None:
        logger.info(f"Target: {pruning_percentage}% reduction in neurons")
    else:
        logger.info(f"Target: {expansion_rate}% expansion rate")
    
    pruned_model, stats = prune_model(
        model=model,
        pruning_type=pruning_type,
        neuron_selection_method=method,
        pruning_percentage=pruning_percentage,
        expansion_rate=expansion_rate,
        show_progress=verbose,
        return_stats=True,
    )
    
    # Log pruning statistics
    logger.info("Pruning complete!")
    logger.info(f"Original parameters: {stats['original_parameters']:,}")
    logger.info(f"Pruned parameters: {stats['pruned_parameters']:,}")
    logger.info(f"Reduction: {stats['reduction']:,} parameters ({stats['percentage_reduction']:.2f}%)")
    
    if stats['expansion_rate'] is not None:
        logger.info(f"Final expansion rate: {stats['expansion_rate']:.2f}%")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Save model and tokenizer
    logger.info(f"Saving pruned model to {output_path}")
    pruned_model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    
    logger.info("Done!")

@cli.command()
@click.option('--model-path', required=True, help='Path or identifier of the model to analyze.')
@click.option('--device', default='auto', help='Device to use for computation (auto, cpu, cuda, cuda:0, etc.)')
def analyze(model_path, device):
    """Analyze a model's architecture and parameter distribution."""
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    # Determine device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    logger.info(f"Loading model from {model_path} to {device}")
    
    # Load model
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device
        )
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise
    
    # Log basic model information
    total_params = count_parameters(model)
    logger.info(f"Model: {model_path}")
    logger.info(f"Total parameters: {total_params:,}")
    
    # Try to identify model architecture
    from ..pruning.utils import get_model_layers, validate_model_for_glu_pruning
    
    layers = get_model_layers(model)
    logger.info(f"Number of layers: {len(layers)}")
    
    # Check if model is compatible with GLU pruning
    is_glu_compatible = validate_model_for_glu_pruning(model)
    logger.info(f"Compatible with GLU pruning: {is_glu_compatible}")
    
    if is_glu_compatible and layers:
        # Get information about the first layer
        first_layer = layers[0]
        mlp = first_layer.mlp
        
        hidden_size = mlp.gate_proj.in_features
        intermediate_size = mlp.gate_proj.out_features
        expansion_ratio = intermediate_size / hidden_size
        
        logger.info(f"Hidden size: {hidden_size}")
        logger.info(f"Intermediate size: {intermediate_size}")
        logger.info(f"Expansion ratio: {expansion_ratio:.2f}x ({expansion_ratio*100:.2f}%)")
        
        # Parameter distribution
        attn_params = sum(p.numel() for name, p in first_layer.named_parameters() if 'self_attn' in name)
        mlp_params = sum(p.numel() for name, p in first_layer.named_parameters() if 'mlp' in name)
        other_params = sum(p.numel() for name, p in first_layer.named_parameters() 
                           if 'self_attn' not in name and 'mlp' not in name)
        
        # Extrapolate to whole model
        total_layer_params = attn_params + mlp_params + other_params
        attn_percentage = (attn_params / total_layer_params) * 100
        mlp_percentage = (mlp_params / total_layer_params) * 100
        other_percentage = (other_params / total_layer_params) * 100
        
        logger.info("\nParameter distribution per layer:")
        logger.info(f"Attention: {attn_params:,} ({attn_percentage:.2f}%)")
        logger.info(f"MLP: {mlp_params:,} ({mlp_percentage:.2f}%)")
        logger.info(f"Other: {other_params:,} ({other_percentage:.2f}%)")
        
        # Estimate potential parameter savings
        for prune_percent in [10, 20, 30, 40, 50]:
            new_intermediate_size = intermediate_size * (1 - prune_percent/100)
            new_expansion_ratio = new_intermediate_size / hidden_size
            param_reduction = (intermediate_size - new_intermediate_size) * (hidden_size + mlp.down_proj.out_features)
            model_reduction_percent = (param_reduction * len(layers)) / total_params * 100
            
            logger.info(f"\nWith {prune_percent}% MLP neuron pruning:")
            logger.info(f"  New expansion ratio: {new_expansion_ratio:.2f}x ({new_expansion_ratio*100:.2f}%)")
            logger.info(f"  Estimated parameter reduction: {model_reduction_percent:.2f}% of model")

if __name__ == '__main__':
    cli()