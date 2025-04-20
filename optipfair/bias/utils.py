"""
Utility functions for bias visualization and analysis.

This module provides helper functions and utilities used across
the bias visualization module.
"""

import os
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

def ensure_directory(directory_path: str) -> None:
    """
    Ensure that a directory exists, creating it if necessary.
    
    Args:
        directory_path: Path to the directory to ensure
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        logger.info(f"Created directory: {directory_path}")

def flatten_dict(nested_dict: Dict[str, Any], prefix: str = '') -> Dict[str, Any]:
    """
    Flatten a nested dictionary into a single-level dictionary.
    
    Args:
        nested_dict: A nested dictionary
        prefix: Prefix for flattened keys
        
    Returns:
        Flattened dictionary
    """
    items = []
    for k, v in nested_dict.items():
        new_key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key).items())
        else:
            items.append((new_key, v))
    return dict(items)

def get_token_differences(tokens1: List[str], tokens2: List[str]) -> List[int]:
    """
    Identify indices where tokens differ between two lists.
    
    Args:
        tokens1: First list of tokens
        tokens2: Second list of tokens
        
    Returns:
        List of indices where tokens differ
    """
    min_len = min(len(tokens1), len(tokens2))
    return [i for i in range(min_len) if tokens1[i] != tokens2[i]]

def clean_token_text(token: str) -> str:
    """
    Clean token text by removing special characters.
    
    Args:
        token: Token to clean
        
    Returns:
        Cleaned token
    """
    # Remove special characters used by various tokenizers
    return token.replace("▁", "").replace("Ġ", "").replace("##", "")

def extract_layer_info(layer_key: str) -> Dict[str, Any]:
    """
    Extract layer type and number from a layer key.
    
    Args:
        layer_key: Layer key string (e.g., "mlp_output_layer_3")
        
    Returns:
        Dictionary with layer type and number
    """
    parts = layer_key.split('_layer_')
    if len(parts) != 2:
        return {"type": "unknown", "number": -1}
    
    try:
        layer_type = parts[0]
        layer_number = int(parts[1])
        return {"type": layer_type, "number": layer_number}
    except (ValueError, IndexError):
        return {"type": "unknown", "number": -1}

def format_metric_value(value: float) -> str:
    """
    Format a metric value for display.
    
    Args:
        value: Numeric value to format
        
    Returns:
        Formatted string
    """
    if value == float('inf') or value == float('-inf'):
        return "inf"
    elif abs(value) < 0.001:
        return f"{value:.2e}"
    else:
        return f"{value:.4f}"