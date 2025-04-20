"""
Tests for the bias visualization module.
"""

import unittest
import torch
import torch.nn as nn
import numpy as np
import sys
import os
from unittest.mock import MagicMock, patch

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from optipfair.bias.activations import (
    register_hooks,
    remove_hooks,
    process_prompt,
    get_activation_pairs,
    get_layer_names,
    select_layers,
)
from optipfair.bias.metrics import (
    calculate_activation_differences,
    calculate_bias_metrics,
)
from optipfair.bias.utils import (
    ensure_directory,
    flatten_dict,
    get_token_differences,
    clean_token_text,
    extract_layer_info,
    format_metric_value,
)

class MockLinear(nn.Linear):
    """Mock Linear layer for testing."""
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)

class MockAttention(nn.Module):
    """Mock attention module that returns a tuple."""
    def __init__(self):
        super().__init__()
        self.q_proj = MockLinear(128, 128)
        self.k_proj = MockLinear(128, 128)
        self.v_proj = MockLinear(128, 128)
        self.o_proj = MockLinear(128, 128)
        
    def forward(self, x):
        return x, None  # Return tuple like real attention

class MockMLP(nn.Module):
    """Mock MLP module with GLU components."""
    def __init__(self):
        super().__init__()
        self.gate_proj = MockLinear(128, 256)
        self.up_proj = MockLinear(128, 256)
        self.down_proj = MockLinear(256, 128)
        
    def forward(self, x):
        return self.down_proj(self.gate_proj(x) * self.up_proj(x))

class MockLayer(nn.Module):
    """Mock transformer layer."""
    def __init__(self):
        super().__init__()
        self.self_attn = MockAttention()
        self.mlp = MockMLP()
        self.input_layernorm = nn.LayerNorm(128)
        
    def forward(self, x):
        attn_out, _ = self.self_attn(x)
        x = x + attn_out
        x = x + self.mlp(x)
        return x

class MockModel(nn.Module):
    """Mock transformer model for testing."""
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([MockLayer() for _ in range(4)])
        self.device = torch.device("cpu")
        
    def forward(self, **kwargs):
        x = torch.randn(1, 10, 128)  # batch_size=1, seq_len=10, hidden_size=128
        for layer in self.model.layers:
            x = layer(x)
        return x

class TestBiasActivations(unittest.TestCase):
    """Test cases for activation capture functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model = MockModel()
        self.tokenizer = MagicMock()
        self.tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
        self.tokenizer.return_value.__getitem__ = lambda x, y: torch.tensor([[1, 2, 3]])
        self.tokenizer.return_value.to = lambda device: self.tokenizer.return_value
        
    def test_register_hooks(self):
        """Test hook registration."""
        handles = register_hooks(self.model)
        
        # Should have multiple hooks
        self.assertGreater(len(handles), 0)
        
        # Clean up
        remove_hooks(handles)
        
    def test_process_prompt(self):
        """Test processing prompt and capturing activations."""
        activations = process_prompt(self.model, self.tokenizer, "test prompt")
        
        # Should capture activations from different components
        self.assertGreater(len(activations), 0)
        
        # Check that we captured from different components
        component_types = set()
        for key in activations.keys():
            parts = key.split('_layer_')
            if len(parts) > 1:
                component_types.add(parts[0])
        
        self.assertIn("attention_output", component_types)
        self.assertIn("mlp_output", component_types)
        
    def test_get_activation_pairs(self):
        """Test getting activation pairs for two prompts."""
        act1, act2 = get_activation_pairs(self.model, self.tokenizer, "prompt1", "prompt2")
        
        # Both should have activations
        self.assertGreater(len(act1), 0)
        self.assertGreater(len(act2), 0)
        
        # Should have same keys
        self.assertEqual(set(act1.keys()), set(act2.keys()))
        
    def test_get_layer_names(self):
        """Test extracting and sorting layer names."""
        activations = {
            "mlp_output_layer_0": torch.randn(1, 10, 128),
            "mlp_output_layer_1": torch.randn(1, 10, 128),
            "attention_output_layer_0": torch.randn(1, 10, 128),
            "attention_output_layer_1": torch.randn(1, 10, 128),
        }
        
        # Test filtering by layer type
        mlp_layers = get_layer_names(activations, "mlp_output")
        self.assertEqual(len(mlp_layers), 2)
        self.assertEqual(mlp_layers[0], "mlp_output_layer_0")
        self.assertEqual(mlp_layers[1], "mlp_output_layer_1")
        
        # Test getting all layers
        all_layers = get_layer_names(activations)
        self.assertEqual(len(all_layers), 4)
        
    def test_select_layers(self):
        """Test layer selection strategies."""
        layer_names = [
            "mlp_output_layer_0",
            "mlp_output_layer_1",
            "mlp_output_layer_2",
            "mlp_output_layer_3",
        ]
        
        # Test first_middle_last
        selected = select_layers(layer_names, "first_middle_last")
        self.assertEqual(len(selected), 3)
        self.assertEqual(selected[0], "mlp_output_layer_0")
        self.assertEqual(selected[1], "mlp_output_layer_2")  # Middle
        self.assertEqual(selected[2], "mlp_output_layer_3")
        
        # Test all
        selected = select_layers(layer_names, "all")
        self.assertEqual(len(selected), 4)
        
        # Test specific indices
        selected = select_layers(layer_names, [0, 3])
        self.assertEqual(len(selected), 2)
        self.assertEqual(selected[0], "mlp_output_layer_0")
        self.assertEqual(selected[1], "mlp_output_layer_3")

class TestBiasMetrics(unittest.TestCase):
    """Test cases for metrics calculation."""
    
    def test_calculate_activation_differences(self):
        """Test calculation of activation differences."""
        act1 = {
            "mlp_output_layer_0": torch.randn(1, 10, 128),
            "attention_output_layer_0": torch.randn(1, 10, 128),
        }
        act2 = {
            "mlp_output_layer_0": torch.randn(1, 10, 128),
            "attention_output_layer_0": torch.randn(1, 10, 128),
        }
        
        differences = calculate_activation_differences(act1, act2)
        
        # Should have differences for all keys
        self.assertEqual(len(differences), 2)
        self.assertIn("mlp_output_layer_0", differences)
        self.assertIn("attention_output_layer_0", differences)
        
        # Differences should be positive
        for diff in differences.values():
            self.assertTrue(torch.all(diff >= 0))
            
    def test_calculate_bias_metrics(self):
        """Test bias metrics calculation."""
        # Create activations with known differences
        act1 = {
            "mlp_output_layer_0": torch.zeros(1, 10, 128),
            "mlp_output_layer_1": torch.zeros(1, 10, 128),
            "attention_output_layer_0": torch.zeros(1, 10, 128),
        }
        act2 = {
            "mlp_output_layer_0": torch.ones(1, 10, 128),
            "mlp_output_layer_1": torch.ones(1, 10, 128) * 0.5,
            "attention_output_layer_0": torch.ones(1, 10, 128) * 0.2,
        }
        
        metrics = calculate_bias_metrics(act1, act2)
        
        # Should have all metric types
        self.assertIn("layer_metrics", metrics)
        self.assertIn("overall_metrics", metrics)
        self.assertIn("component_metrics", metrics)
        
        # Check layer-specific metrics
        self.assertEqual(len(metrics["layer_metrics"]), 3)
        self.assertIn("mean_difference", metrics["layer_metrics"]["mlp_output_layer_0"])
        
        # Check component metrics
        self.assertIn("mlp_output", metrics["component_metrics"])
        self.assertIn("attention_output", metrics["component_metrics"])
        
        # Check progression metrics for MLP
        mlp_metrics = metrics["component_metrics"]["mlp_output"]
        self.assertIn("progression_metrics", mlp_metrics)
        
class TestBiasUtils(unittest.TestCase):
    """Test cases for utility functions."""
    
    def test_ensure_directory(self):
        """Test directory creation function."""
        with patch('os.makedirs') as mock_makedirs:
            ensure_directory("/path/to/test")
            mock_makedirs.assert_called_once_with("/path/to/test")
    
    def test_flatten_dict(self):
        """Test dictionary flattening."""
        nested = {
            "a": 1,
            "b": {
                "c": 2,
                "d": {
                    "e": 3
                }
            }
        }
        
        flat = flatten_dict(nested)
        
        self.assertEqual(flat["a"], 1)
        self.assertEqual(flat["b.c"], 2)
        self.assertEqual(flat["b.d.e"], 3)
        
    def test_get_token_differences(self):
        """Test finding token differences."""
        tokens1 = ["the", "white", "man", "walked"]
        tokens2 = ["the", "black", "man", "walked"]
        
        diff_indices = get_token_differences(tokens1, tokens2)
        
        self.assertEqual(len(diff_indices), 1)
        self.assertEqual(diff_indices[0], 1)  # Index of "white"/"black"
        
    def test_clean_token_text(self):
        """Test token cleaning."""
        self.assertEqual(clean_token_text("▁hello"), "hello")
        self.assertEqual(clean_token_text("Ġworld"), "world")
        self.assertEqual(clean_token_text("##suffix"), "suffix")
        
    def test_extract_layer_info(self):
        """Test layer info extraction."""
        info = extract_layer_info("mlp_output_layer_5")
        self.assertEqual(info["type"], "mlp_output")
        self.assertEqual(info["number"], 5)
        
        # Test invalid format
        info = extract_layer_info("invalid_key")
        self.assertEqual(info["type"], "unknown")
        self.assertEqual(info["number"], -1)
        
    def test_format_metric_value(self):
        """Test metric value formatting."""
        self.assertEqual(format_metric_value(0.1234), "0.1234")
        self.assertEqual(format_metric_value(0.0001), "1.00e-04")
        self.assertEqual(format_metric_value(float('inf')), "inf")

if __name__ == '__main__':
    unittest.main()