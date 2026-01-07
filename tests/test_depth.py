"""
Tests for the depth pruning module.
"""

import unittest
import torch
from torch import nn
import sys
import os

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from optipfair.pruning.depth import (
    validate_layer_removal_params,
    select_layers_to_remove,
    remove_layers_from_model,
    prune_model_depth,
)


class MockTransformerLayer(nn.Module):
    """Mock transformer layer for testing."""
    
    def __init__(self, hidden_size=768):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=8)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.ReLU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)


class MockConfig:
    """Mock configuration for testing."""
    
    def __init__(self, num_hidden_layers=12):
        self.num_hidden_layers = num_hidden_layers


class MockTransformerModel(nn.Module):
    """Mock transformer model for testing."""
    
    def __init__(self, num_layers=12, hidden_size=768):
        super().__init__()
        self.config = MockConfig(num_layers)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([
            MockTransformerLayer(hidden_size) for _ in range(num_layers)
        ])
        
    def __getattr__(self, name):
        if name == 'config':
            return self.config
        return super().__getattr__(name)


class TestDepthPruning(unittest.TestCase):
    """Test cases for depth pruning functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.num_layers = 12
        self.hidden_size = 768
        self.model = MockTransformerModel(self.num_layers, self.hidden_size)
    
    def test_validate_layer_removal_params_num_layers(self):
        """Test parameter validation with num_layers_to_remove."""
        config = validate_layer_removal_params(
            model=self.model,
            num_layers_to_remove=3,
            layer_selection_method="last"
        )
        
        self.assertEqual(config["total_layers"], self.num_layers)
        self.assertEqual(config["num_layers_to_remove"], 3)
        self.assertEqual(config["layer_selection_method"], "last")
        self.assertIsNone(config["layer_indices"])
    
    def test_validate_layer_removal_params_percentage(self):
        """Test parameter validation with depth_pruning_percentage."""
        config = validate_layer_removal_params(
            model=self.model,
            depth_pruning_percentage=25.0,
            layer_selection_method="last"
        )
        
        self.assertEqual(config["total_layers"], self.num_layers)
        self.assertEqual(config["num_layers_to_remove"], 3)  # 25% of 12 = 3
        self.assertEqual(config["layer_selection_method"], "last")
        self.assertIsNone(config["layer_indices"])
    
    def test_validate_layer_removal_params_custom_indices(self):
        """Test parameter validation with custom layer indices."""
        custom_indices = [2, 5, 8]
        config = validate_layer_removal_params(
            model=self.model,
            layer_indices=custom_indices,
            layer_selection_method="first"  # Should be overridden to "custom"
        )
        
        self.assertEqual(config["total_layers"], self.num_layers)
        self.assertEqual(config["num_layers_to_remove"], 3)
        self.assertEqual(config["layer_selection_method"], "custom")
        self.assertEqual(config["layer_indices"], custom_indices)
    
    def test_validate_layer_removal_params_no_params(self):
        """Test validation fails when no parameters are provided."""
        with self.assertRaises(ValueError) as context:
            validate_layer_removal_params(model=self.model)
        self.assertIn("Must specify one of", str(context.exception))
    
    def test_validate_layer_removal_params_multiple_params(self):
        """Test validation fails when multiple parameters are provided."""
        with self.assertRaises(ValueError) as context:
            validate_layer_removal_params(
                model=self.model,
                num_layers_to_remove=3,
                depth_pruning_percentage=25.0
            )
        self.assertIn("mutually exclusive", str(context.exception))
    
    def test_validate_layer_removal_params_invalid_num_layers(self):
        """Test validation fails with invalid num_layers_to_remove."""
        with self.assertRaises(ValueError):
            validate_layer_removal_params(
                model=self.model,
                num_layers_to_remove=0
            )
        
        with self.assertRaises(ValueError):
            validate_layer_removal_params(
                model=self.model,
                num_layers_to_remove=12  # Cannot remove all layers
            )
    
    def test_validate_layer_removal_params_invalid_percentage(self):
        """Test validation fails with invalid depth_pruning_percentage."""
        with self.assertRaises(ValueError):
            validate_layer_removal_params(
                model=self.model,
                depth_pruning_percentage=0.0
            )
        
        with self.assertRaises(ValueError):
            validate_layer_removal_params(
                model=self.model,
                depth_pruning_percentage=100.0
            )
    
    def test_validate_layer_removal_params_invalid_indices(self):
        """Test validation fails with invalid layer indices."""
        with self.assertRaises(ValueError):
            validate_layer_removal_params(
                model=self.model,
                layer_indices=[]  # Empty list
            )
        
        with self.assertRaises(ValueError):
            validate_layer_removal_params(
                model=self.model,
                layer_indices=[15]  # Index out of range
            )
        
        with self.assertRaises(ValueError):
            validate_layer_removal_params(
                model=self.model,
                layer_indices=[2, 2]  # Duplicates
            )
    
    def test_select_layers_to_remove_last(self):
        """Test layer selection with 'last' method."""
        indices = select_layers_to_remove(
            total_layers=12,
            num_layers_to_remove=3,
            layer_selection_method="last"
        )
        
        self.assertEqual(indices, [9, 10, 11])
    
    def test_select_layers_to_remove_first(self):
        """Test layer selection with 'first' method."""
        indices = select_layers_to_remove(
            total_layers=12,
            num_layers_to_remove=3,
            layer_selection_method="first"
        )
        
        self.assertEqual(indices, [0, 1, 2])
    
    def test_select_layers_to_remove_custom(self):
        """Test layer selection with 'custom' method."""
        custom_indices = [2, 5, 8]
        indices = select_layers_to_remove(
            total_layers=12,
            num_layers_to_remove=3,
            layer_selection_method="custom",
            custom_indices=custom_indices
        )
        
        self.assertEqual(indices, [2, 5, 8])
    
    def test_select_layers_to_remove_custom_no_indices(self):
        """Test layer selection with 'custom' method but no indices provided."""
        with self.assertRaises(ValueError):
            select_layers_to_remove(
                total_layers=12,
                num_layers_to_remove=3,
                layer_selection_method="custom"
            )
    
    def test_select_layers_to_remove_invalid_method(self):
        """Test layer selection with invalid method."""
        with self.assertRaises(ValueError):
            select_layers_to_remove(
                total_layers=12,
                num_layers_to_remove=3,
                layer_selection_method="invalid"
            )
    
    def test_remove_layers_from_model(self):
        """Test actual layer removal from model."""
        original_layer_count = len(self.model.model.layers)
        layers_to_remove = [9, 10, 11]  # Remove last 3 layers
        
        modified_model = remove_layers_from_model(
            model=self.model,
            layer_indices_to_remove=layers_to_remove,
            show_progress=False
        )
        
        # Check that the model is modified in place
        self.assertIs(modified_model, self.model)
        
        # Check that the correct number of layers were removed
        self.assertEqual(len(self.model.model.layers), original_layer_count - 3)
        
        # Check that the config was updated
        self.assertEqual(self.model.config.num_hidden_layers, 9)
    
    def test_prune_model_depth_with_num_layers(self):
        """Test complete depth pruning with num_layers_to_remove."""
        original_layer_count = len(self.model.model.layers)
        
        pruned_model = prune_model_depth(
            model=self.model,
            num_layers_to_remove=3,
            layer_selection_method="last",
            show_progress=False
        )
        
        # Check that the model is modified in place
        self.assertIs(pruned_model, self.model)
        
        # Check that the correct number of layers were removed
        self.assertEqual(len(self.model.model.layers), original_layer_count - 3)
        
        # Check that the config was updated
        self.assertEqual(self.model.config.num_hidden_layers, 9)
    
    def test_prune_model_depth_with_percentage(self):
        """Test complete depth pruning with depth_pruning_percentage."""
        original_layer_count = len(self.model.model.layers)
        
        pruned_model = prune_model_depth(
            model=self.model,
            depth_pruning_percentage=25.0,  # 25% of 12 = 3 layers
            layer_selection_method="first",
            show_progress=False
        )
        
        # Check that the model is modified in place
        self.assertIs(pruned_model, self.model)
        
        # Check that the correct number of layers were removed
        self.assertEqual(len(self.model.model.layers), original_layer_count - 3)
        
        # Check that the config was updated
        self.assertEqual(self.model.config.num_hidden_layers, 9)
    
    def test_prune_model_depth_with_custom_indices(self):
        """Test complete depth pruning with custom layer indices."""
        original_layer_count = len(self.model.model.layers)
        custom_indices = [2, 5, 8]
        
        pruned_model = prune_model_depth(
            model=self.model,
            layer_indices=custom_indices,
            show_progress=False
        )
        
        # Check that the model is modified in place
        self.assertIs(pruned_model, self.model)
        
        # Check that the correct number of layers were removed
        self.assertEqual(len(self.model.model.layers), original_layer_count - 3)
        
        # Check that the config was updated
        self.assertEqual(self.model.config.num_hidden_layers, 9)
    
    def test_prune_model_depth_invalid_params(self):
        """Test that invalid parameters raise appropriate errors."""
        with self.assertRaises(ValueError):
            prune_model_depth(
                model=self.model,
                num_layers_to_remove=12,  # Cannot remove all layers
                show_progress=False
            )
        
        with self.assertRaises(ValueError):
            prune_model_depth(
                model=self.model,
                depth_pruning_percentage=100.0,  # Cannot remove all layers
                show_progress=False
            )
    
    def test_layer_indices_ordering(self):
        """Test that layer indices are handled correctly regardless of order."""
        # Create indices in random order
        custom_indices = [8, 2, 5]
        
        pruned_model = prune_model_depth(
            model=self.model,
            layer_indices=custom_indices,
            show_progress=False
        )
        
        # Should still work correctly
        self.assertEqual(len(self.model.model.layers), 9)
        self.assertEqual(self.model.config.num_hidden_layers, 9)
    
    def test_prune_model_depth_with_return_stats(self):
        """Test depth pruning with return_stats=True - this was failing before."""
        from optipfair import prune_model
        
        # Create a fresh model for this test
        test_model = MockTransformerModel(num_layers=12, hidden_size=768)
        original_layer_count = len(test_model.model.layers)
        
        # This should not raise ZeroDivisionError
        pruned_model, stats = prune_model(
            model=test_model,
            pruning_type="DEPTH",
            num_layers_to_remove=3,
            layer_selection_method="last",
            return_stats=True,
            show_progress=False
        )
        
        # Verify stats structure
        self.assertIsInstance(stats, dict)
        self.assertIn("original_parameters", stats)
        self.assertIn("pruned_parameters", stats)
        self.assertIn("reduction", stats)
        self.assertIn("percentage_reduction", stats)
        self.assertIn("original_layer_count", stats)
        self.assertIn("final_layer_count", stats)
        self.assertIn("layers_removed", stats)
        self.assertIn("layer_reduction_percentage", stats)
        
        # Verify stat values
        self.assertEqual(stats["original_layer_count"], 12)
        self.assertEqual(stats["final_layer_count"], 9)
        self.assertEqual(stats["layers_removed"], 3)
        self.assertEqual(stats["layer_reduction_percentage"], 25.0)
        self.assertGreater(stats["original_parameters"], 0)
        self.assertGreater(stats["pruned_parameters"], 0)
        self.assertLess(stats["pruned_parameters"], stats["original_parameters"])
        self.assertGreater(stats["percentage_reduction"], 0)
    
    def test_prune_model_depth_stats_with_percentage(self):
        """Test depth pruning statistics with depth_pruning_percentage."""
        from optipfair import prune_model
        
        test_model = MockTransformerModel(num_layers=12, hidden_size=768)
        
        pruned_model, stats = prune_model(
            model=test_model,
            pruning_type="DEPTH",
            depth_pruning_percentage=25.0,  # 25% of 12 = 3 layers
            layer_selection_method="last",
            return_stats=True,
            show_progress=False
        )
        
        # Verify stats
        self.assertEqual(stats["original_layer_count"], 12)
        self.assertEqual(stats["final_layer_count"], 9)
        self.assertEqual(stats["layers_removed"], 3)
        self.assertEqual(stats["layer_reduction_percentage"], 25.0)
    
    def test_prune_model_depth_stats_with_custom_indices(self):
        """Test depth pruning statistics with custom layer indices."""
        from optipfair import prune_model
        
        test_model = MockTransformerModel(num_layers=12, hidden_size=768)
        custom_indices = [2, 5, 8]
        
        pruned_model, stats = prune_model(
            model=test_model,
            pruning_type="DEPTH",
            layer_indices=custom_indices,
            return_stats=True,
            show_progress=False
        )
        
        # Verify stats
        self.assertEqual(stats["original_layer_count"], 12)
        self.assertEqual(stats["final_layer_count"], 9)
        self.assertEqual(stats["layers_removed"], 3)
        self.assertEqual(stats["layer_reduction_percentage"], 25.0)
    
    def test_get_depth_pruning_statistics(self):
        """Test the get_depth_pruning_statistics function directly."""
        from optipfair.pruning.utils import get_depth_pruning_statistics, count_parameters, get_model_layers
        
        # Create models
        original_model = MockTransformerModel(num_layers=12, hidden_size=768)
        original_params = count_parameters(original_model)
        original_layer_count = len(get_model_layers(original_model))
        
        # Prune without stats
        pruned_model = prune_model_depth(
            model=original_model,
            num_layers_to_remove=3,
            layer_selection_method="last",
            show_progress=False
        )
        
        layers_removed = original_layer_count - len(get_model_layers(pruned_model))
        
        # Call the stats function directly
        stats = get_depth_pruning_statistics(
            original_params=original_params,
            original_layer_count=original_layer_count,
            pruned_model=pruned_model,
            layers_removed=layers_removed,
        )
        
        # Verify all required keys are present
        required_keys = {
            "original_parameters",
            "pruned_parameters",
            "reduction",
            "percentage_reduction",
            "original_layer_count",
            "final_layer_count",
            "layers_removed",
            "layer_reduction_percentage",
        }
        self.assertEqual(set(stats.keys()), required_keys)
        
        # Verify calculations
        self.assertEqual(stats["original_layer_count"], 12)
        self.assertEqual(stats["final_layer_count"], 9)
        self.assertEqual(stats["layers_removed"], 3)
        self.assertEqual(stats["percentage_reduction"], 
                        (stats["reduction"] / stats["original_parameters"]) * 100)


if __name__ == '__main__':
    unittest.main()