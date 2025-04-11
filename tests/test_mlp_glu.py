"""
Tests for the MLPGLUPruning module.
"""

import unittest
import torch
from torch import nn
import sys
import os

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from optipfair.pruning.mlp_glu import (
    compute_neuron_pair_importance_maw,
    compute_neuron_pair_importance_vow,
    compute_neuron_pair_importance_pon,
    prune_neuron_pairs,
    calculate_pruning_percentage_from_expansion_rate,
)

class MockMLP(nn.Module):
    """Mock MLP module for testing."""
    
    def __init__(self, hidden_size=768, intermediate_size=3072):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = nn.SiLU()
        
        # Initialize with normal distribution for testing
        nn.init.normal_(self.gate_proj.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.up_proj.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.down_proj.weight, mean=0.0, std=0.02)

class TestMLPGLUPruning(unittest.TestCase):
    """Test cases for MLP GLU pruning functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.hidden_size = 768
        self.intermediate_size = 3072
        self.mlp = MockMLP(self.hidden_size, self.intermediate_size)
        
        # Test data
        self.gate_weight = self.mlp.gate_proj.weight.data.float()
        self.up_weight = self.mlp.up_proj.weight.data.float()
    
    def test_compute_neuron_pair_importance_maw(self):
        """Test MAW importance calculation."""
        importance = compute_neuron_pair_importance_maw(self.gate_weight, self.up_weight)
        
        # Check shape
        self.assertEqual(importance.shape[0], self.intermediate_size)
        
        # Check that all values are positive
        self.assertTrue(torch.all(importance >= 0))
        
        # Check that it doesn't return all zeros
        self.assertFalse(torch.all(importance == 0))
    
    def test_compute_neuron_pair_importance_vow(self):
        """Test VOW importance calculation."""
        importance = compute_neuron_pair_importance_vow(self.gate_weight, self.up_weight)
        
        # Check shape
        self.assertEqual(importance.shape[0], self.intermediate_size)
        
        # Check that all values are positive or zero (variance is non-negative)
        self.assertTrue(torch.all(importance >= 0))
        
        # Check that it doesn't return all zeros
        self.assertFalse(torch.all(importance == 0))
    
    def test_compute_neuron_pair_importance_pon(self):
        """Test PON importance calculation."""
        importance = compute_neuron_pair_importance_pon(self.gate_weight, self.up_weight)
        
        # Check shape
        self.assertEqual(importance.shape[0], self.intermediate_size)
        
        # Check that all values are positive or zero (L1 norm is non-negative)
        self.assertTrue(torch.all(importance >= 0))
        
        # Check that it doesn't return all zeros
        self.assertFalse(torch.all(importance == 0))
    
    def test_prune_neuron_pairs(self):
        """Test neuron pair pruning function."""
        prune_percentage = 20.0
        
        # Prune the MLP
        new_gate_proj, new_up_proj, new_down_proj, new_size = prune_neuron_pairs(
            self.mlp, prune_percentage
        )
        
        # Check the new size
        expected_size = int(self.intermediate_size * (1 - prune_percentage/100))
        self.assertEqual(new_size, expected_size)
        
        # Check dimensions of new layers
        self.assertEqual(new_gate_proj.in_features, self.hidden_size)
        self.assertEqual(new_gate_proj.out_features, expected_size)
        
        self.assertEqual(new_up_proj.in_features, self.hidden_size)
        self.assertEqual(new_up_proj.out_features, expected_size)
        
        self.assertEqual(new_down_proj.in_features, expected_size)
        self.assertEqual(new_down_proj.out_features, self.hidden_size)
    
    def test_prune_neuron_pairs_zero_percent(self):
        """Test pruning with 0% should keep all neurons."""
        prune_percentage = 0.0
        
        # Prune the MLP
        new_gate_proj, new_up_proj, new_down_proj, new_size = prune_neuron_pairs(
            self.mlp, prune_percentage
        )
        
        # Check the new size equals the original size
        self.assertEqual(new_size, self.intermediate_size)
    
    def test_prune_neuron_pairs_invalid_percentage(self):
        """Test pruning with invalid percentage should raise an error."""
        prune_percentage = 100.0  # Cannot prune all neurons
        
        with self.assertRaises(ValueError):
            prune_neuron_pairs(self.mlp, prune_percentage)
    
    def test_calculate_pruning_percentage_from_expansion_rate(self):
        """Test conversion from expansion rate to pruning percentage."""
        current_intermediate_size = 3072
        current_hidden_size = 768
        
        # Current expansion rate is 400%
        current_expansion_rate = (current_intermediate_size / current_hidden_size) * 100
        self.assertEqual(current_expansion_rate, 400.0)
        
        # Test with target 200% (half the current rate)
        target_expansion_rate = 200.0
        pruning_percentage = calculate_pruning_percentage_from_expansion_rate(
            current_intermediate_size, current_hidden_size, target_expansion_rate
        )
        self.assertEqual(pruning_percentage, 50.0)  # Should prune half the neurons
        
        # Test with target 300% (75% of current)
        target_expansion_rate = 300.0
        pruning_percentage = calculate_pruning_percentage_from_expansion_rate(
            current_intermediate_size, current_hidden_size, target_expansion_rate
        )
        self.assertEqual(pruning_percentage, 25.0)
        
        # Test with target higher than current
        target_expansion_rate = 500.0
        with self.assertRaises(ValueError):
            calculate_pruning_percentage_from_expansion_rate(
                current_intermediate_size, current_hidden_size, target_expansion_rate
            )

if __name__ == '__main__':
    unittest.main()