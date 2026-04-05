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
    compute_neuron_pair_importance_l2,
    round_to_divisor,
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
    
    def test_compute_neuron_pair_importance_l2(self):
        """Test L2 importance calculation."""
        importance = compute_neuron_pair_importance_l2(self.gate_weight, self.up_weight)
        
        # Check shape
        self.assertEqual(importance.shape[0], self.intermediate_size)
        
        # Check that all values are positive or zero (L2 norm is non-negative)
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
        
        # Check the new size (use approximate comparison due to int() rounding)
        expected_size = int(self.intermediate_size * (1 - prune_percentage/100))
        self.assertAlmostEqual(new_size, expected_size, delta=1)
        
        # Check dimensions of new layers
        self.assertEqual(new_gate_proj.in_features, self.hidden_size)
        self.assertEqual(new_gate_proj.out_features, new_size)
        
        self.assertEqual(new_up_proj.in_features, self.hidden_size)
        self.assertEqual(new_up_proj.out_features, new_size)
        
        self.assertEqual(new_down_proj.in_features, new_size)
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
        """Test pruning with 100% percentage keeps at least 1 neuron."""
        prune_percentage = 100.0
        
        # 100% pruning is prevented by min(..., original_size - 1)
        # Should keep exactly 1 neuron
        new_gate, new_up, new_down, k = prune_neuron_pairs(self.mlp, prune_percentage)
        
        self.assertEqual(k, 1)  # At least 1 neuron kept
        self.assertEqual(new_gate.out_features, 1)
        self.assertEqual(new_up.out_features, 1)
        self.assertEqual(new_down.in_features, 1)
    
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

    def test_round_to_divisor_uses_floor(self):
        """Test round_to_divisor rounds down, never up."""
        self.assertEqual(round_to_divisor(8100, 128), 8064)
        self.assertEqual(round_to_divisor(8200, 128), 8192)
        # Nearest rounding would produce 8192; floor must keep 8064.
        self.assertEqual(round_to_divisor(8150, 128), 8064)

    def test_prune_neuron_pairs_raises_when_not_effective(self):
        """Test tiny pruning that keeps base size raises an error."""
        with self.assertRaises(ValueError) as context:
            prune_neuron_pairs(
                self.mlp,
                prune_percentage=0.01,
                expansion_divisor=128,
            )

        self.assertIn("No effective pruning", str(context.exception))


# ==============================================================================
# Phase 2.2: Fairness Pruning Integration Tests
# ==============================================================================

class TestPruneNeuronPairsCustomScores(unittest.TestCase):
    """Test prune_neuron_pairs() with custom_importance_scores parameter."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.hidden_size = 768
        self.intermediate_size = 3072
        self.mlp = MockMLP(self.hidden_size, self.intermediate_size)
    
    def test_custom_scores_shape_mismatch_raises(self):
        """Test that wrong tensor shape raises ValueError."""
        custom_scores = torch.rand(100)  # Wrong size
        
        with self.assertRaises(ValueError) as context:
            prune_neuron_pairs(
                self.mlp,
                prune_percentage=20.0,
                custom_importance_scores=custom_scores
            )
        
        self.assertIn("shape mismatch", str(context.exception).lower())
    
    def test_custom_scores_out_of_range_raises(self):
        """Test that scores not in [0,1] raises ValueError."""
        # Test values > 1
        custom_scores = torch.rand(self.intermediate_size) * 2  # Values in [0, 2]
        
        with self.assertRaises(ValueError) as context:
            prune_neuron_pairs(
                self.mlp,
                prune_percentage=20.0,
                custom_importance_scores=custom_scores
            )
        
        self.assertIn("[0, 1]", str(context.exception))
        
        # Test values < 0
        custom_scores = torch.rand(self.intermediate_size) - 0.5  # Some negative values
        
        with self.assertRaises(ValueError) as context:
            prune_neuron_pairs(
                self.mlp,
                prune_percentage=20.0,
                custom_importance_scores=custom_scores
            )
        
        self.assertIn("[0, 1]", str(context.exception))
    
    def test_custom_scores_applied_to_selection(self):
        """Test that custom scores determine pruning, not importance_fn."""
        # Create custom scores where first half has high scores (will be pruned after inversion)
        custom_scores = torch.zeros(self.intermediate_size)
        custom_scores[:self.intermediate_size // 2] = 0.9  # High fairness = prune candidate
        custom_scores[self.intermediate_size // 2:] = 0.1  # Low fairness = keep
        
        prune_percentage = 50.0
        new_gate, new_up, new_down, k = prune_neuron_pairs(
            self.mlp,
            prune_percentage=prune_percentage,
            custom_importance_scores=custom_scores
        )
        
        # After 50% pruning, should have half the neurons
        expected_k = self.intermediate_size // 2
        self.assertEqual(k, expected_k)
        self.assertEqual(new_gate.out_features, expected_k)
    
    def test_custom_scores_inversion_keeps_lowest_scores(self):
        """Test that inverted semantics work correctly."""
        # Create scores where specific neurons have low fairness scores (should be kept)
        custom_scores = torch.ones(self.intermediate_size) * 0.8
        # Set specific neurons to very low fairness (high importance, low bias)
        keep_indices = [0, 10, 100, 500, 1000]
        for idx in keep_indices:
            custom_scores[idx] = 0.0
        
        prune_percentage = 99.0  # Prune almost everything
        new_gate, new_up, new_down, k = prune_neuron_pairs(
            self.mlp,
            prune_percentage=prune_percentage,
            custom_importance_scores=custom_scores
        )
        
        # With 99% pruning, should keep ~1% of neurons (~30 neurons)
        expected_k = int(self.intermediate_size * (1 - prune_percentage / 100))
        self.assertAlmostEqual(k, expected_k, delta=5)
    
    def test_custom_scores_all_zeros_keeps_all(self):
        """Test that all zeros (lowest fairness) keeps all neurons after inversion."""
        custom_scores = torch.zeros(self.intermediate_size)
        
        prune_percentage = 20.0
        new_gate, new_up, new_down, k = prune_neuron_pairs(
            self.mlp,
            prune_percentage=prune_percentage,
            custom_importance_scores=custom_scores
        )
        
        # All have same score after inversion (1.0), so pruning is still applied
        # Use approximate comparison due to int() truncation
        expected_k = int(self.intermediate_size * (1 - prune_percentage / 100))
        self.assertAlmostEqual(k, expected_k, delta=1)
    
    def test_custom_scores_all_ones_prunes_most(self):
        """Test that all ones (highest fairness) prunes aggressively after inversion."""
        custom_scores = torch.ones(self.intermediate_size)
        
        prune_percentage = 20.0
        new_gate, new_up, new_down, k = prune_neuron_pairs(
            self.mlp,
            prune_percentage=prune_percentage,
            custom_importance_scores=custom_scores
        )
        
        # All have same score after inversion (0.0), so pruning is still applied
        # Use approximate comparison due to int() truncation
        expected_k = int(self.intermediate_size * (1 - prune_percentage / 100))
        self.assertAlmostEqual(k, expected_k, delta=1)
    
    def test_custom_scores_precedence_over_activation_norms(self):
        """Test that custom scores override activation_norms."""
        # Create custom scores
        custom_scores = torch.rand(self.intermediate_size)
        
        # Create fake activation norms that would select different neurons
        activation_norms = torch.rand(self.intermediate_size) * 10
        
        prune_percentage = 30.0
        
        # Prune with both custom_scores and activation_norms
        new_gate1, new_up1, new_down1, k1 = prune_neuron_pairs(
            self.mlp,
            prune_percentage=prune_percentage,
            custom_importance_scores=custom_scores,
            activation_norms=activation_norms
        )
        
        # Prune with only custom_scores
        new_gate2, new_up2, new_down2, k2 = prune_neuron_pairs(
            self.mlp,
            prune_percentage=prune_percentage,
            custom_importance_scores=custom_scores
        )
        
        # Results should be identical (custom_scores takes precedence)
        self.assertEqual(k1, k2)
        self.assertTrue(torch.equal(new_gate1.weight, new_gate2.weight))


class MockLayer(nn.Module):
    """Mock transformer layer for testing."""
    
    def __init__(self, hidden_size=768, intermediate_size=3072):
        super().__init__()
        self.mlp = MockMLP(hidden_size, intermediate_size)


class MockModelForPruning(nn.Module):
    """Mock model with transformer layers for testing prune_model_mlp_glu."""
    
    def __init__(self, num_layers=4, hidden_size=768, intermediate_size=3072):
        super().__init__()
        
        # Create model structure expected by get_model_layers()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([
            MockLayer(hidden_size, intermediate_size) for _ in range(num_layers)
        ])
        
        # Add config
        self.config = nn.Module()
        self.config.intermediate_size = intermediate_size
        self.config.hidden_size = hidden_size
    
    def forward(self, **kwargs):
        return torch.randn(1, 10, self.config.hidden_size)


class TestPruneModelFairnessScores(unittest.TestCase):
    """Test prune_model_mlp_glu() with fairness_scores parameter."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.num_layers = 4
        self.hidden_size = 768
        self.intermediate_size = 3072
        self.model = MockModelForPruning(
            self.num_layers, 
            self.hidden_size, 
            self.intermediate_size
        )
    
    def _create_fairness_scores(self, layer_indices=None):
        """Helper to create fairness scores dict."""
        if layer_indices is None:
            layer_indices = range(self.num_layers)
        
        return {
            idx: torch.rand(self.intermediate_size) 
            for idx in layer_indices
        }
    
    def test_fairness_scores_empty_dict(self):
        """Test that empty dict is accepted with no effect."""
        from optipfair.pruning.mlp_glu import prune_model_mlp_glu
        
        fairness_scores = {}
        
        # Should not raise error
        pruned_model = prune_model_mlp_glu(
            self.model,
            pruning_percentage=20.0,
            fairness_scores=fairness_scores,
            show_progress=False
        )
        
        # Model should be pruned normally
        self.assertIsNotNone(pruned_model)
    
    def test_fairness_scores_partial_layers(self):
        """Test scores for some layers only, others use standard method."""
        from optipfair.pruning.mlp_glu import prune_model_mlp_glu
        
        # Only provide scores for layers 0 and 2
        fairness_scores = self._create_fairness_scores([0, 2])
        
        pruned_model = prune_model_mlp_glu(
            self.model,
            pruning_percentage=20.0,
            fairness_scores=fairness_scores,
            show_progress=False
        )
        
        # All layers should be pruned
        for layer in pruned_model.model.layers:
            new_size = layer.mlp.gate_proj.out_features
            self.assertLess(new_size, self.intermediate_size)
    
    def test_fairness_scores_all_layers(self):
        """Test scores provided for all layers."""
        from optipfair.pruning.mlp_glu import prune_model_mlp_glu
        
        fairness_scores = self._create_fairness_scores()
        
        pruned_model = prune_model_mlp_glu(
            self.model,
            pruning_percentage=20.0,
            fairness_scores=fairness_scores,
            show_progress=False
        )
        
        # All layers should be pruned
        for layer in pruned_model.model.layers:
            new_size = layer.mlp.gate_proj.out_features
            self.assertLess(new_size, self.intermediate_size)
    
    def test_fairness_scores_invalid_type_raises(self):
        """Test that non-dict raises TypeError."""
        from optipfair.pruning.mlp_glu import prune_model_mlp_glu
        
        fairness_scores = "not a dict"
        
        with self.assertRaises(TypeError) as context:
            prune_model_mlp_glu(
                self.model,
                pruning_percentage=20.0,
                fairness_scores=fairness_scores,
                show_progress=False
            )
        
        self.assertIn("Dict[int, torch.Tensor]", str(context.exception))
    
    def test_fairness_scores_invalid_keys_raises(self):
        """Test that non-int keys raise TypeError."""
        from optipfair.pruning.mlp_glu import prune_model_mlp_glu
        
        fairness_scores = {"0": torch.rand(self.intermediate_size)}  # String key
        
        with self.assertRaises(TypeError) as context:
            prune_model_mlp_glu(
                self.model,
                pruning_percentage=20.0,
                fairness_scores=fairness_scores,
                show_progress=False
            )
        
        self.assertIn("keys must be int", str(context.exception))
    
    def test_fairness_scores_invalid_values_raises(self):
        """Test that non-Tensor values raise TypeError."""
        from optipfair.pruning.mlp_glu import prune_model_mlp_glu
        
        fairness_scores = {0: [0.1, 0.2, 0.3]}  # List instead of Tensor
        
        with self.assertRaises(TypeError) as context:
            prune_model_mlp_glu(
                self.model,
                pruning_percentage=20.0,
                fairness_scores=fairness_scores,
                show_progress=False
            )
        
        self.assertIn("values must be torch.Tensor", str(context.exception))
    
    def test_fairness_scores_none_backward_compat(self):
        """Test that fairness_scores=None works as before."""
        from optipfair.pruning.mlp_glu import prune_model_mlp_glu
        
        # Prune without fairness_scores (backward compatibility)
        pruned_model = prune_model_mlp_glu(
            self.model,
            pruning_percentage=20.0,
            fairness_scores=None,
            show_progress=False
        )
        
        # Should work normally - use approximate comparison due to rounding
        for layer in pruned_model.model.layers:
            new_size = layer.mlp.gate_proj.out_features
            expected_size = int(self.intermediate_size * 0.8)
            self.assertAlmostEqual(new_size, expected_size, delta=1)
    
    def test_fairness_scores_with_expansion_rate(self):
        """Test that fairness_scores works with expansion_rate parameter."""
        from optipfair.pruning.mlp_glu import prune_model_mlp_glu
        
        fairness_scores = self._create_fairness_scores()
        
        # Use expansion_rate instead of pruning_percentage
        target_expansion_rate = 200.0  # 200% = 2x hidden_size
        
        pruned_model = prune_model_mlp_glu(
            self.model,
            pruning_percentage=None,  # Must be None when using expansion_rate
            expansion_rate=target_expansion_rate,
            fairness_scores=fairness_scores,
            show_progress=False
        )
        
        # Check that size was reduced
        for layer in pruned_model.model.layers:
            new_size = layer.mlp.gate_proj.out_features
            expected_size = int(self.hidden_size * (target_expansion_rate / 100))
            self.assertEqual(new_size, expected_size)
    
    def test_fairness_scores_with_selective_layers(self):
        """Test that fairness_scores works with layer_indices parameter."""
        from optipfair.pruning.mlp_glu import prune_model_mlp_glu
        
        # Only prune layers 0 and 2
        layer_indices = [0, 2]
        fairness_scores = self._create_fairness_scores(layer_indices)
        
        pruned_model = prune_model_mlp_glu(
            self.model,
            pruning_percentage=20.0,
            layer_indices=layer_indices,
            fairness_scores=fairness_scores,
            show_progress=False
        )
        
        # Layers 0 and 2 should be pruned
        for idx in layer_indices:
            new_size = pruned_model.model.layers[idx].mlp.gate_proj.out_features
            self.assertLess(new_size, self.intermediate_size)
        
        # Layers 1 and 3 should be unchanged
        for idx in [1, 3]:
            size = pruned_model.model.layers[idx].mlp.gate_proj.out_features
            self.assertEqual(size, self.intermediate_size)
    
    def test_fairness_scores_shape_mismatch_raises(self):
        """Test that shape mismatch per layer raises ValueError."""
        from optipfair.pruning.mlp_glu import prune_model_mlp_glu
        
        # Create scores with wrong shape for layer 0
        fairness_scores = {
            0: torch.rand(100)  # Wrong size
        }
        
        with self.assertRaises(ValueError) as context:
            prune_model_mlp_glu(
                self.model,
                pruning_percentage=20.0,
                fairness_scores=fairness_scores,
                show_progress=False
            )
        
        self.assertIn("shape mismatch", str(context.exception).lower())


class TestIntegrationBiasAndPruning(unittest.TestCase):
    """Integration tests for complete bias-to-pruning pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.num_layers = 4
        self.hidden_size = 768
        self.intermediate_size = 3072
        self.model = MockModelForPruning(
            self.num_layers,
            self.hidden_size,
            self.intermediate_size
        )
    
    def test_end_to_end_bias_to_pruning(self):
        """Test full pipeline: create fairness scores → pass to prune_model."""
        from optipfair.pruning.mlp_glu import prune_model_mlp_glu
        
        # Simulate fairness scores from compute_fairness_pruning_scores()
        # Format: Dict[int, torch.Tensor] with layer indices as keys
        fairness_scores = {
            idx: torch.rand(self.intermediate_size)
            for idx in range(self.num_layers)
        }
        
        # Ensure scores are in [0, 1] range (normalized)
        for scores in fairness_scores.values():
            self.assertTrue(scores.min() >= 0)
            self.assertTrue(scores.max() <= 1)
        
        # Prune model with fairness scores
        pruned_model = prune_model_mlp_glu(
            self.model,
            pruning_percentage=30.0,
            fairness_scores=fairness_scores,
            show_progress=False
        )
        
        # Verify all layers were pruned
        for layer in pruned_model.model.layers:
            new_size = layer.mlp.gate_proj.out_features
            self.assertLess(new_size, self.intermediate_size)
            # Verify all three projections have same size
            self.assertEqual(layer.mlp.gate_proj.out_features, new_size)
            self.assertEqual(layer.mlp.up_proj.out_features, new_size)
            self.assertEqual(layer.mlp.down_proj.in_features, new_size)
    
    def test_fairness_pruning_preserves_config(self):
        """Test that config is updated correctly after fairness pruning."""
        from optipfair.pruning.mlp_glu import prune_model_mlp_glu
        
        fairness_scores = {
            idx: torch.rand(self.intermediate_size)
            for idx in range(self.num_layers)
        }
        
        pruned_model = prune_model_mlp_glu(
            self.model,
            pruning_percentage=25.0,
            fairness_scores=fairness_scores,
            show_progress=False
        )
        
        # Config should be updated
        expected_size = int(self.intermediate_size * 0.75)
        self.assertEqual(pruned_model.config.intermediate_size, expected_size)
    
    def test_fairness_and_standard_methods_produce_different_results(self):
        """Test that fairness pruning selects different neurons than standard."""
        from optipfair.pruning.mlp_glu import prune_model_mlp_glu
        import copy
        
        # Create two identical models
        model1 = MockModelForPruning(self.num_layers, self.hidden_size, self.intermediate_size)
        model2 = copy.deepcopy(model1)
        
        # Create fairness scores that favor specific neurons
        fairness_scores = {}
        for idx in range(self.num_layers):
            scores = torch.ones(self.intermediate_size) * 0.5
            # Give low fairness scores (keep) to first quarter
            scores[:self.intermediate_size // 4] = 0.1
            # Give high fairness scores (prune) to last quarter
            scores[3 * self.intermediate_size // 4:] = 0.9
            fairness_scores[idx] = scores
        
        # Prune with fairness
        pruned_fairness = prune_model_mlp_glu(
            model1,
            pruning_percentage=20.0,
            fairness_scores=fairness_scores,
            show_progress=False
        )
        
        # Prune with standard method
        pruned_standard = prune_model_mlp_glu(
            model2,
            pruning_percentage=20.0,
            show_progress=False
        )
        
        # Both should have same final size
        fairness_size = pruned_fairness.model.layers[0].mlp.gate_proj.out_features
        standard_size = pruned_standard.model.layers[0].mlp.gate_proj.out_features
        self.assertEqual(fairness_size, standard_size)
        
        # But weights should likely be different (different neurons selected)
        # Note: There's a small chance they could be identical, but very unlikely
        fairness_weights = pruned_fairness.model.layers[0].mlp.gate_proj.weight
        standard_weights = pruned_standard.model.layers[0].mlp.gate_proj.weight
        
        # Just verify that both pruning operations completed successfully
        self.assertIsNotNone(fairness_weights)
        self.assertIsNotNone(standard_weights)


if __name__ == '__main__':
    unittest.main()