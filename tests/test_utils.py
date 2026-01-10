"""
Tests for the utility functions in the OptiPFair library.
"""

import unittest
import torch
from torch import nn
import sys
import os
from unittest.mock import Mock, patch

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from optipfair.pruning.utils import (
    validate_model_for_glu_pruning,
    get_model_layers,
    count_parameters,
    get_pruning_statistics,
    _prepare_batch_inputs
)

class MockLinear(nn.Linear):
    """Mock Linear layer that tracks initialization."""
    
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        self.initialized = True

class MockMLP(nn.Module):
    """Mock MLP module with gate_proj, up_proj, and down_proj."""
    
    def __init__(self, hidden_size=768, intermediate_size=3072):
        super().__init__()
        self.gate_proj = MockLinear(hidden_size, intermediate_size, bias=False)
        self.up_proj = MockLinear(hidden_size, intermediate_size, bias=False)
        self.down_proj = MockLinear(intermediate_size, hidden_size, bias=False)
        self.act_fn = nn.SiLU()

class MockLayer(nn.Module):
    """Mock transformer layer with MLP component."""
    
    def __init__(self, hidden_size=768, intermediate_size=3072):
        super().__init__()
        self.mlp = MockMLP(hidden_size, intermediate_size)
        self.self_attn = nn.Module()  # Mock attention
        self.layernorm = nn.LayerNorm(hidden_size)

class MockModel(nn.Module):
    """Mock model with the expected layer structure."""
    
    def __init__(self, num_layers=12, hidden_size=768, intermediate_size=3072):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([
            MockLayer(hidden_size, intermediate_size) for _ in range(num_layers)
        ])
        self.config = Mock()
        self.config.intermediate_size = intermediate_size

class TestUtils(unittest.TestCase):
    """Test cases for utility functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model = MockModel()
    
    def test_validate_model_for_glu_pruning_valid(self):
        """Test validation with a valid model."""
        result = validate_model_for_glu_pruning(self.model)
        self.assertTrue(result)
    
    def test_validate_model_for_glu_pruning_invalid_no_mlp(self):
        """Test validation with a model missing MLP component."""
        # Create a model with layers that don't have MLP
        model = MockModel()
        for layer in model.model.layers:
            delattr(layer, 'mlp')
            layer.mlp = nn.Module()  # Not a proper MLP
        
        result = validate_model_for_glu_pruning(model)
        self.assertFalse(result)
    
    def test_validate_model_for_glu_pruning_invalid_missing_proj(self):
        """Test validation with a model missing projection layers."""
        # Create a model with MLPs missing gate_proj
        model = MockModel()
        for layer in model.model.layers:
            delattr(layer.mlp, 'gate_proj')
        
        result = validate_model_for_glu_pruning(model)
        self.assertFalse(result)
    
    def test_validate_model_for_glu_pruning_invalid_wrong_dimensions(self):
        """Test validation with mismatched projection dimensions."""
        # Create a model with mismatched dimensions
        model = MockModel()
        # Make the first layer's up_proj have different dimensions
        layer = model.model.layers[0]
        layer.mlp.up_proj = MockLinear(768, 4096, bias=False)  # Different from gate_proj
        
        result = validate_model_for_glu_pruning(model)
        self.assertFalse(result)
    
    def test_get_model_layers_llama_style(self):
        """Test getting layers from a LLaMA-style model."""
        layers = get_model_layers(self.model)
        self.assertEqual(len(layers), 12)
        self.assertIs(layers[0], self.model.model.layers[0])
    
    def test_get_model_layers_alternate_style(self):
        """Test getting layers from an alternate model style."""
        # Create a model with GPT-2 style architecture
        model = nn.Module()
        model.transformer = nn.Module()
        model.transformer.h = nn.ModuleList([MockLayer() for _ in range(6)])
        
        layers = get_model_layers(model)
        self.assertEqual(len(layers), 6)
        self.assertIs(layers[0], model.transformer.h[0])
    
    def test_get_model_layers_bert_style(self):
        """Test getting layers from a BERT-style model."""
        # Create a model with BERT style architecture
        model = nn.Module()
        model.encoder = nn.Module()
        model.encoder.layer = nn.ModuleList([MockLayer() for _ in range(4)])
        
        layers = get_model_layers(model)
        self.assertEqual(len(layers), 4)
        self.assertIs(layers[0], model.encoder.layer[0])
    
    def test_get_model_layers_not_found(self):
        """Test getting layers when they can't be found."""
        model = nn.Module()  # Empty model
        
        layers = get_model_layers(model)
        self.assertEqual(len(layers), 0)
    
    def test_count_parameters(self):
        """Test counting parameters."""
        # Create a small model with known parameter count
        model = nn.Sequential(
            nn.Linear(10, 20),  # 10*20 + 20 = 220 params
            nn.ReLU(),          # 0 params
            nn.Linear(20, 5)    # 20*5 + 5 = 105 params
        )
        # Total: 325 params
        
        count = count_parameters(model)
        self.assertEqual(count, 325)
    
    def test_get_pruning_statistics(self):
        """Test getting pruning statistics."""
        # Create original and pruned models with known parameter counts
        original_model = MockModel(num_layers=4, hidden_size=768, intermediate_size=3072)
        pruned_model = MockModel(num_layers=4, hidden_size=768, intermediate_size=1536)
        
        # Calculate expected parameter count difference
        # Each layer has:
        # - gate_proj: hidden_size * intermediate_size = 768 * 3072 = 2,359,296 params
        # - up_proj: hidden_size * intermediate_size = 768 * 3072 = 2,359,296 params
        # - down_proj: intermediate_size * hidden_size = 3072 * 768 = 2,359,296 params
        # Total per layer: 7,077,888 params
        # Total for original (4 layers): 28,311,552 params
        
        # For pruned model with intermediate_size = 1536:
        # - gate_proj: 768 * 1536 = 1,179,648 params
        # - up_proj: 768 * 1536 = 1,179,648 params
        # - down_proj: 1536 * 768 = 1,179,648 params
        # Total per layer: 3,538,944 params
        # Total for pruned (4 layers): 14,155,776 params
        
        # Mock the count_parameters function to return these values
        with patch('optipfair.pruning.utils.count_parameters') as mock_count:
            mock_count.side_effect = [28_311_552, 14_155_776]
            
            stats = get_pruning_statistics(original_model, pruned_model)
            
            self.assertEqual(stats["original_parameters"], 28_311_552)
            self.assertEqual(stats["pruned_parameters"], 14_155_776)
            self.assertEqual(stats["reduction"], 14_155_776)
            self.assertEqual(stats["percentage_reduction"], 50.0)
            self.assertEqual(stats["expansion_rate"], 200.0)  # 1536/768 * 100 = 200%


class TestPrepareBatchInputs(unittest.TestCase):
    """Test cases for _prepare_batch_inputs utility function."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device('cpu')
        self.input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
        self.attention_mask = torch.tensor([[1, 1, 1], [1, 1, 0]])
        self.token_type_ids = torch.tensor([[0, 0, 0], [0, 0, 0]])
    
    def test_single_tensor_as_input_ids(self):
        """Test that a single tensor is treated as input_ids."""
        result = _prepare_batch_inputs(self.input_ids, self.device)
        
        self.assertIn('input_ids', result)
        self.assertEqual(len(result), 1)
        self.assertTrue(torch.equal(result['input_ids'], self.input_ids))
    
    def test_dict_batch(self):
        """Test that dict batches are handled correctly."""
        batch = {
            'input_ids': self.input_ids,
            'attention_mask': self.attention_mask,
        }
        
        result = _prepare_batch_inputs(batch, self.device)
        
        self.assertIn('input_ids', result)
        self.assertIn('attention_mask', result)
        self.assertTrue(torch.equal(result['input_ids'], self.input_ids))
        self.assertTrue(torch.equal(result['attention_mask'], self.attention_mask))
    
    def test_dict_batch_with_extra_keys(self):
        """Test that dict batches preserve extra keys."""
        batch = {
            'input_ids': self.input_ids,
            'attention_mask': self.attention_mask,
            'labels': torch.tensor([[2, 3, 4], [5, 6, 7]]),
        }
        
        result = _prepare_batch_inputs(batch, self.device)
        
        self.assertIn('input_ids', result)
        self.assertIn('attention_mask', result)
        self.assertIn('labels', result)
    
    def test_tuple_batch_two_elements(self):
        """Test tuple batch with input_ids and attention_mask."""
        batch = (self.input_ids, self.attention_mask)
        
        result = _prepare_batch_inputs(batch, self.device)
        
        self.assertIn('input_ids', result)
        self.assertIn('attention_mask', result)
        self.assertTrue(torch.equal(result['input_ids'], self.input_ids))
        self.assertTrue(torch.equal(result['attention_mask'], self.attention_mask))
    
    def test_list_batch_two_elements(self):
        """Test list batch with input_ids and attention_mask."""
        batch = [self.input_ids, self.attention_mask]
        
        result = _prepare_batch_inputs(batch, self.device)
        
        self.assertIn('input_ids', result)
        self.assertIn('attention_mask', result)
        self.assertTrue(torch.equal(result['input_ids'], self.input_ids))
        self.assertTrue(torch.equal(result['attention_mask'], self.attention_mask))
    
    def test_tuple_batch_three_elements(self):
        """Test tuple batch with input_ids, attention_mask, and token_type_ids."""
        batch = (self.input_ids, self.attention_mask, self.token_type_ids)
        
        result = _prepare_batch_inputs(batch, self.device)
        
        self.assertIn('input_ids', result)
        self.assertIn('attention_mask', result)
        self.assertIn('token_type_ids', result)
        self.assertTrue(torch.equal(result['input_ids'], self.input_ids))
        self.assertTrue(torch.equal(result['attention_mask'], self.attention_mask))
        self.assertTrue(torch.equal(result['token_type_ids'], self.token_type_ids))
    
    def test_tuple_batch_with_none_values(self):
        """Test tuple batch with None values are skipped."""
        batch = (self.input_ids, None, self.token_type_ids)
        
        result = _prepare_batch_inputs(batch, self.device)
        
        self.assertIn('input_ids', result)
        self.assertNotIn('attention_mask', result)
        self.assertIn('token_type_ids', result)
    
    def test_tuple_batch_single_element(self):
        """Test tuple batch with only input_ids."""
        batch = (self.input_ids,)
        
        result = _prepare_batch_inputs(batch, self.device)
        
        self.assertIn('input_ids', result)
        self.assertEqual(len(result), 1)
    
    def test_unsupported_format_raises_error(self):
        """Test that unsupported formats raise ValueError."""
        with self.assertRaises(ValueError) as context:
            _prepare_batch_inputs("invalid_batch", self.device)
        
        self.assertIn("Unsupported batch format", str(context.exception))
        self.assertIn("str", str(context.exception))
    
    def test_tensors_moved_to_device(self):
        """Test that tensors are moved to the specified device."""
        batch = {'input_ids': self.input_ids, 'attention_mask': self.attention_mask}
        
        result = _prepare_batch_inputs(batch, self.device)
        
        self.assertEqual(result['input_ids'].device, self.device)
        self.assertEqual(result['attention_mask'].device, self.device)
    
    def test_dict_with_none_values(self):
        """Test that dict with None values handles them correctly."""
        batch = {
            'input_ids': self.input_ids,
            'attention_mask': None,
        }
        
        result = _prepare_batch_inputs(batch, self.device)
        
        self.assertIn('input_ids', result)
        # None values should not be included since they're not tensors
        self.assertNotIn('attention_mask', result)


if __name__ == '__main__':
    unittest.main()