"""
Tests for the bias visualization module.
"""

import unittest
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from unittest.mock import MagicMock, patch

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from optipfair.bias.activations import (
    ALLOWED_TARGET_LAYERS,
    register_hooks,
    remove_hooks,
    process_prompt,
    get_prompt_activations,
    get_activation_pairs,
    get_layer_names,
    select_layers,
)
from optipfair.bias.visualization import visualize_prompt_heatmap
from optipfair.bias.visualization import visualize_prompt_layer_heatmap
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
        
        # Create a proper mock tokenizer that returns an object with to() method
        self.tokenizer = MagicMock()
        
        # When the tokenizer is called, it returns a MagicMock with the right methods
        tokenizer_output = MagicMock()
        tokenizer_output.input_ids = torch.tensor([[1, 2, 3]])
        tokenizer_output.to = MagicMock(return_value=tokenizer_output)
        
        # Set the tokenizer to return our mocked output
        self.tokenizer.return_value = tokenizer_output
        
    def test_register_hooks(self):
        """Test hook registration."""
        handles = register_hooks(self.model)
        
        # Should have multiple hooks
        self.assertGreater(len(handles), 0)
        
        # Clean up
        remove_hooks(handles)
        
    def test_process_prompt(self):
        """Test processing prompt and capturing activations."""
        with patch('optipfair.bias.activations.register_hooks') as mock_register:
            # Set up the mock to create a fake _optipfair_activations attribute
            def side_effect(model, target_layers=None):
                model._optipfair_activations = {
                    "mlp_output_layer_0": torch.randn(1, 10, 128),
                    "attention_output_layer_0": torch.randn(1, 10, 128)
                }
                return []
            
            mock_register.side_effect = side_effect
            
            activations = process_prompt(self.model, self.tokenizer, "test prompt")
            
            # Should capture activations from different components
            self.assertGreater(len(activations), 0)
            
            # We expect our two fake activations
            self.assertIn("mlp_output_layer_0", activations)
            self.assertIn("attention_output_layer_0", activations)
        
    def test_get_activation_pairs(self):
        """Test getting activation pairs for two prompts."""
        with patch('optipfair.bias.activations.process_prompt') as mock_process:
            # Set up the process_prompt mock to return fake activations
            mock_process.side_effect = [
                {"mlp_output_layer_0": torch.randn(1, 10, 128), "attention_output_layer_0": torch.randn(1, 10, 128)},
                {"mlp_output_layer_0": torch.randn(1, 10, 128), "attention_output_layer_0": torch.randn(1, 10, 128)}
            ]
            
            act1, act2 = get_activation_pairs(self.model, self.tokenizer, "prompt1", "prompt2")
            
            # Both should have activations
            self.assertGreater(len(act1), 0)
            self.assertGreater(len(act2), 0)
            
            # Should have same keys
            self.assertEqual(set(act1.keys()), set(act2.keys()))

    def test_get_prompt_activations(self):
        """Test getting activations for a single prompt."""
        with patch('optipfair.bias.activations.process_prompt') as mock_process:
            mock_process.return_value = {
                "mlp_output_layer_0": torch.randn(1, 10, 128),
                "attention_output_layer_0": torch.randn(1, 10, 128),
            }

            activations = get_prompt_activations(
                self.model,
                self.tokenizer,
                "prompt1",
            )

            self.assertGreater(len(activations), 0)
            self.assertIn("mlp_output_layer_0", activations)
            self.assertIn("attention_output_layer_0", activations)

    def test_get_prompt_activations_propagates_target_layers(self):
        """Test target_layers is propagated in single-prompt activation API."""
        with patch('optipfair.bias.activations.process_prompt') as mock_process:
            mock_process.return_value = {"down_proj_input_layer_0": torch.randn(1, 10, 256)}

            _ = get_prompt_activations(
                self.model,
                self.tokenizer,
                "prompt1",
                target_layers=["down_proj_input"],
            )

            mock_process.assert_called_once_with(
                model=self.model,
                tokenizer=self.tokenizer,
                prompt="prompt1",
                target_layers=["down_proj_input"],
            )
        
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

    def test_down_proj_input_is_valid_target_layer(self):
        """Validate down_proj_input is accepted as a target layer."""
        self.assertIn("down_proj_input", ALLOWED_TARGET_LAYERS)
        handles = register_hooks(self.model, target_layers=["down_proj_input"])
        try:
            # 4 layers x 1 hook type
            self.assertEqual(len(handles), 4)
        finally:
            remove_hooks(handles)

    def test_down_proj_input_key_registration(self):
        """down_proj_input target registers down_proj_input_layer_* keys."""
        activations = process_prompt(
            self.model,
            self.tokenizer,
            "test prompt",
            target_layers=["down_proj_input"],
        )

        self.assertGreater(len(activations), 0)
        for key in activations:
            self.assertTrue(key.startswith("down_proj_input_layer_"))

    def test_down_proj_and_down_proj_input_shapes(self):
        """down_proj_input has intermediate size, down_proj has hidden size."""
        down_proj_input_activations = process_prompt(
            self.model,
            self.tokenizer,
            "test prompt",
            target_layers=["down_proj_input"],
        )
        down_proj_activations = process_prompt(
            self.model,
            self.tokenizer,
            "test prompt",
            target_layers=["down_proj"],
        )

        self.assertGreater(len(down_proj_input_activations), 0)
        self.assertGreater(len(down_proj_activations), 0)

        # Mock sizes: hidden=128, intermediate=256
        for key, tensor in down_proj_input_activations.items():
            self.assertTrue(key.startswith("down_proj_input_layer_"))
            self.assertEqual(tensor.shape[-1], 256)

        for key, tensor in down_proj_activations.items():
            self.assertTrue(key.startswith("down_proj_layer_"))
            self.assertEqual(tensor.shape[-1], 128)

    def test_combined_down_proj_and_down_proj_input_capture(self):
        """Combined capture returns both down_proj and down_proj_input families."""
        activations = process_prompt(
            self.model,
            self.tokenizer,
            "test prompt",
            target_layers=["down_proj", "down_proj_input"],
        )

        self.assertGreater(len(activations), 0)
        self.assertTrue(any(k.startswith("down_proj_layer_") for k in activations))
        self.assertTrue(any(k.startswith("down_proj_input_layer_") for k in activations))

    def test_backward_compat_down_proj_and_none_behavior(self):
        """Existing down_proj-only and None target behaviors remain unchanged."""
        down_proj_only = process_prompt(
            self.model,
            self.tokenizer,
            "test prompt",
            target_layers=["down_proj"],
        )
        self.assertGreater(len(down_proj_only), 0)
        self.assertTrue(all(k.startswith("down_proj_layer_") for k in down_proj_only))

        default_activations = process_prompt(self.model, self.tokenizer, "test prompt")
        self.assertGreater(len(default_activations), 0)
        self.assertFalse(any(k.startswith("down_proj_input_layer_") for k in default_activations))


class TestVisualizePromptHeatmap(unittest.TestCase):
    """Tests for visualize_prompt_heatmap (single-prompt activation heatmap)."""

    def setUp(self):
        """Reuse the same mock model and tokenizer as TestBiasActivations."""
        self.model = MockModel()
        self.tokenizer = MagicMock()
        tokenizer_output = MagicMock()
        tokenizer_output.input_ids = torch.tensor([[1, 2, 3]])
        tokenizer_output.to = MagicMock(return_value=tokenizer_output)
        self.tokenizer.return_value = tokenizer_output

    def test_runs_without_error_with_valid_layer_key(self):
        """visualize_prompt_heatmap completes without raising for a known layer key."""
        with patch("optipfair.bias.visualization.get_prompt_activations") as mock_gpa, \
             patch("matplotlib.pyplot.show"), \
             patch("matplotlib.pyplot.tight_layout"):
            # Tensor shape: B=1, S=5, H=256 (simulates gate_proj intermediate)
            mock_gpa.return_value = {
                "gate_proj_layer_0": torch.rand(1, 5, 256)
            }
            # Should not raise
            visualize_prompt_heatmap(
                self.model,
                self.tokenizer,
                "test prompt",
                layer_key="gate_proj_layer_0",
                bin_size=64,
                show=False,
            )

    def test_matrix_shape_after_binning(self):
        """Matrix produced has shape (n_tokens, n_bins) = (S, H // bin_size)."""
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend for tests

        with patch("optipfair.bias.visualization.get_prompt_activations") as mock_gpa, \
             patch("matplotlib.pyplot.show"):
            mock_gpa.return_value = {
                "gate_proj_layer_2": torch.ones(1, 4, 256)
            }
            # Capture the matrix via imshow
            captured = {}

            original_imshow = plt.Axes.imshow
            def fake_imshow(self_ax, data, **kwargs):
                captured["matrix"] = data
                return original_imshow(self_ax, data, **kwargs)

            with patch("matplotlib.axes.Axes.imshow", fake_imshow):
                visualize_prompt_heatmap(
                    self.model,
                    self.tokenizer,
                    "test prompt",
                    layer_key="gate_proj_layer_2",
                    bin_size=64,
                    show=False,
                )

            # S=4, H=256, bin_size=64 => n_bins=4
            self.assertIn("matrix", captured)
            matrix = captured["matrix"]
            self.assertEqual(matrix.shape, (4, 4))

    def test_warns_on_missing_layer_key(self):
        """A warning is issued when layer_key is absent from activations."""
        with patch("optipfair.bias.visualization.get_prompt_activations") as mock_gpa, \
             patch("optipfair.bias.visualization.logger") as mock_logger:
            mock_gpa.return_value = {"gate_proj_layer_0": torch.rand(1, 5, 256)}
            visualize_prompt_heatmap(
                self.model,
                self.tokenizer,
                "test prompt",
                layer_key="gate_proj_layer_99",  # Does not exist
                show=False,
            )
            mock_logger.warning.assert_called()

    def test_warns_on_empty_activations(self):
        """A warning is issued when no activations are captured."""
        with patch("optipfair.bias.visualization.get_prompt_activations") as mock_gpa, \
             patch("optipfair.bias.visualization.logger") as mock_logger:
            mock_gpa.return_value = {}
            visualize_prompt_heatmap(
                self.model,
                self.tokenizer,
                "test prompt",
                layer_key="gate_proj_layer_0",
                show=False,
            )
            mock_logger.warning.assert_called()

    def test_no_save_parameter(self):
        """visualize_prompt_heatmap does not accept a 'save' parameter."""
        import inspect
        sig = inspect.signature(visualize_prompt_heatmap)
        self.assertNotIn("save", sig.parameters)

    def test_public_import_from_bias(self):
        """visualize_prompt_heatmap is importable from optipfair.bias."""
        from optipfair.bias import visualize_prompt_heatmap as vph
        self.assertTrue(callable(vph))


class TestVisualizePromptLayerHeatmap(unittest.TestCase):
    """Tests for visualize_prompt_layer_heatmap (layer x neuron-bin heatmap)."""

    def setUp(self):
        self.model = MockModel()
        self.tokenizer = MagicMock()
        tokenizer_output = MagicMock()
        tokenizer_output.input_ids = torch.tensor([[1, 2, 3]])
        tokenizer_output.to = MagicMock(return_value=tokenizer_output)
        self.tokenizer.return_value = tokenizer_output

        # Mock activations: 4 layers, B=1, S=5, H=256
        self.mock_activations = {
            f"gate_proj_layer_{i}": torch.rand(1, 5, 256)
            for i in range(4)
        }

    def test_runs_without_error(self):
        """visualize_prompt_layer_heatmap completes without raising."""
        with patch("optipfair.bias.visualization.get_prompt_activations") as mock_gpa, \
             patch("matplotlib.pyplot.show"):
            mock_gpa.return_value = self.mock_activations
            visualize_prompt_layer_heatmap(
                self.model,
                self.tokenizer,
                "test prompt",
                layer_type="gate_proj",
                bin_size=64,
                show=False,
            )

    def test_matrix_shape_layers_x_bins(self):
        """Matrix shape is (n_layers, n_bins) = (4, 256 // 64)."""
        import matplotlib
        matplotlib.use("Agg")

        captured = {}

        original_imshow = plt.Axes.imshow
        def fake_imshow(self_ax, data, **kwargs):
            captured["matrix"] = data
            return original_imshow(self_ax, data, **kwargs)

        with patch("optipfair.bias.visualization.get_prompt_activations") as mock_gpa, \
             patch("matplotlib.pyplot.show"), \
             patch("matplotlib.axes.Axes.imshow", fake_imshow):
            mock_gpa.return_value = self.mock_activations
            visualize_prompt_layer_heatmap(
                self.model,
                self.tokenizer,
                "test prompt",
                layer_type="gate_proj",
                bin_size=64,
                show=False,
            )

        self.assertIn("matrix", captured)
        # 4 layers, H=256, bin_size=64 => n_bins=4
        self.assertEqual(captured["matrix"].shape, (4, 4))

    def test_warns_on_empty_activations(self):
        """Warning issued when no activations are captured."""
        with patch("optipfair.bias.visualization.get_prompt_activations") as mock_gpa, \
             patch("optipfair.bias.visualization.logger") as mock_logger:
            mock_gpa.return_value = {}
            visualize_prompt_layer_heatmap(
                self.model, self.tokenizer, "test", layer_type="gate_proj", show=False
            )
            mock_logger.warning.assert_called()

    def test_warns_on_no_matching_keys(self):
        """Warning issued when layer_type has no matching keys."""
        with patch("optipfair.bias.visualization.get_prompt_activations") as mock_gpa, \
             patch("optipfair.bias.visualization.logger") as mock_logger:
            mock_gpa.return_value = {"up_proj_layer_0": torch.rand(1, 5, 256)}
            visualize_prompt_layer_heatmap(
                self.model, self.tokenizer, "test", layer_type="gate_proj", show=False
            )
            mock_logger.warning.assert_called()

    def test_no_save_parameter(self):
        """Function does not expose a 'save' parameter."""
        import inspect
        sig = inspect.signature(visualize_prompt_layer_heatmap)
        self.assertNotIn("save", sig.parameters)

    def test_public_import_from_bias(self):
        """visualize_prompt_layer_heatmap is importable from optipfair.bias."""
        from optipfair.bias import visualize_prompt_layer_heatmap as vplh
        self.assertTrue(callable(vplh))


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