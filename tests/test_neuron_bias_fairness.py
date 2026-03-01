"""
Tests for neuron bias analysis and fairness pruning scores.

Covers:
- Change 1: target_layers parameter in register_hooks, process_prompt, get_activation_pairs
- Change 2: analyze_neuron_bias function
- Change 3: compute_fairness_pruning_scores function
- Backward compatibility regression tests
"""

import unittest
import torch
import torch.nn as nn
import sys
import os
from unittest.mock import MagicMock, patch

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from optipfair.bias.activations import (
    ALLOWED_TARGET_LAYERS,
    register_hooks,
    remove_hooks,
    process_prompt,
    get_activation_pairs,
    get_layer_names,
    select_layers,
    analyze_neuron_bias,
    compute_fairness_pruning_scores,
    _normalize,
)


# =============================================================================
# Mock model infrastructure (reused from test_bias_visualization.py)
# =============================================================================


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

    def __init__(self, num_layers=4):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(
            [MockLayer() for _ in range(num_layers)]
        )
        self.device = torch.device("cpu")

    def forward(self, **kwargs):
        x = torch.randn(1, 10, 128)
        for layer in self.model.layers:
            x = layer(x)
        return x


def _make_tokenizer():
    """Create a mock tokenizer for testing."""
    tokenizer = MagicMock()
    tokenizer_output = MagicMock()
    tokenizer_output.input_ids = torch.tensor([[1, 2, 3]])
    tokenizer_output.to = MagicMock(return_value=tokenizer_output)
    tokenizer.return_value = tokenizer_output
    return tokenizer


# =============================================================================
# Change 1 — target_layers tests
# =============================================================================


class TestRegisterHooksTargetLayers(unittest.TestCase):
    """Tests for target_layers parameter in register_hooks."""

    def setUp(self):
        self.model = MockModel()

    def test_no_target_layers_captures_all(self):
        """register_hooks(model) with no args captures all 6 hook types."""
        handles = register_hooks(self.model)
        try:
            # 4 layers × 6 hooks = 24 hooks
            # (attention, mlp_output, gate_proj, up_proj, down_proj, input_norm)
            self.assertEqual(len(handles), 24)
        finally:
            remove_hooks(handles)

    def test_backward_compat_all_hook_types(self):
        """BACKWARD COMPAT: register_hooks(model) captures the same 6 hook types."""
        handles = register_hooks(self.model)
        try:
            # Run a forward pass to populate activations
            self.model(input_ids=torch.tensor([[1, 2, 3]]))

            keys = set(self.model._optipfair_activations.keys())
            # Check all 6 types are present for layer 0
            # Note: input_norm hook is registered but may not fire if the mock
            # forward doesn't call input_layernorm. We verify registration
            # count separately. Here we check the 5 types that do fire.
            expected_prefixes = {
                "attention_output",
                "mlp_output",
                "gate_proj",
                "up_proj",
                "down_proj",
            }
            found_prefixes = set()
            for k in keys:
                parts = k.split("_layer_")
                if len(parts) == 2:
                    found_prefixes.add(parts[0])

            self.assertEqual(found_prefixes, expected_prefixes)

            # Verify input_norm hooks are at least REGISTERED (4 layers)
            # 24 total = 4 layers × 6 hook types
            self.assertEqual(len(handles), 24)
        finally:
            remove_hooks(handles)

    def test_target_layers_gate_proj_only(self):
        """target_layers=["gate_proj"] registers only gate_proj hooks."""
        handles = register_hooks(self.model, target_layers=["gate_proj"])
        try:
            # 4 layers × 1 hook = 4 hooks
            self.assertEqual(len(handles), 4)
        finally:
            remove_hooks(handles)

    def test_target_layers_multiple(self):
        """target_layers=["gate_proj", "up_proj"] registers 2 hook types."""
        handles = register_hooks(
            self.model, target_layers=["gate_proj", "up_proj"]
        )
        try:
            # 4 layers × 2 hooks = 8
            self.assertEqual(len(handles), 8)
        finally:
            remove_hooks(handles)

    def test_target_layers_invalid_raises_valueerror(self):
        """Invalid target_layers raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            register_hooks(self.model, target_layers=["invalid_layer"])
        self.assertIn("invalid_layer", str(ctx.exception))
        self.assertIn("Valid options", str(ctx.exception))

    def test_target_layers_partial_invalid_raises(self):
        """Mix of valid and invalid target_layers raises ValueError."""
        with self.assertRaises(ValueError):
            register_hooks(
                self.model, target_layers=["gate_proj", "nonexistent"]
            )

    def test_no_accidental_substring_match(self):
        """target_layers=["gate_proj"] should NOT capture 'up_proj' etc."""
        handles = register_hooks(self.model, target_layers=["gate_proj"])
        try:
            self.model(input_ids=torch.tensor([[1, 2, 3]]))
            keys = set(self.model._optipfair_activations.keys())
            for k in keys:
                self.assertTrue(
                    k.startswith("gate_proj"),
                    f"Unexpected key captured: {k}",
                )
        finally:
            remove_hooks(handles)


class TestProcessPromptTargetLayers(unittest.TestCase):
    """Tests for target_layers propagation in process_prompt."""

    def setUp(self):
        self.model = MockModel()
        self.tokenizer = _make_tokenizer()

    def test_backward_compat_no_target_layers(self):
        """process_prompt without target_layers returns all hook types."""
        activations = process_prompt(self.model, self.tokenizer, "test")
        self.assertGreater(len(activations), 0)

        prefixes = set()
        for k in activations:
            parts = k.split("_layer_")
            if len(parts) == 2:
                prefixes.add(parts[0])

        # Must have the 5 types that fire during MockModel.forward()
        # (input_norm hook is registered but mock forward doesn't call
        # input_layernorm explicitly, so it may not fire)
        expected = {
            "attention_output",
            "mlp_output",
            "gate_proj",
            "up_proj",
            "down_proj",
        }
        self.assertEqual(prefixes, expected)

    def test_target_layers_filtering(self):
        """process_prompt with target_layers only returns those layers."""
        activations = process_prompt(
            self.model, self.tokenizer, "test", target_layers=["gate_proj"]
        )
        for k in activations:
            self.assertTrue(
                k.startswith("gate_proj"),
                f"Unexpected key: {k}",
            )


class TestGetActivationPairsTargetLayers(unittest.TestCase):
    """Tests for target_layers propagation in get_activation_pairs."""

    def setUp(self):
        self.model = MockModel()
        self.tokenizer = _make_tokenizer()

    def test_backward_compat_no_target_layers(self):
        """get_activation_pairs without target_layers returns all types."""
        act1, act2 = get_activation_pairs(
            self.model, self.tokenizer, "prompt1", "prompt2"
        )
        self.assertGreater(len(act1), 0)
        self.assertGreater(len(act2), 0)
        self.assertEqual(set(act1.keys()), set(act2.keys()))

    def test_target_layers_propagation(self):
        """target_layers propagated to both prompts."""
        act1, act2 = get_activation_pairs(
            self.model,
            self.tokenizer,
            "prompt1",
            "prompt2",
            target_layers=["up_proj"],
        )
        for k in act1:
            self.assertTrue(k.startswith("up_proj"))
        for k in act2:
            self.assertTrue(k.startswith("up_proj"))


# =============================================================================
# Change 2 — analyze_neuron_bias tests
# =============================================================================


class TestAnalyzeNeuronBias(unittest.TestCase):
    """Tests for analyze_neuron_bias function."""

    def setUp(self):
        self.model = MockModel()
        self.tokenizer = _make_tokenizer()
        self.prompt_pairs = [
            ("The doctor said he was tired.", "The doctor said she was tired."),
            ("He is an engineer.", "She is an engineer."),
        ]

    def test_basic_output_structure(self):
        """Returns dict with correct key format and tensor shape."""
        result = analyze_neuron_bias(
            self.model, self.tokenizer, self.prompt_pairs, show_progress=False
        )
        self.assertIsInstance(result, dict)
        self.assertGreater(len(result), 0)

        for key, tensor in result.items():
            # Keys should be like "gate_proj_layer_0" or "up_proj_layer_1"
            self.assertRegex(key, r"^(gate_proj|up_proj)_layer_\d+$")
            # Shape should be [intermediate_size] = [256] for our mock
            self.assertEqual(tensor.shape, torch.Size([256]))
            # Should be on CPU
            self.assertEqual(tensor.device, torch.device("cpu"))

    def test_default_target_layers(self):
        """target_layers=None resolves to ["gate_proj", "up_proj"]."""
        result = analyze_neuron_bias(
            self.model, self.tokenizer, self.prompt_pairs, show_progress=False
        )
        for key in result:
            prefix = key.split("_layer_")[0]
            self.assertIn(prefix, ["gate_proj", "up_proj"])

    def test_custom_target_layers(self):
        """Explicit target_layers are respected."""
        result = analyze_neuron_bias(
            self.model,
            self.tokenizer,
            self.prompt_pairs,
            target_layers=["gate_proj"],
            show_progress=False,
        )
        for key in result:
            self.assertTrue(key.startswith("gate_proj"))

    def test_aggregation_mean(self):
        """aggregation='mean' produces valid output."""
        result = analyze_neuron_bias(
            self.model,
            self.tokenizer,
            self.prompt_pairs,
            aggregation="mean",
            show_progress=False,
        )
        self.assertGreater(len(result), 0)

    def test_aggregation_max(self):
        """aggregation='max' returns max across pairs."""
        result = analyze_neuron_bias(
            self.model,
            self.tokenizer,
            self.prompt_pairs,
            aggregation="max",
            show_progress=False,
        )
        self.assertGreater(len(result), 0)
        for tensor in result.values():
            # Max aggregation means all values should be >= 0
            self.assertTrue(torch.all(tensor >= 0))

    def test_invalid_aggregation_raises(self):
        """Invalid aggregation raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            analyze_neuron_bias(
                self.model,
                self.tokenizer,
                self.prompt_pairs,
                aggregation="median",
                show_progress=False,
            )
        self.assertIn("median", str(ctx.exception))

    def test_empty_prompt_pairs_raises(self):
        """Empty prompt_pairs raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            analyze_neuron_bias(
                self.model, self.tokenizer, [], show_progress=False
            )
        self.assertIn("empty", str(ctx.exception))

    def test_asymmetric_sequence_lengths(self):
        """Handles prompts with different token lengths without error."""
        # The mock tokenizer always returns the same output, but we
        # test the mean-pooling logic by mocking different activation shapes
        with patch(
            "optipfair.bias.activations.get_activation_pairs"
        ) as mock_pairs:
            # Simulate asymmetric shapes: [1, 5, 256] vs [1, 12, 256]
            mock_pairs.return_value = (
                {
                    "gate_proj_layer_0": torch.randn(1, 5, 256),
                    "up_proj_layer_0": torch.randn(1, 5, 256),
                },
                {
                    "gate_proj_layer_0": torch.randn(1, 12, 256),
                    "up_proj_layer_0": torch.randn(1, 12, 256),
                },
            )

            result = analyze_neuron_bias(
                self.model,
                self.tokenizer,
                [("short prompt", "a much longer prompt than the first one")],
                show_progress=False,
            )

            self.assertIn("gate_proj_layer_0", result)
            self.assertEqual(result["gate_proj_layer_0"].shape, torch.Size([256]))

    def test_batch_size_greater_than_one(self):
        """Handles batch_size > 1 correctly (reduces to [intermediate_size])."""
        with patch(
            "optipfair.bias.activations.get_activation_pairs"
        ) as mock_pairs:
            # Shape: [2, 10, 256] (batch_size=2)
            mock_pairs.return_value = (
                {
                    "gate_proj_layer_0": torch.randn(2, 10, 256),
                },
                {
                    "gate_proj_layer_0": torch.randn(2, 8, 256),
                },
            )

            result = analyze_neuron_bias(
                self.model,
                self.tokenizer,
                [("batch prompt 1", "batch prompt 2")],
                target_layers=["gate_proj"],
                show_progress=False,
            )

            self.assertEqual(
                result["gate_proj_layer_0"].shape, torch.Size([256])
            )

    def test_tensors_on_cpu(self):
        """All returned tensors are on CPU."""
        result = analyze_neuron_bias(
            self.model, self.tokenizer, self.prompt_pairs, show_progress=False
        )
        for tensor in result.values():
            self.assertEqual(tensor.device, torch.device("cpu"))


# =============================================================================
# Change 3 — compute_fairness_pruning_scores tests
# =============================================================================


class TestComputeFairnessPruningScores(unittest.TestCase):
    """Tests for compute_fairness_pruning_scores function."""

    def setUp(self):
        self.model = MockModel(num_layers=4)
        # Create mock bias_scores with correct intermediate_size (256)
        self.bias_scores = {}
        for i in range(4):
            self.bias_scores[f"gate_proj_layer_{i}"] = torch.rand(256)
            self.bias_scores[f"up_proj_layer_{i}"] = torch.rand(256)

    def test_basic_output_structure(self):
        """Returns dict with int keys and correct tensor shapes."""
        result = compute_fairness_pruning_scores(
            self.model, self.bias_scores
        )
        self.assertIsInstance(result, dict)
        self.assertGreater(len(result), 0)

        for layer_idx, tensor in result.items():
            self.assertIsInstance(layer_idx, int)
            self.assertEqual(tensor.shape, torch.Size([256]))
            self.assertEqual(tensor.device, torch.device("cpu"))

    def test_all_layers_covered(self):
        """All 4 layers should have scores."""
        result = compute_fairness_pruning_scores(
            self.model, self.bias_scores
        )
        self.assertEqual(set(result.keys()), {0, 1, 2, 3})

    def test_bias_weight_one_pure_bias(self):
        """bias_weight=1.0 should produce scores based only on bias."""
        result = compute_fairness_pruning_scores(
            self.model, self.bias_scores, bias_weight=1.0
        )
        # With bias_weight=1.0: score = 1.0 * bias_norm + 0.0 * (1 - importance_norm)
        # = bias_norm only
        for tensor in result.values():
            # All values should be in [0, 1] (normalized)
            self.assertTrue(torch.all(tensor >= 0))
            self.assertTrue(torch.all(tensor <= 1.0 + 1e-6))

    def test_bias_weight_zero_pure_importance(self):
        """bias_weight=0.0 should produce scores based only on importance."""
        result = compute_fairness_pruning_scores(
            self.model, self.bias_scores, bias_weight=0.0
        )
        # With bias_weight=0.0: score = 0.0 * bias_norm + 1.0 * (1 - importance_norm)
        for tensor in result.values():
            self.assertTrue(torch.all(tensor >= -1e-6))
            self.assertTrue(torch.all(tensor <= 1.0 + 1e-6))

    def test_invalid_bias_weight_raises(self):
        """Out-of-bounds bias_weight raises ValueError."""
        with self.assertRaises(ValueError):
            compute_fairness_pruning_scores(
                self.model, self.bias_scores, bias_weight=1.5
            )
        with self.assertRaises(ValueError):
            compute_fairness_pruning_scores(
                self.model, self.bias_scores, bias_weight=-0.1
            )

    def test_empty_bias_scores_raises(self):
        """Empty bias_scores raises ValueError."""
        with self.assertRaises(ValueError):
            compute_fairness_pruning_scores(self.model, {})

    def test_partial_bias_scores(self):
        """Only layers with matching bias_scores get fairness scores."""
        partial_scores = {
            "gate_proj_layer_0": torch.rand(256),
            "up_proj_layer_0": torch.rand(256),
        }
        result = compute_fairness_pruning_scores(self.model, partial_scores)
        # Only layer 0 should be present
        self.assertEqual(set(result.keys()), {0})

    def test_layers_without_glu_skipped(self):
        """Layers missing GLU attributes are skipped gracefully."""
        # Create a model where one layer lacks gate_proj
        model = MockModel(num_layers=2)
        # Remove gate_proj from layer 1
        del model.model.layers[1].mlp.gate_proj

        bias_scores = {
            "gate_proj_layer_0": torch.rand(256),
            "up_proj_layer_0": torch.rand(256),
            "gate_proj_layer_1": torch.rand(256),
            "up_proj_layer_1": torch.rand(256),
        }

        result = compute_fairness_pruning_scores(model, bias_scores)
        # Layer 1 should be skipped because it lacks gate_proj
        self.assertIn(0, result)
        self.assertNotIn(1, result)


class TestNormalize(unittest.TestCase):
    """Tests for _normalize helper."""

    def test_basic_normalization(self):
        """Normal case: values mapped to [0, 1]."""
        t = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        normed = _normalize(t)
        self.assertAlmostEqual(normed.min().item(), 0.0, places=5)
        self.assertAlmostEqual(normed.max().item(), 1.0, places=3)

    def test_all_same_values(self):
        """When max == min, returns zeros."""
        t = torch.tensor([3.0, 3.0, 3.0])
        normed = _normalize(t)
        self.assertTrue(torch.all(normed == 0))

    def test_single_element(self):
        """Single element tensor."""
        t = torch.tensor([5.0])
        normed = _normalize(t)
        self.assertTrue(torch.all(normed == 0))


# =============================================================================
# Backward Compatibility Regression Tests
# =============================================================================


class TestBackwardCompatibility(unittest.TestCase):
    """Ensure existing code continues to work without changes."""

    def test_register_hooks_old_signature(self):
        """register_hooks(model) still works with single argument."""
        model = MockModel()
        handles = register_hooks(model)
        self.assertGreater(len(handles), 0)
        remove_hooks(handles)

    def test_process_prompt_old_signature(self):
        """process_prompt(model, tokenizer, prompt) with 3 args still works."""
        model = MockModel()
        tokenizer = _make_tokenizer()
        result = process_prompt(model, tokenizer, "test prompt")
        self.assertGreater(len(result), 0)

    def test_get_activation_pairs_old_signature(self):
        """get_activation_pairs(model, tokenizer, p1, p2) with 4 args works."""
        model = MockModel()
        tokenizer = _make_tokenizer()
        act1, act2 = get_activation_pairs(
            model, tokenizer, "prompt1", "prompt2"
        )
        self.assertGreater(len(act1), 0)
        self.assertGreater(len(act2), 0)

    def test_existing_exports_from_init(self):
        """bias/__init__.py continues to export all existing symbols."""
        from optipfair.bias import (
            visualize_bias,
            visualize_mean_differences,
            visualize_heatmap,
            visualize_pca,
            calculate_bias_metrics,
        )
        # All should be callable
        self.assertTrue(callable(visualize_bias))
        self.assertTrue(callable(calculate_bias_metrics))

    def test_new_exports_from_init(self):
        """New symbols are exported from bias/__init__.py."""
        from optipfair.bias import (
            get_activation_pairs,
            analyze_neuron_bias,
            compute_fairness_pruning_scores,
        )
        self.assertTrue(callable(analyze_neuron_bias))
        self.assertTrue(callable(compute_fairness_pruning_scores))

    def test_get_layer_names_still_works(self):
        """get_layer_names utility is unchanged."""
        activations = {
            "mlp_output_layer_0": torch.randn(1, 10, 128),
            "mlp_output_layer_1": torch.randn(1, 10, 128),
            "gate_proj_layer_0": torch.randn(1, 10, 256),
        }
        mlp_layers = get_layer_names(activations, "mlp_output")
        self.assertEqual(len(mlp_layers), 2)

    def test_select_layers_still_works(self):
        """select_layers utility is unchanged."""
        names = [
            "mlp_output_layer_0",
            "mlp_output_layer_1",
            "mlp_output_layer_2",
            "mlp_output_layer_3",
        ]
        selected = select_layers(names, "first_middle_last")
        self.assertEqual(len(selected), 3)

    def test_allowed_target_layers_constant(self):
        """ALLOWED_TARGET_LAYERS is a frozenset with expected values."""
        self.assertIsInstance(ALLOWED_TARGET_LAYERS, frozenset)
        self.assertIn("gate_proj", ALLOWED_TARGET_LAYERS)
        self.assertIn("up_proj", ALLOWED_TARGET_LAYERS)
        self.assertIn("down_proj", ALLOWED_TARGET_LAYERS)
        self.assertIn("mlp_output", ALLOWED_TARGET_LAYERS)
        self.assertIn("attention", ALLOWED_TARGET_LAYERS)
        self.assertIn("input_norm", ALLOWED_TARGET_LAYERS)


if __name__ == "__main__":
    unittest.main()
