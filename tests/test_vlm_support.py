"""
Tests for vision-language model (VLM) support.

VLMs such as Qwen3.5, Qwen3-VL or Gemma 3 multimodal nest their text decoder
next to a vision encoder (model.language_model.layers) and keep the decoder
parameters in a nested text_config. These tests use a small mock with the same
layout to check that pruning and bias analysis target the text decoder only and
keep the nested configuration in sync.
"""

import unittest
from unittest.mock import MagicMock, patch

import torch
from torch import nn
from torch.utils.data import DataLoader
import sys
import os

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transformers import PretrainedConfig

from optipfair import prune_model
from optipfair.bias.activations import (
    compute_fairness_pruning_scores,
    register_hooks,
    remove_hooks,
)
from optipfair.pruning.mlp_glu import get_activation_norms, prune_model_mlp_glu
from optipfair.pruning.utils import (
    get_layers_path,
    get_model_layers,
    get_text_config,
    validate_model_for_glu_pruning,
)

HIDDEN = 32
INTERMEDIATE = 128
NUM_LAYERS = 4
VOCAB = 50
LAYER_TYPES = ["linear_attention", "full_attention", "linear_attention", "full_attention"]


class MockGLUMLP(nn.Module):
    """GLU MLP with gate_proj, up_proj and down_proj."""

    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class MockDecoderLayer(nn.Module):
    """Hybrid decoder layer: full attention (self_attn) or linear attention (linear_attn)."""

    def __init__(self, layer_type):
        super().__init__()
        if layer_type == "full_attention":
            self.self_attn = nn.Linear(HIDDEN, HIDDEN)
        else:
            self.linear_attn = nn.Linear(HIDDEN, HIDDEN)
        self.input_layernorm = nn.LayerNorm(HIDDEN)
        self.mlp = MockGLUMLP(HIDDEN, INTERMEDIATE)

    def forward(self, x):
        attn = self.self_attn if hasattr(self, "self_attn") else self.linear_attn
        x = x + attn(self.input_layernorm(x))
        return x + self.mlp(x)


class MockVisionMLP(nn.Module):
    """Non-GLU vision MLP that must never be pruned."""

    def __init__(self):
        super().__init__()
        self.linear_fc1 = nn.Linear(16, 64)
        self.linear_fc2 = nn.Linear(64, 16)


class MockVisionBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = MockVisionMLP()


class MockVLM(nn.Module):
    """Mimics Qwen3_5ForConditionalGeneration: model.visual + model.language_model."""

    def __init__(self):
        super().__init__()
        text_config = PretrainedConfig(
            hidden_size=HIDDEN,
            intermediate_size=INTERMEDIATE,
            num_hidden_layers=NUM_LAYERS,
            layer_types=list(LAYER_TYPES),
        )
        self.config = PretrainedConfig(text_config=text_config)
        # PretrainedConfig stores unknown dict kwargs as-is; make sure it is a config object
        self.config.text_config = text_config

        self.model = nn.Module()
        self.model.visual = nn.Module()
        self.model.visual.blocks = nn.ModuleList([MockVisionBlock() for _ in range(2)])
        self.model.language_model = nn.Module()
        self.model.language_model.embed_tokens = nn.Embedding(VOCAB, HIDDEN)
        self.model.language_model.layers = nn.ModuleList(
            [MockDecoderLayer(t) for t in LAYER_TYPES]
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None):
        x = inputs_embeds
        if x is None:
            x = self.model.language_model.embed_tokens(input_ids)
        for layer in self.model.language_model.layers:
            x = layer(x)
        return x


class MockTextModel(nn.Module):
    """Text-only LLaMA-style model (model.layers, flat config)."""

    def __init__(self):
        super().__init__()
        self.config = PretrainedConfig(intermediate_size=INTERMEDIATE, num_hidden_layers=2)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(
            [MockDecoderLayer("full_attention") for _ in range(2)]
        )


class TestVLMLayerDiscovery(unittest.TestCase):
    """Decoder layers and text config are found in nested VLM layouts."""

    def test_layers_path_vlm(self):
        self.assertEqual(get_layers_path(MockVLM()), "model.language_model.layers")

    def test_layers_path_text_model_unchanged(self):
        self.assertEqual(get_layers_path(MockTextModel()), "model.layers")

    def test_get_model_layers_returns_text_decoder(self):
        layers = get_model_layers(MockVLM())
        self.assertEqual(len(layers), NUM_LAYERS)
        self.assertTrue(all(isinstance(layer, MockDecoderLayer) for layer in layers))

    def test_validate_vlm(self):
        self.assertTrue(validate_model_for_glu_pruning(MockVLM()))

    def test_get_text_config_nested(self):
        model = MockVLM()
        self.assertIs(get_text_config(model), model.config.text_config)

    def test_get_text_config_flat(self):
        model = MockTextModel()
        self.assertIs(get_text_config(model), model.config)

    def test_get_text_config_without_accessor(self):
        # Older transformers versions have nested text_config but no get_text_config()
        class LegacyConfig(PretrainedConfig):
            @property
            def get_text_config(self):
                raise AttributeError

        model = MockVLM()
        text_config = model.config.text_config
        model.config = LegacyConfig()
        model.config.text_config = text_config
        self.assertIs(get_text_config(model), text_config)


class TestVLMPruning(unittest.TestCase):
    """Pruning modifies only the text decoder and syncs the nested config."""

    def test_mlp_glu_pruning(self):
        model = MockVLM()
        prune_model(model, pruning_type="MLP_GLU", pruning_percentage=25, show_progress=False)

        expected = int(INTERMEDIATE * 0.75)
        for layer in model.model.language_model.layers:
            self.assertEqual(layer.mlp.gate_proj.out_features, expected)
            self.assertEqual(layer.mlp.up_proj.out_features, expected)
            self.assertEqual(layer.mlp.down_proj.in_features, expected)
        self.assertEqual(model.config.text_config.intermediate_size, expected)

        # Vision encoder untouched
        for block in model.model.visual.blocks:
            self.assertEqual(block.mlp.linear_fc1.out_features, 64)

        # Pruned model still runs
        out = model(inputs_embeds=torch.randn(1, 3, HIDDEN))
        self.assertEqual(out.shape, (1, 3, HIDDEN))

    def test_mlp_glu_pruning_text_model_config(self):
        model = MockTextModel()
        prune_model(model, pruning_type="MLP_GLU", pruning_percentage=50, show_progress=False)
        self.assertEqual(model.config.intermediate_size, INTERMEDIATE // 2)

    def test_depth_pruning(self):
        model = MockVLM()
        prune_model(model, pruning_type="DEPTH", layer_indices=[0], show_progress=False)

        self.assertEqual(len(model.model.language_model.layers), NUM_LAYERS - 1)
        self.assertEqual(model.config.text_config.num_hidden_layers, NUM_LAYERS - 1)
        self.assertEqual(model.config.text_config.layer_types, LAYER_TYPES[1:])
        self.assertEqual(len(model.model.visual.blocks), 2)

        out = model(inputs_embeds=torch.randn(1, 3, HIDDEN))
        self.assertEqual(out.shape, (1, 3, HIDDEN))

    def test_data_driven_mlp_pruning(self):
        model = MockVLM()
        input_ids = torch.randint(0, VOCAB, (4, 6))
        dataloader = DataLoader(
            [{"input_ids": ids, "attention_mask": torch.ones_like(ids)} for ids in input_ids],
            batch_size=2,
        )
        prune_model_mlp_glu(
            model,
            neuron_selection_method="MAW",
            pruning_percentage=25,
            dataloader=dataloader,
            show_progress=False,
        )

        # Calibration captured activations on every text decoder layer
        norms = get_activation_norms()
        self.assertEqual(sorted(norms), list(range(NUM_LAYERS)))
        self.assertTrue(all(float(v.sum()) > 0 for v in norms.values()))

        expected = int(INTERMEDIATE * 0.75)
        for layer in model.model.language_model.layers:
            self.assertEqual(layer.mlp.gate_proj.out_features, expected)
        self.assertEqual(model.config.text_config.intermediate_size, expected)

    def test_fairness_scores_cover_text_decoder(self):
        model = MockVLM()
        bias_scores = {
            f"{proj}_layer_{i}": torch.rand(INTERMEDIATE)
            for i in range(NUM_LAYERS)
            for proj in ("gate_proj", "up_proj")
        }
        scores = compute_fairness_pruning_scores(model, bias_scores)
        self.assertEqual(sorted(scores), list(range(NUM_LAYERS)))

        prune_model_mlp_glu(
            model, pruning_percentage=25, fairness_scores=scores, show_progress=False
        )
        self.assertEqual(
            model.config.text_config.intermediate_size, int(INTERMEDIATE * 0.75)
        )


class TestVLMBiasHooks(unittest.TestCase):
    """Activation capture covers every decoder layer, including linear attention."""

    def test_hooks_capture_all_decoder_layers(self):
        model = MockVLM()
        handles = register_hooks(model)
        try:
            with torch.no_grad():
                model(inputs_embeds=torch.randn(1, 3, HIDDEN))
            activations = model._optipfair_activations
        finally:
            remove_hooks(handles)

        for prefix in ("attention_output", "mlp_output", "gate_proj", "up_proj", "down_proj"):
            keys = [k for k in activations if k.startswith(f"{prefix}_layer_")]
            self.assertEqual(len(keys), NUM_LAYERS, prefix)

        # Only text decoder layers are hooked
        self.assertFalse(any("visual" in k for k in activations))


class TestBiasPlotsBFloat16(unittest.TestCase):
    """Bias plots work with bfloat16 activations (models loaded in bfloat16 on GPU)."""

    def setUp(self):
        try:
            import matplotlib
            import sklearn  # noqa: F401
        except ImportError:
            self.skipTest("Visualization dependencies not installed")
        matplotlib.use("Agg")

    def _run(self, plot_fn, layer_key):
        act1 = {layer_key: torch.randn(1, 5, HIDDEN, dtype=torch.bfloat16)}
        act2 = {layer_key: torch.randn(1, 5, HIDDEN, dtype=torch.bfloat16)}
        with patch(
            "optipfair.bias.visualization.get_activation_pairs", return_value=(act1, act2)
        ), patch("matplotlib.pyplot.show"):
            tokenizer = MagicMock()
            tokenizer.tokenize.return_value = [f"tok{i}" for i in range(5)]
            plot_fn(None, tokenizer, ("prompt one", "prompt two"), layer_key=layer_key)

    def test_heatmap_bfloat16(self):
        from optipfair.bias.visualization import visualize_heatmap

        self._run(visualize_heatmap, "mlp_output_layer_0")

    def test_pca_bfloat16(self):
        from optipfair.bias.visualization import visualize_pca

        self._run(visualize_pca, "mlp_output_layer_0")


if __name__ == '__main__':
    unittest.main()
