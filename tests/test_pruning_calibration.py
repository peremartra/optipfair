import unittest
import os
import sys

import torch
from torch import nn

# Use the repository checkout instead of an installed optipfair package.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from optipfair.pruning.depth import analyze_layer_importance
from optipfair.pruning.mlp_glu import (
    get_activation_norms,
    run_calibration_forward_passes,
    setup_mlp_hooks_for_importance,
)


class CalibrationLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.mlp = nn.Module()
        self.mlp.down_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, hidden_states):
        return self.mlp.down_proj(hidden_states)


class CalibrationModel(nn.Module):
    def __init__(self):
        super().__init__()
        hidden_size = 8
        self.embedding = nn.Embedding(32, hidden_size)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(
            [CalibrationLayer(hidden_size) for _ in range(2)]
        )
        self.use_cache_values = []

    def forward(self, input_ids, attention_mask=None, use_cache=None):
        self.use_cache_values.append(use_cache)
        hidden_states = self.embedding(input_ids)
        for layer in self.model.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


class TestPruningCalibration(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model = CalibrationModel()
        self.model.eval()
        self.sequences = [
            torch.tensor([1, 2, 3, 4]),
            torch.tensor([5, 6, 7, 8, 9, 10]),
        ]

    def _collect_activation_norms(self, batches):
        handles = setup_mlp_hooks_for_importance(self.model, torch.device("cpu"))
        try:
            run_calibration_forward_passes(
                self.model, batches, torch.device("cpu"), show_progress=False
            )
            return get_activation_norms()
        finally:
            for handle in handles:
                handle.remove()

    def test_padding_and_batch_partition_do_not_change_activation_norms(self):
        unpadded_batches = [
            {
                "input_ids": sequence.unsqueeze(0),
                "attention_mask": torch.ones_like(sequence).unsqueeze(0),
            }
            for sequence in self.sequences
        ]

        padded_ids = torch.zeros(2, 6, dtype=torch.long)
        padded_mask = torch.zeros(2, 6, dtype=torch.long)
        for index, sequence in enumerate(self.sequences):
            padded_ids[index, : sequence.numel()] = sequence
            padded_mask[index, : sequence.numel()] = 1
        padded_batches = [
            {"input_ids": padded_ids, "attention_mask": padded_mask}
        ]

        norms_unpadded = self._collect_activation_norms(unpadded_batches)
        norms_padded = self._collect_activation_norms(padded_batches)

        self.assertEqual(norms_unpadded.keys(), norms_padded.keys())
        for layer_idx in norms_unpadded:
            torch.testing.assert_close(
                norms_unpadded[layer_idx],
                norms_padded[layer_idx],
                rtol=1e-6,
                atol=1e-6,
            )

    def test_calibration_and_layer_analysis_disable_generation_cache(self):
        batch = {
            "input_ids": torch.tensor([[1, 2, 3, 4]]),
            "attention_mask": torch.ones(1, 4, dtype=torch.long),
        }
        self._collect_activation_norms([batch])
        self.assertTrue(self.model.use_cache_values)
        self.assertTrue(all(value is False for value in self.model.use_cache_values))

        self.model.use_cache_values.clear()
        scores = analyze_layer_importance(
            self.model,
            [batch],
            layers_path="model.layers",
            show_progress=False,
        )
        self.assertEqual(len(scores), 2)
        self.assertTrue(all(value is False for value in self.model.use_cache_values))


if __name__ == "__main__":
    unittest.main()
