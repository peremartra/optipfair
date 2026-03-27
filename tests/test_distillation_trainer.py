"""
Tests for the distillation trainer module.
"""

import os
import sys
import unittest

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from optipfair.distillation.mapping import MAPPING_LAST, MAPPING_UNIFORM
from optipfair.distillation.trainer import distill_model


class MockConfig:
    def __init__(self, num_hidden_layers=4, hidden_size=32):
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size


class MockModelOutput:
    def __init__(self, logits, hidden_states=None):
        self.logits = logits
        self.hidden_states = hidden_states


class MockSmallModel(nn.Module):
    """Minimal mock model that returns logits and optional hidden states."""

    def __init__(self, num_layers=4, hidden_size=32, vocab_size=100):
        super().__init__()
        self.config = MockConfig(num_layers, hidden_size)
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)]
        )
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids, attention_mask=None, output_hidden_states=False):
        x = self.embedding(input_ids)
        hidden_states = []

        for layer in self.layers:
            x = torch.tanh(layer(x))
            if output_hidden_states:
                hidden_states.append(x)

        logits = self.lm_head(x)
        all_hidden_states = (
            tuple([self.embedding(input_ids)] + hidden_states)
            if output_hidden_states
            else None
        )
        return MockModelOutput(logits=logits, hidden_states=all_hidden_states)


class DictDataset(torch.utils.data.Dataset):
    """Dataset that yields dict batches (HuggingFace style)."""

    def __init__(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        self.input_ids = input_ids
        self.attention_mask = attention_mask

    def __len__(self):
        return self.input_ids.size(0)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
        }


def make_tuple_dataloader(batch_size=2, seq_len=8, vocab_size=100, num_batches=3):
    input_ids = torch.randint(0, vocab_size, (batch_size * num_batches, seq_len))
    attention_mask = torch.ones_like(input_ids)
    dataset = TensorDataset(input_ids, attention_mask)
    return DataLoader(dataset, batch_size=batch_size)


def make_dict_dataloader(batch_size=2, seq_len=8, vocab_size=100, num_batches=3):
    input_ids = torch.randint(0, vocab_size, (batch_size * num_batches, seq_len))
    attention_mask = torch.ones_like(input_ids)
    dataset = DictDataset(input_ids, attention_mask)
    return DataLoader(dataset, batch_size=batch_size)


def make_tensor_dataloader(batch_size=2, seq_len=8, vocab_size=100, num_batches=3):
    tensor_data = torch.randint(0, vocab_size, (batch_size * num_batches, seq_len))
    return DataLoader(tensor_data, batch_size=batch_size)


class TestDistillModelValidation(unittest.TestCase):

    def setUp(self):
        self.student = MockSmallModel(num_layers=4)
        self.teacher = MockSmallModel(num_layers=6)
        self.dataloader = make_tuple_dataloader()

    def test_raises_if_all_weights_zero(self):
        with self.assertRaises(ValueError):
            distill_model(
                self.student,
                self.teacher,
                self.dataloader,
                alpha=0.0,
                beta=0.0,
                gamma=0.0,
                delta=0.0,
            )

    def test_raises_if_same_model_object(self):
        with self.assertRaises(ValueError):
            distill_model(self.student, self.student, self.dataloader)

    def test_raises_if_invalid_strategy(self):
        with self.assertRaises(ValueError):
            distill_model(
                self.student,
                self.teacher,
                self.dataloader,
                layer_mapping_strategy="original_indices",
                gamma=0.1,
            )


class TestDistillModelReturnValues(unittest.TestCase):

    def setUp(self):
        self.student = MockSmallModel(num_layers=4)
        self.teacher = MockSmallModel(num_layers=6)
        self.dataloader = make_tuple_dataloader()

    def test_returns_model_when_return_stats_false(self):
        result = distill_model(
            self.student,
            self.teacher,
            self.dataloader,
            epochs=1,
            show_progress=False,
            return_stats=False,
        )
        self.assertIsInstance(result, nn.Module)

    def test_returns_tuple_when_return_stats_true(self):
        result = distill_model(
            self.student,
            self.teacher,
            self.dataloader,
            epochs=1,
            show_progress=False,
            return_stats=True,
        )
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_stats_dict_has_required_keys(self):
        _, stats = distill_model(
            self.student,
            self.teacher,
            self.dataloader,
            epochs=1,
            show_progress=False,
            return_stats=True,
        )
        required_keys = {
            "epochs",
            "learning_rate",
            "accumulation_steps",
            "effective_batch_size",
            "alpha",
            "beta",
            "gamma",
            "delta",
            "temperature",
            "skew_alpha",
            "layer_mapping_strategy",
            "layer_map_used",
            "loss_history",
            "epoch_times_seconds",
            "total_time_seconds",
            "avg_time_per_epoch",
        }
        self.assertEqual(required_keys, set(stats.keys()))

    def test_loss_history_has_five_keys(self):
        _, stats = distill_model(
            self.student,
            self.teacher,
            self.dataloader,
            epochs=1,
            show_progress=False,
            return_stats=True,
        )
        self.assertEqual(
            set(stats["loss_history"].keys()),
            {"total", "task", "logits", "trajectory", "derivative"},
        )

    def test_layer_map_is_none_when_features_off(self):
        _, stats = distill_model(
            self.student,
            self.teacher,
            self.dataloader,
            gamma=0.0,
            delta=0.0,
            epochs=1,
            show_progress=False,
            return_stats=True,
        )
        self.assertIsNone(stats["layer_map_used"])
        self.assertIsNone(stats["layer_mapping_strategy"])

    def test_epoch_times_length_matches_epochs(self):
        n_epochs = 2
        _, stats = distill_model(
            self.student,
            self.teacher,
            self.dataloader,
            epochs=n_epochs,
            show_progress=False,
            return_stats=True,
        )
        self.assertEqual(len(stats["epoch_times_seconds"]), n_epochs)
        self.assertEqual(len(stats["loss_history"]["total"]), n_epochs)


class TestDistillModelBatchFormats(unittest.TestCase):

    def setUp(self):
        self.student = MockSmallModel(num_layers=4)
        self.teacher = MockSmallModel(num_layers=6)

    def test_supports_dict_batches(self):
        dataloader = make_dict_dataloader()
        _, stats = distill_model(
            self.student,
            self.teacher,
            dataloader,
            epochs=1,
            show_progress=False,
            return_stats=True,
        )
        self.assertEqual(stats["epochs"], 1)

    def test_supports_tuple_batches(self):
        dataloader = make_tuple_dataloader()
        _, stats = distill_model(
            self.student,
            self.teacher,
            dataloader,
            epochs=1,
            show_progress=False,
            return_stats=True,
        )
        self.assertEqual(stats["epochs"], 1)

    def test_supports_tensor_batches(self):
        dataloader = make_tensor_dataloader()
        _, stats = distill_model(
            self.student,
            self.teacher,
            dataloader,
            epochs=1,
            show_progress=False,
            return_stats=True,
        )
        self.assertEqual(stats["epochs"], 1)


class TestDistillModelLayerStrategy(unittest.TestCase):

    def setUp(self):
        self.student = MockSmallModel(num_layers=4)
        self.teacher = MockSmallModel(num_layers=6)
        self.dataloader = make_tuple_dataloader()

    def test_uniform_strategy_records_map(self):
        _, stats = distill_model(
            self.student,
            self.teacher,
            self.dataloader,
            layer_mapping_strategy=MAPPING_UNIFORM,
            gamma=0.1,
            epochs=1,
            show_progress=False,
            return_stats=True,
        )
        self.assertEqual(stats["layer_mapping_strategy"], MAPPING_UNIFORM)
        self.assertIsInstance(stats["layer_map_used"], list)

    def test_last_strategy_records_map(self):
        _, stats = distill_model(
            self.student,
            self.teacher,
            self.dataloader,
            layer_mapping_strategy=MAPPING_LAST,
            gamma=0.1,
            epochs=1,
            show_progress=False,
            return_stats=True,
        )
        self.assertEqual(stats["layer_mapping_strategy"], MAPPING_LAST)
        self.assertIsInstance(stats["layer_map_used"], list)


if __name__ == "__main__":
    unittest.main()
