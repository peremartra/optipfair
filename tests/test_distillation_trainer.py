"""
Tests for the distillation trainer module.
"""

import inspect
import os
import sys
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import PretrainedConfig

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from optipfair.distillation.loss import compute_distillation_loss
from optipfair.distillation.mapping import MAPPING_LAST, MAPPING_UNIFORM
from optipfair.distillation.trainer import distill_model
from optipfair.pruning.utils import get_text_config


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

    def __init__(
        self, num_layers=4, hidden_size=32, vocab_size=100, nested_config=False
    ):
        super().__init__()
        if nested_config:
            text_config = PretrainedConfig(
                num_hidden_layers=num_layers,
                hidden_size=hidden_size,
            )
            text_config.use_cache = True
            self.config = PretrainedConfig(text_config=text_config)
            self.config.text_config = text_config
        else:
            self.config = MockConfig(num_layers, hidden_size)
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)]
        )
        self.lm_head = nn.Linear(hidden_size, vocab_size)
        self.cache_values = []
        self.forward_modes = []

    def forward(self, input_ids, attention_mask=None, output_hidden_states=False):
        self.forward_modes.append(self.training)
        text_config = get_text_config(self)
        self.cache_values.append(
            (
                getattr(self.config, "use_cache", "<missing>"),
                getattr(text_config, "use_cache", "<missing>"),
            )
        )
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


class MockMalformedHiddenStatesModel(MockSmallModel):
    """Model that returns fewer hidden states than its configured layer count."""

    def forward(self, input_ids, attention_mask=None, output_hidden_states=False):
        outputs = super().forward(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
        )
        if output_hidden_states:
            outputs.hidden_states = outputs.hidden_states[:-1]
        return outputs


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


class DictDatasetWithLabels(torch.utils.data.Dataset):
    """Dataset that yields dict batches including labels."""

    def __init__(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def __len__(self):
        return self.input_ids.size(0)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
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

    def test_explicit_map_must_be_a_list_of_valid_teacher_indices(self):
        invalid_maps = (
            ([0, 1], 0.1, 0.0),
            ([0, 1, 2, 6], 0.1, 0.0),
            ([0, 1, 2, -1], 0.1, 0.0),
            ([0, 1, True, 3], 0.1, 0.0),
            ((0, 1, 2, 3), 0.1, 0.0),
            ([0, 2, 1, 3], 0.1, 0.1),
        )

        for layer_map, gamma, delta in invalid_maps:
            with self.subTest(layer_map=layer_map, delta=delta):
                with self.assertRaises(ValueError):
                    distill_model(
                        self.student,
                        self.teacher,
                        self.dataloader,
                        layer_map=layer_map,
                        gamma=gamma,
                        delta=delta,
                    )

    def test_explicit_map_is_keyword_only(self):
        parameter = inspect.signature(distill_model).parameters["layer_map"]
        self.assertEqual(parameter.kind, inspect.Parameter.KEYWORD_ONLY)

    def test_hidden_state_count_mismatch_fails_before_loss(self):
        student = MockMalformedHiddenStatesModel(num_layers=4)
        teacher = MockSmallModel(num_layers=6)

        with patch(
            "optipfair.distillation.trainer.compute_distillation_loss",
            wraps=compute_distillation_loss,
        ) as mocked_loss:
            with self.assertRaisesRegex(
                ValueError,
                r"student returned 3 \(expected 4\), teacher returned 6 \(expected 6\)",
            ):
                distill_model(
                    student,
                    teacher,
                    make_tuple_dataloader(num_batches=1),
                    gamma=0.1,
                    epochs=1,
                    scheduler="none",
                    show_progress=False,
                )

        mocked_loss.assert_not_called()

    def test_raises_if_invalid_scheduler(self):
        with self.assertRaises(ValueError):
            distill_model(
                self.student,
                self.teacher,
                self.dataloader,
                scheduler="linear",
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

    def test_runs_with_scheduler_none(self):
        _, stats = distill_model(
            self.student,
            self.teacher,
            self.dataloader,
            scheduler="none",
            epochs=1,
            show_progress=False,
            return_stats=True,
        )
        self.assertEqual(stats["epochs"], 1)

    def test_runs_with_scheduler_cosine(self):
        # num_batches=8, accumulation_steps=2, warmup_ratio=0.25:
        # total_steps=4, warmup_steps=1, cosine_steps=3 → exercises SequentialLR path.
        dataloader = make_tuple_dataloader(num_batches=8)
        _, stats = distill_model(
            self.student,
            self.teacher,
            dataloader,
            scheduler="cosine",
            warmup_ratio=0.25,
            accumulation_steps=2,
            epochs=1,
            show_progress=False,
            return_stats=True,
        )
        self.assertEqual(stats["epochs"], 1)

    def test_scheduler_cosine_steps_both_blocks(self):
        # num_batches=5, accumulation_steps=2, warmup_ratio=0.5:
        # total_steps=2, warmup_steps=1, cosine_steps=1 → SequentialLR.
        # Main block fires at batch 1 and 3 (2 optimizer steps).
        # Remainder block fires at batch 4 (1 optimizer step).
        # Both scheduler.step() call sites are exercised.
        dataloader = make_tuple_dataloader(num_batches=5)
        _, stats = distill_model(
            self.student,
            self.teacher,
            dataloader,
            scheduler="cosine",
            warmup_ratio=0.5,
            accumulation_steps=2,
            epochs=1,
            show_progress=False,
            return_stats=True,
        )
        self.assertEqual(stats["epochs"], 1)


class TestDistillModelCacheHandling(unittest.TestCase):
    def test_existing_flat_cache_values_are_restored(self):
        student = MockSmallModel(num_layers=4)
        teacher = MockSmallModel(num_layers=6)
        student.config.use_cache = True
        teacher.config.use_cache = False

        distill_model(
            student,
            teacher,
            make_tuple_dataloader(),
            epochs=1,
            accumulation_steps=1,
            show_progress=False,
        )

        self.assertTrue(
            all(root is False and text is False for root, text in student.cache_values)
        )
        self.assertTrue(
            all(root is False and text is False for root, text in teacher.cache_values)
        )
        self.assertIs(student.config.use_cache, True)
        self.assertIs(teacher.config.use_cache, False)

    def test_optimizer_uses_only_trainable_parameters(self):
        student = MockSmallModel(num_layers=4)
        teacher = MockSmallModel(num_layers=6)
        frozen_parameter = student.layers[0].weight
        frozen_parameter.requires_grad_(False)
        frozen_before = frozen_parameter.detach().clone()
        trainable_parameters = [
            parameter for parameter in student.parameters() if parameter.requires_grad
        ]
        optimizer_class = torch.optim.AdamW

        with patch(
            "optipfair.distillation.trainer.AdamW",
            wraps=optimizer_class,
        ) as mocked_optimizer:
            distill_model(
                student,
                teacher,
                make_tuple_dataloader(),
                epochs=1,
                accumulation_steps=1,
                show_progress=False,
            )

        optimizer_parameters = mocked_optimizer.call_args.args[0]
        self.assertEqual(optimizer_parameters, trainable_parameters)
        torch.testing.assert_close(frozen_parameter, frozen_before)

    def test_requires_at_least_one_trainable_student_parameter(self):
        student = MockSmallModel(num_layers=4)
        teacher = MockSmallModel(num_layers=6)
        for parameter in student.parameters():
            parameter.requires_grad_(False)

        with self.assertRaisesRegex(ValueError, "at least one trainable parameter"):
            distill_model(
                student,
                teacher,
                make_tuple_dataloader(),
                epochs=1,
                accumulation_steps=1,
                show_progress=False,
            )

    def test_nested_cache_is_disabled_and_restored(self):
        student = MockSmallModel(num_layers=4, nested_config=True)
        teacher = MockSmallModel(num_layers=6, nested_config=True)
        dataloader = make_tuple_dataloader()
        student_text_config = get_text_config(student)
        teacher_text_config = get_text_config(teacher)

        distill_model(
            student,
            teacher,
            dataloader,
            epochs=1,
            accumulation_steps=1,
            show_progress=False,
        )

        self.assertTrue(student.cache_values)
        self.assertTrue(teacher.cache_values)
        self.assertTrue(
            all(root is False and text is False for root, text in student.cache_values)
        )
        self.assertTrue(
            all(root is False and text is False for root, text in teacher.cache_values)
        )
        self.assertFalse(hasattr(student.config, "use_cache"))
        self.assertFalse(hasattr(teacher.config, "use_cache"))
        self.assertIs(student_text_config.use_cache, True)
        self.assertIs(teacher_text_config.use_cache, True)

    def test_cache_is_restored_when_training_raises(self):
        student = MockSmallModel(num_layers=4, nested_config=True)
        teacher = MockSmallModel(num_layers=6, nested_config=True)
        student_text_config = get_text_config(student)
        teacher_text_config = get_text_config(teacher)

        with patch(
            "optipfair.distillation.trainer.compute_distillation_loss",
            side_effect=RuntimeError("loss failed"),
        ):
            with self.assertRaisesRegex(RuntimeError, "loss failed"):
                distill_model(
                    student,
                    teacher,
                    make_tuple_dataloader(),
                    epochs=1,
                    accumulation_steps=1,
                    show_progress=False,
                )

        self.assertFalse(hasattr(student.config, "use_cache"))
        self.assertFalse(hasattr(teacher.config, "use_cache"))
        self.assertIs(student_text_config.use_cache, True)
        self.assertIs(teacher_text_config.use_cache, True)


class TestDistillModelStepCallback(unittest.TestCase):
    def setUp(self):
        self.student = MockSmallModel(num_layers=4)
        self.teacher = MockSmallModel(num_layers=6)
        self.dataloader = make_tuple_dataloader(num_batches=3)

    def test_callback_interval_and_step_payload(self):
        calls = []

        def callback(info):
            calls.append(info)
            self.assertIs(info["model"], self.student)
            self.assertEqual(
                set(info["losses"]),
                {"total", "task", "logits", "trajectory", "derivative"},
            )
            self.assertIsInstance(info["learning_rate"], float)
            self.assertTrue(all(parameter.grad is None for parameter in self.student.parameters()))

        _, stats = distill_model(
            self.student,
            self.teacher,
            self.dataloader,
            epochs=2,
            accumulation_steps=2,
            callback_every_steps=2,
            step_callback=callback,
            show_progress=False,
            return_stats=True,
        )

        self.assertEqual([call["global_step"] for call in calls], [2, 4])
        self.assertEqual([call["total_steps"] for call in calls], [4, 4])
        self.assertEqual([call["epoch"] for call in calls], [0, 1])
        self.assertFalse(stats["stopped_early"])
        self.assertEqual(stats["global_steps_done"], 4)

    def test_callback_defaults_to_every_step_and_restores_train_mode(self):
        calls = []

        def callback(info):
            calls.append(info["global_step"])
            info["model"].eval()

        _, stats = distill_model(
            self.student,
            self.teacher,
            self.dataloader,
            epochs=1,
            accumulation_steps=1,
            step_callback=callback,
            show_progress=False,
            return_stats=True,
        )

        self.assertEqual(calls, [1, 2, 3])
        self.assertTrue(all(self.student.forward_modes))
        self.assertTrue(self.student.training)
        self.assertEqual(stats["global_steps_done"], 3)

    def test_callback_stops_after_requested_step(self):
        calls = []

        def callback(info):
            calls.append(info["global_step"])
            return info["global_step"] == 2

        _, stats = distill_model(
            self.student,
            self.teacher,
            self.dataloader,
            epochs=3,
            accumulation_steps=1,
            step_callback=callback,
            show_progress=False,
            return_stats=True,
        )

        self.assertEqual(calls, [1, 2])
        self.assertTrue(stats["stopped_early"])
        self.assertEqual(stats["global_steps_done"], 2)

    def test_callback_can_capture_reloadable_student_weights(self):
        saved_state = {}

        def callback(info):
            saved_state.update(
                {
                    name: tensor.detach().clone()
                    for name, tensor in info["model"].state_dict().items()
                }
            )

        distill_model(
            self.student,
            self.teacher,
            self.dataloader,
            epochs=1,
            accumulation_steps=1,
            step_callback=callback,
            show_progress=False,
        )

        reloaded_student = MockSmallModel(num_layers=4)
        reloaded_student.load_state_dict(saved_state)
        for name, tensor in reloaded_student.state_dict().items():
            torch.testing.assert_close(tensor, self.student.state_dict()[name])

    def test_callback_validation(self):
        with self.assertRaisesRegex(ValueError, "requires step_callback"):
            distill_model(
                self.student,
                self.teacher,
                self.dataloader,
                callback_every_steps=1,
            )

        for invalid_interval in (0, -1, 1.5, True):
            with self.subTest(interval=invalid_interval):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    distill_model(
                        self.student,
                        self.teacher,
                        self.dataloader,
                        step_callback=lambda _: None,
                        callback_every_steps=invalid_interval,
                    )

    def test_callback_exception_propagates_and_restores_cache(self):
        student = MockSmallModel(num_layers=4, nested_config=True)
        teacher = MockSmallModel(num_layers=6, nested_config=True)
        student_text_config = get_text_config(student)

        def callback(_):
            raise RuntimeError("callback failed")

        with self.assertRaisesRegex(RuntimeError, "callback failed"):
            distill_model(
                student,
                teacher,
                self.dataloader,
                epochs=1,
                accumulation_steps=1,
                step_callback=callback,
                show_progress=False,
            )

        self.assertFalse(hasattr(student.config, "use_cache"))
        self.assertIs(student_text_config.use_cache, True)


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


class TestDistillModelLabelMasking(unittest.TestCase):

    def setUp(self):
        self.student = MockSmallModel(num_layers=4)
        self.teacher = MockSmallModel(num_layers=6)

    @staticmethod
    def _fake_loss(**kwargs):
        loss = kwargs["student_logits"].sum() * 0 + 1.0
        loss_dict = {
            "total": 1.0,
            "task": 1.0,
            "logits": 1.0,
            "trajectory": 0.0,
            "derivative": 0.0,
        }
        return loss, loss_dict

    def test_respects_user_provided_labels(self):
        input_ids = torch.tensor([[10, 11, 12, 13]])
        attention_mask = torch.tensor([[1, 1, 0, 0]])
        labels = torch.tensor([[90, 91, -100, -100]])
        dataloader = DataLoader(
            DictDatasetWithLabels(input_ids, attention_mask, labels),
            batch_size=1,
        )

        with patch(
            "optipfair.distillation.trainer.compute_distillation_loss",
            side_effect=self._fake_loss,
        ) as mocked_loss:
            distill_model(
                self.student,
                self.teacher,
                dataloader,
                epochs=1,
                show_progress=False,
                accumulation_steps=1,
                return_stats=False,
            )

        passed_labels = mocked_loss.call_args.kwargs["labels"]
        self.assertTrue(torch.equal(passed_labels, labels))

    def test_builds_masked_labels_from_attention_mask(self):
        input_ids = torch.tensor([[20, 21, 22, 23]])
        attention_mask = torch.tensor([[1, 1, 0, 0]])
        dataloader = DataLoader(
            DictDataset(input_ids, attention_mask),
            batch_size=1,
        )

        with patch(
            "optipfair.distillation.trainer.compute_distillation_loss",
            side_effect=self._fake_loss,
        ) as mocked_loss:
            distill_model(
                self.student,
                self.teacher,
                dataloader,
                epochs=1,
                show_progress=False,
                accumulation_steps=1,
                return_stats=False,
            )

        passed_labels = mocked_loss.call_args.kwargs["labels"]
        expected_labels = torch.tensor([[20, 21, -100, -100]])
        self.assertTrue(torch.equal(passed_labels.cpu(), expected_labels))


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

    def test_nested_text_config_maps_layers_in_distillation(self):
        student = MockSmallModel(num_layers=4, nested_config=True)
        teacher = MockSmallModel(num_layers=6, nested_config=True)
        _, stats = distill_model(
            student,
            teacher,
            self.dataloader,
            layer_mapping_strategy=MAPPING_UNIFORM,
            gamma=0.1,
            epochs=1,
            show_progress=False,
            return_stats=True,
        )

        self.assertEqual(stats["layer_map_used"], [0, 1, 3, 4])


class TestDistillModelExplicitLayerMap(unittest.TestCase):

    def test_non_contiguous_map_reaches_delta_loss_and_stats(self):
        student = MockSmallModel(num_layers=3)
        teacher = MockSmallModel(num_layers=5)
        layer_map = [0, 2, 3]

        with patch(
            "optipfair.distillation.trainer.compute_distillation_loss",
            wraps=compute_distillation_loss,
        ) as mocked_loss:
            _, stats = distill_model(
                student,
                teacher,
                make_tuple_dataloader(num_batches=1),
                layer_mapping_strategy="ignored-by-explicit-map",
                layer_map=layer_map,
                gamma=0.1,
                delta=0.1,
                alpha=0.4,
                beta=0.4,
                epochs=1,
                accumulation_steps=1,
                scheduler="none",
                show_progress=False,
                return_stats=True,
            )

        self.assertEqual(mocked_loss.call_args.kwargs["layer_map"], layer_map)
        self.assertEqual(stats["layer_mapping_strategy"], "explicit")
        self.assertEqual(stats["layer_map_used"], layer_map)

    def test_repeated_and_unordered_map_is_allowed_without_delta(self):
        student = MockSmallModel(num_layers=4)
        teacher = MockSmallModel(num_layers=6)
        layer_map = [4, 2, 2, 0]

        _, stats = distill_model(
            student,
            teacher,
            make_tuple_dataloader(num_batches=1),
            layer_map=layer_map,
            gamma=0.1,
            epochs=1,
            accumulation_steps=1,
            scheduler="none",
            show_progress=False,
            return_stats=True,
        )

        self.assertEqual(stats["layer_map_used"], layer_map)

    def test_map_without_feature_losses_warns_and_is_not_reported_as_used(self):
        student = MockSmallModel(num_layers=4)
        teacher = MockSmallModel(num_layers=6)
        with self.assertWarnsRegex(UserWarning, "will not be used"):
            _, stats = distill_model(
                student,
                teacher,
                make_tuple_dataloader(num_batches=1),
                layer_map=[0, 1, 2, 3],
                epochs=1,
                scheduler="none",
                show_progress=False,
                return_stats=True,
            )

        self.assertIsNone(stats["layer_map_used"])
        self.assertIsNone(stats["layer_mapping_strategy"])

    def test_invalid_map_is_rejected_without_feature_losses(self):
        student = MockSmallModel(num_layers=4)
        teacher = MockSmallModel(num_layers=6)

        with self.assertRaisesRegex(ValueError, "one index per student layer"):
            distill_model(
                student,
                teacher,
                make_tuple_dataloader(num_batches=1),
                layer_map=[0, 1],
                epochs=1,
                scheduler="none",
                show_progress=False,
            )


if __name__ == "__main__":
    unittest.main()
