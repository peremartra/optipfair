"""
Tests for the distillation loss module.
"""

import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from optipfair.distillation.loss import compute_distillation_loss


class TestComputeDistillationLoss(unittest.TestCase):

    def setUp(self):
        """Create minimal synthetic tensors for testing."""
        self.batch, self.seq, self.vocab, self.hidden = 2, 8, 100, 32
        self.n_layers = 4

        self.student_logits = torch.randn(self.batch, self.seq, self.vocab)
        self.teacher_logits = torch.randn(self.batch, self.seq, self.vocab)
        self.labels = torch.randint(0, self.vocab, (self.batch, self.seq))

        self.student_hiddens = [
            torch.randn(self.batch, self.seq, self.hidden) for _ in range(self.n_layers)
        ]
        self.teacher_hiddens = [
            torch.randn(self.batch, self.seq, self.hidden) for _ in range(self.n_layers)
        ]
        self.layer_map = list(range(self.n_layers))

    def test_returns_tuple(self):
        """The function should return a tensor and a metrics dictionary."""
        loss, loss_dict = compute_distillation_loss(
            self.student_logits,
            self.teacher_logits,
            None,
            None,
            self.labels,
        )
        self.assertIsInstance(loss, torch.Tensor)
        self.assertIsInstance(loss_dict, dict)

    def test_loss_dict_always_has_five_keys(self):
        """The loss dictionary should always expose the same five components."""
        _, loss_dict = compute_distillation_loss(
            self.student_logits,
            self.teacher_logits,
            None,
            None,
            self.labels,
        )
        self.assertEqual(
            set(loss_dict.keys()),
            {"total", "task", "logits", "trajectory", "derivative"},
        )

    def test_disabled_components_are_zero(self):
        """Trajectory and derivative must be zero when both feature weights are disabled."""
        _, loss_dict = compute_distillation_loss(
            self.student_logits,
            self.teacher_logits,
            None,
            None,
            self.labels,
            gamma=0.0,
            delta=0.0,
        )
        self.assertEqual(loss_dict["trajectory"], 0.0)
        self.assertEqual(loss_dict["derivative"], 0.0)

    def test_total_loss_is_tensor_with_grad(self):
        """The total loss should support backpropagation."""
        self.student_logits.requires_grad_(True)
        loss, _ = compute_distillation_loss(
            self.student_logits,
            self.teacher_logits,
            None,
            None,
            self.labels,
        )
        self.assertTrue(loss.requires_grad)

    def test_feature_alignment_components_nonzero(self):
        """Feature losses should contribute when enabled and valid states are provided."""
        _, loss_dict = compute_distillation_loss(
            self.student_logits,
            self.teacher_logits,
            self.student_hiddens,
            self.teacher_hiddens,
            self.labels,
            layer_map=self.layer_map,
            gamma=0.1,
            delta=0.1,
        )
        self.assertGreater(loss_dict["trajectory"], 0.0)
        self.assertGreater(loss_dict["derivative"], 0.0)

    def test_auto_layer_map_when_none_and_features_active(self):
        """Feature alignment should auto-resolve layer mapping when none is provided."""
        _, loss_dict = compute_distillation_loss(
            self.student_logits,
            self.teacher_logits,
            self.student_hiddens,
            self.teacher_hiddens,
            self.labels,
            layer_map=None,
            gamma=0.1,
            delta=0.1,
        )
        self.assertGreater(loss_dict["trajectory"], 0.0)
        self.assertGreater(loss_dict["derivative"], 0.0)

    def test_loss_dict_values_are_floats(self):
        """All exported loss metrics should be plain Python floats."""
        _, loss_dict = compute_distillation_loss(
            self.student_logits,
            self.teacher_logits,
            None,
            None,
            self.labels,
        )
        for key, value in loss_dict.items():
            self.assertIsInstance(value, float, f"Key '{key}' is not a float")

    def test_default_parameters_sum_to_one(self):
        """Default task and logits weights should sum to one when features are off."""
        import inspect

        sig = inspect.signature(compute_distillation_loss)
        alpha = sig.parameters["alpha"].default
        beta = sig.parameters["beta"].default
        gamma = sig.parameters["gamma"].default
        delta = sig.parameters["delta"].default
        self.assertEqual(gamma, 0.0)
        self.assertEqual(delta, 0.0)
        self.assertAlmostEqual(alpha + beta, 1.0, places=6)


if __name__ == "__main__":
    unittest.main()