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

    def test_masked_positions_do_not_change_loss_components(self):
        """Loss components should be invariant to changes on masked token positions."""
        batch, seq, vocab, hidden = 1, 6, 32, 8
        n_layers = 3
        layer_map = list(range(n_layers))

        torch.manual_seed(42)
        student_logits_a = torch.randn(batch, seq, vocab)
        teacher_logits_a = torch.randn(batch, seq, vocab)
        student_hiddens_a = [torch.randn(batch, seq, hidden) for _ in range(n_layers)]
        teacher_hiddens_a = [torch.randn(batch, seq, hidden) for _ in range(n_layers)]

        student_logits_b = student_logits_a.clone()
        teacher_logits_b = teacher_logits_a.clone()
        student_hiddens_b = [h.clone() for h in student_hiddens_a]
        teacher_hiddens_b = [h.clone() for h in teacher_hiddens_a]

        # Masked labels at positions 3, 4, 5 map to shifted positions 2, 3, 4.
        labels = torch.tensor([[5, 6, 7, -100, -100, -100]])

        student_logits_b[:, 2:5, :] = torch.randn(batch, 3, vocab)
        teacher_logits_b[:, 2:5, :] = torch.randn(batch, 3, vocab)
        for idx in range(n_layers):
            student_hiddens_b[idx][:, 2:5, :] = torch.randn(batch, 3, hidden)
            teacher_hiddens_b[idx][:, 2:5, :] = torch.randn(batch, 3, hidden)

        _, loss_a = compute_distillation_loss(
            student_logits_a,
            teacher_logits_a,
            student_hiddens_a,
            teacher_hiddens_a,
            labels,
            layer_map=layer_map,
            alpha=0.25,
            beta=0.25,
            gamma=0.25,
            delta=0.25,
        )
        _, loss_b = compute_distillation_loss(
            student_logits_b,
            teacher_logits_b,
            student_hiddens_b,
            teacher_hiddens_b,
            labels,
            layer_map=layer_map,
            alpha=0.25,
            beta=0.25,
            gamma=0.25,
            delta=0.25,
        )

        self.assertAlmostEqual(loss_a["task"], loss_b["task"], places=6)
        self.assertAlmostEqual(loss_a["logits"], loss_b["logits"], places=6)
        self.assertAlmostEqual(loss_a["trajectory"], loss_b["trajectory"], places=6)
        self.assertAlmostEqual(loss_a["derivative"], loss_b["derivative"], places=6)

    def test_selected_logits_loss_matches_masked_reference_and_gradients(self):
        torch.manual_seed(19)
        student_logits = torch.randn(2, 7, 64, requires_grad=True)
        teacher_logits = torch.randn(2, 7, 64)
        labels = torch.tensor(
            [
                [-100, -100, 4, 5, 6, -100, -100],
                [-100, -100, -100, 7, 8, 9, -100],
            ]
        )
        temperature = 1.7
        skew_alpha = 0.35

        loss, loss_dict = compute_distillation_loss(
            student_logits,
            teacher_logits,
            None,
            None,
            labels,
            alpha=0.0,
            beta=1.0,
            temperature=temperature,
            skew_alpha=skew_alpha,
        )
        loss.backward()
        actual_gradient = student_logits.grad.clone()

        reference_student = student_logits.detach().clone().requires_grad_(True)
        valid_mask = labels[..., 1:] != -100
        teacher_probs = torch.softmax(
            teacher_logits[..., :-1, :] / temperature, dim=-1
        ).detach()
        student_probs = torch.softmax(
            reference_student[..., :-1, :] / temperature, dim=-1
        )
        mixed_probs = skew_alpha * student_probs + (1.0 - skew_alpha) * teacher_probs
        kl_per_token = (
            teacher_probs
            * (
                torch.log(teacher_probs + 1e-9)
                - torch.log(mixed_probs + 1e-9)
            )
        ).sum(dim=-1)
        reference_loss = (
            (kl_per_token * valid_mask).sum() / valid_mask.sum()
        ) * temperature**2
        reference_loss.backward()

        self.assertAlmostEqual(loss_dict["logits"], reference_loss.item(), places=6)
        torch.testing.assert_close(actual_gradient, reference_student.grad, rtol=1e-5, atol=1e-7)

    def test_equal_logits_with_large_vocabulary_have_near_zero_loss(self):
        torch.manual_seed(53)
        logits = torch.randn(1, 2, 248320)
        labels = torch.tensor([[0, 1]])

        for temperature in (1.0, 2.0):
            with self.subTest(temperature=temperature):
                _, loss_dict = compute_distillation_loss(
                    logits,
                    logits,
                    None,
                    None,
                    labels,
                    alpha=0.0,
                    beta=1.0,
                    temperature=temperature,
                )
                self.assertGreaterEqual(loss_dict["logits"], -5e-7)
                self.assertLessEqual(abs(loss_dict["logits"]), 5e-7)

    def test_large_vocabulary_loss_and_gradient_match_symmetric_reference(self):
        torch.manual_seed(53)
        student = torch.randn(1, 3, 248320, requires_grad=True)
        teacher = torch.randn(1, 3, 248320, requires_grad=True)
        labels = torch.tensor([[0, 1, -100]])
        temperature = 2.0
        skew_alpha = 0.4

        loss, loss_dict = compute_distillation_loss(
            student,
            teacher,
            None,
            None,
            labels,
            alpha=0.0,
            beta=1.0,
            temperature=temperature,
            skew_alpha=skew_alpha,
        )
        loss.backward()

        reference_student = student.detach().clone().requires_grad_(True)
        teacher_probs = torch.softmax(teacher.detach()[:, 0, :] / temperature, dim=-1)
        student_probs = torch.softmax(reference_student[:, 0, :] / temperature, dim=-1)
        mixed_probs = skew_alpha * student_probs + (1.0 - skew_alpha) * teacher_probs
        reference = (
            teacher_probs * (torch.log(teacher_probs + 1e-9) - torch.log(mixed_probs + 1e-9))
        ).sum() * temperature**2
        reference.backward()

        self.assertAlmostEqual(loss_dict["logits"], reference.item(), delta=1e-6)
        self.assertIsNone(teacher.grad)
        self.assertGreater(student.grad.norm().item(), 0.0)
        torch.testing.assert_close(student.grad, reference_student.grad, rtol=1e-5, atol=1e-9)

    def test_logits_loss_does_not_backpropagate_to_teacher(self):
        torch.manual_seed(53)
        student = torch.randn(1, 3, 32, requires_grad=True)
        teacher = torch.randn(1, 3, 32, requires_grad=True)
        labels = torch.tensor([[0, 1, 2]])

        loss, _ = compute_distillation_loss(
            student, teacher, None, None, labels, alpha=0.0, beta=1.0
        )
        loss.backward()

        self.assertIsNone(teacher.grad)
        self.assertGreater(student.grad.norm().item(), 0.0)

    def test_logits_loss_handles_batches_without_valid_tokens(self):
        student_logits = torch.randn(1, 4, 16, requires_grad=True)
        teacher_logits = torch.randn(1, 4, 16)
        labels = torch.full((1, 4), -100)

        loss, loss_dict = compute_distillation_loss(
            student_logits,
            teacher_logits,
            None,
            None,
            labels,
            alpha=0.0,
            beta=1.0,
        )
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(loss_dict["logits"], 0.0)
        self.assertTrue(torch.isfinite(student_logits.grad).all())

    def test_all_masked_float16_large_logits_do_not_overflow(self):
        labels = torch.full((1, 4), -100)
        for alpha, beta in ((0.0, 1.0), (1.0, 0.0)):
            with self.subTest(alpha=alpha, beta=beta):
                student_logits = torch.ones(1, 4, 65536, dtype=torch.float16, requires_grad=True)
                teacher_logits = torch.ones_like(student_logits)

                loss, loss_dict = compute_distillation_loss(
                    student_logits,
                    teacher_logits,
                    None,
                    None,
                    labels,
                    alpha=alpha,
                    beta=beta,
                )
                loss.backward()

                self.assertEqual(loss.item(), 0.0)
                self.assertEqual(loss_dict["task"], 0.0)
                self.assertEqual(loss_dict["logits"], 0.0)
                torch.testing.assert_close(student_logits.grad, torch.zeros_like(student_logits))


if __name__ == "__main__":
    unittest.main()