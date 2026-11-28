"""
End-to-end checks of the OptiPFair pipeline on Apple Silicon (MPS).

A tiny randomly initialised Llama model is built from a config (no download), then the full
analyze -> depth prune -> width prune (data-driven) -> distill pipeline runs on MPS. Pruning
decisions on MPS are compared against the same pipeline on CPU so backend-specific numerical
issues surface as test failures rather than silently different models.

The whole module is skipped when the MPS backend is unavailable (Linux, Windows, Intel Macs, CI).
"""

import copy
import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from optipfair import analyze_layer_importance, distill_model, prune_model
from optipfair.device import get_device, is_mps_available

try:
    from transformers import LlamaConfig, LlamaForCausalLM
except ImportError:  # pragma: no cover - transformers is a hard dependency
    LlamaForCausalLM = None

VOCAB_SIZE = 128


def _tiny_llama():
    config = LlamaConfig(
        vocab_size=VOCAB_SIZE,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        tie_word_embeddings=False,
        # GQA + SDPA without an attention mask uses enable_gqa=True, which aborts on MPS with
        # torch 2.6 ("mps.matmul ... Failed to infer result type"). Eager keeps GQA coverage.
        attn_implementation="eager",
    )
    torch.manual_seed(0)
    return LlamaForCausalLM(config).eval()


def _batches(num_batches=4, batch_size=2, seq_len=12, seed=1):
    """Dict batches with right padding on the second row of every batch."""
    generator = torch.Generator().manual_seed(seed)
    batches = []
    for _ in range(num_batches):
        input_ids = torch.randint(1, VOCAB_SIZE, (batch_size, seq_len), generator=generator)
        attention_mask = torch.ones_like(input_ids)
        attention_mask[1, seq_len // 2:] = 0
        batches.append({'input_ids': input_ids, 'attention_mask': attention_mask})
    return batches


@unittest.skipUnless(is_mps_available(), "requires Apple Silicon MPS backend")
@unittest.skipIf(LlamaForCausalLM is None, "requires transformers")
class TestPipelineOnMPS(unittest.TestCase):

    def setUp(self):
        self.mps = get_device("mps")
        self.cpu_model = _tiny_llama()
        self.mps_model = copy.deepcopy(self.cpu_model).to(self.mps)
        self.batches = _batches()

    def test_layer_importance_matches_cpu(self):
        cpu_scores = analyze_layer_importance(self.cpu_model, self.batches, show_progress=False)
        mps_scores = analyze_layer_importance(self.mps_model, self.batches, show_progress=False)
        self.assertEqual(cpu_scores.keys(), mps_scores.keys())
        for idx in cpu_scores:
            self.assertAlmostEqual(cpu_scores[idx], mps_scores[idx], places=3)

    def test_depth_pruning_on_mps(self):
        pruned = prune_model(
            self.mps_model, pruning_type="DEPTH", layer_indices=[1, 2], show_progress=False
        )
        self.assertEqual(len(pruned.model.layers), 2)
        self.assertEqual(next(pruned.parameters()).device.type, "mps")
        with torch.no_grad():
            logits = pruned(**{k: v.to(self.mps) for k, v in self.batches[0].items()}).logits
        self.assertTrue(torch.isfinite(logits).all())

    def _width_prune(self, model, dataloader):
        return prune_model(
            model,
            pruning_type="MLP_GLU",
            neuron_selection_method="MAW",
            pruning_percentage=25,
            dataloader=dataloader,
            show_progress=False,
        )

    def test_static_width_pruning_matches_cpu(self):
        cpu_pruned = self._width_prune(self.cpu_model, None)
        mps_pruned = self._width_prune(self.mps_model, None)
        for cpu_layer, mps_layer in zip(cpu_pruned.model.layers, mps_pruned.model.layers):
            self.assertEqual(cpu_layer.mlp.gate_proj.out_features, 96)
            self.assertTrue(
                torch.allclose(cpu_layer.mlp.gate_proj.weight, mps_layer.mlp.gate_proj.weight.cpu())
            )

    def test_data_driven_width_pruning_matches_cpu(self):
        cpu_pruned = self._width_prune(self.cpu_model, self.batches)
        mps_pruned = self._width_prune(self.mps_model, self.batches)
        self.assertEqual(next(mps_pruned.parameters()).device.type, "mps")
        for cpu_layer, mps_layer in zip(cpu_pruned.model.layers, mps_pruned.model.layers):
            self.assertEqual(
                cpu_layer.mlp.down_proj.in_features, mps_layer.mlp.down_proj.in_features
            )
            # Same neurons kept => identical surviving weights.
            self.assertTrue(
                torch.allclose(cpu_layer.mlp.up_proj.weight, mps_layer.mlp.up_proj.weight.cpu())
            )

    def test_distillation_on_mps(self):
        teacher = self.mps_model
        student = prune_model(
            copy.deepcopy(teacher), pruning_type="DEPTH", layer_indices=[2], show_progress=False
        )
        student.train()
        before = student.lm_head.weight.detach().clone()

        trained, stats = distill_model(
            student_model=student,
            teacher_model=teacher,
            dataloader=self.batches,
            gamma=0.1,
            delta=0.1,
            alpha=0.5,
            beta=0.3,
            epochs=1,
            accumulation_steps=2,
            show_progress=False,
            return_stats=True,
        )

        self.assertEqual(next(trained.parameters()).device.type, "mps")
        self.assertTrue(all(torch.isfinite(torch.tensor(stats["loss_history"]["total"]))))
        self.assertFalse(torch.equal(before, trained.lm_head.weight.detach()))

    def test_generation_after_pruning(self):
        pruned = prune_model(
            self.mps_model, pruning_type="MLP_GLU", pruning_percentage=25, show_progress=False
        )
        input_ids = self.batches[0]['input_ids'][:1].to(self.mps)
        output = pruned.generate(input_ids, max_new_tokens=5, do_sample=False, pad_token_id=0)
        self.assertEqual(output.shape[1], input_ids.shape[1] + 5)


if __name__ == "__main__":
    unittest.main()
