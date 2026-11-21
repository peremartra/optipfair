"""
Tests for the vocabulary pruning module.

All fixtures are tiny models / tokenizers built in memory (no network).
"""

import copy
import json
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from click.testing import CliRunner
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaConfig,
    LlamaForCausalLM,
    PreTrainedTokenizerFast,
)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from optipfair import prune_model
from optipfair.cli.commands import cli
from optipfair.pruning.vocab import (
    _BYTE_FALLBACK_TOKENS,
    analyse_vocab_usage,
    build_keep_set,
    build_pruned_tokenizer,
    prune_model_vocab,
    save_vocab_pruned,
)

TRAIN_CORPUS = [
    "The quick brown fox jumps over the lazy dog.",
    "Pack my box with five dozen liquor jugs!",
    "Sphinx of black quartz, judge my vow.",
    "Compute 12 + 35 = 47, then 47 * 2 = 94.",
    "If x = 3 and y = 4, the hypotenuse is 5.",
    "Machine learning models can be pruned to save memory.",
    "Vocabulary pruning removes rarely used tokens.",
] * 20

CALIBRATION = [
    "Compute 12 + 35 = 47, then 47 * 2 = 94.",
    {"text": "If x = 3 and y = 4, the hypotenuse is 5."},
    {"problem": "What is 2 + 2?", "solution": "2 + 2 = 4."},
]

EOS = "<|endoftext|>"
PAD = "<|pad|>"


def make_byte_level_tokenizer():
    """GPT-2 style byte-level BPE tokenizer trained on TRAIN_CORPUS."""
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=500,
        special_tokens=[EOS, PAD],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tok.train_from_iterator(TRAIN_CORPUS, trainer)
    return PreTrainedTokenizerFast(tokenizer_object=tok, eos_token=EOS, pad_token=PAD)


def make_byte_fallback_tokenizer():
    """Llama-2 style BPE: <0xNN> byte fallback, specials inside model.vocab,
    and a TemplateProcessing post-processor that prepends <s> (at a high id)."""
    vocab = {"<unk>": 0}
    for name in _BYTE_FALLBACK_TOKENS:
        vocab[name] = len(vocab)
    for piece in ["a", "b", "c", "x", "y", "ab", "abc", "xy"]:
        vocab[piece] = len(vocab)
    vocab["</s>"] = len(vocab)
    vocab["<s>"] = len(vocab)
    merges = [("a", "b"), ("ab", "c"), ("x", "y")]
    tok = Tokenizer(models.BPE(vocab=vocab, merges=merges, unk_token="<unk>", byte_fallback=True))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.decoder = decoders.Sequence([decoders.ByteFallback(), decoders.Fuse()])
    tok.add_special_tokens(["<unk>", "</s>", "<s>"])
    tok.post_processor = processors.TemplateProcessing(
        single="<s> $A", special_tokens=[("<s>", vocab["<s>"])]
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=tok, bos_token="<s>", eos_token="</s>", unk_token="<unk>"
    )


def make_model(tokenizer, tied=True):
    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=len(tokenizer),
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=64,
        tie_word_embeddings=tied,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    return LlamaForCausalLM(config).eval()


class TestAnalyseAndKeepSet(unittest.TestCase):
    def setUp(self):
        self.tokenizer = make_byte_level_tokenizer()

    def test_analyse_counts_all_row_formats(self):
        usage = analyse_vocab_usage(self.tokenizer, CALIBRATION)
        self.assertEqual(usage["num_documents"], 3)
        self.assertEqual(usage["vocab_size"], len(self.tokenizer))
        ids = self.tokenizer("2 + 2 = 4.", add_special_tokens=False)["input_ids"]
        for tid in ids:
            self.assertIn(str(tid), usage["frequency"])
        self.assertIn(self.tokenizer.eos_token_id, usage["used_ids"])

    def test_keep_set_size_and_always_kept(self):
        usage = analyse_vocab_usage(self.tokenizer, CALIBRATION)
        frequency = {int(k): v for k, v in usage["frequency"].items()}
        base = build_keep_set(self.tokenizer, frequency, keep_top_k=0)
        keep = build_keep_set(self.tokenizer, frequency, keep_top_k=10)

        self.assertGreaterEqual(len(keep), len(base) + 10)
        self.assertEqual(keep, sorted(keep))
        vocab = self.tokenizer.get_vocab()
        for byte_tok in pre_tokenizers.ByteLevel.alphabet():
            self.assertIn(vocab[byte_tok], base)
        self.assertIn(self.tokenizer.eos_token_id, base)
        self.assertIn(self.tokenizer.pad_token_id, base)

    def test_keep_set_prefers_frequent_tokens(self):
        usage = analyse_vocab_usage(self.tokenizer, CALIBRATION)
        frequency = {int(k): v for k, v in usage["frequency"].items()}
        base = set(build_keep_set(self.tokenizer, frequency, keep_top_k=0))
        keep = set(build_keep_set(self.tokenizer, frequency, keep_top_k=5))
        top = sorted((t for t in frequency if t not in base), key=lambda t: (-frequency[t], t))[:5]
        self.assertTrue(set(top) <= keep - base)

    def test_keep_set_closed_over_merges(self):
        usage = analyse_vocab_usage(self.tokenizer, CALIBRATION)
        frequency = {int(k): v for k, v in usage["frequency"].items()}
        keep = set(build_keep_set(self.tokenizer, frequency, keep_top_k=5))
        spec = json.loads(self.tokenizer.backend_tokenizer.to_str())
        id_to_str = {i: s for s, i in spec["model"]["vocab"].items()}
        vocab = spec["model"]["vocab"]
        kept_strs = {id_to_str[i] for i in keep if i in id_to_str}
        for a, b in spec["model"]["merges"]:
            if a + b in kept_strs:
                self.assertIn(vocab[a], keep)
                self.assertIn(vocab[b], keep)

    def test_keep_set_caps_at_vocab(self):
        keep = build_keep_set(self.tokenizer, {}, keep_top_k=10_000)
        self.assertEqual(len(keep), len(self.tokenizer))

    def test_keep_set_rejects_bad_args(self):
        with self.assertRaises(ValueError):
            build_keep_set(self.tokenizer, {}, keep_top_k=-1)
        with self.assertRaises(ValueError):
            build_keep_set(self.tokenizer, {}, keep_top_k=0, extra_buffer_ids=[10**6])

    def test_byte_fallback_tokens_always_kept(self):
        tokenizer = make_byte_fallback_tokenizer()
        keep = build_keep_set(tokenizer, {}, keep_top_k=0)
        vocab = tokenizer.get_vocab()
        for name in _BYTE_FALLBACK_TOKENS:
            self.assertIn(vocab[name], keep)
        self.assertIn(vocab["<s>"], keep)


class TestBuildPrunedTokenizer(unittest.TestCase):
    def test_roundtrip_on_kept_text(self):
        tokenizer = make_byte_level_tokenizer()
        usage = analyse_vocab_usage(tokenizer, CALIBRATION)
        frequency = {int(k): v for k, v in usage["frequency"].items()}
        keep = build_keep_set(tokenizer, frequency, keep_top_k=len(frequency))
        new_tok, new2old = build_pruned_tokenizer(tokenizer, keep)

        self.assertEqual(new_tok.get_vocab_size(), len(keep))
        self.assertEqual(new2old.tolist(), keep)
        text = "Compute 12 + 35 = 47, then 47 * 2 = 94."
        old_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        new_ids = new_tok.encode(text, add_special_tokens=False).ids
        self.assertEqual(new2old[new_ids].tolist(), old_ids)
        self.assertEqual(new_tok.decode(new_ids), text)

    def test_unseen_text_still_encodable(self):
        tokenizer = make_byte_level_tokenizer()
        keep = build_keep_set(tokenizer, {}, keep_top_k=0)
        new_tok, _ = build_pruned_tokenizer(tokenizer, keep)
        text = "Zebras ñandú 🙂"
        self.assertEqual(new_tok.decode(new_tok.encode(text).ids), text)

    def test_merges_consistent(self):
        tokenizer = make_byte_level_tokenizer()
        keep = build_keep_set(tokenizer, {}, keep_top_k=50)
        new_tok, _ = build_pruned_tokenizer(tokenizer, keep)
        spec = json.loads(new_tok.to_str())
        vocab = spec["model"]["vocab"]
        self.assertTrue(spec["model"]["merges"])
        for a, b in spec["model"]["merges"]:
            self.assertIn(a, vocab)
            self.assertIn(b, vocab)
            self.assertIn(a + b, vocab)

    def test_string_form_merges(self):
        tokenizer = make_byte_level_tokenizer()
        spec = json.loads(tokenizer.backend_tokenizer.to_str())
        spec["model"]["merges"] = [" ".join(m) for m in spec["model"]["merges"]]
        legacy = SimpleNamespace(backend_tokenizer=SimpleNamespace(to_str=lambda: json.dumps(spec)))
        keep = build_keep_set(tokenizer, {}, keep_top_k=50)

        new_tok, _ = build_pruned_tokenizer(legacy, keep)
        ref_tok, _ = build_pruned_tokenizer(tokenizer, keep)
        text = "The quick brown fox"
        self.assertEqual(new_tok.encode(text).ids, ref_tok.encode(text).ids)

    def test_byte_fallback_and_post_processor_remapped(self):
        tokenizer = make_byte_fallback_tokenizer()
        frequency = {tid: 1 for tid in tokenizer("ab ab", add_special_tokens=False)["input_ids"]}
        keep = build_keep_set(tokenizer, frequency, keep_top_k=1)
        new_tok, new2old = build_pruned_tokenizer(tokenizer, keep)
        old_vocab = tokenizer.get_vocab()
        new_vocab = new_tok.get_vocab()

        self.assertNotIn("abc", new_vocab)
        enc = new_tok.encode("abc")
        # BOS is prepended with its *new* id.
        self.assertEqual(enc.ids[0], new_vocab["<s>"])
        self.assertNotEqual(new_vocab["<s>"], old_vocab["<s>"])
        # "c" was dropped, so it falls back to its byte token.
        self.assertEqual(enc.tokens[1:], ["ab", "<0x63>"])
        self.assertEqual(new_tok.decode(enc.ids), "abc")
        self.assertEqual(int(new2old[new_vocab["<s>"]]), old_vocab["<s>"])

    def test_missing_special_raises(self):
        tokenizer = make_byte_level_tokenizer()
        keep = [i for i in build_keep_set(tokenizer, {}, 0) if i != tokenizer.eos_token_id]
        with self.assertRaises(ValueError):
            build_pruned_tokenizer(tokenizer, keep)

    def test_non_bpe_tokenizer_rejected(self):
        wl = Tokenizer(models.WordLevel(vocab={"<unk>": 0, "hi": 1}, unk_token="<unk>"))
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=wl, unk_token="<unk>")
        with self.assertRaises(ValueError):
            build_keep_set(tokenizer, {}, keep_top_k=0)
        with self.assertRaises(ValueError):
            build_keep_set(SimpleNamespace(), {}, keep_top_k=0)


class TestPruneModelVocab(unittest.TestCase):
    def _check_logits_preserved(self, tied):
        tokenizer = make_byte_level_tokenizer()
        model = make_model(tokenizer, tied=tied)
        original = copy.deepcopy(model)

        pruned, new_tok, new2old = prune_model_vocab(
            model, tokenizer, CALIBRATION, keep_top_k=20, show_progress=False
        )
        n = len(new2old)
        self.assertIs(pruned, model)
        self.assertLess(n, len(tokenizer))
        self.assertEqual(len(new_tok), n)
        self.assertEqual(pruned.config.vocab_size, n)
        self.assertEqual(pruned.get_input_embeddings().weight.shape[0], n)
        self.assertEqual(pruned.get_output_embeddings().weight.shape[0], n)
        self.assertTrue(
            torch.equal(
                pruned.get_input_embeddings().weight,
                original.get_input_embeddings().weight[new2old],
            )
        )
        if tied:
            self.assertIs(
                pruned.get_output_embeddings().weight, pruned.get_input_embeddings().weight
            )
        else:
            self.assertTrue(
                torch.equal(
                    pruned.get_output_embeddings().weight,
                    original.get_output_embeddings().weight[new2old],
                )
            )

        # Token-id config fields are remapped to the new space.
        self.assertEqual(pruned.config.eos_token_id, new_tok.eos_token_id)
        self.assertEqual(pruned.config.pad_token_id, new_tok.pad_token_id)
        self.assertEqual(int(new2old[new_tok.eos_token_id]), tokenizer.eos_token_id)
        self.assertEqual(new_tok.eos_token, EOS)

        # Same text -> same tokens -> same logits on the kept columns.
        text = "Compute 12 + 35 = 47"
        old_ids = tokenizer(text, return_tensors="pt")["input_ids"]
        new_ids = new_tok(text, return_tensors="pt")["input_ids"]
        self.assertTrue(torch.equal(new2old[new_ids], old_ids))
        with torch.no_grad():
            ref = original(old_ids).logits[..., new2old]
            out = pruned(new_ids).logits
        torch.testing.assert_close(out, ref)

    def test_tied_embeddings(self):
        self._check_logits_preserved(tied=True)

    def test_untied_embeddings(self):
        self._check_logits_preserved(tied=False)

    def test_noop_when_keeping_everything(self):
        tokenizer = make_byte_level_tokenizer()
        model = make_model(tokenizer)
        pruned, new_tok, new2old = prune_model_vocab(
            model, tokenizer, CALIBRATION, keep_top_k=10_000, show_progress=False
        )
        self.assertIs(new_tok, tokenizer)
        self.assertEqual(pruned.get_input_embeddings().weight.shape[0], len(tokenizer))
        self.assertEqual(new2old.tolist(), list(range(len(tokenizer))))

    def test_invalid_inputs(self):
        tokenizer = make_byte_level_tokenizer()
        model = make_model(tokenizer)
        with self.assertRaises(ValueError):
            prune_model_vocab(model, tokenizer, [], keep_top_k=10, show_progress=False)
        with self.assertRaises(ValueError):
            prune_model_vocab(model, tokenizer, CALIBRATION, keep_top_k=-1, show_progress=False)
        with self.assertRaises(ValueError):
            prune_model_vocab(
                model, tokenizer, {"a": [], "b": []}, keep_top_k=1, show_progress=False
            )

    def test_generate_runs(self):
        tokenizer = make_byte_level_tokenizer()
        model = make_model(tokenizer)
        pruned, new_tok, _ = prune_model_vocab(
            model, tokenizer, CALIBRATION, keep_top_k=20, show_progress=False
        )
        ids = new_tok("2 + 2", return_tensors="pt")["input_ids"]
        out = pruned.generate(ids, max_new_tokens=3, do_sample=False)
        self.assertEqual(out.shape[1], ids.shape[1] + 3)
        self.assertLess(int(out.max()), len(new_tok))


class TestPruneModelDispatcher(unittest.TestCase):
    def test_vocab_branch_returns_tokenizer_and_stats(self):
        tokenizer = make_byte_level_tokenizer()
        model = make_model(tokenizer)
        result = prune_model(
            model,
            pruning_type="VOCAB",
            tokenizer=tokenizer,
            calibration_dataset=CALIBRATION,
            keep_top_k=20,
            show_progress=False,
            return_stats=True,
        )
        self.assertEqual(len(result), 3)
        pruned, new_tok, stats = result
        self.assertEqual(stats["original_vocab_size"], len(tokenizer))
        self.assertEqual(stats["pruned_vocab_size"], len(new_tok))
        self.assertGreater(stats["reduction"], 0)

        model2 = make_model(tokenizer)
        pruned2, new_tok2 = prune_model(
            model2,
            pruning_type="VOCAB",
            tokenizer=tokenizer,
            calibration_dataset=CALIBRATION,
            keep_top_k=20,
            show_progress=False,
        )
        self.assertEqual(len(new_tok2), len(new_tok))

    def test_vocab_branch_requires_args(self):
        tokenizer = make_byte_level_tokenizer()
        model = make_model(tokenizer)
        with self.assertRaises(ValueError):
            prune_model(model, pruning_type="VOCAB", calibration_dataset=CALIBRATION, keep_top_k=5)
        with self.assertRaises(ValueError):
            prune_model(model, pruning_type="VOCAB", tokenizer=tokenizer, keep_top_k=5)


class TestSaveAndCLI(unittest.TestCase):
    def test_save_and_reload(self):
        tokenizer = make_byte_level_tokenizer()
        model = make_model(tokenizer)
        pruned, new_tok, new2old = prune_model_vocab(
            model, tokenizer, CALIBRATION, keep_top_k=20, show_progress=False
        )
        with tempfile.TemporaryDirectory() as tmp:
            out = save_vocab_pruned(pruned, new_tok, new2old, tmp, keep_top_k=20)
            with open(out / "vocab_mapping.json") as f:
                mapping = json.load(f)
            self.assertEqual(mapping, {"keep_top_k": 20, "new2old": new2old.tolist()})

            reloaded = AutoModelForCausalLM.from_pretrained(str(out)).eval()
            re_tok = AutoTokenizer.from_pretrained(str(out))
            self.assertEqual(len(re_tok), len(new2old))
            self.assertEqual(re_tok.eos_token, EOS)
            self.assertEqual(re_tok.eos_token_id, reloaded.config.eos_token_id)
            ids = re_tok("2 + 2", return_tensors="pt")["input_ids"]
            with torch.no_grad():
                torch.testing.assert_close(reloaded(ids).logits, pruned(ids).logits)

    def test_cli_prune_vocab(self):
        tokenizer = make_byte_level_tokenizer()
        model = make_model(tokenizer)
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp:
            data = os.path.join(tmp, "calib.jsonl")
            with open(data, "w") as f:
                for row in CALIBRATION:
                    f.write(json.dumps(row) + "\n")
            out = os.path.join(tmp, "out")
            with patch("optipfair.cli.commands.AutoModelForCausalLM") as m_model, patch(
                "optipfair.cli.commands.AutoTokenizer"
            ) as m_tok:
                m_model.from_pretrained.return_value = model
                m_tok.from_pretrained.return_value = tokenizer
                result = runner.invoke(
                    cli,
                    [
                        "prune-vocab",
                        "--model-path",
                        "tiny",
                        "--calibration-data",
                        data,
                        "--keep-top-k",
                        "20",
                        "--output-path",
                        out,
                        "--device",
                        "cpu",
                        "--quiet",
                    ],
                )
            self.assertEqual(result.exit_code, 0, result.output + repr(result.exception))
            for name in (
                "config.json",
                "model.safetensors",
                "tokenizer.json",
                "vocab_mapping.json",
            ):
                self.assertTrue(os.path.exists(os.path.join(out, name)), name)
            reloaded = AutoTokenizer.from_pretrained(out)
            self.assertLess(len(reloaded), len(tokenizer))

    def test_cli_rejects_split_for_file(self):
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp:
            data = os.path.join(tmp, "calib.txt")
            with open(data, "w") as f:
                f.write("hello\n")
            result = runner.invoke(
                cli,
                [
                    "prune-vocab",
                    "--model-path",
                    "x",
                    "--calibration-data",
                    data,
                    "--split",
                    "train",
                    "--keep-top-k",
                    "1",
                    "--output-path",
                    tmp,
                ],
            )
            self.assertNotEqual(result.exit_code, 0)
            self.assertIn("--split", result.output)


if __name__ == "__main__":
    unittest.main()
