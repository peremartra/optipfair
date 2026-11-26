"""
Tests for device selection helpers (CUDA / Apple Silicon MPS / CPU).
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import torch
from click.testing import CliRunner

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from optipfair import device as device_mod
from optipfair.cli.commands import cli
from optipfair.device import (
    DEVICE_ENV_VAR,
    empty_cache,
    get_default_dtype,
    get_device,
    memory_allocated,
    synchronize,
)


def _backends(cuda=False, mps=False):
    """Patch backend availability without touching real hardware."""
    return (
        patch.object(device_mod.torch.cuda, "is_available", return_value=cuda),
        patch.object(device_mod, "is_mps_available", return_value=mps),
    )


class TestGetDevice(unittest.TestCase):
    """Device resolution order and validation."""

    def setUp(self):
        env = patch.dict(os.environ, {}, clear=False)
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop(DEVICE_ENV_VAR, None)

    def _resolve(self, spec, cuda=False, mps=False):
        cuda_patch, mps_patch = _backends(cuda, mps)
        with cuda_patch, mps_patch:
            return get_device(spec)

    def test_auto_prefers_cuda(self):
        self.assertEqual(self._resolve("auto", cuda=True, mps=True), torch.device("cuda"))

    def test_auto_picks_mps_without_cuda(self):
        self.assertEqual(self._resolve("auto", mps=True), torch.device("mps"))

    def test_auto_falls_back_to_cpu(self):
        self.assertEqual(self._resolve("auto"), torch.device("cpu"))

    def test_none_and_empty_mean_auto(self):
        self.assertEqual(self._resolve(None, mps=True), torch.device("mps"))
        self.assertEqual(self._resolve("", mps=True), torch.device("mps"))

    def test_auto_is_case_insensitive(self):
        self.assertEqual(self._resolve(" AUTO ", mps=True), torch.device("mps"))

    def test_env_var_overrides_auto(self):
        os.environ[DEVICE_ENV_VAR] = "cpu"
        self.assertEqual(self._resolve(None, cuda=True, mps=True), torch.device("cpu"))
        self.assertEqual(self._resolve("auto", cuda=True, mps=True), torch.device("cpu"))

    def test_env_var_auto_value_uses_detection(self):
        os.environ[DEVICE_ENV_VAR] = "auto"
        self.assertEqual(self._resolve(None, mps=True), torch.device("mps"))

    def test_explicit_device_beats_env_var(self):
        os.environ[DEVICE_ENV_VAR] = "cpu"
        self.assertEqual(self._resolve("mps", mps=True), torch.device("mps"))

    def test_explicit_cuda_index_is_preserved(self):
        self.assertEqual(self._resolve("cuda:1", cuda=True), torch.device("cuda:1"))

    def test_torch_device_passthrough(self):
        self.assertEqual(self._resolve(torch.device("cpu")), torch.device("cpu"))

    def test_unavailable_mps_raises(self):
        with self.assertRaisesRegex(RuntimeError, "MPS"):
            self._resolve("mps")

    def test_unavailable_cuda_raises(self):
        with self.assertRaisesRegex(RuntimeError, "CUDA"):
            self._resolve("cuda")

    def test_unavailable_env_device_raises(self):
        os.environ[DEVICE_ENV_VAR] = "mps"
        with self.assertRaises(RuntimeError):
            self._resolve(None)

    def test_unknown_device_raises_value_error(self):
        with self.assertRaisesRegex(ValueError, "Unknown device"):
            self._resolve("tpu-ish")


class TestDefaultDtype(unittest.TestCase):
    """Default dtype per backend."""

    def test_cpu_is_float32(self):
        self.assertEqual(get_default_dtype("cpu"), torch.float32)

    def test_mps_prefers_bfloat16(self):
        with patch.object(device_mod, "is_mps_available", return_value=True), patch.object(
            device_mod, "_mps_supports_bfloat16", return_value=True
        ):
            self.assertEqual(get_default_dtype("mps"), torch.bfloat16)

    def test_mps_without_bfloat16_uses_float32(self):
        with patch.object(device_mod, "is_mps_available", return_value=True), patch.object(
            device_mod, "_mps_supports_bfloat16", return_value=False
        ):
            self.assertEqual(get_default_dtype("mps"), torch.float32)

    def test_cuda_ampere_is_bfloat16(self):
        with patch.object(torch.cuda, "is_available", return_value=True), patch.object(
            torch.cuda, "get_device_capability", return_value=(8, 0)
        ):
            self.assertEqual(get_default_dtype("cuda"), torch.bfloat16)

    def test_cuda_pre_ampere_is_float16(self):
        with patch.object(torch.cuda, "is_available", return_value=True), patch.object(
            torch.cuda, "get_device_capability", return_value=(7, 5)
        ):
            self.assertEqual(get_default_dtype("cuda"), torch.float16)


class TestBackendDispatch(unittest.TestCase):
    """empty_cache / synchronize only touch the requested backend."""

    def _run(self, func, device, cuda=True, mps=True):
        cuda_backend = MagicMock()
        mps_backend = MagicMock()
        name = func.__name__
        with patch.object(torch.cuda, "is_available", return_value=cuda), patch.object(
            device_mod, "is_mps_available", return_value=mps
        ), patch.object(torch.cuda, name, getattr(cuda_backend, name)), patch.object(
            torch.mps, name, getattr(mps_backend, name)
        ):
            func(device)
        return getattr(cuda_backend, name).called, getattr(mps_backend, name).called

    def test_empty_cache_cpu_is_noop(self):
        self.assertEqual(self._run(empty_cache, "cpu"), (False, False))

    def test_empty_cache_mps_only(self):
        self.assertEqual(self._run(empty_cache, torch.device("mps")), (False, True))

    def test_empty_cache_cuda_only(self):
        self.assertEqual(self._run(empty_cache, "cuda:0"), (True, False))

    def test_empty_cache_none_clears_available_backends(self):
        self.assertEqual(self._run(empty_cache, None), (True, True))
        self.assertEqual(self._run(empty_cache, None, cuda=False), (False, True))

    def test_synchronize_mps_only(self):
        self.assertEqual(self._run(synchronize, "mps"), (False, True))

    def test_synchronize_cpu_is_noop(self):
        self.assertEqual(self._run(synchronize, "cpu"), (False, False))

    def test_memory_allocated_cpu_is_none(self):
        self.assertIsNone(memory_allocated("cpu"))


class TestCLIDevice(unittest.TestCase):
    """--device handling in the CLI."""

    def setUp(self):
        self.runner = CliRunner()
        env = patch.dict(os.environ, {}, clear=False)
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop(DEVICE_ENV_VAR, None)

    def test_unavailable_device_is_usage_error(self):
        with patch.object(device_mod, "is_mps_available", return_value=False):
            result = self.runner.invoke(
                cli, ['analyze', '--model-path', 'test-model', '--device', 'mps']
            )
        self.assertEqual(result.exit_code, 2)
        self.assertIn("MPS", result.output)

    @patch('optipfair.cli.commands.AutoModelForCausalLM')
    def test_auto_device_selects_mps(self, mock_model_class):
        mock_model = MagicMock()
        mock_model.config.model_type = "llama"
        mock_model_class.from_pretrained.return_value = mock_model
        with patch.object(torch.cuda, "is_available", return_value=False), patch.object(
            device_mod, "is_mps_available", return_value=True
        ), patch('optipfair.cli.commands.count_parameters', return_value=1):
            self.runner.invoke(cli, ['analyze', '--model-path', 'test-model'])
        self.assertEqual(
            mock_model_class.from_pretrained.call_args.kwargs['device_map'], 'mps'
        )

    @patch('optipfair.cli.commands.AutoModelForCausalLM')
    @patch('optipfair.cli.commands.AutoTokenizer')
    @patch('optipfair.cli.commands.prune_model')
    def test_prune_auto_dtype_follows_device(
        self, mock_prune_model, mock_tokenizer, mock_model_class
    ):
        mock_model_class.from_pretrained.return_value = MagicMock()
        mock_tokenizer.from_pretrained.return_value = MagicMock()
        mock_prune_model.return_value = (
            MagicMock(),
            {
                'original_parameters': 2,
                'pruned_parameters': 1,
                'reduction': 1,
                'percentage_reduction': 50.0,
                'expansion_rate': 200.0,
            },
        )
        with patch.object(torch.cuda, "is_available", return_value=False), patch.object(
            device_mod, "is_mps_available", return_value=True
        ), patch.object(device_mod, "_mps_supports_bfloat16", return_value=True), patch(
            'os.makedirs'
        ):
            result = self.runner.invoke(
                cli,
                ['prune', '--model-path', 'test-model', '--pruning-percentage', '20',
                 '--output-path', './test-output'],
            )
        self.assertEqual(result.exit_code, 0, result.output)
        kwargs = mock_model_class.from_pretrained.call_args.kwargs
        self.assertEqual(kwargs['device_map'], 'mps')
        self.assertEqual(kwargs['torch_dtype'], torch.bfloat16)


@unittest.skipUnless(device_mod.is_mps_available(), "requires Apple Silicon MPS backend")
class TestRealMPS(unittest.TestCase):
    """Exercise the helpers against the real MPS backend when present."""

    def test_helpers_on_mps(self):
        mps = get_device("mps")
        x = torch.ones(1024, device=mps)
        before = memory_allocated(mps)
        self.assertIsInstance(before, int)
        self.assertGreater(before, 0)
        synchronize(mps)
        del x
        empty_cache(mps)

    def test_default_dtype_is_usable(self):
        dtype = get_default_dtype("mps")
        self.assertIn(dtype, (torch.bfloat16, torch.float32))
        torch.zeros(4, dtype=dtype, device="mps").sum().item()


if __name__ == "__main__":
    unittest.main()
