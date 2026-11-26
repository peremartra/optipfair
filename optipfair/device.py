"""
Device helpers for running OptiPFair on NVIDIA GPUs (CUDA), Apple Silicon GPUs (MPS) or CPU.

OptiPFair's pruning, analysis and distillation functions run on whatever device the model
already lives on. These helpers cover the remaining backend-specific pieces: picking a device,
picking a sensible default dtype, releasing cached memory and synchronizing before timing.

Device resolution order for ``get_device()`` / ``get_device("auto")``:

1. The ``OPTIPFAIR_DEVICE`` environment variable, if set (e.g. ``OPTIPFAIR_DEVICE=cpu``).
2. CUDA, if available.
3. MPS (Apple Silicon / Metal), if available.
4. CPU.
"""

import os
from typing import Optional, Union

import torch

DeviceLike = Union[str, torch.device, None]

DEVICE_ENV_VAR = "OPTIPFAIR_DEVICE"
AUTO_DEVICE = "auto"


def is_mps_available() -> bool:
    """Return True if the Apple Silicon (Metal Performance Shaders) backend is usable."""
    mps_backend = getattr(torch.backends, "mps", None)
    return bool(mps_backend is not None and mps_backend.is_available())


def _auto_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if is_mps_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_device(device: DeviceLike = None) -> torch.device:
    """
    Resolve a device specification to a ``torch.device``.

    Args:
        device: ``None`` or ``"auto"`` to pick the best available backend (honouring the
            ``OPTIPFAIR_DEVICE`` environment variable), or an explicit device such as
            ``"cuda"``, ``"cuda:1"``, ``"mps"`` or ``"cpu"``.

    Returns:
        The resolved ``torch.device``.

    Raises:
        RuntimeError: If an explicitly requested backend is not available. An explicit request
            fails loudly instead of silently falling back to a much slower CPU run.
        ValueError: If the device string is not recognised by PyTorch.
    """
    if isinstance(device, torch.device):
        requested = device
    else:
        spec = (device or "").strip().lower()
        if spec in ("", AUTO_DEVICE):
            spec = os.environ.get(DEVICE_ENV_VAR, "").strip().lower() or AUTO_DEVICE
        if spec == AUTO_DEVICE:
            return _auto_device()
        try:
            requested = torch.device(spec)
        except RuntimeError as exc:
            raise ValueError(
                f"Unknown device '{spec}'. Use 'auto', 'cpu', 'cuda', 'cuda:N' or 'mps'."
            ) from exc

    if requested.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is not available.")
    if requested.type == "mps" and not is_mps_available():
        raise RuntimeError(
            "MPS device requested but the MPS backend is not available. MPS requires an Apple "
            "Silicon Mac, macOS 12.3+ and a PyTorch build with MPS support."
        )
    return requested


def _mps_supports_bfloat16() -> bool:
    # bfloat16 on MPS needs macOS 14+; probing is more reliable than parsing OS versions.
    try:
        torch.zeros(1, dtype=torch.bfloat16, device="mps")
        return True
    except (RuntimeError, TypeError):
        return False


def get_default_dtype(device: DeviceLike = None) -> torch.dtype:
    """
    Return a sensible default dtype for loading models on ``device``.

    - CUDA: ``bfloat16`` on Ampere or newer, ``float16`` otherwise.
    - MPS: ``bfloat16`` when supported (macOS 14+), otherwise ``float32``. ``float16`` is avoided
      as a fallback because models trained in bfloat16 (e.g. Gemma) can overflow in float16.
    - CPU: ``float32``.

    Args:
        device: Device specification accepted by :func:`get_device`.

    Returns:
        The recommended ``torch.dtype``.
    """
    resolved = get_device(device)
    if resolved.type == "cuda":
        if torch.cuda.get_device_capability(resolved)[0] >= 8:
            return torch.bfloat16
        return torch.float16
    if resolved.type == "mps":
        return torch.bfloat16 if _mps_supports_bfloat16() else torch.float32
    return torch.float32


def _device_type(device: DeviceLike) -> Optional[str]:
    if device is None:
        return None
    if isinstance(device, torch.device):
        return device.type
    return torch.device(device).type


def empty_cache(device: DeviceLike = None) -> None:
    """
    Release cached allocator memory back to the system.

    Args:
        device: Device whose cache should be released. If ``None``, the caches of every
            available accelerator backend are released. CPU is a no-op.
    """
    device_type = _device_type(device)
    if device_type in (None, "cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
    if device_type in (None, "mps") and is_mps_available() and hasattr(torch, "mps"):
        torch.mps.empty_cache()


def synchronize(device: DeviceLike = None) -> None:
    """
    Wait for all queued kernels on ``device`` to finish.

    GPU work is asynchronous on both CUDA and MPS, so timing code must synchronize before
    reading the clock.

    Args:
        device: Device to synchronize. If ``None``, every available accelerator backend is
            synchronized. CPU is a no-op.
    """
    device_type = _device_type(device)
    if device_type in (None, "cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    if device_type in (None, "mps") and is_mps_available() and hasattr(torch, "mps"):
        torch.mps.synchronize()


def memory_allocated(device: DeviceLike = None) -> Optional[int]:
    """
    Return the number of bytes currently allocated by tensors on ``device``.

    Args:
        device: Device to query. Defaults to the device chosen by :func:`get_device`.

    Returns:
        Bytes allocated, or ``None`` for CPU (not tracked by PyTorch).
    """
    resolved = get_device(device)
    if resolved.type == "cuda":
        return torch.cuda.memory_allocated(resolved)
    if resolved.type == "mps" and hasattr(torch, "mps"):
        return torch.mps.current_allocated_memory()
    return None
