# Installation

## Requirements

OptiPFair requires:

- Python 3.8 or higher
- PyTorch >1.11.0
- Transformers 4.25.0 or higher

## Installing from PyPI

```bash
pip install optipfair
```

## Installing from Source

You can install the latest development version from GitHub:

```bash
git clone https://github.com/yourusername/optipfair.git
cd optipfair
pip install -e .
```

## Supported Hardware

OptiPFair runs on NVIDIA GPUs (CUDA), Apple Silicon GPUs (MPS / Metal) and CPU. Pruning,
layer-importance analysis and distillation run on whatever device the model is loaded on, and
`optipfair.get_device()` picks the best available backend (CUDA, then MPS, then CPU). Set the
`OPTIPFAIR_DEVICE` environment variable (for example `OPTIPFAIR_DEVICE=cpu`) to override the
automatic choice. See [Device Selection](usage.md#device-selection-cuda-apple-silicon-cpu) for details.

On Apple Silicon, install the standard PyTorch wheel (`pip install torch`); MPS support is
included. `bfloat16` on MPS requires macOS 14 or newer.

PyTorch 2.7 or newer is recommended on Apple Silicon. With PyTorch 2.6, models that use
grouped-query attention (Llama 3, Qwen, ...) can abort the process on MPS during `generate()`
(`LLVM ERROR: Failed to infer result type(s)` from `mps.matmul`), because
`scaled_dot_product_attention(..., enable_gqa=True)` is not supported there. If you must stay on
PyTorch 2.6, load the model with `attn_implementation="eager"`.