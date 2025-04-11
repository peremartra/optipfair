# Installation

## Requirements

OptiPFair requires:

- Python 3.8 or higher
- PyTorch 1.10.0 or higher
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

## Optional Dependencies

OptiPFair has several optional dependency groups:

### Development Tools

Install development dependencies for contributing to OptiPFair:

```bash
pip install -e ".[dev]"
```

This includes:
- pytest
- black
- flake8
- isort
- mypy

### Evaluation Tools

Install evaluation dependencies for benchmarking pruned models:

```bash
pip install -e ".[eval]"
```

This includes:
- datasets
- numpy
- pandas
- matplotlib
- seaborn

## Verifying the Installation

To verify that OptiPFair is installed correctly, run:

```bash
python -c "import optipfair; print(optipfair.__version__)"
```

Or check the CLI:

```bash
optipfair --help
```

## GPU Support

For best performance, we recommend using a CUDA-capable GPU. Make sure you have the appropriate version of PyTorch installed for your CUDA version.

To check if PyTorch is using your GPU:

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Number of GPUs: {torch.cuda.device_count()}")
```

OptiPFair will automatically use available GPUs when loading and processing models.