## [0.2.1] - 2025-11-24

### 🎉 New Features

#### Hardware-Optimized Pruning with expansion_divisor
- **expansion_divisor Parameter**: New parameter to round intermediate layer sizes to specific multiples (32, 64, 128, 256)
- **GPU Optimization**: Ensures tensor dimensions are optimized for modern GPU/TPU architectures
- **Flexible Integration**: Works seamlessly with both `pruning_percentage` and `expansion_rate` parameters
- **Automatic Rounding**: Intelligently rounds to the nearest multiple after pruning calculation

### ✨ Enhancements

- **Extended API**: New `expansion_divisor` parameter in `prune_model()` and `prune_model_mlp_glu()`
- **Hardware Alignment**: Better memory access patterns for tensor cores and SIMD operations
- **Validation System**: Comprehensive error checking for valid divisor values and parameter combinations
- **Utility Function**: New `round_to_divisor()` function for precise rounding logic

### 🔧 Technical Details

#### New Functions
- `round_to_divisor()`: Rounds values to nearest multiple of specified divisor

#### Modified Functions
- `prune_model()`: Added `expansion_divisor` parameter with validation
- `prune_model_mlp_glu()`: Integrated expansion_divisor validation and propagation
- `prune_neuron_pairs()`: Added rounding logic after initial pruning calculation

### 📚 Documentation

- Updated API reference with expansion_divisor examples
- Added comprehensive usage guide for hardware optimization
- Created Jupyter notebook example: `examples/expansion_divisor_example.ipynb`
- Updated README.md with hardware-optimized pruning section
- Updated examples/README.md with new tutorial link
- Enhanced LLM reference manual with expansion_divisor documentation

### 🧪 Testing

- Complete test suite in `tests/test_expansion_divisor.py`
- Validation tests for all allowed values
- Rounding behavior tests
- Integration tests with different pruning methods
- Edge case testing

### ⚠️ Important Notes

- `expansion_divisor` cannot be used alone - requires either `pruning_percentage` or `expansion_rate`
- Valid values: `None` (default), `32`, `64`, `128`, `256`
- Rounding maintains bounds: result is always ≥1 and ≤ original size

### 🔒 Compatibility

- Fully backward compatible with v0.2.0
- Works with all neuron selection methods (MAW, VOW, PON)
- Compatible with both static and data-driven pruning

---

# Changelog

All notable changes to OptiPFair will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2025-10-27

### 🎉 Major Features

#### Data-Driven Width Pruning
- **Hybrid Importance Calculation**: Implemented data-driven neuron selection combining static weights with activation statistics
- **Activation Capture System**: PyTorch hooks infrastructure to collect neuron activations during calibration
- **CFSP Method Integration**: Implementation based on "CFSP: An Efficient Structured Pruning Framework for LLMs with Coarse-to-Fine Activation Information" (arXiv:2409.13199v2)

### ✨ Enhancements

- **Extended API**: New `dataloader` parameter in `prune_model()` for calibration data
- **Automatic Method Selection**: Intelligent switching between static and hybrid pruning based on dataloader presence
- **Memory Optimization**: CPU-based activation storage during calibration to minimize VRAM usage
- **Better Error Messages**: Comprehensive validation with clear error messages for incompatible configurations

### 🔧 Technical Details

#### New Functions
- `compute_neuron_pair_importance_maw_hybrid()`: Hybrid importance calculation using Equation 8 from CFSP paper
- `setup_mlp_hooks_for_importance()`: Register forward hooks for activation capture
- `get_activation_norms()`: Retrieve accumulated L2 norms from calibration
- `run_calibration_forward_passes()`: Execute calibration with progress tracking

#### Modified Functions
- `prune_model()`: Added `dataloader` parameter
- `prune_model_mlp_glu()`: Integrated calibration workflow and hybrid pruning logic
- `prune_neuron_pairs()`: Extended to support both static and hybrid importance calculation

### 📚 Documentation

- Updated API reference with data-driven pruning examples
- Added comprehensive usage guide for hybrid pruning
- Created Jupyter notebook example: `examples/data_driven_pruning.ipynb`
- Updated README with quick start guide for data-driven pruning

### 🧪 Testing

- Validated on Gemma, LLaMA, and Mistral model families
- Confirmed backward compatibility with existing static pruning code
- Added validation for dataloader compatibility with pruning methods

### ⚠️ Breaking Changes

None - This release is fully backward compatible with v0.1.x

### 🔒 Compatibility

- Only `neuron_selection_method="MAW"` supports data-driven pruning
- VOW and PON methods remain static-only (will raise `ValueError` if used with dataloader)
- Supports PyTorch dataloaders with dict or tuple batch formats

---

## [0.1.5] - 2024-XX-XX

### Added
- Layer importance analysis
- Depth pruning functionality

### Fixed
- Various bug fixes and improvements

---

## [0.1.0] - 2024-XX-XX

### Added
- Bias visualization tools
- Initial release
- MLP GLU pruning support
- MAW, VOW, PON neuron selection methods
- CLI interface