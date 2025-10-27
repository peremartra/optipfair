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