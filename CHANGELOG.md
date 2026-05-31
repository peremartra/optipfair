## [Unreleased]

---

## [0.4.1] - 2026-05-31

### 🔧 Bug Fixes

#### Distillation Losses Now Ignore Padding Tokens (closes #34)
- Fixed an issue where distillation labels were copied from `input_ids` without masking padded positions, preventing `ignore_index=-100` from taking effect.
- Updated `compute_distillation_loss()` so logits, trajectory, and derivative components reduce only over valid tokens.
- Trainer now respects user-provided `labels`; when labels are absent and `attention_mask` is available, it generates labels and masks padding with `-100`.

#### `analyze_neuron_bias` Now Accepts `batch_size` Parameter (closes #33)
- The reference manual documented `batch_size` as a valid parameter but the function signature did not accept it, causing `TypeError` at runtime.
- Resolved the API/documentation mismatch: the parameter is now clearly documented as not supported at the function level; prompt pairs are processed individually (one pair per forward pass) to handle asymmetric sequence lengths correctly.
- Updated docstring to reflect actual behavior and remove misleading `batch_size` references.

### ✨ New Features

#### Activation Capture at `down_proj` Input (closes #35)
- New target layer type `"down_proj_input"` captures activations at the input of `down_proj` using a forward pre-hook, exposing the expanded MLP space (`[B, S, intermediate_size]`).
- Existing `"down_proj"` behavior (post-projection, `[B, S, hidden_size]`) is fully unchanged.
- `"down_proj_input"` is explicit opt-in: not included when `target_layers=None`.
- Keys stored as `down_proj_input_layer_{i}`.
- New tests in `test_bias_visualization.py` cover validation, key naming, shapes, combined capture, and backward compatibility.

### 🚀 Status
- Promoted from **Alpha** to **Beta** (`Development Status :: 4 - Beta`).

### 🧪 Testing & Quality
- All existing tests pass; no breaking changes to the public API.

---

## [0.4.0] - 2026-04-17

### 🎉 New Features

#### Knowledge Distillation (closes #21)
- **New Function**: `opf.distill_model()` — full knowledge distillation pipeline for recovering student model quality after pruning
  - Temperature-scaled logit distillation (KL divergence)
  - Optional hidden-state matching loss (`beta` weight)
  - Optional attention matching loss (`gamma` weight)
  - Automatic layer mapping: `"uniform"` (spread student layers across teacher) or `"last"` (align to last N teacher layers)
  - Integrated `SequentialLR` scheduler support
  - Automatic device and dtype handling
- **New Constants**: `MAPPING_UNIFORM`, `MAPPING_LAST` exported at package level for explicit mapping control
- **Python API Only**: designed to follow a `prune_model` call to recover accuracy with teacher guidance

#### Documentation & Examples
- Dedicated Knowledge Distillation guide: `docs/knowledge_distillation.md`
- Two new notebooks: `knowledge_distillation.ipynb` (full control) and `knowledge_distillation_express.ipynb` (quick start)

### 🔧 Bug Fixes

#### Width Pruning No Longer Increases Model Size (closes #27)
- **Fixed**: Using `expansion_divisor` with a small `pruning_percentage` could produce a pruned model with *more* neurons than the original when divisor rounding cancelled the intended reduction
- **Solution**: After computing the target neuron count, a strict validation check raises `ValueError` if the rounded value is ≥ the original intermediate size, surfacing the invalid parameter combination early with a clear message
- **Impact**: All combinations of `pruning_percentage` + `expansion_divisor` now either produce a strictly smaller model or raise an explicit error

### 🔧 Technical Improvements

#### Depth Pruning Syncs `config.layer_types` After Pruning (closes #20)
- **Fixed**: `prune_model_depth()` did not update `model.config.layer_types` after removing layers
- **Problem**: Hybrid-architecture models (e.g., Qwen3.5 with GatedDeltaNet SSM blocks) allocate KV-cache buffers based on `config.layer_types`; a stale list caused `IndexError` during inference on the pruned model
- **Solution**:
  - `config.layer_types` entries are now removed in *reverse index order* after depth pruning, preventing index-shift corruption
  - `layer_idx` attributes on remaining layers are reassigned to match their new positions
- **Impact**: Depth pruning now works correctly with hybrid SSM/attention architectures

### 🧪 Testing & Quality
- All existing tests remain passing; no breaking changes to the public API

---

## [0.3.0] - 2026-03-02

### 🎉 New Features

#### Fairness-Aware Pruning
- **New Function**: `analyze_neuron_bias()` - Analyze per-neuron bias contributions across multiple demographic prompt pairs
  - Computes activation-based bias scores for individual neurons
  - Supports multiple aggregation methods (mean, max) across sequence positions
  - Works with GLU architecture MLP layers (gate_proj, up_proj)
- **New Function**: `compute_fairness_pruning_scores()` - Combine bias and importance scores for balanced pruning
  - Configurable `bias_weight` parameter (0.0 to 1.0) to adjust fairness vs. performance trade-offs
  - Returns fairness pruning scores for each layer
  - Enables fairness-aware neuron selection strategies

#### Enhanced Pruning Integration
- **Modified**: `prune_model_mlp_glu()` - Improved compatibility with fairness-aware workflows
- **Documentation**: Added comprehensive fairness-aware pruning guide with examples

### 📚 Documentation Enhancements

#### New Fairness-Aware Pruning Section
- Complete guide to fairness-aware pruning workflow with:
  - Step-by-step tutorial for `analyze_neuron_bias()`
  - Step-by-step tutorial for `compute_fairness_pruning_scores()`
  - Understanding the bias_weight parameter with recommended configurations
  - Complete end-to-end example combining bias analysis with pruning
  - Common patterns for fairness-aware analysis
- New example notebook: `fairness_aware_pruning_demo.ipynb`

#### Updated API Documentation
- Added `analyze_neuron_bias()` to API reference
- Added `compute_fairness_pruning_scores()` to API reference
- Enhanced usage guide with fairness workflows

### 🧪 Testing & Quality
- Compatible with existing pruning functionality
- No breaking changes to existing API
- All existing tests remain passing

---

## [0.2.4] - 2026-01-10

### 🎉 New Features

#### Universal DataLoader Format Support for `analyze_layer_importance`
- **Multi-Format Batch Handling**: `analyze_layer_importance` now automatically detects and handles multiple DataLoader batch formats without requiring HuggingFace dataset utilities
- **Supported Formats**:
  - **Dictionary**: HuggingFace-style `{'input_ids': tensor, 'attention_mask': tensor}`
  - **Tuple/List**: PyTorch `TensorDataset` format `(input_ids, attention_mask, ...)`
  - **Single Tensor**: Direct tensor input treated as `input_ids`
- **Positional Mapping**: Tuple/list elements automatically map to standard transformer arguments: `[0]=input_ids`, `[1]=attention_mask`, `[2]=token_type_ids`, etc.
- **Internal Utility**: New `_prepare_batch_inputs()` function normalizes all formats transparently
- **Debug Logging**: Optional DEBUG-level logging shows format detection and positional mapping
- **Zero Breaking Changes**: Existing code with dict-based DataLoaders works exactly as before

**Closes Issues**: #12, #17, #18

### 📝 Documentation Updates

#### Terminology Update: MAW → PPM
- **New Nomenclature**: The neuron selection method previously known as "MAW (Maximum Absolute Weight)" is now officially documented as **PPM (Peak-to-Peak Magnitude)**, which more accurately describes the calculation method (max + |min|).
- **Backward Compatibility**: The parameter value `"MAW"` is maintained for full backward compatibility and maps to the PPM method.
- **Research Foundation**: PPM is formally described in: *Martra, P. (2025). Fragile Knowledge, Robust Instruction-Following: The Width Pruning Dichotomy in Llama-3.2. ArXiv. https://arxiv.org/abs/2512.22671*
- **Updated Documentation**: All documentation files now reference PPM as the primary name with MAW noted as the legacy parameter value.

#### Clarification: L2 Norm Neuron Selection Method
- **Existing Feature**: The L2 norm method (`neuron_selection_method="L2"`) has been available since early versions
- **How It Works**: Calculates neuron importance using L2 (Euclidean) norms of weight values: `||gate_weight||₂ + ||up_weight||₂`
- **Static Only**: Supports **weight-only (static) pruning** exclusively - not compatible with data-driven mode (dataloader)
- **Documentation Enhancement**: Added explicit warnings in usage guides about L2 limitations vs PPM/MAW data-driven capabilities

### 🧪 Testing

#### Comprehensive Test Coverage for Batch Format Support
- **Unit Tests**: 11 new tests for `_prepare_batch_inputs()` covering all format variations
- **Integration Tests**: 5 new tests for `analyze_layer_importance()` with different DataLoader types
- **Test Coverage**: Dict batches, 2-element tuples, 3+ element tuples, lists, single tensors, None handling, device placement
- **All Tests Pass**: 16 new tests + 95 existing tests = 111 total passing tests

### 🔧 Technical Details

#### Implementation
- **File**: `optipfair/pruning/utils.py`
- **New Function**: `_prepare_batch_inputs(batch, device)` - internal utility with underscore prefix
- **Modified Function**: `analyze_layer_importance()` in `optipfair/pruning/depth.py` now uses normalized batch handling
- **Device Handling**: All tensors automatically moved to model device regardless of input format
- **Error Handling**: Clear ValueError with format hints for unsupported batch types

#### Enhanced Examples
- **layer_importance_analysis.ipynb**: Added section demonstrating TensorDataset (tuple format) usage
- **docs/usage.md**: New examples showing analyze_layer_importance with various DataLoader formats

### 🔒 Compatibility

- **Fully Backward Compatible**: All existing code continues to work without modification
- **No API Changes**: Function signatures unchanged, new functionality is transparent
- **Python**: Requires Python >=3.8 (unchanged)
- **Dependencies**: No new dependencies added

---

## [0.2.3] - 2025-12-04

### 🐛 Bug Fixes

#### Fixed Hybrid Importance Calculation
- **compute_neuron_pair_importance_maw_hybrid()**: Simplified and fixed fórmula for hybrid importance calculation
- **Improved Accuracy**: Now correctly combines static weight magnitudes with dynamic activation statistics
- **Better Performance**: Reduced unnecessary calculations while maintaining correctness
- **Consistent Methodology**: Uses MAW (Maximum Absolute Weight) consistently across all MLP components
- **No API Changes**: Fully backward compatible, internal optimization only

### 🔧 Technical Details

#### Fixed Functions
- `compute_neuron_pair_importance_maw_hybrid()`: Corrected importance score calculation:
  - Static Component: Uses MAW (max + |min|) for gate_proj and up_proj layers
  - Normalization: Scales each component to [0,1] for balanced weighting
  - Hybrid Fusion: Multiplies structural potential by activation norms
  - Validation: All tests pass, no breaking changes

### 🔒 Compatibility

- Fully backward compatible with v0.2.2 and earlier
- No changes to public API
- No changes to function signatures
- Internal optimization only

---

## [0.2.2] - 2025-11-26

### 🎉 New Features

#### Selective Layer Width Pruning
- **layer_indices for MLP_GLU**: Extended `layer_indices` parameter to support selective neuron pruning in specific layers
- **Contextual Usage**: For DEPTH pruning, specifies layers to remove; for MLP_GLU, specifies layers to prune
- **Preservation Strategy**: Allows preserving critical layers (e.g., first/last) at full capacity while pruning others
- **Full Compatibility**: Works seamlessly with all MLP_GLU features (expansion_rate, expansion_divisor, dataloader, all methods)

#### Simplified Hybrid Importance Calculation
- **Optimized MAW Hybrid**: Simplified `compute_neuron_pair_importance_maw_hybrid()` to use simple MAW for gate_proj and up_proj
- **Focused Complexity**: Maintains complex activation-weighted calculation only for down_proj where it has most impact
- **Better Performance**: Faster execution by reducing unnecessary calculations
- **Consistent Formula**: Uses same MAW method (max + |min|) as static pruning for gate/up components

### ✨ Enhancements

- **Extended API**: `layer_indices` parameter now works for both DEPTH and MLP_GLU pruning types
- **Smart Validation**: Comprehensive error checking for layer indices (range, duplicates, empty lists, types)
- **Enhanced Statistics**: `get_pruning_statistics()` now reports selective pruning info (pruned_layers, total_layers)
- **Selective Calibration**: Hooks only registered on selected layers when using data-driven pruning with layer_indices
- **CLI Support**: Updated `--layer-indices` help text to mention both pruning types
- **Backward Compatible**: `layer_indices=None` maintains default behavior (prunes all layers)

### 🔧 Technical Details

#### Modified Functions
- `prune_model()`: Updated docstring and passes `layer_indices` to `prune_model_mlp_glu()`
- `prune_model_mlp_glu()`: Added `layer_indices` parameter with full validation and filtering logic
- `setup_mlp_hooks_for_importance()`: Now accepts `layer_indices` to register hooks only on selected layers
- `compute_neuron_pair_importance_maw_hybrid()`: Simplified to use MAW for gate/up, complex calculation only for down
- `get_pruning_statistics()`: Detects and reports selective pruning information
- CLI `commands.py`: Removed restriction blocking `layer_indices` for MLP_GLU, added parsing logic

### 📚 Documentation

- **README.md**: New "Selective Layer Width Pruning" section with examples and use cases
- **Reference Manual**: Comprehensive section with 4+ usage examples and best practices
- **New Example File**: `examples/selective_layer_width_pruning.py` with 5 complete examples
- **Updated Roadmap**: Marked selective pruning as completed in v0.2.2
- **API Documentation**: Updated parameter descriptions for contextual meaning

### 🧪 Testing

- Complete test suite in `tests/test_selective_layer_pruning.py`
- 12 comprehensive test cases covering:
  - Basic selective pruning (single and multiple layers)
  - All neuron selection methods (MAW, VOW, PON)
  - Compatibility with expansion_rate and expansion_divisor
  - Data-driven pruning with layer_indices
  - Invalid input handling and validation
  - Statistics reporting
  - Weight preservation in unpruned layers
  - Result consistency and reproducibility

### 💡 Use Cases

1. **Preserve Critical Layers**: Keep first and last layers at full capacity
2. **Importance-Based**: Target least important layers identified by analysis
3. **Domain Adaptation**: Implement asymmetric pruning strategies
4. **Experimental**: Test different layer-wise pruning patterns

### 🔒 Compatibility

- Fully backward compatible with v0.2.1
- Works with all neuron selection methods (MAW, VOW, PON)
- Compatible with both static and data-driven pruning
- Integrates with expansion_rate and expansion_divisor

### ⚠️ Important Notes

- `layer_indices` validation ensures indices are valid, unique integers within model range
- Empty lists raise `ValueError`
- Selective pruning with dataloader only calibrates on specified layers (more efficient)
- Statistics include `pruned_layers` and `total_layers` when selective pruning is detected

---

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