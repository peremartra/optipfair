# Usage Guide
This guide provides detailed instructions on how to use the core functionalities of OptiPFair, from pruning models to analyzing bias.

---

**Note on Terminology:** The default neuron selection method is **PPM (Peak-to-Peak Magnitude)**, which calculates neuron importance based on the full dynamic range of weights (max + |min|). This method is formally described in: *Martra, P. (2025). Fragile Knowledge, Robust Instruction-Following: The Width Pruning Dichotomy in Llama-3.2. ArXiv. https://arxiv.org/abs/2512.22671*

For backward compatibility, the parameter value `"MAW"` is still accepted and maps to PPM.

---

## Python API

OptiPFair provides a simple Python API for pruning models.

### Basic Usage

```python
import optipfair as opf
from transformers import AutoModelForCausalLM

# Load a pre-trained model
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")

# Prune the model with default settings (10% pruning, PPM method)
pruned_model = opf.prune_model(model=model)

# Save the pruned model
pruned_model.save_pretrained("./pruned-model")
```

### Advanced Usage

```python
import optipfair as opf

# Prune with custom settings
pruned_model, stats = opf.prune_model(
    model=model,
    pruning_type="MLP_GLU",              # Type of pruning to apply
    neuron_selection_method="MAW",       # Method to calculate neuron importance
    pruning_percentage=20,               # Percentage of neurons to prune
    # expansion_rate=140,                # Alternatively, specify target expansion rate
    show_progress=True,                  # Show progress during pruning
    return_stats=True                    # Return pruning statistics
)

# Print pruning statistics
print(f"Original parameters: {stats['original_parameters']:,}")
print(f"Pruned parameters: {stats['pruned_parameters']:,}")
print(f"Reduction: {stats['reduction']:,} parameters ({stats['percentage_reduction']:.2f}%)")
```

## Knowledge Distillation (Available from v0.4.0)

OptiPFair provides KD via Python API using `import optipfair as opf` and `opf.distill_model`.

```python
import optipfair as opf

trained_student, stats = opf.distill_model(
    student_model=student_model,
    teacher_model=teacher_model,
    dataloader=dataloader,
    alpha=0.6,
    beta=0.4,
    gamma=0.0,
    delta=0.0,
    temperature=2.0,
    skew_alpha=0.4,
    epochs=3,
    learning_rate=4e-5,
    scheduler="cosine",
    warmup_ratio=0.05,
    accumulation_steps=4,
    return_stats=True,
)
```

See the full guide in [Knowledge Distillation](knowledge_distillation.md).

## Command-Line Interface

OptiPFair provides a command-line interface for pruning models:

### Basic Usage

```bash
# Prune a model with default settings (10% pruning, PPM method)
optipfair prune --model-path meta-llama/Llama-3.2-1B --output-path ./pruned-model
```

### Advanced Usage

```bash
# Prune with custom settings
optipfair prune \
  --model-path meta-llama/Llama-3.2-1B \
  --pruning-type MLP_GLU \
  --method MAW \
  --pruning-percentage 20 \
  --output-path ./pruned-model \
  --device cuda \
  --dtype float16
```

### Analyzing a Model

```bash
# Analyze a model's architecture and parameter distribution
optipfair analyze --model-path meta-llama/Llama-3.2-1B
```

## Neuron Selection Methods

OptiPFair supports four methods for calculating neuron importance:

### PPM (Peak-to-Peak Magnitude)

The PPM method identifies neurons based on the peak-to-peak magnitude of weights (max + |min|), capturing the full dynamic range of each neuron's weight values. This is typically the most effective method for GLU architectures. Use parameter value `"MAW"` for backward compatibility.

```python
pruned_model = prune_model(
    model=model,
    neuron_selection_method="MAW",  # PPM method ("MAW" for compatibility)
    pruning_percentage=20
)
```

### VOW (Variance of Weights)

The VOW method identifies neurons based on the variance of their weight values.

```python
pruned_model = prune_model(
    model=model,
    neuron_selection_method="VOW",
    pruning_percentage=20
)
```

### PON (Product of Norms)

The PON method uses the product of L1 norms to identify important neurons.

```python
pruned_model = prune_model(
    model=model,
    neuron_selection_method="PON",
    pruning_percentage=20
)
```

### L2 (L2 Norm)

The L2 method calculates neuron importance using L2 norms of weight values.

```python
pruned_model = prune_model(
    model=model,
    neuron_selection_method="L2",
    pruning_percentage=20
)
```

**Note:** Data-driven pruning (hybrid mode) is only available with the PPM method (`"MAW"`). VOW, PON, and L2 support static (weight-only) pruning only.

## Data-Driven Pruning (v0.2.0+)

### Overview

Data-driven pruning enhances neuron selection by incorporating activation statistics from real data. Instead of relying solely on weight magnitudes, this hybrid approach analyzes how neurons actually behave with your specific data distribution.

### When to Use Data-Driven Pruning

**Use data-driven pruning when:**
- 🎯 You have domain-specific data (medical, legal, code, etc.)
- 📊 You want to preserve task-specific capabilities
- 🔬 You need more intelligent neuron selection
- ⚡ You can afford a one-time calibration pass

**Use static pruning when:**
- ⚡ You need fastest possible pruning
- 🌐 You're pruning for general-purpose use
- 💾 You don't have representative calibration data

### Basic Usage
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader, TensorDataset
from optipfair import prune_model

# 1. Load model
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
tokenizer.pad_token = tokenizer.eos_token

# 2. Prepare calibration data
texts = ["Your domain-specific examples here..."] * 500
inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'])
dataloader = DataLoader(dataset, batch_size=8)

# 3. Prune with calibration
pruned_model = prune_model(
    model=model,
    neuron_selection_method="MAW",
    pruning_percentage=20,
    dataloader=dataloader,  # ← Enables data-driven pruning
    show_progress=True
)
```

### Calibration Data Guidelines

#### Dataset Size
- **Minimum:** 50-100 samples
- **Recommended:** 500-1000 samples
- **Maximum:** 5000+ samples (diminishing returns)

#### Data Quality
✅ **Good calibration data:**
- Representative of target use case
- Diverse examples from domain
- Natural distribution of inputs
- Similar length to deployment data

❌ **Poor calibration data:**
- Generic/unrelated text
- Single repeated example
- Extreme outliers only
- Wrong language/domain

#### Example: Code Generation Model
```python
# Good: Domain-specific code samples
code_samples = [
    "def fibonacci(n): return n if n < 2 else fibonacci(n-1) + fibonacci(n-2)",
    "class DataLoader: def __init__(self, data): self.data = data",
    "import numpy as np\narray = np.zeros((10, 10))",
    # ... 500 more diverse code examples
]

# Bad: Generic text
bad_samples = [
    "The quick brown fox jumps over the lazy dog",
    "Hello world",
    # ... unrelated to code
]
```

### Batch Size Recommendations

| Model Size | VRAM | Batch Size | Calibration Samples |
|-----------|------|------------|-------------------|
| < 1B params | 8GB | 16-32 | 500-1000 |
| 1-3B params | 16GB | 8-16 | 500-1000 |
| 3-7B params | 24GB | 4-8 | 300-500 |
| 7-13B params | 40GB+ | 2-4 | 200-300 |

### Understanding the Hybrid Method

Data-driven pruning uses the CFSP (Coarse-to-Fine Structured Pruning) methodology:

**Equation 8 from CFSP paper:**
```
Importance(neuron_i) = 
    activation_component(neuron_i) +    # Data-driven (down_proj)
    weight_component_up(neuron_i) +     # Static (up_proj)
    weight_component_gate(neuron_i)     # Static (gate_proj)
```

**Components:**
1. **Activation Component (down_proj):** Measures how much each neuron activates with real data
2. **Weight Components (up_proj, gate_proj):** Traditional magnitude-based importance

This combination ensures:
- Neurons important for your data are preserved
- Structural integrity is maintained
- Pruning is stable and predictable

### Advanced: Custom Dataloader
```python
from torch.utils.data import Dataset, DataLoader

class CustomCalibrationDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )
    
    def __len__(self):
        return len(self.encodings['input_ids'])
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx]
        }

# Use custom dataset
dataset = CustomCalibrationDataset(my_texts, tokenizer)
dataloader = DataLoader(dataset, batch_size=8, shuffle=False)

pruned_model = prune_model(model, dataloader=dataloader, pruning_percentage=20)
```

### Comparison: Static vs Data-Driven
```python
# Test both methods
import copy

# Static pruning
model_static = copy.deepcopy(model)
pruned_static = prune_model(
    model_static,
    pruning_percentage=20,
    dataloader=None  # Static
)

# Data-driven pruning
model_datadriven = copy.deepcopy(model)
pruned_datadriven = prune_model(
    model_datadriven,
    pruning_percentage=20,
    dataloader=calibration_dataloader  # Hybrid
)

# Evaluate on your benchmark
# Typically data-driven shows 2-5% better performance retention
```

### Troubleshooting

#### Error: "Data-driven pruning with dataloader is only supported for 'MAW' method"
**Solution:** Change `neuron_selection_method` to `"MAW"` (PPM method):
```python
pruned = prune_model(model, neuron_selection_method="MAW", dataloader=dl)  # PPM method
```

#### Out of Memory during calibration
**Solutions:**
1. Reduce batch size: `DataLoader(dataset, batch_size=2)`
2. Reduce calibration samples: Use 100-200 samples instead of 1000
3. Use smaller max_length: `tokenizer(..., max_length=256)`

#### Calibration taking too long
**Solutions:**
1. Use fewer samples (100-300 is often sufficient)
2. Increase batch size if VRAM allows
3. Use shorter sequences

### Performance Tips

1. **Use FP16/BF16:** Load model with `torch_dtype=torch.float16` for faster calibration
2. **Shuffle Data:** Shuffle calibration dataloader for better representation
3. **Cache Dataset:** Pre-tokenize and cache your calibration dataset
4. **Monitor VRAM:** Use `torch.cuda.empty_cache()` if needed
```python
# Optimized example
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B",
    torch_dtype=torch.float16,  # Faster calibration
    device_map="auto"
)

dataloader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True,  # Better representation
    num_workers=2  # Parallel data loading
)
```


### VOW (Variance of Weights)

The VOW method identifies neurons based on the variance of their weight values. This can be useful for certain specific architectures.

```python
pruned_model = prune_model(
    model=model,
    neuron_selection_method="VOW",
    pruning_percentage=20
)
```

### PON (Product of Norms)

The PON method uses the product of L1 norms to identify important neurons. This is an alternative approach that may be useful in certain contexts.

```python
pruned_model = prune_model(
    model=model,
    neuron_selection_method="PON",
    pruning_percentage=20
)
```

## Pruning Percentage vs Expansion Rate

OptiPFair supports two ways to specify the pruning target:

### Pruning Percentage

Directly specify what percentage of neurons to remove:

```python
pruned_model = prune_model(
    model=model,
    pruning_percentage=20  # Remove 20% of neurons
)
```

### Expansion Rate

Specify the target expansion rate (ratio of intermediate size to hidden size) as a percentage:

```python
pruned_model = prune_model(
    model=model,
    expansion_rate=140  # Target 140% expansion rate
)
```

This approach is often more intuitive when comparing across different model scales.

## Depth Pruning

Depth pruning removes entire transformer layers. When calling `prune_model(..., pruning_type="DEPTH", return_stats=True)`, the returned `stats` dictionary includes depth-specific fields:

```python
pruned_model, stats = prune_model(
    model=model,
    pruning_type="DEPTH",
    num_layers_to_remove=3,          # or depth_pruning_percentage / layer_indices
    layer_selection_method="last",  # "last" (default), "first", or "custom"
    return_stats=True,
)

print(stats)
# {
#   'original_parameters': int,           # Parameter count before pruning
#   'pruned_parameters': int,             # Parameter count after pruning
#   'reduction': int,                     # Absolute reduction in parameters
#   'percentage_reduction': float,        # Percentage reduction of parameters
#   'original_layer_count': int,          # Layers before pruning
#   'final_layer_count': int,             # Layers after pruning
#   'layers_removed': int,                # Number of removed layers
#   'layer_reduction_percentage': float   # Percentage of layers removed
# }
```

- Depth pruning stats do not include `expansion_rate` (only relevant for MLP/GLU width pruning).
- Internally, stats are captured before modifying the model to ensure correctness and avoid deepcopy issues.

## Depth Pruning

OptiPFair also supports depth pruning, which removes entire transformer layers from models. This is more aggressive than neuron-level pruning but can lead to significant efficiency gains.

### Python API

#### Basic Depth Pruning

```python
from optipfair import prune_model

# Remove 2 layers from the end of the model
pruned_model = prune_model(
    model=model,
    pruning_type="DEPTH",
    num_layers_to_remove=2
)
```

#### Depth Pruning by Percentage

```python
# Remove 25% of layers
pruned_model = prune_model(
    model=model,
    pruning_type="DEPTH",
    depth_pruning_percentage=25.0
)
```

#### Depth Pruning with Specific Layer Indices

```python
# Remove specific layers (e.g., layers 2, 5, and 8)
pruned_model = prune_model(
    model=model,
    pruning_type="DEPTH",
    layer_indices=[2, 5, 8]
)
```

### Command-Line Interface

#### Basic Depth Pruning

```bash
# Remove 2 layers from the end of the model
optipfair prune \
  --model-path meta-llama/Llama-3.2-1B \
  --pruning-type DEPTH \
  --num-layers-to-remove 2 \
  --output-path ./depth-pruned-model
```

#### Depth Pruning by Percentage

```bash
# Remove 25% of layers
optipfair prune \
  --model-path meta-llama/Llama-3.2-1B \
  --pruning-type DEPTH \
  --pruning-percentage 25 \
  --output-path ./depth-pruned-model
```

#### Depth Pruning with Specific Layers

```bash
# Remove specific layers
optipfair prune \
  --model-path meta-llama/Llama-3.2-1B \
  --pruning-type DEPTH \
  --layer-indices "2,5,8" \
  --output-path ./depth-pruned-model
```

## Comparing Pruning Types

### MLP GLU vs Depth Pruning

| Feature | MLP GLU Pruning | Depth Pruning |
|---------|-----------------|---------------|
| **Granularity** | Neuron-level | Layer-level |
| **Aggressiveness** | Moderate | High |
| **Parameter Reduction** | Gradual | Significant |
| **Model Structure** | Preserved | Layers removed |
| **Fine-tuning Need** | Minimal | Recommended |
| **Efficiency Gains** | Moderate | High |

### When to Use Each Method

**Use MLP GLU Pruning when:**
- You want gradual parameter reduction
- You need to preserve model structure
- You have limited time for fine-tuning
- You need precise control over expansion rates

**Use Depth Pruning when:**
- You need significant efficiency gains
- You can afford to fine-tune the model
- You have very large models with many layers
- You need maximum inference speed improvement

## Evaluating Pruned Models

After pruning, you can use OptiPFair's evaluation tools to assess the performance of the pruned model:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from optipfair.evaluation.benchmarks import time_inference, compare_models_inference

# Load original and pruned models
original_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
pruned_model = AutoModelForCausalLM.from_pretrained("./pruned-model")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")

# Compare inference speed
comparison = compare_models_inference(
    original_model,
    pruned_model,
    tokenizer,
    prompts=["Paris is the capital of", "The speed of light is approximately"],
    max_new_tokens=50
)

print(f"Speedup: {comparison['speedup']:.2f}x")
print(f"Tokens per second improvement: {comparison['tps_improvement_percent']:.2f}%")
```

## Layer Importance Analysis

OptiPFair includes functionality to analyze the importance of transformer layers using cosine similarity. This helps identify which layers contribute most to the model's transformations, informing depth pruning decisions.

### Basic Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader
from optipfair import analyze_layer_importance

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

# Prepare your dataset (user responsibility)
# Example with a simple dataset
from datasets import load_dataset
dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train[:100]')

def tokenize_function(examples):
    return tokenizer(
        examples['text'],
        truncation=True,
        padding='max_length',
        max_length=512,
        return_tensors='pt'
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True)
dataloader = DataLoader(tokenized_dataset, batch_size=8)

# Analyze layer importance
importance_scores = analyze_layer_importance(model, dataloader)

# Results: {0: 0.890395, 1: 0.307580, 2: 0.771541, ...}
print(importance_scores)
```

### Advanced Usage

```python
# With manual architecture specification
importance_scores = analyze_layer_importance(
    model=model,
    dataloader=dataloader,
    layers_path='transformer.h',  # For GPT-2 style models
    show_progress=True
)

# Analyze specific layers for depth pruning
# Higher scores indicate layers that transform data more significantly
# Lower scores indicate "passive" layers that could be candidates for removal
sorted_layers = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
print("Most important layers:", sorted_layers[:3])
print("Least important layers:", sorted_layers[-3:])
```

### Multi-Architecture Support

The function automatically detects transformer layers for different architectures:

- **LLaMA/Qwen/Mistral**: `model.layers`
- **GPT-2/DistilGPT2**: `transformer.h`  
- **T5**: `encoder.block` or `decoder.block`
- **BERT**: `encoder.layer`

If automatic detection fails, specify the path manually:

```python
# Manual specification for custom architectures
importance_scores = analyze_layer_importance(
    model=model,
    dataloader=dataloader,
    layers_path='model.custom_transformer_layers'
)
```

### Integration with Depth Pruning

Use importance scores to inform depth pruning decisions:

```python
# Analyze layer importance
importance_scores = analyze_layer_importance(model, dataloader)

# Identify least important layers
sorted_layers = sorted(importance_scores.items(), key=lambda x: x[1])
layers_to_remove = [layer_idx for layer_idx, score in sorted_layers[:4]]

# Apply depth pruning to remove least important layers
pruned_model = prune_model(
    model=model,
    pruning_type="DEPTH",
    layer_indices=layers_to_remove
)
```

### DataLoader Format Support (v0.2.4+)

Starting from OptiPFair v0.2.4, `analyze_layer_importance` automatically handles multiple DataLoader batch formats, making it compatible with both HuggingFace datasets and native PyTorch structures.

#### Supported Batch Formats

**1. Dictionary Format (HuggingFace)**

```python
from datasets import load_dataset
from torch.utils.data import DataLoader

# HuggingFace datasets return dict batches
dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train[:100]')
tokenized = dataset.map(tokenize_function, batched=True)
tokenized.set_format(type='torch', columns=['input_ids', 'attention_mask'])
dataloader = DataLoader(tokenized, batch_size=8)

# Batch format: {'input_ids': tensor, 'attention_mask': tensor}
importance_scores = analyze_layer_importance(model, dataloader)
```

**2. Tuple Format (TensorDataset)**

```python
from torch.utils.data import DataLoader, TensorDataset

# Tokenize texts manually
inputs = tokenizer(
    texts,
    truncation=True,
    padding='max_length',
    max_length=512,
    return_tensors='pt'
)

# TensorDataset returns tuples
dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'])
dataloader = DataLoader(dataset, batch_size=8)

# Batch format: (input_ids, attention_mask)
# Automatically mapped: [0]=input_ids, [1]=attention_mask
importance_scores = analyze_layer_importance(model, dataloader)
```

**3. List Format (Custom Datasets)**

```python
class CustomDataset(Dataset):
    def __getitem__(self, idx):
        return [self.input_ids[idx], self.attention_mask[idx]]

# Batch format: [input_ids, attention_mask]
# Same positional mapping as tuples
importance_scores = analyze_layer_importance(model, dataloader)
```

**4. Single Tensor Format**

```python
# Dataset with only input_ids (no attention_mask)
dataset = TensorDataset(input_ids_tensor)
dataloader = DataLoader(dataset, batch_size=8)

# Batch format: single tensor
# Automatically treated as input_ids
importance_scores = analyze_layer_importance(model, dataloader)
```

#### Positional Mapping for Tuple/List Formats

When using tuple or list batches, elements are automatically mapped to standard transformer arguments:

- `[0]` → `input_ids` (required)
- `[1]` → `attention_mask` (optional)
- `[2]` → `token_type_ids` (optional)
- `[3]` → `position_ids` (optional)
- `[4]` → `head_mask` (optional)
- `[5]` → `inputs_embeds` (optional)

**Note**: All formats are fully backward compatible. Existing code continues to work without modifications.

---

## Fairness-Aware Pruning (NEW in v0.3.0)

OptiPFair v0.3.0 introduces fairness-aware pruning, which combines bias analysis with pruning decisions to create models that are both smaller and potentially less biased.

### Overview

Traditional pruning focuses solely on minimizing performance loss. Fairness-aware pruning adds an additional dimension: identifying and potentially removing neurons that contribute to demographic bias.

The workflow consists of two main steps:

1. **Analyze Neuron Bias**: Identify which neurons contribute most to bias across demographic groups
2. **Compute Fairness Scores**: Combine bias scores with importance scores for balanced pruning decisions

### Step 1: Analyze Neuron Bias

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import optipfair as opf
from optipfair.bias import analyze_neuron_bias

# Load model
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")

# Define prompt pairs that differ only in demographic attributes
prompt_pairs = [
    ("The male nurse was helpful.", "The female nurse was helpful."),
    ("White doctor examined the patient.", "Black doctor examined the patient."),
    ("Young engineer designed the system.", "Old engineer designed the system."),
]

# Analyze per-neuron bias
bias_scores = analyze_neuron_bias(
    model=model,
    tokenizer=tokenizer,
    prompt_pairs=prompt_pairs,
    target_layers=["gate_proj", "up_proj"],  # Analyze these MLP components
    aggregation="mean",                       # "mean" or "max" across tokens
    show_progress=True
)

# bias_scores maps layer names to bias tensors
print(f"Analyzed {len(bias_scores)} layers")
```

### Step 2: Compute Fairness Pruning Scores

```python
from optipfair.bias import compute_fairness_pruning_scores

# Combine bias with importance
fairness_scores = compute_fairness_pruning_scores(
    model=model,
    bias_scores=bias_scores,
    bias_weight=0.45  # Balance fairness (0.0-1.0) vs performance
)

# fairness_scores maps layer indices to pruning score tensors
# Higher scores = safer to prune (low bias + low importance)
for layer_idx, scores in fairness_scores.items():
    safe_neurons = (scores > 0.75).sum().item()
    print(f"Layer {layer_idx}: {safe_neurons} neurons safe to prune")
```

### Understanding bias_weight Parameter

The `bias_weight` parameter controls the trade-off between fairness and performance:

| bias_weight | Use Case |
|-------------|----------|
| **0.0** | Pure performance - ignore bias (standard pruning) |
| **0.2** | Performance-critical - secondary fairness concerns |
| **0.4-0.5** | **Balanced - good compression + fairness (RECOMMENDED)** |
| **0.7** | Fairness-critical - reduce bias even at performance cost |
| **1.0** | Pure fairness - prioritize bias reduction over all |

### Complete Fairness-Aware Pruning Example

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import optipfair as opf
from optipfair.bias import analyze_neuron_bias, compute_fairness_pruning_scores

# 1. Load model
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")

# 2. Define demographic test pairs
prompt_pairs = [
    ("The Christian employee worked hard.", "The Muslim employee worked hard."),
    ("The wealthy student studied diligently.", "The poor student studied diligently."),
]

# 3. Analyze neuron-level bias
print("Step 1: Analyzing neuron bias...")
bias_scores = analyze_neuron_bias(
    model=model,
    tokenizer=tokenizer,
    prompt_pairs=prompt_pairs,
    target_layers=["gate_proj", "up_proj"],
    aggregation="mean",
    show_progress=True
)

# 4. Compute fairness pruning scores
print("Step 2: Computing fairness scores...")
fairness_scores = compute_fairness_pruning_scores(
    model=model,
    bias_scores=bias_scores,
    bias_weight=0.45  # Balanced approach
)

# 5. Analyze which neurons are safe to prune
print("\nStep 3: Analyzing results...")
for layer_idx, scores in fairness_scores.items():
    high_score = (scores > 0.75).sum().item()
    print(f"Layer {layer_idx}: {high_score} neurons are safe to prune (score > 0.75)")

# 6. Perform standard pruning
# (Current implementation - use fairness analysis to guide understanding)
print("\nStep 4: Pruning model...")
pruned_model, stats = opf.prune_model(
    model=model,
    pruning_type="MLP_GLU",
    neuron_selection_method="MAW",
    pruning_percentage=15,
    show_progress=True,
    return_stats=True
)

print(f"\nPruning complete: {stats['percentage_reduction']:.2f}% reduction")
print("Next: Evaluate bias metrics to measure fairness improvement")

# 7. Re-evaluate bias after pruning (optional)
# Re-run analyze_neuron_bias on pruned_model to compare
```

### Practical Tips

**1. Choosing Prompt Pairs**

Create prompt pairs that:
- Differ in exactly ONE demographic attribute
- Are otherwise identical in structure and content
- Cover the demographic dimensions you care about (gender, race, age, religion, etc.)
- Use natural, realistic language

**2. Selecting bias_weight**

Start with `bias_weight=0.4-0.5` for balanced results. Adjust based on:
- If performance drops too much → decrease bias_weight (e.g., 0.3)
- If bias reduction is insufficient → increase bias_weight (e.g., 0.6)

**3. Interpreting Fairness Scores**

- **High scores (>0.75)**: Safe to prune - low bias AND low importance
- **Medium scores (0.4-0.75)**: Moderate risk - evaluate case-by-case  
- **Low scores (<0.4)**: Risky to prune - high bias OR high importance

**4. Example Notebook**

For a complete working example with visualizations, see:
- [examples/fairness_aware_pruning_demo.ipynb](../examples/fairness_aware_pruning_demo.ipynb)

---