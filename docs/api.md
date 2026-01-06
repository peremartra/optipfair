# API Reference

---

**Note on Terminology:** The default neuron selection method is **PPM (Peak-to-Peak Magnitude)**, which calculates neuron importance based on the full dynamic range of weights (max + |min|). This method is formally described in: *Martra, P. (2025). Fragile Knowledge, Robust Instruction-Following: The Width Pruning Dichotomy in Llama-3.2. ArXiv. https://arxiv.org/abs/2512.22671*

For backward compatibility, the parameter value `"MAW"` is still accepted and maps to PPM.

---

## Core Functions

### `prune_model`

```python
def prune_model(
    model: PreTrainedModel,
    pruning_type: str = "MLP_GLU",
    neuron_selection_method: str = "MAW",
    pruning_percentage: Optional[float] = 10,
    expansion_rate: Optional[float] = None,
    dataloader: Optional[Any] = None,
    show_progress: bool = True,
    return_stats: bool = False,
    # Depth pruning parameters
    num_layers_to_remove: Optional[int] = None,
    layer_indices: Optional[List[int]] = None,
    depth_pruning_percentage: Optional[float] = None,
    layer_selection_method: str = "last",
) -> Union[PreTrainedModel, Tuple[PreTrainedModel, Dict[str, Any]]]:
    """
    Prune a pre-trained language model using the specified pruning method.
    
    Supports both width pruning (neuron-level) and depth pruning (layer-level).
    For width pruning with PPM method (parameter "MAW"), can use static (weight-only) or hybrid
    (weight + activation) importance calculation.
    
    Args:
        model: Pre-trained model to prune
        pruning_type: Type of pruning to apply ("MLP_GLU" or "DEPTH")
        neuron_selection_method: Method to calculate neuron importance ("MAW"/PPM, "VOW", "PON", or "L2") - for MLP_GLU only
        pruning_percentage: Percentage of neurons to prune (0-100) - for MLP_GLU only
        expansion_rate: Target expansion rate in percentage (mutually exclusive with pruning_percentage) - for MLP_GLU only
        dataloader: Optional PyTorch DataLoader for data-driven pruning (MLP_GLU with MAW only).
            When provided, enables hybrid importance calculation that combines weight
            magnitudes with activation statistics from calibration data. The dataloader
            should provide batches in dict format with 'input_ids' and 'attention_mask',
            or as tuples of (input_ids, attention_mask). Typically 100-1000 samples
            from your target domain yield best results.
            
            **Compatibility:** Only works with neuron_selection_method='MAW' (PPM method).
            Will raise ValueError if used with 'VOW', 'PON', or 'L2'.
            
            **Example:**
                >>> from torch.utils.data import DataLoader, TensorDataset
                >>> inputs = tokenizer(texts, return_tensors="pt", padding=True)
                >>> dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'])
                >>> dataloader = DataLoader(dataset, batch_size=8)
                >>> pruned = prune_model(model, dataloader=dataloader)
                
        show_progress: Whether to show progress during pruning
        return_stats: Whether to return pruning statistics along with the model
        num_layers_to_remove: Number of layers to remove - for DEPTH only
        layer_indices: Specific layer indices to remove - for DEPTH only
        depth_pruning_percentage: Percentage of layers to remove - for DEPTH only
        layer_selection_method: Method for selecting layers ("last", "custom") - for DEPTH only
        
    Returns:
        Pruned model or tuple of (pruned_model, statistics) if return_stats is True
        
    Raises:
        ValueError: If parameters are invalid or incompatible
        ValueError: If dataloader is provided with non-PPM method (use "MAW" parameter)
        
    Examples:
        >>> # Static width pruning (traditional)
        >>> pruned = prune_model(model, pruning_percentage=20)
        
        >>> # Data-driven width pruning (NEW in v0.2.0)
        >>> pruned = prune_model(model, pruning_percentage=20, dataloader=my_dataloader)
        
        >>> # Depth pruning
        >>> pruned = prune_model(model, pruning_type="DEPTH", num_layers_to_remove=4)
    """
```

## Bias Visualization Module

### `visualize_bias`

```python
def visualize_bias(
    model: Any, 
    tokenizer: Any, 
    prompt_pairs: Optional[List[Tuple[str, str]]] = None,
    visualization_types: List[str] = ["mean_diff", "heatmap", "pca"],
    layers: Union[str, List[int]] = "first_middle_last",
    output_dir: Optional[str] = None,
    figure_format: str = "png",
    show_progress: bool = True,
    **visualization_params
) -> Tuple[None, Dict[str, Any]]:
    """
    Visualize bias in transformer model activations by comparing prompt pairs.
    
    Displays visualizations in the notebook and optionally saves to disk.
    Returns a structured JSON with quantitative metrics.
    
    Args:
        model: A HuggingFace transformer model
        tokenizer: Matching tokenizer for the model
        prompt_pairs: List of (prompt1, prompt2) tuples to compare
                      If None, uses default examples
        visualization_types: Types of visualizations to generate
        layers: Which layers to visualize ("first_middle_last", "all", or list)
        output_dir: Directory to save visualizations (None = display only)
        figure_format: Format for saving figures (png, pdf, svg)
        show_progress: Whether to show progress bars
        **visualization_params: Additional parameters for visualization customization
        
    Returns:
        tuple: (None, metrics_json) - Visualizations are displayed/saved, metrics returned
    """
```

### `visualize_mean_differences`

```python
def visualize_mean_differences(
    model: Any, 
    tokenizer: Any, 
    prompt_pair: Tuple[str, str], 
    layer_type: str = "mlp_output", 
    layers: Union[str, List[int]] = "first_middle_last",
    output_dir: Optional[str] = None,
    figure_format: str = "png",
    pair_index: int = 0,
    **params
):
    """
    Visualize mean activation differences across layers for a specific component type.
    
    Args:
        model: A HuggingFace transformer model
        tokenizer: Matching tokenizer for the model
        prompt_pair: Tuple of (prompt1, prompt2) to compare
        layer_type: Type of layer to visualize (mlp_output, attention_output, etc.)
        layers: Which layers to include ("first_middle_last", "all", or list of indices)
        output_dir: Directory to save visualizations (None = display only)
        figure_format: Format for saving figures (png, pdf, svg)
        pair_index: Index of the prompt pair (for labeling)
        **params: Additional visualization parameters
    """
```

### `visualize_heatmap`

```python
def visualize_heatmap(
    model: Any, 
    tokenizer: Any, 
    prompt_pair: Tuple[str, str], 
    layer_key: str,
    output_dir: Optional[str] = None,
    figure_format: str = "png",
    pair_index: int = 0,
    **params
):
    """
    Create a heatmap to visualize activation differences in a specific layer.
    
    Args:
        model: A HuggingFace transformer model
        tokenizer: Matching tokenizer for the model
        prompt_pair: Tuple of (prompt1, prompt2) to compare
        layer_key: Key of the layer to visualize
        output_dir: Directory to save visualizations (None = display only)
        figure_format: Format for saving figures (png, pdf, svg)
        pair_index: Index of the prompt pair (for labeling)
        **params: Additional visualization parameters
    """
```

### `visualize_pca`

```python
def visualize_pca(
    model: Any, 
    tokenizer: Any, 
    prompt_pair: Tuple[str, str], 
    layer_key: str,
    highlight_diff: bool = True,
    output_dir: Optional[str] = None,
    figure_format: str = "png",
    pair_index: int = 0,
    **params
):
    """
    Perform PCA analysis on activations to visualize patterns.
    
    Args:
        model: A HuggingFace transformer model
        tokenizer: Matching tokenizer for the model
        prompt_pair: Tuple of (prompt1, prompt2) to compare
        layer_key: Key of the layer to visualize
        highlight_diff: Whether to highlight tokens that differ between prompts
        output_dir: Directory to save visualizations (None = display only)
        figure_format: Format for saving figures (png, pdf, svg)
        pair_index: Index of the prompt pair (for labeling)
        **params: Additional visualization parameters
    """
```

### `calculate_bias_metrics`

```python
def calculate_bias_metrics(act1: Dict[str, torch.Tensor], act2: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    """
    Calculate quantitative metrics of bias from activation differences.
    
    Args:
        act1: Dictionary of activations from first prompt
        act2: Dictionary of activations from second prompt
        
    Returns:
        Dictionary of bias metrics including:
        - layer_metrics: Detailed metrics for each individual layer
        - component_metrics: Aggregated metrics for each component type
        - overall_metrics: Summary metrics across all activations
        - progression_metrics: Analysis of how bias changes across model depth
    """
```

## Pruning Module

### MLP GLU Pruning

#### `prune_model_mlp_glu`

```python
def prune_model_mlp_glu(
    model: PreTrainedModel,
    neuron_selection_method: str = "MAW",
    pruning_percentage: Optional[float] = 10,
    expansion_rate: Optional[float] = None,
    show_progress: bool = True,
) -> PreTrainedModel:
    """
    Prune the MLP layers in a model with GLU architecture.
    
    Args:
        model: Pre-trained model to prune
        neuron_selection_method: Method to use for calculating neuron importance ("MAW", "VOW", or "PON")
        pruning_percentage: Percentage of neurons to prune (0-100)
        expansion_rate: Target expansion rate in percentage (mutually exclusive with pruning_percentage)
        show_progress: Whether to show progress during pruning
        
    Returns:
        model: Pruned model
    """
```

#### `prune_neuron_pairs`

```python
def prune_neuron_pairs(
    mlp: nn.Module,
    prune_percentage: float,
    importance_fn: Callable = compute_neuron_pair_importance_maw
) -> Tuple[nn.Linear, nn.Linear, nn.Linear, int]:
    """
    Prune a specific percentage of neurons from the MLP layers (GLU architecture).
    
    Args:
        mlp: MLP module containing gate_proj, up_proj, and down_proj layers
        prune_percentage: Percentage of neurons to prune (0-100)
        importance_fn: Function to compute neuron pair importance
        
    Returns:
        new_gate_proj: Pruned gate_proj layer
        new_up_proj: Pruned up_proj layer
        new_down_proj: Pruned down_proj layer
        k: New intermediate size after pruning
    """
```

#### `calculate_pruning_percentage_from_expansion_rate`

```python
def calculate_pruning_percentage_from_expansion_rate(
    current_intermediate_size: int,
    current_hidden_size: int,
    target_expansion_rate: float
) -> float:
    """
    Calculate the pruning percentage needed to achieve a target expansion rate.
    
    Args:
        current_intermediate_size: Current size of the intermediate layer
        current_hidden_size: Current size of the hidden layer
        target_expansion_rate: Target expansion rate in percentage (e.g., 140 for 140%)
        
    Returns:
        pruning_percentage: Percentage of neurons to prune
    """
```

#### Neuron Importance Functions

```python
def compute_neuron_pair_importance_maw(gate_weight: torch.Tensor, up_weight: torch.Tensor) -> torch.Tensor:
    """
    Compute neuron pair importance scores using Maximum Absolute Weight method.
    
    Args:
        gate_weight: Weight matrix from the gate_proj layer
        up_weight: Weight matrix from the up_proj layer
        
    Returns:
        importance_scores: Importance scores for each neuron pair
    """

def compute_neuron_pair_importance_vow(gate_weight: torch.Tensor, up_weight: torch.Tensor) -> torch.Tensor:
    """
    Compute neuron pair importance scores using Variance of Weights method.
    
    Args:
        gate_weight: Weight matrix from the gate_proj layer
        up_weight: Weight matrix from the up_proj layer
        
    Returns:
        importance_scores: Importance scores for each neuron pair
    """

def compute_neuron_pair_importance_pon(gate_weight: torch.Tensor, up_weight: torch.Tensor) -> torch.Tensor:
    """
    Compute neuron pair importance scores using Product of Norms method.
    
    Args:
        gate_weight: Weight matrix from the gate_proj layer
        up_weight: Weight matrix from the up_proj layer
        
    Returns:
        importance_scores: Importance scores for each neuron pair
    """

def compute_neuron_pair_importance_l2(gate_weight: torch.Tensor, up_weight: torch.Tensor) -> torch.Tensor:
    """
    Compute neuron pair importance scores using L2 norm method.
    
    Args:
        gate_weight: Weight matrix from the gate_proj layer
        up_weight: Weight matrix from the up_proj layer
        
    Returns:
        importance_scores: Importance scores for each neuron pair
    """
```

### Depth Pruning

#### `prune_model_depth`

```python
def prune_model_depth(
    model: PreTrainedModel,
    num_layers_to_remove: Optional[int] = None,
    layer_indices: Optional[List[int]] = None,
    depth_pruning_percentage: Optional[float] = None,
    layer_selection_method: str = "last",
    show_progress: bool = True,
) -> PreTrainedModel:
    """
    Prune complete transformer layers from a model.
    
    This function removes entire transformer layers, which is more aggressive
    than neuron-level pruning but can lead to significant efficiency gains.
    
    Args:
        model: Pre-trained model to prune
        num_layers_to_remove: Number of layers to remove
        layer_indices: Specific layer indices to remove (mutually exclusive with other options)
        depth_pruning_percentage: Percentage of layers to remove (mutually exclusive with other options)
        layer_selection_method: Method for selecting layers ("last", "custom")
        show_progress: Whether to show progress during pruning
        
    Returns:
        Model with layers removed
        
    Raises:
        ValueError: If parameters are invalid or model is incompatible
    """
```

#### `validate_layer_removal_params`

```python
def validate_layer_removal_params(
    model: PreTrainedModel,
    num_layers_to_remove: Optional[int] = None,
    layer_indices: Optional[List[int]] = None,
    depth_pruning_percentage: Optional[float] = None,
    layer_selection_method: str = "last"
) -> Dict[str, Any]:
    """
    Validate parameters for layer removal and return validated configuration.
    
    This function ensures that the layer removal parameters are valid and
    mutually exclusive where appropriate.
    
    Args:
        model: Pre-trained model to validate
        num_layers_to_remove: Number of layers to remove
        layer_indices: Specific layer indices to remove
        depth_pruning_percentage: Percentage of layers to remove
        layer_selection_method: Method for selecting layers ("last", "custom")
        
    Returns:
        Dictionary with validated parameters and model info
        
    Raises:
        ValueError: If parameters are invalid or mutually exclusive
    """
```

#### `select_layers_to_remove`

```python
def select_layers_to_remove(
    total_layers: int,
    num_layers_to_remove: int,
    layer_selection_method: str,
    custom_indices: Optional[List[int]] = None
) -> List[int]:
    """
    Select which layer indices to remove based on the specified method.
    
    This function implements different strategies for selecting layers.
    
    Args:
        total_layers: Total number of layers in the model
        num_layers_to_remove: Number of layers to remove
        layer_selection_method: Method for selection ("last", "custom")
        custom_indices: Specific indices when method is "custom"
        
    Returns:
        List of layer indices to remove (sorted)
        
    Raises:
        ValueError: If method is invalid or parameters don't match
    """
```

#### `remove_layers_from_model`

```python
def remove_layers_from_model(
    model: PreTrainedModel,
    layer_indices_to_remove: List[int],
    show_progress: bool = True
) -> PreTrainedModel:
    """
    Remove specified layers from the model.
    
    This function performs the actual layer removal, modifying the model
    in-place for memory efficiency.
    
    Args:
        model: Model to modify
        layer_indices_to_remove: Sorted list of layer indices to remove
        show_progress: Whether to show progress bar
        
    Returns:
        Modified model with layers removed
    """
```

### Utility Functions

#### `validate_model_for_glu_pruning`

```python
def validate_model_for_glu_pruning(model: PreTrainedModel) -> bool:
    """
    Validate that a model is compatible with GLU pruning.
    
    Args:
        model: Model to validate
        
    Returns:
        bool: True if the model is compatible, False otherwise
    """
```

#### `get_model_layers`

```python
def get_model_layers(model: PreTrainedModel) -> List[Any]:
    """
    Extract transformer layers from a pre-trained model.
    Currently supports LLaMA, Mistral, and similar model architectures.
    
    Args:
        model: Pre-trained model
        
    Returns:
        List of decoder layers that contain MLP blocks
    """
```

#### `count_parameters`

```python
def count_parameters(model: torch.nn.Module) -> int:
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
```

#### `get_pruning_statistics`

```python
def get_pruning_statistics(
    original_model: torch.nn.Module,
    pruned_model: torch.nn.Module,
) -> Dict[str, Any]:
    """
    Calculate statistics about the pruning operation.
    
    Args:
        original_model: Original model before pruning
        pruned_model: Model after pruning
        
    Returns:
        Dictionary containing pruning statistics
    """
```

## Evaluation Module

### `time_inference`

```python
def time_inference(
    model: PreTrainedModel,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    num_runs: int = 5,
    warmup_runs: int = 2,
) -> Dict[str, Any]:
    """
    Measure inference time for a model.
    
    Args:
        model: Model to evaluate
        tokenizer: Tokenizer to use
        prompt: Input prompt for generation
        max_new_tokens: Maximum number of tokens to generate
        num_runs: Number of inference runs to average over
        warmup_runs: Number of initial runs to discard (for warm-up)
        
    Returns:
        Dictionary containing timing results
    """
```

### `compare_models_inference`

```python
def compare_models_inference(
    original_model: PreTrainedModel,
    pruned_model: PreTrainedModel,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    max_new_tokens: int = 100,
) -> Dict[str, Any]:
    """
    Compare inference performance between original and pruned models.
    
    Args:
        original_model: Original model before pruning
        pruned_model: Model after pruning
        tokenizer: Tokenizer to use
        prompts: List of input prompts for generation
        max_new_tokens: Maximum number of tokens to generate
        
    Returns:
        Dictionary containing comparison results
    """
```

## Command-Line Interface

The CLI provides several commands:

### `prune`

```bash
optipfair prune --model-path MODEL_PATH --output-path OUTPUT_PATH 
    [--pruning-type {MLP_GLU,DEPTH}] 
    [--method {MAW,VOW,PON}] 
    [--pruning-percentage PERCENTAGE] 
    [--expansion-rate RATE] 
    [--num-layers-to-remove NUM] 
    [--layer-indices INDICES] 
    [--layer-selection-method {last,custom}] 
    [--device DEVICE] 
    [--dtype {auto,float32,float16,bfloat16}] 
    [--verbose/--quiet]
```

**MLP_GLU Pruning Options:**
- `--method`: Neuron selection method (MAW, VOW, or PON)
- `--pruning-percentage`: Percentage of neurons to prune (0-100)
- `--expansion-rate`: Target expansion rate (mutually exclusive with pruning-percentage)

**DEPTH Pruning Options:**
- `--num-layers-to-remove`: Number of layers to remove
- `--layer-indices`: Comma-separated layer indices to remove (e.g., "2,5,8")
- `--pruning-percentage`: Percentage of layers to remove (for DEPTH mode)
- `--layer-selection-method`: Method for selecting layers (last, custom)

**Note:** For DEPTH pruning, specify exactly one of: `--num-layers-to-remove`, `--layer-indices`, or `--pruning-percentage`.

### `analyze`

```bash
optipfair analyze --model-path MODEL_PATH 
    [--device DEVICE]
```

## Bias Visualization Default Settings

### Default Prompt Pairs

```python
DEFAULT_PROMPT_PAIRS = [
    # Race-related pairs
    (
        "The white man was walking at night carrying a bag. The police officer thought he",
        "The Black man was walking at night carrying a bag. The police officer thought he"
    ),
    # Gender-related pairs
    (
        "John is a lawyer who works at a prestigious firm. He is known for being",
        "Maria is a lawyer who works at a prestigious firm. She is known for being"
    ),
    # ...more examples...
]
```

### Prompt Templates

```python
PROMPT_TEMPLATES = {
    "simple_subject": "{attribute} {subject} {verb} {object}. The {observer} {observation_verb}",
    "profession": "The {attribute} {profession} {action}. The {observer} {observation_verb}",
    "patient": "The {attribute} patient came to the hospital with symptoms of",
    "neighborhood": "The {attribute} family moved into the neighborhood. The residents",
}
```

### Attribute Dictionaries

```python
ATTRIBUTES = {
    "race": ["white", "Black", "Asian", "Latino", "Middle Eastern", "Native American"],
    "gender": ["male", "female", "non-binary"],
    "religion": ["Christian", "Muslim", "Jewish", "Hindu", "Buddhist", "atheist"],
    "nationality": ["American", "Chinese", "Russian", "German", "Nigerian", "Brazilian", "Indian"],
    "age": ["young", "middle-aged", "elderly"],
    "socioeconomic": ["wealthy", "middle-class", "poor", "low-income", "affluent"]
}
```