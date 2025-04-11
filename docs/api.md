# API Reference

## Core Functions

### `prune_model`

```python
def prune_model(
    model: PreTrainedModel,
    pruning_type: str = "MLP_GLU",
    neuron_selection_method: str = "MAW",
    pruning_percentage: Optional[float] = 10,
    expansion_rate: Optional[float] = None,
    show_progress: bool = True,
    return_stats: bool = False,
) -> Union[PreTrainedModel, Tuple[PreTrainedModel, Dict[str, Any]]]:
    """
    Prune a pre-trained language model using the specified pruning method.
    
    Args:
        model: Pre-trained model to prune
        pruning_type: Type of pruning to apply (currently only "MLP_GLU" is supported)
        neuron_selection_method: Method to calculate neuron importance ("MAW", "VOW", or "PON")
        pruning_percentage: Percentage of neurons to prune (0-100)
        expansion_rate: Target expansion rate in percentage (mutually exclusive with pruning_percentage)
        show_progress: Whether to show progress during pruning
        return_stats: Whether to return pruning statistics along with the model
        
    Returns:
        Pruned model or tuple of (pruned_model, statistics) if return_stats is True
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
    [--pruning-type {MLP_GLU}] 
    [--method {MAW,VOW,PON}] 
    [--pruning-percentage PERCENTAGE] 
    [--expansion-rate RATE] 
    [--device DEVICE] 
    [--dtype {auto,float32,float16,bfloat16}] 
    [--verbose/--quiet]
```

### `analyze`

```bash
optipfair analyze --model-path MODEL_PATH 
    [--device DEVICE]
```