"""
Tests for selective layer width pruning functionality.

This module tests the layer_indices parameter for MLP_GLU pruning,
ensuring that only specified layers are pruned while others remain unchanged.
"""

import pytest
import torch
from transformers import AutoModelForCausalLM
from optipfair import prune_model
from optipfair.pruning.utils import get_model_layers


class DummyDataLoader:
    """Minimal dataloader for testing data-driven pruning."""
    def __init__(self, batch_size=2, seq_len=8, num_batches=2):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.num_batches = num_batches
    
    def __iter__(self):
        for _ in range(self.num_batches):
            yield {
                'input_ids': torch.randint(0, 100, (self.batch_size, self.seq_len)),
                'attention_mask': torch.ones(self.batch_size, self.seq_len, dtype=torch.long)
            }
    
    def __len__(self):
        return self.num_batches


@pytest.fixture
def model():
    """Load a small test model (LLaMA architecture)."""
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    return AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="cpu"
    )


def test_selective_pruning_basic(model):
    """Test that only specified layers are pruned."""
    layers = get_model_layers(model)
    num_layers = len(layers)
    
    # Store original intermediate sizes for all layers
    original_sizes = [layer.mlp.gate_proj.out_features for layer in layers]
    
    # Prune only layers 0, 2, 4
    layer_indices = [0, 2, 4]
    pruned_model = prune_model(
        model=model,
        pruning_type="MLP_GLU",
        neuron_selection_method="MAW",
        pruning_percentage=20,
        layer_indices=layer_indices,
        show_progress=False
    )
    
    # Check that only specified layers were pruned
    pruned_layers = get_model_layers(pruned_model)
    for idx, layer in enumerate(pruned_layers):
        current_size = layer.mlp.gate_proj.out_features
        if idx in layer_indices:
            # Should be pruned (smaller)
            assert current_size < original_sizes[idx], \
                f"Layer {idx} should be pruned but has same size"
        else:
            # Should remain unchanged
            assert current_size == original_sizes[idx], \
                f"Layer {idx} should not be pruned but size changed"


def test_selective_pruning_single_layer(model):
    """Test pruning a single layer."""
    layers = get_model_layers(model)
    original_sizes = [layer.mlp.gate_proj.out_features for layer in layers]
    
    # Prune only layer 5
    pruned_model = prune_model(
        model=model,
        pruning_type="MLP_GLU",
        pruning_percentage=30,
        layer_indices=[5],
        show_progress=False
    )
    
    pruned_layers = get_model_layers(pruned_model)
    for idx, layer in enumerate(pruned_layers):
        current_size = layer.mlp.gate_proj.out_features
        if idx == 5:
            assert current_size < original_sizes[idx]
        else:
            assert current_size == original_sizes[idx]


def test_selective_pruning_with_expansion_rate(model):
    """Test selective pruning with expansion_rate parameter."""
    layers = get_model_layers(model)
    original_sizes = [layer.mlp.gate_proj.out_features for layer in layers]
    
    layer_indices = [1, 3, 5]
    pruned_model = prune_model(
        model=model,
        pruning_type="MLP_GLU",
        pruning_percentage=None,  # Must be None when using expansion_rate
        expansion_rate=260,  # Target 260% expansion rate
        layer_indices=layer_indices,
        show_progress=False
    )
    
    pruned_layers = get_model_layers(pruned_model)
    for idx, layer in enumerate(pruned_layers):
        current_size = layer.mlp.gate_proj.out_features
        if idx in layer_indices:
            assert current_size < original_sizes[idx]
        else:
            assert current_size == original_sizes[idx]


def test_selective_pruning_with_expansion_divisor(model):
    """Test selective pruning with expansion_divisor parameter."""
    layers = get_model_layers(model)
    original_sizes = [layer.mlp.gate_proj.out_features for layer in layers]
    
    layer_indices = [0, 10, 20]
    expansion_divisor = 128
    
    pruned_model = prune_model(
        model=model,
        pruning_type="MLP_GLU",
        pruning_percentage=15,
        expansion_divisor=expansion_divisor,
        layer_indices=layer_indices,
        show_progress=False
    )
    
    pruned_layers = get_model_layers(pruned_model)
    for idx, layer in enumerate(pruned_layers):
        current_size = layer.mlp.gate_proj.out_features
        if idx in layer_indices:
            # Should be pruned and divisible by expansion_divisor
            assert current_size < original_sizes[idx]
            assert current_size % expansion_divisor == 0, \
                f"Layer {idx} size {current_size} not divisible by {expansion_divisor}"
        else:
            assert current_size == original_sizes[idx]


def test_selective_pruning_with_dataloader(model):
    """Test selective data-driven pruning with dataloader."""
    layers = get_model_layers(model)
    original_sizes = [layer.mlp.gate_proj.out_features for layer in layers]
    
    dataloader = DummyDataLoader(batch_size=2, seq_len=8, num_batches=3)
    layer_indices = [2, 4, 6]
    
    pruned_model = prune_model(
        model=model,
        pruning_type="MLP_GLU",
        neuron_selection_method="MAW",
        pruning_percentage=25,
        dataloader=dataloader,
        layer_indices=layer_indices,
        show_progress=False
    )
    
    pruned_layers = get_model_layers(pruned_model)
    for idx, layer in enumerate(pruned_layers):
        current_size = layer.mlp.gate_proj.out_features
        if idx in layer_indices:
            assert current_size < original_sizes[idx]
        else:
            assert current_size == original_sizes[idx]


def test_selective_pruning_all_methods(model):
    """Test selective pruning with all neuron selection methods."""
    methods = ["MAW", "VOW", "PON", "L2"]
    layer_indices = [1, 5, 9]
    
    for method in methods:
        layers = get_model_layers(model)
        original_sizes = [layer.mlp.gate_proj.out_features for layer in layers]
        
        pruned_model = prune_model(
            model=model,
            pruning_type="MLP_GLU",
            neuron_selection_method=method,
            pruning_percentage=15,
            layer_indices=layer_indices,
            show_progress=False
        )
        
        pruned_layers = get_model_layers(pruned_model)
        for idx, layer in enumerate(pruned_layers):
            current_size = layer.mlp.gate_proj.out_features
            if idx in layer_indices:
                assert current_size < original_sizes[idx], \
                    f"Method {method}: Layer {idx} should be pruned"
            else:
                assert current_size == original_sizes[idx], \
                    f"Method {method}: Layer {idx} should not be pruned"


def test_selective_pruning_statistics(model):
    """Test that pruning statistics correctly report selective pruning."""
    layer_indices = [0, 5, 10, 15, 20]
    
    pruned_model, stats = prune_model(
        model=model,
        pruning_type="MLP_GLU",
        pruning_percentage=30,
        layer_indices=layer_indices,
        return_stats=True,
        show_progress=False
    )
    
    # Check statistics
    assert "original_parameters" in stats
    assert "pruned_parameters" in stats
    assert "percentage_reduction" in stats
    assert stats["pruned_parameters"] < stats["original_parameters"]
    
    # Check selective pruning info
    assert "pruned_layers" in stats
    assert "total_layers" in stats
    assert stats["pruned_layers"] == len(layer_indices)


def test_invalid_layer_indices(model):
    """Test error handling for invalid layer_indices."""
    layers = get_model_layers(model)
    num_layers = len(layers)
    
    # Test out of range indices
    with pytest.raises(ValueError, match="Invalid layer indices"):
        prune_model(
            model=model,
            pruning_type="MLP_GLU",
            pruning_percentage=20,
            layer_indices=[0, num_layers, num_layers + 1],  # Out of range
            show_progress=False
        )
    
    # Test negative indices
    with pytest.raises(ValueError, match="Invalid layer indices"):
        prune_model(
            model=model,
            pruning_type="MLP_GLU",
            pruning_percentage=20,
            layer_indices=[-1, 0, 5],
            show_progress=False
        )
    
    # Test empty list
    with pytest.raises(ValueError, match="cannot be an empty list"):
        prune_model(
            model=model,
            pruning_type="MLP_GLU",
            pruning_percentage=20,
            layer_indices=[],
            show_progress=False
        )
    
    # Test duplicate indices
    with pytest.raises(ValueError, match="duplicate values"):
        prune_model(
            model=model,
            pruning_type="MLP_GLU",
            pruning_percentage=20,
            layer_indices=[0, 5, 5, 10],
            show_progress=False
        )
    
    # Test non-integer values
    with pytest.raises(TypeError, match="must be integers"):
        prune_model(
            model=model,
            pruning_type="MLP_GLU",
            pruning_percentage=20,
            layer_indices=[0, "5", 10],
            show_progress=False
        )


def test_layer_indices_none_prunes_all(model):
    """Test that layer_indices=None prunes all layers (backward compatibility)."""
    layers = get_model_layers(model)
    original_sizes = [layer.mlp.gate_proj.out_features for layer in layers]
    
    # Prune with layer_indices=None (default behavior)
    pruned_model = prune_model(
        model=model,
        pruning_type="MLP_GLU",
        pruning_percentage=20,
        layer_indices=None,
        show_progress=False
    )
    
    # All layers should be pruned
    pruned_layers = get_model_layers(pruned_model)
    for idx, layer in enumerate(pruned_layers):
        current_size = layer.mlp.gate_proj.out_features
        assert current_size < original_sizes[idx], \
            f"Layer {idx} should be pruned when layer_indices=None"


def test_selective_pruning_preserves_unpruned_weights(model):
    """Test that unpruned layers have identical weights after pruning."""
    layers = get_model_layers(model)
    
    # Store original weights for unpruned layers
    unpruned_indices = [1, 3, 7, 11]
    original_weights = {}
    for idx in unpruned_indices:
        original_weights[idx] = {
            'gate_proj': layers[idx].mlp.gate_proj.weight.data.clone(),
            'up_proj': layers[idx].mlp.up_proj.weight.data.clone(),
            'down_proj': layers[idx].mlp.down_proj.weight.data.clone(),
        }
    
    # Prune different layers
    pruned_model = prune_model(
        model=model,
        pruning_type="MLP_GLU",
        pruning_percentage=25,
        layer_indices=[0, 2, 4, 6, 8, 10],
        show_progress=False
    )
    
    # Check that unpruned layers have identical weights
    pruned_layers = get_model_layers(pruned_model)
    for idx in unpruned_indices:
        assert torch.allclose(
            pruned_layers[idx].mlp.gate_proj.weight.data,
            original_weights[idx]['gate_proj']
        ), f"Layer {idx} gate_proj weights changed"
        
        assert torch.allclose(
            pruned_layers[idx].mlp.up_proj.weight.data,
            original_weights[idx]['up_proj']
        ), f"Layer {idx} up_proj weights changed"
        
        assert torch.allclose(
            pruned_layers[idx].mlp.down_proj.weight.data,
            original_weights[idx]['down_proj']
        ), f"Layer {idx} down_proj weights changed"


def test_selective_pruning_consistency(model):
    """Test that selective pruning produces consistent results."""
    layer_indices = [2, 5, 8]
    
    # Prune twice with same parameters
    pruned_model_1 = prune_model(
        model=model,
        pruning_type="MLP_GLU",
        neuron_selection_method="MAW",
        pruning_percentage=20,
        layer_indices=layer_indices,
        show_progress=False
    )
    
    # Need fresh model for second run
    model2 = AutoModelForCausalLM.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    
    pruned_model_2 = prune_model(
        model=model2,
        pruning_type="MLP_GLU",
        neuron_selection_method="MAW",
        pruning_percentage=20,
        layer_indices=layer_indices,
        show_progress=False
    )
    
    # Check that results are identical
    layers_1 = get_model_layers(pruned_model_1)
    layers_2 = get_model_layers(pruned_model_2)
    
    for idx in range(len(layers_1)):
        size_1 = layers_1[idx].mlp.gate_proj.out_features
        size_2 = layers_2[idx].mlp.gate_proj.out_features
        assert size_1 == size_2, f"Layer {idx} has inconsistent size across runs"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
