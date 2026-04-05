"""
Tests for expansion_divisor parameter functionality
"""

import pytest
import torch
from transformers import AutoModelForCausalLM
from optipfair import prune_model


@pytest.fixture
def small_model():
    """Load a small model for testing"""
    model_name = "meta-llama/Llama-3.2-1B"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    return model


class TestExpansionDivisorValidation:
    """Test validation of expansion_divisor parameter"""
    
    def test_valid_divisor_values(self, small_model):
        """Test that valid divisor values are accepted"""
        valid_divisors = [32, 64, 128, 256]
        
        for divisor in valid_divisors:
            # Should not raise an error
            pruned_model = prune_model(
                model=small_model,
                pruning_type="MLP_GLU",
                pruning_percentage=10,
                expansion_divisor=divisor,
                show_progress=False
            )
            assert pruned_model is not None
    
    def test_invalid_divisor_value(self, small_model):
        """Test that invalid divisor values raise ValueError"""
        invalid_divisors = [16, 48, 512, 1024, -128, 0]
        
        for divisor in invalid_divisors:
            with pytest.raises(ValueError, match="expansion_divisor must be one of"):
                prune_model(
                    model=small_model,
                    pruning_type="MLP_GLU",
                    pruning_percentage=10,
                    expansion_divisor=divisor,
                    show_progress=False
                )
    
    def test_divisor_alone_raises_error(self, small_model):
        """Test that using expansion_divisor alone raises ValueError"""
        with pytest.raises(ValueError, match="expansion_divisor cannot be used alone"):
            prune_model(
                model=small_model,
                pruning_type="MLP_GLU",
                pruning_percentage=None,
                expansion_rate=None,
                expansion_divisor=128,
                show_progress=False
            )
    
    def test_divisor_with_pruning_percentage(self, small_model):
        """Test that expansion_divisor works with pruning_percentage"""
        # Should not raise an error
        pruned_model = prune_model(
            model=small_model,
            pruning_type="MLP_GLU",
            pruning_percentage=20,
            expansion_divisor=128,
            show_progress=False
        )
        assert pruned_model is not None
    
    def test_divisor_with_expansion_rate(self, small_model):
        """Test that expansion_divisor works with expansion_rate"""
        # Should not raise an error
        pruned_model = prune_model(
            model=small_model,
            pruning_type="MLP_GLU",
            expansion_rate=200,
            expansion_divisor=128,
            show_progress=False
        )
        assert pruned_model is not None


class TestExpansionDivisorRounding:
    """Test rounding behavior of expansion_divisor"""
    
    def test_rounding_with_divisor(self, small_model):
        """Test that intermediate size is divisible by divisor"""
        divisors = [32, 64, 128, 256]
        
        for divisor in divisors:
            pruned_model, stats = prune_model(
                model=small_model,
                pruning_type="MLP_GLU",
                pruning_percentage=20,
                expansion_divisor=divisor,
                show_progress=False,
                return_stats=True
            )
            
            # Get intermediate size from first layer
            if hasattr(pruned_model, 'model') and hasattr(pruned_model.model, 'layers'):
                intermediate_size = pruned_model.model.layers[0].mlp.gate_proj.out_features
                
                # Check divisibility
                assert intermediate_size % divisor == 0, (
                    f"Intermediate size {intermediate_size} is not divisible by {divisor}"
                )
    
    def test_rounding_near_boundary(self, small_model):
        """Test rounding behavior near divisor boundaries"""
        # Use a pruning percentage that should result in a size near a boundary
        pruned_model, stats = prune_model(
            model=small_model,
            pruning_type="MLP_GLU",
            pruning_percentage=15,
            expansion_divisor=128,
            show_progress=False,
            return_stats=True
        )
        
        # Get intermediate size
        if hasattr(pruned_model, 'model') and hasattr(pruned_model.model, 'layers'):
            intermediate_size = pruned_model.model.layers[0].mlp.gate_proj.out_features
            
            # Should be divisible by 128
            assert intermediate_size % 128 == 0
            
            # Should be positive
            assert intermediate_size > 0
    
    def test_no_rounding_when_divisor_none(self, small_model):
        """Test that no rounding occurs when divisor is None"""
        # Prune with divisor=None
        pruned_no_divisor, stats_no_divisor = prune_model(
            model=small_model,
            pruning_type="MLP_GLU",
            pruning_percentage=20,
            expansion_divisor=None,
            show_progress=False,
            return_stats=True
        )
        
        # Prune with same percentage but with divisor
        pruned_with_divisor, stats_with_divisor = prune_model(
            model=small_model,
            pruning_type="MLP_GLU",
            pruning_percentage=20,
            expansion_divisor=128,
            show_progress=False,
            return_stats=True
        )
        
        # Get intermediate sizes
        if hasattr(pruned_no_divisor, 'model') and hasattr(pruned_no_divisor.model, 'layers'):
            size_no_divisor = pruned_no_divisor.model.layers[0].mlp.gate_proj.out_features
            size_with_divisor = pruned_with_divisor.model.layers[0].mlp.gate_proj.out_features
            
            # They might be different (unless size_no_divisor was already divisible by 128)
            if size_no_divisor % 128 != 0:
                assert size_no_divisor != size_with_divisor


class TestExpansionDivisorWithMethods:
    """Test expansion_divisor with different neuron selection methods"""
    
    def test_divisor_with_maw(self, small_model):
        """Test expansion_divisor with MAW method"""
        pruned_model, stats = prune_model(
            model=small_model,
            pruning_type="MLP_GLU",
            neuron_selection_method="MAW",
            pruning_percentage=20,
            expansion_divisor=128,
            show_progress=False,
            return_stats=True
        )
        
        assert pruned_model is not None
        assert stats['pruned_parameters'] < stats['original_parameters']
    
    def test_divisor_with_vow(self, small_model):
        """Test expansion_divisor with VOW method"""
        pruned_model, stats = prune_model(
            model=small_model,
            pruning_type="MLP_GLU",
            neuron_selection_method="VOW",
            pruning_percentage=20,
            expansion_divisor=64,
            show_progress=False,
            return_stats=True
        )
        
        assert pruned_model is not None
        assert stats['pruned_parameters'] < stats['original_parameters']
    
    def test_divisor_with_pon(self, small_model):
        """Test expansion_divisor with PON method"""
        pruned_model, stats = prune_model(
            model=small_model,
            pruning_type="MLP_GLU",
            neuron_selection_method="PON",
            pruning_percentage=20,
            expansion_divisor=256,
            show_progress=False,
            return_stats=True
        )
        
        assert pruned_model is not None
        assert stats['pruned_parameters'] < stats['original_parameters']
    
    def test_divisor_with_l2(self, small_model):
        """Test expansion_divisor with L2 method"""
        pruned_model, stats = prune_model(
            model=small_model,
            pruning_type="MLP_GLU",
            neuron_selection_method="L2",
            pruning_percentage=20,
            expansion_divisor=256,
            show_progress=False,
            return_stats=True
        )
        
        assert pruned_model is not None
        assert stats['pruned_parameters'] < stats['original_parameters']


class TestExpansionDivisorEdgeCases:
    """Test edge cases for expansion_divisor"""
    
    def test_high_pruning_with_large_divisor(self, small_model):
        """Test high pruning percentage with large divisor"""
        # This might result in very small intermediate size
        pruned_model, stats = prune_model(
            model=small_model,
            pruning_type="MLP_GLU",
            pruning_percentage=50,
            expansion_divisor=256,
            show_progress=False,
            return_stats=True
        )
        
        # Should still work and produce valid model
        assert pruned_model is not None
        
        # Check that intermediate size is positive and divisible
        if hasattr(pruned_model, 'model') and hasattr(pruned_model.model, 'layers'):
            intermediate_size = pruned_model.model.layers[0].mlp.gate_proj.out_features
            assert intermediate_size > 0
            assert intermediate_size % 256 == 0
    
    def test_low_pruning_with_small_divisor(self, small_model):
        """Test low pruning percentage with small divisor"""
        pruned_model, stats = prune_model(
            model=small_model,
            pruning_type="MLP_GLU",
            pruning_percentage=5,
            expansion_divisor=32,
            show_progress=False,
            return_stats=True
        )
        
        assert pruned_model is not None
        
        # Check divisibility
        if hasattr(pruned_model, 'model') and hasattr(pruned_model.model, 'layers'):
            intermediate_size = pruned_model.model.layers[0].mlp.gate_proj.out_features
            assert intermediate_size % 32 == 0

    def test_tiny_pruning_with_divisor_raises_no_effective_pruning(self, small_model):
        """Test tiny pruning with divisor fails if resulting size is not reduced."""
        with pytest.raises(ValueError, match="No effective pruning"):
            prune_model(
                model=small_model,
                pruning_type="MLP_GLU",
                pruning_percentage=0.01,
                expansion_divisor=128,
                show_progress=False,
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
