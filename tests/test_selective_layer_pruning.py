"""
Tests for layer_indices with MLP_GLU width pruning.

The model config stores a single intermediate_size shared by every layer, so
width pruning must keep all MLPs at the same size. Selective width pruning
(pruning only some layers) is rejected before the model is modified, because
the resulting model could not be saved and reloaded.
"""

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from optipfair import prune_model
from optipfair.pruning.mlp_glu import prune_model_mlp_glu
from optipfair.pruning.utils import get_model_layers

NUM_LAYERS = 4
INTERMEDIATE = 256


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
    """Small random LLaMA model (no download needed)."""
    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=INTERMEDIATE,
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=4,
        num_key_value_heads=4,
    )
    return LlamaForCausalLM(config).eval()


def _mlp_sizes(model):
    return [layer.mlp.gate_proj.out_features for layer in get_model_layers(model)]


@pytest.mark.parametrize("layer_indices", [[0], [0, 2], [1, 2, 3]])
def test_selective_width_pruning_rejected(model, layer_indices):
    """Pruning only some layers raises ValueError and leaves the model untouched."""
    original_weight = get_model_layers(model)[0].mlp.gate_proj.weight.clone()

    with pytest.raises(ValueError, match="Selective width pruning is not supported"):
        prune_model(
            model=model,
            pruning_type="MLP_GLU",
            pruning_percentage=25,
            layer_indices=layer_indices,
            show_progress=False
        )

    assert _mlp_sizes(model) == [INTERMEDIATE] * NUM_LAYERS
    assert model.config.intermediate_size == INTERMEDIATE
    assert torch.equal(get_model_layers(model)[0].mlp.gate_proj.weight, original_weight)


def test_selective_width_pruning_rejected_with_dataloader(model):
    """The check runs before calibration, so data-driven pruning is rejected too."""
    with pytest.raises(ValueError, match="Selective width pruning is not supported"):
        prune_model(
            model=model,
            pruning_type="MLP_GLU",
            neuron_selection_method="MAW",
            pruning_percentage=25,
            dataloader=DummyDataLoader(),
            layer_indices=[0, 1],
            show_progress=False
        )
    assert _mlp_sizes(model) == [INTERMEDIATE] * NUM_LAYERS


def test_layer_indices_covering_all_layers_equals_none(model):
    """A list with every layer (in any order) behaves exactly like layer_indices=None."""
    reference = LlamaForCausalLM(model.config).eval()
    reference.load_state_dict(model.state_dict())

    prune_model(
        model=model,
        pruning_type="MLP_GLU",
        pruning_percentage=25,
        layer_indices=list(reversed(range(NUM_LAYERS))),
        show_progress=False
    )
    prune_model(
        model=reference,
        pruning_type="MLP_GLU",
        pruning_percentage=25,
        layer_indices=None,
        show_progress=False
    )

    for layer, ref_layer in zip(get_model_layers(model), get_model_layers(reference)):
        assert torch.equal(layer.mlp.gate_proj.weight, ref_layer.mlp.gate_proj.weight)
        assert torch.equal(layer.mlp.down_proj.weight, ref_layer.mlp.down_proj.weight)


def test_layer_indices_none_prunes_all_uniformly(model):
    """layer_indices=None prunes every layer to the same size and updates the config."""
    prune_model(
        model=model,
        pruning_type="MLP_GLU",
        pruning_percentage=25,
        expansion_divisor=32,
        layer_indices=None,
        show_progress=False
    )

    sizes = _mlp_sizes(model)
    assert len(set(sizes)) == 1
    assert sizes[0] < INTERMEDIATE
    assert sizes[0] % 32 == 0
    assert model.config.intermediate_size == sizes[0]


def test_pruned_model_reloads(model, tmp_path):
    """The pruned model saves and reloads with the standard class and identical logits."""
    prune_model(
        model=model,
        pruning_type="MLP_GLU",
        pruning_percentage=25,
        show_progress=False
    )
    model.save_pretrained(tmp_path)
    reloaded = LlamaForCausalLM.from_pretrained(tmp_path).eval()

    input_ids = torch.randint(0, 128, (1, 6))
    with torch.no_grad():
        assert torch.allclose(model(input_ids).logits, reloaded(input_ids).logits, atol=1e-5)


def test_heterogeneous_model_rejected(model):
    """A model whose MLPs already have different sizes cannot be width pruned."""
    mlp = get_model_layers(model)[1].mlp
    mlp.gate_proj = torch.nn.Linear(64, 128, bias=False)
    mlp.up_proj = torch.nn.Linear(64, 128, bias=False)
    mlp.down_proj = torch.nn.Linear(128, 64, bias=False)

    with pytest.raises(ValueError, match="different intermediate sizes"):
        prune_model_mlp_glu(model, pruning_percentage=25, show_progress=False)


def test_invalid_layer_indices(model):
    """Malformed layer_indices still produce specific errors."""
    with pytest.raises(ValueError, match="Invalid layer indices"):
        prune_model(model=model, pruning_type="MLP_GLU", pruning_percentage=20,
                    layer_indices=[0, NUM_LAYERS], show_progress=False)

    with pytest.raises(ValueError, match="Invalid layer indices"):
        prune_model(model=model, pruning_type="MLP_GLU", pruning_percentage=20,
                    layer_indices=[-1, 0], show_progress=False)

    with pytest.raises(ValueError, match="cannot be an empty list"):
        prune_model(model=model, pruning_type="MLP_GLU", pruning_percentage=20,
                    layer_indices=[], show_progress=False)

    with pytest.raises(ValueError, match="duplicate values"):
        prune_model(model=model, pruning_type="MLP_GLU", pruning_percentage=20,
                    layer_indices=[0, 1, 1, 2], show_progress=False)

    with pytest.raises(TypeError, match="must be integers"):
        prune_model(model=model, pruning_type="MLP_GLU", pruning_percentage=20,
                    layer_indices=[0, "1", 2, 3], show_progress=False)
