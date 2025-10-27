"""
Validation Script - Phase D: Compare Static vs Hybrid Pruning
Model: HuggingFaceTB/SmolLM2-135M
Device: MPS (Apple Silicon)
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader, TensorDataset
from optipfair import prune_model

print("=" * 80)
print("PHASE D VALIDATION: Static vs Hybrid Pruning Comparison")
print("=" * 80)

# Configuration
MODEL_NAME = "HuggingFaceTB/SmolLM2-135M"
DEVICE = "mps"  # Apple Silicon GPU
PRUNING_PERCENTAGES = [10, 20]
BATCH_SIZE = 4
NUM_SAMPLES = 16
MAX_LENGTH = 128

print(f"\n📋 Configuration:")
print(f"   Model: {MODEL_NAME}")
print(f"   Device: {DEVICE}")
print(f"   Pruning percentages: {PRUNING_PERCENTAGES}")
print(f"   Calibration samples: {NUM_SAMPLES}")

# Helper function to count parameters
def count_parameters(model):
    """Count total parameters in model"""
    return sum(p.numel() for p in model.parameters())

def get_intermediate_size(model):
    """Get intermediate size from first MLP layer"""
    from optipfair.pruning.utils import get_model_layers
    layers = get_model_layers(model)
    if layers:
        return layers[0].mlp.gate_proj.out_features
    return None

# Prepare calibration dataloader
print(f"\n📊 Creating calibration dataloader...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

texts = [
    "The capital of France is Paris.",
    "Machine learning is a subset of artificial intelligence.",
    "Python is a popular programming language.",
    "The Earth orbits around the Sun.",
    "Water boils at 100 degrees Celsius.",
    "Shakespeare wrote Romeo and Juliet.",
    "The Pacific Ocean is the largest ocean.",
    "Mathematics is the study of numbers.",
    "The Great Wall of China is visible from space.",
    "DNA contains genetic information.",
    "Albert Einstein developed the theory of relativity.",
    "The human brain has billions of neurons.",
    "Climate change affects global temperatures.",
    "Photosynthesis converts light into energy.",
    "The Moon orbits the Earth.",
    "Gravity keeps planets in orbit.",
][:NUM_SAMPLES]

inputs = tokenizer(
    texts,
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=MAX_LENGTH
)

dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'])
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"✅ Dataloader ready: {len(dataloader)} batches")

# Results storage
results = {}

print("\n" + "=" * 80)
print("RUNNING VALIDATION TESTS")
print("=" * 80)

for pruning_pct in PRUNING_PERCENTAGES:
    print(f"\n{'─' * 80}")
    print(f"Testing {pruning_pct}% Pruning")
    print(f"{'─' * 80}")
    
    # =========================================================================
    # Test 1: Static MAW (without dataloader)
    # =========================================================================
    print(f"\n🔹 Test 1: Static MAW ({pruning_pct}%)")
    print("   Loading model...")
    
    model_static = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        device_map=DEVICE
    )
    
    original_params = count_parameters(model_static)
    original_intermediate = get_intermediate_size(model_static)
    
    print(f"   Original parameters: {original_params:,}")
    print(f"   Original intermediate size: {original_intermediate}")
    print(f"   Pruning...")
    
    pruned_static, stats_static = prune_model(
        model=model_static,
        neuron_selection_method="MAW",
        pruning_percentage=pruning_pct,
        dataloader=None,  # ← STATIC
        show_progress=False,
        return_stats=True
    )
    
    static_params = count_parameters(pruned_static)
    static_intermediate = get_intermediate_size(pruned_static)
    
    print(f"   ✅ Static pruning complete")
    print(f"      Pruned parameters: {static_params:,}")
    print(f"      Reduction: {stats_static['reduction']:,} ({stats_static['percentage_reduction']:.2f}%)")
    print(f"      New intermediate size: {static_intermediate}")
    print(f"      Expansion rate: {stats_static['expansion_rate']:.2f}%")
    
    # Clean up
    del model_static
    torch.mps.empty_cache()
    
    # =========================================================================
    # Test 2: Hybrid MAW (with dataloader)
    # =========================================================================
    print(f"\n🔹 Test 2: Hybrid MAW ({pruning_pct}%)")
    print("   Loading model...")
    
    model_hybrid = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        device_map=DEVICE
    )
    
    print(f"   Original parameters: {original_params:,}")
    print(f"   Original intermediate size: {original_intermediate}")
    print(f"   Running calibration + pruning...")
    
    pruned_hybrid, stats_hybrid = prune_model(
        model=model_hybrid,
        neuron_selection_method="MAW",
        pruning_percentage=pruning_pct,
        dataloader=dataloader,  # ← HYBRID
        show_progress=False,
        return_stats=True
    )
    
    hybrid_params = count_parameters(pruned_hybrid)
    hybrid_intermediate = get_intermediate_size(pruned_hybrid)
    
    print(f"   ✅ Hybrid pruning complete")
    print(f"      Pruned parameters: {hybrid_params:,}")
    print(f"      Reduction: {stats_hybrid['reduction']:,} ({stats_hybrid['percentage_reduction']:.2f}%)")
    print(f"      New intermediate size: {hybrid_intermediate}")
    print(f"      Expansion rate: {stats_hybrid['expansion_rate']:.2f}%")
    
    # Clean up
    del model_hybrid
    torch.mps.empty_cache()
    
    # =========================================================================
    # Compare Results
    # =========================================================================
    print(f"\n📊 Comparison for {pruning_pct}% pruning:")
    print(f"   {'Metric':<30} {'Static':<20} {'Hybrid':<20} {'Match?':<10}")
    print(f"   {'-' * 80}")
    
    params_match = static_params == hybrid_params
    intermediate_match = static_intermediate == hybrid_intermediate
    
    print(f"   {'Total Parameters':<30} {static_params:<20,} {hybrid_params:<20,} {'✅ YES' if params_match else '❌ NO':<10}")
    print(f"   {'Intermediate Size':<30} {static_intermediate:<20} {hybrid_intermediate:<20} {'✅ YES' if intermediate_match else '❌ NO':<10}")
    print(f"   {'Expansion Rate':<30} {stats_static['expansion_rate']:<20.2f} {stats_hybrid['expansion_rate']:<20.2f} {'✅ YES' if abs(stats_static['expansion_rate'] - stats_hybrid['expansion_rate']) < 0.01 else '❌ NO':<10}")
    
    # Store results
    results[pruning_pct] = {
        'static': {
            'parameters': static_params,
            'intermediate_size': static_intermediate,
            'expansion_rate': stats_static['expansion_rate']
        },
        'hybrid': {
            'parameters': hybrid_params,
            'intermediate_size': hybrid_intermediate,
            'expansion_rate': stats_hybrid['expansion_rate']
        },
        'sizes_match': params_match and intermediate_match
    }

# ============================================================================
# Final Summary
# ============================================================================
print("\n" + "=" * 80)
print("VALIDATION SUMMARY")
print("=" * 80)

all_passed = True

for pruning_pct, result in results.items():
    status = "✅ PASS" if result['sizes_match'] else "❌ FAIL"
    print(f"\n{pruning_pct}% Pruning: {status}")
    
    if result['sizes_match']:
        print(f"   Both methods produced identical model sizes:")
        print(f"   - Parameters: {result['static']['parameters']:,}")
        print(f"   - Intermediate size: {result['static']['intermediate_size']}")
        print(f"   - Expansion rate: {result['static']['expansion_rate']:.2f}%")
    else:
        print(f"   ⚠️  Size mismatch detected:")
        print(f"   Static:  {result['static']['parameters']:,} params, intermediate={result['static']['intermediate_size']}")
        print(f"   Hybrid:  {result['hybrid']['parameters']:,} params, intermediate={result['hybrid']['intermediate_size']}")
        all_passed = False

print("\n" + "=" * 80)
if all_passed:
    print("🎉 ALL VALIDATIONS PASSED!")
    print("Static and Hybrid pruning produce identical model architectures.")
else:
    print("⚠️  VALIDATION FAILED!")
    print("Static and Hybrid pruning produced different model sizes.")
print("=" * 80)