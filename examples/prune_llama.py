"""
Example script demonstrating how to prune a LLaMA model using OptiPFair.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from optipfair import prune_model
from optipfair.evaluation.benchmarks import time_inference

# Configuration
MODEL_NAME = "meta-llama/Llama-3.2-1B"  # Change to your model
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PRUNING_PERCENTAGE = 20
OUTPUT_PATH = "./pruned-llama"
TEST_PROMPT = "Paris is the capital of"

def main():
    print(f"Loading model {MODEL_NAME} to {DEVICE}...")
    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if "cuda" in DEVICE else torch.float32,
        device_map=DEVICE
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Test the original model
    print("\nTesting original model generation:")
    test_output = generate_text(model, tokenizer, TEST_PROMPT)
    print(f"Generated text: {test_output}")
    
    # Benchmark original model
    print("\nBenchmarking original model...")
    original_timing = time_inference(model, tokenizer, TEST_PROMPT, max_new_tokens=50)
    print(f"Original tokens per second: {original_timing['tokens_per_second']:.2f}")
    
    # Prune the model
    print(f"\nPruning model with {PRUNING_PERCENTAGE}% pruning...")
    pruned_model, stats = prune_model(
        model=model,
        pruning_type="MLP_GLU",
        neuron_selection_method="MAW",
        pruning_percentage=PRUNING_PERCENTAGE,
        show_progress=True,
        return_stats=True
    )
    
    # Print pruning statistics
    print("\nPruning statistics:")
    print(f"Original parameters: {stats['original_parameters']:,}")
    print(f"Pruned parameters: {stats['pruned_parameters']:,}")
    print(f"Reduction: {stats['reduction']:,} parameters ({stats['percentage_reduction']:.2f}%)")
    
    # Test the pruned model
    print("\nTesting pruned model generation:")
    pruned_output = generate_text(pruned_model, tokenizer, TEST_PROMPT)
    print(f"Generated text: {pruned_output}")
    
    # Benchmark pruned model
    print("\nBenchmarking pruned model...")
    pruned_timing = time_inference(pruned_model, tokenizer, TEST_PROMPT, max_new_tokens=50)
    print(f"Pruned tokens per second: {pruned_timing['tokens_per_second']:.2f}")
    
    # Calculate speedup
    speedup = pruned_timing['tokens_per_second'] / original_timing['tokens_per_second']
    print(f"Speedup: {speedup:.2f}x")
    
    # Save pruned model
    print(f"\nSaving pruned model to {OUTPUT_PATH}...")
    pruned_model.save_pretrained(OUTPUT_PATH)
    tokenizer.save_pretrained(OUTPUT_PATH)
    print("Done!")

def generate_text(model, tokenizer, prompt, max_new_tokens=50):
    """Generate text using the model."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=5,
            early_stopping=True
        )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    main()