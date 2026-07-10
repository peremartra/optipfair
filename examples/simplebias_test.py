# simple_test.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from optipfair.bias import visualize_mean_differences

# Use a small model for quick testing
model_name = "meta-llama/Llama-3.2-1B"  # Or another model you have access to
try:
    # Load with minimal resources for testing
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16, 
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Test with a simple prompt pair
    prompt_pair = (
        "The white doctor examined the patient. The nurse thought",
        "The Black doctor examined the patient. The nurse thought"
    )

    # Call the visualization function
    visualize_mean_differences(
        model, 
        tokenizer, 
        prompt_pair, 
        layer_type="mlp_output", 
        layers="first_middle_last"
    )
    
    print("Visualization test successful!")
except Exception as e:
    print(f"Error encountered: {e}")