import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# --- YOU MUST UPDATE THESE TWO PATHS ---
# Path to the directory where your FP32 model is stored locally
input_dir = "A:\LLM\.cache\huggingface\hub\models--wzhouad--gemma-2-9b-it-WPO-HB"

# Path to the directory where the converted FP16 model will be saved
output_dir = "A:\LLM\.cache\huggingface\hub\models--wzhouad--gemma-2-9b-it-WPO-HB_FP16"
# -------------------------------------

# Make sure the output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load the tokenizer from the local path
print(f"Loading tokenizer from {input_dir}...")
tokenizer = AutoTokenizer.from_pretrained(input_dir)

# Load the model in FP32 from the local path
print(f"Loading FP32 model from {input_dir}...")
model = AutoModelForCausalLM.from_pretrained(
    input_dir,
    torch_dtype=torch.float32,
    device_map="cpu"
    # device_map="auto" # use this if you have enough GPU VRAM
)

# Convert the model to FP16 and save it to the new local directory
print("Converting model to FP16 and saving to disk...")
model.half().save_pretrained(
    output_dir,
    safe_serialization=True,
    max_shard_size="5GB"
)
tokenizer.save_pretrained(output_dir)

print(f"Model successfully converted and saved to {output_dir}")
print("You can now use this new FP16 model in your mergekit config.yaml.")