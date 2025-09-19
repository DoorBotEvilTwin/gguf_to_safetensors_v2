import os
import argparse
import json
from gguf import GGUFReader
from typing import List, Dict, Any

def extract_and_save_tokenizer_files(gguf_path: str, output_dir: str) -> None:
    """
    Extracts tokenizer metadata from a GGUF file and saves it as
    tokenizer.json, tokenizer_config.json, and special_tokens_map.json.
    """
    print(f"Loading GGUF file for tokenizer metadata: {gguf_path}")
    reader = GGUFReader(gguf_path, 'r')

    # --- Extract raw metadata from GGUF ---
    try:
        vocab_list_raw = reader.get_field("tokenizer.ggml.tokens").parts[0]
        merges_list = reader.get_field("tokenizer.ggml.merges").parts[0]
        
        bos_token_id = int(reader.get_field("tokenizer.ggml.bos_token_id").parts[0])
        eos_token_id = int(reader.get_field("tokenizer.ggml.eos_token_id").parts[0])
        unk_token_id = int(reader.get_field("tokenizer.ggml.unknown_token_id").parts[0])
        padding_token_id = int(reader.get_field("tokenizer.ggml.padding_token_id").parts[0])
        
        model_max_length = int(reader.get_field("llama.context_length").parts[0])
        
        # Optional: chat template
        chat_template = None
        try:
            chat_template = reader.get_field("tokenizer.chat_template").parts[0]
        except KeyError:
            pass # Chat template might not always be present
            
        # Convert raw vocab bytes to strings
        vocab_list = [token.decode('utf-8', errors='ignore') for token in vocab_list_raw]

    except Exception as e:
        print(f"Fatal Error: Could not extract essential tokenizer metadata from GGUF. Error: {e}")
        return

    # --- 1. Create tokenizer.json ---
    try:
        # The vocab for tokenizer.json needs to be a dict of {token_string: id}
        vocab_dict = {token: i for i, token in enumerate(vocab_list)}

        tokenizer_json_data = {
            "version": "1.0",
            "truncation": None,
            "padding": None,
            "added_tokens": [], # GGUF doesn't typically store this separately
            "normalizer": {
                "type": "Sequence",
                "normalizers": [
                    {"type": "NFC"},
                    {"type": "Replace", "pattern": " ", "content": " "}, # Example, adjust if needed
                ]
            },
            "pre_tokenizer": {
                "type": "ByteLevel", # Common for BPE models like GPT2/Llama
                "add_prefix_space": False, # Based on tokenizer.ggml.add_space_prefix = 0
                "splits_by_unicode_script": False,
                "trim_offsets": True
            },
            "post_processor": {
                "type": "ByteLevel",
                "truncation": None,
                "padding": None,
                "add_prefix_space": False,
                "trim_offsets": True
            },
            "decoder": {
                "type": "ByteLevel",
                "add_prefix_space": False,
                "trim_offsets": True
            },
            "model": {
                "type": "BPE",
                "vocab": vocab_dict,
                "merges": merges_list,
                "dropout": None,
                "unk_token": vocab_list[unk_token_id] if 0 <= unk_token_id < len(vocab_list) else "<unk>"
            }
        }
        
        tokenizer_json_path = os.path.join(output_dir, "tokenizer.json")
        with open(tokenizer_json_path, 'w', encoding='utf-8') as f:
            json.dump(tokenizer_json_data, f, indent=None, separators=(',', ':')) # Compact format
        print(f"Created tokenizer.json at {tokenizer_json_path}")
    except Exception as e:
        print(f"Warning: Could not create tokenizer.json. Error: {e}")

    # --- 2. Create tokenizer_config.json ---
    try:
        tokenizer_config_data = {
            "model_max_length": model_max_length,
            "padding_side": "left", # Common default for causal models
            "tokenizer_class": "LlamaTokenizer", # Mistral uses LlamaTokenizer
            "clean_up_tokenization_spaces": False,
            "add_bos_token": bool(reader.get_field("tokenizer.ggml.add_bos_token").parts[0]),
            "add_eos_token": bool(reader.get_field("tokenizer.ggml.add_eos_token").parts[0]),
        }
        if chat_template:
            tokenizer_config_data["chat_template"] = chat_template

        tokenizer_config_path = os.path.join(output_dir, "tokenizer_config.json")
        with open(tokenizer_config_path, 'w', encoding='utf-8') as f:
            json.dump(tokenizer_config_data, f, indent=2)
        print(f"Created tokenizer_config.json at {tokenizer_config_path}")
    except Exception as e:
        print(f"Warning: Could not create tokenizer_config.json. Error: {e}")

    # --- 3. Create special_tokens_map.json ---
    try:
        special_tokens_map_data = {}
        
        def get_token_string(token_id, default_str):
            if 0 <= token_id < len(vocab_list):
                return vocab_list[token_id]
            return default_str

        special_tokens_map_data["bos_token"] = get_token_string(bos_token_id, "<|begin_of_text|>")
        special_tokens_map_data["eos_token"] = get_token_string(eos_token_id, "<|end_of_text|>")
        special_tokens_map_data["unk_token"] = get_token_string(unk_token_id, "<unk>")
        special_tokens_map_data["pad_token"] = get_token_string(padding_token_id, "<pad>")

        special_tokens_map_path = os.path.join(output_dir, "special_tokens_map.json")
        with open(special_tokens_map_path, 'w', encoding='utf-8') as f:
            json.dump(special_tokens_map_data, f, indent=2)
        print(f"Created special_tokens_map.json at {special_tokens_map_path}")
    except Exception as e:
        print(f"Warning: Could not create special_tokens_map.json. Error: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Extracts tokenizer metadata from a GGUF file and saves it as Hugging Face tokenizer files."
    )
    parser.add_argument("--gguf-file", required=True, help="Path to the original GGUF file to read metadata from.")
    parser.add_argument("--output-dir", required=True, help="Path to the directory where the tokenizer files will be saved.")
    args = parser.parse_args()

    if not os.path.isfile(args.gguf_file):
        print(f"Error: GGUF file not found at {args.gguf_file}")
        return
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Created output directory: {args.output_dir}")

    extract_and_save_tokenizer_files(args.gguf_file, args.output_dir)
    
    print("\nTokenizer file generation complete.")

if __name__ == "__main__":
    main()