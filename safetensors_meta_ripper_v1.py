import os
import argparse
import json
import glob
from safetensors import safe_open
from gguf import GGUFReader
from gguf.constants import Keys
from typing import List, Dict, Any

def create_safetensors_index(shards_dir: str, output_dir: str) -> None:
    """Creates the model.safetensors.index.json file by scanning shard files."""
    shard_pattern = os.path.join(shards_dir, '*.safetensors')
    shard_files = sorted(glob.glob(shard_pattern))

    if not shard_files:
        print(f"Error: No .safetensors files found in directory: {shards_dir}")
        return

    print(f"Found {len(shard_files)} shard files to index.")

    index_data: Dict[str, Any] = {"metadata": {}, "weight_map": {}}
    total_size = 0

    for shard_file in shard_files:
        shard_basename = os.path.basename(shard_file)
        try:
            with safe_open(shard_file, framework="pt", device="cpu") as f:
                for tensor_name in f.keys():
                    index_data["weight_map"][tensor_name] = shard_basename
            
            shard_size = os.path.getsize(shard_file)
            total_size += shard_size
        except Exception as e:
            print(f"Warning: Could not process shard {shard_basename}. Error: {e}")
            continue

    index_data["metadata"]["total_size"] = total_size
    
    index_filepath = os.path.join(output_dir, "model.safetensors.index.json")
    try:
        with open(index_filepath, 'w', encoding='utf-8') as f:
            json.dump(index_data, f, indent=2)
        print(f"Successfully created safetensors index file: {index_filepath}")
    except Exception as e:
        print(f"Error: Failed to write index file. Error: {e}")

def extract_and_save_gguf_configs(reader: GGUFReader, output_dir: str) -> None:
    """Extracts metadata from GGUF and saves config, tokenizer, and generation files."""
    
    config = {}
    # --- config.json ---
    try:
        arch = reader.get_field(Keys.General.ARCHITECTURE).name.lower()
        model_type_map = {"llama": "llama", "mistral": "mistral", "gemma": "gemma"}
        model_type = model_type_map.get(arch, arch)

        config = {
            "architectures": [arch.capitalize()],
            "model_type": model_type,
            "hidden_size": reader.get_int_value(f"{model_type}.embedding_length"),
            "intermediate_size": reader.get_int_value(f"{model_type}.feed_forward_length"),
            "num_attention_heads": reader.get_int_value(f"{model_type}.attention.head_count"),
            "num_hidden_layers": reader.get_int_value(f"{model_type}.block_count"),
            "num_key_value_heads": reader.get_int_value(f"{model_type}.attention.head_count_kv"),
            "rms_norm_eps": reader.get_float_value(f"{model_type}.attention.layer_norm_rms_epsilon"),
            "vocab_size": len(reader.get_field(Keys.Tokenizer.VOCAB)),
            "rope_theta": reader.get_float_value(f"{model_type}.rope.freq_base"),
            "max_position_embeddings": reader.get_int_value(f"{model_type}.context_length"),
        }
        with open(os.path.join(output_dir, "config.json"), 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
        print("Created config.json")
    except Exception as e:
        print(f"Warning: Could not create config.json. Some values may be missing. Error: {e}")

    # --- tokenizer_config.json ---
    try:
        tokenizer_config = {
            "model_max_length": config.get("max_position_embeddings", 4096),
            "padding_side": "left",
            "tokenizer_class": "LlamaTokenizer",
        }
        # Add chat template if it exists
        try:
            chat_template = reader.get_str_value("tokenizer.chat_template")
            tokenizer_config["chat_template"] = chat_template
        except (KeyError, ValueError):
            pass # Field does not exist
        
        with open(os.path.join(output_dir, "tokenizer_config.json"), 'w', encoding='utf-8') as f:
            json.dump(tokenizer_config, f, indent=2)
        print("Created tokenizer_config.json")
    except Exception as e:
        print(f"Warning: Could not create tokenizer_config.json. Error: {e}")

    # --- tokenizer.json ---
    try:
        vocab = [item.piece for item in reader.get_field(Keys.Tokenizer.VOCAB)]
        merges = reader.get_field(Keys.Tokenizer.MERGES)
        
        tokenizer_data = {
            "version": "1.0",
            "model": {
                "type": "BPE",
                "vocab": {token: i for i, token in enumerate(vocab)},
                "merges": merges,
            },
            "added_tokens": [],
        }
        with open(os.path.join(output_dir, "tokenizer.json"), 'w', encoding='utf-8') as f:
            json.dump(tokenizer_data, f, indent=None, separators=(',', ':'))
        print("Created tokenizer.json")
    except Exception as e:
        print(f"Warning: Could not create tokenizer.json. Error: {e}")

    # --- special_tokens_map.json ---
    try:
        special_map = {}
        # Use a helper to avoid crashing on missing keys
        def add_special_token(key_name, gguf_id_key):
            try:
                token_id = reader.get_int_value(gguf_id_key)
                token_str = vocab[token_id]
                special_map[key_name] = token_str
            except (KeyError, ValueError, IndexError):
                pass
        
        add_special_token("bos_token", "tokenizer.ggml.bos_token_id")
        add_special_token("eos_token", "tokenizer.ggml.eos_token_id")
        add_special_token("unk_token", "tokenizer.ggml.unknown_token_id")
        
        with open(os.path.join(output_dir, "special_tokens_map.json"), 'w', encoding='utf-8') as f:
            json.dump(special_map, f, indent=2)
        print("Created special_tokens_map.json")
    except Exception as e:
        print(f"Warning: Could not create special_tokens_map.json. Error: {e}")

    # --- generation_config.json ---
    try:
        gen_config = {"_from_model_config": True}
        try:
            gen_config["bos_token_id"] = reader.get_int_value("tokenizer.ggml.bos_token_id")
            gen_config["eos_token_id"] = reader.get_int_value("tokenizer.ggml.eos_token_id")
        except (KeyError, ValueError):
            pass
        
        with open(os.path.join(output_dir, "generation_config.json"), 'w', encoding='utf-8') as f:
            json.dump(gen_config, f, indent=2)
        print("Created generation_config.json")
    except Exception as e:
        print(f"Warning: Could not create generation_config.json. Error: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Generate safetensors index and config files for a sharded model directory."
    )
    parser.add_argument(
        "--gguf-file", 
        required=True, 
        help="Path to the original GGUF file to read metadata from."
    )
    parser.add_argument(
        "--shards-dir", 
        required=True, 
        help="Path to the directory containing the sharded .safetensors files."
    )
    args = parser.parse_args()

    if not os.path.isfile(args.gguf_file):
        print(f"Error: GGUF file not found at {args.gguf_file}")
        return
    if not os.path.isdir(args.shards_dir):
        print(f"Error: Shards directory not found at {args.shards_dir}")
        return

    print(f"Loading GGUF metadata from: {args.gguf_file}")
    reader = GGUFReader(args.gguf_file, 'r')

    # Generate config files from GGUF header and save them to the shards directory
    extract_and_save_gguf_configs(reader, args.shards_dir)

    # Generate the safetensors index from the actual shard files
    create_safetensors_index(args.shards_dir, args.shards_dir)
    
    print("\nMetadata ripping complete.")

if __name__ == "__main__":
    main()