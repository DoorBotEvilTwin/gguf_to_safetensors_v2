import os
import argparse
import torch
import numpy as np
from safetensors.torch import save_file
from safetensors import safe_open
from typing import Dict, Tuple, List
from gguf import GGUFReader, dequantize

def load_gguf_and_extract_metadata(gguf_path: str) -> Tuple[GGUFReader, List[Dict]]:
    """Load GGUF file and extract metadata for all tensors."""
    print(f"Loading GGUF file: {gguf_path}")
    reader = GGUFReader(gguf_path, 'r')
    tensors_metadata = []
    for tensor in reader.tensors:
        tensor_metadata = {
            'name': tensor.name,
            'shape': tuple(tensor.shape.tolist()),
            'n_elements': tensor.n_elements,
            'n_bytes': tensor.n_bytes,
            'type': tensor.tensor_type,
        }
        tensors_metadata.append(tensor_metadata)
    return reader, tensors_metadata

def get_dequantized_tensor_size_in_bytes(tensor_info: Dict, use_bf16: bool) -> int:
    """Calculates the size of a tensor after it has been dequantized to FP16 or BF16."""
    bytes_per_element = 2
    return tensor_info['n_elements'] * bytes_per_element

def get_hf_name(gguf_name: str) -> str:
    """Translates a GGUF tensor name to its Hugging Face equivalent for Llama/Mistral models."""
    name_map = {
        "token_embd.weight": "model.embed_tokens.weight",
        "output_norm.weight": "model.norm.weight",
        "output.weight": "lm_head.weight",
    }
    if gguf_name in name_map:
        return name_map[gguf_name]

    if gguf_name.startswith("blk."):
        parts = gguf_name.split('.')
        layer_num = parts[1]
        layer_part = ".".join(parts[2:])
        
        block_map = {
            "attn_norm.weight": "input_layernorm.weight",
            "ffn_norm.weight": "post_attention_layernorm.weight",
            "attn_q.weight": "self_attn.q_proj.weight",
            "attn_k.weight": "self_attn.k_proj.weight",
            "attn_v.weight": "self_attn.v_proj.weight",
            "attn_output.weight": "self_attn.o_proj.weight",
            "ffn_gate.weight": "mlp.gate_proj.weight",
            "ffn_up.weight": "mlp.up_proj.weight",
            "ffn_down.weight": "mlp.down_proj.weight",
        }
        
        if layer_part in block_map:
            return f"model.layers.{layer_num}.{block_map[layer_part]}"

    print(f"Warning: No mapping found for tensor '{gguf_name}'. Using original name.")
    return gguf_name

def convert_gguf_to_safetensors_by_size(gguf_path: str, output_path: str, use_bf16: bool, shard_size_gb: float):
    """Converts a GGUF file to .safetensors, sharding and renaming tensors for HF compatibility."""
    reader, tensors_metadata = load_gguf_and_extract_metadata(gguf_path)
    print(f"Extracted metadata for {len(tensors_metadata)} tensors from GGUF file.")

    shard_size_bytes = int(shard_size_gb * 1024**3)
    print(f"Target shard size set to ~{shard_size_gb} GB ({shard_size_bytes} bytes).")

    output_dir = os.path.dirname(output_path)
    if not output_dir:
        output_dir = "."
    base_name = os.path.basename(output_path).replace('.safetensors', '')
    
    tensors_in_current_chunk: dict[str, torch.Tensor] = {}
    current_chunk_size_bytes = 0
    num_chunks = 0
    
    total_shards = 0
    temp_size = 0
    for tensor_info in tensors_metadata:
        dequantized_size = get_dequantized_tensor_size_in_bytes(tensor_info, use_bf16)
        if temp_size > 0 and (temp_size + dequantized_size) > shard_size_bytes:
            total_shards += 1
            temp_size = 0
        temp_size += dequantized_size
    if temp_size > 0:
        total_shards += 1
    print(f"Model will be split into {total_shards} shards.")

    for i, tensor_info in enumerate(tensors_metadata):
        gguf_tensor_name = tensor_info['name']
        dequantized_size = get_dequantized_tensor_size_in_bytes(tensor_info, use_bf16)

        if current_chunk_size_bytes > 0 and (current_chunk_size_bytes + dequantized_size) > shard_size_bytes:
            num_chunks += 1
            chunk_path = os.path.join(output_dir, f"{base_name}-{num_chunks:05d}-of-{total_shards:05d}.safetensors")
            
            print(f"\nCurrent chunk size ({current_chunk_size_bytes / 1024**3:.2f} GB) exceeds limit.")
            print(f"Saving chunk {num_chunks} with {len(tensors_in_current_chunk)} tensors to {chunk_path}...\n")
            save_file(tensors_in_current_chunk, chunk_path)
            
            tensors_in_current_chunk.clear()
            current_chunk_size_bytes = 0

        tensor_data = reader.get_tensor(i)
        weights_np = dequantize(tensor_data.data, tensor_data.tensor_type).copy()
        target_dtype = torch.bfloat16 if use_bf16 else torch.float16
        
        try:
            weights_tensor = torch.from_numpy(weights_np).to(target_dtype)
        except Exception as e:
            print(f"Warning: Could not convert {gguf_tensor_name} directly. Error: {e}. Using float32 fallback.")
            weights_tensor = torch.from_numpy(weights_np.astype(np.float32)).to(target_dtype)

        # --- CORRECTED RENAMING LOGIC ---
        hf_tensor_name = get_hf_name(gguf_tensor_name)
        
        print(f"Processed tensor ({i+1}/{len(tensors_metadata)}): {gguf_tensor_name} -> {hf_tensor_name} | Size: {dequantized_size/1024**2:.2f} MB")
        
        tensors_in_current_chunk[hf_tensor_name] = weights_tensor
        current_chunk_size_bytes += dequantized_size
        
        del weights_np
        del tensor_data

    if tensors_in_current_chunk:
        num_chunks += 1
        chunk_path = os.path.join(output_dir, f"{base_name}-{num_chunks:05d}-of-{total_shards:05d}.safetensors")
        print(f"\nSaving final chunk {num_chunks} with {len(tensors_in_current_chunk)} tensors to {chunk_path}...\n")
        save_file(tensors_in_current_chunk, chunk_path)

    print("All tensors have been dequantized, renamed, and saved into sharded safetensor files.")

def main():
    parser = argparse.ArgumentParser(
        description="Convert GGUF to HF-compatible sharded safetensors, renaming tensors correctly."
    )
    parser.add_argument("--input", required=True, help="Path to the input GGUF file.")
    parser.add_argument("--output", required=True, help="Base path for the final output sharded .safetensors files.")
    parser.add_argument("--bf16", action="store_true", help="Convert tensors to BF16 format instead of the default FP16.")
    parser.add_argument(
        "--shard-size", 
        type=float, 
        default=5.0, 
        help="Maximum size of each shard in Gigabytes (GB). Default: 5.0"
    )
    args = parser.parse_args()

    convert_gguf_to_safetensors_by_size(args.input, args.output, args.bf16, args.shard_size)

if __name__ == "__main__":
    main()