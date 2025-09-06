# GGUF to Sharded Safetensors Converter (gguf_to_safetensors_v2)

This script is an advanced conversion tool designed to upconvert any GGUF model (e.g., Q4_K_M, Q8_0) into a sharded FP16 or BF16 Hugging Face model, fully compatible with modern ML tools like `mergekit` and the `transformers` library.

It builds upon the foundational work of the original [gguf_to_safetensors converter by purinnohito](https://github.com/purinnohito/gguf_to_safetensors), extending the concept with critical features for handling very large models on consumer hardware.

## The Goal: Unlocking GGUF Models

Modern open-source AI relies on the ability to merge, modify, and fine-tune existing models. While GGUF is an excellent format for inference, it is a dead-end for model composition. The primary motivation for this script was to "unlock" GGUF-only models, upconverting them to the standard sharded safetensors format so they can participate in the broader model development ecosystem.

## Key Features

This script provides a robust solution for converting even massive 20B+ parameter models.

-   **Upconversion:** Converts low-precision quantized weights (e.g., Q8_0, Q4_K_M) into higher-precision FP16 or BF16 formats.
-   **Correct Tensor Renaming:** Automatically translates tensor names from the GGUF convention (e.g., `blk.0.attn_norm.weight`) to the standard Hugging Face convention (`model.layers.0.input_layernorm.weight`), which is critical for compatibility.
-   **Size-Based Sharding:** Instead of creating one enormous file, the script splits the output into multiple smaller files (shards) of a user-defined size (e.g., 5 GB). This is essential for filesystem compatibility, ease of sharing, and use with tools that expect sharded models.
-   **Memory Efficiency:** The entire conversion is chunked to operate within a limited RAM footprint. The script processes the GGUF tensor by tensor, saving a shard to disk and clearing memory before continuing. This allows for the conversion of models far larger than the available system RAM.
-   **Standardized Naming:** Output shards are automatically named using the Hugging Face standard: `model-0000X-of-0000Y.safetensors`. The script pre-calculates the total number of shards to ensure filenames are generated correctly.

## Requirements

-   Python 3.8+
-   Required Python libraries:
    -   `torch`
    -   `numpy`
    -   `safetensors`
    -   `gguf`

## Setup

1.  **Save the Script:** Save the final Python script as `gguf_to_safetensors.py`.

2.  **Install Dependencies:** Open your terminal or command prompt and install the necessary libraries using pip:
    ```bash
    pip install torch numpy safetensors gguf
    ```

## Usage

The script is run from the command line and accepts several arguments to control the conversion process.

### Basic Command Structure

```bash
python gguf_to_safetensors.py --input <path_to_gguf> --output <base_output_name> [options]
```

### Arguments

-   `--input`: (Required) Path to the source GGUF file you want to convert.
-   `--output`: (Required) Path and base name for the output files. The directory will be created if it doesn't exist. For example, `--output ./my_model/model.safetensors` will create files like `./my_model/model-00001-of-00010.safetensors`.
-   `--shard-size`: (Optional) The maximum size for each shard in Gigabytes (GB). Defaults to `5.0`.
-   `--bf16`: (Optional) Use this flag to convert to BFloat16 instead of the default Float16.

### Examples

**Example 1: Standard Conversion to FP16**

Convert a GGUF file into 5 GB shards with FP16 precision.

```bash
python gguf_to_safetensors.py --input ./models/MyMistral-Q8_0.gguf --output ./output_model/MyMistral-FP16.safetensors --shard-size 5.0
```
This command creates a directory named `output_model` containing the sharded `.safetensors` files.

**Example 2: Fast Windows Conversion**

If the script and model are in the same directory, the command is very simple:

```bash
python gguf_to_safetensors.py --input RandomModel-Q8_0.gguf --output RandomModel-FP16.safetensors
```

**Example 3: Conversion to BF16 with 8 GB Shards**

```bash
python gguf_to_safetensors.py ^
    --input ./models/LargeModel-Q4_K_M.gguf ^
    --output ./converted_models/LargeModel-BF16/model.safetensors ^
    --shard-size 8.0 ^
    --bf16
```
This creates a folder named `LargeModel-BF16` containing all the weight files, split into ~8 GB `.safetensors` shards in BF16 format.

## Post-Conversion: Completing the Model Directory

This script focuses on accurately converting the model weights. After running it, you will have a set of correctly named and formatted `.safetensors` shards.

To make the model fully usable, you must add the necessary configuration and tokenizer files to the same directory:
-   `config.json`
-   `tokenizer.json`
-   `tokenizer_config.json`
-   `special_tokens_map.json`
-   `generation_config.json`
-   `model.safetensors.index.json`

The recommended approach is to **copy these files from the original, non-quantized base model** (e.g., from the official `mistralai/Mistral-7B-v0.1` repository). After copying the files, you will need to generate the `model.safetensors.index.json` file that maps the tensors to your newly created shards. This can be done with a separate utility script.

Another file, `safetensors_meta_ripper_v1.py` is provided, which can generate the `model.safetensors.index.json`, but other files are *not* able to be extracted yet without further codebase improvements.
