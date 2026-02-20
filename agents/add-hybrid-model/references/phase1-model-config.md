# Phase 1: Model Config JSON

## Goal

Create the HuggingFace-format config file so aiconfigurator can read the model's architecture parameters.

## Steps

### 1. Obtain the model's `config.json`

```bash
# Download from HuggingFace
wget https://huggingface.co/<org>/<model>/raw/main/config.json

# Or copy from a local model directory
cp /path/to/model/config.json /tmp/
```

Focus on these fields:

**Standard fields (always required):**
- `architectures` — list with one architecture class name, e.g. `["MiMoV2FlashForCausalLM"]`
- `num_hidden_layers`, `hidden_size`, `num_attention_heads`, `num_key_value_heads`
- `head_dim` (or derivable as `hidden_size / num_attention_heads`)
- `intermediate_size`, `vocab_size`, `max_position_embeddings`

**MoE fields (if model has MoE layers):**
- `num_experts_per_tok` — number of experts selected per token
- `num_local_experts` or `n_routed_experts` — total number of experts
- `moe_intermediate_size` — FFN hidden size inside each expert

**Hybrid-specific fields (new per-layer pattern fields):**
- Layer type patterns: `hybrid_layer_pattern`, `layer_types`, `full_attention_interval`
- Alternative attention dims: `swa_num_kv_heads`, `swa_head_dim`, `sliding_window_size`
- Linear attention dims: `linear_num_kv_heads`, `linear_key_head_dim`, `linear_value_head_dim`
- Shared expert: `shared_expert_intermediate_size`, `shared_intermediate_size`

**MTP (Multi-Token Prediction) fields:**
- `num_mtp_modules` or `num_nextn_predict_layers` — number of extra draft tokens
- `use_mtp` — boolean flag

**Quantization (if pre-quantized):**
- `quantization_config` — nested object with `quant_type`, `weight_bits`, etc.

### 2. Name the file correctly

```
src/aiconfigurator/model_configs/<Org>--<Model>_config.json
```

Replace `/` with `--` in the HuggingFace model ID.

| HuggingFace ID | Config filename |
|---|---|
| `XiaomiMiMo/MiMo-V2-Flash` | `XiaomiMiMo--MiMo-V2-Flash_config.json` |
| `Qwen/Qwen3.5-397B-A17B` | `Qwen--Qwen3.5-397B-A17B_config.json` |
| `nvidia/DeepSeek-V3.2-NVFP4` | `nvidia--DeepSeek-V3.2-NVFP4_config.json` |

If the model has a separate quantization config (e.g. NVFP4), also create:
```
src/aiconfigurator/model_configs/<Org>--<Model>_hf_quant_config.json
```

### 3. Identify new hybrid-specific fields

Compare against existing model configs and mark fields that are new:

```json
{
  "architectures": ["MiMoV2FlashForCausalLM"],
  "num_hidden_layers": 28,
  "hidden_size": 2048,
  "num_attention_heads": 16,
  "num_key_value_heads": 8,             // standard field
  "head_dim": 128,                      // standard field
  "intermediate_size": 2048,
  "vocab_size": 151936,
  "max_position_embeddings": 32768,
  "num_experts_per_tok": 2,             // standard MoE field
  "num_local_experts": 64,              // standard MoE field
  "moe_intermediate_size": 1024,        // standard MoE field
  "hybrid_layer_pattern": "GSGS...",    // <- new hybrid field
  "moe_layer_freq": 1,                  // <- new hybrid field
  "swa_num_kv_heads": 8,                // <- new hybrid field
  "swa_head_dim": 64,                   // <- new hybrid field
  "sliding_window_size": 4096           // <- new hybrid field
}
```

### 4. Minimal required content

At minimum, include every field that `_parse_hf_config_json()` reads. Check `utils.py` line ~449–507 for the full list of parsed fields. Missing fields cause a `KeyError` at parse time.

Fields read unconditionally:
```python
config["architectures"][0]
config["num_hidden_layers"]
config["hidden_size"]
config["num_attention_heads"]
config["vocab_size"]
config["max_position_embeddings"]
```

Fields read with `.get()` (optional, have defaults):
```python
config.get("num_key_value_heads") or 0
config.get("intermediate_size") or 0
config.get("head_dim") or config.get("attention_head_dim") or (hidden_size // n)
config.get("num_experts_per_tok", 0)
config.get("num_local_experts") or config.get("n_routed_experts") or 0
config.get("moe_intermediate_size", 0) or config.get("intermediate_size", 0)
```

## Example: MiMo-V2-Flash

```json
{
  "architectures": ["MiMoV2FlashForCausalLM"],
  "model_type": "mimo_v2_flash",
  "num_hidden_layers": 28,
  "hidden_size": 2048,
  "num_attention_heads": 16,
  "num_key_value_heads": 8,
  "head_dim": 128,
  "intermediate_size": 2048,
  "vocab_size": 151936,
  "max_position_embeddings": 32768,
  "num_experts_per_tok": 2,
  "num_local_experts": 64,
  "moe_intermediate_size": 1024,
  "hybrid_layer_pattern": "GSGSGSGSGSGSGSG...",
  "moe_layer_freq": 1,
  "swa_num_kv_heads": 8,
  "swa_head_dim": 64,
  "swa_v_head_dim": 64,
  "global_v_head_dim": 128,
  "sliding_window_size": 4096
}
```

## Example: VLM (Qwen3-VL style)

VLMs nest LLM parameters under `text_config`. Include the full nested structure:

```json
{
  "architectures": ["Qwen3VLMoeForConditionalGeneration"],
  "model_type": "qwen3_vl_moe",
  "text_config": {
    "num_hidden_layers": 94,
    "hidden_size": 3584,
    "num_attention_heads": 28,
    "num_key_value_heads": 4,
    "intermediate_size": 2048,
    "num_experts_per_tok": 4,
    "num_local_experts": 128,
    "moe_intermediate_size": 2560,
    "vocab_size": 152064,
    "max_position_embeddings": 32768
  },
  "vision_config": {
    "depth": 32,
    "hidden_size": 1280,
    "num_heads": 16,
    "intermediate_size": 5120,
    "patch_size": 14,
    "out_hidden_size": 3584,
    "spatial_merge_size": 2,
    "temporal_patch_size": 2,
    "in_channels": 3
  }
}
```

## Common Issues

| Issue | Cause | Fix |
|------|------|------|
| `KeyError: 'num_hidden_layers'` | Standard field missing from JSON | Add the missing field |
| `ValueError: architecture X not supported` | Architecture not in `ARCHITECTURE_TO_MODEL_FAMILY` | Complete Phase 2 |
| Incorrect `head_dim` computed | Config has `head_dim` but code falls through to formula | Ensure `head_dim` is present in JSON |
| `moe_inter_size` is 0 when it shouldn't be | Field named `moe_intermediate_size` vs `intermediate_size` | Match exact key name in config |
