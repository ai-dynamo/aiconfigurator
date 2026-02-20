# Phase 2: Registration in `common.py`

## Goal

Register the new model so aiconfigurator can recognize its architecture, route it to the correct model class, and expose it as a supported model.

## File

`src/aiconfigurator/sdk/common.py`

---

## Step 1: Add to `ModelFamily` (new families only)

**When**: Only if the model requires a new model class (Situation S3). Skip if reusing an existing family.

```python
# common.py line ~282
# Before:
ModelFamily = {"GPT", "LLAMA", "MOE", "DEEPSEEK", "NEMOTRONNAS", "NEMOTRONH"}

# After (example adding HYBRIDMOE):
ModelFamily = {"GPT", "LLAMA", "MOE", "DEEPSEEK", "NEMOTRONNAS", "NEMOTRONH", "HYBRIDMOE"}
```

Naming convention: use ALL_CAPS, descriptive of the primary architectural characteristic.

| Example family | Used for |
|---|---|
| `HYBRIDMOE` | SWA + global attention + MoE (MiMo-V2-Flash) |
| `QWEN35MOE` | Linear attention + full attention + MoE (Qwen3.5-MoE) |
| `NEMOTRONH` | Mamba + Transformer + MoE (NemotronH) |

---

## Step 2: Add to `ARCHITECTURE_TO_MODEL_FAMILY`

**Always required.** Without this, `_parse_hf_config_json()` raises `ValueError: architecture not supported`.

```python
# common.py line ~283
ARCHITECTURE_TO_MODEL_FAMILY = {
    # ... existing entries ...
    "MiMoV2FlashForCausalLM": "HYBRIDMOE",          # add
    "Qwen3_5MoeForConditionalGeneration": "QWEN35MOE", # add (VLM wrapper)
    "Qwen3NextForCausalLM": "QWEN35MOE",             # add (same family, different arch)
}
```

> **Important**: For VLMs, the architecture string is the wrapper (e.g. `ForConditionalGeneration`), not the inner LLM architecture. Register the wrapper.

---

## Step 3: Add to `DefaultHFModels`

**Always required** for the model to be accepted by the CLI without explicit HuggingFace download.

```python
# common.py line ~220
DefaultHFModels = {
    # ... existing models ...
    # New Hybrid Models
    "XiaomiMiMo/MiMo-V2-Flash",
    "Qwen/Qwen3.5-397B-A17B",
}
```

---

## Step 4: Add a config dataclass (if hybrid-specific params exist)

**When**: Only if the model has structured config fields beyond the standard set (layers, hidden, n, n_kv, d, inter, vocab, context, topk, num_experts, moe_inter). Standard fields are already handled by `_parse_hf_config_json()` without a dataclass.

Use a frozen `@dataclass` so it can be stored in `extra_params` and passed through the system unchanged.

```python
# common.py — add near other config dataclasses (search for NemotronHConfig)

import dataclasses  # already imported

@dataclasses.dataclass(frozen=True)
class MiMoConfig:
    """Configuration for MiMo V2 Flash hybrid model (SWA + Global Attention + MoE)."""
    hybrid_layer_pattern: str   # per-layer type string, e.g. "GSGSGSGSG..." (G=global, S=SWA)
    moe_layer_freq: int         # how often MoE appears in the layer pattern
    swa_num_kv_heads: int       # KV heads for SWA attention layers
    swa_head_dim: int           # head dim for SWA queries
    swa_v_head_dim: int         # value head dim for SWA layers
    global_v_head_dim: int      # value head dim for global attention layers
    sliding_window_size: int    # SWA window length
```

```python
@dataclasses.dataclass(frozen=True)
class Qwen35MoEConfig:
    """Configuration for Qwen3.5-MoE hybrid model (linear + full attention + MoE)."""
    layer_types: list           # per-layer type list: "full_attention" or "linear_attention"
    linear_num_kv_heads: int
    linear_num_value_heads: int
    linear_key_head_dim: int
    linear_value_head_dim: int
    shared_expert_inter_size: int  # dense shared expert size (0 if absent)
```

### Existing reference dataclasses

| Dataclass | Used by |
|---|---|
| `NemotronHConfig` | NemotronH (Mamba + Transformer + MoE) |
| `DeepSeekMLAConfig` | GLM5, DeepSeek MLA variants (MLA attention dims) |
| `VisionEncoderConfig` | Qwen3-VL, KimiK2.5 (VLM vision tower params) |

---

## Special cases

### MLA (Multi-head Latent Attention) variants

Models like GLM5 use MLA attention from DeepSeek but with different dims. Use the **existing** `DEEPSEEK` family and `DeepSeekMLAConfig` dataclass:

```python
ARCHITECTURE_TO_MODEL_FAMILY = {
    "GlmMoeDsaForCausalLM": "DEEPSEEK",  # MLA variant -> reuse DEEPSEEK family
}
```

### Models sharing a family

Multiple architecture strings can map to the same family. The model class handles config differences via `extra_params`:

```python
ARCHITECTURE_TO_MODEL_FAMILY = {
    "Qwen3_5MoeForConditionalGeneration": "QWEN35MOE",  # VLM wrapper
    "Qwen3NextForCausalLM": "QWEN35MOE",                # Pure LLM variant
}
```

### MTP models

Models with Multi-Token Prediction (e.g. MiniMax-M2.5) do **not** need a new family — use `MOE` family. The MTP behavior is controlled by `nextn` from `raw_config["num_mtp_modules"]` in `task.py`.

---

## Validation

```bash
# Quick Python check — no GPU needed
python3 -c "
from aiconfigurator.sdk.common import ARCHITECTURE_TO_MODEL_FAMILY, ModelFamily
assert 'MiMoV2FlashForCausalLM' in ARCHITECTURE_TO_MODEL_FAMILY, 'mapping missing'
assert ARCHITECTURE_TO_MODEL_FAMILY['MiMoV2FlashForCausalLM'] in ModelFamily, 'family missing'
print('OK:', ARCHITECTURE_TO_MODEL_FAMILY['MiMoV2FlashForCausalLM'])
"
```

## Common Issues

| Issue | Cause | Fix |
|------|------|------|
| `ValueError: architecture X is not supported` | Missing entry in `ARCHITECTURE_TO_MODEL_FAMILY` | Add entry |
| `ValueError: model family Y not valid` | Family string not in `ModelFamily` set | Add to `ModelFamily` |
| CLI rejects model path | HF ID not in `DefaultHFModels` | Add to `DefaultHFModels` |
| `dataclass` not frozen | Forgot `frozen=True` | Always use `@dataclasses.dataclass(frozen=True)` |
| Two architectures needed for same model | VLM has a wrapper arch + inner LLM arch | Register the wrapper arch only; parse inner from `text_config` in `utils.py` |
