# Phase 3: Config Parsing in `utils.py`

## Goal

Extend `_parse_hf_config_json()` to read hybrid-specific fields from the JSON and populate `extra_params` with the dataclass defined in Phase 2.

## File

`src/aiconfigurator/sdk/utils.py`

---

## Understanding `_parse_hf_config_json()`

**Location**: line ~429

This function converts a raw HuggingFace `config.json` dict into a normalized dict used throughout the codebase. It:
1. Reads standard fields (`layers`, `hidden_size`, `n`, `n_kv`, `d`, `inter`, `vocab`, `context`, `topk`, `num_experts`, `moe_inter`)
2. Reads architecture-specific fields into `extra_params` (a frozen dataclass or `None`)
3. Returns a flat dict with all values

**The `extra_params` block** (line ~466) is where to add new hybrid model parsing:

```python
extra_params = None
if architecture == "NemotronHForCausalLM":
    extra_params = common.NemotronHConfig(...)
elif architecture == "DeciLMForCausalLM":
    ...
# <- add your elif branch here
```

---

## Step 1: Add parsing for standard hybrid models

Add an `elif` branch for each new architecture that has extra fields:

```python
elif architecture == "MiMoV2FlashForCausalLM":
    extra_params = common.MiMoConfig(
        hybrid_layer_pattern=config["hybrid_layer_pattern"],
        moe_layer_freq=config.get("moe_layer_freq", 1),
        swa_num_kv_heads=config["swa_num_kv_heads"],
        swa_head_dim=config.get("swa_head_dim", config.get("head_dim", hidden_size // n)),
        swa_v_head_dim=config.get("swa_v_head_dim", config.get("head_dim", hidden_size // n)),
        global_v_head_dim=config.get("global_v_head_dim", config.get("head_dim", hidden_size // n)),
        sliding_window_size=config["sliding_window_size"],
    )
```

> **Tip**: Use `.get("field", default)` for optional fields so old/incomplete configs still parse.

---

## Step 2: Normalize layer pattern fields

Some models express layer types as a per-layer list; others use a scalar interval. Normalize to a canonical list in the parser so `models.py` always receives a list:

```python
elif architecture in ("Qwen3_5MoeForConditionalGeneration", "Qwen3NextForCausalLM"):
    # Normalize layer_types: list or derive from full_attention_interval
    layer_types = config.get("layer_types")
    if layer_types is None:
        interval = config.get("full_attention_interval", 4)
        num_layers = config["num_hidden_layers"]
        layer_types = [
            "full_attention" if i % interval == 0 else "linear_attention"
            for i in range(num_layers)
        ]

    extra_params = common.Qwen35MoEConfig(
        layer_types=layer_types,
        linear_num_kv_heads=config.get("linear_num_kv_heads", config.get("num_key_value_heads", 0)),
        linear_num_value_heads=config.get("linear_num_value_heads", 0),
        linear_key_head_dim=config.get("linear_key_head_dim", config.get("head_dim", hidden_size // n)),
        linear_value_head_dim=config.get("linear_value_head_dim", config.get("head_dim", hidden_size // n)),
        shared_expert_inter_size=config.get("shared_intermediate_size", 0),
    )
```

---

## Step 3: Handle VLM `text_config` nesting

VLMs (architectures ending in `ForConditionalGeneration`) nest LLM params under `text_config`. Without special handling, standard field access (`config["num_hidden_layers"]`) will `KeyError`.

Add the architecture to `VLM_ARCHITECTURES` (defined near top of `utils.py`):

```python
VLM_ARCHITECTURES = {
    "KimiK25ForConditionalGeneration",
    "Qwen3VLMoeForConditionalGeneration",
    "Qwen3_5MoeForConditionalGeneration",  # <- add if VLM wrapper
}
```

The existing guard at the start of `_parse_hf_config_json()` then automatically uses `text_config`:

```python
# Already in utils.py — just add your arch to VLM_ARCHITECTURES
if architecture in VLM_ARCHITECTURES:
    config = config.get("text_config", config)
```

Then for VLMs with a vision tower, also parse `vision_config`. Check whether `VisionEncoderConfig` already covers your case, or add new fields:

```python
elif architecture == "Qwen3VLMoeForConditionalGeneration":
    vc = original_config.get("vision_config", {})  # use original_config before text_config override
    extra_params = common.VisionEncoderConfig(
        depth=vc["depth"],
        hidden_size=vc["hidden_size"],
        num_heads=vc["num_heads"],
        intermediate_size=vc["intermediate_size"],
        patch_size=vc["patch_size"],
        out_hidden_size=vc.get("out_hidden_size", vc["hidden_size"]),
        spatial_merge_size=vc.get("spatial_merge_size", 2),
        temporal_patch_size=vc.get("temporal_patch_size", 2),
        in_channels=vc.get("in_channels", 3),
    )
```

---

## Step 4: Handle MLA (Multi-head Latent Attention) variants

Models that use MLA attention (have `kv_lora_rank` in config) should map to `DEEPSEEK` family and use `DeepSeekMLAConfig`:

```python
elif architecture == "GlmMoeDsaForCausalLM":
    if "kv_lora_rank" in config:
        extra_params = common.DeepSeekMLAConfig(
            q_lora_rank=config.get("q_lora_rank", 0),
            kv_lora_rank=config["kv_lora_rank"],
            qk_nope_head_dim=config["qk_nope_head_dim"],
            qk_rope_head_dim=config["qk_rope_head_dim"],
            v_head_dim=config["v_head_dim"],
        )
```

---

## Return value

`_parse_hf_config_json()` returns a dict. The `extra_params` key carries the dataclass (or `None`):

```python
return {
    "architecture": architecture,
    "layers": layers,
    "n": n,
    "n_kv": n_kv,
    "d": d,
    "hidden_size": hidden_size,
    "inter_size": inter_size,
    "vocab": vocab,
    "context": context,
    "topk": topk,
    "num_experts": num_experts,
    "moe_inter_size": moe_inter_size,
    "extra_params": extra_params,   # <- your dataclass lands here
}
```

This dict is then passed to `get_model()` via `model_info = _get_model_info(model_path)`, and `extra_params` is forwarded to the model class constructor.

---

## Validation

```python
from aiconfigurator.sdk.utils import get_model_config_from_model_path

info = get_model_config_from_model_path("XiaomiMiMo/MiMo-V2-Flash")
print(info["architecture"])   # "MiMoV2FlashForCausalLM"
print(info["extra_params"])   # MiMoConfig(hybrid_layer_pattern=..., ...)
assert info["extra_params"].swa_num_kv_heads == 8
```

---

## Common Issues

| Issue | Cause | Fix |
|------|------|------|
| `KeyError: 'num_hidden_layers'` | VLM config needs `text_config` nesting but arch not in `VLM_ARCHITECTURES` | Add architecture to `VLM_ARCHITECTURES` |
| `extra_params` is `None` | `elif` branch not reached (architecture string mismatch) | Print `config["architectures"][0]` and compare exactly |
| `TypeError: __init__() missing argument` | Dataclass field not in config JSON | Use `.get("field", default)` in parser |
| Field value is wrong type | JSON stores as string, code expects int | Cast: `int(config["field"])` |
| `frozen=True` dataclass mutated | Tried to set field after init | Frozen dataclasses are immutable; all values must be in `__init__` |
| Layer count mismatch | `layer_types` list length != `num_hidden_layers` | Assert `len(layer_types) == layers` in parser |
