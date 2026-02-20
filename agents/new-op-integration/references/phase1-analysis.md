# Phase 1: Model Architecture Analysis

## Goal

Understand the new model configuration so that aiconfigurator can recognize and parse it correctly.

## Steps

### 1. Obtain and analyze `config.json`

```bash
# Download from HuggingFace
wget https://huggingface.co/<org>/<model>/raw/main/config.json

# Or use a local model directory
ls /path/to/model/config.json
```

Focus on these fields:
- `architectures` — model architecture identifier (for example `["NewModelForCausalLM"]`)
- `num_hidden_layers`, `hidden_size`, `num_attention_heads`, `num_key_value_heads`
- `intermediate_size`, `vocab_size`, `max_position_embeddings`
- MoE fields: `num_experts_per_tok`, `num_local_experts`/`n_routed_experts`, `moe_intermediate_size`
- Special fields: any new keys not in the list above

### 2. Compare with known models and identify new fields

Compare with existing model configs and mark fields that are new or different:

```json
{
  "architectures": ["NewModelForCausalLM"],
  "num_hidden_layers": 32,
  "hidden_size": 4096,
  "num_attention_heads": 32,
        "num_key_value_heads": 8,        // standard field
        "new_specific_field_1": 64,      // <- new field
        "new_specific_field_2": 2048     // <- new field
}
```

### 3. Add `ModelFamily` (if a new family is required)

**File**: `src/aiconfigurator/sdk/common.py`

If the new model requires a completely new model class (Situation S3), add a value to `ModelFamily`:

```python
# common.py (around line 280)
ModelFamily = {"GPT", "LLAMA", "MOE", "DEEPSEEK", "NEMOTRONNAS", "NEMOTRONH", "NEW_MODEL"}  # <- add
```

> **Note**: If the model is only a variant of an existing family (for example another Llama architecture), you can skip this step.

### 4. Add architecture mapping

**File**: `src/aiconfigurator/sdk/common.py`

Add the mapping in `ARCHITECTURE_TO_MODEL_FAMILY`:

```python
# common.py (around line 281)
ARCHITECTURE_TO_MODEL_FAMILY = {
    "LlamaForCausalLM": "LLAMA",
    "Qwen2ForCausalLM": "LLAMA",
    # ... existing mappings ...
    "NewModelForCausalLM": "NEW_MODEL",  # <- add
}
```

### 5. Add a config dataclass (if there are special structured fields)

**File**: `src/aiconfigurator/sdk/common.py`

Only required if the model has **unique structured config** (see `NemotronHConfig`).
Most models do not require this; parameters can be parsed directly.

```python
# common.py — use only when there are multiple related special parameters
@dataclass(frozen=True)
class NewModelConfig:
    """Configuration for NewModel-specific parameters."""
    new_field_1: int
    new_field_2: int
    # ...
```

### 6. Add parsing logic

**File**: `src/aiconfigurator/sdk/utils.py`

Add new-model field parsing in `_parse_hf_config_json()`:

```python
# utils.py — inside `_parse_hf_config_json()` (after around line 466)

# After existing NemotronH handling:
extra_params = None
if architecture == "NemotronHForCausalLM":
    extra_params = common.NemotronHConfig(...)
elif architecture == "NewModelForCausalLM":                    # <- add
    extra_params = common.NewModelConfig(                       # <- add
        new_field_1=config["new_field_1"],                      # <- add
        new_field_2=config.get("new_field_2", 2048),            # <- add (with default)
    )                                                           # <- add
```

> **Important**: Parsed data is passed to `models.py` through `extra_params`.
> Standard fields (`layers`, `hidden_size`, `n`, `n_kv`, etc.) are already parsed by `_parse_hf_config_json`.
> You only need to handle **new non-standard fields**.

### 7. Consume new parameters in `models.py` (if a new model class is required)

**File**: `src/aiconfigurator/sdk/models.py`

Add instantiation logic in `get_model()`. Follow existing models as reference:

```python
# models.py — inside `get_model()`

# Follow DeepSeekModel or NemotronHModel initialization patterns
if model_family == "NEW_MODEL":
    model = NewModel(
        topk, num_experts, moe_inter_size,  # if MoE
        model_path, model_family, architecture, layers, n, n_kv, d,
        hidden, inter, vocab, context, model_config,
    )
    if extra_params is not None:
        model.set_extra_config(extra_params)  # pass special config
```

## Validation

```bash
# Verify CLI recognizes the new model
aiconfigurator cli support --model-path <new-model-path-or-hf-id> --system h200_sxm --backend trtllm

# Expected: architecture is recognized even if perf data is not ready.
# If you see "architecture is not supported", mapping is missing/incorrect.
# If you see "data not found", mapping works but data collection (Phase 5) is still needed.
```

## Common Issues

| Issue | Cause | Fix |
|------|------|------|
| `architecture is not supported` | Missing mapping in `ARCHITECTURE_TO_MODEL_FAMILY` | Add mapping in `common.py` |
| `num_key_value_heads is None` | Field is null in `config.json` | Use `config.get(field) or 0` in `utils.py` |
| `extra_params` not propagated | New dataclass is not returned by `_parse_hf_config_json` | Check `extra_params` assignment path |
| Parsed values are incorrect | Mismatch between config key names and code | Re-check HuggingFace config keys carefully |
