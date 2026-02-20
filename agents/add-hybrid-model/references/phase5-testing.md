# Phase 5: Testing and Verification

## Goal

Add unit tests for config parsing and run end-to-end CLI validation to confirm the model is fully integrated.

---

## Step 1: Unit tests for config parsing

**File**: `tests/unit/sdk/test_utils.py`

Add a test function for each new architecture. The test exercises `_parse_hf_config_json()` directly so it runs without a GPU or network access.

### Template

```python
def test_parse_<model_name>_config():
    """Test that <ModelName> config is parsed correctly."""
    config = {
        "architectures": ["<ArchForCausalLM>"],
        # --- all standard fields ---
        "num_hidden_layers": 28,
        "hidden_size": 2048,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "intermediate_size": 2048,
        "vocab_size": 151936,
        "max_position_embeddings": 32768,
        # --- MoE fields (if applicable) ---
        "num_experts_per_tok": 2,
        "num_local_experts": 64,
        "moe_intermediate_size": 1024,
        # --- hybrid-specific fields ---
        "hybrid_layer_pattern": "GSGSGSGSG...",
        "moe_layer_freq": 1,
        "swa_num_kv_heads": 8,
        "swa_head_dim": 64,
        "swa_v_head_dim": 64,
        "global_v_head_dim": 128,
        "sliding_window_size": 4096,
    }

    result = _parse_hf_config_json(config)

    # Standard field assertions
    assert result["architecture"] == "<ArchForCausalLM>"
    assert result["layers"] == 28
    assert result["n"] == 16
    assert result["n_kv"] == 8
    assert result["d"] == 128
    assert result["topk"] == 2
    assert result["num_experts"] == 64

    # Hybrid-specific assertions
    assert isinstance(result["extra_params"], common.MiMoConfig)
    assert result["extra_params"].swa_num_kv_heads == 8
    assert result["extra_params"].sliding_window_size == 4096
```

### Example: Qwen3.5-MoE (layer_types list input)

```python
def test_parse_qwen35_moe_config():
    config = {
        "architectures": ["Qwen3_5MoeForConditionalGeneration"],
        "text_config": {
            "num_hidden_layers": 94,
            "hidden_size": 3584,
            "num_attention_heads": 28,
            "num_key_value_heads": 4,
            "intermediate_size": 2048,
            "vocab_size": 152064,
            "max_position_embeddings": 32768,
            "num_experts_per_tok": 4,
            "num_local_experts": 128,
            "moe_intermediate_size": 2560,
            "layer_types": ["full_attention", "linear_attention"] * 47,
        }
    }
    result = _parse_hf_config_json(config)
    assert result["architecture"] == "Qwen3_5MoeForConditionalGeneration"
    assert isinstance(result["extra_params"], common.Qwen35MoEConfig)
    assert len(result["extra_params"].layer_types) == 94
    assert result["extra_params"].layer_types[0] == "full_attention"
```

### Example: Qwen3-Next (full_attention_interval scalar fallback)

```python
def test_parse_qwen3_next_config():
    config = {
        "architectures": ["Qwen3NextForCausalLM"],
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
        "full_attention_interval": 4,   # scalar instead of list
    }
    result = _parse_hf_config_json(config)
    assert isinstance(result["extra_params"], common.Qwen35MoEConfig)
    # Scalar interval=4 should produce a list of length 28
    assert len(result["extra_params"].layer_types) == 28
    # Layer 0 is full attention (0 % 4 == 0)
    assert result["extra_params"].layer_types[0] == "full_attention"
    # Layer 1 is linear attention (1 % 4 != 0)
    assert result["extra_params"].layer_types[1] == "linear_attention"
```

---

## Step 2: Run unit tests

```bash
# Run only the config parsing tests
pytest tests/unit/sdk/test_utils.py -v

# Run just the new test
pytest tests/unit/sdk/test_utils.py::test_parse_mimo_v2_flash_config -v
```

All existing tests must still pass — the change should be purely additive.

---

## Step 3: End-to-end CLI validation

These checks run without GPU hardware; they exercise the model registration and config parsing code paths.

### 3a. Check architecture recognition

```bash
python3 -c "
from aiconfigurator.sdk.models import get_model_family
family = get_model_family('XiaomiMiMo/MiMo-V2-Flash')
print('family:', family)
assert family == 'HYBRIDMOE', f'Expected HYBRIDMOE, got {family}'
print('OK')
"
```

### 3b. Check model instantiation (CPU only)

```python
from aiconfigurator.sdk import models, config as cfg_module

mc = cfg_module.ModelConfig(tp_size=1)
model = models.get_model("XiaomiMiMo/MiMo-V2-Flash", mc, "trtllm")
print("context_ops:", [op._name for op in model.context_ops])
print("generation_ops:", [op._name for op in model.generation_ops])
assert len(model.context_ops) > 0
assert len(model.generation_ops) > 0
```

### 3c. CLI support check

```bash
aiconfigurator cli support --model-path XiaomiMiMo/MiMo-V2-Flash --system h200_sxm --backend trtllm
```

Expected output: architecture recognized, even if perf data is not yet available.

- If you see `"architecture is not supported"` → Phase 2 registration missing
- If you see `"data not found"` → registration works but perf data needed (expected for new models)
- If you see a full support table → model is fully integrated

---

## Step 4: Verify `check_is_moe()`

```python
from aiconfigurator.sdk.models import check_is_moe

result = check_is_moe("XiaomiMiMo/MiMo-V2-Flash")
print("is_moe:", result)  # should be True if model has MoE layers
assert result == True
```

---

## Reference PR test patterns

| PR | New test function | What it tests |
|---|---|---|
| PR 410 | `test_parse_qwen35_moe_config()` | `Qwen35MoEConfig` from list `layer_types` |
| PR 410 | `test_parse_qwen3_next_config()` | `Qwen35MoEConfig` from scalar `full_attention_interval` |
| PR 406 | Inline assertions in integration test | `DeepSeekMLAConfig` from GLM5 config |

---

## Checklist before opening PR

```
□ pytest tests/unit/sdk/test_utils.py -v → all PASSED
□ get_model_family("<HF ID>") → returns correct family
□ get_model("<HF ID>", mc, "trtllm") → non-empty context_ops and generation_ops
□ check_is_moe("<HF ID>") → correct True/False
□ aiconfigurator cli support → no "architecture not supported" error
□ No changes to existing tests
```

---

## Common Issues

| Issue | Cause | Fix |
|------|------|------|
| `ModuleNotFoundError` in test | Import path wrong | Use `from aiconfigurator.sdk.utils import _parse_hf_config_json` |
| Test fails with `KeyError` | Config dict missing a field that the parser requires | Add the missing field to the test config dict |
| `extra_params is None` in test | `elif` branch not reached (architecture mismatch) | Print `config["architectures"][0]` to verify exact string |
| `AssertionError` on layer count | `layer_types` list length != `num_hidden_layers` | Fix normalization logic in Phase 3 parser |
| CLI gives `data not found` | Expected; perf data not yet collected | Not a bug; data collection is a separate workflow |
