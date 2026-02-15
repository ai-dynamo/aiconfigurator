# Phase 1: Model Architecture Analysis

## Objective

Understand the DSA architecture and extract configuration parameters from the model.

## Step-by-Step Process

### 1. Locate Model Config

```bash
# HuggingFace model config
wget https://huggingface.co/deepseek-ai/DeepSeek-V3.2/raw/main/config.json
```

### 2. Identify DSA Parameters

Key parameters in `config.json`:

```json
{
  "architectures": ["DeepseekV32ForCausalLM"],
  "model_type": "deepseek_v32",
  
  // Standard MLA params
  "hidden_size": 7168,
  "num_attention_heads": 128,
  "q_lora_rank": 1536,
  "kv_lora_rank": 512,
  "qk_rope_head_dim": 64,
  "qk_nope_head_dim": 128,
  "v_head_dim": 128,
  
  // DSA-specific params
  "index_n_heads": 64,
  "index_head_dim": 128,
  "index_topk": 2048,
  "first_k_dense_replace": 3
}
```

### 3. Architecture Mapping

Add to `src/aiconfigurator/sdk/common.py`:

```python
# Architecture to model family mapping
ARCH_MAPPING = {
    "DeepseekV3ForCausalLM": ModelFamily.DEEPSEEK,
    "DeepseekV32ForCausalLM": ModelFamily.DEEPSEEK,  # V3.2
}

# DSA config dataclass
@dataclass
class DeepSeekV32Config:
    index_n_heads: int = 64
    index_head_dim: int = 128
    index_topk: int = 2048
    first_k_dense_replace: int = 3
```

### 4. Config Parsing

Add to `src/aiconfigurator/sdk/utils.py`:

```python
def _parse_hf_config_json(config: dict) -> dict:
    # ... existing parsing ...
    
    # V3.2 DSA params
    if "index_n_heads" in config:
        extra_params["index_n_heads"] = config["index_n_heads"]
        extra_params["index_head_dim"] = config.get("index_head_dim", 128)
        extra_params["index_topk"] = config.get("index_topk", 2048)
        extra_params["first_k_dense_replace"] = config.get("first_k_dense_replace", 3)
    
    return {
        # ... existing fields ...
        "extra_params": extra_params,
    }
```

## Architecture Differences: V3 vs V3.2

| Feature | V3 | V3.2 |
|---------|----|----|
| Architecture | `DeepseekV3ForCausalLM` | `DeepseekV32ForCausalLM` |
| Attention | All layers MLA | Layer 0-2: MLA, Layer 3-60: **DSA** |
| DSA params | ❌ None | `index_n_heads`, `index_topk`, `first_k_dense_replace` |
| Complexity | O(n²) | O(n×k) for DSA layers |

## Validation

Test config parsing:

```python
from aiconfigurator.sdk.utils import _parse_hf_config_json

config = {
    "architectures": ["DeepseekV32ForCausalLM"],
    "index_n_heads": 64,
    "index_topk": 2048,
}

parsed = _parse_hf_config_json(config)
assert parsed["extra_params"]["index_n_heads"] == 64
assert parsed["extra_params"]["index_topk"] == 2048
```
