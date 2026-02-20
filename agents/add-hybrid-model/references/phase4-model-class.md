# Phase 4: Model Class in `models.py`

## Goal

Implement a new model class that separates compute ops by layer type, wire it into `get_model()`, and update `check_is_moe()`.

## File

`src/aiconfigurator/sdk/models.py`

---

## When to add a new class vs reuse an existing one

| Situation | Action |
|---|---|
| New architecture, same op structure as an existing family | Add to `get_model()` `elif` chain only — no new class needed |
| Existing family, but with new config fields (`extra_params`) | Add `elif` in `get_model()` and pass `extra_params` to existing class constructor |
| Genuinely new layer-type combination (new hybrid) | Create a new model class |

For hybrid models (S3), a new class is always required because op counts differ per layer type.

---

## Step 1: Understand `BaseModel`

**Location**: line ~362

`BaseModel.__init__()` takes standard params and builds `self.context_ops` / `self.generation_ops`.

Key attributes:
```python
self._num_layers    # total transformer layers
self._num_heads     # attention heads
self._n_kv          # key-value heads
self._d             # head dim
self._hidden        # hidden size
self._inter         # FFN intermediate size
self._vocab         # vocabulary size
self._context       # max context length
self._nextn         # MTP extra tokens (0 = disabled)

self.context_ops    # list[Operation] — prefill compute ops
self.generation_ops # list[Operation] — decode compute ops
```

---

## Step 2: Create the new model class

Place the new class **before** `get_model()`. Follow `NemotronHModel` as the closest reference for hybrid models.

### Key design pattern: deferred initialization

For hybrid models, op construction requires the per-layer pattern from `extra_params`. Store the config as instance variables *before* calling `super().__init__()`, which triggers `_build_ops()`:

```python
class HybridMoEModel(BaseModel):
    """
    Hybrid MoE model with heterogeneous layers (e.g. SWA attention + global attention + MoE FFN).
    Op costs are computed per layer type rather than uniformly scaled.
    """

    def __init__(
        self,
        model_path, model_family, architecture,
        layers, n, n_kv, d, hidden, inter, vocab, context,
        model_config,
        nextn,
        moe_inter, num_experts, topk,
        extra_params,  # MiMoConfig
    ):
        # Store hybrid config BEFORE super().__init__ which calls _build_ops
        self._hybrid_config = extra_params
        self._moe_inter = moe_inter
        self._num_experts = num_experts
        self._topk = topk

        super().__init__(
            model_path, model_family, architecture, layers, n, n_kv, d,
            hidden, inter, vocab, context, model_config, nextn,
        )

    def _build_ops(self):
        """Override to build separate op lists for each layer type."""
        cfg = self._hybrid_config
        pattern = cfg.hybrid_layer_pattern  # e.g. "GSGSGSGSG..."

        # Count layer types
        n_global = pattern.count("G")  # global attention + MoE
        n_swa = pattern.count("S")     # SWA attention + MoE

        # ... build context_ops and generation_ops using n_global / n_swa as scale factors ...
        # See existing MOEModel for op construction patterns
        pass
```

### Scale factors for each layer type

The scale factor in each `Operation(name, scale_factor, ...)` call represents "how many identical copies exist". For hybrid models, use the layer counts per type rather than `self._num_layers`:

```python
# Wrong: uses total layers (mixes different layer types)
ops.ContextAttention("ctx_attn", self._num_layers, ...)

# Correct: separate scale factors per type
ops.ContextAttention("ctx_global_attn", n_global, ...)   # global attention layers only
ops.ContextAttention("ctx_swa_attn", n_swa, ...)         # SWA layers only
```

### MTP scale factor (if model has MTP)

If `self._nextn > 0`, apply the same MTP scaling used in `MOEModel` and `DeepSeekModel`:

```python
from aiconfigurator.sdk.models import calc_expectation

self._mtp_scale_factor = (
    1.0 / (1 + calc_expectation(self._nextn, self._nextn_accept_rates))
    * (self._nextn + self._num_layers) / self._num_layers
    if self._nextn > 0
    else 1.0
)
# Apply to all generation ops that are token-batch-size-dependent
```

---

## Step 3: Wire into `get_model()`

**Location**: line ~137

Find the `elif model_family == "NEMOTRONH":` block and add after it:

```python
elif model_family == "HYBRIDMOE":
    model = HybridMoEModel(
        model_path, model_family, architecture,
        layers, n, n_kv, d, hidden, inter, vocab, context,
        model_config,
        nextn,
        moe_inter_size, num_experts, topk,
        extra_params,
    )
```

> **Note**: The argument order must match the constructor. Check what `model_info` provides:
> - `layers, n, n_kv, d, hidden, inter, vocab, context` — from standard parsing
> - `topk, num_experts, moe_inter_size` — MoE fields
> - `extra_params` — your hybrid config dataclass
> - `nextn` — from `model_config.nextn` (set by `task.py` before calling `get_model`)

---

## Step 4: Update `check_is_moe()`

**Location**: line ~325

If the new family contains MoE layers, it must return `True`:

```python
def check_is_moe(model_path: str) -> bool:
    family = get_model_family(model_path)
    if family in ("MOE", "DEEPSEEK", "HYBRIDMOE", "QWEN35MOE"):  # add new family
        return True
    if family == "NEMOTRONH":
        ...
    return False
```

---

## Reference op construction patterns

### Standard ops shared across most models

```python
# Embedding
ops.Embedding("embedding", 1, self._vocab, self._hidden)

# LayerNorm / RMSNorm
ops.ElementWise("norm", self._num_layers, self._hidden)

# Dense FFN (SwiGLU: 2 GEMMs + activation)
ops.GEMM("ffn_gate", self._num_layers, self._hidden, self._inter)
ops.GEMM("ffn_up",   self._num_layers, self._hidden, self._inter)
ops.GEMM("ffn_down", self._num_layers, self._inter,  self._hidden)

# Output projection
ops.GEMM("lm_head", 1, self._hidden, self._vocab)
```

### Attention ops

```python
# Context (prefill) attention
ops.ContextAttention("ctx_attn", n_layers, self._n_kv, self._d, self._context)

# Generation (decode) attention
ops.GenerationAttention("gen_attn", n_layers, self._n_kv, self._d, self._context)

# QKV projection
ops.GEMM("qkv_proj", n_layers,
         self._hidden, (self._num_heads + 2 * self._n_kv) * self._d)

# Output projection
ops.GEMM("o_proj", n_layers, self._num_heads * self._d, self._hidden)
```

### MoE ops

```python
# Router
ops.GEMM("moe_router", n_moe_layers, self._hidden, self._num_experts)

# MoE dispatch (gather/scatter communication)
ops.MoEDispatch("moe_dispatch", n_moe_layers,
                self._num_experts, self._topk, self._moe_ep_size)

# MoE compute (experts)
ops.MoE("moe_experts", n_moe_layers,
        self._hidden, self._moe_inter, self._num_experts, self._topk)
```

### Sliding-window attention

Same op class as standard attention, but with different head dimensions from the hybrid config:

```python
ops.ContextAttention("ctx_swa_attn", n_swa_layers,
                     cfg.swa_num_kv_heads, cfg.swa_head_dim, cfg.sliding_window_size)
```

---

## Example: NemotronH (reference hybrid model)

`NemotronHModel` is the closest existing reference. Key features:
- Uses `set_nemotronh_config(extra_params)` for deferred initialization
- Builds separate ops for Mamba layers, attention layers, MoE layers
- Uses `pattern.count("M")`, `pattern.count("T")`, `pattern.count("E")` for counts

---

## Validation

```python
from aiconfigurator.sdk import models, config as cfg_module

mc = cfg_module.ModelConfig(tp_size=8, moe_ep_size=4)
model = models.get_model("XiaomiMiMo/MiMo-V2-Flash", mc, "trtllm")

print("context ops:", [op._name for op in model.context_ops])
print("generation ops:", [op._name for op in model.generation_ops])
assert len(model.context_ops) > 0
assert len(model.generation_ops) > 0

from aiconfigurator.sdk.models import check_is_moe
assert check_is_moe("XiaomiMiMo/MiMo-V2-Flash") == True
```

---

## Common Issues

| Issue | Cause | Fix |
|------|------|------|
| `AssertionError: num_heads not divisible by tp_size` | `BaseModel.__init__` validates head divisibility | Verify tp_size choices in tests; not a code bug |
| `context_ops` is empty | `_build_ops()` not called or returned early | Ensure `_build_ops()` is called from `__init__` (base class does this) |
| `extra_params is None` in `get_model()` | Phase 3 parsing not reached | Verify `elif` branch in `_parse_hf_config_json()` |
| `TypeError: cannot unpack non-sequence` | Constructor arg order mismatch | Match exactly the order in `model_info` dict |
| Wrong total latency | Scale factors use `self._num_layers` instead of per-type counts | Use `pattern.count(type_char)` for each type |
| `check_is_moe()` returns False | New family not added to the `if family in (...)` check | Update `check_is_moe()` |
