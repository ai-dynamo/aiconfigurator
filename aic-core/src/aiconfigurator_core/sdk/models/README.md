# `models/` Package

This package implements the model layer of the AIConfigurator SDK. Each model class defines the operation pipeline (context and generation ops) for a specific LLM architecture family.

## `blocks/`

`blocks/` holds composition helpers — reusable pipeline-fragment builders that model classes call to construct parts of their op lists — not model classes. Modules under `blocks/` must stay side-effect-free: no `@register_model` is allowed here.

- `vit.py` — `build_encoder_ops()` for ViT-based vision encoders (moved from `models/vit_ops.py`, which remains as a compatibility shim)
- `moe.py` — `MoEBlockShape` + `build_moe_block_ops()`, the single MoE-block wiring site (fused and large-EP emission), plus the `register_moe_block` variant registry and `LARGE_EP_READY_FAMILIES` (see [MoE Blocks and Large EP](#moe-blocks-and-large-ep))

## Package Structure

```
models/
  __init__.py        # get_model() factory, auto-discovery, re-exports
  base.py            # BaseModel class + _MODEL_REGISTRY + @register_model decorator
  helpers.py         # Standalone utilities: model info lookup, quant defaults, family resolution
  gpt.py             # GPTModel
  llama.py           # LLAMAModel (also covers Qwen2, Qwen3, MiMo)
  moe.py             # MOEModel (Mixtral, Qwen3MoE, MiniMax-M2, gpt-oss, etc.)
  deepseek.py        # DeepSeekModel (also serves the KIMIK25 family — Kimi K2.5)
  deepseek_v32.py    # DeepSeekV32Model (DeepSeek V3.2 / GLM-5 with DSA attention)
  deepseek_v4.py     # DeepSeekV4Model (mHC + SWA/CSA/HCA compressed attention)
  nemotron_nas.py    # NemotronNas (PUZZLE NAS models)
  nemotron_h.py      # NemotronHModel (Mamba + MoE + Transformer hybrid)
  hybrid_moe.py      # HybridMoEModel (MiMo-V2-Flash, Llama 4)
  qwen35.py          # Qwen35Model (Qwen3.5 hybrid GDN + full-attention)
```

## How It Works

### Registry-Driven Model Creation

Model classes register themselves using the `@register_model` decorator. The decorator accepts one or more family names — most classes register one family, but a class can register multiple if it serves multiple families with branching inside `create()`:

```python
from aiconfigurator_core.sdk.models.base import BaseModel, register_model

@register_model("LLAMA")
class LLAMAModel(BaseModel):
    ...

@register_model("DEEPSEEK", "KIMIK25")  # one class, two families
class DeepSeekModel(BaseModel):
    @classmethod
    def create(cls, model_info, model_config, backend_name):
        ...  # DeepSeek V3 / R1 + Kimi K2.5 construction
```

When the package is imported, all model modules are auto-discovered via `pkgutil.iter_modules`, which triggers the decorators and populates `_MODEL_REGISTRY`.

`get_model()` resolves a HuggingFace model path to a model instance:

```
model_path
  -> _get_model_info()                # parse HF config, extract architecture + params
  -> _architecture_to_model_family()  # "LlamaForCausalLM" -> "LLAMA"
  -> _apply_model_quant_defaults()    # infer quant modes from model config
  -> _MODEL_REGISTRY["LLAMA"]         # look up registered class
  -> cls.create(model_info, ...)      # construct via classmethod factory
```

### `create()` Classmethod

Each model class has a `create(cls, model_info, model_config, backend_name)` classmethod that handles construction. Per-family construction details (MoE prefix args, post-construction hooks like `set_hybrid_config`) live inside `create()`, keeping `get_model()` itself generic.

| Model | Why it overrides `create()` |
|-------|-----------------------------|
| GPTModel | Standard args, simple wrapper |
| LLAMAModel | Standard args, simple wrapper |
| MOEModel | Passes MoE-specific args (`topk`, `num_experts`, `moe_inter_size`) and threads `backend_name` |
| DeepSeekModel | Passes MoE args and threads `backend_name` (serves DEEPSEEK + KIMIK25) |
| DeepSeekV32Model | Passes MoE args, resolves the DSA quant/skip `extra_params`, threads `backend_name` |
| DeepSeekV4Model | Single class, MoE prefix args |
| NemotronHModel | Passes MoE args + calls `set_hybrid_config()` after construction |
| HybridMoEModel | Passes MoE args + calls `set_hybrid_config()` after construction |
| NemotronNas | Applies block configs from `extra_params` after construction |
| Qwen35Model | Standard args, simple wrapper |

### Key Files

- **`base.py`** — `BaseModel` defines the shared constructor (model metadata, quant config, layer counts) and the `get_kvcache_*` helpers. `_MODEL_REGISTRY` and `register_model(*families)` live here.
- **`helpers.py`** — Pure functions for model discovery (`_get_model_info`, `get_model_family`, `check_is_moe`), quantization defaults (`_apply_model_quant_defaults`), and MTP math (`mtp_scale_factor`).
- **`__init__.py`** — The `get_model()` entry point, auto-discovery loop, and backward-compatible re-exports.

### Architecture-to-Family Mapping

The mapping from HuggingFace architecture names (e.g., `LlamaForCausalLM`) to model families (e.g., `LLAMA`) lives in `sdk/common.py:ARCHITECTURE_TO_MODEL_FAMILY`. This is separate from the model registry because `sdk/utils.py` needs it during config parsing, before any model classes are involved.

## MoE Blocks and Large EP

The per-regime wideEP model classes are gone, and so is the `create()` dispatch that selected them: `DeepSeekModel.create()` used to 3-way dispatch (`WideEPDeepSeekModel` / `TrtllmWideEPDeepSeekModel` / default) on `enable_wideep` and the wideEP config, `DeepSeekV32Model` the same for its `*V32` variants, and `MOEModel` 2-way to `SGLangEPMOEModel` on sglang with `moe_backend == "deepep_moe"`. Today each family has ONE class building ONE graph per config, and the regime is selected inside the MoE block:

- **`blocks/moe.py::build_moe_block_ops`** is the single MoE-block wiring site. Model classes derive an `MoEBlockShape` from their checkpoint geometry and hand it to the builder (with their model-owned workload-distribution string and scale factor). When `ModelConfig.moe_comm_backend` — an internal field the enumerator sets per parallel tuple, never a user flag — names a comm backend for the phase, the builder emits the large-EP `MoEAllToAll`/`ModeledEPMoE` graph; otherwise the fused dispatch/MoE/dispatch pipeline. `ModelConfig.num_gpus_per_node` must be set alongside the comm backend (large-EP construction raises otherwise — see `helpers.large_ep_gpus_per_node`). An optional internal `moe_comm_query_profile` records the measured `(ep_size, node_num)` keys for an explicitly approved donor profile; it changes only the A2A lookup geometry, while `ModeledEPMoE` continues to use the deployed EP size.
- **`blocks/moe.py::register_moe_block`** is how family/framework/system deviations specialize the builder instead of adding model classes. Two variants ship registered: DEEPSEEK-on-sglang and DEEPSEEKV32-on-sglang strip the router GEMM under deepep backends (the legacy wideEP graphs never wired one there).
- **`blocks/moe.py::LARGE_EP_READY_FAMILIES`** = `{MOE, DEEPSEEK, DEEPSEEKV32, KIMIK25}` — the families whose classes are wired for the large-EP emission. The enumerator assigns `moe_comm_backend` only inside this set.

Current wiring status per family:

| Family (module) | MoE block source | Large EP |
|---|---|---|
| MOE (`moe.py`) | builder, both regimes | ready |
| DEEPSEEK / KIMIK25 (`deepseek.py`) | hand-wired fused spans; builder for large EP | ready |
| DEEPSEEKV32 (`deepseek_v32.py`) | hand-wired fused spans; builder for large EP | ready |
| QWEN3VL_MOE (`qwen3vl.py`) | rides MOEModel | excluded from `LARGE_EP_READY_FAMILIES`: `Qwen3VLMoEModel.create()` does not forward `backend_name`, so the builder would pick the wrong large-EP shared-expert/reduce flavor; wiring it is a documented follow-up |
| HYBRIDMOE (`hybrid_moe.py`) | builder (fused only) | not wired — raises `ValueError` on `moe_comm_backend` |
| MINIMAXM3 (`minimax_m3.py`) | builder (fused only; shared triplet stays hand-wired) | not wired — raises `ValueError` on `moe_comm_backend` |
| QWEN35 (`qwen35.py`), GEMMA4 (`gemma4.py`), NEMOTRONH (`nemotron_h.py`), DEEPSEEKV4 (`deepseek_v4.py`) | hand-wired MoE spans | not wired (builder adoption is backlog; DeepSeek-V4 owns its own MegaMoE path) |

Moving the remaining hand-wired MoE spans (including the DeepSeek fused spans, whose generation dialect differs from the builder's transcription) onto the builder is tracked backlog, not a prerequisite for large-EP work: a family becomes large-EP-ready by adopting the builder for its MoE block and joining `LARGE_EP_READY_FAMILIES`.

## Adding a New Model

### Case 1: New architecture in an existing family

If the new model uses the same operation pipeline as an existing family (e.g., a new Llama variant):

**1 file to edit:**
- `sdk/common.py` — Add the architecture name to `ARCHITECTURE_TO_MODEL_FAMILY`:
  ```python
  ARCHITECTURE_TO_MODEL_FAMILY = {
      ...
      "NewLlamaForCausalLM": "LLAMA",  # <-- add this
  }
  ```

That's it. The existing `LLAMAModel` handles the rest.

### Case 2: New model family (standard, no MoE)

**2 files to edit:**

**1. Create `models/new_model.py`:**

```python
from aiconfigurator_core.sdk.models.base import BaseModel, register_model


@register_model("NEWMODEL")
class NewModel(BaseModel):
    @classmethod
    def create(cls, model_info, model_config, backend_name):
        return cls(
            model_info["model_path"],
            model_info["model_family"],
            model_info["architecture"],
            model_info["layers"],
            model_info["n"],
            model_info["n_kv"],
            model_info["d"],
            model_info["hidden_size"],
            model_info["inter_size"],
            model_info["vocab"],
            model_info["context"],
            model_config,
            model_info["extra_params"],
        )

    def __init__(self, *args) -> None:
        super().__init__(*args)
        # Build your context_ops and generation_ops pipelines here
        self.context_ops = [...]
        self.generation_ops = [...]
```

**2. Edit `sdk/common.py`:**

```python
ARCHITECTURE_TO_MODEL_FAMILY = {
    ...
    "NewModelForCausalLM": "NEWMODEL",
}
ModelFamily = {
    ...
    "NEWMODEL",
}
```

### Case 3: New model family with custom construction

If the model needs MoE args, post-construction setup, or variant dispatch, override `create()` accordingly:

```python
@register_model("NEWMOE")
class NewMoEModel(BaseModel):
    @classmethod
    def create(cls, model_info, model_config, backend_name):
        model = cls(
            model_info["topk"],
            model_info["num_experts"],
            model_info["moe_inter_size"],
            model_info["model_path"],
            model_info["model_family"],
            model_info["architecture"],
            model_info["layers"],
            model_info["n"],
            model_info["n_kv"],
            model_info["d"],
            model_info["hidden_size"],
            model_info["inter_size"],
            model_info["vocab"],
            model_info["context"],
            model_config,
        )
        model.set_some_config(model_info["extra_params"])
        return model

    def __init__(self, topk, num_experts, moe_inter_size, *args) -> None:
        super().__init__(*args)
        self._topk = topk
        self._num_experts = num_experts
        self._moe_inter_size = moe_inter_size
        # ...
```

### Case 4: One class serving multiple families

If a new family reuses the construction logic of an existing class with minor branching (the `KIMIK25` pattern), extend the existing class's `@register_model` decorator:

```python
@register_model("EXISTING_FAMILY", "NEW_FAMILY")
class ExistingModel(BaseModel):
    @classmethod
    def create(cls, model_info, model_config, backend_name):
        if model_info["model_family"] == "NEW_FAMILY":
            ...  # new-family-specific path
        else:
            ...  # existing-family path (the original create() body)
```

Plus the `common.py` mapping for the new architecture and family name.

### `model_info` dict keys

The `model_info` dict passed to `create()` contains these keys. Most come from `utils.py:get_model_config_from_model_path()`; two are injected by `get_model()` before calling `create()`:

| Key | Type | Description |
|-----|------|-------------|
| `model_path` | `str` | HuggingFace model path or local path *(injected by `get_model()`)* |
| `model_family` | `str` | Resolved family name (e.g., `"LLAMA"`) *(injected by `get_model()`)* |
| `architecture` | `str` | HuggingFace architecture (e.g., `"LlamaForCausalLM"`) |
| `layers` | `int` | Number of transformer layers |
| `n` | `int` | Number of attention heads |
| `n_kv` | `int` | Number of key-value heads |
| `d` | `int` | Head size |
| `hidden_size` | `int` | Hidden dimension |
| `inter_size` | `int` | Intermediate (FFN) size |
| `vocab` | `int` | Vocabulary size |
| `context` | `int` | Max context length |
| `topk` | `int` | MoE top-k experts (0 for dense models) |
| `num_experts` | `int` | Number of MoE experts (0 for dense models) |
| `moe_inter_size` | `int` | MoE intermediate size |
| `extra_params` | `any` | Architecture-specific config (`NemotronHConfig`, `HybridMoEConfig`, `DeepSeekV4Config`, `Qwen35Config`, `list[BlockConfig]`, or `dict`) |
| `raw_config` | `dict` | Raw HuggingFace config.json contents |
