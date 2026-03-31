# `models/` Package

This package implements the model layer of the AIConfigurator SDK. Each model class defines the operation pipeline (context and generation ops) for a specific LLM architecture family.

## Package Structure

```
models/
  __init__.py        # get_model() factory, auto-discovery, re-exports
  base.py            # BaseModel class + _MODEL_REGISTRY + @register_model decorator
  helpers.py         # Standalone utilities: model info lookup, quant defaults, family resolution
  gpt.py             # GPTModel
  llama.py           # LLAMAModel (also covers Qwen2, Qwen3, MiMo)
  moe.py             # MOEModel (Mixtral, Qwen3MoE, MiniMax-M2, etc.)
  deepseek.py        # DeepSeekModel, TrtllmWideEPDeepSeekModel, WideEPDeepSeekModel
  nemotron_nas.py    # NemotronNas (PUZZLE NAS models)
  nemotron_h.py      # NemotronHModel (Mamba + MoE + Transformer hybrid)
  hybrid_moe.py      # HybridMoEModel (MiMo-V2-Flash, Llama 4)
```

## How It Works

### Registry-Driven Model Creation

Model classes register themselves using the `@register_model` decorator:

```python
from aiconfigurator.sdk.models.base import BaseModel, register_model

@register_model("LLAMA")
class LLAMAModel(BaseModel):
    ...
```

When the package is imported, all model modules are auto-discovered via `pkgutil.iter_modules`, which triggers the decorators and populates `_MODEL_REGISTRY`.

`get_model()` resolves a HuggingFace model path to a model instance:

```
model_path
  -> _get_model_info()          # parse HF config, extract architecture + params
  -> _architecture_to_model_family()  # "LlamaForCausalLM" -> "LLAMA"
  -> _apply_model_quant_defaults()    # infer quant modes from model config
  -> _MODEL_REGISTRY["LLAMA"]         # look up registered class
  -> cls.create(model_info, ...)      # construct via classmethod factory
```

### `create()` Classmethod

Each model class has a `create(cls, model_info, model_config, backend_name)` classmethod that handles construction. `BaseModel.create()` provides a default implementation for simple models (GPT, LLAMA). Models with custom needs override it:

| Model | Why it overrides `create()` |
|-------|-----------------------------|
| MOEModel | Passes MoE-specific args (`topk`, `num_experts`, `moe_inter_size`) |
| DeepSeekModel | 3-way dispatch based on backend and WideEP config |
| NemotronHModel | Passes MoE args + calls `set_hybrid_config()` after construction |
| HybridMoEModel | Passes MoE args + calls `set_hybrid_config()` after construction |
| NemotronNas | Applies block configs from `extra_params` after construction |

### Key Files

- **`base.py`** - `BaseModel` defines the shared constructor (model metadata, quant config, layer counts). `_MODEL_REGISTRY` and `register_model` live here.
- **`helpers.py`** - Pure functions for model discovery (`_get_model_info`, `get_model_family`, `check_is_moe`), quantization defaults (`_apply_model_quant_defaults`), and MTP math (`calc_expectation`).
- **`__init__.py`** - The `get_model()` entry point, auto-discovery loop, and backward-compatible re-exports.

### Architecture-to-Family Mapping

The mapping from HuggingFace architecture names (e.g., `LlamaForCausalLM`) to model families (e.g., `LLAMA`) lives in `sdk/common.py:ARCHITECTURE_TO_MODEL_FAMILY`. This is separate from the model registry because `sdk/utils.py` needs it during config parsing, before any model classes are involved.

## Adding a New Model

### Case 1: New architecture in an existing family

If the new model uses the same operation pipeline as an existing family (e.g., a new Llama variant):

**1 file to edit:**
- `sdk/common.py` - Add the architecture name to `ARCHITECTURE_TO_MODEL_FAMILY`:
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
from aiconfigurator.sdk.models.base import BaseModel, register_model

@register_model("NEWMODEL")
class NewModel(BaseModel):
    def __init__(self, *args) -> None:
        super().__init__(*args)
        # Build your context_ops and generation_ops pipelines here
        self.context_ops = [...]
        self.generation_ops = [...]
```

The default `BaseModel.create()` handles construction automatically.

**2. Edit `sdk/common.py`:**

```python
ARCHITECTURE_TO_MODEL_FAMILY = {
    ...
    "NewModelForCausalLM": "NEWMODEL",
}
```

### Case 3: New model family with custom construction

If the model needs MoE args, post-construction setup, or variant dispatch, override `create()`:

```python
@register_model("NEWMOE")
class NewMoEModel(BaseModel):
    @classmethod
    def create(cls, model_info: dict, model_config, backend_name: str) -> "NewMoEModel":
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
        # Any post-construction setup
        model.set_some_config(model_info["extra_params"])
        return model

    def __init__(self, topk, num_experts, moe_inter_size, *args) -> None:
        super().__init__(*args)
        self._topk = topk
        self._num_experts = num_experts
        self._moe_inter_size = moe_inter_size
        # ...
```

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
| `extra_params` | `any` | Architecture-specific config (NemotronHConfig, HybridMoEConfig, list[BlockConfig], or dict) |
| `raw_config` | `dict` | Raw HuggingFace config.json contents |
