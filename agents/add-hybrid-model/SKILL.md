---
name: add-hybrid-model
description: |
  Add support for a new hybrid architecture model to aiconfigurator.
  Use when adding a model with heterogeneous layer types: mixed attention variants
  (SWA + global, linear + full), mixed FFN types (dense + MoE), or both.
  This skill guides through: Situation Assessment → Model Config → Registration → Parsing → Model Class → Testing.
---

# Add Hybrid Architecture Model Skill

> **Background**: "Hybrid" means the model has layers of different types rather than a uniform stack.
> Examples: MiMo-V2-Flash (SWA + global attention + MoE), Qwen3.5-MoE (linear attention + full attention + MoE),
> NemotronH (Mamba + Transformer + MoE). Each layer type needs separate op cost modeling.

## Step 0: Determine the Situation (Mandatory)

Before making any changes, classify which situation you are in. Required effort varies significantly.

| Situation | Description | Required Phases |
|-----------|-------------|-----------------|
| **S1: Existing family, same ops** | Architecture fits an existing family (e.g., another LLAMA or MOE variant) | Phase 1 only |
| **S2: Existing family, new config fields** | Architecture fits an existing family but has extra config params (e.g., MTP support, MLA dims) | Phases 1–2 + part of 3 |
| **S3: New hybrid family** | Model has fundamentally new layer-type heterogeneity needing a new model class | Full Phases 1–5 |

**How to decide**:

```
1. Read config.json; note the architecture field.
2. Check whether config["architectures"][0] exists in ARCHITECTURE_TO_MODEL_FAMILY (common.py line ~283).
   -> Exists AND layer structure is uniform -> S1
   -> Exists but has new config fields (MTP, MLA, VLM nesting) -> S2
3. If architecture is unknown, check whether any existing model class can represent it:
   -> All layer types already modeled (e.g., standard attention + standard MoE) -> S1 or S2
   -> At least one layer type combination is new (e.g., linear-attention + MoE, SWA + dense) -> S3
```

---

## Key File Paths (Quick Reference)

| File | Path | Purpose |
|------|------|---------|
| Architecture mapping & families | `src/aiconfigurator/sdk/common.py` | `ModelFamily`, `ARCHITECTURE_TO_MODEL_FAMILY`, `DefaultHFModels`, config dataclasses |
| Config parsing | `src/aiconfigurator/sdk/utils.py` | `_parse_hf_config_json()` |
| Model classes | `src/aiconfigurator/sdk/models.py` | `BaseModel`, all model subclasses, `get_model()`, `check_is_moe()` |
| Model config JSONs | `src/aiconfigurator/model_configs/` | One JSON per model; filename = `<Org>--<Model>_config.json` |
| Unit tests | `tests/unit/sdk/test_utils.py` | Config parsing tests |
| Support matrix | `src/aiconfigurator/systems/support_matrix.csv` | Tested model/system/backend combinations |

---

## Phase 1: Model Config JSON

**Goal**: Create the HuggingFace-format config file for aiconfigurator to read.

**Reference**: `agents/add-hybrid-model/references/phase1-model-config.md`

**Done criteria**: File exists at `src/aiconfigurator/model_configs/<Org>--<Model>_config.json` with all required fields, including hybrid-specific per-layer pattern fields.

---

## Phase 2: Registration in `common.py`

**Goal**: Register the new architecture, family, model ID, and config dataclass so the rest of the codebase can recognize it.

**Reference**: `agents/add-hybrid-model/references/phase2-registration.md`

**Done criteria**: `_architecture_to_model_family("NewArchForCausalLM")` returns the expected family string; model ID appears in `DefaultHFModels`.

---

## Phase 3: Config Parsing in `utils.py`

**Goal**: Extend `_parse_hf_config_json()` to parse hybrid-specific fields (layer type patterns, special attention dims, etc.) into the config dataclass as `extra_params`.

**Reference**: `agents/add-hybrid-model/references/phase3-config-parsing.md`

**Done criteria**: `get_model_config_from_model_path("<model-id>")["extra_params"]` returns the correct config dataclass instance with all fields populated.

---

## Phase 4: Model Class in `models.py`

**Goal**: Implement a new model class (S3 only) that separates ops by layer type and correctly accounts for each layer type's compute cost, then wire it into `get_model()`.

**Reference**: `agents/add-hybrid-model/references/phase4-model-class.md`

**Done criteria**: `get_model("<model-id>", model_config, backend)` returns a model with non-empty `context_ops` and `generation_ops`; `check_is_moe()` returns correct value.

---

## Phase 5: Testing and Verification

**Goal**: Add unit tests for config parsing and run end-to-end CLI validation.

**Reference**: `agents/add-hybrid-model/references/phase5-testing.md`

**Done criteria**: `pytest tests/unit/sdk/test_utils.py -v` passes; `aiconfigurator cli support` recognizes the model.

---

## Full Integration Checklist

```
□ Phase 1 — model_configs/<Org>--<Model>_config.json created with all fields
□ Phase 2 — ARCHITECTURE_TO_MODEL_FAMILY entry added in common.py
□ Phase 2 — ModelFamily set updated (if new family)
□ Phase 2 — Model HF ID added to DefaultHFModels in common.py
□ Phase 2 — Config dataclass added to common.py (if hybrid-specific params exist)
□ Phase 3 — _parse_hf_config_json() updated in utils.py
□ Phase 3 — VLM: text_config nesting handled (if ForConditionalGeneration)
□ Phase 4 — New model class created in models.py (S3 only)
□ Phase 4 — get_model() dispatches to new class
□ Phase 4 — check_is_moe() updated (if model contains MoE layers)
□ Phase 5 — Unit test added to tests/unit/sdk/test_utils.py
□ Phase 5 — CLI validation passes
```

---

## Lessons Learned (From Real Hybrid Model PRs)

### 1. Deferred Initialization for Per-Layer-Pattern Models

**Problem**: Layer type breakdown (which layers are SWA, which are global, etc.) is only known after the config is loaded, but `BaseModel.__init__` builds ops immediately.

**Rule**: Store the hybrid config as an instance variable _before_ calling `super().__init__()`, then use it inside `_build_ops()` (or `set_<model>_config()`). See `NemotronHModel` as the reference pattern.

### 2. Scale Factor Must Reflect Actual Layer Counts

**Problem**: Using `self._num_layers` as the scale factor for all ops in a hybrid model double-counts or under-counts.

**Rule**: Count layers of each type from the pattern (e.g., `pattern.count("G")` for global layers) and use those counts as individual scale factors.

### 3. MoE Check Must Cover New Families

**Problem**: `check_is_moe()` in `models.py` only knows about `MOE` and `DEEPSEEK`; new hybrid families are missed.

**Rule**: Whenever a new family contains MoE layers, add it to the `if family in (...)` check in `check_is_moe()`.

### 4. VLM Config Nesting

**Problem**: VLMs (architectures ending in `ForConditionalGeneration`) nest LLM params under `text_config`. Standard field access (`config["num_hidden_layers"]`) will fail.

**Rule**: Add the architecture to `VLM_ARCHITECTURES` in `utils.py` and use `config = config.get("text_config", config)` before reading standard fields. Parse `vision_config` separately.

### 5. MTP Scale Factor for Non-DeepSeek Models

**Problem**: MTP (Multi-Token Prediction) was originally only in DeepSeek. Extending it to other families (e.g., MOE for MiniMax-M2.5) requires removing `assert self._nextn == 0` guards.

**Rule**: Apply the same `_mtp_scale_factor` formula used in `DeepSeekModel` wherever generation ops are built. Read `nextn` from `raw_config["num_mtp_modules"]` (or `num_nextn_predict_layers`).

### 6. Pattern Fields Can Be List or Scalar

**Problem**: Some models express layer types as a per-layer list (`["full", "linear", "full", ...]`), others as a scalar interval (`full_attention_interval: 4`).

**Rule**: Normalize in `_parse_hf_config_json()`: if a list exists, use it; otherwise generate it from the scalar. Store the canonical list in the dataclass.
