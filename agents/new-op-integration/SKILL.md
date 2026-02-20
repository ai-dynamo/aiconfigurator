---
name: new-op-integration
description: |
  Integrate new operations into aiconfigurator to support new models. 
  Use when adding support for new model architectures or new attention/compute mechanisms.
  This skill guides through: Situation Assessment → Analysis → Mock → Profile → Integration → Collection.
---

# New Operation Integration Skill

> **Official reference**: `docs/add_a_new_model.md` — read it first for the overall decision tree.

## Step 0: Determine the Situation (Mandatory)

Before making any changes, determine which situation you are in. The required effort varies significantly.

| Situation | Description | Required Phases |
|-----------|-------------|-----------------|
| **S1: Simple Variant** | The model architecture is already supported (e.g., a new Llama variant); only an architecture mapping is needed | Partial Phase 1 |
| **S2: New Data Needed** | The architecture mapping exists, but operation parameters differ (e.g., a new MoE model with different `num_experts`/`topk`) | Phases 1 + 5 |
| **S3: New Operation Needed** | The model introduces a brand-new compute primitive (e.g., convolution, new attention mechanism) | Full Phases 1 → 5 |

**How to decide**:

```
1. Read the target model's config.json.
2. Check whether config["architectures"][0] exists in ARCHITECTURE_TO_MODEL_FAMILY.
   -> If yes, it is likely S1 or S2.
3. Analyze model structure for compute units unsupported by aiconfigurator.
   -> No unsupported op -> S1 or S2
   -> Unsupported op exists -> S3
4. For S2, verify whether existing perf data covers the new parameter combinations.
   -> Covered -> S1
   -> Not covered -> S2
```

---

## Key File Paths (Quick Reference)

| File | Path | Purpose |
|------|------|---------|
| Architecture mapping & enums | `src/aiconfigurator/sdk/common.py` | `ModelFamily` enum, `ARCHITECTURE_TO_MODEL_FAMILY` mapping |
| Config parsing | `src/aiconfigurator/sdk/utils.py` | `_parse_hf_config_json()`, `get_model_config_from_model_path()` |
| Operation definitions | `src/aiconfigurator/sdk/operations.py` | `Operation` base class and all subclasses |
| Performance database | `src/aiconfigurator/sdk/perf_database.py` | Data loading + query methods |
| Model definitions | `src/aiconfigurator/sdk/models.py` | `BaseModel` and model subclasses |
| Performance result type | `src/aiconfigurator/sdk/performance_result.py` | `PerformanceResult` (float-like, includes power/energy) |
| Performance data directory | `src/aiconfigurator/systems/data/{system}/{backend}/{version}/` | `*_perf.txt` data files |
| Test case definitions | `collector/common_test_cases.py` | Standard benchmark parameter grids |
| Collection utilities | `collector/helper.py` | `benchmark_with_power()`, `log_perf()` |
| Collection entrypoint | `collector/collect.py` | Unified collection orchestration |
| Backend collection scripts | `collector/{backend}/collect_{op}.py` | Backend-specific collection implementations |

---

## Phase 1: Model Architecture Analysis

**Goal**: Understand model configuration and add architecture recognition support.

**Reference**: `agents/new-op-integration/references/phase1-analysis.md`

**Done criteria**: `aiconfigurator cli support --model-path <model> --system h200_sxm` correctly identifies the model.

---

## Phase 2: Build a Mock Layer

**Goal**: Build a standalone runnable operation layer for profiling and data collection, and define the collector-facing modeling contract.

**Reference**: `agents/new-op-integration/references/phase2-mock-layer.md`

**Done criteria**: The mock layer runs independently with correct input/output shapes; axis choice (`x=b*s` vs separate `b,s`) and collector schema are documented in this phase.

---

## Phase 3: Align nsys Profiles

**Goal**: Validate that mock-layer kernel behavior matches full-model E2E inference.

**Reference**: `agents/new-op-integration/references/phase3-nsys-alignment.md`

**Done criteria**: Kernel names match and latency is within a reasonable range (<2x gap).

---

## Phase 4: SDK Integration

**Goal**: Integrate the new operation into the aiconfigurator modeling stack.

**Reference**: `agents/new-op-integration/references/phase4-integration.md`

**Done criteria**: `operation.query(db, x=...)` returns `PerformanceResult`; CLI outputs complete modeling results.

---

## Phase 5: Performance Data Collection

**Goal**: Collect benchmark data across batch/sequence combinations.

**Reference**: `agents/new-op-integration/references/phase5-collection.md`

**Done criteria**: `*_perf.txt` files are generated with expected data point counts.

---

## Full Integration Checklist

After all phases are complete, run final verification:

```
□ common.py: `ModelFamily` enum value added (if needed)
□ common.py: `ARCHITECTURE_TO_MODEL_FAMILY` mapping added
□ utils.py: `config.json` parsing logic added
□ operations.py: new `Operation` subclass implemented (`query` + `get_weights`)
□ perf_database.py: data loader + query method implemented
□ models.py: new model class implemented (or existing one updated)
□ systems/data/: performance data files placed correctly
□ collector/: collection scripts run correctly
□ CLI validation: `aiconfigurator cli support/default` works
□ Tests: existing tests remain intact
□ Phase 2 documented: axis decision (`x=b*s` vs separate `b,s`) justified from SOL and reflected in collector schema
```

---

## Lessons Learned (From Real Integration Work)

The following lessons come from DeepSeek-V3.2 DSA integration and apply to any new operation.

### 1. Operation Granularity Must Match Existing Operations

**Problem**: If a new op includes sub-ops already modeled independently (such as GEMMs), you will double-count.
**Rule**: Align granularity with the most similar existing op (for example, MLA).
- If MLA collects attention kernels only, your new op should do the same.
- If the new op must be collected at a coarser granularity (whole attention block), remove included GEMMs in `models.py`.
- Verification: compare static per-layer latency with `cli default` between baseline and new model under the same config.

### 2. Data Format Must Exactly Match Existing Patterns

**Problem**: `perf_database.py` interpolation/query infrastructure assumes strict dict structures.
**Rule**: Copy the load/query/interpolation pattern of the closest existing op.
- **Nested dict levels**: MLA context uses 5 levels `data[quant][kv_dtype][num_heads][s][b]`
- **Leaf type**: use `defaultdict()` (not `defaultdict(dict)`), otherwise dedup logic with `try/except KeyError` breaks because keys are created silently
- **Interpolation axis order**: context `(x=num_heads, y=s, z=b)`; generation `(x=num_heads, y=b, z=s)`; must match dict nesting
- **`_interp_3d` returns dict**: wrap `{"latency": ..., "energy": ...}` into `PerformanceResult`
- **TP emulation**: collect on single GPU with `local_num_heads = global_heads // tp_size` and `Mapping(world_size=1, tp_size=1)`

### 3. Do Not Fallback to a Different-Granularity Operation

**Problem**: If fallback op granularity differs, estimates are wrong.
**Rule**: When new-op data is unavailable, raise `PerfDataNotAvailableError` rather than silently falling back. Fallback is allowed only when granularity is truly equivalent.

### 4. Validate FP8 / Quantization Support in the Runtime Framework

**Problem**: Model configs may infer FP8, but framework implementation of the new op may not support it.
**Rule**:
- Validate FP8 KV-cache execution in collector (with `try/except`).
- If unsupported, force override to supported dtype in `models.py` and log it.
- Verify support differences by SM generation (for example SM90 vs SM100+ kernel paths).

### 5. Proper Phase 3 Alignment Method

**Do**: extract the target submodule from a full decoder layer (for example `decoder.self_attn`) and benchmark it with exactly the same inputs as the standalone collector mock layer.
**Do not**: compare only kernel-name lists (necessary but not sufficient).
**Pass criteria**: latency gap < 10% and consistent CUDA Graph capture behavior.

### 6. Use Static Mode for Per-Op Differential Diagnosis

When modeling looks wrong, compare per-op per-layer latency directly via `op.query(db, ...)` to isolate problematic ops quickly:
```python
for op in model.context_ops:
    r = op.query(db, x=b*s, batch_size=b, s=s, prefix=0, beam_width=1)
    print(f"{op._name}: {float(r)/op._scale_factor:.4f} ms/layer")
```
