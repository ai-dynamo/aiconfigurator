---
description: >
  Adding/changing collector case coverage: base grids, model shapes, capability floors.
paths:
  - "collector/**"
  - "tests/unit/collector/**"
---

# Collector Case Authoring

How to add or change collection coverage. The declaration surface is exactly
two kinds of YAML plus one capability table — if you feel the need for a new
kind of rule, re-read `layer_permissions.md` first.

## The equation

```text
plan cases = dedup( expand(base sweep grid) ∪ expand(model shapes) )
```

- **Base grids** (`cases/base_ops/<op>.yaml`) give interpolation its uniform
  support: token/batch/seq sweeps, parallelism sizes, routing distributions.
  One file per op. Density here is the only knob for collection cost.
- **Model shapes** (`cases/models/<architecture>_cases.yaml`,
  `model_case_values`) guarantee exact hits for real models and keep
  correlated dimensions together (query heads / KV heads / head_dim / window
  are ONE tuple — never crossed with another model's values).
- **Dedup** happens on physical tuples when base and model expansions overlap.

## Adding coverage

| Goal | Action |
|---|---|
| New architecture | one new `cases/models/<architecture>_cases.yaml` |
| New model, same architecture | add path to `model_paths` in the existing file |
| New checkpoint quant artifact | new `model_case_values` row / `framework_quantization` allowed_modes — never merge quant-distinct artifacts into one row |
| Change a shared sweep | edit the op's `cases/base_ops/<op>.yaml` |
| New op collector | only when no existing op can produce the data points |
| Op needs newer SM than some GPUs | axis `min_sm` (quant modes) or `cases/capabilities.yaml` |

## Model file structure (unchanged from v2 core)

- `architecture`, `model_path`, `model_paths` (aliases), `include_base`,
  `base_ops`, `model_ops`, `framework_specific_base_ops`,
  `model_specific_base_ops` — op activation.
- `all_frameworks_op_cases` / `framework_specific_op_cases` — op sections;
  the only meaningful value is `cases: all` (activation). There are no
  selectors: `case_ids` / `contains` / `indices` / `ranges` / `limit` /
  `rules` no longer exist.
- `model_case_values` / `framework_specific_model_case_values` — shapes.
  `model_aliases` = one physical case, many artifact names (shape-only ops).
  `model_paths` inside a row = one case per path (runtime-sensitive).

## Legitimate shape narrowing: declare, never filter

There are no selectors, so "I need fewer shapes for this model" has exactly
one legal form: **declare the correlation and let the generator expand only
valid combinations**. Route by who owns the fact:

| The constraint belongs to | Where it goes | Existing example |
|---|---|---|
| the model (head count, valid TP shards, windows) | a field on its `model_case_values.<op>` row | `attention` profiles: `tensor_parallel_sizes` — the generator expands only valid shards of that tuple |
| the op's universal math (identities, budgets) | generator constraint / base-op budget field | `tp*ep == num_gpu`; `batch*seq <= max_context_tokens` |
| the platform (dtype floor, memory) | `capabilities.yaml` / the collector memory filter | see `layer_permissions.md` |
| the framework (kernel limits) | probe-and-raise / `FIXME(kernel-limit)` | see `layer_permissions.md` |

If the op's generator does not yet support the model-correlated field you
need (e.g. `mla_bmm` sweeps a global head grid with no model axis), the fix
is to extend that generator to honor a declared field — a mechanism change,
so propose it to the human. Do NOT approximate it with any post-generation
filtering; that is how the selector engine was born. Over-collection in the
meantime is acceptable: extra valid points are interpolation support, and
invalid ones fail into the classified log.

Subset selection is a RUNTIME concern, never persisted to YAML:

```bash
python3 collect.py --backend sglang --model-path <model> --gpu b200_sxm   # one model
python3 collect.py --backend trtllm --ops moe --case-filter "tp=4"        # substring filter
python3 collect.py --backend trtllm --resume                              # finish an interrupted run
```

## What NOT to do

1. No generation recipes inside model op sections — shapes go in
   `model_case_values`, recipes in `base_ops/`.
2. No per-model narrowing of another op's grid "to save time" — extra points
   are cheap; missing interpolation support is not. Tune the base grid instead.
3. No new YAML keys that condition on batch/seq/token/feature values.
4. Do not edit `capabilities.yaml` for a framework gap — hardware facts only.
