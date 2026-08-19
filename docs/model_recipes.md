# Model recipes — traced reference facts + drift checking (pilot)

Status: **draft / pilot** (GLM-5.2 family). This page documents the recipe
reference artifacts and the `recipe_check` validator so reviewers can evaluate
the approach on one model.

## Positioning

Model classes stay the single production path, built from **config
interpretation** (`_get_model_info` + quant defaults) the way they are today —
authored by an engineer or by an AI following `docs/add_a_new_model.md`. The
recipe is a **reference**, not a second source of truth and not a runtime
dependency:

```
framework traces ──extract──▶ recipe (recipes/*.recipe.yaml)
                                  │
        (authoring evidence)      │      (drift detection)
  AI / engineer writes the        ├──▶ check_model_against_recipe(model, recipe)
  model class from config         │        MATCH / TOLERATED / DIVERGENT / UNCHECKED
  interpretation, consulting ◀────┘        (unit-testable, no perf DB needed)
  the recipe for what the
  framework actually runs
```

## What a recipe records

`recipes/<org>--<model>.recipe.yaml` (schema `aic-model-recipe/v0`), extracted
per (model, framework version, quant identity):

- `layer_kinds.<kind>.<phase>.layer_ops`: the **ordered** per-layer op sequence
  with module identity, traced tensor shapes, kernels, and framework call
  paths (`file:line` chains).
- `layer_map`: layer-kind counts for the real checkpoint depth, read from the
  checkpoint config (GLM-5.2: 21 full-indexer / 57 shared-indexer / 3 dense of 78).
- `guards`: shape-dependent branches **derived** from kernel/op-set diffs
  between prefill lengths straddling a config threshold (GLM: `index_topk=2048`
  separates dense-MHA FA3 from flashmla-sparse), flagged `needs_human_confirm`.
- `sharding_rules`: per-param TP rules validated from real tp1-vs-tp2 loads.
- `quant_methods_by_module`, `weights_by_kind`, provenance (`evidence: real`).

## The checker

`sdk/models/recipe_check.py :: check_model_against_recipe(model, recipe)`
introspects the already-built op graph (no perf-database queries, never
blocks) and reports per **op family**:

- Comparisons are owned by registered **op-family checkers** whose `matches()`
  keys on traced evidence (attention-backend class, `moe::` span, module
  naming) — never a model name. New framework behavior = one new checker;
  a traced block nothing claims is reported `UNCHECKED`, not skipped.
- **Tolerated divergences are rules**: sglang fuses the shared expert into the
  routed MoE (traced 257 experts / topk 9; router logits 256-wide); the MoE
  checker recognizes the collected 256/topk-8 + shared-FFN decomposition and
  reports `TOLERATED` (owner decision — bounded effect, ≤3% TTFT / <0.5% TPOT
  @ b=128, not worth a collection-pipeline change). Anything else is
  `DIVERGENT`.

What it catches today (pinned by `tests/unit/sdk/models/test_recipe_check.py`):

| Check vs GLM-5.2 recipes | hand `DeepSeekV32Model` |
|---|---|
| attention layer coverage / skip-indexer fraction / kv identity | MATCH |
| fused shared expert → decomposition | TOLERATED |
| MoE layer coverage (78 modeled vs 75 traced) + dense head layers | **DIVERGENT** (pilot finding: `first_k_dense_replace=3` layers not modeled) |
| FP8: DSA module gemm key bf16 vs traced fp8_block deep_gemm projections | **DIVERGENT** (needs collector provenance) |

## Numeric context (pilot shadow diff)

A trace-faithful shadow model built from these recipes was diffed against the
hand model on b200_sxm / sglang 0.5.14 (opharness pilot,
`recipes/PILOT_REPORT.md` there): attention and scaffold agreed exactly;
end-to-end deltas −0.4%…−3.5% were fully attributable to the dense-head gap
above. That shadow constructor is deliberately NOT part of this repo — the
recipe stays a reference; construction stays config-driven.

## Out of scope (pilot)

CP/WideEP checks, memory modeling checks, non-DSA attention families, and the
probe/extractor themselves (opharness workspace; candidate `collector/` stage
if this direction is funded).
