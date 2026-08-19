# Model facts — the explicit hf→config conversion, checked (pilot)

Status: **draft / pilot** (GLM-5.2 family). This page documents the model-facts
seam so reviewers can evaluate the approach on one model.

## Positioning

Model classes stay the single production path, authored (by an engineer or an
AI) against **config interpretation** — exactly as today. What changes is that
the conversion feeding `get_model` becomes an explicit, checkable step:

```
HF config ──► assemble_model_facts(model_path, model_config, backend, system)
                 ├─ structural facts: layer kinds (GLM-5.2: 3 dense-head /
                 │    18 full-indexer / 57 shared-indexer), expert config
                 │    (256 routed, topk 8, +1 shared, inter 2048), kv identity,
                 │    branch params (index_topk=2048)
                 ├─ quant resolution: checkpoint inference + the system-aware
                 │    resolve_* remaps, folded into ONE choke point
                 └─ declared APPROXIMATIONS: deliberate simplifications with
                      rationale + measured impact bound
                       │
     check_facts_against_dryrun(facts, references/dryrun/*.json)   ← evidence
                       │
              verified facts ──► get_model builds the model
                       │
     check_model_against_facts(model, facts)                       ← structure
```

Checks report `MATCH / APPROX / DIVERGENT / UNCHECKED`, never block, and run
at authoring time + CI — not in the build hot path. The evidence artifact is
the **raw dry-run JSON** (real framework loads of depth-cut dummy variants,
`references/dryrun/`); there is no intermediate format.

## Declared approximations (facts vs modeling, both owner decisions)

Facts always record the TRUE structure. An approximation declares how a model
is *allowed* to blur it — so checks distinguish intent from drift:

| Rule | What it allows | Measured impact (GLM-5.2 pilot) |
|---|---|---|
| `dense_head_as_moe` | `first_k_dense_replace` dense head layers counted as MoE (simpler model classes) | −0.4%…−3.5% e2e, overestimate direction (conservative) |
| `fused_shared_expert_decomposed` | runtime fuses the shared expert into the routed MoE (traced 257 experts / topk 9); modeled as 256/topk-8 + shared FFN, matching collected data | ≤3% TTFT / <0.5% TPOT @ b=128 vs faithful fused query |

## The unified quant resolution (and the Mocker gap it closes)

`resolve_model_quant_modes` folds `_apply_model_quant_defaults` + the
system-aware `resolve_*` helpers into one call. Previously these were each
caller's duty: `cli/api.py` carries three copies, `task_v2` inlines a fourth —
and the **`compile_engine` path (Rust `build_aic_engine` / Dynamo Mocker) ran
none**, so an embedded caller silently kept e.g. native-FP4 compute
assumptions for an nvfp4 checkpoint on Hopper. `compile_engine` now calls the
unified resolver (explicit quant kwargs still win).

## What the checks catch today (pinned in `tests/unit/sdk/models/test_model_facts.py`)

- facts ↔ dry-run: kv identity and per-module quant classes MATCH; the fused
  shared expert reports APPROX via its declared rule; a layer kind without
  dry-run evidence reports UNCHECKED (coverage is surfaced, never silenced).
- hand model ↔ facts: attention coverage / skip-indexer fraction (21 full of
  78) MATCH; dense-head-as-MoE reports APPROX; **GLM-5.2-FP8 DSA quant key
  reports DIVERGENT** — the checkpoint's attention projections execute
  fp8_block deep_gemm kernels while the model keys the DSA module tables with
  bf16 (and the 0.5.14 skip-indexer rows were collected at bf16 only). Open
  finding pending collector provenance; the pinned test flips when settled.

## Out of scope (pilot)

Wiring model classes to consume facts directly (follow-up migration), CP/WideEP
facts, memory facts, non-DSA families, and the probe itself (opharness
workspace; candidate `collector/` stage if this direction is funded).
