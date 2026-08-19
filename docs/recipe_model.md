# RecipeModel — trace-driven model building (pilot)

Status: **draft / pilot** (GLM-5.2 family only). This page documents the
architecture skeleton introduced by the recipe-model pilot so reviewers can
evaluate the approach on one model before any hand model is migrated.

## Why

Hand-written model classes (`sdk/models/*.py`) encode what we *believe* a
framework executes: which projections exist, how the shared expert is placed,
which layers skip the DSA indexer, how weights shard under TP. That knowledge
drifts silently as frameworks evolve. A **recipe** is the same knowledge
**machine-extracted from real framework execution traces**, so the model
builder, the collector, and the report all consume one set of facts.

The GLM-5.2 pilot (sglang 0.5.16 traces on SM90, shadow-diffed against
`DeepSeekV32Model` on the sglang 0.5.14 b200_sxm database) validated the
approach and caught real drift:

| Finding | Disposition |
|---|---|
| Per-kind attention decomposition (21 full / 57 shared / 3 dense of 78 layers, counts read from the checkpoint config) reproduces the hand model's fraction amortization **exactly** | validation |
| sglang fuses the shared expert into the routed MoE (traced `w13=[257,4096,6144]`, `topk_ids=[*,9]`; router logits stay 256-wide) — hand model + collected grid use 256/topk-8 + separate shared FFN | **tolerated divergence** — mapped by policy, no collection change (impact ≤3% TTFT, <0.5% TPOT @ b=128) |
| GLM-5.2's 3 dense head layers (`first_k_dense_replace=3`) are not modeled by the hand model (all 78 layers counted as MoE) | align model-side (cheap; recipe already models it) |
| GLM-5.2-FP8 attention projections run fp8_block deep_gemm kernels, but the hand model keys DSA module tables with `gemm=bfloat16`, and 0.5.14 skip-indexer rows exist for bfloat16 only | needs collector provenance; documented `dsa_gemm_override` workaround |

Full report: the opharness workspace `recipes/PILOT_REPORT.md` (probe +
extractor live there for now; they move into `collector/` when the approach is
funded).

## What a recipe is

`aiconfigurator_core/recipes/<org>--<model>.recipe.yaml`
(schema `aic-model-recipe/v0`), extracted per (model, framework version,
quant identity). Contents:

- `layer_kinds.<kind>.<phase>.layer_ops`: the **ordered** per-layer op sequence
  with module identity, traced tensor shapes, kernels, and framework call paths
  (`file:line` chains) — one representative layer, asserted homogeneous.
- `layer_map`: layer-kind counts for the real checkpoint depth, read from the
  checkpoint config (`indexer_types` × `mlp_layer_types` for GLM).
- `guards`: shape-dependent branches, **derived** from kernel/op-set diffs
  between prefill lengths straddling a config threshold (GLM: `index_topk=2048`
  separates dense-MHA FA3 from flashmla-sparse), flagged
  `needs_human_confirm` — never hand-written.
- `sharding_rules`: per-param TP rules validated from real tp1-vs-tp2 loads
  (GLM: `q_b/kv_b/lm_head/embed` dim0/tp, `o_proj` dim1/tp, indexer +
  `fused_qkv_a` replicated, experts dim1/tp).
- `quant_methods_by_module`, `weights_by_kind`, provenance (trace files,
  extractor hash, `evidence: real`).

## How RecipeModel maps facts to queries

`sdk/models/recipe.py :: get_recipe_model(model_path, model_config, backend)`
fills `context_ops` / `generation_ops` like any hand model; the evaluation
layer is untouched.

- Attention: one `ContextDSAModule`/`GenerationDSAModule` per layer kind
  (`dsa_full_layer_fraction` 1.0 for full-indexer kinds, 0.0 for shared),
  quant modes from the traced quant-method classes, FMHA dtype from the traced
  attention input dtype.
- MoE: expert count / intermediate size from traced expert weights, topk from
  traced `topk_ids`, subject to the **mapping policy** below.
- Dense MLP: GEMM widths from traced weights.
- Scaffold the trace cannot see (residual+norm, embedding, logits, P2P):
  the same generic formulas hand models use.

### Facts vs policy

The recipe always records what ran. Where reality diverges from the collected
data grid, an explicit policy decides the query, and every application is
appended to `model.mapping_notes`:

- `moe_policy="decompose_fused_shared"` (default): a fused shared expert
  (traced experts > router width) is mapped back onto the collected
  decomposition — MoE(router_width, topk−n_shared) + shared-expert FFN GEMMs,
  decode overlapped. Owner decision: re-collecting fused shapes would ripple
  through the whole collection pipeline for a bounded, small effect.
- `moe_policy="faithful"`: query traced shapes as-is (lands in the empirical
  tier under `DatabaseMode.HYBRID`); used to re-quantify the tolerated
  divergence whenever data or frameworks change.
- Coverage gaps raise `RecipeGapError` — a layer kind that was never traced,
  an unmapped quant class, an absent recipe. No silent fallbacks.

## Try it

```python
from aiconfigurator.sdk import config
from aiconfigurator.sdk.backends.factory import get_backend
from aiconfigurator.sdk.models import get_model, get_recipe_model
from aiconfigurator.sdk.perf_database import get_database
from aiconfigurator.sdk import common

db = get_database("b200_sxm", "sglang", "0.5.14")
db.set_default_database_mode(common.DatabaseMode.HYBRID)
mc = lambda: config.ModelConfig(tp_size=1, moe_tp_size=1, moe_ep_size=1)
rc = config.RuntimeConfig(batch_size=32, beam_width=1, isl=4096, osl=256)
backend = get_backend("sglang")

recipe = backend.run_static(get_recipe_model("zai-org/GLM-5.2", mc(), "sglang"), db, rc, "static")
hand = backend.run_static(get_model("zai-org/GLM-5.2", mc(), "sglang"), db, rc, "static")
# compare recipe.get_context_latency_dict() vs hand's — attention/scaffold match
# exactly; the residual is the dense-head layers the hand model does not model
```

## Out of scope (pilot)

Context parallelism, WideEP, memory modeling (reuses the DSA formula),
non-DSA attention families, and the extractor/probe themselves (currently in
the opharness workspace; they become a collector stage if this is funded).
