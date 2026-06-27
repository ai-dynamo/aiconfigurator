# Collector V2 Case Pruning

## Goal

Collector V2 should keep Collector V1's useful measurement coverage, retain
intentional V2 additions, and avoid scheduling cases created only by unrelated
Cartesian-product axes or repeated model artifact names.

The compatibility invariant is:

```text
V1 physical cases ⊆ cleaned V2 physical cases
```

For every migrated operation, `removed_v1_cases` must be zero. A lower total
case count is not sufficient evidence because a plan can add many new cases
while still deleting old interpolation anchors.

## Scope

This work changes Collector inputs and case generation only:

- `collector/cases/**/*.yaml`
- `collector/case_generator.py`
- `collector/model_cases.py`
- operation-local case getters
- Collector tests and documentation

It does not change AIC SDK or Rust lookup behavior, EngineSpec, or Dynamo
Planner. Collector output must continue to satisfy the existing consumers.
Consumer problems found during the audit belong in separate changes.

## Baselines

- Attention V1: `a4827ce203e9fbc24fe6c6779a7eaa2a7dc79f1a`, immediately before
  Collector V2.
- Encoder attention: the original hardcoded grid from PR #1092, because the
  pre-V2 attention baseline predates this collector.
- Collector V2 before pruning: upstream `66c6e05fef00cbee6546847fa2280116ef4a38cd`.

The comparison uses physical benchmark inputs, not model aliases or scheduler
task IDs. Measurement fields such as latency and power are not case identity.

## Population flow

```text
additive YAML profiles
    -> select the model/backend operations
    -> expand correlated structural tuples
    -> reject unreachable/backend-unsupported tuples
    -> stable operation-local deduplication
    -> apply model and SM selectors
    -> benchmark queue
```

The important distinction is between an axis list and a structural profile.
For attention, `(query heads, KV heads, head dimension, window size, TP)` is a
correlated tuple. Combining global lists for those fields produces shapes that
no model uses. Batch and sequence axes can still be swept after the structural
tuple is selected.

YAML remains additive. Deduplication happens after enough information is known
to identify the actual benchmark point. The first occurrence wins so ordering
and resume behavior remain deterministic.

## Safe deduplication rules

1. Never change a downstream consumer to make a Collector case appear useful.
2. Preserve a field if any current Python or Rust consumer distinguishes it.
3. Collapse artifact aliases only for shape-only collectors where the model
   path is neither loaded by the benchmark nor part of its persisted key.
4. Do not alias BF16, FP8, or NVFP4 checkpoints before a module benchmark has
   resolved path-dependent native quantization. Once quantization is explicit,
   an operation-local key may collapse only truly identical benchmark inputs.
5. For standalone MLA, total heads and TP may produce the same local-head
   kernel. The getter deduplicates on `(dtype, local heads, batch, sequence)`.
6. SM exclusions are pre-execution skips. They are not expected failures after
   a case has already run.
7. Unknown or unproved equivalence is retained. It is better to prune less than
   to silently remove a V1 interpolation anchor.

## Attention result

The following canonical B200/SM100 counts compare physical attention cases.
`Removed` is always measured against the V1 baseline.

| Backend | Operation | V1 | V2 before | Cleaned V2 | Added | Removed |
|---|---|---:|---:|---:|---:|---:|
| SGLang | context | 33,714 | 122,676 | 50,901 | 17,187 | 0 |
| SGLang | generation | 19,484 | 53,654 | 39,556 | 20,072 | 0 |
| TRT-LLM | context | 63,192 | 143,739 | 75,483 | 12,291 | 0 |
| TRT-LLM | generation | 40,240 | 155,582 | 54,318 | 14,078 | 0 |
| vLLM | context | 40,392 | 84,296 | 50,932 | 10,540 | 0 |
| vLLM | generation | 36,288 | 68,920 | 53,638 | 17,350 | 0 |
| vLLM XPU | context | 16,188 | 16,188 | 17,838 | 1,650 | 0 |
| vLLM XPU | generation | 26,322 | 26,322 | 30,728 | 4,406 | 0 |
| **Total** | | **275,820** | **671,377** | **373,394** | **97,574** | **0** |

The unpruned V2 vLLM grids also removed 10,098 context and 8,774 generation
V1 cases, all from the historical `(head_dim=128, window=128)` region. The
legacy profile restores those points, while model-native profiles add valid
new window/head combinations without recreating the global Cartesian product.

Encoder attention retains all 7,008 original hardcoded cases and adds 671
model-native cases, for 7,679 total.

## Other operation results

These are deterministic shared YAML recipe counts. Backend-specific expansion
may multiply a recipe by dtype, TP/EP, or token lists.

| Operation | V1 | V2 before | Cleaned V2 | Notes |
|---|---:|---:|---:|---|
| GEMM | 35,742 | 35,742 | 35,742 | unchanged |
| ComputeScale | 1,628 | 1,628 | 1,628 | shared recipe unchanged; V2 also activates SGLang/vLLM |
| MoE common | 1,797 | 4,548 | 2,931 | artifact duplicates removed; V2 physical additions retained |
| MLA context specs | 220 | 550 | 220 | getter emits 1,760 unique loader keys |
| MLA generation specs | 362 | 885 | 362 | getter emits 2,896 unique loader keys |
| Mamba | 8 | 8 | 12 | four V1-compatible interpolation anchors added |
| GDN | 16 | 16 | 16 | unchanged |
| mHC | 8 | 8 | 4 | four unique shape/phase profiles; token expansion is unchanged |
| MLA BMM pre/post | 400 / 448 | 400 / 448 | 400 / 448 | unchanged |

For MoE on B200, the cleaned schedules remove repeated artifact tasks without
removing any expanded physical tuple. For example, SGLang drops from 366,072
V2 tasks to 218,514 while retaining all 211,920 distinct conservative tuples.

DSA module population removes artifact-only repetition only where quantization
and architecture are already explicit. SGLang keeps its checkpoint tasks
because native quantization is path-dependent; TRT-LLM and vLLM can safely
deduplicate after their explicit module inputs are known. Every projected key
set is unchanged:

| Backend | Operation | Scheduled before | Scheduled cleaned | Removed projected keys |
|---|---|---:|---:|---:|
| SGLang | context | 792 | 792 | 0 |
| SGLang | generation | 48 | 48 | 0 |
| TRT-LLM | context | 46,848 | 23,424 | 0 |
| TRT-LLM | generation | 35,328 | 17,664 | 0 |
| vLLM | context | 70,272 | 35,136 | 0 |
| vLLM | generation | 35,328 | 17,664 | 0 |

The analogous GLM MoE artifacts are deliberately not merged: SGLang selects
native FP8/NVFP4 MoE quantization by artifact path, so those paths represent
real additional measurements rather than scheduler duplicates.

## Review checklist

When adding or changing a model profile:

1. Identify whether the collector loads the model or only uses its dimensions.
2. Keep correlated dimensions in one profile; do not append them to unrelated
   global axes.
3. State whether quantization is selected by the model artifact or expanded by
   the collector.
4. Compare physical key sets, not only totals.
5. Require `V1 - candidate == ∅` for full/raw collection.
6. Verify targeted model plans do not activate unrelated base operations.
7. Keep an old synthetic anchor when an unchanged consumer still queries it,
   even if the current model metadata would choose a different value.

## Validation

```bash
pytest -q tests/unit/collector
ruff check collector tests/unit/collector
ruff format --check collector tests/unit/collector
```

The unit coverage loads test-only V1 physical-key fixtures and executes the
current backend getter bodies, so batch, sequence, precision, head topology,
window, and phase keys are all checked for zero removal. It also freezes encoder
anchors, per-operation recipe counts, selector narrowing, alias handling, and
operation-local physical deduplication. No coverage fixture is loaded by the
Collector runtime.
