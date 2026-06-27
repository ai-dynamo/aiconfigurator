# Collector V2 Case Pruning

## Goal

Collector V2 should keep Collector V1's useful measurement coverage, retain
intentional V2 additions, and avoid scheduling cases created only by unrelated
Cartesian-product axes or repeated model artifact names.

For full/raw collection, the compatibility invariant is:

```text
V1 physical cases ⊆ cleaned V2 physical cases
```

For every migrated operation with a frozen historical baseline,
`removed_v1_cases` must be zero. A lower total case count is not sufficient
evidence because a plan can add many new cases while still deleting old
interpolation anchors.

Targeted model collection has a different contract: it is model-exact. It does
not inherit unrelated synthetic V1 interpolation anchors when the selected
model has an explicit structural profile.

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

The comparison uses consumer-visible physical lookup keys, not model aliases,
scheduler task IDs, latency, or power measurements.

## Three identities

Collector population must keep three different identities separate:

1. **Recipe identity** explains why YAML requested a case. Multiple model
   documents may provide provenance for the same work.
2. **Benchmark invocation identity** contains everything that can change the
   executed kernel or runtime setup, including path-dependent checkpoint
   quantization.
3. **Persisted physical key** is the unchanged key used by the current AIC
   consumer to load a measurement.

Deduplication is safe only when benchmark invocation identity is equivalent and
the persisted physical key is also equivalent. A consumer-key collision alone
does not prove two invocations are interchangeable. In particular, a BF16,
FP8, or NVFP4 checkpoint path may select different runtime behavior before its
native quantization becomes explicit.

## Population flow

```text
additive YAML profiles
    -> select the model/backend operations
    -> expand correlated structural tuples
    -> reject unreachable/backend-unsupported tuples
    -> derive operation-local invocation identity
    -> stable deduplication of proven-equivalent invocations
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
2. Preserve a field if it changes either the benchmark invocation or any
   current Python/Rust consumer key.
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

## Pruning decisions

| Situation | Population behavior | Reason |
|---|---|---|
| Head/KV-head/head-dim/window values from different models | Keep correlated model profiles; do not cross them | Cross-model tuples are not deployable shapes |
| Shape-only collector with base/FP8/NVFP4 names and an independent quant axis | Canonicalize artifact aliases | Artifact name does not change the invocation or persisted key |
| Module collector that reads checkpoint-native quantization | Retain each path until native quantization is explicit | The path can change the executed kernel |
| Different total-head/TP pairs with the same standalone-MLA local-head key | Deduplicate in the getter | Both invocation and current loader key are equivalent |
| Experimental op with no production consumer | Keep the registry entry, omit it from default model plans | Explicit research runs remain possible without default collection cost |
| Equivalence is uncertain | Retain the cases | Conservative pruning avoids silent coverage loss |

## Attention result

The following canonical B200/SM100 counts compare full/raw physical attention
cases. `Removed` is always measured against the V1 baseline.

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
5. Derive benchmark invocation identity before deciding that artifact names or
   quantization variants are duplicates.
6. Require `V1 - candidate == ∅` for full/raw collection.
7. Verify targeted model plans do not activate unrelated base operations or
   inherit unrelated synthetic V1 anchors.
8. Keep an old synthetic anchor when an unchanged consumer still queries it,
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
