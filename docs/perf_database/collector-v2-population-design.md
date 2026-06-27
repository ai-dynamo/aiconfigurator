# Design: Collector v2 Rule Population and Physical-Case Deduplication

## Status

Proposed.

## Summary

Collector v2 should allow model and backend coverage to grow by appending YAML
rules without requiring every YAML author to reason about scheduler duplication,
consumer lookup keys, or collector-v1 compatibility.

This document proposes treating case population as a small compiler:

```text
YAML rules
  -> parse and validate
  -> expand logical candidates lazily
  -> normalize aliases and defaults
  -> filter unreachable and unsupported candidates
  -> derive invocation and physical-row identities
  -> deduplicate and detect conflicts
  -> enforce compatibility manifests
  -> emit a deterministic executable plan and an explanation report
```

YAML remains additive and model-centric. Python code remains authoritative for
physical identity and consumer compatibility. This division lets the YAML
catalog grow while preventing duplicate collection and accidental loss of old
coverage.

## Motivation

Collector v2 moved model intent into YAML, which is the right ownership model,
but several failure modes remain possible as the catalog grows:

1. Independent dimensions can form structurally impossible Cartesian products.
2. Multiple model artifacts can schedule the same kernel when quantization is
   already an independent axis.
3. A collector, output writer, Python database loader, and Rust database loader
   can silently disagree about the identity or ordering of a case.
4. Comparing only case counts cannot prove that collector-v1 lookup keys are
   preserved.
5. Local `seen` sets in individual collectors deduplicate different notions of
   identity and are difficult to audit globally.

The current collector already contains pieces of the desired design: explicit
base-op activation, correlated attention profiles, model aliases, targeted
model plans, and several collector-local physical-key deduplicators. The
proposal makes these mechanisms a common population contract.

There is also a current split-brain integration point to remove. `model_cases.py`
loads YAML to build op selectors, while `case_generator.py` scans the model YAML
catalog again and uses `COLLECTOR_MODEL_PATH` to narrow generated values. As a
result, a selector plan and a generated case list can follow related but not
identical merge semantics. The proposed compiler becomes the single YAML-to-plan
path. During migration it may adapt typed logical cases back to legacy tuples,
but backend collectors must not independently reload the catalog.

## Goals

- Let authors add independent YAML rules without deep-merging their dimensions
  into one global Cartesian product.
- Remove duplicate benchmark invocations after canonical normalization.
- Reject conflicting rows instead of relying on last-write-wins behavior.
- Filter model-unreachable and backend-unsupported candidates before workers
  are launched.
- Guarantee that a full/raw plan contains every protected collector-v1
  physical key.
- Preserve provenance so every scheduled or skipped case can be explained.
- Keep plan generation deterministic and memory-bounded.
- Support incremental migration one op at a time.

## Non-goals

- The YAML format will not become a general expression language.
- The planner will not infer arbitrary model architecture facts from checkpoint
  code at collection time.
- The first implementation will not generate Python and Rust query code from a
  shared schema.
- The planner will not replace framework-specific benchmark setup.
- This design does not automatically reduce every valid but low-value sampling
  point; sampling policy remains an explicit sweep-policy concern.

## Design principle

YAML authors append rules. They do not define physical identity.

A rule describes why a workload should exist and which logical dimensions it
uses. An op schema in Python decides how that workload is normalized, whether
it is reachable, how it is invoked, and how its output maps to a database row.

This distinction is important. Deduplication can remove two candidates that
normalize to the same physical key, but it cannot determine that two distinct
keys represent an impossible model topology. Structural correlation must come
from a model profile or a structured capability constraint.

## Proposed YAML semantics

Rules are additive clauses. Each clause expands independently, and the planner
takes the union of their normalized output. Lists from different clauses are
never implicitly crossed.

```yaml
schema_version: 2
op: attention_context

rules:
  - id: collector_v1_compat
    profiles:
      - query_heads: 64
        kv_heads: 8
        head_dim: 128
        window_size: 0
        tensor_parallel_sizes: [1, 2, 4, 8]
      - query_heads: 48
        kv_heads: 8
        head_dim: 128
        window_size: 4096
        tensor_parallel_sizes: [1, 2, 4, 8]
    sweep:
      batch_sizes: [1, 2, 4, 8, 16]
      sequence_lengths: [128, 512, 2048, 8192]

  - id: qwen3_vl_delta
    when:
      model_families: [qwen3_vl]
      backends: [sglang, trtllm, vllm]
    profiles:
      - query_heads: 32
        kv_heads: 8
        head_dim: 128
        window_size: 0
        tensor_parallel_sizes: [1, 2, 4, 8, 16]

model_aliases:
  Qwen/Qwen3-32B-FP8: Qwen/Qwen3-32B

compatibility:
  preserve_manifest: collector_v1/attention_context.json
```

### Structured predicates

`when`, `requires`, and `exclude` should accept schema-validated mappings, not
Python-like expression strings. Initially supported predicates should remain
small:

- backend membership;
- SM membership or numeric range;
- model family or canonical model membership;
- precision/quantization membership;
- divisibility and maximum/minimum constraints for TP, EP, heads, batch, and
  sequence length.

More complex op semantics belong in the op schema's `is_reachable` or
`is_supported` implementation.

### Alias semantics

A model alias states that two artifact names share model structure. It does not
silently change quantization policy. If the artifact name implies a precision,
that implication is normalized into an explicit quantization field before the
physical key is derived.

Aliases must not be used when an artifact changes runtime behavior, valid
parallel ranges, activation, kernel selection, or model loading.

## Planner data model

### RuleSource

Identifies the YAML file, rule id, model path, architecture, backend, and SM
context responsible for a candidate. Sources are accumulated during
deduplication rather than discarded.

### LogicalCase

A typed, normalized candidate before scheduling. It contains semantic fields
such as query heads, KV heads, head dimension, TP size, quantization, batch, and
sequence length. It must not depend on the string representation used by
`create_test_case_id`.

### InvocationKey

Identifies one benchmark execution. Two logical cases with the same
`InvocationKey` are run once even if several model rules require them.

The invocation key may include setup fields that do not appear in the database
query. For example, a collector that loads a real checkpoint may need a model
artifact in its invocation key even when a shape-only microbenchmark does not.

### PhysicalRowKey

Identifies one performance database row after output normalization. It includes
every field that changes the consumer lookup result and excludes provenance-only
fields such as an alias artifact name.

One invocation may emit several physical rows when a subprocess performs an
internal sweep. Schemas should predict those rows when practical and must
validate actual rows during output finalization.

### QueryKey

Represents the key constructed by an AIC consumer. The first implementation
does not need to call consumers from the planner, but checked-in parity fixtures
must demonstrate that Python and Rust build the same key from representative
rows.

### PlannedCase

Contains the normalized logical case, invocation key, predicted row keys,
merged provenance, and any expected-failure metadata needed by the runner.

### CaseDecision

Records a candidate's outcome and reason:

```text
scheduled
duplicate_invocation
unreachable_model_shape
unsupported_backend
unsupported_hardware
excluded_by_rule
conflicting_metadata
compatibility_removal
```

The report aggregates decisions but can also dump them as JSON for auditing.

## Op schema contract

Each migrated op registers an implementation similar to:

```python
class OpCaseSchema(Protocol):
    op_name: str

    def expand(self, rule: Rule, context: PlanContext) -> Iterable[LogicalCase]: ...
    def normalize(self, case: LogicalCase, context: PlanContext) -> LogicalCase: ...
    def is_reachable(self, case: LogicalCase, context: PlanContext) -> Decision: ...
    def is_supported(self, case: LogicalCase, context: PlanContext) -> Decision: ...
    def invocation_key(self, case: LogicalCase) -> Hashable: ...
    def predicted_row_keys(self, case: LogicalCase) -> Iterable[Hashable]: ...
    def row_key(self, output_row: Mapping[str, object]) -> Hashable: ...
```

The registry is the only source of physical identity for planning. Backend
collectors may retain temporary assertions during migration, but they should
not independently invent different deduplication keys.

## Population pipeline

### 1. Parse and validate

Load YAML with strict unknown-field validation. Resolve model aliases and rule
references, but retain source locations for diagnostics.

### 2. Resolve additive rules

Select rules for the requested model, backend, and SM. Do not merge sweep lists
across different rule ids. Duplicate rule ids are an error by default. Reusing
or modifying a rule requires an explicit relationship such as `extends` or
`replace`; it must never trigger an implicit `dict.update()` overlay. This
prevents an additive catalog from becoming a large rectangle again when two
files happen to use the same descriptive id.

### 3. Expand lazily

Each rule yields candidates as an iterator. The planner must not materialize a
large Cartesian product before it can apply cheap structural filters.

Within a rule, explicitly independent sweep dimensions may still form a
product. Correlated dimensions such as heads, head dimension, window, and valid
TP sizes belong in profile records.

### 4. Normalize

Normalization applies canonical enums, default values, model aliases, MHA KV
sentinels, quantization aliases, and op-specific field normalization before any
key is computed.

Normalization must be idempotent:

```text
normalize(normalize(case)) == normalize(case)
```

### 5. Filter reachability and capability

Reachability answers whether the case represents a declared model topology.
Capability answers whether the selected backend/hardware/kernel supports it.
Keeping these decisions separate makes reports actionable.

### 6. Deduplicate invocations

Use an ordered mapping from `InvocationKey` to `PlannedCase`. A duplicate merges
provenance and compatible metadata. Conflicting expected-failure or setup
metadata is an error unless an explicit merge policy exists.

The planner must never use last-rule-wins for a physical identity conflict.

### 7. Validate physical rows

At plan time, detect collisions among predicted physical-row keys. During
output finalization, derive the actual row key and reject conflicting duplicate
rows. Identical duplicate rows may be coalesced with their provenance recorded.

### 8. Enforce compatibility

For full/raw plans, compare generated physical keys with a checked-in legacy
manifest:

```text
legacy physical keys <= generated physical keys
```

Missing keys fail plan generation unless listed in a reviewed removal allowlist
with a reason and expiry/version. Targeted model plans do not need to contain
the complete legacy grid.

### 9. Emit deterministic output

Sort cases by a schema-defined stable ordering after deduplication. Shuffling,
when requested for execution balance, happens later in the runner and uses an
explicit seed.

## Compatibility manifests

Manifests should contain normalized physical keys, not collector case strings
or only aggregate counts. Because common attention manifests contain hundreds
of thousands of keys, store one deterministic compressed JSON Lines file per
backend/op under `tests/fixtures/collector_v1/`, together with a small
human-readable summary. Do not store only a total hash: CI must be able to
report the exact missing keys.

```json
{"op":"attention_context","backend":"vllm","key":["fp8","fp8",8,128,0,64,2048,4]}
```

Manifest generation must be a separate explicit maintenance command. Normal
tests consume checked-in manifests but never rewrite them.

Suggested commands:

```bash
python -m collector.planner manifest create --source collector-v1 --output ...
python -m collector.planner manifest diff --base ... --plan ...
```

## Plan explanation

`--plan-only` should report both scheduler and physical coverage:

```text
attention_context (vllm, sm100)
  expanded logical candidates:       90,824
  unreachable model shapes:         -24,512
  unsupported backend/hardware:      -8,116
  duplicate invocations:             -7,264
  scheduled invocations:             50,932

  collector-v1 physical keys:        40,392
  retained collector-v1 keys:        40,392
  true physical additions:           10,540
  removed protected keys:                 0
```

`--plan-json` should write all aggregate counters plus optional per-case
decisions and provenance. This artifact can be attached to collection PRs.

## Proposed package layout

```text
collector/planner/
  __init__.py
  models.py            # contexts, rules, cases, decisions, reports
  compiler.py          # common lazy population pipeline
  registry.py          # op-name -> OpCaseSchema
  predicates.py        # small structured predicate vocabulary
  manifests.py         # compatibility manifest loading and diffing
  reporting.py         # text and JSON explain output
  schemas/
    attention.py
    mla.py
    gemm.py
    moe.py
    state_space.py
    dsv4.py

tests/fixtures/collector_v1/
  <backend>/<op>.json.gz
```

`collector/model_cases.py` remains responsible for resolving model, base-op,
framework, and SM YAML sources. It should produce typed additive rules for the
compiler instead of directly representing only string selectors.

`collector/collect.py` remains the execution coordinator. It consumes the
compiled invocation plan and continues to own shuffle, limit, resume, worker,
and expected-failure behavior.

## Incremental rollout

### Phase 0: shadow planning and reports

- Add core models, registry, compiler, and reporting.
- Adapt one existing case list into logical candidates without changing which
  cases execute.
- Compare old and new invocation lists in tests and `--plan-only` output.
- Do not enforce deduplication in production yet.

### Phase 1: attention and encoder

- Register attention schemas for SGLang, TRT-LLM, vLLM, and XPU.
- Move current profile normalization and physical-key logic behind the schema.
- Check in collector-v1 attention manifests.
- Enable invocation deduplication and compatibility enforcement for these ops.

Attention is the best first migration because it has the largest case volume,
well-understood consumer keys, and the clearest benefit.

### Phase 2: MLA and state-space ops

- Migrate micro MLA, MLA module, MLA BMM, WideEP MLA, GDN, Mamba, and MHC.
- Replace collector-local `seen` sets with schema assertions.
- Add Python/Rust row-to-query parity fixtures for ops consumed by Rust.

### Phase 3: remaining common and model-specific ops

- Migrate GEMM, ComputeScale, MoE, WideEP MoE, and DSV4.
- Make physical-row conflict validation mandatory during output finalization.
- Require every collector-v1-compatible op to name a manifest.

### Phase 4: schema-v2 YAML default

- Make additive rules the documented default.
- Retain a compatibility reader for schema-v1 YAML during one release window.
- Remove local deduplication only after plan and output-key parity are proven.

## Estimated implementation size

The estimates below are changed lines, not net-new lines, and include refactoring
existing logic into the common contract.

| Work item | Production code | Tests/tools | Likely files |
|---|---:|---:|---:|
| Core models, compiler, registry, predicates | 600-900 | 350-500 | 6-9 |
| Reporting, manifest tooling, CLI integration | 350-550 | 250-400 | 5-7 |
| Attention/encoder schema migration | 500-800 | 400-600 | 8-12 |
| MLA and state-space schema migration | 700-1,050 | 400-600 | 10-15 |
| GEMM/MoE/ComputeScale/DSV4 migration | 900-1,400 | 450-700 | 10-16 |
| Remaining active-op adapters and parity glue | 450-800 | 300-500 | 6-10 |
| Python/Rust consumer parity fixtures | 150-250 | 250-400 | 5-8 |
| Total full migration | 3,500-5,500 | 2,000-3,000 | 25-40 |

A useful first slice covering additive rules, the common compiler,
`InvocationKey`, explain output, and the main attention/encoder/MLA adapters is
approximately 1,500-2,200 production lines plus 1,000-1,500 test/tool lines
across 10-15 files. It can be reviewed independently before introducing hard
physical-row manifest enforcement.

The complete estimate covers roughly 32 active registry op names grouped into
about 14-16 schema families. Precise `PhysicalRowKey` support requires adapters
for collectors such as MLA module, Mamba, GDN, and DSV4, where one invocation
performs an internal sweep and emits multiple rows.

The manifest data itself may contain many rows but should be generated and is
not included in the hand-written code estimate.

## Testing strategy

Every schema should have the following contract tests:

1. normalization is idempotent;
2. case ordering is deterministic;
3. aliases that should collapse share an invocation and row key;
4. artifacts with different runtime policy do not collapse;
5. structurally impossible profiles are rejected before scheduling;
6. unsupported backend/SM combinations receive a stable reason code;
7. no two scheduled invocations predict conflicting physical rows;
8. the protected legacy manifest is a subset of the full/raw output;
9. targeted model output contains only the selected model's reachable profiles;
10. representative output rows map to the same Python and Rust query key.

Integration tests should run the new compiler in shadow mode against current
collector outputs, compare exact ordered keys, and then switch the op to
enforcement mode only after parity is established.

## Failure behavior

The planner should fail before launching workers when it encounters:

- an unknown YAML field;
- an unknown op schema;
- a conflicting invocation definition;
- a predicted physical-row collision with incompatible metadata;
- a protected legacy-key removal;
- a rule that expands beyond a configurable safety threshold before filtering.

Expected runtime kernel failures remain handled by the existing SM exception
and resume mechanisms.

## Alternatives considered

### Deduplicate only after full Cartesian expansion

Rejected. It wastes memory and cannot remove structurally impossible cases that
have distinct physical keys.

### Put arbitrary filter expressions in YAML

Rejected. It creates an untyped programming language, makes schema evolution
unsafe, and spreads kernel semantics across configuration files.

### Keep per-collector `seen` sets

Rejected as the long-term contract. Local sets are useful assertions during
migration but cannot enforce global provenance, compatibility manifests, or
consumer-key parity.

### Include model path in every key

Rejected. Many microbenchmarks are shape-only, and including artifact names
would preserve exactly the duplicate collection this design is intended to
remove. Model path belongs in an invocation key only when it changes setup or
runtime behavior.

## Acceptance criteria

The design is complete when:

- adding two YAML rules that normalize to one invocation schedules one worker;
- every deduplicated case retains all contributing rule/model provenance;
- full/raw plans cannot remove a protected collector-v1 physical key silently;
- impossible model profiles and unsupported backend cases are reported with
  distinct reason codes;
- output finalization detects conflicting physical rows;
- Python and Rust parity fixtures cover every Rust-consumed migrated op;
- `--plan-only` reports logical, invocation, and physical-key counts; and
- migrated collectors no longer need independent production deduplication logic.
