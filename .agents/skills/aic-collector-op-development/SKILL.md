---
name: aic-collector-op-development
description: Design, add, review, or modify AIC Collector operations and their case population. Use for new Collector ops, backend registry entries, collector case YAML, case generators/getters, pruning or deduplication, persisted perf keys, framework-version routing, and audits of whether generated cases match Python/Rust consumers.
---

# AIC Collector OP Development

Build the smallest Collector change whose benchmark invocations are valid and
whose persisted rows satisfy the existing AIC consumer contract.

## Required reading

Before editing, read:

1. `packages/aiconfigurator-core/collector/README.md`
2. `packages/aiconfigurator-core/collector/cases/README.md`
3. `docs/perf_database/collector-v2-population-design.md`, especially
   **Three identities**, **Population flow**, and **Safe deduplication rules**
4. The relevant `packages/aiconfigurator-core/collector/<backend>/registry.py`, collector module, and model/base
   case YAML
5. Every Python and Rust loader/query that consumes the op's perf file

If the task proceeds to GPU collection, also use `$aic-auto-collect`.

## Non-negotiable rules

- Treat Collector output as an existing-consumer contract. Do not modify SDK,
  Rust, EngineSpec, or Dynamo Planner merely to make a generated case useful.
  Consumer changes require separate explicit scope.
- Do not call a case invalid merely because the synthetic collector cannot run
  it. If the framework and consumer support the shape, fix or record the
  Collector gap instead of pruning coverage.
- Do not restore coverage with a different kernel under the same logical label.
  Persisted quant/backend labels must describe the invocation that actually ran.
- Do not modify historical snapshots to make an alignment test pass. Use
  read-only comparisons as review evidence.
- Do not add permanent `legacy_*` runtime concepts for a one-time migration.
  Put production defaults in normal base-op YAML and keep migration comparisons
  outside the runtime schema.
- Do not ship a deduplication path unless current repository-owned YAML produces
  at least one duplicate invocation. Record that stage's before/after counts
  and prove that the unique invocation and persisted-key sets are unchanged.
  Remove no-op deduplication instead of manufacturing duplicate-only fixtures
  for it.
- Treat the public getter and any subprocess or inner re-expansion as one
  population contract. Full/raw and targeted entry paths must consume the same
  resolved YAML quantization and SM gates; do not reconstruct the policy with a
  second hard-coded hardware heuristic.
- A case-population change may touch `packages/aiconfigurator-core/collector/collect.py` only for model/op
  plan selection required by the resolved case plan. Generic resume, retry,
  checkpoint, logging, and output-finalization behavior require separate
  explicit scope.
- Anchor collector behavior to the requested framework version. Prefer one
  verified `__compat__`/version route over speculative multi-version branches.

## Step 1: Establish the consumer contract

Search from the perf filename and op name through both producers and consumers:

```bash
rg -n "<op_name>|<perf_filename>|query_<op>" collector src rust tests
rg -n "PerfFile|PerfDataFilename|log_perf" collector src rust
```

Record a temporary contract table; do not commit it unless it is useful product
documentation:

| Axis | Recipe source | Changes invocation? | Persisted key? | Consumer query values | Evidence |
|---|---|---:|---:|---|---|

Answer before coding:

- Which columns form the exact lookup key?
- Which axes interpolate, and which require an exact bucket?
- Which TP, EP, head, window, quant, dtype, phase, and backend values can the
  consumer request?
- Does a missing key fail, use an empirical fallback, or silently select another
  path?
- Does the op have a production consumer? If not, keep it registry-only or
  explicitly experimental instead of adding it to default model plans.

Never assume an adjacent key is a fallback. Verify it in the loader/query code.

## Step 2: Define three identities

Keep these separate:

1. **Recipe identity**: why YAML requested the work.
2. **Benchmark invocation identity**: every value that can change the executed
   model, kernel, runtime setup, or quantization.
3. **Persisted physical key**: the columns used by current consumers.

Deduplicate only when both invocation identity and persisted key are equivalent.
A persisted-key collision is a bug unless the invocations are proven identical.
If repository-owned cases map distinct invocation identities to one consumer
key, fail population with the conflicting owners and key. Do not silently use
first-wins or widen the SDK/Rust schema inside a Collector-only change.

Examples:

- Model aliases may collapse for a shape-only synthetic benchmark.
- Checkpoint paths must remain separate while native quantization or module
  behavior is path-dependent.
- W4A16 and W4A8 are distinct when activation precision changes.
- NVFP4, MXFP4, FP8, and INT4 labels are not interchangeable because geometry
  happens to match.

## Step 3: Design cases without accidental Cartesian products

- Keep independent workload axes such as batch or token count as lists.
- Keep correlated structural axes in one profile. Typical correlated fields are
  `(heads, KV heads, head dimension, window, TP)` and
  `(experts, top-k, hidden, intermediate, TP, EP, quant)`.
- Put shared production defaults in `packages/aiconfigurator-core/collector/cases/base_ops/<op>.yaml`.
- Put model-native topology and artifact policy in
  `packages/aiconfigurator-core/collector/cases/models/*_cases.yaml`.
- Make targeted structural population exact when a model profile exists; it
  may still reuse shared workload sweeps. Full/raw collection may union
  defaults and model profiles, then stably deduplicate.
- Use stable first-wins deduplication by default. If a later selector can
  distinguish equivalent recipe representations, choose and document a
  canonical representative that remains selectable (for example the smallest
  TP for a local-head key); otherwise deduplication can erase targeted coverage.
- Add a generic synthetic default only when a consumer or interpolation need is
  demonstrated.

For an existing op, compare current and candidate physical key sets during the
change. This is change-specific evidence, not a permanent compatibility mode.
Report `kept`, `added`, `removed`, and `deduplicated` separately; totals alone
cannot reveal coverage loss. Migration baselines, exact V1 totals, and
historical snapshots are review evidence, not permanent unit-test contracts
unless an unchanged consumer explicitly depends on that exact inventory.

## Step 4: Implement one vertical slice

Touch only the layers the op needs:

1. Base/model YAML and model-plan selection
2. Case generator/getter when YAML cannot express the required correlation
3. Backend collector implementation
4. Backend registry and exact framework version route
5. Persisted schema/logging
6. Focused tests for generated cases and consumer-visible keys

Before adding a helper, schema field, or filter—and again before finalizing—find
its producer in current repository-owned YAML and its reader on a production
population path. If a later design decision removes either side, delete the
orphan. Do not retain speculative aliases, phase selectors, compatibility
modes, or one-time audit scaffolding.

## Step 5: Prune and skip only with evidence

Acceptable evidence includes:

- Exact-version framework source showing the path is unsupported
- A minimal runtime repro on the target framework/GPU
- A proven invocation/key duplicate
- A documented absence from every production consumer and default model plan

Place runtime/SM/version exclusions as close as possible to the backend that
owns the limitation. Keep shared generators deterministic; avoid framework
imports or runtime availability probes in shared YAML population.

Pay special attention to boundaries:

- `EP>1` and `TP>1`, including combinations queried independently
- Per-rank dimensions after TP/EP sharding
- Global versus sliding-window attention
- Hopper versus Blackwell quantization support
- Native versus converted checkpoint artifacts
- Context versus generation phases

If the framework supports a boundary but the collector harness does not, file a
TODO/issue and keep the missing coverage visible. Do not describe it as pruned
unsupported input.

## Step 6: Validate in layers

### Static population

- Generate full/raw and representative targeted plans through each changed
  op's registry/public getter; testing only a shared case generator is not
  sufficient.
- Count generator recipes, raw getter tasks, post-selector scheduled tasks,
  token-expanded benchmark invocations, and unique persisted keys separately.
- For every deduplication, record stage-local before/after, unique invocation,
  and unique persisted-key counts using repository-owned inputs. Name the
  stage being counted (generator recipes, raw getter tasks, post-selector tasks,
  or token-expanded invocations). If the count at the dedupe stage does not
  decrease, remove the deduplication path.
- When a runtime or subprocess expands an inner sweep, compare its quantization
  and SM policy with the outer getter. Test both sides of every changed hardware
  gate and assert that unsupported precision labels are absent.
- Apply model, SM, and framework-version selectors before reporting a count as
  the final scheduled queue; raw getter counts are not final plan counts.
- Assert invocation IDs and persisted keys have no unexplained duplicates.
- Assert required consumer query keys are covered.
- Assert unrelated model dimensions never cross.
- Assert targeted plans do not inherit unrelated defaults or ops.

### Unit checks

Prefer behavior tests over copied migration inventories or AST tests of which
helper a function calls. Keep focused tests for:

- Structural correlation and boundary TP/EP values
- Quant/artifact policy
- Alias versus path-sensitive behavior
- Stable deduplication on the real invocation/key identity
- Registry/version routing and plan selection
- Persisted key names and local/global dimension semantics

Run at minimum:

```bash
.venv/bin/pytest -q tests/unit/collector
.venv/bin/ruff check collector tests/unit/collector
.venv/bin/ruff format --check collector tests/unit/collector
git diff --check
```

### Runtime smoke

Run inside the exact target framework image and record the installed package
version. Include representative cases for every material branch, not merely the
smallest cases:

- Each quantization/kernel path
- EP1 and EP>1 when supported
- Low and high TP
- Global and each supported window family
- Context and generation
- At least one model/artifact per path-sensitive branch

Then verify the produced rows can be loaded and queried by the unchanged AIC
consumer. A successful kernel timing alone is insufficient.

## Definition of done

Do not call the op complete until the handoff states:

- Exact framework version, GPU/SM, backend, and `__compat__` route
- Consumer lookup contract and fallback behavior
- Case counts by backend/op: kept, added, removed, deduplicated, skipped
- Evidence for every prune/skip
- Unit/lint/runtime-smoke results
- A real consumer query or support-matrix sanity check
- Remaining gaps, especially required EP/TP/quant/window buckets
- Whether the change touched Collector only; if not, why broader scope was
  explicitly authorized
- A final changed-file and changed-symbol audit that accounts for every
  production Python change and finds no orphan schema/helper, no no-op
  deduplication, and no unrelated orchestration behavior
