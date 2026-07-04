# Collector Layer Permissions

Which layer of the collector may hold which kind of rule. This is the single
most important reference when "fixing" a failing case: pick the layer first,
then act. Violating these boundaries is how the retired sm_exceptions rule
engine grew to 1400 lines of shape-sniping rules.

## The one-line architecture

```text
plan cases = dedup( expand(base_ops sweep grids) ∪ expand(model_case_values shapes) )
runnable   = plan cases ∩ hardware capability floors  −  hang denylist
result     = perf rows  +  classified failure records   (failure is DATA, not a defect)
```

## Permission table

| Layer | Allowed | Forbidden |
|---|---|---|
| `cases/base_ops/*.yaml` | sweep density/ranges; axis-level `min_sm` on quant/precision modes | per-model special cases; shape exclusion rules |
| `cases/models/*_cases.yaml` | model structural shapes, correlated tuples, artifact/quant policy, op activation (`cases: all`) | selectors, match rules, a second generator recipe |
| `cases/capabilities.yaml` | positive dtype/op → min-SM floors (hardware facts only) | negative "drop" rules; shape axes; framework-version conditions; per-backend nesting |
| `<backend>/registry.py` | version routing (`VersionRoute`); maturity markers `unverified=True` (op × backend) and `unverified_sms=(...)` (op × backend × SM) | any shape-level information |
| `collect_*.py` collector code | **dispatch**: pick kernel path by SM/version, record `kernel_source`; runtime probe → **raise a classified exception**; **memory-feasibility filter** (see below) | silent `continue`/skip of a queued case; any other case filtering; patching code so one shape passes |
| `cases/denylist.yaml` | cases that HANG or kill the node (dated, with reason) | ordinary crashes — those belong to the failure log |
| executor (`collect.py`) | — mechanism is off-limits during case-fixing tasks | changing checkpoint/case-ID/failure-classification logic to make a case pass |
| failure records (`errors_*.json`, summary) | append-only observation | feeding failures back into ANY declaration layer as new rules |

## The dispatch/skip discriminator

> A legal branch changes HOW a case runs. An illegal branch changes WHETHER it runs.

- Legal: `if sm >= 100: use kernel A else kernel B` — both sides produce a data
  point (and record which kernel).
- Illegal: `if sm < 90: continue` — the case vanishes with no data and no
  failure record.

A collector has exactly two legal responses to a queued case: **execute it, or
raise**. "Whether it runs" is decided only outside the collector, in three
auditable places: capability floors (generation time), registry `unverified`
markers, and the hang denylist.

## The one sanctioned in-collector filter: memory feasibility

Device memory is a permanent hardware fact, but its expression needs
op-specific arithmetic (KV cache, weights, activation peaks) that only the
collector knows. Therefore a collector MAY filter cases on memory
feasibility, under exactly these conditions:

1. **Generation time only** — inside `get_*_test_cases()`, so the case is
   never queued. Never a runtime `continue`.
2. **Predicate = size vs capacity, nothing else** — preferred:
   `estimated_footprint(shape) > device_total_memory * safety_factor` with
   memory queried live (`torch.cuda.get_device_properties`); acceptable
   fallback: a shape metric against a per-SM/per-capacity constant. NEVER
   framework-version or model-name conditions — that is exception smuggling.
3. **Drops are counted, never silent** — log one line:
   `<op>: dropped N/M cases (memory budget, device=<GB>)`.

Anything beyond this pattern goes back to the normal rule: run it or raise.

## Parking framework kernel limits: `FIXME(kernel-limit)`

Framework kernel limits (backend × SM constraints with shape flavor, e.g.
"Blackwell attention rejects GQA ratio >= 32 unless divisible by 32") are
neither hardware facts (not capabilities.yaml material) nor version bugs
(too stable to ignore). Their home is a **`FIXME(kernel-limit)` comment at
the invocation site in the owning collector**, stating the claimed limit,
its origin, and that it is unverified. The affected cases simply fail at
runtime meanwhile.

Lifecycle: on the next framework version bump, the upgrade audit greps
`FIXME(kernel-limit)`, verifies each claim against the framework source, and
either implements it as a probe (ask the framework's own selector — never a
prediction) or a guard that raises a classified error citing the framework
source line, or deletes the note. Do not implement guards from unverified
claims, and never express these limits in YAML.

## Module boundary: a collector task may only touch collector/

The collector's outward surface is its **data contract**: the perf row schema
(columns written by `helper.log_perf`), the canonical filenames
(`registry_types.PerfFile`), and the parquet finalization format. Everything
downstream — `src/aiconfigurator/sdk/perf_database.py`, the Rust
`aiconfigurator-core` operators, `tools/support_matrix/`, packaged data under
`src/aiconfigurator/systems/data/` — consumes that contract.

Rules:

1. A collector task may modify `collector/` and `tests/unit/collector/`.
   **Nothing else.** Not the SDK, not Rust, not tools, not systems data, not
   the generator — even when the collector change "obviously needs" a
   consumer-side change to be useful.
2. **Changing the data contract requires explicit human approval.** Adding a
   column, a new perf file, a new key dimension, or renaming anything the SDK
   parses is a producer+consumer coordinated change. If you discover the need
   mid-task: STOP, write up the proposal (new dimension, why, which consumers
   must change), and hand it to the human. Do not implement both sides
   "while you're at it".
3. The reverse holds too: SDK/modeling tasks do not reach into `collector/`.
4. These rule files themselves (`.claude/rules/collector/`) are human-owned
   policy. Do not edit them as a side effect of a fix task; propose changes
   instead.

## Meta rules

1. **Default action for a failing case is NO CHANGE.** Confirm it is recorded
   and classified in the failure log, then stop. Escalate only per the decision
   tree in `failure_handling.md`.
2. **Mechanism changes need explicit human approval.** The executor, case
   generator engine, failure classification, and case-ID format are mechanism.
   Propose changes; do not fold them into a "fix this case" task.
3. **Inexpressibility is the guardrail.** The YAML schemas here deliberately
   cannot express shape conditions or version predicates. Do not extend the
   schemas to allow them.
4. **A test that only works with `-m unit` unmarked is invisible to CI.** Mark
   new collector tests `pytest.mark.unit`.
