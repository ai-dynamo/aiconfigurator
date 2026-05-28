# AIC Rust Migration Job Definition

Here is the project plan. Don’t assume. Don’t hide confusion. Surface tradeoffs. Ask me for anything that needs clarity.

## Goal

Migrate AIC Python SDK engine-step latency logic to Rust, with the ultimate goal of speeding up each engine-step API call.

The top priority is low per-call latency. At the same time, the Rust code should be clean, readable, modular, and extensible as long as that does not sacrifice performance. Prefer the minimum coherent implementation that solves the problem. Do not add speculative abstractions or future features.

## Current Iteration Goal

Create the migration map, parity smoke tests, and benchmark harness needed to safely migrate AIC Python SDK engine-step latency logic into Rust.

Do not implement the full Rust migration until the migration map, parity harness, and benchmark harness are in place.

## Context

There is a POC in `./rust`. It partially works, but it oversimplifies AIC’s architecture. The goal is not to polish the POC as-is. The goal is to build a Rust crate that matches the Python SDK’s observable behavior while improving per-engine-step call latency.

I have coarser-grained tickets here:

https://linear.app/nvidia/project/aic-refactor-reuse-rust-engine-step-latency-api-aa6fc06a9e9d/issues

Feel free to propose sub-issues or new tickets if needed, but do not create them without summarizing why they are needed.

## Relevant Architecture

Current flow:

```
Python frontend -> Python SDK -> perf DB CSV files
```

Target flow:

```
Python frontend -> Rust SDK/core through ABI/bindings -> perf DB CSV files
```

Ownership boundaries:

- AIC core engine-step latency logic should move to Rust.
- AIC CLI, collectors, config generators, Pareto analysis, and other orchestration/UI layers should remain in Python.
- Primary implementation changes should be in `./rust`.
- Python changes are allowed only for tests, thin bindings, or existing Rust integration points such as `src/aiconfigurator/sdk/rust_engine_step.py`.
- Do not refactor unrelated Python SDK internals.

Design guidance:

- Use the Python SDK as the behavior reference, not necessarily the implementation template.
- Preserve the Python SDK’s modular concepts such as `operators/`, `models/`, and `backends/`.
- The Python SDK is still undergoing refactors. Do not blindly translate deprecated or redundant Python code into Rust.
- For example, `perf_database.py` has known redundancy as described in AIC-533:
https://linear.app/nvidia/issue/AIC-533/phase-45-remove-deprecated-perfdatabasequery-methods-update-test
- Also do not copy Python's eager perf DB startup shape. `PerfDatabase.__init__`
  still eagerly calls many op-family loaders as a transition compromise while
  tests migrate to lazy loading. Rust should load only metadata at
  database/session construction, then load each perf-file family lazily on first
  use or explicitly prewarm only the op families required by the current
  model/backend slice.
- Keep one table owner per op family. Do not duplicate deprecated Python
  `PerfDatabase.query_*` compatibility wrappers as Rust `PerfDatabase` methods
  when the behavior belongs in an operator/table module. Preserve parity with
  operator/query-boundary tests before removing or deduplicating Python-era
  paths.
- If Python behavior appears deprecated, redundant, buggy, or unclear, document the issue before deciding whether Rust should match it or intentionally diverge.

Core rule:

```
Match Python SDK observable behavior, not its internal structure.
```

## Tasks

### Phase 0: Migration Map

Status: complete for the current iteration.

Create a migration map before implementing the full Rust migration.

Deliverables:

1. Map relevant Python SDK modules/files to proposed Rust modules/paths.
2. Identify public behaviors the Rust implementation must match.
3. Identify deprecated or redundant Python logic that should not be copied directly.
4. Propose the Rust module design.
5. Create a system diagram showing current flow and target flow.
6. List open questions, architecture tradeoffs, and stop conditions.

### Phase 1: Parity Smoke Harness

Status: complete for the current iteration. The smoke tests live under
`rust/aiconfigurator-core/parity_tests/` and are expected to xfail until the
Rust op graph is complete.

Add pytest parity smoke tests comparing the existing Python SDK against the current Rust integration.

The tests should use:

```
src/aiconfigurator/sdk/rust_engine_step.py
```

for the Rust implementation path, and compare it against the Python SDK for reported metrics.

Initial smoke coverage:

- MiniMaxAI/MiniMax-M2.5
- Kimi-K2.5
- vLLM 0.19.0
- Sampled forward-pass parameters

The current Rust implementation is expected to be wrong or incomplete in some cases. These tests may be marked as `xfail` or otherwise made non-blocking for now, but failures must be explicit and explained.

Do not let expected current failures hide unrelated regressions.

Example parity case:

```
MiniMaxAI/MiniMax-M2.5
System: b200_sxm
Backend: vLLM 0.19.0
Forward pass:
- ISL = 1024
- OSL = 2
- prefix = 0
```

Python AIC:

```
run_static(mode="static_ctx") = 41.879 ms
run_static(mode="static_gen") = 5.808 ms
total = 47.687 ms
```

Rust AIC FPM:

```
ForwardPassMetrics:
- prefill_reqs = 1
- prefill_tokens = 1024
- prefill_kv = 0
- decode_reqs = 1
- decode_kv = 1024

forward_pass_time_ms = 30.050 ms
```

End-state parity target:

```
Rust and Python metrics differ by <1% on the agreed parity suite.
```

### Phase 2: Benchmark Harness

Status: complete for the current iteration. The benchmark harness lives at
`rust/aiconfigurator-core/parity_tests/benchmark_engine_step.py`.

Add or reuse a script to benchmark forward-step latency for the Python SDK and Rust core.

The benchmark should produce numbers that show actual speed improvement and can be reused to track Rust core performance over time.

Benchmark requirements:

1. Measure Python SDK per-step latency.
2. Measure Rust core per-step latency.
3. Separate cold-start/setup cost from warm hot-path per-call latency where possible.
4. Use reproducible input parameters.
5. Report p50/p90/p99 per-call latency if practical.
6. Report speedup ratio.
7. Include enough detail that another engineer can rerun the benchmark.

End-state performance target:

```
Rust hot-path engine-step calls are at least 3x faster than the Python SDK hot path.
```

### Phase 3: Rust Implementation

Status: pending. This is the next implementation phase.

Only start full implementation after Phase 0-2 are in place.

Implementation guidance:

1. Work module by module according to the migration map.
2. Match Python SDK observable behavior.
3. Avoid copying deprecated or redundant Python internals.
4. Keep Rust code modular and readable without sacrificing hot-path performance.
5. Keep perf DB loading and expensive setup out of the per-step hot path where possible.
6. Run parity tests and benchmark checks as the implementation progresses.

### Phase 4: Comprehensive Parity Scan

Status: pending until the Rust implementation is feature-complete.

After the Rust implementation is feature-complete, run a full parity scan across
the AIC-supported search space. The Phase 1 smoke tests cover only two models and
are not sufficient final coverage.

Comprehensive scan requirements:

1. Enumerate every model/system/backend/version/configuration entry that AIC
   advertises as supported.
2. Compare Python SDK and Rust core latency outputs through stable public
   interfaces.
3. Cover static, mixed-step, agg, and disagg paths where applicable.
4. Run the scan in parallel or shards if needed; the requirement is full
   coverage, not single-process execution.
5. Report every mismatch with enough case metadata to reproduce it.
6. Treat <1% latency drift as the final parity target unless a documented and
   approved exception exists.
7. Keep the comprehensive scan separate from the lightweight smoke suite so
   normal development feedback remains fast.

## Dependencies / Relationships

- Phase 0 defines the migration/design map.
- Phase 1 provides correctness guardrails.
- Phase 2 provides performance guardrails.
- Phase 3 uses the migration map, parity tests, and benchmark harness to implement the Rust migration safely.
- Phase 4 validates full parity across the supported AIC search space before the
  Rust core becomes the source of truth.

Do not treat the phases as isolated checklist items. If work in one phase reveals missing requirements in another, update the plan and surface the change.

## Constraints / Non-goals

- Do not remove the Python SDK in this project. Python SDK deprecation/removal belongs to a separate project.
- Do not move AIC CLI, collectors, config generators, or Pareto analysis into Rust.
- Do not refactor unrelated Python SDK internals.
- Do not blindly port deprecated Python code.
- Do not add speculative Rust abstractions or future features.
- Do not optimize for cleanliness at the cost of hot-path latency.
- Do not optimize for latency in a way that makes the code impossible to understand or maintain.
- Existing Python tests should continue to pass.
- The Rust implementation should become the source of truth for AIC core engine-step latency logic, while Python remains the orchestration and user-facing layer.

## Acceptance Criteria for Current Iteration

The current iteration is complete when:

1. A migration map exists and covers the relevant Python SDK files/modules.
2. A proposed Rust module structure is documented.
3. A system diagram shows current flow and target flow.
4. Parity smoke tests exist for selected model/system/backend/input cases.
5. Known current Rust mismatches are marked as `xfail` or reported explicitly with reasons.
6. A benchmark script exists for Python SDK vs Rust per-step latency.
7. Benchmark output includes reproducible parameters and speedup numbers.
8. Current decisions, remaining open questions, tradeoffs, and stop conditions are documented.
9. Full Rust migration work does not proceed until the map, parity harness, and benchmark harness are ready.

## Final Project Acceptance Criteria

The full project is successful when:

1. Rust and Python SDK metrics differ by <1% on the agreed parity suite.
2. Rust hot-path engine-step calls are at least 3x faster than the Python SDK hot path.
3. Existing Python tests pass.
4. Python CLI, collectors, config generators, and Pareto analysis remain Python-owned.
5. Rust owns the core engine-step latency implementation.
6. A comprehensive parity scan has run at least once across the supported AIC search space.
7. Any intentional divergence from Python behavior is documented and approved.

## Test Strategy

Before implementing Rust logic, derive a test matrix that includes:

1. Happy paths.
2. Boundary cases.
3. Invalid inputs.
4. Unsupported model/system/backend/version tuples.
5. Deprecated or ambiguous Python behavior.
6. Perf DB lookup and missing-data cases.
7. Cold-start vs warm hot-path performance cases.
8. Regression cases for existing Python behavior.
9. A comprehensive, shardable full-scan suite for all AIC-supported entries,
   run after feature completeness rather than as the fast smoke loop.

Prefer tests through public interfaces over tests of private implementation details.

If a test case depends on an ambiguous API or product decision, list it as an open question instead of guessing.

## Stop Conditions

Stop and ask before proceeding if:

1. Python behavior and desired Rust behavior conflict.
2. A required Python integration change violates the `./rust` primary implementation boundary.
3. A parity case depends on deprecated or redundant Python behavior.
4. The benchmark result depends heavily on cold startup or perf DB loading rather than hot-path calls.
5. The existing POC architecture conflicts with the target modular design.
6. A required model/system/backend/version tuple is missing from the perf DB.
7. The test harness cannot compare Python and Rust metrics through stable public interfaces.
8. Achieving <1% parity appears incompatible with the intended Rust architecture.
9. Achieving 3x speedup appears incompatible with parity or readability.
10. A task requires changing unrelated Python SDK internals.

## Expected Final Summary

At the end, summarize:

1. What changed.
2. What migration map was created.
3. What parity tests were added and which are expected to fail for now.
4. What benchmark script was added or reused.
5. What checks were run.
6. What remains ambiguous.
7. What tradeoffs were surfaced.
8. What the next implementation step should be.
