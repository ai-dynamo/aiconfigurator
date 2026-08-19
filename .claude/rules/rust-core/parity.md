---
description: >
  Regression discipline for the compiled engine: the frozen golden fixtures
  are the engine-step spec, deliberate modeling changes carry their golden
  diff, and per-op performance values are computed ONLY in the Rust engine
  (single oracle — #1357 Phase 3).
paths:
  - "aic-core/rust/aiconfigurator-core/**"
  - "aic-core/src/aiconfigurator_core/sdk/operations/**"
  - "aic-core/src/aiconfigurator_core/sdk/perf_database.py"
  - "aic-core/src/aiconfigurator_core/sdk/engine.py"
  - "aic-core/src/aiconfigurator_core/sdk/rust_engine_step.py"
  - "tests/unit/sdk/test_opspec_coverage.py"
  - "tests/cross_package/test_single_oracle_contract.py"
  - "tests/cross_package/test_query_shim_baseline.py"
---

# Rust-Core Regression Discipline (single-oracle era)

The compiled engine (`aic-core/rust/aiconfigurator-core`) is the ONLY
engine-step executor AND the only per-op performance oracle — op-level and
FPM models alike. The Python engine-step path and the Python per-call query
math are gone; final answers are frozen in `parity_tests/goldens/` (and,
for the per-run synthetic FPM fixture, inline `_FPM_*_FROZEN` tables) and
the parity suites assert live-Rust-vs-frozen.

## Rule 1 — engine-step changes carry their golden diff

Any PR that changes what the compiled engine computes (operators, loaders,
interpolation, selection rules, engine composition) MUST in the same PR:

1. Keep `test_engine_step_parity.py` / `test_compile_engine_parity.py`
   green — either the change is answer-preserving (goldens untouched), or it
   is a deliberate modeling change and the PR refreshes the affected records
   with `parity_tests/pin_goldens.py --refresh <keys>` (or `--refresh-all`)
   and lets the GOLDEN DIFF carry the review: reviewers see exactly which
   numbers moved and by how much. Never refresh to silence an unexplained
   failure.
2. Anchor new behavior: an oracle test in the Rust `#[cfg(test)]` module
   (hand-derived or generated from the modeling spec — there is no live
   Python reference to generate against), and/or a parity case pinned via
   `pin_goldens.py` (append-only mode) when a new config class becomes
   reachable.
3. Keep the `rust-engine-step-parity` CI job green — it is the enforcement
   mechanism, not this document.

## Rule 2 — the single-oracle invariant (per-op values live in Rust ONLY)

Per-op performance VALUES — latency, energy, the SOL decomposition — are
computed only by the compiled engine. Python owns model/topology
composition, data loading, orchestration, and presentation; it never owns
estimation math. Concretely:

- Do NOT add Python-side interpolation, roofline/SOL formulas,
  empirical-utilization estimates, or per-call table lookups anywhere under
  `aic-core/src/aiconfigurator_core/sdk/` (banned def shapes: the
  `_query_*` and `_lookup_*` prefixes, `get_sol`, `get_empirical`). The
  correct home is the Rust operator/table layer plus, if needed, a new
  engine FFI.
- `PerfDatabase.query_*` and `Operation.query` are a FROZEN set of
  deprecated engine-routed shims (plus explicit tombstones that raise),
  removed together in the deprecation-cleanup PR. New per-op access goes
  through `EngineHandle.evaluate_ops_json` / `evaluate_ops_sol_json`, the
  per-phase surface (`run_static_per_op`), or whole runs. `Operation.query`
  overrides are limited to the orchestration whitelist (AFD comm ops, the
  deprecated `Mamba2` composite) whose bodies compose ENGINE-evaluated twin
  ops.
- The deliberate-edit gates are
  `tests/cross_package/test_single_oracle_contract.py` (frozen surfaces,
  banned def names, whitelists) and
  `tests/cross_package/test_query_shim_baseline.py` (the merge-base-captured
  legacy baseline + frozen legacy-surface manifest). A PR that must grow a
  whitelist there needs an explicit justification in its description.
- Name-based guards cannot catch a determined rename; treat any Python code
  that turns shapes+tables into latency as a violation regardless of its
  name.

## Adding a new Operation

A new `Operation` subclass must get a `_to_opspec` branch in
`sdk/engine.py`, an `Op` variant in `operators/op.rs` (**append at the tail**
— bincode variant indices are positional; mid-enum insertion requires an
`ENGINE_SPEC_SCHEMA_VERSION` bump on BOTH sides), the `engine/spec.rs`
round-trip fixture, and a pinned parity case (`pin_goldens.py`).
`tests/unit/sdk/test_opspec_coverage.py` fails until the op converts or
carries a justified `EXEMPT` entry — an unconvertible op in a shipped
model is a HARD ERROR at estimation time, not a silent fallback. If the op
should be reachable through the deprecation-window shims, give it an
`_ENGINE_QUERY_SHAPE` (and a `_engine_query_plan` hook when its legacy
kwargs need mapping) — never a Python `query` body.

## Selection rules are regression surface too

Table/slice/kernel selection changes move golden numbers exactly like
formula changes; the same golden-diff rule applies. Python dicts iterate in
file/insertion order; Rust `BTreeMap` iterates sorted — any "first
available" fallback needs a load-order record (see `quants_in_load_order` /
`first_distribution` in `perf_database/{moe,moe_expert_compute}.rs`).

## Known intentional splits (do not "fix" without the tracking issue)

- AFD and the VL encoder phase are Python-side ORCHESTRATION; their per-op
  values cross the engine FFIs (the AFD comm ops evaluate standard
  P2P/NCCL/ElementWise twins through the single-op plumbing).
- `SOL_FULL` is a per-call diagnostic (never a selectable default mode); it
  is served by `evaluate_ops_sol_json` and raises for op families whose SOL
  path does not export its decomposition.
- Estimate-only systems (a spec yaml with no collected data) load under the
  SOL view only (`load_with_sources_opts` tolerance); every other mode keeps
  the loud missing-directory gate.
- The Python-loaded tables are the RAW collected data plane (enumeration,
  charts, support matrix) — no load-time SOL clamp or grid pre-expansion;
  the engine clamps and interpolates its own load.
- Rust reads parquet only (no `.txt` legacy loading) — new data drops must
  ship parquet.
