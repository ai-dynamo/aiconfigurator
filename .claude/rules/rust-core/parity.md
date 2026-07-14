---
description: >
  Parity discipline for the compiled engine: the frozen Python op/query math
  is the spec; latency-affecting changes must land on both sides with an
  oracle anchor in the same PR.
paths:
  - "rust/aiconfigurator-core/**"
  - "src/aiconfigurator/sdk/operations/**"
  - "src/aiconfigurator/sdk/perf_database.py"
  - "src/aiconfigurator/sdk/perf_interp/**"
  - "src/aiconfigurator/sdk/engine.py"
  - "src/aiconfigurator/sdk/rust_engine_step.py"
  - "tests/unit/sdk/test_opspec_coverage.py"
---

# Rust-Core Parity Discipline (Python-path freeze)

The compiled engine (`rust/aiconfigurator-core`) is at full numeric parity
with the Python engine step for the SILICON, HYBRID, and EMPIRICAL database
modes, guarded by `rust/aiconfigurator-core/parity_tests/` (engine-step,
compile-engine, and perf gates). The Python op/query math is **frozen**: it
is the reference the parity suite compares against, and it is scheduled for
retirement (see the Python-path freeze tracking issue).

## The rule

Any PR that changes latency-affecting behavior in:

- `src/aiconfigurator/sdk/operations/**` (op query math, SOL formulas,
  slice/kernel selection, transfer ladder),
- the query/loader layer of `src/aiconfigurator/sdk/perf_database.py`,
- `src/aiconfigurator/sdk/perf_interp/**`,

MUST in the same PR:

1. **Mirror the change** in the corresponding `rust/aiconfigurator-core/src/`
   operator/table (the layering matches: dispatch + estimators in
   `operators/`, algorithm-free loaders/accessors in `perf_database/`).
2. **Anchor it**: a Python-generated oracle in the Rust `#[cfg(test)]` module
   (1e-9, `uv run python` against a `shared_layer=False` view — copy the
   existing oracle-test pattern in `operators/gemm.rs`), and/or a case in
   `parity_tests/test_engine_step_parity.py` when a new config class becomes
   reachable.
3. Keep the `rust-engine-step-parity` CI job green — it is the enforcement
   mechanism, not this document.

## Adding a new Operation

A new `Operation` subclass must get a `_to_opspec` branch in
`sdk/engine.py`, an `Op` variant in `operators/op.rs` (**append at the tail**
— bincode variant indices are positional; mid-enum insertion requires an
`ENGINE_SPEC_SCHEMA_VERSION` bump on BOTH sides), the `engine/spec.rs`
round-trip fixture, and a parity case. `tests/unit/sdk/test_opspec_coverage.py`
fails until the op converts or carries a justified `EXEMPT` entry.

## Selection rules are parity surface too

Table/slice/kernel selection must match rule-for-rule, including fallback
order and tie-breaks. Python dicts iterate in file/insertion order; Rust
`BTreeMap` iterates sorted — any "first available" fallback needs a
load-order record on the Rust side (see `quants_in_load_order` /
`first_distribution` in `perf_database/{moe,wideep_moe}.rs` for the pattern).

## Known intentional splits (do not "fix" without the tracking issue)

- SOL / SOL_FULL modes delegate to the Python step (routing gate in
  `sdk/rust_engine_step.py`).
- AFD and the VL encoder phase are Python-side orchestration; their per-op
  values move to Rust via the planned op-list evaluation FFI, not by porting
  the orchestration.
- Rust reads parquet only (no `.txt` legacy loading) — new data drops must
  ship parquet.
- Energy/power does not yet cross the FFI (rust-routed reports show 0.0W)
  until the energy follow-up PR lands.
