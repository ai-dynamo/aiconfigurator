<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# PROPOSAL: rewrite `.claude/rules/rust-core/parity.md` for the golden-diff era

Rule files are human-owned policy (repo-guide rule 2), so this PR does not
edit `.claude/rules/rust-core/parity.md`; it proposes the replacement text
below. Motivation: with the Python engine-step path deleted (dedup-plan
Gate 3), the current rule is wrong in three ways —

1. It declares "the frozen Python op/query math is the spec" and mandates
   mirroring into it for engine-step changes; the engine-step reference is
   now the FROZEN GOLDEN FIXTURES, not live Python code.
2. Its "known intentional splits" are stale: SOL was ported (PR-2 #1496),
   energy crosses the FFI (PR-2), the op-list evaluation FFI landed, and
   FPM runs on the compiled engine (#1461 + the walk deletion in this PR).
3. It does not describe the post-freeze golden workflow (`pin_goldens.py`,
   provenance-marked pins, golden-diff review).
4. Its `paths:` predate the aic-core restructure: they name the
   `src/aiconfigurator/sdk/**` compatibility shims instead of the real
   sources under `aic-core/src/aiconfigurator_core/sdk/**` (and the crate
   moved under `aic-core/rust/`). The proposal below uses the
   repository-root-relative real locations.

One thing the rule must KEEP: the per-call Python query stack
(`operations/*.py` `_query_*_table` bodies, `perf_database.query_*`,
`perf_interp/`) is still live — the AFD comm ops, the sanity notebook, and
the op-internal SOL/empirical couplings (ops whose silicon query falls back
into their own SOL math) consume it — so per-op query math remains
DUAL-implemented until the #1357 Phase-3 sequel deletes it
family-by-family. The dual rule therefore
survives for the per-op surface, while the step-level rule becomes
golden-diff.

---

## Proposed replacement text for `.claude/rules/rust-core/parity.md`

```markdown
---
description: >
  Regression discipline for the compiled engine: the frozen golden fixtures
  are the engine-step spec; deliberate modeling changes carry their golden
  diff. Per-op query math is still dual-implemented (AFD/tooling) until
  the #1357 Phase-3 retirement.
paths:
  - "aic-core/rust/aiconfigurator-core/**"
  - "aic-core/src/aiconfigurator_core/sdk/operations/**"
  - "aic-core/src/aiconfigurator_core/sdk/perf_database.py"
  - "aic-core/src/aiconfigurator_core/sdk/perf_interp/**"
  - "aic-core/src/aiconfigurator_core/sdk/engine.py"
  - "aic-core/src/aiconfigurator_core/sdk/rust_engine_step.py"
  - "tests/unit/sdk/test_opspec_coverage.py"
---

# Rust-Core Regression Discipline (golden-diff era)

The compiled engine (`rust/aiconfigurator-core`) is the ONLY engine-step
executor — op-level and FPM models alike. The Python engine-step path is
gone; its final answers are frozen in `parity_tests/goldens/` (and, for the
per-run synthetic FPM fixture, inline `_FPM_*_FROZEN` tables) and the
parity suites assert live-Rust-vs-frozen.

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

## Rule 2 — the per-call Python query stack is still dual (for now)

`aic-core/src/aiconfigurator_core/sdk/operations/**` `_query_*_table`
bodies, `perf_database.query_*`, and `perf_interp/**` remain live
consumers' substrate: the AFD comm ops (empirical nccl/p2p/mem_op queries
plus the `_sum_latency` fallback loop),
`tools/sanity_check/validate_database.ipynb`, and the op-internal
SOL/empirical couplings (ops whose silicon queries fall back into their own
SOL math). Until the #1357 Phase-3
retirement lands, a latency-affecting change to that per-call math MUST
still be mirrored in the corresponding Rust operator (and vice versa), with
an anchor per Rule 1 — otherwise AFD/tooling silently diverge from the
engine step.

## Adding a new Operation

A new `Operation` subclass must get a `_to_opspec` branch in
`sdk/engine.py`, an `Op` variant in `operators/op.rs` (**append at the tail**
— bincode variant indices are positional; mid-enum insertion requires an
`ENGINE_SPEC_SCHEMA_VERSION` bump on BOTH sides), the `engine/spec.rs`
round-trip fixture, and a pinned parity case (`pin_goldens.py`).
`tests/unit/sdk/test_opspec_coverage.py` fails until the op converts or
carries a justified `EXEMPT` entry — an unconvertible op in a shipped
model is now a HARD ERROR at estimation time, not a silent fallback.

## Selection rules are regression surface too

Table/slice/kernel selection changes move golden numbers exactly like
formula changes; the same golden-diff rule applies. Python dicts iterate in
file/insertion order; Rust `BTreeMap` iterates sorted — any "first
available" fallback needs a load-order record (see `quants_in_load_order` /
`first_distribution` in `perf_database/{moe,wideep_moe}.rs`).

## Known intentional splits (do not "fix" without the tracking issue)

- AFD and the VL encoder phase are Python-side orchestration; their per-op
  values cross the op-list evaluation FFI (the AFD synthetic comm ops stay
  on the per-call stack).
- SOL_FULL stays a Python-side PER-CALL diagnostic (the sanity notebook's
  raw-tuple contract); it is not a selectable default mode on either engine.
- Rust reads parquet only (no `.txt` legacy loading) — new data drops must
  ship parquet.
```
