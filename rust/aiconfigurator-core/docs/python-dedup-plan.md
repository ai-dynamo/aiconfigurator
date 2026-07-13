<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Phase 2 Execution Plan — Rust-default flip and Python latency-path removal

**Status (2026-07-13): NOT STARTED — plan of record, not yet implemented.**
The switch is still opt-in: `engine_step_backend` defaults to Python
(`config.py`), `should_use_rust_engine_step` returns Python unless it is set to
`"rust"`, and the `operations/*.py` `query()` methods plus the
`perf_database.py` latency-query methods are all still present. None of the
gates below has run.

**New blocker since drafting:** the Rust engine-step is **SILICON-only** (no
`util_empirical` layer), so `should_use_rust_engine_step` delegates
HYBRID/EMPIRICAL databases back to Python for answer-parity. A *global* default
flip (Gate 1) therefore cannot ship until the empirical-layer port lands
(issue #1333); until then any flip is silicon-only or staged per-family.

**Done since drafting:** the full parity scan is complete
(`parity-scan-report.md`: gate CLOSED, 0 REGRESSION), and the
engine-step parity, compile-engine parity, and engine-step perf gates all run
in CI (`.github/workflows/build-test.yml`).

**Branch base:** `main` tip (post #1200 / #1201).

## Motivation

Phase 1.5 (#1200) made Python build the op list and Rust execute it, and
deleted the **Rust** model layer (`models/`, `backends/`, `factory.rs`, …).
#1201 added the capacity API. Both are merged.

But the symmetric duplication is still live, with the polarity flipped:
the **Python** latency-execution stack — `operations/*.py` `query()` bodies,
the `perf_database.py` latency-query methods, and the Python op-walk in
`backends/base_backend.py` — now mirrors the Rust `operators/` +
`perf_database/` + `engine/` code. Two engines compute the same step latency.

The Rust path exists but is **opt-in**: `should_use_rust_engine_step`
(`sdk/rust_engine_step.py:222`) returns `"python"` unless
`runtime_config.engine_step_backend == "rust"` or the env var is set. Every
default CLI / SDK / webapp run still walks ops in Python.

Phase 2 makes Rust the default execution path and removes the duplicated
Python latency code.

## Goal

> Rust is the only step-latency engine. Python keeps model construction,
> the OpSpec walk, the agg-sweep scheduler, and the memory model — and
> drops the `Operation.query()` latency math that Rust already owns.

After Phase 2:
- `engine_step_backend` defaults to `"rust"`. The `"python"` value is
  retained one release as an escape hatch, then removed.
- `operations/*.py` keeps `__init__` / `get_weights()` / attribute storage
  (load-bearing for `compile_engine` and the memory model); the `query()`
  methods and their query-time helpers are deleted.
- `perf_database.py` keeps data loaders, `system_spec`, and weight/metadata
  accessors; the latency-query methods (`query_gemm`, `query_*`) are deleted.
- `base_backend.py` keeps the agg-sweep scheduler, `_get_memory_usage`, and
  the Rust-routing branch; the Python step-latency branch
  (`_run_context_phase` / `_run_generation_phase`) is deleted.

## Why this is safe (primary-source evidence)

The deletable scope was verified, not assumed:

1. **`Operation.query()` has exactly one consumer — the Python op-walk.**
   A repo-wide `.query(` grep across `memory.py`, `inference_summary.py`,
   `vllm_backend.py` returns empty; the only callers are inside
   `base_backend._run_context_phase` / `_run_generation_phase`. Those are the
   branch Rust replaces.
2. **The memory model does not use `query()`.**
   `base_backend._get_memory_usage` (`:1344`) sums `op.get_weights()`
   (config-time) and reads `database.system_spec["misc"][…]` (DB metadata).
   No latency query. So `get_weights()` and the perf-DB metadata accessors
   must stay; the latency-query methods need not.
3. **`compile_engine` reads instance attributes, never `query()`.**
   The E0 OpSpec audit proved every Rust `Op`
   field maps to a build-time Python `Operation` attribute (`op._n`,
   `op._scale_factor`, …). `_to_opspec` walks attributes; deleting `query()`
   does not touch it.

## Keep / delete inventory

Every candidate is **partial** — none of these files is deleted whole.

| Module | LoC (file) | Delete | Keep | Notes |
| --- | --- | --- | --- | --- |
| `sdk/operations/*.py` | 11 952 | `query()` methods + query-time helpers (the latency math, dominant fraction) | `__init__`, attribute storage, `get_weights()`, `get_*` config accessors | `compile_engine` (`_to_opspec`) + memory model depend on the kept parts. |
| `sdk/backends/base_backend.py` | 1 429 | Python step-latency branch: `_run_context_phase`, `_run_generation_phase`, the `else` after `should_use_rust_engine_step`, Python mixed/decode-step math | agg-sweep scheduler (`run_agg`, `find_best_agg_result_under_constraints`), `_get_memory_usage`, Rust-routing branch, cache-key helpers | After the flip the Python branch is dead; the scheduler still drives and calls the Rust step helpers. |
| `sdk/perf_database.py` | 2 474 | latency-query methods (`query_gemm`/`query_attention`/`query_moe`/… — callers were `operations.query()`, now deleted) | parquet loaders, `system_spec`, weight/metadata accessors, support-matrix reads | Consumed by memory model, support matrix, `task.py`, `predict*` — those stay. |
| `sdk/interpolation.py` | 785 | nothing yet | all | Still used by `perf_database.py` (kept readers) and `webapp/.../profiling`. The `operations/*.py` callers go away, but it is not orphaned. Re-audit at delete time. |
| `sdk/inference_session.py` | 1 888 | nothing structural | all | `run_static`/`run_agg` are thin delegations to `base_backend`; the op-walk is not here. Stays as orchestration. |
| `sdk/rust_engine_step.py` | 479 | the `should_use_rust_engine_step` gate (once `"python"` is removed) | the Rust step-estimate helpers + handle cache | Becomes unconditional; the live bridge. |

Rough net: ~3–5 kLoC removed from Python, 0 added. (Measure the exact
`query()`-only fraction at implementation time via an AST pass; file totals
above are the upper bound.)

## Gates

Three hard gates, in order. Do not collapse them into one PR.

### Gate 1 — Default flip, scan- and DRIFT-gated

Flip `engine_step_backend` default to `"rust"`. Before merge:
- Full-matrix scan must hold the Phase 1.5 bar: `STRICT_PASS >= 1906`,
  `REGRESSION == 0`.
- The residual DRIFT entries (current list in the completed scan,
  `parity-scan-report.md`) were held out of Phase 1.5 scope.
  A *global* default flip silently ships them.
  Each must be either resolved or **formally accepted** (listed by family,
  with the >5% throughput delta documented) as a precondition. Decide and
  record: is the flip global, or staged per-family (the `rust_engine_step.py:382`
  comment hints some families already default to Rust)?
- The 164-surface smoke harness (`parity_tests/test_engine_step_parity.py`,
  `test_compile_engine_parity.py`) passes bit-identical-or-within-tolerance.

**No deletion in this PR.** Both engines stay; only the default changes.

### Gate 2 — Golden capture (replace the live differential oracle)

The parity tests compare **Python vs Rust live**. Deleting the Python path
destroys the regression detector future Rust changes rely on. Before any
deletion:
- Capture current Python `run_static` / `run_agg` / step-latency outputs as
  golden fixtures across the smoke matrix.
- Rewrite `test_engine_step_parity.py` / `test_compile_engine_parity.py` to
  assert **Rust vs golden** instead of **Rust vs live-Python**.
- Land the goldens + rewired tests as their own PR, green, before Gate 3.

### Gate 3 — Delete the duplicated Python latency code

Only after Gates 1–2 hold and have soaked one release cycle:
- Delete per the keep/delete table.
- Remove the `"python"` value of `engine_step_backend` (and the
  `should_use_rust_engine_step` gate); the CLI/SDK arg becomes deprecated
  no-op for one cycle, then dropped.
- Re-run goldens (Gate 2) + smoke harness; numbers unchanged.

## PR sequence

| # | PR | Lands | Gated? |
| --- | --- | --- | --- |
| **P0** | DRIFT triage | Resolve or formally accept the residual DRIFT entries; record the decision alongside `parity-scan-report.md`. No code beyond fixes. | — |
| **P1** | Default flip | `engine_step_backend` defaults to `"rust"`; `"python"` retained. Full scan + smoke green. Both engines present. | **GATE 1** |
| **P2** | Golden oracle | Capture Python goldens; rewire parity tests to Rust-vs-golden. | **GATE 2** |
| **P3** | Delete Python latency path | `operations/*.py` `query()`, `base_backend` Python branch, `perf_database` latency-query methods. Re-audit `interpolation.py`. | **GATE 3** |
| **P4** | Retire the switch | Remove `should_use_rust_engine_step` + the `"python"` arg value (deprecation cycle elapsed). | parity re-run |

P0 → P1 → P2 → P3 → P4, strictly sequential. P2 may start once P1 is in.

## Out of scope

- Re-running collectors or regenerating perf-DB data.
- Perf-DB schema or support-matrix CSV format changes.
- CLI / generator / Pareto / webapp behaviour changes (the flip is internal
  to `sdk/`; webapp keeps its `interpolation.py` use).
- The Dynamo-side `estimate_num_gpu_blocks` rewrite — completed downstream in
  the Dynamo repo; no longer tracked here.
- The `#1208` OOM-budget-sharing follow-up.

## Risks

| Risk | Mitigation |
| --- | --- |
| Default flip silently regresses a DRIFT family. | P0 gate: triage/accept the residual DRIFT (4 in the completed scan) before P1. |
| Deleting Python destroys the differential oracle. | Gate 2: capture goldens + rewire tests before any deletion. |
| `query()` deletion nicks a kept consumer (memory / OpSpec walk). | AST pass to confirm `query()` has no caller outside the deleted branch; `get_weights()` / attrs / loaders explicitly retained. |
| `interpolation.py` assumed dead but webapp/perf-DB still use it. | Marked "keep, re-audit"; not in P3's delete set without a fresh consumer grep. |
| `rust_engine_step` handle cache or rayon introduces non-determinism once it is the only path. | Smoke harness runs `RAYON_NUM_THREADS=1` and `=8`, asserts identical output (carried over from Phase 1.5 E5). |

## Acceptance criteria

1. **Default is Rust** with full scan `STRICT_PASS >= 1906`, `REGRESSION == 0`,
   and the residual DRIFT entries (4 in the completed scan) resolved or accepted.
2. **Goldens replace the live oracle**; parity tests are Rust-vs-golden and green.
3. **Python `query()` latency math removed**; `compile_engine` and
   `_get_memory_usage` unaffected (their kept dependencies proven).
4. **No CLI / generator / Pareto / webapp behaviour change.**
5. **LoC discipline:** net −3 to −5 kLoC on `sdk/`; `sdk/models/` unchanged.

## Pointers

- Completed parity scan (DRIFT list, gate status): `parity-scan-report.md`.
- Scan procedure: `parity-scan-runbook.md`.
- Engine-step / compile-engine / perf gates in CI: `.github/workflows/build-test.yml`.
- The opt-in switch this plan flips: `sdk/rust_engine_step.py`
  (`should_use_rust_engine_step`).
- Empirical-layer port that gates the global default flip: issue #1333.
- Architecture reference: `design_doc.html`.
