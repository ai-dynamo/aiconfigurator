<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Phase 2 Execution Plan — Rust-default flip and Python latency-path removal

**Status (2026-08-01): REVISED — consolidated to three PRs; PR-1 in flight.**
Since the 2026-07-13 draft, #1355 closed every SILICON parity divergence AND
ported the util-space empirical layer, so the compiled engine now answers
SILICON / HYBRID / EMPIRICAL itself (`rust_engine_step.py`,
`_RUST_SUPPORTED_DATABASE_MODES`). The old Gate-1 blocker (#1333) is gone and
the original P0 (DRIFT triage) is substantially done. The remaining gaps, and
which PR owns each:

- **Energy/power does not cross the FFI.** The static path already
  compensates (the rust branch of `_run_static_breakdown` runs the Python
  phase runners for energy only), but the agg mixed/decode step paths return
  `energy_wms=0.0` — an unconditional default flip would silently zero agg
  `power_w` on the systems that ship measured power columns (b200_sxm,
  h200_sxm, gb200). → PR-1 handles this by *routing*, not porting: default
  resolution delegates power-carrying databases to the Python step. The
  actual energy-across-FFI port (rust perf leaves deliberately carry no
  power today, `perf_interp.rs`) lands in PR-2 and removes that routing carve-out.
  Note the energy compensation also means the Python `query()` layer is a
  live dependency of rust-routed static runs — another reason deletion
  cannot precede the energy port.
- **AFD** is Python-only (no rust operators; `RustEngineUnsupportedError`
  → Python-step fallback). Harmless at flip time, blocks deletion. → PR-2.
- **SOL / SOL_FULL** modes route to the Python step by design. Blocks
  deletion, not the flip. → PR-2 (port or retire the modes).

## Revised PR sequence (2026-08-01) — supersedes the P0–P4 table below

| # | PR | Scope | Maps to |
| --- | --- | --- | --- |
| **PR-1** | Rust default | Flip default resolution to the compiled engine; power-carrying and non-`PerfDatabase` databases delegate to the Python step; explicit `"rust"`/`"python"` keep force semantics (`"python"` is the escape hatch, retained one release). Propagate `engine_step_backend` into the internal `RuntimeConfig` constructions (`run_mixed` passes 1–3, `_get_genonly_step_latency`) so the escape hatch binds the whole composition. Forced-rust full-matrix scan as merge evidence. No deletion. Bake one release cycle. | P0+P1 |
| **PR-2** | Golden anchor + gap closure | Capture Python `run_static`/`run_agg`/step goldens while Python is alive; rewire parity tests to Rust-vs-golden; energy-across-FFI (removes PR-1's power routing carve-out); resolve AFD (per-op values via the planned op-list evaluation FFI, per `.claude/rules/rust-core/parity.md`) and SOL/SOL_FULL (port or retire). The op-list FFI must also restore the **user-facing per-op breakdown**: today the Rust path collapses `InferenceSummary` per-op dicts (`get_per_ops_data()`, `get_context_latency_dict()`, the "Context breakdown" summary section) into a single synthetic key (`rust_engine_step_*`), and the per-op latencies computed inside `run_context_ops` are summed and discarded before crossing the FFI — AFD orchestration alone is not the only consumer. These goldens double as the EngineSpec IR anchor required by the model-builder plan. | P2 + gaps |
| **PR-3** | Delete + retire switch | Delete per the keep/delete inventory below; `"python"` value becomes a warning no-op (deprecation shim = folded P4); propose the `.claude/rules` dual-implementation → golden-diff rewrite (human-owned, proposal only). | P3+P4 |

Strictly sequential. PR-2 may start once PR-1 merges; PR-3 waits for PR-1 to
bake one release cycle. Rationale for 3 (not 2, not 5): goldens must merge
green from the live Python path *before* any deletion (a golden captured in
the deletion PR risks self-referential regeneration and reverts with it), and
the flip must bake separately from deletion so a post-flip drift still has a
one-env-var rollback; conversely P0's heavy lifting shipped in #1355 and P4 is
a one-line shim, so neither deserves a standalone PR anymore.

**Done since the original draft:** the 2026-07 opt-in-rust parity scan
completed (`parity-scan-report.md`: gate CLOSED, 0 REGRESSION — historical
evidence for the opt-in path, distinct from PR-1's own forced-rust scan,
which reruns against the flipped default as its merge evidence); engine-step
parity, compile-engine parity, and perf gates in CI; #1355 (SILICON audit +
HYBRID/EMPIRICAL port) merged.

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
default CLI / SDK run still walks ops in Python.

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
| `sdk/interpolation.py` | 785 | nothing yet | all | Still used by `perf_database.py` (kept readers). The `operations/*.py` callers go away, but it is not orphaned. Re-audit at delete time. |
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
- **Per-op breakdown precondition**: the Rust path must expose per-op
  latencies through the op-list evaluation FFI and populate the
  `InferenceSummary` per-op dicts with real op names (not the synthetic
  `rust_engine_step_*` keys) — or the loss of op-level analysis is formally
  accepted and documented here. Until then the Python step is the only way
  to obtain a per-op breakdown, and deleting it removes that capability.
- Delete per the keep/delete table.
- Remove the `"python"` value of `engine_step_backend` (and the
  `should_use_rust_engine_step` gate); the CLI/SDK arg becomes deprecated
  no-op for one cycle, then dropped.
- Re-run goldens (Gate 2) + smoke harness; numbers unchanged.

## PR sequence (SUPERSEDED — see "Revised PR sequence (2026-08-01)" above)

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
- CLI / generator / Pareto behaviour changes (the flip is internal to `sdk/`).
- The Dynamo-side `estimate_num_gpu_blocks` rewrite — completed downstream in
  the Dynamo repo; no longer tracked here.
- The `#1208` OOM-budget-sharing follow-up.

## Risks

| Risk | Mitigation |
| --- | --- |
| Default flip silently regresses a DRIFT family. | P0 gate: triage/accept the residual DRIFT (4 in the completed scan) before P1. |
| Deleting Python destroys the differential oracle. | Gate 2: capture goldens + rewire tests before any deletion. |
| `query()` deletion nicks a kept consumer (memory / OpSpec walk). | AST pass to confirm `query()` has no caller outside the deleted branch; `get_weights()` / attrs / loaders explicitly retained. |
| `interpolation.py` assumed dead but perf-DB still uses it. | Marked "keep, re-audit"; not in P3's delete set without a fresh consumer grep. |
| `rust_engine_step` handle cache or rayon introduces non-determinism once it is the only path. | Smoke harness runs `RAYON_NUM_THREADS=1` and `=8`, asserts identical output (carried over from Phase 1.5 E5). |
| PR-3 deletes the Python step while the Rust path still collapses the per-op breakdown to a synthetic key — SDK users permanently lose op-level latency analysis. | Gate 3 precondition: per-op values cross the op-list evaluation FFI and populate `InferenceSummary` before deletion, or the loss is formally accepted and recorded. |

## Acceptance criteria

1. **Default is Rust** with full scan `STRICT_PASS >= 1906`, `REGRESSION == 0`,
   and the residual DRIFT entries (4 in the completed scan) resolved or accepted.
2. **Goldens replace the live oracle**; parity tests are Rust-vs-golden and green.
3. **Python `query()` latency math removed**; `compile_engine` and
   `_get_memory_usage` unaffected (their kept dependencies proven).
4. **No CLI / generator / Pareto behaviour change.**
5. **LoC discipline:** net −3 to −5 kLoC on `sdk/`; `sdk/models/` unchanged.

## Pointers

- Completed parity scan (DRIFT list, gate status): `parity-scan-report.md`.
- Scan procedure: `parity-scan-runbook.md`.
- Engine-step / compile-engine / perf gates in CI: `.github/workflows/build-test.yml`.
- The opt-in switch this plan flips: `sdk/rust_engine_step.py`
  (`should_use_rust_engine_step`).
- Empirical-layer port that gates the global default flip: issue #1333.
- Architecture reference: `design_doc.html`.
