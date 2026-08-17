<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Phase 2 Execution Plan — Rust-default flip and Python latency-path removal

**Status (2026-08-14): PR-1 (#1454), PR-2 (#1496) and PR-2.5 (#1508) MERGED;
PR-3 (#1521 — the PR carrying this text) delivers the ENGINE-STEP-path
retirement and is in flight until it merges — see "PR-3 disposition"
below.** The original PR-3 scope ("delete the per-call query stack
wholesale") was revised after FPM (#1384) landed a live consumer of that
stack; #1461's Rust FPM port (Op::FpmForward + fpm_sol.rs) then let PR-3
delete the Python FPM walk too.

## PR-3 disposition (2026-08-14) — what was deleted, what was kept, and why

**Deleted (the engine-step path, both op-level and FPM):**

- `base_backend`'s Python step branches: the phase runners
  (`_run_context_phase` / `_run_generation_phase`), `run_mixed`'s three-pass
  composition, `_get_fpm_mix_step_latency`, the encoder `op.query()`
  fallback loop, and the `RustEngineUnsupportedError` "parity by
  delegation" rescue arms — an inexpressible op graph is a hard error now
  (the opspec coverage tripwire keeps that unreachable for shipped ops). A
  non-`PerfDatabase` database on a step surface raises `TypeError` (the
  compiled engine resolves perf data from disk by identity). En route, the
  two #1461 leftover guards that still forced FPM static/decode onto the
  Python walk were closed (rust-first, verified answer-preserving to full
  precision on the synthetic parity fixture).
- `fpm_forward.py`'s query machinery: `query`/`query_totals`/
  `query_pass_baseline`, the parquet+sidecar loader and validators, the
  perf_interp configs, and `_oplevel_sol_fn` (the per-op DatabaseMode.SOL
  roofline closure). The Rust core owns FPM end to end
  (`perf_database/fpm_forward.rs`, `operators/fpm_forward.rs`,
  `operators/fpm_sol.rs`); `FPMForwardOp` keeps only the construction
  surface `_to_opspec` and the memory model consume.
- The `"python"` value of `engine_step_backend` became a warn-once
  deprecation NO-OP (routes to the compiled engine; accepted one release
  cycle, then dropped). Unknown values now raise. The gate keeps only the
  non-`PerfDatabase` delegation (consumed by the AFD orchestration).
- The live-Python golden capture harness (`regenerate_goldens.py` + guard
  tests): the goldens are frozen artifacts; `pin_goldens.py` appends/
  refreshes records from the live rust engine (provenance-marked
  `post_freeze_pins`), making the golden diff the review artifact. The FPM
  parity class freezes its Python-side references inline
  (`_FPM_*_FROZEN`) since its dataset is generated per-run.
- The relative Rust-vs-Python CI perf gate and the benchmark's python arm;
  the python-vs-rust support-matrix compare machinery
  (`scan_rust_parity.py`, `--compare-engine-step-backends`); the
  `prediction_regression_gate` python pin flipped to rust.
- The dead `Mamba2` composite op class (+ its opspec EXEMPT entry).

**Kept — the PER-CALL query stack (`operations/*.py` `query()` +
`_query_*_table`, `perf_database.query_*`, `perf_interp/`,
`util_empirical.py`) stays intact.** Its load-bearing consumers:

1. **AFD comm ops** (`afd_transfer.py`, permanent Python orchestration)
   query `query_nccl` / `query_p2p` / `query_mem_op` EMPIRICALLY, and
   `_sum_latency` keeps its `op.query()` fallback loop.
2. **`tools/sanity_check/validate_database.ipynb`** (+ its e2e) exercises
   10 `PerfDatabase.query_*` methods per-call, including the SOL_FULL
   raw-tuple diagnostic — which therefore stays as-is.
3. **Internal couplings** that make partial carving unsafe: GEMM's silicon
   path re-queries SOL (fp8_static floor), `_correct_sol` needs table
   lookups at load time, mamba has no mode dispatch, MSA's empirical path
   divides DSA-SOL by DSA-SILICON.

(FPM was the third hard dependent when PR-3 was first scoped — its
roofline queried every op-level op in SOL — but #1461 moved that to
`fpm_sol.rs`, which is what unlocked deleting the walk.)

**Sequel ladder (tracked in #1357 Phase 3):**

1. **PR-4 — notebook re-oracle** (independent of PR-3; can run in
   parallel): re-oracle `validate_database.ipynb` onto the per-op
   evaluation FFI (needs a small FFI addition if the sol_math/sol_mem
   decomposition plots are to survive; rust computes both components
   internally). This is the prerequisite that unblocks the per-call
   deletion — the notebook is the `query_*` facade's biggest live
   consumer.
2. **PR-5 — per-call query-stack retirement** (needs PR-3 + PR-4): delete
   the per-call stack family-by-family (#1357's thin-delegation shape),
   retiring `query_*`, the empirical/silicon table bodies, and
   `util_empirical`'s math (keep the provenance constants) — with the AFD
   comm-table queries re-pointed at the op-list FFI or kept as the last
   per-call island. The deprecated `Mamba2` composite's disposition lands
   here too. **DONE — this PR.** Landed in one PR rather than
   family-by-family: the pinned pre-retirement baseline
   (`tests/cross_package/test_query_shim_baseline.py`, 97 cases captured
   from the Python math before deletion) made the whole-surface swap
   verifiable at once. `query_*`/`Operation.query` survive one release as
   engine-routed deprecation shims (5 tombstones raise: the two GEMM
   overhead sub-table queries, the two legacy deepep walks, and the
   per-phase `query_trtllm_alltoall`); AFD's three query points and the
   `Mamba2` composite's five legs evaluate standard twin ops through the
   single-op plumbing.
3. **Deprecation-cleanup PR** (time-locked): drop the `"python"`
   `engine_step_backend` value, the routing gate, and the CLI choice after
   the one-release bake PR-3 starts — plus the PR-5 shims
   (`PerfDatabase.query_*`, `Operation.query`, `_evaluate_single_op`'s
   deprecation plumbing) and the `Mamba2` composite export.

**Post-PR-5 invariant (the single-oracle rule):** per-op performance VALUES
(latency, energy, SOL decomposition) are computed ONLY by the compiled
engine (`aic-core/rust/aiconfigurator-core`). Python owns model/topology
composition, data loading, and orchestration — never estimation math. New
per-op access goes through the op-list FFI (`EngineHandle.evaluate_ops_json`
/ `evaluate_ops_sol_json`), the per-phase surface (`run_static_per_op`), or
whole runs; there is no supported per-call Python query surface after the
shims' window closes. Enforced by
`tests/cross_package/test_single_oracle_contract.py` (frozen shim surface,
no `_query_*_table`/`get_sol`/`get_empirical` defs, whitelisted `query`
overrides, `perf_interp` stays deleted) and mirrored in `.coderabbit.yaml`
path instructions; the `.claude/rules/rust-core/parity.md` Rule 2 update landed with this
migration at maintainer direction.

The keep/delete inventory and Gate-3 text below are retained as the
original plan of record; where they conflict with this disposition, the
disposition wins.

**Status (2026-08-06): PR-1 (#1454) MERGED 2026-08-05; PR-2 in flight.**
PR-2 closes every gap listed below and lands the Gate-2 golden anchor. State
of the former gaps:

- **Energy/power crosses the FFI (PR-2).** Rust perf leaves carry the
  measured `{latency, power, energy}` triple (`perf_interp.rs::LeafValue`);
  power blends with the Python engine's exact semantics (linear lerp /
  median util-hold / IDW site mean — NOT energy-lerp), and per-op
  `energy_wms` rides the op-list FFI. The PR-1 power routing carve-out and
  the static-path "run the Python phase runners for energy only" double
  evaluation are both DELETED; the Python `query()` layer is no longer a
  live dependency of rust-routed runs. Known dormant gap: the WideEP/deepep
  table families remain latency-only on the Rust side — no shipped wideep
  parquet carries power columns today, so both engines report 0.0 there;
  thread it when wideep power data first ships (#1439 territory).
- **AFD** sources its per-op values from the compiled engine via the thin
  op-list evaluation FFI (`evaluate_context_ops` / `evaluate_generation_ops`,
  index-addressed into the compiled spec); the A/F partitioning, stride
  integration, and the five synthetic comm ops stay Python-side permanently.
  (Correction to the earlier text: AFD never routed through
  `RustEngineUnsupportedError` fallback — its decode pipeline called
  `Operation.query()` directly in `AFDInferenceSession._sum_latency`; that
  call site is now the FFI insertion point, with the Python loop as
  fallback.) The VL encoder phase likewise: shape math stays Python, per-op
  values come from `evaluate_ops_json` (encoder ops are deliberately NOT in
  the compiled spec).
- **SOL is ported; SOL_FULL is a Python-side per-call diagnostic (PR-2).**
  The compiled engine dispatches `DatabaseMode::Sol` per family in front of
  the already-ported SOL formulas; `_RUST_SUPPORTED_DATABASE_MODES` includes
  SOL, emptying the gate's mode-based delegation. SOL_FULL is declared a
  permanently Python-side PER-CALL diagnostic (#1357's alternative
  disposition): `query_*(..., database_mode=SOL_FULL)` keeps returning the
  raw `(sol_time, sol_math, sol_mem)` tuple the sanity-check notebook
  plots, while mode entry rejects it as a DEFAULT mode (it has never worked
  there — the phase runners cannot consume a bare tuple). The per-call
  branches are engine-step-dead and delete together with `query()` in PR-3,
  which must also decide the sanity notebook's replacement oracle.

## Revised PR sequence (2026-08-01) — supersedes the P0–P4 table below

| # | PR | Scope | Maps to |
| --- | --- | --- | --- |
| **PR-1** | Rust default | Flip default resolution to the compiled engine; power-carrying and non-`PerfDatabase` databases delegate to the Python step; explicit `"rust"`/`"python"` keep force semantics (`"python"` is the escape hatch, retained one release). Propagate `engine_step_backend` into the internal `RuntimeConfig` constructions (`run_mixed` passes 1–3, `_get_genonly_step_latency`) so the escape hatch binds the whole composition. Forced-rust full-matrix scan as merge evidence. No deletion. Bake one release cycle. | P0+P1 |
| **PR-2** | Golden anchor + gap closure — **DELIVERED by this PR** | Goldens captured from the live Python path (engine-step 346 case/surface records incl. error classes, compile-engine references, per-op dicts for the compile-engine subset; byte-idempotent `regenerate_goldens.py`); parity suites rewired to Rust-vs-golden with anti-vacuous guards. Energy crosses the FFI (power carve-out and the static energy double-run deleted). The op-list evaluation FFI restores the user-facing per-op breakdown (real op names + per-op energy + real provenance tags replace the `rust_engine_step_*` synthetic keys) and feeds AFD `_sum_latency` / the VL encoder phase. SOL ported; SOL_FULL kept as a Python-side per-call diagnostic (default-mode entry rejects it). Issue #1456 (site-transfer tie-break order) fixed and anchored by smoke cases. | P2 + gaps |
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

The Rust path became the DEFAULT in PR-1 (#1454), and PR-2 closed the
remaining delegations: `should_use_rust_engine_step`
(`sdk/rust_engine_step.py`) now keeps only the explicit `"python"` escape
hatch and the non-`PerfDatabase` duck-type delegation. The paragraphs below
describe the pre-Phase-2 starting point this plan set out to remove.

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
- The smoke harness (`parity_tests/test_engine_step_parity.py`,
  `test_compile_engine_parity.py` — 346 golden-backed case/surface pairs as
  of PR-2) passes bit-identical-or-within-tolerance.

**No deletion in this PR.** Both engines stay; only the default changes.

### Gate 2 — Golden capture (replace the live differential oracle) — SATISFIED (PR-2)

The parity tests compared **Python vs Rust live**; deleting the Python path
would have destroyed the regression detector future Rust changes rely on.
PR-2 delivered:
- Python `run_static` / step-latency outputs captured as golden fixtures
  across the full smoke matrix (`parity_tests/goldens/`, including error
  classes — error symmetry is a first-class golden value — plus per-op
  latency/energy/source dicts for the compile-engine subset).
- `test_engine_step_parity.py` / `test_compile_engine_parity.py` assert
  **Rust vs golden**; `regenerate_goldens.py` (Python side pinned, thread
  caps pinned, byte-idempotent) is the sanctioned regeneration path.
  Anti-vacuous guard tests prove the comparison detects drift.
- The rewiring exposed and fixed a vacuous comparison: the mixed-step
  helper's bare `RuntimeConfig` had rust-routed the "python" side since the
  PR-1 flip (silent self-comparison); it is now pinned to the Python step.

### Gate 3 — Delete the duplicated Python latency code

Only after Gates 1–2 hold and have soaked one release cycle:
- **Per-op breakdown precondition — SATISFIED (PR-2)**: the op-list
  evaluation FFI exposes per-op `(name, latency_ms, energy_wms, source)`
  and the bridge populates the `InferenceSummary` per-op dicts with real op
  names, energies, and provenance tags; the compile-engine parity subset
  anchors the per-op dicts against goldens.
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
| PR-3 deletes the Python step while the Rust path still collapses the per-op breakdown to a synthetic key — SDK users permanently lose op-level latency analysis. | RESOLVED (PR-2): per-op values cross the op-list evaluation FFI and populate `InferenceSummary` with real names/energies/sources; golden-anchored. |
| WideEP/deepep tables are latency-only on the Rust side; the MoE-dispatch wrapper sums only `.latency_ms` of its inner comm results (discarding energy the comm loaders already carry); DSA's `has_power` gate is per-source in Rust vs first-source in Python. Energy silently diverges if power data ships for those tables before the ports. | All dormant today (no shipped comm/wideep/DSA parquet carries power; both engines report 0.0 on every affected path — verified). Thread them together with the first relevant power drop (#1439 territory). |
| Residual exact-tie enumeration-order gaps of the #1456 class (grid_hold `nearest_key` toward-smaller-key; attention `_ref_head_size` / DSA `bs_slice` sorted-vs-insertion order) — all self-documented in code, all fire only on bitwise-equal distance ties. | Pre-existing (not introduced by Phase 2). Fix with the same load-order-record pattern as the GEMM site fix if a real tie ever surfaces in the scan. |

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
