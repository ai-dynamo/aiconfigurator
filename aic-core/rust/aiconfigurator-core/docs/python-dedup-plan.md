<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Phase 2/3 Record — Rust-default flip and Python latency-path removal

**Status (2026-08-23): FULLY LANDED — this document is a record, not a
plan. Phase 2 merged as PR-1 (#1454), PR-2 (#1496), PR-2.5 (#1508) and
PR-3 (#1521, the PR that first carried this text); the Phase 3 sequel
ladder below closed out with PR-4 (#1547), PR-5 (#1552), PR-6 (#1555)
and the deprecation-cleanup PR (#1566), and issue #1357 is closed.**
Nothing here is scheduled work. What survives the plan is the two
deferrals on ladder item 4 (kv-cache bytes, the AFD orchestration
quartet) and the two dormant gaps in the closing section.

The superseded planning scaffolding — the gate ladder, the keep/delete
inventory, the P0–P4 and PR-1/2/3 sequence tables, the acceptance
criteria and the pre-deletion safety audit — was removed once every gate
had been satisfied and every PR merged. Recover it from git history if
you need to see what was promised versus what shipped.

## Background

Phase 1.5 (#1200) made Python build the op list and Rust execute it, and
deleted the **Rust** model layer (`models/`, `backends/`, `factory.rs`);
#1201 added the capacity API. That left the symmetric duplication with
the polarity flipped: the Python latency stack — `operations/*.py`
`query()` bodies, `perf_database.py`'s latency-query methods, and the
op-walk in `backends/base_backend.py` — mirrored the Rust `operators/` +
`perf_database/` + `engine/` code, so two engines computed the same step
latency. Phase 2 made Rust the default execution path and removed the
duplicate; Phase 3 finished the job down to the data plane.

PR-3 delivered the ENGINE-STEP-path retirement — see the disposition
below. Its original scope ("delete the per-call query stack wholesale")
was revised after FPM (#1384) landed a live consumer of that stack;
#1461's Rust FPM port (Op::FpmForward + fpm_sol.rs) then let PR-3 delete
the Python FPM walk too.

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
- (Correction, cleanup PR: an earlier revision listed the dead `Mamba2`
  composite as deleted here. It was NOT — PR-3 kept it as a deprecated
  five-leg composite on the query whitelist; it was deleted with the
  per-call shims in the deprecation-cleanup PR.)

**Kept by PR-3 — the PER-CALL query stack (`operations/*.py` `query()` +
`_query_*_table`, `perf_database.query_*`, `perf_interp/`,
`util_empirical.py`) survived this PR intact**, then was retired by PR-5
and deleted outright by the deprecation-cleanup PR (ladder items 2 and 4
below). The three dependencies that made it uncarvable *at the time*:

1. **AFD comm ops** (`afd_transfer.py`, permanent Python orchestration)
   queried `query_nccl` / `query_p2p` / `query_mem_op` EMPIRICALLY, and
   `_sum_latency` kept its `op.query()` fallback loop. (Now: the comm ops
   evaluate standard twin ops through the engine's single-op plumbing.)
2. **`tools/sanity_check/validate_database.ipynb`** (+ its e2e) exercised
   10 `PerfDatabase.query_*` methods per-call, including the SOL_FULL
   raw-tuple diagnostic. (Now: re-oracled by PR-4/PR-5; SOL_FULL is served
   by `evaluate_ops_sol_json`.)
3. **Internal couplings** that made partial carving unsafe: GEMM's silicon
   path re-queried SOL (fp8_static floor), `_correct_sol` needed table
   lookups at load time, mamba had no mode dispatch, MSA's empirical path
   divided DSA-SOL by DSA-SILICON. (Resolved by retiring the whole surface
   at once against a pinned baseline rather than family-by-family.)

(FPM was the third hard dependent when PR-3 was first scoped — its
roofline queried every op-level op in SOL — but #1461 moved that to
`fpm_sol.rs`, which is what unlocked deleting the walk.)

**Sequel ladder (tracked in #1357 Phase 3):**

1. **PR-4 — notebook re-oracle** — **DELIVERED by #1547**: the
   sol_math/sol_mem decomposition crosses the FFI
   (`AicEngine.evaluate_ops_sol_json`, `SolComponents` riding on
   `PerformanceResult`), and `validate_database.ipynb` /
   `create_charts.py` source per-op values from the compiled engine
   (`tools/sanity_check/engine_reference.py` over a model-less
   `EngineHandle.for_database` probe). The charts show OP-LEVEL
   estimates (context attention includes the fused extras; gemm
   fp8_static charts the op model). The one residual left open here —
   the `query_trtllm_alltoall` per-phase chart, which no op-level
   evaluation expresses — was closed by PR-5 (#1552): the chart walks
   the raw phase table (`_trtllm_alltoall_data`) directly and scores it
   against a closed-form wire model mirrored in the notebook, verified
   0-mismatch against the retired facade. That mirror is a knowing
   exception to the single-oracle rule below, confined to the
   diagnostic notebook.
2. **PR-5 — per-call query-stack retirement** (needs PR-3 + PR-4): delete
   the per-call stack family-by-family (#1357's thin-delegation shape),
   retiring `query_*`, the empirical/silicon table bodies, and
   `util_empirical`'s math (keep the provenance constants) — with the AFD
   comm-table queries re-pointed at the op-list FFI or kept as the last
   per-call island. The deprecated `Mamba2` composite's disposition lands
   here too. **DONE — this PR.** Landed in one PR rather than
   family-by-family: the pinned pre-retirement baseline
   (`tests/cross_package/test_query_shim_baseline.py`, 120 cases captured
   from the Python math before deletion) made the whole-surface swap
   verifiable at once. `query_*`/`Operation.query` survive one release as
   engine-routed deprecation shims (5 tombstones raise: the two GEMM
   overhead sub-table queries, the two legacy deepep walks, and the
   per-phase `query_trtllm_alltoall`); AFD's three query points and the
   `Mamba2` composite's five legs evaluate standard twin ops through the
   single-op plumbing.
3. **PR-6 — data-plane FFI + weight physics** (independent of the
   deprecation window; runs while the PR-5 shims bake). **DONE — this
   PR.** The engine table view (`perf_database/table_view.rs`,
   `AicEngine.table_view_json`) re-folds the raw parquet sources into
   every retired Python loader's exact nested-dict shape — values, key
   TYPES (rehydrated to enums/ints/tuples in
   `sdk/engine_table_view.py`), insertion order, and empty subtrees —
   and every `PerfDatabase._<family>_data` attribute now binds it, so
   the notebook grids, support matrix, task validation gate, and every
   other consumer kept their code unchanged. Per-op weight physics moved
   to `Op::weight_bytes` (batch FFI `weights_ops_json`); the base
   `Operation.get_weights` routes there and the per-class `_weights`
   math is deleted. The `_GEMM/_MOE_QUANT_UTIL_LEVEL` dicts are
   projections of the Rust tables. Verified by a pinned pre-deletion
   baseline (7 databases × every table attribute + support matrices +
   per-op weights, `tests/cross_package/test_data_plane_baseline.py`;
   retired in the deprecation-cleanup PR once the equivalence landed).
   Deferred to the cleanup PR from the original cut: the Python-side shared-layer
   source resolution (`_build_op_sources` still feeds the engine's
   source map — moving it is orthogonal to the view work), kv-cache
   bytes (model-level polymorphism, needs a model-geometry spec), and
   the moe/moe_comm/dsa/dsv4 parsers, which survive as TEST-ONLY
   schema-contract fixtures for the collector suite's format handshake
   (no production path parses perf files in Python anymore).
4. **Deprecation-cleanup PR — removal + pyo3 op unification**
   (time-locked). **DONE — this PR.** Landed exactly as planned, in four
   segments:
   - *Deprecated-surface removal:* the `"python"` `engine_step_backend`
     value, `PerfDatabase.query_*` / `Operation.query` (shims and
     tombstones alike), `_evaluate_single_op`'s re-moding dimension, and
     the `Mamba2` composite. `_evaluate_single_op` survives as the
     PERMANENT internal single-op plumbing behind the AFD twins and the
     fallback loop.
   - *Source resolution:* `_build_op_sources` moved to the engine
     (`perf_database/source_resolution.rs`; schema v13 —
     `EngineConfig` dropped the Python-resolved `perf_db_sources` map for
     `enable_shared_layer` + `strict_provenance` policy flags; structured
     resolver warnings re-emit through the Python warn-once registries).
   - *pyo3 op unification:* the Rust op structs ARE the Python classes
     (`py_ops.rs`, 32 families + the base); `models/*.py` construct them
     directly, `operations/*.py` keeps thin data-plane shells, and the
     `_to_opspec` serializer family, the per-op weight FFI wrapper and
     the two-sided `ENGINE_SPEC_SCHEMA_VERSION` sync are deleted
     (`engine_spec_schema_version()` is the single owner; ops
     self-serialize via `_spec_json` / `ops_json_from_ops`, which refuses
     the RetiredDeepEp dispatch tombstone). The pickle prerequisite
     resolved via `__getnewargs_ex__` (shell identity survives
     `ProcessPoolExecutor` fork+spawn).
   - *Baseline + parser retirement:* both migration baselines retired
     with the surfaces they froze (`query_shim_baseline.json` with the
     shims; `data_plane_baseline.json` + its codec once the last parser
     left) — succeeded by the synthetic-parquet view-shape suites, which
     don't depend on shipped data staying byte-stable. The last Python
     perf parser (`load_dsv4_sparse_op_data`) retired when the
     `_dsv4_csa_topk_calib_data` view attribute landed; the collector
     contract tests assert against the engine view / frozen schema
     literals.

   Still deferred (unchanged from PR-6): kv-cache bytes (model-level
   polymorphism, needs a model-geometry spec) and the AFD orchestration
   quartet (Python-side by design; per-op values already cross the
   engine's single-op plumbing — see the `afd_transfer.py` module TODO
   for the port triggers).

**Post-PR-5 invariant (the single-oracle rule):** per-op performance VALUES
(latency, energy, SOL decomposition) are computed ONLY by the compiled
engine (`aic-core/rust/aiconfigurator-core`). Python owns model/topology
composition, model-config loading and shared-layer SOURCE SELECTION, and
orchestration — never estimation math, and (since PR-6) never
perf-data parsing: the engine loads and serves the performance tables,
Python consumes them through the table-view FFI. New
per-op access goes through the op-list FFI (`EngineHandle.evaluate_ops_json`
/ `evaluate_ops_sol_json`), the per-phase surface (`run_static_per_op`), or
whole runs; there is no per-call Python query surface at all — the
deprecation shims completed their one-release window and were deleted with
the cleanup PR. Enforced by
`tests/cross_package/test_single_oracle_contract.py` (no per-call
`PerfDatabase` query surface, no `_query_*_table`/`get_sol`/`get_empirical`
defs, whitelisted `query` overrides, a frozen per-file def inventory,
`perf_interp` stays deleted) and mirrored in `.coderabbit.yaml`
path instructions; the `.claude/rules/rust-core/parity.md` Rule 2 update landed with this
migration at maintainer direction.

## Known dormant gaps (carried forward)

Two items outlived the plan. Neither changes a shipped number today; both
are recorded here because the code that would trip them is still in place.

- **Comm-table energy is latency-only on the Rust side.** The WideEP/deepep
  families and the MoE-dispatch wrapper carry no power columns
  (`perf_database/wideep.rs`, `moe_a2a.rs`), the wrapper sums only
  `.latency_ms` of its inner comm results, and DSA's `has_power` gate
  evaluates per-source. No shipped comm/wideep/DSA parquet carries power
  today, so every affected path reports 0.0. Thread these together with the
  first relevant power drop — issue #1439.
- **Exact-tie enumeration-order gaps of the #1456 class.** `grid_hold`'s
  `nearest_key` breaks toward the smaller key; attention's `_ref_head_size`
  and DSA's `bs_slice` sort where Python used insertion order. All are
  self-documented at their sites and fire only on bitwise-equal distance
  ties, so none is reachable by shipped data. If a real tie ever surfaces in
  a scan, fix it with the load-order-record pattern
  (`quants_in_load_order` / `first_distribution` in
  `perf_database/{moe,moe_expert_compute}.rs`) that closed #1456 itself.

## Pointers

- Completed parity scan (DRIFT list, gate status): `parity-scan-report.md`.
- Scan procedure: `parity-scan-runbook.md`.
- Engine-step / compile-engine / perf gates in CI:
  `.github/workflows/build-test.yml`.
- The surviving routing gate: `sdk/rust_engine_step.py`
  (`should_use_rust_engine_step` — reduced to the non-`PerfDatabase`
  delegation described in the disposition above).
- Architecture reference: `design_doc.html`.
