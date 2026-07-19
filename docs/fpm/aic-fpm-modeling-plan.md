# AIC-FPM Modeling Plan (Task 2) — realigned to the as-built collector

> Revision: 2026-07-19
>
> Supersedes: Task 2 sections (M0–M4, §5.6–§5.7) of `aic-fpm-integration-plan.md` (2026-07-13)
>
> Basis: `feature/fpm-collector` @ `cf1b3cbb` (Task 1 as built, 142 unit tests green) and
> `feature/fpm-modeling` @ `cae1aef7` (upstream/main after the #1322 aic-core restructure)

## 1. What changed since the original plan

Task 1 was implemented with different mechanics than the 2026-07-13 plan assumed. Task 2
consumes the **as-built** contract, not the planned one:

| Topic | Original plan | As built (authoritative) |
|---|---|---|
| Physical axes | per-request `(B,S,P)` / `(B,L)` | per-DP-rank iteration totals `(batch_size, total_prefill_tokens, total_kv_read_tokens)`, `partition_policy=balanced_v1` |
| Point grid | collector-owned Halton/maximin, frozen holdout IDs | Dynamo PR11509 native self-benchmark owns the grid; runtime-admitted, no holdout set |
| Repeats | 5 repeats, median, CV/quarantine | single sample per point (`dynamo_native_single_sample_v1`); DP max is baked into `latency_ms` |
| Publication | candidate dir + human promotion | collector writes the formal pair directly (locked, atomic, conflict-fail merge) |
| Sidecar | predictor ID, block size, per-axis min/max | schema v5: hashes, run identities, counts — **no predictor/domain fields** |
| Backends | vllm/sglang/trtllm probes | vLLM only; `PP=1`, `CP=1` fixed; one backend policy `baseline_auto` |

Consequences adopted in this plan:

1. **Interpolation reuses `perf_interp` v2** (Python engine + existing Rust port with parity
   infrastructure) instead of the plan's bespoke log-quadratic predictors (§5.7). The collected
   grid is structured-but-runtime-pruned; `perf_interp`'s Grid/ScatteredSites handles that, and
   we get cross-language parity machinery for free.
2. **Per-cell domain is derived from rows at load time** (min/max per axis + exact-key set),
   since the sidecar carries no domain metadata. Extending the sidecar is a cross-module
   contract change and is out of scope for V1.
3. **Per-cell MAPE holdout qualification (M1.4) is dropped for V1** — with a single-sample,
   runtime-owned grid there are no frozen holdout IDs to qualify against. Accuracy is instead
   covered by (a) golden interpolation fixtures, (b) the existing
   `tools/prediction_regression_gate/` + accuracy-tracking machinery once real data is promoted.
4. **Energy follows the existing zero-energy convention** of the Rust engine-step path
   (`base_backend.py:440-441` zeroes energy dicts) rather than a new hard-error path. The
   original plan predates this precedent; consistency wins. Documented as a known limitation.

## 2. Data contract consumed (fixed by Task 1 — do not change here)

Location (post-restructure the canonical tree is `aic-core/src/aiconfigurator_core/systems/data/`;
`src/aiconfigurator/systems` is a symlink to it):

```text
systems/data/<system>/<backend>/<version>/
  fpm_forward_perf.parquet
  fpm_forward_perf.metadata.json     # schema_name=aic_fpm_forward_perf, schema_version=5
```

Row key (24 columns, `collector/fpm_forward/database.py:_ROW_KEY`):
`cell_id, model_path, system, backend, backend_version, weight_quantization, gemm_quant_mode,
moe_quant_mode, fmha_quant_mode, comm_quant_mode, kv_cache_dtype, tp, pp, dp, moe_tp, moe_ep,
cp, backend_axis, backend_policy, workload_kind, batch_size, total_prefill_tokens,
total_kv_read_tokens, partition_policy`. Value column: `latency_ms`.

Coordinate semantics (verified against `native_artifact._expected_scheduled` /
`_validate_fpm`): every DP rank executes the **same** point, so the workload columns are
**local per-DP-rank totals**:

- prefill: `batch_size=B` local requests, `total_prefill_tokens=Σ new tokens`,
  `total_kv_read_tokens=Σ past-KV tokens` (0 for ordinary prefill);
- decode: one token per request, `total_prefill_tokens=0`, `total_kv_read_tokens=Σ KV lengths`;
- `latency_ms = max over DP ranks of scheduler wall_time` for that point.

This matches the modeling core's existing convention that ops are queried with **local**
batch (`base_backend.py` phases; MoE ops re-globalize internally). Returning the collected
max-over-ranks latency as the per-rank estimate is consistent with
`Engine::forward_pass_time_ms` taking the max across attention-DP ranks.

Loader obligations (mirroring the writer's semantics):

- require both files; recompute and match `parquet_sha256` (the sidecar is the commit record —
  an unmatched pair after an interrupted writer must be rejected);
- reject unknown `schema_name`/`schema_version`;
- reject duplicate physical keys, non-finite/non-positive latency, mixed `backend_version`;
- reject queries whose cell is absent or whose point is outside the derived finite domain with
  `PerfDataNotAvailableError` (Python) / `AicError::PerfDatabase` (Rust) — no clamp, no
  fallback to another cell.

## 3. Design

### M0 — `forward_model` switch (default unchanged)

- Add `forward_model: str = "op_level"` to **`ModelConfig`**
  (`aic-core/src/aiconfigurator_core/sdk/config.py:11`). ModelConfig, not RuntimeConfig,
  because the op-list rewrite happens inside `get_model`, which receives ModelConfig only.
  Values: `op_level | fpm`; anything else raises.
- Thread it exactly like `engine_step_backend`:
  TaskV2 field + `build_model_config` (`src/aiconfigurator/sdk/task_v2.py:354, 1074`),
  v1-compat passthrough (`task_v1_compat.py:83`), CLI flag `--forward-model`
  (`src/aiconfigurator/cli/main.py`, `cli/api.py`), webapp bridges, `pareto_analysis.py`.
  Generator is **not** touched in V1 (module boundary).
- While M2 (Rust) is not landed, `forward_model=fpm` forces the Python engine step:
  `should_use_rust_engine_step` (`rust_engine_step.py:224`) returns False for FPM models.

### M1 — Python loader, `FPMForwardOp`, centralized rewrite

New code lives in the modeling core (`aic-core/src/aiconfigurator_core/sdk/`):

1. `PerfDataFilename.fpm_forward = "fpm_forward_perf.parquet"` (`common.py:968`).
2. A loader in `perf_database.py` producing per-cell tables + derived domains, with the §2
   validations. Cells are keyed by the identity columns; the sidecar is validated once per
   directory.
3. `FPMForwardOp(phase)` in `operations/` (subclass of `Operation`,
   `operations/base.py:114`). `query()` mapping from the existing phase-loop kwargs
   (`base_backend.py:303-388`):
   - prefill: `(batch_size=B, s=effective_isl, prefix=P)` →
     `(B, total_prefill_tokens=B*s, total_kv_read_tokens=B*prefix)`;
   - decode: `(batch_size=B·(nextn+1), s=isl+i+1)` →
     `(B, total_kv_read_tokens=B*s)`;
   - exact-key hit returns the stored row; otherwise `perf_interp.query` with a new
     `fpm_forward` `OpInterpConfig` (axes above; value transform chosen via fixtures against
     real GLM-5.2 data — start with log-token axes like the attention configs);
   - `perf_interp` requires a whole-model `sol_fn` even without extrapolation: in-domain
     ragged-bracket recovery and ScatteredSites cross-site transfer both rescale by SOL
     ratios. V1 ships a crude per-rank roofline built from the ORIGINAL op lists at rewrite
     time (dense-FLOPs + attention-pair term for prefill; weight bytes + KV bytes for
     decode) — only per-axis scaling trends matter, constants cancel in the ratios;
   - domain check happens **before** `perf_interp` (its boundary util-hold semantics are
     wrong for whole-model latency; out-of-domain must error, not hold);
   - energy: 0.0 (documented convention, §1.4).
4. Centralized rewrite at the `get_model` return site
   (`models/__init__.py:115`): when `forward_model == "fpm"`,
   `context_ops = [FPMForwardOp(prefill)]`, `generation_ops = [FPMForwardOp(decode)]`.
   - Tolerate property-backed op lists (`nemotron_nas.py:89-106` has setters).
   - Reject models with non-empty `encoder_ops` (no FPM data for encoders).
   - Model metadata, parallelism, and public model type are unchanged; no model class
     rewrites its own lists.

Cell selection maps model/runtime identity onto row identity:

- `system/backend/version` — from the `PerfDatabase` instance;
- `gemm/moe/fmha/comm/kvcache` quant modes, `tp/pp/dp/moe_tp/moe_ep/cp` — from `ModelConfig`
  (names align one-to-one with row columns);
- backend policy — V1 requires the `baseline_auto` profile
  (`moe_backend=None, attention_backend=None, enable_wideep=False, enable_eplb=False`);
  a ModelConfig that deviates fails with a clear unsupported-policy error;
- `model_path` — see Open decision D1.

### M2 — Rust `FpmForward` op + schema bump

Change surface (from the Rust core map):

1. New op struct + module under `src/operators/`, variant in `Op`
   (`operators/op.rs:91-137`), extend `name()` (`:182`), `query()` (`:243`), and the
   exhaustiveness/round-trip guard `all_op_variants()` (`engine/spec.rs:522-591`).
2. Bump `ENGINE_SPEC_SCHEMA_VERSION` 2→3 in lockstep: `config.rs:24` and Python
   `engine.py:92`. Old specs must fail with `UnsupportedSchemaVersion` (existing gate,
   `spec.rs:109-136`).
3. Python→Rust: `_fpm_forward` converter + `_to_opspec` branch (`engine.py:509-599`).
4. New `perf_database/fpm_forward.rs` table on `PerfDatabase` (`perf_database/mod.rs:87-104`)
   reading the same parquet pair (Rust already reads parquet directly), wired through the
   Rust `perf_interp` port with the **same** config as Python; misses/out-of-domain surface
   as `AicError::PerfDatabase` → `PyValueError`.
5. Parity: extend the live parity suites (`parity_tests/test_compile_engine_parity.py`,
   `test_engine_step_parity.py`) with FPM-mode cases over a synthetic dataset in a temp
   systems root; `PARITY_RTOL = 0.01` as today. If parity cannot be met, V1 ships
   exact-match only in Rust (mirrors the original plan's fallback).

### M3 — explicit pure/mixed branch (both languages)

The mixed-step split is name-based (`"context_attention"`/`"generation_attention"` string
match — Python `base_backend._get_mix_step_latency:977-1026`, Rust
`session.rs:get_mix_step_ops:132/169/194`). A whole-model op matches neither filter and would
be silently summed once in pass 1 with the wrong workload — so FPM models must **never**
reach that code:

- add an explicit FPM branch ahead of the 3-pass split in both languages:
  - mixed step = `FPM_prefill(B_ctx, tokens_ctx, kv_ctx)` +
    `[FPM_decode(B_gen, kv_gen) − FPM_decode_baseline(B_gen)]` — the **marginal-decode
    composition** (owner-approved 2026-07-19, superseding the plain sum in the original
    plan §1/M3). A mixed step is one shared forward pass: weight reads, kernel launches,
    and fixed per-step overheads are paid once, by the prefill component; a full
    pure-decode step would pay them twice. Sampling the decode curve at its KV-axis floor
    (`max(B, domain_min)` — one KV token per request is the physical minimum) isolates the
    shared-pass part, so the subtraction keeps only the KV-read/attention cost that
    genuinely adds to the iteration. Residuals: the gen tokens' GEMM marginal is dropped
    (small when `B_gen ≪ ctx_tokens`, slight underestimate — the plain sum overestimates,
    so truth is bracketed), and the subtraction doubles single-sample noise variance;
  - a generation-only step (`ctx_tokens == 0`) has no pass to ride on and keeps the full
    decode latency;
  - pure prefill / pure decode = one query;
  - empty work = 0; no mixed row is ever synthesized;
- the branch lives in the shared engine/backend entry points (`rank_latency_ms`,
  `_get_mix_step_latency`, `_get_genonly_step_latency`) so static, agg, and disagg consumers
  inherit it through their existing calls;
- guard: in op-level mode nothing changes byte-for-byte; assert FPM op lists are exactly
  `[FPMForwardOp]` before taking the branch.

### M4 — consumer and compatibility verification

Both modes across: Python static, Rust native engine, `RustForwardPassPerfModel`
native/correction path, agg, disagg, TaskV2, CLI.

Required tests (unit tests under `tests/unit/sdk/`, mirroring existing layout):

- default `op_level` yields byte-identical op lists and predictions (regression gate stays
  green);
- FPM rewrite yields exactly one op per phase list; encoder models rejected;
- loader: missing sidecar, hash mismatch, duplicate keys, mixed versions, unsupported
  schema, non-finite latency all fail;
- exact-match rows for P=0 prefill, P>0 prefill, decode; interpolation golden fixtures;
- out-of-domain and missing-cell queries fail explicitly;
- mixed = pure prefill + pure decode (both languages);
- DP: dp>1 queries use local axes and reproduce collected values exactly;
- old `EngineSpec` versions fail with the version error;
- producer-consumer test (I1 analogue): a synthetic parquet+sidecar pair written **from the
  documented schema** (not by importing collector code — module boundary), loaded in a temp
  systems root, queried from Python and Rust.

## 4. Work order

```text
M0 config switch  ──►  M1 Python op + loader (synthetic fixtures)
                              │
                              ▼
                  M3 Python mixed branch  ──►  M4 Python consumer tests
                              │
                              ▼
                  M2 Rust op + schema bump + parity  ──►  M4 Rust/parity tests
```

Python-first is deliberate: M0+M1+M3 are shippable with `fpm` forcing the Python engine
step; M2 removes that restriction. Real-data validation (GLM-5.2 vLLM cells from Task 1)
follows once a dataset lands in the tree.

## 4.1 Implementation status (2026-07-19)

M0 + M1 + M3 + the Python half of M4 are implemented on `feature/fpm-modeling`:
`operations/fpm_forward.py` (loader + op + interp configs + sol builder), the `get_model`
rewrite, `ModelConfig.forward_model` threaded through TaskV2 / v1-compat / CLI
(`--forward-model`), the explicit FPM mixed/genonly branches with the Rust engine-step
forced off, and `tests/unit/sdk/test_fpm_forward.py` (33 tests). D1 shipped as
exact-model-path-then-unique-fallback; D2 shipped as ScatteredSites (per-phase configs,
one-line swap to Grid for the LOO bake-off); D3 shipped as zero-energy; D4 deferred to M2.
The mixed step ships the marginal-decode composition (§M3), replacing the original plan's
plain prefill+decode sum. M2 (Rust op + schema bump + parity) and real-data LOO
qualification remain.

## 5. Open decisions (need owner sign-off before implementation)

- **D1 — model identity matching.** Rows key on the HF `model_path` string; `ModelConfig`
  carries no path. Proposal: select by all other identity columns; if exactly one
  `model_path` remains, use it; if several, require an explicit path from the consumer
  (TaskV2 flows have it). Alternative: add a `model_path` field to ModelConfig.
- **D2 — interpolation config.** Start from `perf_interp` Grid over
  `(batch_size, total_prefill_tokens, total_kv_read_tokens)` with log-scaled token axes;
  freeze after fixture evaluation on the first real GLM-5.2 cell. Fallback: exact-match only.
- **D3 — energy.** Zero-energy convention (recommended, matches Rust engine-step precedent)
  vs hard error on energy queries in FPM mode (original plan §M1).
- **D4 — schema bump timing.** Bump `ENGINE_SPEC_SCHEMA_VERSION` in M2 only (recommended)
  vs pre-emptively in M0.

## 6. Non-goals (unchanged from the original plan unless noted)

- No mixed-workload data; no TTFT/ITL/queueing model; no replacement of the
  online-correction/regression API (`ForwardPassPerfModel` stays the public Rust surface —
  the compiled engine simply contains FPM ops when selected).
- No sidecar schema changes, no collector or generator edits (cross-module).
- No silent extrapolation, no cross-cell fallback, no implicit backend fallback.
- No per-cell MAPE holdout gate in V1 (moved to prediction-gate tooling; see §1.3 — this is
  a change from the original plan).
