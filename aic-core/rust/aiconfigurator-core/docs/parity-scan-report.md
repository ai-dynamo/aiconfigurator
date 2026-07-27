<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Rust↔Python Engine-Step Parity Coverage

**Generated: 2026-06-16 (UTC). Commit: `048c3a7f`** (final confirming rescan).
This is the parity sibling of `perf-speedup-report.md`; re-stamp the date/commit
whenever the scan is regenerated.

**Status as of 2026-06-16 — gate CLOSED.** **6 engine parity bugs found and
fixed** (§5); probe 0 DRIFT / 0 REGRESSION; Pareto 0 REGRESSION, 99.8% pass. The
4 residual Pareto DRIFT are **frontier-extent differences in the high-throughput
corner, not engine errors** — the user-facing frontier curves agree within ~1%
(§6). Two small, pre-existing limitations documented for fast-follow (§8).
Acceptance criterion: **0 REGRESSION + every DRIFT explained** (§9). The
confirming full rescan on the final build (commit `048c3a7f`) has completed and
reproduces these results exactly — **Phase 2 Python-deletion may proceed** (§9).

## 1. Why this report

Phase 2 flips the Rust engine-step to the default latency path and then deletes
the duplicated Python latency code (the per-op `query()` walk in
`base_backend`, the `operations/*.py` latency queries, and the
`perf_database` latency lookups). That deletion is gated on proving the Rust
engine reproduces the Python engine's latency across the **entire** supported
model × system × backend × version matrix.

This report is that gate's evidence: the coverage swept, the methodology, the
results, and every parity bug found and fixed along the way.

- Plan: [`aic-core/rust/aiconfigurator-core/docs/python-dedup-plan.md`](./python-dedup-plan.md)
- Scan runbook: [`aic-core/rust/aiconfigurator-core/docs/parity-scan-runbook.md`](./parity-scan-runbook.md)
- Harness: `tools/support_matrix/scan_rust_parity.py`

## 2. Coverage

The scan enumerates every supported `(model, system, backend, version, mode)`
combination from the support matrix.

| Dimension | Count | Values |
|---|---|---|
| Entries | **2,158** | agg + disagg per config |
| Models | 55 | HuggingFace IDs |
| Architectures | 20 | see below |
| Backends | 3 | sglang (892), vllm (660), trtllm (606) |
| Versions | 4 per backend | latest engine releases |
| Systems | 10 | b200_sxm (424), h200_sxm (328), gb200 (300), h100_sxm (264), b300_sxm (248), gb300 (246), rtx_pro_6000_server (118), l40s (116), a100_sxm (92), b60 (22) |
| Serving modes | 2 | agg (1,079), disagg (1,079) |

**Architectures swept:** DeciLM, DeepSeek-V3, DeepSeek-V3.2, DeepSeek-V4,
Gemma4, GLM-MoE-DSA, GPT-OSS, Kimi-K2.5, Llama, Llama4, MiMo, MiMo-V2-Flash,
MiniMax-M2, NemotronH, Qwen3, Qwen3-MoE, Qwen3-VL, Qwen3-VL-MoE, Qwen3.5,
Qwen3.5-MoE.

## 3. Methodology

Two layers, both comparing the **Rust** engine-step (`engine_step_backend="rust"`)
against the **Python** engine-step on the identical task config.

### Probe layer (fast, regression net)
- One single-point `cli_estimate` per entry: `isl=256, osl=256, prefix=128`,
  parallelism by model size class.
- Compares `ttft` and `tpot`. **Pass = rtol ≤ 1%** (atol 1e-3 ms).
- Catches per-op latency divergence cheaply across all 2,158 entries.

### Pareto layer (slow, end-to-end)
- Full `cli_default` agg-vs-disagg sweep per entry; compares the throughput/
  latency Pareto frontier.
- Verdicts: `STRICT_PASS` (per-row rtol ≤ 1%), `ENVELOPE_PASS` (frontier
  rtol ≤ 5% when discrete row-selection differs), `DRIFT`, `REGRESSION`.

Tolerances are baked-in constants in the runner, not CLI flags.

## 4. Probe results — 0 DRIFT, 0 REGRESSION

| Outcome | Count | Meaning |
|---|---|---|
| **PASS** | 1,875 | Rust within 1% of Python (the vast majority bit-identical) |
| **BOTH_ERROR_PASS** | 280 | Python and Rust both raise the *same* error (missing data / OOM) — symmetric, not a parity gap |
| **PY_ERROR_ONLY** | 3 | Python raises, Rust succeeds — a Python-side data-availability gap (§5) |
| **DRIFT** | **0** | — |
| **REGRESSION** | **0** | — |

Among the 1,875 PASS entries the agreement is far tighter than the 1% gate
(final build `048c3a7f`):

- Max absolute drift: **0.41% ttft / 0.18% tpot**.
- Mean absolute drift: **0.003% ttft / 0.005% tpot**.
- Only 4 entries exceed 0.1% drift (all < 0.5%); the rest are bit-identical to
  the Python engine.

## 5. Parity bugs found and fixed

Six engine bugs surfaced; all are fixed on `simonec/rust-fixes`. Each was
validated by an end-to-end re-scan, not just a module-level test. All are
**pre-existing** in the Rust engine core (reproduce on `main`), not introduced
by this work.

| Commit | Area | Root cause | Fix |
|---|---|---|---|
| `293f2366` | GEMM | Rust `query_two_d` fp8-static scale-table used `inner_only=true`, stricter than Python's clamp → out-of-envelope queries errored | Align bilinear fallback to Python's clamp (`inner_only=false`) |
| `821c9f12` | DSA context | Missing top-k piecewise dispatch + wrong 3-D lookup branch | Port top-k-piecewise + robust-3D batch-scaling |
| `344f79ed` | shared interp | `interp_2d_1d_grid` lacked an exact-hit short-circuit → ragged-grid undercount | Add exact-hit short-circuit |
| `109d7c48` | DeepSeek-V4 | (1) head slice selected by `native_heads` + tp axis instead of the rank-local head resolved against CSV keys; (2) generation used smooth grid interpolation instead of Python's ragged batch-scaling | Resolve local head key, collapse the tp axis, `robust_lookup_batch_{inner,outer}` |
| `49751d1e` | mixed-step | Pass-3 queried `generation_attention` at `decode_batch.max(1)` even with **no decode requests** (prefill-only step) → spurious batch-1 term; Python guards `if gen_tokens>0` | Skip pass-3 when `decode_batch == 0` |
| `b195bbfd` | generation attention | Single in-grid `interp_2d_1d_grid_strict` over `[n][s][b]` diverged from Python's `interp_3d(n,b,s)` at ragged/extrapolation corners (large decode batch × long kv) | Rebuild grid `[n][b][s]`, densify at load (`extrapolate_data_grid`), clamp→densify→clamp, 5-sample s-averaging |

### Methodology note — the probe masked corner bugs

Bugs `49751d1e` (low-batch prefill) and `b195bbfd` (large-batch-×-long-kv decode)
were **invisible to the single-point probe** (`bs=16, isl/osl=256`), which sits
in a low-drift pocket. They surfaced only in the Pareto layer (frontier-shape
DRIFTs) and were localized with a one-time multi-config drift sweep, which
dropped from **37 → 4** `>1%` configs after both fixes. Lesson: a one-config
probe gives false "0 DRIFT" confidence; the
Pareto sweep + drift map are the real corner coverage.

### DeepSeek-V4 detail (the hard one)

Two coupled divergences, the first *masking* the second in the net probe drift:

- **Head-key selection.** The model passes the op a rank-local head count
  (`num_heads = native // tp`). Python resolves the data slice by that local
  count against the CSV head keys `{64, 128}` and **ignores the CSV `tp_size`
  column** (its loaders keep the last row per cell = the max-tp measurement).
  Rust selected by `native_heads` and used `tp_size` as an interpolation axis,
  landing on the wrong (sparse) slice.
- **Ragged generation lookup.** The generation table is ragged (e.g.
  `s_total=385` is measured only at `batch=2`). Python's robust lookup scales
  the largest measured `bp ≤ query_b` by `query_b / bp`; Rust smoothly
  interpolated the batch axis instead, under-counting decode attention in the
  mixed step. The same fix cleared DeepSeek-V4-Flash agg as well — its drift was
  purely this ragged path (Flash has no head divergence; disagg passed because
  it skips the mixed-step overlap).

After the fix, DeepSeek-V4-Pro agg/disagg and Flash agg are **bit-identical** to
the Python engine (ttft/tpot within 0.001%). Regression test:
`dsv4_pro_head_resolution_and_ragged_generation`.

## 6. Pareto results

The full `cli_default` agg-vs-disagg Pareto comparison (cloud, commit
`fe6bdcd7` = the GEMM/DSA/interp/V4 fixes, **before** `49751d1e`/`b195bbfd`):

| Outcome | Count | Meaning |
|---|---|---|
| **STRICT_PASS** | 2,021 | Per-row frontier within 1% |
| **ENVELOPE_PASS** | 9 | Frontier within the 5% envelope (discrete row-selection differs) |
| **DRIFT** | 4 | Frontier-shape difference (§ below) |
| **ERROR** | 124 | 120 symmetric (both engines error identically) + 4 Python-only (Rust succeeds, §7) |
| **REGRESSION** | **0** | — |

99.8% of non-error entries pass (STRICT or ENVELOPE). The `49751d1e`/`b195bbfd`
fixes only tighten Rust→Python parity (they reproduce Python's reference more
faithfully), so they cannot turn a STRICT_PASS into a DRIFT; a fresh rescan on
the final build is the confirming gate (§8).

### The 4 DRIFT entries — frontier-extent differences, not engine error

| Entry (all disagg) | probe drift | reqlat-curve overlap | envelope-extreme Δ |
|---|---|---|---|
| `Qwen/Qwen3-30B-A3B` · gb200 · vllm 0.19.0 | 0.0% | mean **0.09%** / max 0.36% | 19% (peak-throughput endpoint) |
| `moonshotai/Kimi-K2.5` · h200_sxm · vllm 0.14.0 | 0.0% | mean **0.96%** / max 6.47% | 6% (min-latency endpoint; tpot & peak-tput identical) |
| `moonshotai/Kimi-K2.5` · h200_sxm · vllm 0.19.0 | 0.0% | mean **1.23%** / max 6.88% | 6% (min-latency endpoint; tpot & peak-tput identical) |
| `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` · gb300 · sglang 0.5.10 | 0.0% | mean **0.15%** / max 0.97% | 7% (extreme endpoints; curve <1%) |

A one-time drift classifier compared the two engines'
user-facing frontier **curves** (request_latency vs tokens/s/user) and the
**envelope extremes** the comparator checks. The result resolves the ambiguity:

- **The frontier curves agree** wherever both engines have operating points —
  mean gap **0.09–1.23%** (≤6.9% at the very edges). No per-config engine error
  on the bulk of the frontier.
- **The DRIFT flag comes from frontier *extent*** — the two frontiers terminate
  at slightly different endpoint operating points (e.g. Qwen: Rust's peak
  throughput is 19% lower; Kimi: Rust's min-latency floor is 6% *lower*, i.e.
  Rust "better"; Kimi's tpot & peak-throughput extremes are bit-identical). The
  differences go **both directions** → **no systematic regression**.
- These endpoints live in the **high-throughput saturation corner** — exactly
  the region documented in `CLAUDE.md` known-issue #2 ("results can be overly
  optimistic in the low-speed, high-throughput region").

**Verdict:** the 4 DRIFT are frontier-extent/discreteness in the saturation
corner, not engine regressions. The engine is materially correct (curves within
~1%). They are documented, not "fixed," because there is nothing per-config to
fix.

## 7. Known non-blocking observations

- **Python-only errors (Rust succeeds)** — 3 at the probe layer
  (`DeepSeek-V3.2`, `GLM-5-NVFP4`, `GLM-5-FP8` on `b200_sxm/sglang/0.5.10` agg),
  4 at the Pareto layer (the same three plus the base `GLM-5`). Python raises
  *"Context DSA module data not available"*; the Rust engine resolves the same
  query. This is a Python-side data-availability gap, not a Rust parity bug —
  Rust is the more-robust side. Tracked separately from this gate.
- **280 × `BOTH_ERROR_PASS`** — both engines raise the identical error (missing
  perf data for an uncollected combo, or model-does-not-fit OOM). Symmetric by
  construction; counts as parity.
- **DeepSeek-V4 head-key quirk (documented, intentionally mirrored).** Python's
  DSV4 head resolver assumes the CSV `num_heads` column is the rank-local head
  count, but the collected data is the model's *total* head count swept over
  `tp_size` (latency falls as `tp` rises). The result is that a 128-total-head
  model can be served from a 64-local-head data slice. The Rust engine
  faithfully reproduces this to pass the parity gate; because Phase 2 deletes
  the Python path, the surviving Rust engine inherits the quirk. Flagged here so
  a future data re-collection (head axis = local, with a real `tp` axis) can
  retire it deliberately rather than by accident.

## 8. Known limitations (documented, not fixed)

These are **pre-existing**, **bounded**, and live in the high-throughput corner
already caveated by `CLAUDE.md` known-issue #2. They were deliberately *not*
fixed in this PR because the fixes touch the highest-blast-radius shared paths
(serving every passing entry) for a sub-2% gain — a bad trade. Fast-follow.

- **GEMM large-`n` extrapolation (large-vocab logits).** The GEMM table maxes at
  `n=65536`; the logits/vocab GEMM queries `n≈128k`, so it extrapolates. Rust vs
  Python differ ~30% *on that op*, which dilutes to ~1% tpot only on small dense
  models at low batch (e.g. Llama-3.1-8B bs=16/tp=1); larger models clean. Fix
  would touch the shared GEMM query path (`gemm.rs`).
- **Context-attention ragged grid.** Generation attention was ported to Python's
  densify + s-averaging semantics (`b195bbfd`); **context** attention was not.
  It shows as a ~1.5% prefill-ttft divergence on some disagg frontier configs
  (e.g. Qwen3-30B-A3B). Same `extrapolate_data_grid` fix pattern would apply.
- **4 Pareto frontier-extent DRIFTs** (§6) — endpoint differences in the
  saturation corner; curves agree <1% mean; no regression.

## 9. Gate status

The acceptance criterion is **"0 REGRESSION + every DRIFT explained"**, not
"0 DRIFT". The latter is unachievable by engine fixes: Nemotron-3-Nano is
bit-identical per-config (<0.08%) yet still flags a frontier-shape DRIFT — the
comparator is sensitive to which discrete near-tied operating points each
frontier selects, which no amount of engine accuracy removes.

| Gate | Status |
|---|---|
| Probe parity (2,158 entries, rtol ≤ 1%) | ✅ 0 DRIFT / 0 REGRESSION |
| Discovered engine bugs fixed | ✅ 6 fixes landed |
| Pareto: 0 REGRESSION | ✅ |
| Pareto: every DRIFT explained | ✅ 4 = frontier-extent/discreteness (§6), 2 known-limits (§8) |

**Confirming rescan — DONE, gate closed (2026-06-16, commit `048c3a7f`).** The
Pareto numbers above predated `49751d1e`/`b195bbfd`; the confirming run on the
final build has now completed and reproduces them exactly. Rather than rescan
only the non-`STRICT_PASS` rows, the **full** matrix was re-run on the final
build for a clean single-commit certification (the cloud host has no memory
pressure):

- **Probe (2,158):** 1875 PASS / 280 BOTH_ERROR_PASS / 3 PY_ERROR_ONLY —
  **0 DRIFT / 0 REGRESSION** (0 drift/reg in every one of the 10 systems). The
  bs=1 prefill-ttft fix (`49751d1e`) was separately verified: the pre-fix
  +3.6–12% bs=1 ttft gap across all backends is now 0.000%.
- **Pareto (2,158):** 2021 STRICT_PASS / 9 ENVELOPE_PASS / 124 ERROR /
  **4 DRIFT / 0 REGRESSION** — identical verdict counts to the pre-fix baseline.
  Regression-checked rigorously: **0 rows where Python passed and Rust did not**,
  in any bucket. The 4 DRIFT are exactly the frontier-extent entries in §6; the
  124 ERROR are 120 symmetric both-engine + 4 Python-only (Rust the robust side).

The drift map (37→4) and the per-entry classifier provided the engine-level
evidence; this rescan is the end-to-end confirmation. **Phase 2 Python-deletion
may proceed.**
