<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Rust↔Python Engine-Step Parity Coverage

**Status as of 2026-06-15.** Probe layer: **complete, 0 DRIFT / 0 REGRESSION**.
Pareto layer: **running** (cloud both-rescan in progress) — see §6.

## 1. Why this report

Phase 2 flips the Rust engine-step to the default latency path and then deletes
the duplicated Python latency code (the per-op `query()` walk in
`base_backend`, the `operations/*.py` latency queries, and the
`perf_database` latency lookups). That deletion is gated on proving the Rust
engine reproduces the Python engine's latency across the **entire** supported
model × system × backend × version matrix.

This report is that gate's evidence: the coverage swept, the methodology, the
results, and every parity bug found and fixed along the way.

- Plan: [`rust/aiconfigurator-core/docs/phase-2-python-dedup-plan.md`](../rust/aiconfigurator-core/docs/phase-2-python-dedup-plan.md)
- Scan runbook: [`rust/aiconfigurator-core/docs/phase-2-parity-scan-runbook.md`](../rust/aiconfigurator-core/docs/phase-2-parity-scan-runbook.md)
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

Among the 1,875 PASS entries the agreement is far tighter than the 1% gate:

- Max absolute drift: **0.41% ttft / 0.45% tpot**.
- Mean absolute drift: **0.002% ttft / 0.008% tpot**.
- Only 33 entries exceed 0.1% drift (all still < 0.5%); the rest are
  bit-identical to the Python engine.

## 5. Parity bugs found and fixed

Four engine bugs surfaced during the scan; all are fixed on
`rust-parity/cloud-scan-runbook`. Each was validated by an end-to-end probe
re-scan, not just a module-level test.

| Commit | Area | Root cause | Fix |
|---|---|---|---|
| `aff78394` | GEMM | Rust `query_two_d` fp8-static scale-table used `inner_only=true`, stricter than Python's clamp → out-of-envelope queries errored | Align bilinear fallback to Python's clamp (`inner_only=false`) |
| `04191715` | DSA context | Missing top-k piecewise dispatch + wrong 3-D lookup branch | Port top-k-piecewise + robust-3D batch-scaling |
| `3ec52ed7` | shared interp | `interp_2d_1d_grid` lacked an exact-hit short-circuit → ragged-grid undercount | Add exact-hit short-circuit |
| `fe6bdcd7` | DeepSeek-V4 | (1) head slice selected by `native_heads` + tp axis instead of the rank-local head resolved against CSV keys; (2) generation used smooth grid interpolation instead of Python's ragged batch-scaling | Resolve local head key, collapse the tp axis, `robust_lookup_batch_{inner,outer}` |

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

## 6. Pareto layer — in progress

The cloud both-rescan is running the full `cli_default` Pareto comparison. This
report will be finalized when it completes. The probe layer (above) already
proves per-op latency parity; the Pareto layer additionally exercises the
config-search and frontier-selection logic end-to-end.

**Expected, not a regression — comparator discreteness.** A small set of
entries (across Kimi-K2.5, Nemotron-3-Nano, Qwen3-30B-A3B) showed probe drift
≈ 0 but a Pareto-only DRIFT in an earlier run. This is the comparator picking a different
*discrete* frontier point when two configs are near-tied — the engine latencies
agree; the selection is on a knife-edge. These are expected to land as
`ENVELOPE_PASS` (frontier within 5%) and are **not** engine bugs; the engine is
not changed for them.

## 7. Known non-blocking observations

- **3 × `PY_ERROR_ONLY`** — `DeepSeek-V3.2`, `GLM-5-NVFP4`, `GLM-5-FP8` on
  `b200_sxm/sglang/0.5.10` agg. Python raises *"Context DSA module data not
  available"*; the Rust engine resolves the same query. This is a Python-side
  data-availability gap, not a Rust parity bug — Rust is the more-robust side.
  Tracked separately from this gate.
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

## 8. Gate status

| Gate | Status |
|---|---|
| Probe parity (all 2,158 entries, rtol ≤ 1%) | ✅ 0 DRIFT / 0 REGRESSION |
| All discovered engine bugs fixed | ✅ 4 fixes landed |
| Pareto parity (end-to-end frontier) | ⏳ cloud rescan in progress |

Phase 2's Python-deletion step proceeds once the Pareto layer lands green
(modulo the documented comparator-discreteness entries).
