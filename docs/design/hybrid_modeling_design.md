# HYBRID modeling: util-space empirical + transfer policy

Design for the data-calibrated HYBRID / EMPIRICAL prediction path. Conclusions only —
not the exploration that produced them.

## Problem

A silicon lookup interpolates collected data and is accurate where data exists. When it
cannot serve a shape / quant / op / version, HYBRID predicts the gap; the request fails
only if that empirical/transfer path also has no coverage. The old HYBRID used a fixed
per-op `scale_factor` constant (`latency = SOL / k`), which has a large, unbounded error
tail.

## Model: util-space empirical

Every op's empirical estimate is

```
latency = SOL(query) / util ,   util = SOL / measured ∈ (0, 1]
```

`util` (the achieved kernel efficiency) is read best-effort from collected data in
per-axis normalised log space. One-dimensional curves use the two bracketing samples
(`k=2`, `p=1` inverse-distance weighting), with exact hits preserved and out-of-range
queries clamped to boundary util. Generic multi-dimensional grids retain nearest-
neighbour lookup because their axes may mix numeric coordinates with categorical/kernel
regimes. Operations can explicitly opt into the ragged 2-D bracket helper when they know
the axes: GenerationAttention interpolates exact-head raw `(batch, sequence)` curves in
physical coordinates; Context DSA uses log coordinates and admits only batch curves whose
original sequence range covers the query. Neither path requires a Cartesian product, and
both fall back to the existing NN/transfer chain on a miss. SOL is analytic and
quant-aware, so it carries the structural part of a query (sizes, quant coefficients)
while `util` carries only efficiency. Because the SOL ratio cancels, the reconstruction
is largely insensitive to absolute SOL error — it removes the constant-fallback tail
(GEMM leave-one-out mean MAPE 64–341% → 7–15%).

Implementation: `sdk/operations/util_empirical.py` (`UtilGrid`, `build_samples`,
`grid_for`, `estimate`). A single scalar `util` per sample — **not** a per-component
(compute / mem) split (see Decisions).

## Honest gaps

When no calibration data is available for a slice (no own-shape and no transfer
reference), `estimate()` raises `EmpiricalNotImplementedError` instead of fabricating a
`SOL / constant`. Coverage gaps surface instead of silently producing a wrong number.
Genuinely table-less ops (mem / p2p / element-wise) keep their analytic formulas and
never call `estimate()`.

## Transfer kinds + policy

When an op's own slice has no data, HYBRID borrows utilisation from a neighbour. The
ways to borrow are first-class (`common.TransferKind`) and each is independently enabled
by a **transfer policy** (`PerfDatabase.transfer_policy`). Ordered by decreasing
confidence:

| kind | borrows from | mechanism |
|---|---|---|
| `xshape` | nearest collected shape, **same quant** | NN on shape features (MoE topk/experts/hidden/inter; attention head_size); query's own SOL |
| `xquant` | a sibling quant in the **same** `(memory, compute)` profile | same SOL coefficients → util transfers directly |
| `xprofile` | a quant in a **different** profile | borrow + rescale util by the per-quant level ratio `e(query)/e(ref)` |
| `xop` | a **related op's** util (e.g. MSA ← DSA) | `util_scale` level-alignment hook, manual `k` |
| `xversion` | a **sibling backend version's** measured data (the shared layer) | load-time data sourcing (`enable_shared_layer`) |

A query falls through the ladder (own data → `xshape` → `xquant` → `xprofile`), using the
first that has data; a disabled kind is skipped → next, or raise.

`transfer_policy` is a frozenset of enabled kinds with presets (`off` / `conservative` /
`balanced` / `aggressive`; default = all on = backward-compatible). It is read at query
time, so a cached DB can be retuned. External surfaces:

- CLI: `--transfer-policy` (default / estimate subcommands)
- YAML / Python: a flat `transfer_policy` field on the v2 `Task`
- support matrix: `AIC_SM_TRANSFERS` env

## Key decisions

- **Single scalar `util`, not a compute/mem SOL split.** The split helps only when the
  binding roofline bound differs between source and target. It essentially never does:
  NN returns same-regime samples, and a quant's compute and memory factors are correlated
  so the roofline knee barely moves across quants (0 of 84992 GEMM shapes flip for
  bf16↔fp8). A split was implemented and reverted.
- **Cross-profile = per-quant efficiency level.** Cross-profile error is ~pure systematic
  kernel-efficiency bias, not a regime mismatch. A single per-quant level `e(q)` (median
  achieved util, data-derived where collected, inferred otherwise) corrects it:
  `util(query) ≈ util(ref) · e(query)/e(ref)` (cross-profile LOO 58% → 24%). Ratios are
  ~stack-stable (e.g. w4a16/fp8 = 0.17 on b200, 0.18 on h100).
- **Windowed (sliding-window) attention.** Prefer the real windowed slice. When it is
  absent or too sparse to interpolate, derive from the `window=0` (full-attention)
  measurement scaled by the window-aware SOL ratio — a bounded fallback (~25–50% vs
  measured). SOL already caps the score region / decode KV length at the window, so the
  physical invariant *windowed ≤ full* holds by construction.
- **Cross-op via `util_scale`.** A manual level-alignment scale `k` pulls a borrowed op's
  util to the target op's level (e.g. MSA runs ~1× DSA, MHA↔MLA ~1.4×). Not auto-calibrated
  by design — a single injection point.

## Provenance (observability)

The transfer kind that produced a value is captured (`capture_provenance()` wraps a run;
`estimate()` notes its tier at the single chokepoint, ops tag the specific kind). The worst
(least-confident) tier per PASS is recorded in a `Source` column. Predictions are unchanged —
this is purely observability.

### Support matrix: silicon-first with hybrid rescue (default)

The support matrix runs each cell in **pure SILICON first** (shared layer off) and re-runs
**only the genuine FAILs** in HYBRID. A `FAIL → PASS` transition is therefore unambiguously a
hybrid rescue, which lets the `Source` be derived from the two-pass outcome **without any
per-row provenance plumbing**:

| silicon | hybrid | `Source` |
|---|---|---|
| PASS | — | `silicon` |
| FAIL | PASS, an empirical tier fired | that tier (`xshape` / `xquant` / `xprofile` / `xop`) |
| FAIL | PASS, no empirical tier | `xversion` (sibling-version shared layer) |
| FAIL | FAIL | FAIL |
| HW / FRAMEWORK incompatible | — | unchanged (not rescuable, not retried) |

The `xversion` derivation is the key win: because the silicon pass runs with the shared layer
**off**, a silicon-fail / hybrid-pass with no empirical tier can only be the shared layer, so
shared-layer rescues are attributed correctly (a single all-HYBRID pass cannot distinguish a
sibling-row hit from a primary-row hit and mislabels them `silicon`). Runtime is ~unchanged
since only the minority of FAILs get a second pass. Set `AIC_SM_ALLOW_HYBRID=0` for a
pure-silicon matrix (no rescue).

## Scope

util-space empirical is wired for GEMM; context / generation attention (+ cross-head_size,
windowed); MoE (regular, WideEP, low-latency kernel selection); MLA; DSA; DSV4
(MHC / context / generation); MSA; NCCL + custom_allreduce. Sibling-version shared layer
is enabled for EMPIRICAL as well as HYBRID.

## Boundary with SILICON (and the silicon interpolation refactor)

The split of responsibility is by **coverage**, not by axis type:

- **SILICON stays pure.** It uses collected rows for same-slice interpolation and any
  explicitly supported numeric boundary handling. A missing shape / window / quant /
  op is a failed silicon lookup handed to HYBRID; SILICON does not borrow another slice.
- **HYBRID / `util_empirical` owns all gap-filling.** Every transfer — `xshape`,
  windowed, `xquant`, `xprofile`, `xop`, `xversion` — lives here and stays here. None of
  it migrates into SILICON. The request raises only after these permitted fallback tiers
  also fail to find calibration data.

A parallel refactor makes the silicon interpolation engine able to interpolate in
*latency-space or util-space* and pick the better per point. That improves **in-grid
accuracy**; it does **not** absorb the HYBRID transfers into the silicon lookup. The
only thing the two share is the util definition: `util = SOL /
measured` with `SOL = max(sol_math, sol_mem)` (single scalar — a compute/mem split was
tried and rejected) and the same per-op `get_sol`. Keeping that definition consistent on
both sides is the entire contract; no code coupling is required.

## Validation

`tools/accuracy_regression_testing/validate_empirical_fidelity.py` runs the same model and
workload independently in explicit SILICON and EMPIRICAL modes. Results below use strict
own-data EMPIRICAL (`--transfer-policy off`), so no HYBRID or transfer fallback is included.
APE is `abs(empirical - silicon) / silicon`; each cell is **mean / p90 / max APE (%)** with
the number of comparable points in parentheses. Unless marked, a row contains ten
deterministic irregular off-grid points per phase on B200/SGLang 0.5.10.

Across every phase and sample kind in the primary matrix, 518/546 silicon-eligible pairs
are comparable: **1.47% mean, 0.85% median, 3.67% p90, 8.68% max, and 1.04% WAPE**.

| model / configuration | type | quantization | prefill | decode | encoder |
|---|---|---|---:|---:|---:|
| **Primary off-grid aggregate** | mixed | mixed | **0.92 / 1.98 / 7.05 (n=185)** | **2.37 / 4.88 / 8.68 (n=170)** | — |
| Qwen3-32B | dense | BF16 | 0.73 / 1.34 / 1.53 (n=10) | 1.03 / 1.52 / 3.59 (n=10) | — |
| Qwen3-32B | dense | FP8 | 0.99 / 1.30 / 4.30 (n=10) | 2.28 / 4.69 / 6.82 (n=10) | — |
| Qwen3-32B | dense | FP8-static | 0.56 / 1.18 / 1.37 (n=10) | 2.23 / 2.63 / 3.37 (n=10) | — |
| Llama-3.1-70B | dense | FP8-static | 0.71 / 1.02 / 1.10 (n=10) | 3.39 / 3.74 / 3.86 (n=10) | — |
| Qwen3.5-27B | dense/GDN | BF16 | 0.39 / 0.53 / 1.70 (n=10) | 1.79 / 2.89 / 3.02 (n=10) | — |
| Qwen3-235B-A22B | MoE | BF16 | 0.68 / 0.40 / 4.19 (n=10) | 2.42 / 3.38 / 3.47 (n=10) | — |
| Qwen3-235B-A22B | MoE | FP8 | 0.73 / 0.82 / 3.59 (n=10) | 2.90 / 3.82 / 3.85 (n=10) | — |
| Qwen3-235B-A22B | MoE | NVFP4 | 0.26 / 0.52 / 0.67 (n=10) | 4.15 / 5.89 / 5.96 (n=10) | — |
| MiniMax-M2.5 | MoE | FP8-block | 0.24 / 0.30 / 1.07 (n=10) | 2.60 / 3.93 / 4.24 (n=10) | — |
| MiniMax-M2.5 | MoE | NVFP4 | 0.48 / 1.46 / 1.79 (n=10) | 3.67 / 6.52 / 7.04 (n=10) | — |
| Nemotron-Nano | hybrid/MoE | BF16 | 0.15 / 0.34 / 0.34 (n=10) | 3.43 / 5.53 / 7.28 (n=10) | — |
| Kimi-K2.5 | MLA/MoE | BF16 + INT4-WO | — | — | — |
| DeepSeek-V3.2 (TP2/PP2) | DSA | FP8-block | 2.62 / 5.91 / 5.91 (n=5) | 0.51 / 0.82 / 0.98 (n=10) | — |
| DeepSeek-V3.2 (TP8) | DSA | FP8-block | 2.00 / 6.41 / 7.05 (n=10) | 0.74 / 1.36 / 2.33 (n=14)¹ | — |
| GLM-5 (PP2) | DSA | BF16 | 1.30 / 2.21 / 2.25 (n=10) | 0.69 / 1.59 / 1.69 (n=10) | — |
| GLM-5 | DSA | FP8 | 1.35 / 2.10 / 3.40 (n=10) | 2.01 / 3.26 / 3.74 (n=10) | — |
| GLM-5 | DSA | NVFP4 | 2.07 / 2.81 / 3.64 (n=10) | 1.31 / 2.29 / 2.66 (n=10) | — |
| DeepSeek-V4-Pro (PP2) | DSV4 | FP8-block | 0.89 / 1.30 / 1.42 (n=10) | 3.74 / 6.26 / 8.68 (n=10) | — |
| DeepSeek-V4-Pro | DSV4 | FP8 + MXFP4/MXFP8 | 1.12 / 1.32 / 1.98 (n=10) | 2.20 / 4.93 / 5.36 (n=10) | — |
| DeepSeek-V4-Flash | DSV4 | FP8 + MXFP4/MXFP8 | 1.12 / 1.54 / 1.81 (n=10) | — | — |
| Qwen3-VL-32B image | dense/VL | BF16 | 0.58 (n=1)² | 0.29 (n=1)² | 5.17 (n=1)² |
| Qwen3-VL-30B-A3B image | MoE/VL | BF16 | 0.34 (n=1)² | 2.45 (n=1)² | 5.32 (n=1)² |
| **Cross-stack off-grid aggregate**³ | mixed | mixed | **0.89 / 1.50 / 9.87 (n=64)** | **1.07 / 1.83 / 3.25 (n=64)** | — |

1. The TP8 decode row uses all 14 boundary-util extrapolation probes; its ten irregular
   probes have 0.84% mean APE with the same 1.36% p90 and 2.33% max.
2. VL image values are one-point smoke checks, not distributions.
3. The separate stack sweep covers B200/H100/B300 and SGLang/TRT-LLM/vLLM. Its 9.87%
   maximum is the DSA `index_topk` transition; the B300/TRT-LLM MoE case has no SILICON
   reference and is excluded.

The primary off-grid coverage is 185/195 (94.9%) for prefill and 170/180 (94.4%) for
decode. Five DeepSeek-V3.2 TP2/PP2 prefill probes are OOM. Kimi-K2.5 is a strict
EMPIRICAL own-data gap, while DeepSeek-V4-Flash decode has no SILICON reference; neither
is included in MAPE.
