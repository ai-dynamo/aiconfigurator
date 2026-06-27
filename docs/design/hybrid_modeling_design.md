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
latency = SOL(query) / util ,   util = SOL / measured > 0
```

`util` is an effective calibration factor, not a bounded physical efficiency. It can
exceed 1 when hardware/kernel effects are not represented by the analytic SOL baseline;
large values are data/model sanity signals and are not silently clamped. It is read
best-effort from the operation's calibration rows in per-axis normalised log space. Every
numeric grid uses the same two-neighbour inverse-distance weighting (`k=2`, `p=1`): exact hits are
preserved, each query axis is clamped to the measured range, and the two nearest samples
are blended without requiring a Cartesian product. Operations select categorical/kernel-
regime slices before building the grid; the shared estimator only sees numeric
coordinates. SOL is analytic and
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
(least-confident) tier per successful run is recorded in a `Source` column.

### Support matrix: silicon-first with hybrid rescue (default)

The support matrix runs each cell in **SILICON first** (including declared shared-layer
collected rows). It retries only structured performance-data misses, plus explicitly
classified framework/data gaps, in HYBRID. Arbitrary programming errors are not retried.
`PASS` is reserved for measured-silicon support; a successful rescue is reported separately
as `HYBRID_PASS`:

| silicon | hybrid | `Status` | `Source` |
|---|---|---|---|
| PASS | — | `PASS` | `silicon` |
| structured/known data gap | PASS, an empirical tier fired | `HYBRID_PASS` | that tier (`xshape` / `xquant` / `xprofile` / `xop`) |
| structured/known data gap | PASS, no empirical tier | `HYBRID_PASS` | `empirical` |
| structured/known data gap | FAIL | original silicon status | empty |
| programming error or HW incompatible | — | original status | empty |

Shared-layer hits remain `silicon`: they are real collected rows merged before lookup, and
the loader does not retain per-row source provenance after the merge. Replay commands use
the database mode that produced the row. Set `AIC_SM_ALLOW_HYBRID=0` for a SILICON-only
matrix (no rescue).

## Scope

util-space empirical is wired for GEMM; context / generation attention (+ cross-head_size,
windowed); MoE (regular, WideEP, low-latency kernel selection); MLA; DSA; DSV4
(MHC / context / generation); MSA; NCCL + custom_allreduce.

## Boundary with SILICON

The split of responsibility is by **coverage**, not by axis type:

- **SILICON stays collected-data-only.** It uses active-version rows plus manifest-declared
  shared rows from sibling versions/frameworks, then applies same-slice interpolation and
  explicitly supported numeric boundary handling. It never uses SOL/util transfer.
- **HYBRID / `util_empirical` owns all gap-filling.** Every transfer — `xshape`,
  windowed, `xquant`, `xprofile`, `xop` — lives here and stays here. The request raises
  only after the collected-data lookup and these permitted fallback tiers
  also fail to find calibration data.

Individual SILICON operations may use latency-space interpolation or explicit
util-preserving boundary handling inside their collected numeric slice. That does not
absorb the HYBRID transfers into SILICON. Where util is used, both paths share the same
definition: `util = SOL / measured`, with `SOL = max(sol_math, sol_mem)` and the same
per-op `get_sol`.

## Validation

The following offline study ran the same model and workload independently in explicit
SILICON and EMPIRICAL modes. Results use strict own-data EMPIRICAL
(`--transfer-policy off`), so no HYBRID or transfer fallback is included.
SILICON follows its production policy and includes manifest-declared shared collected rows;
formula-only EMPIRICAL loads only the active version.
APE is `abs(empirical - silicon) / silicon`; each cell is **mean / p90 / max APE (%)** with
the number of comparable points in parentheses. Unless marked, a row contains ten
deterministic irregular off-grid points per phase on B200/SGLang 0.5.10.

Across every phase and sample kind in the primary matrix, 518/546 silicon-eligible pairs
are comparable: **1.88% mean, 1.05% median, 4.99% p90, 10.18% max, and 1.67% WAPE**.

| model / configuration | type | quantization | prefill | decode | encoder |
|---|---|---|---:|---:|---:|
| **Primary off-grid aggregate** | mixed | mixed | **1.45 / 4.34 / 10.07 (n=185)** | **2.64 / 5.66 / 8.91 (n=170)** | — |
| Qwen3-32B | dense | BF16 | 0.40 / 0.62 / 0.80 (n=10) | 2.35 / 3.39 / 4.49 (n=10) | — |
| Qwen3-32B | dense | FP8 | 0.43 / 0.96 / 1.17 (n=10) | 2.83 / 5.38 / 5.66 (n=10) | — |
| Qwen3-32B | dense | FP8-static | 0.35 / 0.68 / 0.89 (n=10) | 3.13 / 4.72 / 5.64 (n=10) | — |
| Llama-3.1-70B | dense | FP8-static | 0.35 / 0.55 / 1.04 (n=10) | 3.41 / 5.24 / 5.96 (n=10) | — |
| Qwen3.5-27B⁴ | dense/GDN | BF16 | 4.48 / 5.12 / 5.28 (n=10) | 5.43 / 6.57 / 7.24 (n=10) | — |
| Qwen3-235B-A22B | MoE | BF16 | 0.36 / 0.36 / 2.63 (n=10) | 2.92 / 4.24 / 4.99 (n=10) | — |
| Qwen3-235B-A22B | MoE | FP8 | 0.43 / 0.54 / 2.22 (n=10) | 3.76 / 4.84 / 7.31 (n=10) | — |
| Qwen3-235B-A22B | MoE | NVFP4 | 0.34 / 0.56 / 0.58 (n=10) | 4.26 / 6.12 / 6.13 (n=10) | — |
| MiniMax-M2.5 | MoE | FP8-block | 0.29 / 0.41 / 1.29 (n=10) | 1.64 / 2.37 / 3.32 (n=10) | — |
| MiniMax-M2.5 | MoE | NVFP4 | 0.58 / 1.79 / 2.95 (n=10) | 2.36 / 4.00 / 4.13 (n=10) | — |
| Nemotron-Nano | hybrid/MoE | BF16 | 0.09 / 0.18 / 0.34 (n=10) | 3.85 / 6.39 / 7.03 (n=10) | — |
| Kimi-K2.5 | MLA/MoE | BF16 + INT4-WO | — | — | — |
| DeepSeek-V3.2 (TP2/PP2) | DSA | FP8-block | 4.08 / 7.92 / 7.92 (n=5) | 0.39 / 0.87 / 1.02 (n=10) | — |
| DeepSeek-V3.2 (TP8) | DSA | FP8-block | 4.09 / 10.03 / 10.07 (n=10) | 0.65 / 1.40 / 2.24 (n=14)¹ | — |
| GLM-5 (PP2) | DSA | BF16 | 1.77 / 2.47 / 2.56 (n=10) | 0.53 / 1.14 / 1.39 (n=10) | — |
| GLM-5 | DSA | FP8 | 2.89 / 4.93 / 6.54 (n=10) | 0.96 / 2.48 / 2.99 (n=10) | — |
| GLM-5 | DSA | NVFP4 | 4.33 / 6.97 / 9.50 (n=10) | 1.07 / 2.24 / 2.50 (n=10) | — |
| DeepSeek-V4-Pro (PP2) | DSV4 | FP8-block | 1.60 / 3.96 / 4.55 (n=10) | 3.86 / 6.31 / 8.91 (n=10) | — |
| DeepSeek-V4-Pro | DSV4 | FP8 + MXFP4/MXFP8 | 0.85 / 1.29 / 1.64 (n=10) | 2.10 / 5.00 / 5.68 (n=10) | — |
| DeepSeek-V4-Flash | DSV4 | FP8 + MXFP4/MXFP8 | 1.15 / 1.73 / 1.83 (n=10) | — | — |
| Qwen3-VL-32B image | dense/VL | BF16 | 0.81 (n=1)² | 2.79 (n=1)² | 3.31 (n=1)² |
| Qwen3-VL-30B-A3B image | MoE/VL | BF16 | 0.63 (n=1)² | 4.70 (n=1)² | 3.42 (n=1)² |
| **Cross-stack off-grid aggregate**³ | mixed | mixed | **1.02 / 2.33 / 15.72 (n=64)** | **2.14 / 4.82 / 6.63 (n=64)** | — |

1. The TP8 decode row uses all 14 boundary-util extrapolation probes; its ten irregular
   probes have 0.73% mean APE with the same 1.40% p90 and 2.24% max.
2. VL image values are one-point smoke checks, not distributions.
3. The separate stack sweep covers B200/H100/B300 and SGLang/TRT-LLM/vLLM. Its 15.72%
   maximum is the DSA `index_topk` transition; the B300/TRT-LLM MoE case has no SILICON
   reference and is excluded.
4. Qwen3.5's SILICON path reuses sibling-version collected GDN kernels; strict EMPIRICAL
   has no active-version GDN table and uses its analytic fallback.

The primary off-grid coverage is 185/195 (94.9%) for prefill and 170/180 (94.4%) for
decode. Five DeepSeek-V3.2 TP2/PP2 prefill probes are OOM. Kimi-K2.5 is a strict
EMPIRICAL own-data gap, while DeepSeek-V4-Flash decode has no SILICON reference; neither
is included in MAPE.

### Data-quality diagnostic: very large util

An observed util of 14.58 is not treated as a valid efficiency ceiling or silently
clamped. It comes from the B200/SGLang 0.5.10 context-attention row at
`b=8, s=16384, n=64, n_kv=1, head_dim=256, window_size=0`: SOL is 31.275 ms while the
stored latency is 2.145 ms. The same table reports the supposedly full-attention
`window_size=0` case faster than windows 128 and 1024. The collector writes the database
value `0` directly into SGLang's runtime field, whose full-attention sentinel is `-1`, so
this row measured a near-zero left window rather than full attention. This is a bad-data
signal requiring recollection, not a reason to cap util; the collector/data repair is
kept outside this modeling PR.
