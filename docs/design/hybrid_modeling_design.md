# HYBRID modeling: util-space empirical + transfer policy

Design for the data-calibrated HYBRID / EMPIRICAL prediction path. Conclusions only —
not the exploration that produced them.

## Problem

SILICON mode interpolates collected data and is accurate where data exists, but raises
for any shape / quant / op / version it never measured. HYBRID must predict those gaps.
The old HYBRID used a fixed per-op `scale_factor` constant (`latency = SOL / k`), which
has a large, unbounded error tail.

## Model: util-space empirical

Every op's empirical estimate is

```
latency = SOL(query) / util ,   util = SOL / measured ∈ (0, 1]
```

`util` (the achieved kernel efficiency) is read best-effort from collected data by
nearest-neighbour lookup in per-axis normalised log space. SOL is analytic and
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
`estimate()` notes its tier at the single chokepoint, ops tag the specific kind). The
support matrix records the worst (least-confident) tier per PASS in a `Source` column and
splits passes into *silicon* vs *hybrid:&lt;tier&gt;*. Predictions are unchanged — this is
purely observability.

## Scope

util-space empirical is wired for GEMM; context / generation attention (+ cross-head_size,
windowed); MoE (regular, WideEP, low-latency kernel selection); MLA; DSA; DSV4
(MHC / context / generation); MSA; NCCL + custom_allreduce. Sibling-version shared layer
is enabled for EMPIRICAL as well as HYBRID.

## Boundary with SILICON (and the silicon interpolation refactor)

The split of responsibility is by **coverage**, not by axis type:

- **SILICON stays pure.** It interpolates only *within* the collected grid and raises
  outside coverage. It does **not** extrapolate across an uncollected shape / window /
  quant / op. This keeps SILICON reproducible and honest.
- **HYBRID / `util_empirical` owns all gap-filling.** Every transfer — `xshape`,
  windowed, `xquant`, `xprofile`, `xop`, `xversion` — lives here and stays here. None of
  it migrates into SILICON.

A parallel refactor makes the silicon interpolation engine able to interpolate in
*latency-space or util-space* and pick the better per point. That improves **in-grid
accuracy**; it does **not** absorb the HYBRID transfers (silicon must not reach an
uncovered slice). The only thing the two share is the util definition: `util = SOL /
measured` with `SOL = max(sol_math, sol_mem)` (single scalar — a compute/mem split was
tried and rejected) and the same per-op `get_sol`. Keeping that definition consistent on
both sides is the entire contract; no code coupling is required.

## Validation

| config | metric | result |
|---|---|---|
| Llama-3.1-70B (h100/trtllm, fp8), run_static | MAPE | 23.6% (constant) → **7.0%** |
| DeepSeek-V3 (h200/trtllm, bf16), run_static | MAPE | **3.63%** |
| GLM-5 gate | median MAPE | **1.47%** |

Per-op silicon-vs-empirical invariant audits (MLA / DSA / dsv4 × trtllm / sglang / vllm)
recover silicon to ≈ 1.0 at collected points.
