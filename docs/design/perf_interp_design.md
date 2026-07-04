# perf_interp — the unified perf-table interpolation engine

Status: shipped with PR #1303. This document covers the design, the
leave-one-out accuracy evidence gathered during the migration, and the
legacy defensive machinery the redesign made unnecessary.

## 1. Problem

Every op in the SDK (GEMM, attention, MLA, DSA/CSA, MoE, Mamba, comm, ...)
answers SILICON queries from a collected perf table: a nested dict
`data[x][y]...[axis_n] -> {latency, power, energy}`. Three realities shaped
the old code and motivated the redesign:

1. **Tables are ragged, not Cartesian.** Real collection is corner-truncated
   (large seq x large batch is skipped for cost/OOM: attention sub-grids are
   ~75% filled with a staircase frontier), and GEMM's `(n, k)` are *scattered
   real matmul shapes* — a dense m-sweep collected at one model's `(n, k)`
   forms no product with any other shape (the reverse `(k, n)` usually does
   not exist).
2. **Queries extrapolate.** Deployments ask for token counts, sequence
   lengths and batch sizes beyond what was benched.
3. **Physics is known per op.** Every op has an analytic speed-of-light
   `SOL(coords) = max(compute, memory)` roofline, and its utilization
   `util = SOL/latency` is a smooth, bounded quantity.

The legacy answer was *load-time grid pre-expansion*
(`extrapolate_data_grid`): densify every table into a rectangle at load, so
query-time interpolation only ever saw complete grids. That decision is the
root of most of the defensive code catalogued in section 5: rectangularizing
scattered shapes mangles them, the synthesized points polluted lookups
(forcing raw-vs-expanded double bookkeeping), and axis-by-axis expansion
produced order-dependent interpolation-of-interpolation.

## 2. Design

One shared N-axis resolver (`sdk/perf_interp/engine.py`) driven by a
declarative per-op record (`sdk/perf_interp/config.py`). Adding an op is
adding a config, not a code path.

### Resolution chain (the whole engine in four steps)

```
1. exact hit             -> return the measured leaf verbatim
2. resolve in the data   -> Grid: nested bracket+blend (value-transform space)
                            ScatteredSites: site curve eval; unknown site ->
                            nearest-site transfer in util space
3. beyond the range      -> hold the boundary util (k_tail-median anchor),
                            latency = SOL(query) / util
4. nothing to anchor on  -> raise (structured miss; never fabricate)
```

### Two resolvers — the only two table shapes that exist

| Resolver | Table shape | Handling |
|---|---|---|
| `ScatteredSites(site_axes, curve_axis)` | GEMM: scattered `(n,k)` sites, each owning an m-curve | exact site answers from its **own** curve; an uncollected shape borrows **util** from the nearest collected shapes (log-space IDW, filtered to sites whose curve covers the query m, 2.0-octave miss gate) |
| `Grid()` | everything else, 1..N axes | per level: exact key collapses; otherwise bracket + blend; a ragged branch is dropped and weight renormalized; past the staircase frontier = ordinary out-of-range util-hold |

The engine is N-axis: context DSA/CSA carries a past-KV axis
(`[heads][prefix][seq][batch]`), and the same op can be 4-axis in new
collections but 3-axis in legacy files — the shape is detected **per
(quant, arch, backend) slice at query time**, never assumed globally.

### Invariants (deliberately not knobs)

- **Measured points are returned exactly.** The engine never smooths
  collected data — GEMM-m wave-quantization sawteeth survive iff collected,
  which is the collector's job, not the interpolator's.
- **Cross-site transfer and extrapolation are always in util space.** A
  neighbouring shape's *latency* means nothing at the query shape, but its
  *efficiency* transfers, and `SOL(query)` re-applies the exact scaling.
  Extrapolation is unbounded by design: outside the data, held efficiency +
  analytic SOL is the only honest signal at any distance.
- **Energy rides as average power** (`energy/latency`, smooth and bounded),
  blended with the same weights, then re-multiplied.

### The one knob: per-axis value transform

`value_transform` (`raw | sqrt | util`) is the space used for interpolation
*between* measured points, and it applies **per axis** (`transform_axis`) —
curvature is a property of the axis, not the table. Context-type attention is
~seq^2 along seq but ~linear along batch/heads; the harness measured global
sqrt at 9.4% median interior error vs 2.0% with sqrt confined to the seq axis,
while on the seq axis itself sqrt beats raw 3.2% vs 6.0%. (The legacy
pre-expansion encoded the same fact as `sqrt_y_value=True`.)

Recipe: **sqrt-on-seq for context-type ops (~seq^2); RAW everywhere else** —
DSA/CSA measured raw-linear per axis (their top-k saturation knee is itself
collected, so brackets never straddle it far). `util` as an *interpolation*
space is available but off by default: between two bracketing anchors the SOL
roughly cancels, and its `max(compute, mem)` ridge kink can hurt.

## 3. Validation philosophy

Accuracy is judged by **leave-one-out against measured latency**: hold out a
collected point (or an entire `(n,k)` site, or a seq row, or the boundary
shell), predict it from the rest, compare. Deviation from the *previous
system's* prediction is a sanity signal only — the old prediction is not
ground truth; single-digit % deviation is expected (interpolation is
inherently a few-percent instrument), and only 20-30%+ warrants
investigation. Every knob above had to win its LOO A/B or be deleted.

## 4. Accuracy evidence (real tables, leave-one-out)

Data: h100_sxm sglang 0.5.10 / vllm 0.19.0, gb200 sglang 0.5.10.

| Fold | median | p90 |
|---|---|---|
| GEMM interior (300 folds) | 4.87% | 15.6% |
| GEMM frontier (util-hold extrapolation) | **0.96%** | 8.4% |
| GEMM **site-holdout** (drop an entire `(n,k)` shape, predict from neighbours) | **3.63%** | 14.7% (0 misses) |
| context attention interior (batch-axis blends) | 2.00% | 9.7% |
| context attention seq-row holdout: sqrt vs raw | **3.21% vs 5.99%** | — |
| context attention global-sqrt vs per-axis sqrt (interior) | 9.44% -> **2.00%** | — |
| context MLA interior / frontier | 2.19% / 5.47% | 10.0% / 45% |
| context DSA, **4-axis engine path** interior / frontier | **0.84% / 2.45%** | 10.9% / 14.1% (0 misses) |
| CSA (gb200) plain crossing vs regime-aware | **1.72% vs 1.92%** | 4.3% / 4.3% |
| CSA knee-just-above, signed | plain **+0.57%** vs regime −2.94% | — |

Notable robustness result: 1705 ragged queries across four op families on the
raw (un-expanded) tables produced **zero crashes and zero Qhull errors** —
the corner-truncated frontier is fully absorbed by util-hold.

Known weakness (tracked): frontier *tails* are fat (p90 up to ~45-49% for
attention/MLA min-side edges) — extrapolating *below* the smallest collected
sizes is overhead-dominated and the SOL ratio overshoots. Median behaviour is
single-digit everywhere.

## 5. Defensive machinery this design retires

The pattern across all of these: a workaround for load-time pre-expansion, or
a hand-rolled copy of util-hold, or protection against an interpolation
method (cubic/scattered griddata) that the engine no longer uses.

### 5.1 The top-k regime-aware piecewise (the exemplar)

`interp_dsa_context_topk_piecewise_from_raw` (PR #903) restricted DSA context
interpolation to *same-regime* anchors around the `index_topk` boundary, and
required a pristine *raw* table to do it. Archaeology showed its two
motivations precisely:

1. the then-**cubic** 3-D interpolation overshot/underestimated just above
   the knee — cubic ringing at kinks is a real numerical phenomenon;
2. it had to consult **raw** rows because pre-expansion had polluted the
   working table with synthesized points.

The engine removes both root causes structurally: bracket blends are
**linear** (cannot overshoot a kink, and the knee `s = topk` is itself
collected, so brackets rarely straddle it), and tables are **raw** (no
pre-expansion). Leave-one-out on every real config — DeepseekV3.2 bf16/fp8,
GLM-5, and DSV4 CSA on gb200 — confirmed plain linear crossing ties or beats
the piecewise, including *just above the boundary* (+0.57% vs −2.94%), i.e.
the original symptom cannot be reproduced under the new mechanism. Deleted
with its tests.

### 5.2 Hand-rolled util-holds (three independent copies)

- DSA generation's `boundary_util_value` — freeze boundary util, scale by
  SOL ratio;
- MoE's `_estimate_overflow_with_last_token_util` (~65 lines) — same, for
  token overflow;
- `scale_matrix`'s clamp + `SOL(q)/SOL(boundary)` ratio — its own comment
  said "freeze utilization at the clamped boundary".

All three are the engine's step 3 verbatim. Deleted; their call sites are one
engine query each.

### 5.3 Raw-vs-expanded double bookkeeping

DSA, DSV4 and generation attention each kept a `deepcopy` of the loaded
table *before* expansion (`_raw_*_data`) so regime lookups and boundary
anchoring could avoid synthesized points, plus logic to prefer measured rows
over expanded ones ("an exact batch in the working table may itself be a
load-time extrapolation target"). With no pre-expansion the table *is* the
raw data: the copies became aliases and the preference logic was deleted.

### 5.4 Robust-lookup layers

`_dsv4_robust_3d_lookup` / `_dsv4_lookup_prefix_resolved` (exact -> cubic ->
covering-batch interpolation -> largest-lower-batch x batch scaling; per-
prefix slices + clamped 1-D across prefix) existed to survive ragged clouds
that crashed scattered cubic interpolation (QhullError on degenerate axes).
The Grid resolver's bracket + branch-drop + util-hold handles the same
inputs natively; both helpers and their eight white-box tests are gone.

### 5.5 Pre-expansion itself

`extrapolate_data_grid`, its 12 per-op `_extrapolate` methods and 18
`_*_TARGET_*` grid constants (~30k synthesized nodes per quant-mode,
order-dependent interpolation-of-interpolation, and the source of the
"Skipping interpolation" warning spam on ragged inputs) are deleted. Also
gone: raw-linear extrapolations that could undershoot launch-overhead floors
or go negative below the smallest collected size (NCCL/allreduce message
sizes, MoE token underflow, DSV4 decode below min `s_total`).

### 5.6 Contracts deliberately preserved

Not everything old was defensive cruft — some behaviors are deliberate and
test-locked, and survive unchanged at op level:

- `compute_scale` is a quantization-overhead *delta*: beyond its grid it is
  held **flat** at the clamped boundary (not SOL-scaled).
- Mamba's degradation contract: a genuine data miss returns plain SOL
  (`source="sol"`), it does not raise.
- MoE singleton-underflow still raises (a lone 1024-token row cannot define
  the low-token launch floor); HYBRID falls through to the empirical path.
- The CSA top-k DELTA *calibration* (measured collector-vs-representative
  correction) is orthogonal to interpolation and kept verbatim.

## 6. Current state and remaining work

- All op families resolve through `perf_interp`; `interpolation.py`'s
  interp family has exactly one remaining consumer, `_lookup_sparse_kernel`
  (the DSV4 CP kernel-DELTA correction helper, which has no natural SOL).
  Retiring it — and with it the scipy `griddata` dependency — requires either
  a validated proxy SOL for the sparse kernels or acceptance of a behavior
  change in its batch-scaling fallback; parked as an explicit decision.
- The DeepEP `sms` axis has no scaling story (only sm=20 is collected);
  off-grid `sms` snaps to the nearest collected value until data exists.
- Frontier-tail accuracy (min-side extrapolation) is the main quality
  follow-up; medians are single-digit everywhere.
