<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# DeepEP-LL decode latency modeling

## 1. Scope

This document defines the Stage 1 latency model for **DeepEP low-latency
(DeepEP-LL) decode only**. It does not change DeepEP high-throughput (HT),
DeepEP V2, or TensorRT-LLM communication models.

The modeled decode block is:

```text
FP8 DeepEP-LL dispatch -> standard fused MoE compute -> BF16 DeepEP-LL combine
```

Dispatch and combine use measured DeepEP-LL curves as calibration and then
apply a cached Monte Carlo routing/load model. Expert compute uses the existing
ordinary `MoE` performance table at the model's workload distribution, such as
`power_law_1.01` for DeepSeek or `power_law_1.2` for the default model. Compute
is not multiplied by a second Monte Carlo imbalance factor.

## 2. Token convention

Let:

- \(B_r\): source tokens presented to each rank for one decode step;
- \(P\): number of ranks/GPUs in the MoE expert-parallel group;
- \(K\): Top-K experts selected by each token;
- \(N\): total experts;
- \(H\): hidden size.

`MoEAllToAll` receives **per-rank source tokens**, so its table/Monte Carlo
query uses \(B_r\). The corresponding global source-token count is

\[
B = P B_r,
\]

and the routing matrix contains \(B K=P B_r K\) token-to-expert assignments.
The model does not multiply communication tokens by attention-DP again.

## 3. Logical routes and physical bottlenecks

Routing can be represented by a \(P\times P\) matrix \(M\), where
\(M_{ij}\) is the number of expert assignments sent from source rank \(i\) to
destination rank \(j\). These are \(P^2\) **logical flows**, not \(P^2\)
independent physical channels.

The critical communication resources are the \(P\) GPU endpoints. For one
path (NVLink/MNVL or IB), endpoint \(i\)'s directional loads are

\[
TX_i=\sum_j M_{ij},\qquad RX_i=\sum_j M_{ji}.
\]

The hardware bandwidth values in `SystemSpec` are bytes/s per GPU in one
direction. RDMA, IB, and NVLink can transmit and receive concurrently, so a
full-duplex endpoint is modeled with

\[
L_{endpoint}=\max(\max_i TX_i,\max_i RX_i),
\]

not `send + recv`. This aggregation is why the model sizes \(P\) endpoints
while still sampling the full \(P^2\) traffic matrix.

## 4. Per-assignment payload

DeepEP source behavior fixes different wire payloads for the two LL phases.
The upstream low-latency test calculates exactly these byte counts in
[`tests/legacy/test_low_latency.py`](https://github.com/deepseek-ai/DeepEP/blob/01dc3aaac82068020353dce2c302e38153c0bfaa/tests/legacy/test_low_latency.py#L218-L224),
and its published benchmark setup explicitly identifies FP8 dispatch and BF16
combine in the
[`README`](https://github.com/deepseek-ai/DeepEP/blob/01dc3aaac82068020353dce2c302e38153c0bfaa/README.md#L43-L44).

### 4.1 Dispatch: FP8 plus metadata

For hidden size divisible by 128,

\[
S_d = H + 4\frac{H}{128} + 16\quad\text{bytes}.
\]

The terms are \(H\) FP8 activation bytes, one four-byte scale per 128 values,
and 16 bytes of routing metadata.

### 4.2 Combine: BF16

\[
S_c = 2H\quad\text{bytes}.
\]

The implementation makes the phase dtypes explicit (`fp8` for dispatch and
`bfloat16` for combine). The current DeepEP collector persists both phases as
`comm_dtype="default"`. That legacy key has a phase-specific physical meaning:
`default` is equivalent to FP8 for dispatch and BF16 for combine. It is not a
general wildcard, so an unrelated request such as FP16 or NVFP4 cannot use it.
Resolution prefers an exact typed row, then the `fp8_block -> fp8` alias, and
only then a physically compatible `default` row.

## 5. Calibration lookup

For each phase, calibration walks the full shape under each dtype candidate.
The order is:

1. exact topology plus requested dtype;
2. exact topology plus the `fp8_block -> fp8` alias;
3. exact topology plus a physically compatible legacy `default`;
4. single-domain donor plus requested dtype;
5. single-domain donor plus the alias;
6. single-domain donor plus compatible `default`.

An unavailable shape, invalid OLS curve, or unusable one-point curve advances
to the next candidate. Configuration errors do not. This distinction matters
for mixed-schema tables: the presence of an FP8 row for a different shape
must not hide a valid `default` row for the requested shape.

Within one donor dtype, the physical single-domain EP width is preferred;
remaining node-1 curves are then tried in stable EP order. This makes the
coverage rule (“at least one viable node-1 curve”) identical to runtime even
when the preferred donor curve is present but unusable.

No interpolation or nearest-neighbor substitution is permitted along
\(H\), \(K\), or \(N\). Within the selected curve, an exact token point is
used directly; otherwise the existing token-axis interpolation is used. Both
cases are exact-topology calibration when the curve's \(P\) and node topology
match the request; token interpolation alone does not turn it into a donor.

Calibration provenance has four states: `ExactOls`, `ExactOneShot`,
`DonorOls`, and `DonorOneShot`. One-shot and donor results carry fallback
diagnostics. All four pass through Monte Carlo and therefore remain
`Estimated`, including an exact measured token point.

Legacy DeepEP-LL rows contain `node_num` but no EP axis. Their EP is restored
using the system's actual `num_gpus_per_node`: GB200/GB300 node-1 rows are EP4,
while HGX node-1 rows are EP8. The HT legacy adapter intentionally retains its
historical EP=`node_num * 8` behavior.

## 6. OLS, startup latency, and fitted bandwidth

When the selected curve has at least two points, every token point
\((x_i,y_i)\) participates in an unweighted ordinary least-squares fit:

\[
y=ax+b.
\]

Stage 1 uses

\[
t_0=\max(b,0)
\]

as the fixed startup/kernel overhead and interprets \(a\) as the fitted
per-source-token variable-time slope. A finite negative intercept caused by
measurement noise is clamped to zero. A curve is rejected when it has fewer
than two points for OLS, non-positive slope, zero token variance, or
non-finite output.

A one-point curve may instead borrow a system-level startup estimate. The
pool contains valid multi-point OLS curves from the same database/system,
framework version, `deepep_ll` backend, phase, SMS, and equivalent physical
dtype. Typed and `default` copies of the same physical curve count once. The
borrowed startup is the standard median of their intercepts. For the point
\((B_1,T_1)\), the one-shot slope is

\[
a=\frac{T_1-t_0}{B_1}.
\]

The point is rejected, and resolution continues, when no system startup is
available, \(B_1\leq0\), \(T_1\leq t_0\), or the result is non-finite. This is
a coverage rule as well as a runtime rule: coverage cannot admit a candidate
that runtime cannot calibrate.

For phase payload \(S\), the fitted effective one-direction bandwidth is

\[
\beta_{fit}=\frac{KS}{a\times 10^{-3}},
\]

where \(a\) is in milliseconds per source token. This bandwidth is a
diagnostic interpretation of the OLS slope; the runtime equivalently retains
the measured variable-time term \(\max(T_{base}-t_0,0)\).

## 7. Two bandwidth sources

The model deliberately retains two independent bandwidth sources:

- \(\beta_{fit}\) comes from the parquet curve's OLS slope. It includes the
  behavior of the measured LL kernel and communication stack.
- \(\beta_{spec}\) comes from the system YAML (`SystemSpec`) and is a hardware
  path limit. It is not a fitted slope.

For an exact-topology curve, the runtime uses the measured variable-time term
and does not apply a \(\beta_{spec}\) floor. For a single-domain donor, it
compares the fit-derived and topology-spec **times** and selects the slower
one. It never silently replaces a measured slope with the larger advertised
bandwidth.

## 8. Topology paths

Ranks and experts use continuous placement. Each rank owns \(N/P\) adjacent
expert IDs.

- GB200/GB300: ranks in the same NVL72 rack use MNVL/NVSwitch. Traffic beyond
  72 GPUs uses `inter_rack_bw` (IB), falling back to `inter_node_bw` only when
  the rack bandwidth field is absent.
- B200/B300/H100/H200: ranks in one physical node use NVLink. Cross-node
  traffic uses `inter_node_bw` (IB/RDMA).

NVLink/MNVL and IB paths may progress in parallel, so the two path times are
combined with `max`. Within each path, endpoint time uses `max(TX,RX)` as
described in section 3.

## 9. Monte Carlo routing model

Every positive power-law exponent, including `power_law_1.0`, executes a
deterministic 4,096-trial Monte Carlo estimate and caches its P50. A value of
1.0 remains a sampled long-tail distribution; it is not the balanced case.
Explicit `uniform` and `balanced` routing first guarantee exactly
\(BK/P=B_rK\) destination assignments per rank and then distribute that quota
within each rank. Balanced exact-topology latency can therefore be evaluated once because
its endpoint imbalance is exactly one. Balanced donor modeling still executes
4,096 random assignment trials to expose source-to-destination path variation.

For each power-law trial:

1. independently sample one power-law weight for every expert;
2. normalize the weights to \(BK\) assignments and round to integer quotas;
3. cap each quota at \(B\), so one token cannot select the same expert twice;
4. use the existing rank-round-robin correction to restore the exact quota
   sum;
5. exchange the busiest contiguous expert rank with rank 0, preserving the
   established AIC worst-rank convention;
6. randomly shuffle token processing order and construct an exact-quota
   Top-K assignment: select any expert whose remaining quota equals the
   remaining token count, then sample all other experts proportionally to
   remaining quota without replacement within the token;
7. aggregate the logical routes onto endpoint TX/RX loads for NVLink/MNVL and
   IB separately.

The assignment conserves every expert quota exactly, never selects one expert
twice for a token, and cannot leave an infeasible residual quota. Hot experts
are not sorted onto different tokens or ranks; random token/source mapping can
therefore place hot traffic on the same endpoint, which is the behavior this
communication model needs to sample.

The power-law **quota generation** continues to follow the semantics of
`collector/helper.py::_generate_power_law_distribution`, including the quota
cap, rank-round-robin correction, and busiest-rank relabeling. The assignment
step intentionally differs. The A2A collector exercises real router-style
random routes, while helper's deterministic descending fill primarily serves
compute-workload construction. Equal Rust and PyTorch seeds also do not imply
equal random streams; cross-language parity fixtures must share sampled
weights explicitly.

A model string
`power_law_<alpha>` preserves its alpha; bare `power_law` and unknown names
use the Stage 1 default \(\alpha=1.2\).

The random generator is `ChaCha8Rng` with base seed
`0xA1C0_DEE5_EED0_0001`. Each trial derives a stable sub-seed from that base,
so parallel execution and cache misses remain reproducible.

## 10. Final communication latency

For each trial, let \(\alpha_{comm,trial}\) be the busiest full-duplex logical
endpoint load divided by the uniform average endpoint load.

When the selected curve matches the requested topology exactly,

\[
T_{trial}^{exact}=t_0+
  \alpha_{comm,trial}\max(T_{base}-t_0,0).
\]

The measured curve already carries that topology's communication behavior, so
advertised NVLink/IB bandwidth does not rescale or floor this branch.
For explicit `balanced`/`uniform` routing, the exact-topology result preserves
the measured or token-interpolated (T_{base}) verbatim. This is algebraically
the same expression when (T_{base}\geq t_0), and it deliberately preserves
the anchor when measurement noise puts an individual point below the fitted
OLS intercept.

When the curve is a single-domain donor, let \(T_{NVLink-spec,trial}\) and
\(T_{IB-spec,trial}\) be the path times computed from sampled bytes and
`SystemSpec` bandwidths. Then

\[
T_{trial}^{donor}=t_0+\max\left(
  \alpha_{comm,trial}\max(T_{base}-t_0,0),
  T_{NVLink-spec,trial},
  T_{IB-spec,trial}
\right).
\]

The runtime scalar is the standard median of the 4,096 complete trial
latencies (the average of the middle two values for an even sample count):

\[
T_{dispatch/combine}=P50(T_{trial}).
\]

The target load is applied directly to the measured variable-time term. It is
not divided by a separately simulated uniform baseline. `t0` is added once
and is never amplified by routing skew.

The cache key contains \(B_r,P,K,N,\alpha\), phase, topology-domain size,
payload, fitted variable time, and calibration mode. Donor entries also key on
both path bandwidths; exact entries deliberately do not. The bounded cache has
capacity 4,096 and stores the P50 latency consumed by runtime.
Every result on this modeled LL path is tagged `Estimated`.

In `HYBRID` mode, any viable exact, one-shot, or donor calibration executes
normally. Only after every candidate is exhausted does the operation return
`EmpiricalNotImplemented`. `SILICON` preserves the typed performance-data
miss, while invalid topology and EPLB requests preserve their configuration
errors.

## 11. Expert compute

DeepEP-LL expert compute uses the existing standard fused `MoE` predictor:

\[
T_{compute}=f_{MoE}(B, H, I, K, N, EP, \text{quant}, \text{distribution}).
\]

The query uses the model's already-resolved distribution curve. DeepSeek, for
example, emits `power_law_1.01`; the generic default emits
`power_law_1.2`. No `MoeExpertCompute` wide-EP table and no additional
Monte Carlo rank multiplier are used for LL. The large-EP graph also does not
insert ordinary EP pre/post-dispatch operators around this compute op.

## 12. Calibration evidence

The six checked-in SGLang 0.5.12 DeepEP-LL parquet files contain 192
phase/shape curves. An OLS fit over every token point gave:

- 192/192 positive slopes;
- 192/192 positive raw intercepts, with a minimum of approximately 6.02 µs;
- median \(R^2\) approximately 0.99909;
- minimum \(R^2\) approximately 0.9522.

For GB200, \(H=7168,K=8,N=256,EP=4\):

| Phase | OLS intercept \(b\) | OLS slope \(a\) | \(\beta_{fit}\) | NVLink spec |
|---|---:|---:|---:|---:|
| dispatch | 15.935 µs | 0.0934 µs/token | 634.3 GB/s | 900 GB/s |
| combine | 16.861 µs | 0.1476 µs/token | 776.8 GB/s | 900 GB/s |

![GB200 DeepEP-LL OLS feasibility](deepep_ll_gb200_ols.svg)

The fitted slopes are below the 900 GB/s hardware specification. The exact
EP4 result uses the measured curve plus Monte Carlo skew without a spec floor;
the separate specification remains a guardrail when that curve is used as a
donor for an unmeasured topology.

## 13. Stage 1 fixed assumptions

- balanced source-token count across ranks;
- continuous expert placement and \(N\) divisible by \(P\);
- existing AIC independent power-law weight/quota semantics;
- 4,096 trials for every power-law request and for balanced donors; balanced
  exact topology uses its one equivalent deterministic endpoint trial;
- P50 latency as the runtime scalar;
- full-duplex `max(send, recv)`;
- NVLink/MNVL and IB progress in parallel;
- all system bandwidths are per-GPU, one direction;
- DeepEP-LL EPLB is unsupported and returns an explicit error;
- LogFMT is not modeled.

## 14. Stage 2, Stage 3, and Stage 4 TODO

### Stage 2: validate the controlled model

- collect routing-skew sweeps and cross-topology LL curves;
- validate OLS stability and small-message nonlinearity;
- compare fitted and measured effective bandwidth;
- validate NVLink/IB overlap and endpoint contention.

### Stage 3: collect \(\alpha_{exp}\)

- collect model-specific expert hit-rate distributions;
- derive and validate the measured expert-imbalance factor \(\alpha_{exp}\);
- publish the model-specific artifact and metadata under the collector
  contract.

Stage 3 produces the measurement input. It does not change this Stage-1
Monte Carlo implementation to consume a real expert distribution.

### Stage 4: collector-driven routing

- collect the actual expert hit-rate vector, expert placement, or routing
  trace;
- allow Monte Carlo to accept the measured probability vector directly;
- optionally replay traces instead of sampling;
- support placement-aware EPLB;
- permit custom routing models without estimating a power-law alpha at all.

This makes the Stage 1 power-law generator a fallback rather than a permanent
requirement.
