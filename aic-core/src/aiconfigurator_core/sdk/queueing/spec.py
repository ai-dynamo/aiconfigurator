# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared types for the queueing (pass-calendar) model.

Every quantity in this package is derived from scheduler semantics or
queueing theory — there are NO fitted constants. See
docs/design/queueing_model.md for the term-by-term provenance table.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Protocol


class TimingModel(Protocol):
    """Timing provider interface.

    (batch_size, mean_isl, mean_prefix) is the native parameterization of
    the SDK's own phase estimators (`BaseBackend._run_context_phase`), i.e.
    the minimal description of one prefill batch; `DatabaseTimingModel`
    delegates to them directly. The Protocol exists so the evaluator can
    also be driven by synthetic timing functions during validation, which
    cancels timing out of the validation residual.

    Implementations MAY additionally provide
    ``mixed_pass_ms(ctx_tokens, gen_tokens, isl, osl, prefix) -> float``
    for the duration of one fused prefill+decode pass; the calendar prefers
    it when present and otherwise composes ``prefill_ms + decode_ms``
    (which double-counts the shared non-attention cost — see
    ``DatabaseTimingModel.mixed_pass_ms``).
    """

    def prefill_ms(self, batch_size: int, mean_isl: int, mean_prefix: int) -> float:
        """Latency of one prefill batch: batch_size requests, mean effective
        prompt length mean_isl of which mean_prefix is cached."""
        ...

    def decode_ms(self, batch_size: int, context_len: int) -> float:
        """Latency of one decode iteration for batch_size sequences at mean
        context length context_len."""
        ...


@dataclass(frozen=True)
class WorkloadSpec:
    """Stationary workload characterization.

    The model covers stationary regimes: (isl, osl, prefix) — fixed, or
    described by marginal quantile streams (W2) — under a closed-loop
    concurrency cap or an open-loop Poisson-like rate (W1). Timestamped
    traces enter by reduction: quasi-stationary windowing + empirical
    quantile extraction (see the design doc's fidelity contract); raw
    non-stationary replay stays out of scope for the analytical model.
    """

    isl: int
    osl: int
    prefix: int = 0
    concurrency: Optional[int] = None  # closed loop in-flight cap
    request_rate: Optional[float] = None  # open loop, requests/s
    num_requests: Optional[int] = None  # benchmark length N for mean(N)
    # Variable-length workloads: deterministic stratified shape streams.
    # Each entry is one stratum of the length distribution (inverse-CDF
    # midpoints — see ``stratified_quantiles``); slots draw their own
    # (isl, osl) from these streams via a coprime-stride rotation (zero
    # RNG, converges to the marginal over the steady window). ``isl`` /
    # ``osl`` above stay the NOMINAL means, used for capacity-style
    # arithmetic (admission caps, identity anchor, output-token rate).
    # Heterogeneity must live inside the batch: a mixture of homogeneous
    # fixed-shape runs keeps each component's convoy structure and cannot
    # reproduce the measured desynchronization (h20e trtllm tp4,
    # isl cv=0.25: steady TTFT 1.8s -> 0.66s, throughput within 10%).
    # Scope: closed-loop agg calendars; the disagg tandem stays
    # fixed-shape for now.
    isl_quantiles: Optional[tuple] = None
    osl_quantiles: Optional[tuple] = None
    # W3 joint shape strata: tuple of (isl, prefix, osl) triples drawn as a
    # unit (one stratum = one real trace record at an isl-ordered quantile
    # midpoint — see ``stratified_shape_tuples``), so isl<->osl correlation
    # and per-request prefix hits survive; marginal quantile streams cannot
    # carry either. Mutually exclusive with isl_quantiles/osl_quantiles.
    shape_tuples: Optional[tuple] = None
    # W3 empirical inter-arrival strata (ms), open loop only: replaces the
    # default exponential strata, normalized so the stream mean is exactly
    # 1000/request_rate — pass raw trace inter-arrivals (zeros allowed:
    # batched arrivals) and control the rate independently.
    arrival_quantiles: Optional[tuple] = None
    # Per-request client/frontend turnaround: the time between a slot
    # freeing (previous request's completion, which is when a closed-loop
    # client dispatches the replacement) and the replacement becoming
    # VISIBLE to the scheduler (HTTP receive -> tokenize -> IPC -> waiting
    # queue). At 0 the replacement lands exactly on the pass boundary and
    # always catches the next pass — a knife-edge that real deployments
    # never hit: any eps > 0 makes arrivals miss the boundary and wait out
    # the pass in flight, which cascades into cohort clumping (validated on
    # b300/vllm-0.24: eps ~= 15 ms turns TTFT p50 from 135 into 523 ms at
    # C=32 with throughput and ITL unchanged). This is a timing-layer
    # quantity: measure it, don't fit it (e.g. c=1 TTFT minus the perf-DB
    # prefill latency; it is the same physical overhead the legacy additive
    # dispatch term approximates).
    turnaround_ms: float = 0.0
    # Size-dependent slope of the same turnaround path (turnaround_ms is the
    # fixed part): client serialize -> HTTP transfer -> frontend tokenize all
    # scale with prompt length, so a request dispatched at t becomes visible
    # to the scheduler at t + turnaround_ms + isl * ingest_us_per_token/1000.
    # This is an ARRIVAL-PLANE mapping, not a service cost: trace timestamps
    # record client dispatch, but the engine's FCFS queue orders by scheduler
    # arrival — for near-simultaneous dispatches (e.g. Mooncake's 3s-bucketed
    # batch arrivals) the size slope reorders the burst shortest-first, which
    # dominates burst TTFT (measured h20e/trtllm 1.3.0rc20: a 23k-token and a
    # 1k-token prompt dispatched together ALWAYS serve small-first — order
    # flips only once the small lags by 60-100 ms, i.e. ~3.6 us/token; same
    # inversion visible in a real Mooncake replay: 96% of same-bucket
    # big-then-small pairs emit first token small-first while client dispatch
    # order is 0% inverted). Measure it like turnaround_ms, don't fit it:
    # predictions are ordering-sensitive but magnitude-insensitive (any c > 0
    # yields the same same-instant ordering; +-2x shifts TTFT < 1%).
    ingest_us_per_token: float = 0.0

    def __post_init__(self):
        if (self.concurrency is None) == (self.request_rate is None):
            raise ValueError("specify exactly one of concurrency / request_rate")
        if self.osl < 1 or self.isl < 1:
            raise ValueError("isl and osl must be >= 1")
        if not 0 <= self.prefix <= self.isl:
            raise ValueError("prefix must be between 0 and isl")
        if self.concurrency is not None and self.concurrency < 1:
            raise ValueError("concurrency must be >= 1")
        if self.request_rate is not None and self.request_rate <= 0:
            raise ValueError("request_rate must be > 0")
        if self.num_requests is not None and self.num_requests < 1:
            raise ValueError("num_requests must be >= 1")
        if self.turnaround_ms < 0:
            raise ValueError("turnaround_ms must be >= 0")
        if self.ingest_us_per_token < 0:
            raise ValueError("ingest_us_per_token must be >= 0")
        for name in ("isl_quantiles", "osl_quantiles"):
            qs = getattr(self, name)
            if qs is None:
                continue
            qs = tuple(int(v) for v in qs)
            object.__setattr__(self, name, qs)
            if not qs or any(v < 1 for v in qs):
                raise ValueError(f"{name} must be a non-empty tuple of ints >= 1")
        if self.isl_quantiles is not None and self.prefix > min(self.isl_quantiles):
            raise ValueError("prefix must not exceed the smallest isl quantile")
        if self.shape_tuples is not None:
            if self.isl_quantiles or self.osl_quantiles:
                raise ValueError("shape_tuples and isl/osl_quantiles are mutually exclusive")
            tt = tuple((int(a), int(b), int(c)) for a, b, c in self.shape_tuples)
            object.__setattr__(self, "shape_tuples", tt)
            if not tt or any(i < 1 or o < 1 or not 0 <= px < i for i, px, o in tt):
                raise ValueError("shape_tuples entries must satisfy isl>=1, osl>=1, 0<=prefix<isl")
        if self.arrival_quantiles is not None:
            if self.request_rate is None:
                raise ValueError("arrival_quantiles requires an open-loop workload (request_rate)")
            aq = tuple(float(v) for v in self.arrival_quantiles)
            object.__setattr__(self, "arrival_quantiles", aq)
            if not aq or any(v < 0 for v in aq) or sum(aq) <= 0:
                raise ValueError("arrival_quantiles must be non-empty, non-negative, with positive mean")

    @property
    def effective_isl(self) -> int:
        return max(1, self.isl - self.prefix)


def stratified_quantiles(values, k: int = 16) -> tuple:
    """Deterministic inverse-CDF midpoints of an empirical length sample:
    ``k`` strata, each represented by the sample quantile at the stratum
    midpoint. Zero-RNG companion to ``WorkloadSpec.isl_quantiles`` /
    ``osl_quantiles``."""
    vs = sorted(int(v) for v in values)
    if not vs:
        raise ValueError("values must be non-empty")
    n = len(vs)
    return tuple(vs[min(n - 1, int((i + 0.5) / k * n))] for i in range(k))


def stratified_shape_tuples(records, k: int = 32) -> tuple:
    """Deterministic joint strata from trace records: sort (isl, prefix, osl)
    triples by total work (isl + osl), then take the record at each stratum
    midpoint. One stratum = one REAL record, so correlations and prefix hits
    ride along for free. Zero-RNG companion to ``WorkloadSpec.shape_tuples``."""
    rs = sorted(((int(i), int(p), int(o)) for i, p, o in records), key=lambda t: t[0] + t[2])
    if not rs:
        raise ValueError("records must be non-empty")
    n = len(rs)
    return tuple(rs[min(n - 1, int((j + 0.5) / k * n))] for j in range(k))


def workload_fidelity(wl: "WorkloadSpec") -> str:
    """The W-tier of the workload description an evaluation consumed — the
    input side of the fidelity contract (design doc, "Workload-fidelity
    contract"): W0 fixed shape, W1 + open-loop arrivals, W2 + shape
    marginals; W3 (joint shape/prefix streams) and W4 (temporal structure)
    are contract placeholders, not implemented. Orthogonal features
    collapse to the highest tier, with components spelled out so the string
    stays readable without the table."""
    open_loop = wl.request_rate is not None
    joint = bool(wl.shape_tuples or wl.arrival_quantiles)
    marginals = bool(wl.isl_quantiles or wl.osl_quantiles)
    tier = 3 if joint else (2 if marginals else (1 if open_loop else 0))
    parts = [
        "open-loop" if open_loop else "closed-loop",
        "joint-shapes" if wl.shape_tuples else ("shape-marginals" if marginals else "fixed-shape"),
    ]
    if wl.arrival_quantiles:
        parts.append("empirical-arrivals")
    return f"W{tier}({', '.join(parts)})"


def _shape_stream(quantiles: Optional[tuple], fallback: int):
    """Deterministic low-discrepancy rotation over the strata: index n maps
    to quantiles[(n * stride) % k] with a golden-ratio stride coprime to k,
    so consecutive draws sweep the distribution instead of clustering, and
    the stream is exactly reproducible (resume-safe, no RNG)."""
    if not quantiles:
        while True:
            yield fallback
    k = len(quantiles)
    stride = max(1, round(k * 0.6180339887))
    from math import gcd

    while gcd(stride, k) != 1:
        stride += 1
    n = 0
    while True:
        yield quantiles[(n * stride) % k]
        n += 1


@dataclass(frozen=True)
class EngineSpec:
    """Engine scheduling parameters (names follow vLLM; per-backend calendars
    reinterpret them where the engine's knobs differ)."""

    max_num_batched_tokens: int = 8192
    max_num_seqs: int = 256
    enable_chunked_prefill: bool = True
    # One-pass scheduling lookahead (vLLM AsyncScheduler, default-ON since
    # vLLM 0.24): the batch for pass k+1 is fixed while pass k executes, so
    # an arrival during pass k joins pass k+2 at the earliest — every
    # admission pays up to one extra pass of TTFT. Decode-side effects of
    # async scheduling (hidden per-step CPU gap) belong to the timing layer,
    # not here. Default False preserves the synchronous calendar.
    # The flag's semantics is "does a NEW ARRIVAL miss one extra pass", not
    # "does the engine overlap anything": TRT-LLM's overlap scheduler
    # (disable_overlap_scheduler=False, default) overlaps execution prep for
    # the already-admitted batch while admission still sees fresh arrivals,
    # so TRT-LLM maps to False. Measured (h20e_sxm, trtllm 1.3.0rc20,
    # Qwen3-32B tp4, isl4096/osl256): True overpredicts steady TTFT by
    # 25-30% at C>=32, False lands within ~4%.
    async_scheduling: bool = False
    # SGLang-specific (used by the sglang calendar only)
    max_prefill_tokens: Optional[int] = None  # defaults to max_num_batched_tokens
    chunked_prefill_size: Optional[int] = None  # defaults to max_num_batched_tokens
    # default True: AIC's generator deploys SGLang agg with mixed chunk ON
    # (rule_plugin/sglang.rule), so the calendar matches the deployed engine;
    # False selects the alternating (dedicated prefill batch) calendar
    enable_mixed_chunk: bool = True
    # TRT-LLM-specific (used by the trtllm calendar only)
    guaranteed_no_evict: bool = False
    kv_capacity_tokens: Optional[int] = None  # needed by guaranteed_no_evict

    def __post_init__(self):
        if self.max_num_batched_tokens < 1 or self.max_num_seqs < 1:
            raise ValueError("max_num_batched_tokens and max_num_seqs must be >= 1")
        if self.guaranteed_no_evict and (self.kv_capacity_tokens is None or self.kv_capacity_tokens < 1):
            # fail loudly instead of silently ignoring the admission cap
            raise ValueError("guaranteed_no_evict requires positive kv_capacity_tokens")


@dataclass
class Distribution:
    """Discrete weighted distribution (TTFT/ITL are mixtures of pass-calendar
    mass points, not smooth densities — this representation is exact)."""

    values: list = field(default_factory=list)
    weights: list = field(default_factory=list)

    def add(self, value: float, weight: float = 1.0) -> None:
        self.values.append(float(value))
        self.weights.append(float(weight))

    def _sorted(self):
        pairs = sorted(zip(self.values, self.weights, strict=True))
        total = sum(w for _, w in pairs)
        return pairs, total

    @property
    def mean(self) -> float:
        total = sum(self.weights)
        if total <= 0:
            return float("nan")
        return sum(v * w for v, w in zip(self.values, self.weights, strict=True)) / total

    def quantile(self, q: float) -> float:
        pairs, total = self._sorted()
        if not pairs or total <= 0:
            return float("nan")
        target = q * total
        acc = 0.0
        for v, w in pairs:
            acc += w
            if acc >= target:
                return v
        return pairs[-1][0]

    @property
    def p50(self) -> float:
        return self.quantile(0.50)

    @property
    def p90(self) -> float:
        return self.quantile(0.90)

    @property
    def p99(self) -> float:
        return self.quantile(0.99)

    @property
    def maximum(self) -> float:
        return max(self.values) if self.values else float("nan")

    def shifted(self, delta_ms: float) -> Distribution:
        """New distribution with every mass point shifted by delta_ms
        (used for additive latency stages, e.g. a vision encoder ahead of
        the LLM prefill)."""
        out = Distribution()
        out.values = [v + delta_ms for v in self.values]
        out.weights = list(self.weights)
        return out

    def scaled_mix(self, other: Distribution, self_weight: float, other_weight: float) -> Distribution:
        out = Distribution()
        s_total = sum(self.weights) or 1.0
        o_total = sum(other.weights) or 1.0
        for v, w in zip(self.values, self.weights, strict=True):
            out.add(v, w / s_total * self_weight)
        for v, w in zip(other.values, other.weights, strict=True):
            out.add(v, w / o_total * other_weight)
        return out


@dataclass
class QueueingReport:
    """Full output of the queueing model.

    ttft_steady / itl / tpot are steady-state (deployment capability);
    ttft_transient is the initial-burst admission staircase (cold start /
    synchronized-burst behavior); ttft_mean_n blends them for a benchmark
    of num_requests, making the N-dependence of the blended mean explicit.
    """

    ttft_steady: Distribution
    ttft_transient: Distribution
    itl: Distribution
    tpot: Distribution
    throughput_rps: float
    output_tokens_per_s: float
    e2e: Distribution = field(default_factory=Distribution)
    backend: str = ""
    mode: str = "agg"  # agg | disagg | static
    num_requests: Optional[int] = None
    # disagg decomposition (0 for agg)
    kv_transfer_ms: float = 0.0
    prefill_queue_ms: float = 0.0
    # input-side fidelity tier this report consumed (see workload_fidelity
    # and the design doc's fidelity contract); evaluators set it so
    # downstream consumers can gate on prediction quality without
    # re-deriving what the workload description contained
    workload_fidelity: str = "W0(closed-loop, fixed-shape)"
    # trace-replay diagnostics (evaluate_open_loop arrival_trace mode only):
    # one dict per request in trace order — arrival_ms, isl, prefix, osl,
    # ttft_ms, e2e_ms. None outside trace mode; not part of the summary
    # contract, intended for per-request diffing against reference
    # simulators/live replays.
    per_request: Optional[list] = None

    @property
    def ttft_mean_n(self) -> float:
        """Blended mean for a benchmark of N requests: the transient window
        covers the initial concurrency burst; the rest is steady state."""
        n = self.num_requests
        w = len(self.ttft_transient.values)
        if n is None or n <= 0 or not self.ttft_transient.values:
            return self.ttft_steady.mean
        w = min(w, n)
        return (w * self.ttft_transient.mean + (n - w) * self.ttft_steady.mean) / n

    def to_columns(self, prefix: str = "") -> dict:
        """Flatten into additive summary-dataframe columns."""
        p = prefix
        return {
            f"{p}ttft_steady_mean": self.ttft_steady.mean,
            f"{p}ttft_steady_p50": self.ttft_steady.p50,
            f"{p}ttft_steady_p90": self.ttft_steady.p90,
            f"{p}ttft_steady_p99": self.ttft_steady.p99,
            f"{p}ttft_transient_mean": self.ttft_transient.mean,
            f"{p}ttft_transient_max": self.ttft_transient.maximum,
            f"{p}ttft_mean_n": self.ttft_mean_n,
            f"{p}itl_mean": self.itl.mean,
            f"{p}itl_p50": self.itl.p50,
            f"{p}itl_p99": self.itl.p99,
            f"{p}tpot_mean_calendar": self.tpot.mean,
            f"{p}tpot_p99_calendar": self.tpot.p99,
        }
