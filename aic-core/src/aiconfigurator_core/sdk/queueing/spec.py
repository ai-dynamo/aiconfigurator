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

    The model covers stationary regimes only: fixed (isl, osl, prefix) with
    either a closed-loop concurrency cap or an open-loop Poisson rate.
    Timestamped traces / non-stationary arrivals are out of scope for the
    analytical correction — use simulation-level tooling for those.
    """

    isl: int
    osl: int
    prefix: int = 0
    concurrency: Optional[int] = None  # closed loop in-flight cap
    request_rate: Optional[float] = None  # open loop, requests/s
    num_requests: Optional[int] = None  # benchmark length N for mean(N)
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

    def __post_init__(self):
        if (self.concurrency is None) == (self.request_rate is None):
            raise ValueError("specify exactly one of concurrency / request_rate")
        if self.osl < 1 or self.isl < 1:
            raise ValueError("isl and osl must be >= 1")
        if self.turnaround_ms < 0:
            raise ValueError("turnaround_ms must be >= 0")

    @property
    def effective_isl(self) -> int:
        return max(1, self.isl - self.prefix)


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
