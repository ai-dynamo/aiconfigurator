# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Disaggregated P/D tandem model.

TTFT_disagg = W_prefill + S_prefill + KV_handoff + decode_first_pass

Semantics verified against dynamo mocker offline disagg replay
(tools/queueing_oracle README): the user-visible first token is emitted by
the DECODE stage (the prefill worker's token is internal); the handoff delay
defers decode enqueue; decode workers pay zero prefill compute but the
prompt admission still consumes scheduler budget.

Rate matching is an OUTPUT here: for a candidate (n_prefill, n_decode) the
closed-loop fixed point yields throughput and both stage utilizations;
imbalance surfaces as prefill queueing (TTFT) or decode saturation
(TPOT/throughput) instead of the legacy 0.9/0.92 scalar derates.

The router dispatch policy is exposed as `prefill_inflight_cap` (kappa):
  None  = engine-batched admission (matches this package's evaluator and the
          DES round-robin driver)
  1     = serialized prefills per worker (approximates dynamo kv_router's
          pending-queue admission; measured impact on TTFT mean ~20%)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from .spec import Distribution, EngineSpec, QueueingReport, TimingModel, WorkloadSpec


@dataclass(frozen=True)
class DisaggSpec:
    num_prefill_workers: int
    num_decode_workers: int
    kv_transfer_bandwidth_gbps: float = 0.0  # 0 disables the handoff term
    kv_bytes_per_token: int = 0
    prefill_inflight_cap: Optional[int] = None  # kappa; None = engine-batched


def kv_handoff_ms(wl: WorkloadSpec, spec: DisaggSpec) -> float:
    """mocker common/utils.rs compute_prefill_handoff_delay_ms:
    flat isl * bytes_per_token / bandwidth."""
    if spec.kv_transfer_bandwidth_gbps <= 0 or spec.kv_bytes_per_token <= 0:
        return 0.0
    return wl.isl * spec.kv_bytes_per_token / (spec.kv_transfer_bandwidth_gbps * 1e9) * 1000.0


def _prefill_pass_ms(wl: WorkloadSpec, eng: EngineSpec, timing: TimingModel, batch: int, chunk_per_req: int) -> float:
    return timing.prefill_ms(batch, wl.prefix + chunk_per_req, wl.prefix)


def evaluate_disagg(
    wl: WorkloadSpec,
    prefill_eng: EngineSpec,
    decode_eng: EngineSpec,
    timing: TimingModel,
    spec: DisaggSpec,
    backend: str = "vllm",
) -> QueueingReport:
    """Tandem pass-calendar recursion (primary path).

    Same methodology as the agg evaluator: deterministic pass-level
    recursion of both worker pools plus the handoff delay queue. No RNG,
    no per-token events; validated against the DES DisaggSimulator."""
    if wl.concurrency is None:
        raise ValueError("disagg model currently supports closed-loop workloads")
    return _tandem_recursion(wl, prefill_eng, decode_eng, timing, spec, backend)


@dataclass
class _Req:
    arrival_ms: float
    remaining_prefill: int = 0
    generated: int = 0
    first_token_ms: float = -1.0
    last_token_ms: float = -1.0
    gaps: list = None
    is_initial_burst: bool = False


class _Pool:
    """One worker pool: each worker runs back-to-back passes over its own
    request list; workers become free at busy_until and are driven lazily."""

    def __init__(self, n: int):
        self.busy_until = [0.0] * n
        self.queues: list[list[_Req]] = [[] for _ in range(n)]
        self._rr = 0

    def dispatch(self, req: _Req, now_ms: float) -> None:
        widx = self._rr % len(self.queues)
        self.queues[widx].append(req)
        self._rr += 1
        # an idle worker's clock must not lag behind the arrival: passes for
        # this request can only start once it exists
        if self.busy_until[widx] < now_ms:
            self.busy_until[widx] = now_ms

    def next_event(self) -> float:
        candidates = [t for t, q in zip(self.busy_until, self.queues, strict=True) if q]
        return min(candidates) if candidates else float("inf")


def _tandem_recursion(wl, prefill_eng, decode_eng, timing, spec, backend):
    c = wl.concurrency
    handoff = kv_handoff_ms(wl, spec)
    prefill = _Pool(spec.num_prefill_workers)
    decode = _Pool(spec.num_decode_workers)
    in_transfer: list[tuple[float, _Req]] = []  # (ready_ms, req)

    for _ in range(c):
        r = _Req(arrival_ms=0.0, remaining_prefill=wl.effective_isl, gaps=[], is_initial_burst=True)
        prefill.dispatch(r, 0.0)

    completions = 0
    warmup = 4 * c
    target = 8 * c
    steady_start_ms = None
    now = 0.0

    ttft_transient = Distribution()
    ttft_steady = Distribution()
    itl = Distribution()
    tpot = Distribution()
    steady_completions = 0

    def run_prefill_pass(widx: int, start_ms: float) -> float:
        q = prefill.queues[widx]
        budget = prefill_eng.max_num_batched_tokens
        cap = spec.prefill_inflight_cap or len(q)
        batch_count = 0
        batch_isl = 0
        batch_prefix = 0
        finished = []
        for r in list(q)[:cap]:
            if budget <= 0:
                break
            chunk = min(r.remaining_prefill, budget)
            computed_before = wl.prefix + (wl.effective_isl - r.remaining_prefill)
            r.remaining_prefill -= chunk
            budget -= chunk
            batch_count += 1
            batch_isl += computed_before + chunk
            batch_prefix += computed_before
            if r.remaining_prefill == 0:
                finished.append(r)
        if batch_count == 0:
            return start_ms
        dur = timing.prefill_ms(batch_count, batch_isl // batch_count, batch_prefix // batch_count)
        end = start_ms + dur
        for r in finished:
            q.remove(r)
            in_transfer.append((end + handoff, r))
        return end

    def run_decode_pass(widx: int, start_ms: float):
        nonlocal completions, steady_start_ms, steady_completions
        q = decode.queues[widx]
        emitters = [r for r in q if r.generated < wl.osl]
        if not emitters:
            return start_ms, []
        ctx = sum(wl.isl + r.generated for r in emitters) // len(emitters)
        dur = timing.decode_ms(len(emitters), ctx)
        end = start_ms + dur
        done = []
        for r in emitters:
            r.generated += 1
            if r.first_token_ms < 0:
                r.first_token_ms = end
                ttft_ms = end - r.arrival_ms
                if r.is_initial_burst:
                    ttft_transient.add(ttft_ms)
                elif completions >= warmup:
                    ttft_steady.add(ttft_ms)
            else:
                r.gaps.append(end - r.last_token_ms)
            r.last_token_ms = end
            if r.generated >= wl.osl:
                done.append(r)
        for r in done:
            q.remove(r)
        return end, done

    max_iters = 2000 * (target)
    for _ in range(max_iters):
        if completions >= target:
            break
        # release transferred KV whose handoff has elapsed up to `now`
        in_transfer.sort(key=lambda x: x[0])
        # next event across pools and transfers
        t_pf = min((prefill.busy_until[i] for i, q in enumerate(prefill.queues) if q), default=float("inf"))
        t_dc = min(
            (decode.busy_until[i] for i, q in enumerate(decode.queues) if any(r.generated < wl.osl for r in q)),
            default=float("inf"),
        )
        t_tr = in_transfer[0][0] if in_transfer else float("inf")
        now = min(t_pf, t_dc, t_tr)
        if now == float("inf"):
            raise RuntimeError("disagg tandem recursion stalled")

        while in_transfer and in_transfer[0][0] <= now:
            _, r = in_transfer.pop(0)
            decode.dispatch(r, now)

        for i, q in enumerate(prefill.queues):
            if q and prefill.busy_until[i] <= now:
                prefill.busy_until[i] = run_prefill_pass(i, now)
        for i, q in enumerate(decode.queues):
            if q and decode.busy_until[i] <= now:
                end, done = run_decode_pass(i, now)
                decode.busy_until[i] = end
                for r in done:
                    completions += 1
                    if completions == warmup:
                        steady_start_ms = end
                    if completions > warmup and not r.is_initial_burst:
                        steady_completions += 1
                        for g in r.gaps:
                            itl.add(g)
                        if r.gaps:
                            tpot.add(sum(r.gaps) / len(r.gaps))
                    nxt = _Req(arrival_ms=end, remaining_prefill=wl.effective_isl, gaps=[])
                    prefill.dispatch(nxt, end)
    else:
        raise RuntimeError("disagg tandem recursion did not converge")

    window_ms = now - (steady_start_ms or 0.0)
    x = steady_completions / (window_ms / 1000.0) if window_ms > 0 else 0.0
    return QueueingReport(
        ttft_steady=ttft_steady,
        ttft_transient=ttft_transient,
        itl=itl,
        tpot=tpot,
        throughput_rps=x,
        output_tokens_per_s=x * wl.osl,
        backend=backend,
        mode="disagg",
        num_requests=wl.num_requests,
        kv_transfer_ms=handoff,
        prefill_queue_ms=max(0.0, ttft_steady.mean - handoff - (itl.mean if itl.values else 0.0)),
    )


def evaluate_disagg_closed_form(
    wl: WorkloadSpec,
    prefill_eng: EngineSpec,
    decode_eng: EngineSpec,
    timing: TimingModel,
    spec: DisaggSpec,
    backend: str = "vllm",
) -> QueueingReport:
    """Sweep fast path: Little's-law fixed point, mean estimates only.
    Coarser than the tandem recursion (closed-loop arrivals are smoother
    than the M/D/c assumption); use the recursion for reported numbers."""
    if wl.concurrency is None:
        raise ValueError("disagg model currently supports closed-loop workloads")
    c = wl.concurrency
    handoff = kv_handoff_ms(wl, spec)
    b_pf = prefill_eng.max_num_batched_tokens
    ctx = wl.isl + wl.osl // 2

    # prefill service per request (no decode interference on prefill workers)
    if spec.prefill_inflight_cap == 1:
        # kv_router-style serialization: each prompt runs alone
        chunks = math.ceil(wl.effective_isl / b_pf)
        s_pf = sum(
            _prefill_pass_ms(wl, prefill_eng, timing, 1, min(b_pf, wl.effective_isl - i * b_pf)) for i in range(chunks)
        )
    else:
        # engine-batched: m prompts share each pass's budget
        m = max(1, min(c, b_pf // wl.effective_isl)) if wl.effective_isl <= b_pf else 1
        chunks = math.ceil(m * wl.effective_isl / b_pf)
        s_pf = chunks * _prefill_pass_ms(wl, prefill_eng, timing, m, min(wl.effective_isl, b_pf // m)) / 1.0
    # per-request occupancy of a prefill server (a batched pass of m requests
    # consumes s_pf wall-time for m requests)
    m_batch = (
        1
        if spec.prefill_inflight_cap == 1
        else max(1, min(c, b_pf // wl.effective_isl))
        if wl.effective_isl <= b_pf
        else 1
    )
    s_pf_per_req = s_pf / m_batch

    # fixed point on throughput X (req/ms)
    c_d = float(max(1, c - 1))
    x = 0.0
    t_dec = timing.decode_ms(max(1, round(c_d / spec.num_decode_workers)), ctx)
    for _ in range(50):
        b_d = max(1.0, c_d / spec.num_decode_workers)
        t_dec = timing.decode_ms(max(1, round(b_d)), ctx)
        r_decode = wl.osl * t_dec
        x_decode_cap = c_d / r_decode if r_decode > 0 else float("inf")
        # prefill pool utilization; closed-loop arrivals are completion-
        # regulated (smoother than Poisson), so the wait is the busy-server
        # pass residual rather than an M/D/c queue
        rho_p = min(0.999, x * s_pf_per_req / spec.num_prefill_workers) if x > 0 else 0.0
        w_p = rho_p * s_pf / 2.0
        r_total = w_p + s_pf + handoff + r_decode
        x_new = min(
            c / r_total, x_decode_cap, spec.num_prefill_workers / s_pf_per_req if s_pf_per_req > 0 else float("inf")
        )
        if abs(x_new - x) < 1e-9:
            x = x_new
            break
        x = 0.5 * x + 0.5 * x_new
        c_d = max(1.0, c - x * (w_p + s_pf + handoff))

    rho_p = min(0.999, x * s_pf_per_req / spec.num_prefill_workers)
    w_p = rho_p * s_pf / 2.0

    # user-visible TTFT: prefill wait + service + handoff + first decode pass
    ttft_mean = w_p + s_pf + handoff + t_dec
    ttft_steady = Distribution()
    ttft_steady.add(ttft_mean)

    # transient: initial burst staircases through the prefill pool
    ttft_transient = Distribution()
    per_worker_burst = math.ceil(c / spec.num_prefill_workers)
    for k in range(1, per_worker_burst + 1):
        passes = math.ceil(k * wl.effective_isl / b_pf)
        ttft_transient.add(
            passes * _prefill_pass_ms(wl, prefill_eng, timing, 1, min(b_pf, wl.effective_isl)) + handoff + t_dec
        )

    itl = Distribution()
    itl.add(t_dec)  # decode-only stage: no prefill interference (single mass)
    tpot = Distribution()
    tpot.add(t_dec)

    return QueueingReport(
        ttft_steady=ttft_steady,
        ttft_transient=ttft_transient,
        itl=itl,
        tpot=tpot,
        throughput_rps=x * 1000.0,
        output_tokens_per_s=x * 1000.0 * wl.osl,
        backend=backend,
        mode="disagg",
        num_requests=wl.num_requests,
        kv_transfer_ms=handoff,
        prefill_queue_ms=w_p,
    )
