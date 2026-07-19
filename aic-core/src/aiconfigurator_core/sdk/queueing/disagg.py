# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Disaggregated P/D tandem model: the pass-calendar recursion for disagg.

Serving-flow semantics (matching the disagg deployment and the DES oracle):

  - the prefill pool computes the prompt and produces the FIRST token — that
    IS the TTFT token, streamed to the user from the prefill worker;
  - the KV cache is then handed to a decode worker. Transfers are flows on a
    per-worker-NIC fabric with max-min fair sharing, so fan-out (a clump of
    completions leaving one prefill worker) and fan-in (several prefill
    workers landing on one decode worker) slow each other down by the
    computed fair share — the handoff lands in the FIRST ITL GAP, not TTFT;
  - the decode worker continues the same sequence from token 2 and pays no
    prefill compute (KV-connector semantics: transferred KV counts as
    computed tokens).

Rate matching is an OUTPUT here, not an input: for a candidate
(num_prefill_workers, num_decode_workers) the closed-loop recursion yields
throughput and both stages' behavior directly — pool imbalance surfaces as
prefill queueing (TTFT) or decode saturation (ITL/throughput) instead of
scalar throughput derates.

The router dispatch policy is exposed as ``prefill_inflight_cap`` (kappa):
  None  = engine-batched admission (all queued prompts share a prefill
          pass's token budget — matches the DES round-robin driver)
  1     = serialized prefills per worker (approximates a kv_router-style
          pending-queue admission; measured impact on TTFT mean ~20%)

Same methodology as the agg evaluator (`calendar.evaluate_closed_loop`):
deterministic pass-level recursion, no RNG, no per-token events. Validated
against the DES ``DisaggSimulator`` (tools/queueing_oracle) — the disagg
families of the validation gate compare the two with identical timing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from .spec import Distribution, EngineSpec, QueueingReport, TimingModel, WorkloadSpec


@dataclass(frozen=True)
class DisaggSpec:
    """Deployment shape + KV-transfer fabric for the tandem model.

    Bandwidths are nominal single-direction Byte/s per worker NIC (the AIC
    system spec convention: ``node.inter_node_bw`` / ``node.intra_node_bw``),
    de-rated by ``bw_efficiency`` (default 0.8, mirroring the spec's own
    ``mem_bw_empirical_scaling_factor`` convention).
    ``kv_bytes_per_token == 0`` disables transfer modeling entirely
    (zero-delay handoff).
    """

    num_prefill_workers: int
    num_decode_workers: int
    kv_bytes_per_token: int = 0
    egress_bytes_per_s: float = 0.0  # per prefill-worker NIC
    ingress_bytes_per_s: float = 0.0  # per decode-worker NIC
    bw_efficiency: float = 0.8
    prefill_inflight_cap: Optional[int] = None  # kappa; None = engine-batched


class _TransferFabric:
    """Max-min fair sharing of per-worker NIC bandwidth (deterministic
    fluid recompute). Same math as the DES oracle's ``TransferFabric``
    (tools/queueing_oracle/vllm_sim.py); the disagg gate families compare
    the two sides through it, so drift between the copies fails the gate."""

    _EPS_BYTES = 1e-6
    _EPS_MS = 1e-6  # sub-ns: absorbs float dust at large virtual times

    def __init__(self, spec: DisaggSpec):
        self._egress = spec.egress_bytes_per_s * spec.bw_efficiency
        self._ingress = spec.ingress_bytes_per_s * spec.bw_efficiency
        self._flows: dict[int, list] = {}  # fid -> [src, dst, remaining, rate, payload]
        self._finished: list[tuple[float, object]] = []  # (finish_ms, payload)
        self._next_fid = 0
        self._t_ms = 0.0

    def has_flows(self) -> bool:
        return bool(self._flows) or bool(self._finished)

    def _internal_next_ms(self) -> Optional[float]:
        times = [f[2] / f[3] for f in self._flows.values() if f[3] > 0]
        if not times:
            return None
        return self._t_ms + max(min(times) * 1000.0, self._EPS_MS)

    def _advance(self, now_ms: float) -> None:
        """Piecewise fluid advance: rates change at every completion, so the
        clock must stop at each internal completion point (collecting the
        finished flow with its TRUE finish time and recomputing rates)
        before continuing — a single linear step would both mis-share
        bandwidth after the completion and swallow the completion event
        when a caller submits with a future timestamp."""
        while self._t_ms < now_ms:
            t_star = self._internal_next_ms()
            step_to = now_ms if (t_star is None or t_star > now_ms) else t_star
            dt_s = (step_to - self._t_ms) / 1000.0
            if dt_s > 0:
                for f in self._flows.values():
                    f[2] = max(0.0, f[2] - f[3] * dt_s)
            self._t_ms = step_to
            done = sorted(
                fid
                for fid, f in self._flows.items()
                if f[2] <= self._EPS_BYTES or (f[3] > 0 and f[2] / f[3] * 1000.0 <= self._EPS_MS)
            )
            if done:
                for fid in done:
                    self._finished.append((self._t_ms, self._flows.pop(fid)[4]))
                self._recompute()
            elif step_to >= now_ms:
                break

    def _recompute(self) -> None:
        if not self._flows:
            return
        caps: dict = {}
        members: dict = {}
        for fid, f in self._flows.items():
            for ep, cap in ((("e", f[0]), self._egress), (("i", f[1]), self._ingress)):
                caps.setdefault(ep, cap)
                members.setdefault(ep, []).append(fid)
        unfixed = set(self._flows)
        while unfixed:
            share, ep = min(
                (caps[ep] / sum(1 for x in m if x in unfixed), ep)
                for ep, m in members.items()
                if any(x in unfixed for x in m)
            )
            for fid in [x for x in members[ep] if x in unfixed]:
                f = self._flows[fid]
                f[3] = max(share, 0.0)
                unfixed.discard(fid)
                other = ("i", f[1]) if ep[0] == "e" else ("e", f[0])
                if other != ep:
                    caps[other] = max(0.0, caps[other] - share)
            caps[ep] = 0.0

    def submit(self, src: int, dst: int, num_bytes: float, now_ms: float, payload) -> None:
        self._advance(now_ms)
        self._flows[self._next_fid] = [src, dst, max(1.0, float(num_bytes)), 0.0, payload]
        self._next_fid += 1
        self._recompute()

    def pop_completed(self, now_ms: float) -> list[tuple[float, object]]:
        """Advance to now_ms and return [(finish_ms, payload)] for flows
        that completed by then — finish_ms is the flow's TRUE completion
        time, which can precede now_ms when the caller's clock jumped."""
        self._advance(now_ms)
        ready = [(t, p) for t, p in self._finished if t <= now_ms]
        self._finished = [(t, p) for t, p in self._finished if t > now_ms]
        return ready

    def next_completion_ms(self) -> Optional[float]:
        if self._finished:
            return min(t for t, _ in self._finished)
        return self._internal_next_ms()


@dataclass
class _Req:
    arrival_ms: float
    remaining_prefill: int
    generated: int = 0
    first_token_ms: float = -1.0
    last_token_ms: float = -1.0
    prefill_start_ms: float = -1.0
    xfer_submit_ms: float = -1.0
    pool_arrival_ms: float = 0.0  # when the req joined its CURRENT pool
    gaps: list = field(default_factory=list)
    is_initial_burst: bool = False


class _Pool:
    """One worker pool: each worker runs back-to-back passes over its own
    queue; workers become free at busy_until and are driven lazily.

    Requests can sit in a queue with a pool_arrival_ms in the future
    (a pass computed in one loop iteration timestamps its completions and
    replacements at the pass END): a pass must never consume a request
    that has not arrived by the pass start, or causality breaks and the
    limit cycle locks a phantom phase."""

    def __init__(self, n: int):
        self.busy_until = [0.0] * n
        self.queues: list[list[_Req]] = [[] for _ in range(n)]
        self._rr = 0

    def next_worker(self) -> int:
        widx = self._rr % len(self.queues)
        self._rr += 1
        return widx

    def dispatch(self, req: _Req, now_ms: float, widx: Optional[int] = None) -> None:
        # busy_until is deliberately NOT bumped here: it means only "the
        # worker is executing until t". A dispatch can carry a FUTURE
        # timestamp (a pass computed in one loop iteration timestamps its
        # outputs at the pass end), and bumping would freeze the worker
        # until then, swallowing every pass it could still run in between.
        # next_start() = max(busy_until, earliest arrival) covers idleness.
        if widx is None:
            widx = self.next_worker()
        req.pool_arrival_ms = now_ms
        self.queues[widx].append(req)

    def next_start(self, widx: int, eligible) -> float:
        """Earliest time worker widx can start a pass over its ELIGIBLE
        queued requests (inf if none): the worker must be free AND at
        least one eligible request must have arrived."""
        arrivals = [r.pool_arrival_ms for r in self.queues[widx] if eligible(r)]
        if not arrivals:
            return math.inf
        return max(self.busy_until[widx], min(arrivals))


def evaluate_disagg(
    wl: WorkloadSpec,
    prefill_eng: EngineSpec,
    decode_eng: EngineSpec,
    prefill_timing: TimingModel,
    decode_timing: TimingModel,
    spec: DisaggSpec,
    backend: str = "vllm",
    warmup_generations: int = 4,
    window_generations: int = 4,
    initial_stagger_ms: float = 0.0,
) -> QueueingReport:
    """Run the tandem pass-calendar recursion for a closed-loop workload.

    ``prefill_timing`` / ``decode_timing`` are separate so heterogeneous
    deployments (different GPUs or parallelisms per pool) price each stage
    with its own estimators; pass the same object for homogeneous setups.

    ``initial_stagger_ms`` spaces the initial burst's arrivals: the tandem
    system is MULTI-STABLE (the steady limit cycle depends on the initial
    cohort phase — e.g. a "large slow prefill batch" cycle vs a "small
    fast batch" pipeline cycle), so single-phase results are one attractor
    among several. Use ``evaluate_disagg_mixed`` for phase-robust output.
    """
    if wl.concurrency is None:
        raise ValueError("the disagg tandem model requires a closed-loop workload")
    c = wl.concurrency
    prefill = _Pool(spec.num_prefill_workers)
    decode = _Pool(spec.num_decode_workers)
    fabric = _TransferFabric(spec) if (spec.kv_bytes_per_token > 0 and spec.egress_bytes_per_s > 0) else None
    transfer_bytes = wl.isl * spec.kv_bytes_per_token

    for k in range(c):
        t0 = k * initial_stagger_ms
        prefill.dispatch(_Req(arrival_ms=t0, remaining_prefill=wl.effective_isl, is_initial_burst=True), t0)

    completions = 0
    warmup_reqs = warmup_generations * c
    target = (warmup_generations + window_generations) * c
    steady_start_ms = None
    now = 0.0

    ttft_transient = Distribution()
    ttft_steady = Distribution()
    itl = Distribution()
    tpot = Distribution()
    e2e = Distribution()
    xfer_durations: list[float] = []
    prefill_waits: list[float] = []
    steady_completions = 0

    def run_prefill_pass(widx: int, start_ms: float) -> float:
        """One static prefill pass: queued prompts (up to kappa) share the
        token budget; completers emit their FIRST token at pass end."""
        nonlocal completions, steady_start_ms, steady_completions
        q = prefill.queues[widx]
        budget = prefill_eng.max_num_batched_tokens
        arrived = [r for r in q if r.pool_arrival_ms <= start_ms]
        cap = spec.prefill_inflight_cap or len(arrived)
        batch_count = 0
        batch_isl = 0
        batch_prefix = 0
        finished: list[_Req] = []
        for r in arrived[:cap]:
            if budget <= 0:
                break
            if r.prefill_start_ms < 0:
                r.prefill_start_ms = start_ms
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
        end = start_ms + prefill_timing.prefill_ms(batch_count, batch_isl // batch_count, batch_prefix // batch_count)
        for r in finished:
            q.remove(r)
            # the prefill worker emits the first (TTFT) token off the final
            # chunk's logits — user-visible from this stage
            r.generated = 1
            r.first_token_ms = end
            r.last_token_ms = end
            ttft_ms = end - r.arrival_ms
            if r.is_initial_burst:
                ttft_transient.add(ttft_ms)
            elif completions >= warmup_reqs:
                ttft_steady.add(ttft_ms)
                prefill_waits.append(r.prefill_start_ms - r.arrival_ms)
            if r.generated >= wl.osl:
                _complete(r, end)  # osl == 1 finishes on the prefill worker
            elif fabric is None:
                decode.dispatch(r, end)
            else:
                dst = decode.next_worker()
                r.xfer_submit_ms = end
                fabric.submit(widx, dst, transfer_bytes, end, (dst, r))
        return end

    def run_decode_pass(widx: int, start_ms: float) -> float:
        """One decode iteration: the running set (capped at max_num_seqs)
        emits one token each; no prefill compute on decode workers."""
        q = decode.queues[widx]
        emitters = [r for r in q if r.generated < wl.osl and r.pool_arrival_ms <= start_ms][: decode_eng.max_num_seqs]
        if not emitters:
            return start_ms
        ctx = sum(wl.isl + r.generated for r in emitters) // len(emitters)
        end = start_ms + decode_timing.decode_ms(len(emitters), ctx)
        for r in emitters:
            r.generated += 1
            r.gaps.append(end - r.last_token_ms)  # gap 1 carries the handoff
            r.last_token_ms = end
            if r.generated >= wl.osl:
                q.remove(r)
                _complete(r, end)
        return end

    def _complete(r: _Req, end_ms: float) -> None:
        nonlocal completions, steady_start_ms, steady_completions
        completions += 1
        if completions == warmup_reqs:
            steady_start_ms = end_ms
        if completions > warmup_reqs and not r.is_initial_burst:
            steady_completions += 1
            for g in r.gaps:
                itl.add(g)
            if r.gaps:
                tpot.add(sum(r.gaps) / len(r.gaps))
            e2e.add(end_ms - r.arrival_ms)
        prefill.dispatch(_Req(arrival_ms=end_ms, remaining_prefill=wl.effective_isl), end_ms)

    def _any_req(_r: _Req) -> bool:
        return True

    def _decoding(r: _Req) -> bool:
        return r.generated < wl.osl

    max_iters = 200 * (warmup_generations + window_generations) * max(1, wl.osl)
    for _ in range(max_iters):
        if completions >= target:
            break
        t_pf = min((prefill.next_start(i, _any_req) for i in range(len(prefill.queues))), default=math.inf)
        t_dc = min((decode.next_start(i, _decoding) for i in range(len(decode.queues))), default=math.inf)
        t_tr = fabric.next_completion_ms() if fabric and fabric.has_flows() else None
        now = min(t_pf, t_dc, t_tr if t_tr is not None else math.inf)
        if now == math.inf:
            raise RuntimeError(
                f"disagg tandem recursion stalled (C={c}, {spec.num_prefill_workers}P"
                f"{spec.num_decode_workers}D) — invalid configuration"
            )

        if fabric is not None:
            for t_done, (dst, r) in fabric.pop_completed(now):
                xfer_durations.append(t_done - r.xfer_submit_ms)
                decode.dispatch(r, t_done, widx=dst)

        for i in range(len(prefill.queues)):
            if prefill.next_start(i, _any_req) <= now:
                prefill.busy_until[i] = run_prefill_pass(i, now)
        for i in range(len(decode.queues)):
            if decode.next_start(i, _decoding) <= now:
                decode.busy_until[i] = run_decode_pass(i, now)
    else:
        raise RuntimeError("disagg tandem recursion did not converge within max_iters")

    window_ms = now - (steady_start_ms if steady_start_ms is not None else 0.0)
    throughput = steady_completions / (window_ms / 1000.0) if window_ms > 0 else 0.0

    return QueueingReport(
        ttft_steady=ttft_steady,
        ttft_transient=ttft_transient,
        itl=itl,
        tpot=tpot,
        e2e=e2e,
        throughput_rps=throughput,
        output_tokens_per_s=throughput * wl.osl,
        backend=backend,
        mode="disagg",
        num_requests=wl.num_requests,
        kv_transfer_ms=(sum(xfer_durations) / len(xfer_durations)) if xfer_durations else 0.0,
        prefill_queue_ms=(sum(prefill_waits) / len(prefill_waits)) if prefill_waits else 0.0,
    )


def evaluate_disagg_mixed(
    wl: WorkloadSpec,
    prefill_eng: EngineSpec,
    decode_eng: EngineSpec,
    prefill_timing: TimingModel,
    decode_timing: TimingModel,
    spec: DisaggSpec,
    backend: str = "vllm",
    phases: int = 4,
) -> QueueingReport:
    """Phase-robust tandem output: an equal-weight mixture over a
    deterministic set of initial-arrival staggers.

    The tandem system is multi-stable — the steady limit cycle depends on
    the initial cohort phase (simultaneous arrivals lock a large-batch
    prefill cycle; spread arrivals lock a small-batch pipeline cycle, with
    TTFTs differing by multiples). A single phase is therefore one
    attractor among several, and which one a real deployment lands in is
    set by arrival jitter outside the model. The mixture over staggers
    spanning [0, t_solo_prefill] is the phase-agnostic estimate; it stays
    deterministic (no RNG) and each component is a valid limit cycle.
    """
    t_solo = max(1e-6, prefill_timing.prefill_ms(1, wl.isl, wl.prefix))
    offsets = [k * t_solo / max(1, phases - 1) for k in range(max(1, phases))]
    reps = [
        evaluate_disagg(
            wl, prefill_eng, decode_eng, prefill_timing, decode_timing, spec, backend, initial_stagger_ms=off
        )
        for off in offsets
    ]

    def _merge(get) -> Distribution:
        out = Distribution()
        for rep in reps:
            dist = get(rep)
            if not dist.values:
                continue
            w = 1.0 / len(dist.values)
            for v in dist.values:
                out.add(v, w)
        return out

    n = len(reps)
    return QueueingReport(
        ttft_steady=_merge(lambda r: r.ttft_steady),
        ttft_transient=_merge(lambda r: r.ttft_transient),
        itl=_merge(lambda r: r.itl),
        tpot=_merge(lambda r: r.tpot),
        e2e=_merge(lambda r: r.e2e),
        throughput_rps=sum(r.throughput_rps for r in reps) / n,
        output_tokens_per_s=sum(r.output_tokens_per_s for r in reps) / n,
        backend=backend,
        mode="disagg",
        num_requests=wl.num_requests,
        kv_transfer_ms=sum(r.kv_transfer_ms for r in reps) / n,
        prefill_queue_ms=sum(r.prefill_queue_ms for r in reps) / n,
    )
