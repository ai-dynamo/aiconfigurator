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
    computed tokens). The transfer moves the FULL context (per-request isl
    x kv_bytes_per_token): cached prefix saves prefill compute, not handoff
    bytes — the decode pool holds no copy of the prefix KV.

Rate matching is an OUTPUT here, not an input: for a candidate
(num_prefill_workers, num_decode_workers) the recursion yields throughput
and both stages' behavior directly — pool imbalance surfaces as prefill
queueing (TTFT) or decode saturation (ITL/throughput) instead of scalar
throughput derates.

Workload coverage follows the same fidelity contract as the agg calendars
(design doc §3.1): W0 closed loop, W1 open-loop rates (deterministic
exponential arrivals under the correlation-free per-period shuffle),
W2 shape marginals, W3 joint (isl, prefix, osl) strata / empirical
inter-arrival strata / verbatim ``arrival_trace`` replay. All shape and
arrival draws come from the SAME deterministic streams as the agg
calendars (``spec._shape_drawer`` / ``spec._interarrival_stream``), so
agg-vs-disagg comparisons see identical workloads. Open-loop arrivals are
routed to prefill workers by a static round-robin router at the
scheduler-visibility instant (arrival + turnaround + isl x ingest slope
— the same arrival-plane mapping as the agg calendar), matching TRT-LLM's
native disagg router; kv_router-style pending-queue admission is
approximated by ``prefill_inflight_cap=1``.

The router dispatch policy is exposed as ``prefill_inflight_cap`` (kappa):
  None  = engine-batched admission (all queued prompts share a prefill
          pass's token budget — matches the DES round-robin driver)
  1     = serialized prefills per worker (approximates a kv_router-style
          pending-queue admission; measured impact on TTFT mean ~20%)

KV-pressure semantics are NOT modeled (no admission gate, no
hold-until-transfer accounting): engines carrying ``kv_capacity_tokens`` /
``guaranteed_no_evict`` are rejected loudly rather than silently ignored —
the same honesty contract as the agg calendars.

Same methodology as the agg evaluator (`calendar.evaluate_closed_loop`):
deterministic pass-level recursion, no RNG, no per-token events. Validated
against the DES ``DisaggSimulator`` (tools/queueing_oracle) — the disagg
families of the validation gate compare the two with identical timing.
"""

from __future__ import annotations

import math
from bisect import insort
from dataclasses import dataclass, field
from typing import Optional

from .spec import (
    Distribution,
    EngineSpec,
    QueueingReport,
    TimingModel,
    WorkloadSpec,
    _interarrival_stream,
    _shape_drawer,
    workload_fidelity,
)


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
    # kappa: per-pass prefill batch cap on each prefill worker.
    # None = engine-batched (queued prompts share one pass up to the token
    # budget); 1 = solo-serial prefills. Measured mapping (h20e 2P1D,
    # trtllm 1.3.0rc20): TRT-LLM native disaggregated serving runs ctx
    # prefills solo-serial behind a static round-robin router — i.e.
    # kappa=1 (low-load TTFT within ~3%). At the saturation knee (rho -> 1)
    # the deterministic recursion locks the zero-wait pipeline attractor
    # for ANY kappa/phase while jitter drives real deployments into a
    # queued attractor — see design doc failure mode 15; the saturated
    # regime is unaffected.
    prefill_inflight_cap: Optional[int] = None
    # Serving-flow placement of the KV handoff. False (default): the prefill
    # worker's first token streams to the user immediately and the transfer
    # lands in the FIRST ITL GAP (the DES oracle's idealized flow — the gate
    # families pin this arm). True: the user-visible first token waits for
    # the transfer (decode-attach flow) — TTFT = prefill + handoff, gap 1
    # clean. Measured mapping: TRT-LLM native disaggregated serving and the
    # dynamo mocker/frontend are BOTH decode-attach (h20e slow-link A/B,
    # UCX_TLS=cuda_copy,tcp: c=1 TTFT absorbed the full 2.2 s transfer while
    # the decode ITL spike stayed unchanged — exp1 fabric appendix). On
    # NVLink the two flows differ by ~3 ms and are indistinguishable.
    handoff_in_ttft: bool = False

    def __post_init__(self) -> None:
        if self.num_prefill_workers < 1 or self.num_decode_workers < 1:
            raise ValueError("num_prefill_workers and num_decode_workers must be >= 1")
        if self.prefill_inflight_cap is not None and self.prefill_inflight_cap < 1:
            raise ValueError("prefill_inflight_cap must be None (engine-batched) or >= 1")


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
    isl: int
    prefix: int = 0
    osl: int = 1
    generated: int = 0
    first_token_ms: float = -1.0
    last_token_ms: float = -1.0
    prefill_start_ms: float = -1.0
    xfer_submit_ms: float = -1.0
    xfer_ms: float = 0.0
    # open loop: when the router/scheduler first SEES this request
    # (arrival + turnaround + isl x ingest slope); TTFT is measured from
    # arrival_ms — the same arrival-plane convention as the agg calendar
    eligible_ms: float = 0.0
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


def _reject_kv_pressure(prefill_eng: EngineSpec, decode_eng: EngineSpec) -> None:
    """KV-pressure honesty (same contract as the agg calendars): the tandem
    models no KV admission gate and no hold-until-transfer accounting, so
    accepting these inputs would silently return optimistic numbers."""
    for stage, eng in (("prefill", prefill_eng), ("decode", decode_eng)):
        if eng.kv_capacity_tokens or eng.guaranteed_no_evict:
            raise ValueError(
                f"the disagg tandem models no KV-pressure admission semantics ({stage} "
                "engine sets kv_capacity_tokens/guaranteed_no_evict): unset them, or "
                "size the deployment so KV never binds"
            )


@dataclass
class _TandemStats:
    """Raw accounting of one tandem run (report assembly is the caller's)."""

    ttft_transient: Distribution
    ttft_steady: Distribution
    itl: Distribution
    tpot: Distribution
    e2e: Distribution
    xfer_durations: list
    prefill_waits: list
    completions: int
    steady_completions: int
    steady_start_ms: Optional[float]
    end_ms: float

    @property
    def throughput_rps(self) -> float:
        window_ms = self.end_ms - (self.steady_start_ms if self.steady_start_ms is not None else 0.0)
        return self.steady_completions / (window_ms / 1000.0) if window_ms > 0 else 0.0


def _run_tandem(
    prefill: _Pool,
    decode: _Pool,
    fabric: Optional[_TransferFabric],
    prefill_eng: EngineSpec,
    decode_eng: EngineSpec,
    prefill_timing: TimingModel,
    decode_timing: TimingModel,
    spec: DisaggSpec,
    pending: list,
    target: int,
    warmup_reqs: int,
    max_iters: int,
    on_complete=None,
    transient_prewarmup: bool = False,
    max_backlog: Optional[int] = None,
    stall_msg: str = "",
) -> _TandemStats:
    """The tandem event loop — the ONE copy of the pass/handoff semantics
    every disagg entry point drives (closed loop, open loop, trace replay,
    session lanes). ``pending`` holds not-yet-visible arrivals as
    ``(eligible_ms, seq, req)`` tuples sorted by (eligible, seq);
    ``on_complete(req, end_ms)`` runs after a request's accounting and may
    push follow-up work (a closed-loop replacement straight into the
    prefill pool, or a session lane's next turn insorted into ``pending``).
    ``transient_prewarmup`` buckets pre-warmup non-burst TTFTs as transient
    (the open-loop/sessions convention; the closed loop drops them).
    ``max_backlog`` arms the arrived-but-unstarted divergence guard (open
    loop only — endogenous arrivals self-throttle and pass None)."""
    completions = 0
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
    # KV-handoff outbox: passes are computed with FUTURE end timestamps in
    # worker-index order, so direct fabric.submit calls would reach the
    # fluid clock out of time order (worker 0's late pass end advances the
    # fabric past worker 1's earlier one, clamping its flow start — small
    # transfers then appear to take tens of ms). Flows queue here and join
    # the fabric through the event loop, in stamp order.
    xfer_outbox: list = []
    outbox_seq = [0]

    def submit_xfer(end_ms: float, src: int, dst: int, num_bytes: float, req: _Req) -> None:
        insort(xfer_outbox, (end_ms, outbox_seq[0], src, dst, num_bytes, req), key=lambda x: (x[0], x[1]))
        outbox_seq[0] += 1

    def run_prefill_pass(widx: int, start_ms: float) -> float:
        """One static prefill pass: queued prompts (up to kappa) share the
        token budget; completers emit their FIRST token at pass end."""
        nonlocal completions, steady_start_ms, steady_completions
        q = prefill.queues[widx]
        budget = prefill_eng.max_num_batched_tokens
        arrived = [r for r in q if r.pool_arrival_ms <= start_ms]
        # per-pass batch cap: kappa (router admission) AND the engine's own
        # max_batch_size — a ctx worker deployed at bs4 co-schedules at most
        # 4 prompts per pass regardless of the token budget (measured: cc
        # 48k window, kappa-only brackets missed by +34/-? while bs-capped
        # engine-batched admission tracks the live GNE scheduler)
        cap = min(spec.prefill_inflight_cap or len(arrived), prefill_eng.max_num_seqs)
        batch_count = 0
        batch_isl = 0
        batch_prefix = 0
        finished: list[_Req] = []
        for r in arrived[:cap]:
            if budget <= 0:
                break
            if not prefill_eng.enable_chunked_prefill and r.remaining_prefill > budget:
                # chunked prefill off: admission stops once a whole prompt no
                # longer fits the remaining budget (same rule as the agg
                # FusedCalendar; TRT-LLM disagg ctx workers deploy this way)
                break
            if r.prefill_start_ms < 0:
                r.prefill_start_ms = start_ms
            chunk = min(r.remaining_prefill, budget)
            computed_before = r.prefix + (max(1, r.isl - r.prefix) - r.remaining_prefill)
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
            if fabric is not None and spec.handoff_in_ttft and r.osl > 1:
                # decode-attach flow: the first token becomes user-visible
                # only after the KV handoff — emission happens at transfer
                # completion (TTFT = prefill + handoff, gap 1 clean)
                dst = decode.next_worker()
                r.xfer_submit_ms = end
                submit_xfer(end, widx, dst, r.isl * spec.kv_bytes_per_token, r)
                continue
            _emit_first_token(r, end)
            if r.generated >= r.osl:
                _complete(r, end)  # osl == 1 finishes on the prefill worker
            elif fabric is None:
                decode.dispatch(r, end)
            else:
                dst = decode.next_worker()
                r.xfer_submit_ms = end
                submit_xfer(end, widx, dst, r.isl * spec.kv_bytes_per_token, r)
        return end

    def _emit_first_token(r: _Req, t_ms: float) -> None:
        """The prefill worker samples the first (TTFT) token off the final
        chunk's logits; depending on the serving flow it is user-visible at
        the prefill pass end or at handoff completion."""
        r.generated = 1
        r.first_token_ms = t_ms
        r.last_token_ms = t_ms
        ttft_ms = t_ms - r.arrival_ms
        if r.is_initial_burst:
            ttft_transient.add(ttft_ms)
        elif completions >= warmup_reqs:
            ttft_steady.add(ttft_ms)
            prefill_waits.append(r.prefill_start_ms - r.arrival_ms)
        elif transient_prewarmup:
            # open loop/sessions: pre-warmup TTFTs are transient (same
            # bucketing as calendar.evaluate_open_loop)
            ttft_transient.add(ttft_ms)

    def run_decode_pass(widx: int, start_ms: float) -> float:
        """One decode iteration: the running set (capped at max_num_seqs)
        emits one token each; no prefill compute on decode workers."""
        q = decode.queues[widx]
        emitters = [r for r in q if r.generated < r.osl and r.pool_arrival_ms <= start_ms][: decode_eng.max_num_seqs]
        if not emitters:
            return start_ms
        ctx = sum(r.isl + r.generated for r in emitters) // len(emitters)
        end = start_ms + decode_timing.decode_ms(len(emitters), ctx)
        for r in emitters:
            r.generated += 1
            r.gaps.append(end - r.last_token_ms)  # gap 1 carries the handoff
            r.last_token_ms = end
            if r.generated >= r.osl:
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
        if on_complete is not None:
            on_complete(r, end_ms)

    def _any_req(_r: _Req) -> bool:
        return True

    def _decoding(r: _Req) -> bool:
        return r.generated < r.osl

    for _ in range(max_iters):
        if completions >= target:
            break
        t_pf = min((prefill.next_start(i, _any_req) for i in range(len(prefill.queues))), default=math.inf)
        t_dc = min((decode.next_start(i, _decoding) for i in range(len(decode.queues))), default=math.inf)
        t_tr = fabric.next_completion_ms() if fabric and fabric.has_flows() else None
        t_arr = pending[0][0] if pending else math.inf
        t_ob = xfer_outbox[0][0] if xfer_outbox else math.inf
        now = min(t_pf, t_dc, t_arr, t_ob, t_tr if t_tr is not None else math.inf)
        if now == math.inf:
            raise RuntimeError(f"disagg tandem recursion stalled ({stall_msg}) — invalid configuration")

        while pending and pending[0][0] <= now:
            _, _, r = pending.pop(0)
            prefill.dispatch(r, r.eligible_ms)
        if max_backlog is not None and (backlog := _prefill_backlog(prefill, now)) > max_backlog:
            raise RuntimeError(
                f"disagg prefill backlog diverged ({stall_msg}, backlog={backlog}) — "
                "request_rate is at or beyond this deployment's prefill capacity; "
                "no steady state exists"
            )

        if fabric is not None:
            while xfer_outbox and xfer_outbox[0][0] <= now:
                t_s, _, src, dst, num_bytes, r = xfer_outbox.pop(0)
                fabric.submit(src, dst, num_bytes, t_s, (dst, r))
            for t_done, (dst, r) in fabric.pop_completed(now):
                r.xfer_ms = t_done - r.xfer_submit_ms
                xfer_durations.append(r.xfer_ms)
                if r.first_token_ms < 0:  # decode-attach flow: TTFT lands here
                    _emit_first_token(r, t_done)
                decode.dispatch(r, t_done, widx=dst)

        for i in range(len(prefill.queues)):
            if prefill.next_start(i, _any_req) <= now:
                prefill.busy_until[i] = run_prefill_pass(i, now)
        for i in range(len(decode.queues)):
            if decode.next_start(i, _decoding) <= now:
                decode.busy_until[i] = run_decode_pass(i, now)
    else:
        raise RuntimeError("disagg tandem recursion did not converge within max_iters")

    return _TandemStats(
        ttft_transient=ttft_transient,
        ttft_steady=ttft_steady,
        itl=itl,
        tpot=tpot,
        e2e=e2e,
        xfer_durations=xfer_durations,
        prefill_waits=prefill_waits,
        completions=completions,
        steady_completions=steady_completions,
        steady_start_ms=steady_start_ms,
        end_ms=now,
    )


def _prefill_backlog(prefill: _Pool, now_ms: float) -> int:
    """Arrived-but-unstarted requests across the prefill pool."""
    return sum(
        1 for q in prefill.queues for r in q if r.prefill_start_ms < 0 and r.pool_arrival_ms <= now_ms
    )


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
    warmup_requests: int = 128,
    window_requests: int = 512,
    arrival_trace=None,
) -> QueueingReport:
    """Run the tandem pass-calendar recursion (closed OR open loop).

    ``prefill_timing`` / ``decode_timing`` are separate so heterogeneous
    deployments (different GPUs or parallelisms per pool) price each stage
    with its own estimators; pass the same object for homogeneous setups.

    Closed loop (``wl.concurrency``): fixed slots, the replacement is
    dispatched at the completion instant and becomes visible to the prefill
    pool after ``wl.turnaround_ms``; ``warmup_generations`` /
    ``window_generations`` split the limit cycle. ``initial_stagger_ms``
    spaces the initial burst's arrivals: the tandem system is MULTI-STABLE
    (the steady limit cycle depends on the initial cohort phase — e.g. a
    "large slow prefill batch" cycle vs a "small fast batch" pipeline
    cycle), so single-phase results are one attractor among several. Use
    ``evaluate_disagg_mixed`` for phase-robust output.

    Open loop (``wl.request_rate``): arrivals come from the same
    deterministic streams as ``calendar.evaluate_open_loop`` (exponential
    strata under the correlation-free per-period shuffle, or empirical
    ``arrival_quantiles``), or verbatim from ``arrival_trace`` tuples
    ``(arrival_ms, isl, prefix, osl)`` — pairing and ordering preserved,
    per-request diagnostics in ``QueueingReport.per_request`` (with
    ``xfer_ms``, the request's KV-handoff duration on the fabric). The
    router assigns arrivals round-robin across prefill workers at the
    scheduler-visibility instant; there is no burst phase to mix over, so
    a single evaluation is already phase-robust. ``warmup_requests`` /
    ``window_requests`` split the sequence. Raises RuntimeError when the
    prefill backlog diverges (request_rate at or beyond capacity). Note
    TTFT is prefill-side: a saturated transfer fabric or decode pool shows
    up in ITL/e2e/throughput, not TTFT.
    """
    closed_loop = wl.concurrency is not None
    if arrival_trace is not None and closed_loop:
        raise ValueError("arrival_trace requires an open-loop workload (request_rate)")
    if not closed_loop and initial_stagger_ms:
        raise ValueError("initial_stagger_ms shapes the closed-loop initial burst only")
    _reject_kv_pressure(prefill_eng, decode_eng)
    prefill = _Pool(spec.num_prefill_workers)
    decode = _Pool(spec.num_decode_workers)
    fabric = _TransferFabric(spec) if (spec.kv_bytes_per_token > 0 and spec.egress_bytes_per_s > 0) else None

    draw_shape, mean_osl, max_osl = _shape_drawer(wl)

    def _new_req(arrival_ms: float, **kw) -> _Req:
        isl_i, px_i, osl_i = draw_shape()
        return _Req(
            arrival_ms=arrival_ms,
            remaining_prefill=max(1, isl_i - px_i),
            isl=isl_i,
            prefix=px_i,
            osl=osl_i,
            **kw,
        )

    pending: list[_Req] = []  # open loop: not-yet-visible arrivals, scheduler-arrival order
    trace_order: list[_Req] = []
    if closed_loop:
        c = wl.concurrency
        for k in range(c):
            t0 = k * initial_stagger_ms
            prefill.dispatch(_new_req(t0, is_initial_burst=True), t0)
        warmup_reqs = warmup_generations * c
        target = (warmup_generations + window_generations) * c
    else:
        if arrival_trace is not None:
            target = len(arrival_trace)
            warmup_reqs = min(warmup_requests, max(0, target - 1))
            pending = [
                _Req(
                    arrival_ms=float(t_i),
                    remaining_prefill=max(1, int(isl_i) - int(px_i)),
                    isl=int(isl_i),
                    prefix=int(px_i),
                    osl=max(1, int(osl_i)),
                )
                for (t_i, isl_i, px_i, osl_i) in arrival_trace
            ]
            trace_order = list(pending)
            max_osl = max([max_osl] + [r.osl for r in pending])
        else:
            target = warmup_requests + window_requests
            warmup_reqs = warmup_requests
            gap_stream = _interarrival_stream(wl)
            t_arr = 0.0
            for _ in range(target):
                t_arr += next(gap_stream)
                pending.append(_new_req(t_arr))
        for r in pending:
            r.eligible_ms = r.arrival_ms + wl.turnaround_ms + r.isl * wl.ingest_us_per_token / 1000.0
        # arrival-plane mapping (see WorkloadSpec.ingest_us_per_token): the
        # router serves by *scheduler* arrival, and the size-dependent
        # ingest slope reorders near-simultaneous dispatches shortest-first.
        # Stable sort: exact ties keep dispatch order.
        if wl.ingest_us_per_token > 0:
            pending.sort(key=lambda r: r.eligible_ms)

    if closed_loop:

        def on_complete(r: _Req, end_ms: float) -> None:
            # the client dispatches the replacement at the completion
            # instant (arrival_ms=end_ms, the TTFT origin); the prefill pool
            # sees it only after the frontend turnaround — same
            # visibility-delay semantics as the agg calendar. At turnaround 0
            # the replacement lands exactly on the completion/worker-free
            # knife edge, which at high utilization locks the zero-wait
            # pipeline attractor real deployments never hold (validated:
            # h20e 2P1D kappa=1 at the saturation knee).
            prefill.dispatch(_new_req(end_ms), end_ms + wl.turnaround_ms)

        max_iters = 200 * (warmup_generations + window_generations) * max(1, max_osl)
        max_backlog = None
    else:
        on_complete = None
        # pure stall backstop (every iteration performs at least one discrete
        # event: an arrival release, a fabric completion, or a pass)
        max_iters = 400 * (target + 1) * max(1, max_osl)
        # backlog bound before declaring divergence: mirrors the agg open
        # loop's max(4*cap, total/2) with the tandem's admission analog
        # (kappa or the engine seq cap, per prefill worker)
        backlog_cap = spec.num_prefill_workers * (spec.prefill_inflight_cap or prefill_eng.max_num_seqs)
        max_backlog = max(4 * backlog_cap, target // 2)

    stats = _run_tandem(
        prefill,
        decode,
        fabric,
        prefill_eng,
        decode_eng,
        prefill_timing,
        decode_timing,
        spec,
        # (eligible, seq) tuples: seq is the dispatch index, so the sorted
        # order reproduces the stable ingest sort (ties keep dispatch order)
        [(r.eligible_ms, i, r) for i, r in enumerate(pending)],
        target,
        warmup_reqs,
        max_iters,
        on_complete=on_complete,
        transient_prewarmup=not closed_loop,
        max_backlog=max_backlog,
        stall_msg=f"{spec.num_prefill_workers}P{spec.num_decode_workers}D, rate={wl.request_rate}/s",
    )

    per_request = None
    if trace_order:
        per_request = [
            dict(
                arrival_ms=r.arrival_ms,
                isl=r.isl,
                prefix=r.prefix,
                osl=r.osl,
                ttft_ms=(r.first_token_ms - r.arrival_ms) if r.first_token_ms >= 0 else None,
                e2e_ms=(r.last_token_ms - r.arrival_ms) if r.last_token_ms >= 0 else None,
                xfer_ms=r.xfer_ms if r.xfer_submit_ms >= 0 else None,
            )
            for r in trace_order
        ]

    return QueueingReport(
        ttft_steady=stats.ttft_steady,
        ttft_transient=stats.ttft_transient,
        itl=stats.itl,
        tpot=stats.tpot,
        e2e=stats.e2e,
        throughput_rps=stats.throughput_rps,
        output_tokens_per_s=stats.throughput_rps * mean_osl,
        backend=backend,
        mode="disagg",
        num_requests=wl.num_requests,
        kv_transfer_ms=(sum(stats.xfer_durations) / len(stats.xfer_durations)) if stats.xfer_durations else 0.0,
        prefill_queue_ms=(sum(stats.prefill_waits) / len(stats.prefill_waits)) if stats.prefill_waits else 0.0,
        workload_fidelity=workload_fidelity(wl),
        per_request=per_request,
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
    warmup_requests: int = 128,
    window_requests: int = 512,
    arrival_trace=None,
) -> QueueingReport:
    """Phase-robust tandem output: an equal-weight mixture over a
    deterministic set of initial-arrival staggers.

    The closed-loop tandem system is multi-stable — the steady limit cycle
    depends on the initial cohort phase (simultaneous arrivals lock a
    large-batch prefill cycle; spread arrivals lock a small-batch pipeline
    cycle, with TTFTs differing by multiples). A single phase is therefore
    one attractor among several, and which one a real deployment lands in
    is set by arrival jitter outside the model. The mixture over staggers
    spanning [0, t_solo_prefill] is the phase-agnostic estimate; it stays
    deterministic (no RNG) and each component is a valid limit cycle.

    Open-loop workloads have no initial-cohort phase (arrivals are
    externally timed), so a single evaluation is already phase-robust and
    is returned directly.
    """
    if wl.concurrency is None:
        return evaluate_disagg(
            wl,
            prefill_eng,
            decode_eng,
            prefill_timing,
            decode_timing,
            spec,
            backend,
            warmup_requests=warmup_requests,
            window_requests=window_requests,
            arrival_trace=arrival_trace,
        )
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
        workload_fidelity=workload_fidelity(wl),
    )
