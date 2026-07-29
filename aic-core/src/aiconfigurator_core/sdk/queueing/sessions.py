# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Session-loop evaluator: the first W4 capability.

Agentic workloads (Claude-Code-style multi-turn sessions, tool loops) have
ENDOGENOUS arrivals: turn k+1 dispatches only after turn k completes plus a
think/tool gap, so the arrival process depends on the serving speed being
predicted. Neither the closed loop (fixed slots, immediate replacement) nor
the open loop (exogenous timestamps) can express that; exact-trace replay
sidesteps it by consuming MEASURED dispatch times, which requires having run
the workload once.

This evaluator closes the loop a priori: each session is a sequential LANE
of turns; turn k+1's dispatch time is an OUTPUT of the recursion
(completion_k + think gap), mapped to scheduler arrival through the same
plane as everywhere else (+ turnaround epsilon + isl x ingest slope). Lanes
contend for the same engine budget/admission through the backend calendar,
so cross-session interference, prefix-hot follow-up turns and burst pile-ups
all emerge structurally.

Approximations carried (documented, same as the cc-traces replay harness):
one outstanding request per lane (a Claude Code parent blocks on subagents;
overlapping subagents serialize into the lane), and per-turn prefixes are
inputs (compute them with trace.prefix_hits — within-session reuse dominates
at the 90%+ reuse typical of agent loops, so ordering sensitivity of the
oracle across lanes is second-order).
"""

from __future__ import annotations

from bisect import insort
from dataclasses import dataclass

from .calendar import CALENDARS, _Slot
from .disagg import DisaggSpec, _Pool, _reject_kv_pressure, _Req, _run_tandem, _TransferFabric
from .spec import Distribution, EngineSpec, QueueingReport, TimingModel, WorkloadSpec


@dataclass(frozen=True)
class SessionTurn:
    """One turn of a session lane.

    ``think_ms`` is the gap between the PREVIOUS turn's completion and this
    turn's dispatch (client think / tool execution time); for the first turn
    it offsets the session's start instead. Recorded traces reconstruct it as
    ``t_{k+1} - (t_k + api_time_k)`` — clamp/cap on the caller side when the
    recorded gaps contain human idle time that the replay should compress.
    """

    isl: int
    prefix: int
    osl: int
    think_ms: float = 0.0


def evaluate_sessions(
    sessions: list,
    eng: EngineSpec,
    timing: TimingModel,
    backend: str = "vllm",
    session_start_ms: list | None = None,
    stagger_ms: float = 0.0,
    turnaround_ms: float = 0.0,
    ingest_us_per_token: float = 0.0,
    warmup_requests: int = 0,
) -> QueueingReport:
    """Run the pass-calendar recursion over session lanes (W4).

    ``sessions``: list of lanes, each a list of SessionTurn. Lane i starts at
    ``session_start_ms[i]`` (or ``i * stagger_ms``). Returns the usual
    QueueingReport; ``per_request`` entries carry ``session``/``turn`` and
    the EMERGENT ``arrival_ms`` (dispatch instant produced by the recursion —
    compare it against a measured replay to audit the session dynamics
    themselves, not just the latency given arrivals).
    """
    if not sessions or not any(sessions):
        raise ValueError("sessions is empty")
    if session_start_ms is not None and len(session_start_ms) != len(sessions):
        raise ValueError("session_start_ms must match sessions length")
    calendar = CALENDARS[backend]

    all_turns = [t for lane in sessions for t in lane]
    total = len(all_turns)
    mean_isl = max(1, sum(t.isl for t in all_turns) // total)
    mean_osl = max(1, sum(t.osl for t in all_turns) // total)
    # internal spec: carries the plane constants and the mean shape the
    # admission cap arithmetic uses; the rate field is unused by lanes
    wl = WorkloadSpec(
        isl=mean_isl,
        osl=mean_osl,
        request_rate=1.0,
        turnaround_ms=turnaround_ms,
        ingest_us_per_token=ingest_us_per_token,
    )
    cap = calendar.admission_cap(wl, eng)

    def _make_slot(lane_idx: int, turn_idx: int, dispatch_ms: float) -> _Slot:
        t = sessions[lane_idx][turn_idx]
        prefix = min(t.prefix, max(0, t.isl - 1))
        s = _Slot(
            remaining_prefill=max(1, t.isl - prefix),
            isl=t.isl,
            osl=max(1, t.osl),
            prefix=prefix,
            arrival_ms=dispatch_ms,
            eligible_ms=dispatch_ms + turnaround_ms + t.isl * ingest_us_per_token / 1000.0,
        )
        lane_of[id(s)] = (lane_idx, turn_idx)
        return s

    lane_of: dict = {}
    results: dict = {}
    # pending sorted by eligible time (scheduler-arrival order, ties by
    # insertion — insort on a keyed list)
    pending: list = []

    def _enqueue(slot: _Slot) -> None:
        insort(pending, (slot.eligible_ms, len(lane_of), slot), key=lambda x: (x[0], x[1]))

    for i, lane in enumerate(sessions):
        if not lane:
            continue
        start = session_start_ms[i] if session_start_ms is not None else i * stagger_ms
        _enqueue(_make_slot(i, 0, start + lane[0].think_ms))

    slots: list[_Slot] = []
    waiting: list[_Slot] = []
    now = 0.0
    prev_pass_start = 0.0
    completions = 0
    steady_start_ms = None
    steady_completions = 0
    ttft_transient = Distribution()
    ttft_steady = Distribution()
    itl = Distribution()
    tpot = Distribution()
    e2e = Distribution()

    max_osl = max(t.osl for t in all_turns)
    max_passes = 400 * total * max(1, max_osl) // max(1, min(cap, total))
    for _ in range(max_passes):
        if completions >= total:
            break
        horizon = prev_pass_start if eng.async_scheduling else now
        while pending and pending[0][0] <= horizon:
            waiting.append(pending.pop(0)[2])
        while waiting and len(slots) < cap:
            slots.append(waiting.pop(0))

        pass_start = now
        duration, emitters = calendar.step(slots, wl, eng, timing)
        if not emitters and duration <= 0.0:
            if waiting:
                raise RuntimeError("session recursion stalled with waiting work")
            if not pending:
                break
            now = max(now, pending[0][0])
            prev_pass_start = now
            continue
        prev_pass_start = pass_start
        now += duration

        finished: list[_Slot] = []
        for s in emitters:
            s.generated += 1
            if s.first_token_ms < 0:
                s.first_token_ms = now
                if completions >= warmup_requests:
                    ttft_steady.add(now - s.arrival_ms)
                else:
                    ttft_transient.add(now - s.arrival_ms)
            else:
                s.gaps.append(now - s.last_token_ms)
            s.last_token_ms = now
            if s.generated >= s.osl:
                finished.append(s)

        for s in finished:
            completions += 1
            if completions == warmup_requests:
                steady_start_ms = now
            if completions > warmup_requests:
                steady_completions += 1
                for g in s.gaps:
                    itl.add(g)
                if s.gaps:
                    tpot.add(sum(s.gaps) / len(s.gaps))
                e2e.add(now - s.arrival_ms)
            slots.remove(s)
            li, ti = lane_of[id(s)]
            results[(li, ti)] = s
            if ti + 1 < len(sessions[li]):
                nxt = sessions[li][ti + 1]
                _enqueue(_make_slot(li, ti + 1, now + nxt.think_ms))
    else:
        raise RuntimeError("session pass-calendar did not converge within max_passes")

    window_ms = now - (steady_start_ms if steady_start_ms is not None else 0.0)
    throughput = steady_completions / (window_ms / 1000.0) if window_ms > 0 else 0.0

    per_request = []
    for li, lane in enumerate(sessions):
        for ti in range(len(lane)):
            s = results.get((li, ti))
            if s is None:
                continue
            per_request.append(
                dict(
                    session=li,
                    turn=ti,
                    arrival_ms=s.arrival_ms,
                    isl=s.isl,
                    prefix=s.prefix,
                    osl=s.osl,
                    ttft_ms=(s.first_token_ms - s.arrival_ms) if s.first_token_ms >= 0 else None,
                    e2e_ms=(s.last_token_ms - s.arrival_ms) if s.last_token_ms >= 0 else None,
                )
            )

    return QueueingReport(
        ttft_steady=ttft_steady,
        ttft_transient=ttft_transient,
        itl=itl,
        tpot=tpot,
        e2e=e2e,
        throughput_rps=throughput,
        output_tokens_per_s=throughput * mean_osl,
        backend=backend,
        mode="agg",
        num_requests=total,
        workload_fidelity=f"W4(session-loop, lanes={len(sessions)}, turns={total})",
        per_request=per_request,
    )


def evaluate_sessions_disagg(
    sessions: list,
    prefill_eng: EngineSpec,
    decode_eng: EngineSpec,
    prefill_timing: TimingModel,
    decode_timing: TimingModel,
    spec: DisaggSpec,
    backend: str = "vllm",
    session_start_ms: list | None = None,
    stagger_ms: float = 0.0,
    turnaround_ms: float = 0.0,
    ingest_us_per_token: float = 0.0,
    warmup_requests: int = 0,
) -> QueueingReport:
    """Session lanes over the disagg tandem (W4 x disagg).

    Same lane semantics as ``evaluate_sessions`` — turn k+1 dispatches at
    completion_k + think gap, mapped to scheduler arrival through the usual
    plane (turnaround + isl x ingest slope) — driven through the tandem's
    event loop (``disagg._run_tandem``: static-RR prefill router, per-request
    KV handoff on the max-min-fair fabric, first token prefill-side). The
    lane approximations of the agg evaluator carry over unchanged (one
    outstanding request per lane; per-turn prefixes are inputs).

    ``per_request`` entries carry ``session``/``turn``, the EMERGENT
    ``arrival_ms``, and ``xfer_ms`` (the turn's KV-handoff duration).
    """
    if not sessions or not any(sessions):
        raise ValueError("sessions is empty")
    if session_start_ms is not None and len(session_start_ms) != len(sessions):
        raise ValueError("session_start_ms must match sessions length")
    _reject_kv_pressure(prefill_eng, decode_eng)

    all_turns = [t for lane in sessions for t in lane]
    total = len(all_turns)
    mean_osl = max(1, sum(t.osl for t in all_turns) // total)
    max_osl = max(1, max(t.osl for t in all_turns))

    prefill = _Pool(spec.num_prefill_workers)
    decode = _Pool(spec.num_decode_workers)
    fabric = _TransferFabric(spec) if (spec.kv_bytes_per_token > 0 and spec.egress_bytes_per_s > 0) else None

    lane_of: dict = {}
    results: dict = {}
    pending: list = []  # (eligible_ms, seq, req) — _run_tandem's arrival feed

    def _enqueue_turn(lane_idx: int, turn_idx: int, dispatch_ms: float) -> None:
        t = sessions[lane_idx][turn_idx]
        prefix = min(t.prefix, max(0, t.isl - 1))
        r = _Req(
            arrival_ms=dispatch_ms,
            remaining_prefill=max(1, t.isl - prefix),
            isl=t.isl,
            prefix=prefix,
            osl=max(1, t.osl),
            eligible_ms=dispatch_ms + turnaround_ms + t.isl * ingest_us_per_token / 1000.0,
        )
        lane_of[id(r)] = (lane_idx, turn_idx)
        insort(pending, (r.eligible_ms, len(lane_of), r), key=lambda x: (x[0], x[1]))

    for i, lane in enumerate(sessions):
        if not lane:
            continue
        start = session_start_ms[i] if session_start_ms is not None else i * stagger_ms
        _enqueue_turn(i, 0, start + lane[0].think_ms)

    def on_complete(r: _Req, end_ms: float) -> None:
        li, ti = lane_of[id(r)]
        results[(li, ti)] = r
        if ti + 1 < len(sessions[li]):
            _enqueue_turn(li, ti + 1, end_ms + sessions[li][ti + 1].think_ms)

    stats = _run_tandem(
        prefill,
        decode,
        fabric,
        prefill_eng,
        decode_eng,
        prefill_timing,
        decode_timing,
        spec,
        pending,
        total,
        warmup_requests,
        400 * (total + 1) * max(1, max_osl),
        on_complete=on_complete,
        transient_prewarmup=True,
        # endogenous arrivals self-throttle (one outstanding per lane), so
        # the open-loop divergence guard stays unarmed
        max_backlog=None,
        stall_msg=f"{spec.num_prefill_workers}P{spec.num_decode_workers}D sessions",
    )

    per_request = []
    for li, lane in enumerate(sessions):
        for ti in range(len(lane)):
            r = results.get((li, ti))
            if r is None:
                continue
            per_request.append(
                dict(
                    session=li,
                    turn=ti,
                    arrival_ms=r.arrival_ms,
                    isl=r.isl,
                    prefix=r.prefix,
                    osl=r.osl,
                    ttft_ms=(r.first_token_ms - r.arrival_ms) if r.first_token_ms >= 0 else None,
                    e2e_ms=(r.last_token_ms - r.arrival_ms) if r.last_token_ms >= 0 else None,
                    xfer_ms=r.xfer_ms if r.xfer_submit_ms >= 0 else None,
                )
            )

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
        num_requests=total,
        kv_transfer_ms=(sum(stats.xfer_durations) / len(stats.xfer_durations)) if stats.xfer_durations else 0.0,
        prefill_queue_ms=(sum(stats.prefill_waits) / len(stats.prefill_waits)) if stats.prefill_waits else 0.0,
        workload_fidelity=f"W4(session-loop, lanes={len(sessions)}, turns={total})",
        per_request=per_request,
    )
