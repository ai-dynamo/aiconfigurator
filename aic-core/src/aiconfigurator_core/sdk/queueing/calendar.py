# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pass-calendar limit-cycle evaluator.

For a stationary closed-loop workload, the continuous-batching engine is a
deterministic dynamical system: identical requests + closed-loop arrivals
make every pass a pure function of the previous state, so the system enters
a limit cycle after the initial admission staircase. This module evaluates
that recursion at pass granularity (aggregate slot state, no event heap, no
KV manager, no RNG) and reads TTFT/ITL/TPOT distributions off the cycle.

This is an evaluation of the scheduling algorithm's own arithmetic — not a
statistical fit and not a per-request simulation. Step semantics are
anchored to the vLLM v1 scheduler's behavior: one token budget shared by
decode tokens and prefill chunks, the running set served before the waiting
queue in admission order, and chunked prefill consuming whatever budget the
decodes leave. The full clause provenance table and validation record live
in docs/design/queueing_model.md; upstream scheduling changes surface as
validation-gate drift (design doc §6).

Backend calendars:
  - vllm    : fused pass — unified token budget, running decodes spend first,
              chunked prefill shares the remainder (VALIDATED, see design doc §5)
  - trtllm  : fused pass like vllm (max_num_tokens budget); optional
              GUARANTEED_NO_EVICT admission cap (STRUCTURAL, not yet
              validated against a trtllm oracle)
  - sglang  : mixed-chunk pass by default (prefill chunks share the iteration
              with the running decodes — matches AIC's SGLang agg deployment
              rule, which sets enable_mixed_chunk=true); with mixed chunk off,
              alternating passes — dedicated prefill batches pause decode and
              ITL spikes are whole prefill batches
              (STRUCTURAL, not yet validated against an SGLang reference)
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .spec import Distribution, EngineSpec, QueueingReport, TimingModel, WorkloadSpec


# eq=False: slots are identity objects. Cohorts of identical requests reach
# field-equal states in the limit cycle, and slots.index() must find THIS
# slot, not the first field-equal one.
@dataclass(eq=False)
class _Slot:
    remaining_prefill: int
    generated: int = 0
    arrival_ms: float = 0.0
    # when the scheduler can first SEE this request (arrival + client/
    # frontend turnaround); TTFT is still measured from arrival_ms
    eligible_ms: float = 0.0
    first_token_ms: float = -1.0
    last_token_ms: float = -1.0
    # per-slot shape: equals the workload's (isl, osl) for fixed-shape
    # workloads; drawn from the deterministic quantile streams when
    # WorkloadSpec.isl_quantiles / osl_quantiles are set. Heterogeneity
    # lives INSIDE the batch — a mixture of homogeneous runs cannot
    # represent it (each component would keep its own convoy structure,
    # which is exactly what shape diversity destroys; measured h20e
    # trtllm tp4, isl cv=0.25: steady TTFT collapses 1.8s -> 0.66s while
    # throughput moves <10%).
    isl: int = 0
    osl: int = 0
    gaps: list = field(default_factory=list)
    is_initial_burst: bool = False


class BaseCalendar:
    """One engine iteration ("pass") over the aggregate slot state."""

    name = "base"
    validated = False

    def admission_cap(self, wl: WorkloadSpec, eng: EngineSpec) -> int:
        return eng.max_num_seqs

    def step(
        self, slots: list[_Slot], wl: WorkloadSpec, eng: EngineSpec, timing: TimingModel
    ) -> tuple[float, list[_Slot]]:
        """Advance one pass; return (pass_duration_ms, emitting_slots)."""
        raise NotImplementedError


class FusedCalendar(BaseCalendar):
    """vLLM-v1-style fused pass: decode-ready slots spend one budget token
    each (and emit), prefilling slots consume the remaining budget as chunks
    in admission order; prefill completion emits in the same pass."""

    name = "vllm"
    validated = True

    def step(self, slots, wl, eng, timing):
        budget = eng.max_num_batched_tokens
        prefill_completers: list[_Slot] = []
        decode_emitters: list[_Slot] = []
        batch_count = 0
        batch_total_isl = 0
        batch_total_prefix = 0

        for s in slots:
            if s.remaining_prefill > 0:
                if budget <= 0:
                    continue
                if not eng.enable_chunked_prefill and s.remaining_prefill > budget:
                    # chunked prefill off: the scheduler stops admitting once
                    # a whole prompt no longer fits the remaining budget
                    break
                chunk = min(s.remaining_prefill, budget)
                computed_before = wl.prefix + (max(1, s.isl - wl.prefix) - s.remaining_prefill)
                s.remaining_prefill -= chunk
                budget -= chunk
                batch_count += 1
                batch_total_isl += computed_before + chunk
                batch_total_prefix += computed_before
                if s.remaining_prefill == 0:
                    prefill_completers.append(s)
            elif s.generated < s.osl:
                if budget <= 0:
                    continue
                budget -= 1
                decode_emitters.append(s)

        emitters = prefill_completers + decode_emitters
        # a genuinely mixed pass prefers the fused mixed-pass timing hook
        # (one combined batch — shared non-attention cost paid once); the
        # prefill+decode sum is the fallback for timing models without it
        mixed = getattr(timing, "mixed_pass_ms", None)
        if mixed is not None and batch_count > 0 and decode_emitters:
            chunk_tokens = batch_total_isl - batch_total_prefix
            mean_dec_isl = sum(s.isl for s in decode_emitters) // len(decode_emitters)
            mean_dec_osl = sum(s.osl for s in decode_emitters) // len(decode_emitters)
            return mixed(chunk_tokens, len(decode_emitters), mean_dec_isl, mean_dec_osl, wl.prefix), emitters

        prefill_ms = 0.0
        if batch_count > 0:
            mean_isl = batch_total_isl // batch_count
            mean_prefix = batch_total_prefix // batch_count
            prefill_ms = timing.prefill_ms(batch_count, mean_isl, mean_prefix)
        decode_ms = 0.0
        if decode_emitters:
            # prefill completers are NOT decode rows: the fused pass samples
            # their first token off the final chunk's logits (gpu_model_runner
            # logits_indices) — no extra decode-row cost, no budget token
            ctx = sum(s.isl + s.generated for s in decode_emitters) // len(decode_emitters)
            decode_ms = timing.decode_ms(len(decode_emitters), ctx)
        return prefill_ms + decode_ms, emitters


class TrtllmCalendar(FusedCalendar):
    """TRT-LLM in-flight batching is fused like vLLM. GUARANTEED_NO_EVICT
    admits a request only if KV for its full max length is reservable, which
    caps effective concurrency below max_num_seqs."""

    name = "trtllm"
    # h20e_sxm / trtllm 1.3.0rc20, Qwen3-32B tp4, isl4096/osl256, C 1..64:
    # chunked-off ITL bimodal signature reproduced (spike 533-546 pred vs
    # 541-590 meas), chunked-on spike tracks the budget (1067 pred vs ~1062
    # meas), TPOT <=2% at C>=32, KV-capped deep queueing TTFT +-3% at 2-4x
    # overload; known boundary: mid-bs TTFT residuals are timing-layer TPOT
    # bias amplified by osl (failure mode 16), not calendar structure
    validated = True

    def admission_cap(self, wl, eng):
        cap = eng.max_num_seqs
        if eng.guaranteed_no_evict and eng.kv_capacity_tokens:
            cap = min(cap, max(1, eng.kv_capacity_tokens // (wl.isl + wl.osl)))
        return cap


class AlternatingCalendar(BaseCalendar):
    """SGLang calendar, two modes selected by ``EngineSpec.enable_mixed_chunk``:

    - mixed chunk (default — AIC's generator deploys SGLang agg with
      ``enable_mixed_chunk=true``): prefill chunks and the running decodes
      share one iteration (SGLang merges the extend batch with the decode
      batch). Structurally fused like vLLM, except decodes do not debit the
      prefill token budget.
    - mixed chunk off (SGLang's own server default): prefill batches run as
      dedicated iterations (decode paused — the structural source of SGLang
      ITL spikes), decode iterations run alone.

    Budget semantics follow SGLang's ``PrefillAdder`` (schedule_policy.py,
    verified against 0.5.14 source): ``chunked_prefill_size`` is the
    per-iteration extend-token budget SHARED across the batch
    (``alloc = min(extend_input_len, rem_chunk_tokens)``) — NOT a
    per-request chunk cap; ``max_prefill_tokens`` separately caps the
    admitted input tokens; and in mixed mode the running decode rows debit
    both budgets (``num_mixed_decode_tokens``)."""

    name = "sglang"
    # h20e_sxm / sglang 0.5.14, Qwen3-32B tp1, isl4096/osl256, C 1..32, both
    # branches: mixed-chunk spike tracks the shared chunk budget (3992/4010
    # pred vs 3943/3960 meas at C=16/32), X +-3-7%; alternating (server
    # default) TPOT within 1% and X exact at C=32, cohort-locked prefill
    # waves reproduced (spike mass ~0.4% ~= 1/osl, amplitude = one cohort
    # wave). TTFT residuals are timing-layer bias amplified by osl (failure
    # mode 16). Measured with a clean thread-per-slot client; aiperf triggers
    # a deterministic +1x-prefill TTFT artifact in non-mixed mode (tool-side)
    validated = True

    def step(self, slots, wl, eng, timing):
        if eng.enable_mixed_chunk:
            return self._mixed_step(slots, wl, eng, timing)
        return self._alternating_step(slots, wl, eng, timing)

    def _prefill_batch(self, prefilling, wl, eng, mixed_decode_tokens=0):
        """Consume the per-iteration extend budget over `prefilling` slots in
        admission order; return (batch_count, mean_isl, mean_prefix, completers)."""
        input_cap = eng.max_prefill_tokens or eng.max_num_batched_tokens
        chunk_cap = eng.chunked_prefill_size or input_cap
        budget = min(chunk_cap, input_cap) - mixed_decode_tokens
        completers = []
        batch_count = 0
        batch_total_isl = 0
        batch_total_prefix = 0
        for s in prefilling:
            if budget <= 0:
                break
            chunk = min(s.remaining_prefill, budget)
            computed_before = wl.prefix + (max(1, s.isl - wl.prefix) - s.remaining_prefill)
            s.remaining_prefill -= chunk
            budget -= chunk
            batch_count += 1
            batch_total_isl += computed_before + chunk
            batch_total_prefix += computed_before
            if s.remaining_prefill == 0:
                completers.append(s)
        mean_isl = batch_total_isl // batch_count if batch_count else 0
        mean_prefix = batch_total_prefix // batch_count if batch_count else 0
        return batch_count, mean_isl, mean_prefix, completers

    def _mixed_step(self, slots, wl, eng, timing):
        # snapshot decode-ready BEFORE consuming prefill: a slot completing
        # its prefill in this pass emits once (as a completer), not twice
        prefilling = [s for s in slots if s.remaining_prefill > 0]
        decode_emitters = [s for s in slots if s.remaining_prefill == 0 and s.generated < s.osl]
        batch_count = 0
        mean_isl = mean_prefix = 0
        completers: list[_Slot] = []
        if prefilling:
            # mixed decode rows debit the extend budget (PrefillAdder's
            # num_mixed_decode_tokens)
            batch_count, mean_isl, mean_prefix, completers = self._prefill_batch(
                prefilling, wl, eng, mixed_decode_tokens=len(decode_emitters)
            )
        emitters = completers + decode_emitters

        mixed = getattr(timing, "mixed_pass_ms", None)
        if mixed is not None and batch_count > 0 and decode_emitters:
            chunk_tokens = batch_count * max(0, mean_isl - mean_prefix)
            mean_dec_isl = sum(s.isl for s in decode_emitters) // len(decode_emitters)
            mean_dec_osl = sum(s.osl for s in decode_emitters) // len(decode_emitters)
            return mixed(chunk_tokens, len(decode_emitters), mean_dec_isl, mean_dec_osl, wl.prefix), emitters

        prefill_ms = 0.0
        if batch_count:
            prefill_ms = timing.prefill_ms(batch_count, mean_isl, mean_prefix)
        decode_ms = 0.0
        if decode_emitters:
            ctx = sum(s.isl + s.generated for s in decode_emitters) // len(decode_emitters)
            decode_ms = timing.decode_ms(len(decode_emitters), ctx)
        return prefill_ms + decode_ms, emitters

    def _alternating_step(self, slots, wl, eng, timing):
        prefilling = [s for s in slots if s.remaining_prefill > 0]
        if prefilling:
            batch_count, mean_isl, mean_prefix, completers = self._prefill_batch(prefilling, wl, eng)
            return timing.prefill_ms(batch_count, mean_isl, mean_prefix), completers

        emitters = [s for s in slots if s.generated < s.osl]
        if not emitters:
            return 0.0, []
        ctx = sum(s.isl + s.generated for s in emitters) // len(emitters)
        return timing.decode_ms(len(emitters), ctx), emitters


CALENDARS: dict[str, BaseCalendar] = {
    "vllm": FusedCalendar(),
    "trtllm": TrtllmCalendar(),
    "sglang": AlternatingCalendar(),
}


def evaluate_closed_loop(
    wl: WorkloadSpec,
    eng: EngineSpec,
    timing: TimingModel,
    backend: str = "vllm",
    warmup_generations: int = 4,
    window_generations: int = 4,
    ttft_anchor: str = "none",
) -> QueueingReport:
    """Run the pass-calendar recursion for a closed-loop workload.

    One run yields both regimes: the initial burst of C simultaneous
    arrivals produces the transient admission staircase; after
    `warmup_generations` request generations (staircase + cohort echo
    decay) the limit cycle is sampled for `window_generations`.

    ttft_anchor: "none" (default) reports the recursion's own steady TTFT —
    exact for the idealized zero-turnaround client the DES oracle models.
    "identity" relocates the steady TTFT distribution onto the Little's-law
    accounting identity E[TTFT] = C/X - (osl-1)E[TPOT] - turnaround (shape
    from the calendar, location from the identity) — the phase-robust
    estimate for real closed-loop clients, whose measured steady TTFT is
    invariant to client turnaround (see design doc §6.16).
    """
    if wl.concurrency is None:
        raise ValueError("evaluate_closed_loop requires a closed-loop workload")
    calendar = CALENDARS[backend]
    c = min(wl.concurrency, calendar.admission_cap(wl, eng))
    if c < 1:
        raise ValueError("admission cap rejected all concurrency")

    # deterministic per-slot shape streams (fixed-shape workloads yield the
    # nominal isl/osl forever, reproducing the homogeneous recursion exactly);
    # osl uses an offset start so isl/osl strata pair pseudo-independently
    from .spec import _shape_stream

    isl_stream = _shape_stream(wl.isl_quantiles, wl.isl)
    osl_stream = _shape_stream(wl.osl_quantiles, wl.osl)
    if wl.osl_quantiles:
        for _ in range(len(wl.osl_quantiles) // 2):
            next(osl_stream)

    def _new_slot(**kw) -> _Slot:
        isl_i = next(isl_stream)
        osl_i = next(osl_stream)
        return _Slot(
            remaining_prefill=max(1, isl_i - wl.prefix), isl=isl_i, osl=osl_i, **kw
        )

    mean_osl = (
        sum(wl.osl_quantiles) / len(wl.osl_quantiles) if wl.osl_quantiles else wl.osl
    )

    slots = [_new_slot(is_initial_burst=True) for _ in range(c)]
    pending: list[_Slot] = []  # dispatched replacements not yet visible to the scheduler
    now = 0.0
    prev_pass_start = 0.0
    completions = 0
    warmup_reqs = warmup_generations * c
    target = (warmup_generations + window_generations) * c
    steady_start_ms = None

    ttft_transient = Distribution()
    ttft_steady = Distribution()
    itl = Distribution()
    tpot = Distribution()
    e2e = Distribution()
    steady_completions = 0

    max_osl = max(wl.osl_quantiles) if wl.osl_quantiles else wl.osl
    max_passes = 200 * (warmup_generations + window_generations) * max(1, max_osl)
    for _ in range(max_passes):
        if completions >= target:
            break
        # Admission horizon of the pass STARTING now: a synchronous
        # scheduler builds it at its start (arrivals up to `now` make it);
        # an async scheduler built it when the previous pass started, so
        # only arrivals visible by then are in it (one-pass lookahead).
        horizon = prev_pass_start if eng.async_scheduling else now
        if pending:
            admitted = [s for s in pending if s.eligible_ms <= horizon]
            if admitted:
                slots.extend(admitted)  # completion order == FCFS admission order
                pending = [s for s in pending if s.eligible_ms > horizon]
        pass_start = now
        duration, emitters = calendar.step(slots, wl, eng, timing)
        if not emitters and duration <= 0.0:
            if pending:
                # engine idle until the next replacement becomes visible;
                # an idle scheduler builds the wake pass immediately, so
                # the lookahead horizon catches up to the wake instant
                now = max(now, min(s.eligible_ms for s in pending))
                prev_pass_start = now
                continue
            raise RuntimeError(
                f"pass-calendar stalled (backend={backend}, C={c}, "
                f"budget={eng.max_num_batched_tokens}) — invalid configuration"
            )
        prev_pass_start = pass_start
        now += duration

        finished: list[_Slot] = []
        for s in emitters:
            s.generated += 1
            if s.first_token_ms < 0:
                s.first_token_ms = now
                ttft_ms = now - s.arrival_ms
                if s.is_initial_burst:
                    ttft_transient.add(ttft_ms)
                elif completions >= warmup_reqs:
                    ttft_steady.add(ttft_ms)
            else:
                s.gaps.append(now - s.last_token_ms)
            s.last_token_ms = now
            if s.generated >= s.osl:
                finished.append(s)

        for s in finished:
            completions += 1
            if completions == warmup_reqs:
                steady_start_ms = now
            if completions > warmup_reqs and not s.is_initial_burst:
                steady_completions += 1
                for g in s.gaps:
                    itl.add(g)
                if s.gaps:
                    tpot.add(sum(s.gaps) / len(s.gaps))
                e2e.add(now - s.arrival_ms)
            idx = slots.index(s)
            slots[idx : idx + 1] = []
            # closed loop: the client dispatches the replacement at the
            # completion instant (arrival_ms=now, the TTFT origin); the
            # scheduler sees it only after the frontend turnaround
            pending.append(
                _new_slot(arrival_ms=now, eligible_ms=now + wl.turnaround_ms)
            )
    else:
        raise RuntimeError("pass-calendar did not converge within max_passes")

    window_ms = now - (steady_start_ms if steady_start_ms is not None else 0.0)
    throughput = steady_completions / (window_ms / 1000.0) if window_ms > 0 else 0.0

    if ttft_anchor == "identity" and throughput > 0 and ttft_steady.values:
        # Little's-law anchor for the saturated closed loop. Each of the C
        # slots cycles through (TTFT + (osl-1) ITL gaps + client turnaround),
        # so cycle time == C / X exactly, and the steady TTFT MEAN is the
        # accounting identity
        #     E[TTFT] = C/X - (osl-1) * E[TPOT] - turnaround
        # -- independent of arrival phase. The deterministic recursion locks
        # arrivals to one phase of the pass structure; the phase choice only
        # REDISTRIBUTES time between TTFT and TPOT under this identity (it
        # cannot change their sum), and real engines with ms-scale timing
        # jitter do not stay on any single phase (measured: adding +15 ms of
        # client turnaround on a live vLLM 0.24 server moved steady TTFT p50
        # by 0.1 ms). So the calendar contributes the distribution SHAPE and
        # the identity pins its location; no client-side parameter survives.
        identity_mean = (
            wl.concurrency / throughput * 1000.0
            - max(0.0, mean_osl - 1.0) * tpot.mean
            - wl.turnaround_ms
        )
        if identity_mean > 0:
            delta = identity_mean - ttft_steady.mean
            ttft_steady = ttft_steady.shifted(delta)
            e2e = e2e.shifted(delta) if e2e.values else e2e

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
        num_requests=wl.num_requests,
    )


def evaluate_open_loop(
    wl: WorkloadSpec,
    eng: EngineSpec,
    timing: TimingModel,
    backend: str = "vllm",
    warmup_requests: int = 128,
    window_requests: int = 512,
) -> QueueingReport:
    """Run the pass-calendar recursion for an OPEN-loop workload.

    Arrivals come from a deterministic exponential quantile stream (64
    inverse-CDF strata, normalized to an exact mean of 1/request_rate,
    golden-ratio-stride rotation) — the zero-RNG counterpart of Poisson
    arrivals, same construction as the per-slot shape streams. Unlike the
    closed loop, arrivals do not self-throttle: the in-flight population
    floats, arrivals beyond the admission cap wait in a FIFO queue, and
    TTFT includes that queue wait. There is no Little's-law TTFT anchor
    here — the closed-loop cycle identity does not apply to open arrivals.

    Raises RuntimeError when the waiting queue diverges (request_rate at or
    beyond the deployment's capacity — no steady state exists).
    """
    if wl.request_rate is None:
        raise ValueError("evaluate_open_loop requires an open-loop workload (request_rate)")
    from math import gcd, log

    from .spec import _shape_stream

    calendar = CALENDARS[backend]
    cap = calendar.admission_cap(wl, eng)

    isl_stream = _shape_stream(wl.isl_quantiles, wl.isl)
    osl_stream = _shape_stream(wl.osl_quantiles, wl.osl)
    if wl.osl_quantiles:
        for _ in range(len(wl.osl_quantiles) // 2):
            next(osl_stream)
    mean_osl = sum(wl.osl_quantiles) / len(wl.osl_quantiles) if wl.osl_quantiles else wl.osl

    # deterministic exponential inter-arrival strata, exact-mean normalized
    k = 64
    strata = [-log(1.0 - (i + 0.5) / k) for i in range(k)]
    scale = (1000.0 / wl.request_rate) / (sum(strata) / k)
    stride = max(1, round(k * 0.6180339887))
    while gcd(stride, k) != 1:
        stride += 1

    total = warmup_requests + window_requests
    pending: list[_Slot] = []
    t_arr = 0.0
    for n in range(total):
        t_arr += strata[(n * stride) % k] * scale
        isl_i = next(isl_stream)
        pending.append(
            _Slot(
                remaining_prefill=max(1, isl_i - wl.prefix),
                isl=isl_i,
                osl=next(osl_stream),
                arrival_ms=t_arr,
                eligible_ms=t_arr + wl.turnaround_ms,
            )
        )

    slots: list[_Slot] = []
    waiting: list[_Slot] = []  # visible to the scheduler, above the admission cap
    now = 0.0
    prev_pass_start = 0.0
    completions = 0
    steady_start_ms = None

    ttft_transient = Distribution()
    ttft_steady = Distribution()
    itl = Distribution()
    tpot = Distribution()
    e2e = Distribution()
    steady_completions = 0

    max_osl = max(wl.osl_quantiles) if wl.osl_quantiles else wl.osl
    max_passes = 400 * total * max(1, max_osl) // max(1, min(cap, total))
    for _ in range(max_passes):
        if completions >= total:
            break
        horizon = prev_pass_start if eng.async_scheduling else now
        if pending:
            visible = [s for s in pending if s.eligible_ms <= horizon]
            if visible:
                waiting.extend(visible)  # FIFO by arrival (pending is arrival-ordered)
                pending = [s for s in pending if s.eligible_ms > horizon]
        while waiting and len(slots) < cap:
            slots.append(waiting.pop(0))
        if len(waiting) > max(4 * cap, total // 2):
            raise RuntimeError(
                f"open-loop waiting queue diverged (backend={backend}, "
                f"rate={wl.request_rate}/s, cap={cap}) — request_rate is at or "
                "beyond this deployment's capacity; no steady state exists"
            )

        pass_start = now
        duration, emitters = calendar.step(slots, wl, eng, timing)
        if not emitters and duration <= 0.0:
            nxt = min((s.eligible_ms for s in pending), default=float("inf"))
            if waiting:
                # capped out but idle passes cannot happen with waiting work
                raise RuntimeError("open-loop recursion stalled with waiting work")
            if nxt == float("inf"):
                break  # drained: all arrivals processed
            now = max(now, nxt)
            prev_pass_start = now
            continue
        prev_pass_start = pass_start
        now += duration

        finished: list[_Slot] = []
        for s in emitters:
            s.generated += 1
            if s.first_token_ms < 0:
                s.first_token_ms = now
                ttft_ms = now - s.arrival_ms
                if completions >= warmup_requests:
                    ttft_steady.add(ttft_ms)
                else:
                    ttft_transient.add(ttft_ms)
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
            idx = slots.index(s)
            slots[idx : idx + 1] = []
    else:
        raise RuntimeError("open-loop pass-calendar did not converge within max_passes")

    window_ms = now - (steady_start_ms if steady_start_ms is not None else 0.0)
    throughput = steady_completions / (window_ms / 1000.0) if window_ms > 0 else 0.0

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
        num_requests=wl.num_requests,
    )
