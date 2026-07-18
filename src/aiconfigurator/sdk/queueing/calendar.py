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
statistical fit and not a per-request simulation. Provenance of the step
semantics: dynamo lib/mocker scheduler/vllm/core.rs, transcribed via the
parity-verified DES oracle in tools/queueing_oracle (0.0% error vs mocker on
every reported metric for uniform closed-loop workloads).

Backend calendars:
  - vllm    : fused pass — unified token budget, running decodes spend first,
              chunked prefill shares the remainder (VALIDATED vs mocker/DES)
  - trtllm  : fused pass like vllm (max_num_tokens budget); optional
              GUARANTEED_NO_EVICT admission cap (STRUCTURAL, not yet
              validated against a trtllm oracle)
  - sglang  : alternating passes — dedicated prefill batches pause decode,
              decode passes run alone; ITL spikes are whole prefill batches
              (STRUCTURAL, not yet validated against the sglang mocker)
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .spec import Distribution, EngineSpec, QueueingReport, TimingModel, WorkloadSpec


@dataclass
class _Slot:
    remaining_prefill: int
    generated: int = 0
    arrival_ms: float = 0.0
    first_token_ms: float = -1.0
    last_token_ms: float = -1.0
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
        emitters: list[_Slot] = []
        batch_count = 0
        batch_total_isl = 0
        batch_total_prefix = 0

        for s in slots:
            if s.remaining_prefill > 0:
                if budget <= 0:
                    continue
                if not eng.enable_chunked_prefill and s.remaining_prefill > budget:
                    break  # blocked: whole prompt must fit (core.rs semantics)
                chunk = min(s.remaining_prefill, budget)
                computed_before = wl.prefix + (wl.effective_isl - s.remaining_prefill)
                s.remaining_prefill -= chunk
                budget -= chunk
                batch_count += 1
                batch_total_isl += computed_before + chunk
                batch_total_prefix += computed_before
                if s.remaining_prefill == 0:
                    emitters.append(s)
            elif s.generated < wl.osl:
                if budget <= 0:
                    continue
                budget -= 1
                emitters.append(s)

        prefill_ms = 0.0
        if batch_count > 0:
            mean_isl = batch_total_isl // batch_count
            mean_prefix = batch_total_prefix // batch_count
            prefill_ms = timing.prefill_ms(batch_count, mean_isl, mean_prefix)
        decode_ms = 0.0
        if emitters:
            ctx = sum(wl.isl + s.generated for s in emitters) // len(emitters)
            decode_ms = timing.decode_ms(len(emitters), ctx)
        return prefill_ms + decode_ms, emitters


class TrtllmCalendar(FusedCalendar):
    """TRT-LLM in-flight batching is fused like vLLM. GUARANTEED_NO_EVICT
    admits a request only if KV for its full max length is reservable, which
    caps effective concurrency below max_num_seqs."""

    name = "trtllm"
    validated = False

    def admission_cap(self, wl, eng):
        cap = eng.max_num_seqs
        if eng.guaranteed_no_evict and eng.kv_capacity_tokens:
            cap = min(cap, max(1, eng.kv_capacity_tokens // (wl.isl + wl.osl)))
        return cap


class AlternatingCalendar(BaseCalendar):
    """SGLang-style calendar: prefill batches run as dedicated iterations
    (decode paused — the structural source of SGLang ITL spikes), decode
    iterations run alone. Budgets are max_prefill_tokens per prefill batch
    with per-request chunks capped at chunked_prefill_size."""

    name = "sglang"
    validated = False

    def step(self, slots, wl, eng, timing):
        prefilling = [s for s in slots if s.remaining_prefill > 0]
        if prefilling:
            budget = eng.max_prefill_tokens or eng.max_num_batched_tokens
            chunk_cap = eng.chunked_prefill_size or budget
            emitters = []
            batch_count = 0
            batch_total_isl = 0
            batch_total_prefix = 0
            for s in prefilling:
                if budget <= 0:
                    break
                chunk = min(s.remaining_prefill, budget, chunk_cap)
                computed_before = wl.prefix + (wl.effective_isl - s.remaining_prefill)
                s.remaining_prefill -= chunk
                budget -= chunk
                batch_count += 1
                batch_total_isl += computed_before + chunk
                batch_total_prefix += computed_before
                if s.remaining_prefill == 0:
                    emitters.append(s)
            mean_isl = batch_total_isl // batch_count
            mean_prefix = batch_total_prefix // batch_count
            return timing.prefill_ms(batch_count, mean_isl, mean_prefix), emitters

        emitters = [s for s in slots if s.generated < wl.osl]
        if not emitters:
            return 0.0, []
        ctx = sum(wl.isl + s.generated for s in emitters) // len(emitters)
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
) -> QueueingReport:
    """Run the pass-calendar recursion for a closed-loop workload.

    One run yields both regimes: the initial burst of C simultaneous
    arrivals produces the transient admission staircase; after
    `warmup_generations` request generations (staircase + cohort echo
    decay) the limit cycle is sampled for `window_generations`.
    """
    if wl.concurrency is None:
        raise ValueError("evaluate_closed_loop requires a closed-loop workload")
    calendar = CALENDARS[backend]
    c = min(wl.concurrency, calendar.admission_cap(wl, eng))
    if c < 1:
        raise ValueError("admission cap rejected all concurrency")

    slots = [_Slot(remaining_prefill=wl.effective_isl, is_initial_burst=True) for _ in range(c)]
    now = 0.0
    completions = 0
    warmup_reqs = warmup_generations * c
    target = (warmup_generations + window_generations) * c
    steady_start_ms = None

    ttft_transient = Distribution()
    ttft_steady = Distribution()
    itl = Distribution()
    tpot = Distribution()
    steady_completions = 0

    max_passes = 200 * (warmup_generations + window_generations) * max(1, wl.osl)
    for _ in range(max_passes):
        if completions >= target:
            break
        duration, emitters = calendar.step(slots, wl, eng, timing)
        if not emitters and duration <= 0.0:
            raise RuntimeError(
                f"pass-calendar stalled (backend={backend}, C={c}, "
                f"budget={eng.max_num_batched_tokens}) — invalid configuration"
            )
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
            if s.generated >= wl.osl:
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
            idx = slots.index(s)
            slots[idx : idx + 1] = []
            slots.append(_Slot(remaining_prefill=wl.effective_isl, arrival_ms=now))
    else:
        raise RuntimeError("pass-calendar did not converge within max_passes")

    window_ms = now - (steady_start_ms if steady_start_ms is not None else 0.0)
    throughput = steady_completions / (window_ms / 1000.0) if window_ms > 0 else 0.0

    return QueueingReport(
        ttft_steady=ttft_steady,
        ttft_transient=ttft_transient,
        itl=itl,
        tpot=tpot,
        throughput_rps=throughput,
        output_tokens_per_s=throughput * wl.osl,
        backend=backend,
        mode="agg",
        num_requests=wl.num_requests,
    )
