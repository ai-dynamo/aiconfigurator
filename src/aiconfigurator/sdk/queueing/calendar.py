# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pass-calendar limit-cycle evaluator.

For a stationary closed-loop workload, the continuous-batching engine is a
deterministic dynamical system: identical requests + closed-loop arrivals
make every pass a pure function of the previous state, so the system enters
a limit cycle after the initial admission staircase. This module evaluates
that recursion at pass granularity (aggregate slot state, no event heap, no
RNG) and reads TTFT/ITL/TPOT distributions off the cycle.

KV capacity is modeled by block counting: each request holds
ceil(resident_tokens / block_size) blocks; admission and decode growth
allocate against `kv_capacity_tokens`, and exhaustion triggers
preemption-with-recompute (LIFO by default, matching vLLM v1). This covers
the KV-constrained regime where effective concurrency is limited by memory
rather than by max_num_seqs, including the waiting-queue delay it induces.
Block sharing across requests (prefix reuse) is NOT counted — with
`prefix > 0` under KV pressure the model overcounts memory; see the design
doc's failure-mode table.

This is an evaluation of the scheduling algorithm's own arithmetic — not a
statistical fit and not a per-request simulation. Provenance of the step
semantics: dynamo lib/mocker scheduler/vllm/core.rs, transcribed via the
parity-verified DES oracle in tools/queueing_oracle (0.0% error vs mocker on
every reported metric for uniform closed-loop workloads).

Backend calendars:
  - vllm    : fused pass — unified token budget, running set spends first,
              chunked prefill shares the remainder (VALIDATED vs mocker/DES)
  - trtllm  : fused pass like vllm (max_num_tokens budget); optional
              GUARANTEED_NO_EVICT admission cap (STRUCTURAL, not yet
              validated against a trtllm oracle)
  - sglang  : alternating passes — dedicated prefill batches pause decode,
              decode passes run alone; ITL spikes are whole prefill batches
              (STRUCTURAL, not yet validated against the sglang mocker)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .spec import Distribution, EngineSpec, QueueingReport, TimingModel, WorkloadSpec


@dataclass
class _Slot:
    """One in-flight request. `computed` counts tokens whose KV is resident
    and computed (the cached prefix enters as an initial credit); the known
    sequence is isl + generated, so `remaining = isl + generated - computed`
    is the scheduling demand (prompt chunks, recompute after preemption, or
    the single pending decode token)."""

    computed: int
    generated: int = 0
    arrival_ms: float = 0.0
    held_blocks: int = 0
    first_token_ms: float = -1.0
    last_token_ms: float = -1.0
    gaps: list = field(default_factory=list)
    num_preemptions: int = 0
    is_initial_burst: bool = False


class _EngineState:
    """Waiting/running queues + block-counted KV, mirroring the DES core."""

    def __init__(self, wl: WorkloadSpec, eng: EngineSpec):
        self.wl = wl
        self.eng = eng
        self.block_size = eng.block_size
        self.kv_capacity_blocks = eng.kv_capacity_tokens // eng.block_size if eng.kv_capacity_tokens else None
        self.used_blocks = 0
        self.waiting: list[_Slot] = []
        self.running: list[_Slot] = []
        self.total_preemptions = 0

    def new_request(self, arrival_ms: float, initial_burst: bool = False) -> _Slot:
        # fresh requests start with the cached-prefix credit
        return _Slot(computed=self.wl.prefix, arrival_ms=arrival_ms, is_initial_burst=initial_burst)

    def seq_len(self, s: _Slot) -> int:
        return self.wl.isl + s.generated

    def remaining(self, s: _Slot) -> int:
        return self.seq_len(s) - s.computed

    def try_allocate(self, s: _Slot, target_tokens: int) -> bool:
        need = math.ceil(target_tokens / self.block_size) - s.held_blocks
        if need <= 0:
            return True
        if self.kv_capacity_blocks is not None and self.used_blocks + need > self.kv_capacity_blocks:
            return False
        s.held_blocks += need
        self.used_blocks += need
        return True

    def free_all(self, s: _Slot) -> None:
        self.used_blocks -= s.held_blocks
        s.held_blocks = 0

    def preempt_one(self, exclude: _Slot | None = None) -> _Slot | None:
        """vLLM v1 recompute preemption: the victim frees its blocks, loses
        all computed progress (generated tokens remain part of the sequence
        and are recomputed; no prefix credit is assumed on recompute since
        the cached blocks may have been evicted), and returns to the front
        of the waiting queue."""
        candidates = self.running if exclude is None else [s for s in self.running if s is not exclude]
        if not candidates:
            return None
        victim = candidates[-1] if self.eng.preemption_mode == "lifo" else candidates[0]
        self.running.remove(victim)
        self.free_all(victim)
        victim.computed = 0
        victim.num_preemptions += 1
        self.total_preemptions += 1
        self.waiting.insert(0, victim)
        return victim


class BaseCalendar:
    """One engine iteration ("pass") over the engine state."""

    name = "base"
    validated = False

    def admission_cap(self, wl: WorkloadSpec, eng: EngineSpec) -> int:
        return eng.max_num_seqs

    def step(self, state: _EngineState, timing: TimingModel) -> tuple[float, list[_Slot]]:
        """Advance one pass; return (pass_duration_ms, emitting_slots)."""
        raise NotImplementedError


class _PrefillAgg:
    """Mocker-style prefill batch aggregation (mean isl / mean prefix)."""

    def __init__(self):
        self.count = 0
        self.total_isl = 0
        self.total_prefix = 0

    def add(self, prompt_before: int, prompt_tokens: int) -> None:
        self.count += 1
        self.total_isl += prompt_before + prompt_tokens
        self.total_prefix += prompt_before

    def latency_ms(self, timing: TimingModel) -> float:
        if self.count == 0:
            return 0.0
        return timing.prefill_ms(self.count, self.total_isl // self.count, self.total_prefix // self.count)


class FusedCalendar(BaseCalendar):
    """vLLM-v1-style fused pass, transcribed from core.rs: the running set
    spends the unified budget first (decode-ready slots one token each,
    mid-prefill slots their next chunk), then waiting requests are admitted
    while budget, max_num_seqs, and KV blocks allow. Prompt completion emits
    in the same pass; KV exhaustion preempts with recompute; no admissions
    after a preemption within the pass."""

    name = "vllm"
    validated = True

    _BLOCKED = "blocked"
    _SELF_PREEMPTED = "self_preempted"

    def _schedule(self, state: _EngineState, s: _Slot, budget: int, agg: _PrefillAgg, allow_preempt_others: bool):
        """Schedule s's next chunk of work. Returns (tokens_used, status)."""
        wl, eng = state.wl, state.eng
        remaining = state.remaining(s)
        if remaining <= 0:
            return 0, "ready"
        if not eng.enable_chunked_prefill and remaining > budget:
            return 0, self._BLOCKED
        chunk = min(remaining, budget)
        if chunk <= 0:
            return 0, self._BLOCKED
        is_decode_token = remaining == 1 and s.generated > 0
        target = s.computed + chunk
        while not state.try_allocate(s, target):
            victim = state.preempt_one(exclude=s) if allow_preempt_others else None
            if victim is None:
                if is_decode_token:
                    # decode-growth allocation failure without a victim:
                    # skip this token this pass (core.rs emit loop), do not
                    # block the rest of the running set
                    return 0, "skip"
                # partial prefill progress up to whole blocks that still fit
                if state.kv_capacity_blocks is None:
                    return 0, self._BLOCKED
                free_blocks = state.kv_capacity_blocks - state.used_blocks
                fit = (s.held_blocks + free_blocks) * state.block_size
                chunk = min(chunk, max(0, fit - s.computed))
                if chunk <= 0:
                    return 0, self._BLOCKED
                target = s.computed + chunk
                if not state.try_allocate(s, target):
                    return 0, self._BLOCKED
                break
        prompt_before = min(s.computed, wl.isl)
        s.computed = target
        prompt_tokens = min(s.computed, wl.isl) - prompt_before
        if prompt_tokens > 0:
            agg.add(prompt_before, prompt_tokens)
        return chunk, "ok"

    def step(self, state, timing):
        wl, eng = state.wl, state.eng
        budget = eng.max_num_batched_tokens
        agg = _PrefillAgg()
        preemptions_before = state.total_preemptions

        # 1) running set first (core.rs: while running and budget > 0)
        for s in list(state.running):
            if budget <= 0:
                break
            if s not in state.running or s.generated >= wl.osl:
                continue
            used, status = self._schedule(state, s, budget, agg, allow_preempt_others=True)
            if status == self._BLOCKED:
                break
            if status == "skip":
                continue
            budget -= used

        # 2) waiting admissions; none after a preemption this pass
        while (
            state.total_preemptions == preemptions_before
            and state.waiting
            and budget > 0
            and len(state.running) < self.admission_cap(wl, eng)
        ):
            s = state.waiting[0]
            used, status = self._schedule(state, s, budget, agg, allow_preempt_others=True)
            if status in (self._BLOCKED, self._SELF_PREEMPTED):
                break
            state.waiting.pop(0)
            state.running.append(s)
            budget -= used

        prefill_ms = agg.latency_ms(timing)

        # 3) emission: slots fully caught up emit one token this pass
        emitters = [s for s in state.running if state.remaining(s) <= 0 and s.generated < wl.osl]
        decode_ms = 0.0
        if emitters:
            ctx = sum(state.seq_len(s) for s in emitters) // len(emitters)
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
    iterations run alone. Budgets are max_prefill_tokens per batch with
    per-request chunks capped at chunked_prefill_size. Retraction under KV
    exhaustion is approximated as recompute preemption."""

    name = "sglang"
    validated = False

    def step(self, state, timing):
        wl, eng = state.wl, state.eng
        while state.waiting and len(state.running) < self.admission_cap(wl, eng):
            state.running.append(state.waiting.pop(0))

        # extend (prompt-phase) work pending? prompt incomplete or recompute
        extend = [s for s in state.running if s.computed < wl.isl or state.remaining(s) > 1]
        if extend:
            budget = eng.max_prefill_tokens or eng.max_num_batched_tokens
            chunk_cap = eng.chunked_prefill_size or budget
            agg = _PrefillAgg()
            emitters = []
            for s in extend:
                if budget <= 0:
                    break
                if s not in state.running:
                    continue
                chunk = min(state.remaining(s), budget, chunk_cap)
                target = s.computed + chunk
                while not state.try_allocate(s, target):
                    if state.preempt_one(exclude=s) is None:
                        chunk = 0
                        break
                if chunk <= 0:
                    continue
                prompt_before = min(s.computed, wl.isl)
                s.computed = target
                agg.add(prompt_before, min(s.computed, wl.isl) - prompt_before)
                budget -= chunk
                if state.remaining(s) <= 0:
                    emitters.append(s)
            if agg.count == 0:
                return 0.0, []
            return agg.latency_ms(timing), emitters

        emitters = []
        for s in list(state.running):
            if s.generated >= wl.osl or s not in state.running:
                continue
            if not state.try_allocate(s, s.computed + 1) and (
                state.preempt_one(exclude=s) is None or not state.try_allocate(s, s.computed + 1)
            ):
                continue
            s.computed += 1
            emitters.append(s)
        if not emitters:
            return 0.0, []
        ctx = sum(state.seq_len(s) for s in emitters) // len(emitters)
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
    c = wl.concurrency
    if c < 1:
        raise ValueError("concurrency must be >= 1")
    if (
        eng.kv_capacity_tokens is not None
        and math.ceil((wl.isl + 1) / eng.block_size) > eng.kv_capacity_tokens // eng.block_size
    ):
        raise ValueError("a single request exceeds kv_capacity_tokens")

    state = _EngineState(wl, eng)
    for _ in range(c):
        state.waiting.append(state.new_request(0.0, initial_burst=True))
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

    max_passes = 400 * (warmup_generations + window_generations) * max(1, wl.osl)
    for _ in range(max_passes):
        if completions >= target:
            break
        duration, emitters = calendar.step(state, timing)
        if not emitters and duration <= 0.0:
            raise RuntimeError(
                f"pass-calendar stalled (backend={backend}, C={c}, "
                f"budget={eng.max_num_batched_tokens}, "
                f"kv={eng.kv_capacity_tokens}) — invalid configuration"
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
            state.free_all(s)
            state.running.remove(s)
            state.waiting.append(state.new_request(now))
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
        total_preemptions=state.total_preemptions,
        completions_observed=completions,
    )
