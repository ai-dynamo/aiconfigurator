# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Minimal discrete-event replica of Dynamo mocker's vLLM-style scheduler.

Semantics transcribed from dynamo `lib/mocker/src/scheduler/vllm/core.rs`
(local checkout, v1.3.0). One `execute_pass` models one engine iteration:

  1. spend the token budget over the *running* set first (chunked prefill
     continuation + one decode token of budget per caught-up request),
  2. admit from *waiting* while `len(running) < max_num_seqs`, no preemption
     happened this pass, and budget remains,
  3. pass duration = prefill_time(batch aggregates) + decode_time(ready set),
     both from a pluggable perf model (default: mocker's polynomial),
  4. every caught-up request emits exactly one token at pass end; KV pressure
     triggers LIFO/FIFO preemption with full recompute (vLLM v1 style).

KV accounting is block-based with hash-shared full prompt blocks, an LRU
inactive pool for prefix reuse, and anonymous partial/generated blocks.
`kv_mode="token"` disables sharing + prefix caching (pure token counting) so
the fidelity delta of simplified accounting can be measured directly.

Deliberately out of scope (v0): disagg P/D handoff, KV events, routers,
SGLang retraction semantics, multi-turn generated-block reuse.
"""

from __future__ import annotations

import heapq
import math
from collections import OrderedDict
from collections.abc import Callable, Hashable
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

# --------------------------------------------------------------------------
# Perf models
# --------------------------------------------------------------------------


class PolynomialPerfModel:
    """Exact replica of mocker `PerfModel::Polynomial` (common/perf_model.rs)."""

    def prefill_ms(self, batch_count: int, mean_isl: int, mean_prefix: int) -> float:
        if batch_count == 0:
            return 0.0
        tokens = float(batch_count * max(0, mean_isl - mean_prefix))
        return max(0.0, 4.209989e-07 * tokens * tokens + 1.518344e-02 * tokens + 1.650142e01)

    def decode_ms(self, batch_size: int, active_kv_tokens: int, context_length: int, total_kv_tokens: int) -> float:
        if batch_size == 0:
            return 0.0
        p = active_kv_tokens / total_kv_tokens if total_kv_tokens > 0 else 1.0
        return max(1.0, -25.74 * p * p + 54.01 * p + 5.74)


class CallbackPerfModel:
    """Adapter for AIC-style callbacks (predict_prefill / predict_decode).

    prefill_fn(batch, effective_isl, prefix) -> ms
    decode_fn(batch, context_length) -> ms        # AIC signature, osl fixed at 2
    """

    def __init__(self, prefill_fn: Callable[[int, int, int], float], decode_fn: Callable[[int, int], float]):
        self._prefill_fn = prefill_fn
        self._decode_fn = decode_fn

    def prefill_ms(self, batch_count: int, mean_isl: int, mean_prefix: int) -> float:
        if batch_count == 0:
            return 0.0
        return max(0.0, self._prefill_fn(batch_count, max(0, mean_isl - mean_prefix), mean_prefix))

    def decode_ms(self, batch_size: int, active_kv_tokens: int, context_length: int, total_kv_tokens: int) -> float:
        if batch_size == 0:
            return 0.0
        return max(1.0, self._decode_fn(batch_size, context_length))


# --------------------------------------------------------------------------
# KV block manager
# --------------------------------------------------------------------------


class KvManager:
    """Block-based KV accounting with hash sharing + LRU inactive pool.

    Approximates mocker's kvbm-logical backend:
      - hashed full prompt blocks are shared by refcount ("Active"),
      - refcount 0 moves a block to the LRU "Inactive" pool (prefix-reusable),
      - anonymous blocks (partial tails, generated tokens) are never shared,
      - a new allocation evicts the LRU inactive block when at capacity,
      - allocation fails only when capacity is exhausted and nothing is
        evictable -> caller preempts.

    Known simplification vs kvbm-logical: generated full blocks are freed
    anonymously instead of entering the inactive pool with sequence hashes,
    so multi-turn reuse of *generated* text is not modeled.
    """

    def __init__(self, num_blocks: int, block_size: int, enable_sharing: bool = True):
        self.capacity = num_blocks
        self.block_size = block_size
        self.enable_sharing = enable_sharing
        self.active: dict[Hashable, int] = {}  # hash -> refcount
        self.inactive: OrderedDict[Hashable, None] = OrderedDict()  # LRU, oldest first
        self.anon_blocks = 0

    @property
    def num_active_blocks(self) -> int:
        return len(self.active) + self.anon_blocks

    @property
    def used_blocks(self) -> int:
        return self.num_active_blocks + len(self.inactive)

    def match_prefix(self, hashes: tuple[Hashable, ...]) -> int:
        """Number of leading full blocks already cached (active or inactive)."""
        if not self.enable_sharing:
            return 0
        n = 0
        for h in hashes:
            if h in self.active or h in self.inactive:
                n += 1
            else:
                break
        return n

    def _make_room(self) -> bool:
        if self.used_blocks < self.capacity:
            return True
        if self.inactive:
            self.inactive.popitem(last=False)  # evict LRU
            return True
        return False

    def alloc_hashed(self, h: Hashable) -> bool:
        if not self.enable_sharing:
            return self.alloc_anon()
        if h in self.active:
            self.active[h] += 1
            return True
        if h in self.inactive:
            del self.inactive[h]
            self.active[h] = 1
            return True
        if not self._make_room():
            return False
        self.active[h] = 1
        return True

    def alloc_anon(self) -> bool:
        if not self._make_room():
            return False
        self.anon_blocks += 1
        return True

    def free_hashed(self, h: Hashable) -> None:
        if not self.enable_sharing:
            self.free_anon(1)
            return
        rc = self.active.get(h)
        if rc is None:
            return
        if rc > 1:
            self.active[h] = rc - 1
        else:
            del self.active[h]
            self.inactive[h] = None  # newest end of LRU

    def free_anon(self, n: int) -> None:
        self.anon_blocks -= n
        assert self.anon_blocks >= 0, "anon block accounting went negative"


# --------------------------------------------------------------------------
# Requests
# --------------------------------------------------------------------------


class Status(Enum):
    WAITING = "waiting"
    RUNNING = "running"
    PREEMPTED = "preempted"
    DONE = "done"


@dataclass
class Request:
    rid: int
    isl: int
    osl: int
    # hashes for the floor(isl / block_size) *full* prompt blocks
    prompt_hashes: tuple[Hashable, ...] = ()
    arrival_ms: float = 0.0

    # scheduler state
    status: Status = Status.WAITING
    computed: int = 0  # num_computed_tokens
    generated: int = 0
    held_hashed: list = field(default_factory=list)
    anon_blocks: int = 0
    num_preemptions: int = 0

    # metrics
    dispatch_ms: float = -1.0
    admitted_ms: float = -1.0
    cached_tokens_at_admission: int = 0
    token_times: list = field(default_factory=list)
    completed_ms: float = -1.0

    def seq_len(self) -> int:
        return self.isl + self.generated

    def total_blocks(self) -> int:
        return len(self.held_hashed) + self.anon_blocks

    def allocate_blocks(self, need: int, kv: KvManager) -> int:
        """Allocate up to `need` new blocks for this sequence; returns count."""
        n_full_prompt = len(self.prompt_hashes)
        got = 0
        for _ in range(need):
            idx = self.total_blocks()  # next block index
            if idx < n_full_prompt:
                ok = kv.alloc_hashed(self.prompt_hashes[idx])
                if ok:
                    self.held_hashed.append(self.prompt_hashes[idx])
            else:
                ok = kv.alloc_anon()
                if ok:
                    self.anon_blocks += 1
            if not ok:
                break
            got += 1
        return got

    def free_all_blocks(self, kv: KvManager) -> None:
        for h in self.held_hashed:
            kv.free_hashed(h)
        self.held_hashed.clear()
        kv.free_anon(self.anon_blocks)
        self.anon_blocks = 0


# --------------------------------------------------------------------------
# Engine core (one simulated worker)
# --------------------------------------------------------------------------


@dataclass
class EngineArgs:
    num_gpu_blocks: int = 16384
    block_size: int = 64
    max_num_seqs: int = 256
    max_num_batched_tokens: int = 8192
    enable_prefix_caching: bool = True
    enable_chunked_prefill: bool = True
    preemption_mode: str = "lifo"  # lifo (vLLM v1) | fifo
    kv_mode: str = "block"  # block | token (no sharing / no prefix cache)
    worker_type: str = "agg"  # agg | prefill | decode
    # KV handoff delay for disagg (mocker: tokens * bytes_per_token / bw)
    kv_transfer_bandwidth_gbps: float = 0.0  # 0 disables
    kv_bytes_per_token: int = 0


@dataclass
class PassResult:
    end_ms: float
    emissions: list  # [(Request, completed: bool, handoff_delay_ms)]
    made_progress: bool
    num_prefill_batched: int
    num_ready_decode: int


class _Outcome(Enum):
    SCHEDULED = 0
    BLOCKED = 1
    CURRENT_PREEMPTED = 2


class VllmSimCore:
    def __init__(self, args: EngineArgs, perf_model=None):
        self.args = args
        self.perf = perf_model or PolynomialPerfModel()
        sharing = args.kv_mode == "block"
        self.kv = KvManager(args.num_gpu_blocks, args.block_size, enable_sharing=sharing)
        self.waiting: list[Request] = []  # deque semantics; preempted prepend
        self.running: list[Request] = []
        self.total_preemptions = 0

    # -- queue helpers ------------------------------------------------------

    def receive(self, req: Request) -> None:
        req.status = Status.WAITING
        self.waiting.append(req)

    def has_work(self) -> bool:
        return bool(self.waiting or self.running)

    def in_flight(self) -> int:
        return len(self.waiting) + len(self.running)

    def _preempt_one(self) -> Optional[Request]:
        """core.rs SchedulerState::preempt — victim from running per mode."""
        if not self.running:
            return None
        victim = self.running.pop(-1 if self.args.preemption_mode == "lifo" else 0)
        victim.free_all_blocks(self.kv)
        victim.computed = 0
        victim.status = Status.PREEMPTED
        victim.num_preemptions += 1
        self.total_preemptions += 1
        self.waiting.insert(0, victim)  # prepend_waiting
        return victim

    # -- one engine iteration ----------------------------------------------

    def execute_pass(self, now_ms: float) -> PassResult:
        budget = self.args.max_num_batched_tokens
        scheduled: dict[int, int] = {}  # rid -> tokens_used (for preempt refund)
        batch_count = 0
        batch_total_isl = 0
        batch_total_prefix = 0
        preempted_any = False

        # helper closure keeps the schedule loop shape close to core.rs
        def schedule_request(req: Request, from_waiting: bool):
            nonlocal budget, batch_count, batch_total_isl, batch_total_prefix, preempted_any

            cached = 0
            if req.computed == 0 and self.args.enable_prefix_caching:
                hit_blocks = self.kv.match_prefix(req.prompt_hashes)
                # never cache the full prompt: at least 1 token must be computed
                max_cacheable = (req.isl - 1) // self.kv.block_size
                cached = min(hit_blocks, max_cacheable) * self.kv.block_size
            eff_before = req.computed + cached
            prompt_before = min(eff_before, req.isl)
            remaining = req.seq_len() - eff_before
            prompt_remaining = req.isl - prompt_before
            if prompt_remaining > 0 and not self.args.enable_chunked_prefill and prompt_remaining > budget:
                return _Outcome.BLOCKED, 0
            desired = min(remaining, budget)
            if desired == 0 and remaining > 0:
                return _Outcome.BLOCKED, 0

            target = eff_before + desired
            actual_after = target
            while True:
                need = math.ceil(target / self.kv.block_size) - req.total_blocks()
                if need <= 0:
                    req.computed = actual_after
                    break
                got = req.allocate_blocks(need, self.kv)
                if got == need:
                    req.computed = actual_after
                    break
                committed_tokens = req.total_blocks() * self.kv.block_size
                req.computed = min(actual_after, committed_tokens)
                victim = self._preempt_one()
                if victim is None:
                    actual_after = req.computed
                    break
                preempted_any = True
                undone = scheduled.pop(victim.rid, None)
                if undone is not None:
                    budget += undone
                    # note: batch aggregate rollback for preempted prefills
                    # (core.rs also subtracts isl/prefix; matched below via
                    # recording aggregates only after the loop settles)
                if victim is req:
                    return _Outcome.CURRENT_PREEMPTED, 0

            tokens_used = actual_after - eff_before
            if tokens_used == 0 and actual_after < req.seq_len():
                return _Outcome.BLOCKED, 0

            prompt_after = min(actual_after, req.isl)
            prompt_tokens = prompt_after - prompt_before
            scheduled[req.rid] = tokens_used
            if prompt_tokens > 0:
                batch_count += 1
                batch_total_isl += prompt_before + prompt_tokens
                batch_total_prefix += prompt_before
            budget -= tokens_used

            if from_waiting:
                req.status = Status.RUNNING
                self.running.append(req)
                if req.admitted_ms < 0:
                    req.admitted_ms = now_ms
                    req.cached_tokens_at_admission = cached
            return _Outcome.SCHEDULED, tokens_used

        # 1) running set first
        i = 0
        while i < len(self.running) and budget > 0:
            req = self.running[i]
            outcome, _ = schedule_request(req, from_waiting=False)
            if outcome is _Outcome.SCHEDULED:
                i += 1
            elif outcome is _Outcome.BLOCKED:
                break
            else:  # CURRENT_PREEMPTED: req was removed from running at position i
                pass

        # 2) waiting admissions
        while not preempted_any and len(self.running) < self.args.max_num_seqs:
            if not self.waiting:
                break
            req = self.waiting[0]
            self.waiting.pop(0)
            outcome, tokens_used = schedule_request(req, from_waiting=True)
            if outcome is _Outcome.SCHEDULED:
                if tokens_used == 0 and budget == 0:
                    break
            else:
                if outcome is _Outcome.BLOCKED:
                    self.waiting.insert(0, req)  # put back, keep order
                break

        # 3) timing (decode workers pay no prefill compute — KV arrived via
        #    transfer; mocker core.rs predict_prefill_duration returns ZERO)
        prefill_ms = 0.0
        if batch_count > 0 and self.args.worker_type != "decode":
            mean_isl = batch_total_isl // batch_count
            mean_prefix = batch_total_prefix // batch_count
            prefill_ms = self.perf.prefill_ms(batch_count, mean_isl, mean_prefix)
        decode_start = now_ms + prefill_ms

        decode_ms, emissions = self._emit_ready_tokens(decode_start)
        end_ms = decode_start + decode_ms
        return PassResult(
            end_ms=end_ms,
            emissions=emissions,
            made_progress=bool(scheduled) or bool(emissions),
            num_prefill_batched=batch_count,
            num_ready_decode=len(emissions),
        )

    def _emit_ready_tokens(self, decode_start_ms: float):
        ready = [r for r in self.running if r.computed >= r.seq_len() and r.generated < r.osl]
        if not ready:
            return 0.0, []

        if self.args.worker_type == "prefill":
            # first (and only) token is produced by the prefill pass itself
            decode_ms = 0.0
        else:
            active_kv_tokens = self.kv.num_active_blocks * self.kv.block_size
            total_kv_tokens = self.kv.capacity * self.kv.block_size
            context_length = sum(r.seq_len() for r in ready) // len(ready)
            decode_ms = self.perf.decode_ms(len(ready), active_kv_tokens, context_length, total_kv_tokens)
        decode_end = decode_start_ms + decode_ms

        emissions = []
        for req in ready:
            emitted = False
            completed = False
            while True:
                if req.status is not Status.RUNNING:
                    break  # got preempted by an earlier ready request's alloc
                req.generated += 1
                need = math.ceil(req.seq_len() / self.kv.block_size) - req.total_blocks()
                if need <= 0 or req.allocate_blocks(need, self.kv) == need:
                    emitted = True
                    completed = req.generated >= req.osl
                    break
                req.generated -= 1  # sequence.pop()
                victim = self._preempt_one()
                if victim is None or victim is req:
                    break
            if not emitted:
                continue
            handoff_ms = 0.0
            if self.args.worker_type == "prefill":
                # prefill emits exactly one token, then hands off the KV cache
                completed = True
                bw = self.args.kv_transfer_bandwidth_gbps
                bpt = self.args.kv_bytes_per_token
                if bw > 0 and bpt > 0:
                    handoff_ms = req.isl * bpt / (bw * 1e9) * 1000.0
            req.token_times.append(decode_end)
            if completed:
                req.status = Status.DONE
                req.completed_ms = decode_end
                req.free_all_blocks(self.kv)
                self.running.remove(req)
            emissions.append((req, completed, handoff_ms))
        return decode_ms, emissions


# --------------------------------------------------------------------------
# Event-driven multi-worker driver
# --------------------------------------------------------------------------


class Simulator:
    """Virtual-clock driver mirroring the offline replay loop:
    a worker executes a pass whenever it is idle and has work; the pass
    completion is a scheduled event at `end_ms` (replay/offline/engine.rs).
    Dispatch is round-robin. Modes:
      - trace: open loop, requests dispatched at their arrival_ms
      - concurrency: closed loop with an in-flight cap; TTFT measured
        from actual dispatch (matches dynamo.replay concurrency semantics)
    """

    def __init__(self, num_workers: int, args: EngineArgs, perf_model=None, concurrency: Optional[int] = None):
        self.workers = [VllmSimCore(args, perf_model) for _ in range(num_workers)]
        self.busy = [False] * num_workers
        self.stalled = [False] * num_workers
        self.concurrency = concurrency
        self._rr = 0
        self._events: list = []  # (time, seq, kind, payload)
        self._seq = 0
        self.now = 0.0

    def _push(self, t: float, kind: str, payload) -> None:
        heapq.heappush(self._events, (t, self._seq, kind, payload))
        self._seq += 1

    def _dispatch(self, req: Request) -> None:
        wid = self._rr % len(self.workers)
        self._rr += 1
        req.dispatch_ms = self.now
        self.workers[wid].receive(req)
        self.stalled[wid] = False

    def run(self, requests: list[Request]) -> list[Request]:
        pending = sorted(requests, key=lambda r: r.arrival_ms)
        if self.concurrency is None:
            for r in pending:
                self._push(r.arrival_ms, "arrival", r)
            backlog: list[Request] = []
        else:
            backlog = pending
            for _ in range(min(self.concurrency, len(backlog))):
                self._push(0.0, "arrival", backlog.pop(0))

        completed = 0
        total = len(requests)
        while completed < total:
            if not self._events:
                raise RuntimeError(
                    "simulation deadlock: no events but requests in flight (prompt larger than KV capacity?)"
                )
            self.now = self._events[0][0]
            # Drain ALL events at this timestamp (incl. arrivals pushed by
            # completions processed in this drain) before driving workers,
            # matching the mocker offline loop; otherwise a same-instant
            # arrival misses the next pass and eats a full spurious pass
            # of extra TTFT.
            while self._events and self._events[0][0] <= self.now:
                _, _, kind, payload = heapq.heappop(self._events)
                if kind == "arrival":
                    self._dispatch(payload)
                elif kind == "pass_done":
                    wid, emissions = payload
                    self.busy[wid] = False
                    for req, done, _handoff in emissions:
                        if done:
                            completed += 1
                            if self.concurrency is not None and backlog:
                                self._push(self.now, "arrival", backlog.pop(0))
            self._drive_idle_workers()
        return requests

    def _drive_idle_workers(self) -> None:
        for wid, core in enumerate(self.workers):
            if self.busy[wid] or self.stalled[wid] or not core.has_work():
                continue
            result = core.execute_pass(self.now)
            if not result.made_progress and result.end_ms <= self.now:
                # nothing schedulable and nothing emitted: wait for next event
                self.stalled[wid] = True
                continue
            self.busy[wid] = True
            self._push(result.end_ms, "pass_done", (wid, result.emissions))


class DisaggSimulator:
    """P/D-disaggregated driver: requests prefill on the prefill pool, hand
    off after a KV-transfer delay (mocker: isl * kv_bytes_per_token / bw),
    then decode on the decode pool. Round-robin dispatch per pool, mirroring
    a degenerate kv_router. Closed-loop concurrency only (TTFT from dispatch,
    first token timestamp includes the KV handoff delay, matching the
    dynamo.replay disagg report)."""

    def __init__(
        self,
        num_prefill: int,
        num_decode: int,
        prefill_args: EngineArgs,
        decode_args: EngineArgs,
        perf_model=None,
        concurrency: int = 32,
    ):
        self.pools = {
            "prefill": [VllmSimCore(prefill_args, perf_model) for _ in range(num_prefill)],
            "decode": [VllmSimCore(decode_args, perf_model) for _ in range(num_decode)],
        }
        self.busy = {s: [False] * len(cores) for s, cores in self.pools.items()}
        self.stalled = {s: [False] * len(cores) for s, cores in self.pools.items()}
        self._rr = {"prefill": 0, "decode": 0}
        self.concurrency = concurrency
        self._events: list = []
        self._seq = 0
        self.now = 0.0

    def _push(self, t: float, kind: str, payload) -> None:
        heapq.heappush(self._events, (t, self._seq, kind, payload))
        self._seq += 1

    def _dispatch(self, stage: str, req: Request) -> None:
        wid = self._rr[stage] % len(self.pools[stage])
        self._rr[stage] += 1
        self.pools[stage][wid].receive(req)
        self.stalled[stage][wid] = False

    def run(self, requests: list[Request]) -> list[Request]:
        backlog = sorted(requests, key=lambda r: r.arrival_ms)
        for _ in range(min(self.concurrency, len(backlog))):
            req = backlog.pop(0)
            req.dispatch_ms = 0.0
            self._dispatch("prefill", req)

        completed = 0
        total = len(requests)
        self._drive()
        while completed < total:
            if not self._events:
                raise RuntimeError("disagg simulation deadlock")
            self.now = self._events[0][0]
            while self._events and self._events[0][0] <= self.now:
                _, _, kind, payload = heapq.heappop(self._events)
                if kind == "kv_ready":
                    self._dispatch("decode", payload)
                elif kind == "pass_done":
                    stage, wid, emissions = payload
                    self.busy[stage][wid] = False
                    for req, done, handoff in emissions:
                        if not done:
                            continue
                        if stage == "prefill":
                            # mocker offline disagg: the prefill token is
                            # NOT user-visible (collector sits on the decode
                            # stage only); the decode worker receives a clone
                            # of the original request and regenerates all osl
                            # tokens. Handoff delay defers decode enqueue.
                            req.token_times.pop()
                            req.generated = 0
                            req.computed = 0
                            self._push(self.now + handoff, "kv_ready", req)
                        else:
                            completed += 1
                            if backlog:
                                nxt = backlog.pop(0)
                                nxt.dispatch_ms = self.now
                                self._dispatch("prefill", nxt)
            self._drive()
        return requests

    def _drive(self) -> None:
        for stage, cores in self.pools.items():
            for wid, core in enumerate(cores):
                if self.busy[stage][wid] or self.stalled[stage][wid] or not core.has_work():
                    continue
                result = core.execute_pass(self.now)
                if not result.made_progress and result.end_ms <= self.now:
                    self.stalled[stage][wid] = True
                    continue
                self.busy[stage][wid] = True
                self._push(result.end_ms, "pass_done", (stage, wid, result.emissions))
