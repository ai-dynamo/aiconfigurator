# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reference discrete-event simulation of vLLM v1 iteration-level scheduling.

Scheduling semantics are anchored clause-by-clause to the vLLM v1 scheduler
(vllm/v1/core/sched/scheduler.py):

  - unified token budget: one per-step budget (max_num_scheduled_tokens)
    shared by decode tokens and prefill chunks
  - running set scheduled first, in admission order; each decode consumes
    one budget token
  - chunked prefill: a chunk is min(remaining_prompt, remaining_budget)
  - chunked-off gate: with chunked prefill disabled, admission stops once a
    whole prompt no longer fits the remaining budget
  - admission cap: waiting-queue admission stops at max_num_seqs running
  - waiting-admission alloc failure: the request stays queued (allocation
    rolled back), admission stops for the step — no preemption
  - running-path alloc failure: preempt the newest running request (LIFO)
    with full recompute

One engine pass = prefill chunk scheduling followed by one decode emission
for caught-up requests; pass duration = prefill(batch) + decode(rows) from
a pluggable perf model, so scheduling fidelity is separable from timing
fidelity. Block-level KV accounting mirrors the vLLM BlockPool: hashed
blocks are shared by refcount, freed blocks keep their hash in an
eviction-ordered free queue (prefix-reusable until evicted).

The full provenance table and validation record live in
docs/design/queueing_model.md §5.
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


class SyntheticPerfModel:
    """Synthetic timing basis for standalone runs.

    A generic roofline-shaped stand-in: prefill has a fixed launch cost, a
    bandwidth-bound linear term, and a compute-bound quadratic term in batch
    tokens; decode grows with batch size and mean context. The constants are
    arbitrary but produce realistic pass-length ratios. Studies that care
    about timing inject their own callbacks (``CallbackPerfModel``) — the
    oracle's value is scheduling fidelity, not these numbers.
    """

    def prefill_ms(self, batch_count: int, mean_isl: int, mean_prefix: int) -> float:
        if batch_count == 0:
            return 0.0
        tokens = float(batch_count * max(0, mean_isl - mean_prefix))
        return 12.0 + 0.016 * tokens + 5e-07 * tokens * tokens

    def decode_ms(self, batch_size: int, active_kv_tokens: int, context_length: int, total_kv_tokens: int) -> float:
        if batch_size == 0:
            return 0.0
        return max(1.0, 2.5 + 0.05 * batch_size + 0.001 * context_length)


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
    """Block-based KV accounting mirroring the vLLM v1 BlockPool.

    vLLM keeps every KV block in one pool: blocks referenced by running
    requests hold a refcount; freed blocks keep their content hash and move
    to a free queue in eviction order, where a prefix hit can re-activate
    them until they are evicted for a new allocation
    (vllm/v1/core/block_pool.py). This manager reproduces that lifecycle:
      - hashed full prompt blocks are shared by refcount ("active"),
      - refcount 0 moves a block to the LRU "inactive" pool (prefix-reusable),
      - anonymous blocks (partial tails, generated tokens) are never shared,
      - a new allocation evicts the LRU inactive block when at capacity,
      - allocation fails only when capacity is exhausted and nothing is
        evictable -> caller preempts.

    Known simplification vs vLLM: full blocks produced during decode also
    receive hashes in vLLM (generated text is prefix-reusable); here they
    are freed anonymously, so multi-turn reuse of *generated* text is not
    modeled.
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
    # 'token' disables block sharing + prefix cache, to measure the fidelity
    # delta of naive token-count KV accounting against block-level accounting
    kv_mode: str = "block"  # block | token
    worker_type: str = "agg"  # agg | prefill | decode


@dataclass(frozen=True)
class TransferSpec:
    """KV-transfer fabric parameters for disagg (see ``TransferFabric``).

    Bandwidths are nominal single-direction Byte/s per worker NIC — the
    same units as the AIC system spec (``node.inter_node_bw`` /
    ``node.intra_node_bw``; see ``sysspec.transfer_spec_from_system``).
    ``bw_efficiency`` de-rates the nominal line rate to what transfers
    actually achieve (protocol/packetization/scatter-gather overheads);
    the 0.8 default follows the same convention as the system spec's own
    ``mem_bw_empirical_scaling_factor``.
    """

    kv_bytes_per_token: int
    egress_bytes_per_s: float  # per prefill-worker NIC
    ingress_bytes_per_s: float  # per decode-worker NIC
    bw_efficiency: float = 0.8


class TransferFabric:
    """Max-min fair sharing of per-worker NIC bandwidth for KV transfers.

    Transfers are NOT independent: each is a fluid flow from a prefill
    worker's egress to a decode worker's ingress, and concurrent flows
    share endpoint capacity max-min fairly (the multi-stream RDMA
    behaviour). Rates are recomputed whenever a flow starts or finishes,
    so fan-out (one prefill worker's completions racing out of one NIC)
    and fan-in (several prefill workers landing on one decode worker)
    slow each other down by exactly the computed fair share instead of a
    configured constant.
    """

    _EPS_BYTES = 1e-6
    _EPS_MS = 1e-6  # sub-ns: absorbs float dust at large virtual times

    def __init__(self, spec: TransferSpec):
        assert spec.egress_bytes_per_s > 0 and spec.ingress_bytes_per_s > 0
        self._egress = spec.egress_bytes_per_s * spec.bw_efficiency
        self._ingress = spec.ingress_bytes_per_s * spec.bw_efficiency
        # fid -> [src, dst, remaining_bytes, rate_bytes_per_s, payload]
        self._flows: dict[int, list] = {}
        self._finished: list = []  # (finish_ms, payload)
        self._next_fid = 0
        self._t_ms = 0.0  # time of the last fluid-state update

    def has_flows(self) -> bool:
        return bool(self._flows) or bool(self._finished)

    def _internal_next_ms(self):
        times = [f[2] / f[3] for f in self._flows.values() if f[3] > 0]
        if not times:
            return None
        return self._t_ms + max(min(times) * 1000.0, self._EPS_MS)

    def _advance(self, now_ms: float) -> None:
        """Piecewise fluid advance: rates change at every completion, so
        the clock stops at each internal completion point (collecting the
        finished flow with its TRUE finish time and recomputing rates)
        before continuing — a single linear step would mis-share bandwidth
        past a completion point. Completions within the time epsilon also
        finish here (float absorption guard: at large virtual times the
        clock cannot resolve byte dust)."""
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
        """Max-min water-filling: repeatedly fix the flows crossing the
        currently most-contended endpoint at that endpoint's fair share,
        deduct their consumption from the other endpoint they traverse."""
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

    def pop_completed(self, now_ms: float) -> list:
        """Advance to now_ms and return [(finish_ms, payload)] for flows
        completed by then (finish_ms = the flow's true completion time)."""
        self._advance(now_ms)
        ready = [(t, p) for t, p in self._finished if t <= now_ms]
        self._finished = [(t, p) for t, p in self._finished if t > now_ms]
        return ready

    def next_completion_ms(self) -> Optional[float]:
        if self._finished:
            return min(t for t, _ in self._finished)
        return self._internal_next_ms()


@dataclass
class PassResult:
    end_ms: float
    emissions: list  # [(Request, completed: bool)]
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
        self.perf = perf_model or SyntheticPerfModel()
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
        """vLLM v1 preemption: pop the NEWEST running request (LIFO), free
        its KV, reset to zero computed tokens (full recompute), and prepend
        it to the waiting queue. (The priority-policy variant, which picks
        the lowest-priority victim instead, is out of scope.)"""
        if not self.running:
            return None
        victim = self.running.pop()
        victim.free_all_blocks(self.kv)
        victim.computed = 0
        victim.status = Status.PREEMPTED
        victim.num_preemptions += 1
        self.total_preemptions += 1
        self.waiting.insert(0, victim)  # prepend to waiting, keep FCFS order
        return victim

    # -- one engine iteration ----------------------------------------------

    def execute_pass(self, now_ms: float) -> PassResult:
        budget = self.args.max_num_batched_tokens
        scheduled: dict[int, int] = {}  # rid -> tokens_used (for preempt refund)
        batch_count = 0
        batch_total_isl = 0
        batch_total_prefix = 0
        preempted_any = False

        # helper closure mirrors the scheduler's per-request scheduling step
        def schedule_request(req: Request, from_waiting: bool):
            nonlocal budget, batch_count, batch_total_isl, batch_total_prefix, preempted_any

            cached = 0
            if req.computed == 0 and self.args.worker_type == "decode":
                # KV connector semantics: remotely transferred KV counts as
                # computed tokens (num_external_computed_tokens), leaving one
                # position to run locally for logits — the same
                # "cache hit caps at length - 1" rule as the local prefix
                # cache. Blocks for the transferred KV are still allocated
                # locally below.
                cached = req.seq_len() - 1
            elif req.computed == 0 and self.args.enable_prefix_caching:
                hit_blocks = self.kv.match_prefix(req.prompt_hashes)
                # never cache the full prompt: at least one position must be
                # computed to obtain logits (vLLM caps the cache hit at
                # prompt length - 1)
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
            entry_hashed = len(req.held_hashed)
            entry_anon = req.anon_blocks
            entry_computed = req.computed
            while True:
                need = math.ceil(target / self.kv.block_size) - req.total_blocks()
                if need <= 0:
                    req.computed = actual_after
                    break
                got = req.allocate_blocks(need, self.kv)
                if got == need:
                    req.computed = actual_after
                    break
                if from_waiting:
                    # vLLM v1 semantics: a WAITING admission whose allocation
                    # fails is put back intact and admission stops — it does
                    # NOT preempt running requests. Roll back this attempt's
                    # partial allocation.
                    while len(req.held_hashed) > entry_hashed:
                        self.kv.free_hashed(req.held_hashed.pop())
                    extra_anon = req.anon_blocks - entry_anon
                    if extra_anon > 0:
                        self.kv.free_anon(extra_anon)
                        req.anon_blocks = entry_anon
                    req.computed = entry_computed
                    return _Outcome.BLOCKED, 0
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
                    # (scheduler also releases the victim's scheduled work;
                    # matched below via recording aggregates only after the
                    # loop settles)
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

        # 3) timing. Decode workers never accumulate prefill work here: the
        #    KV-connector jump above leaves at most one local position per
        #    admission, billed like any other row.
        prefill_ms = 0.0
        if batch_count > 0:
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

        # decode rows exclude prefill completers (generated == 0): the fused
        # pass samples their first token off the final chunk's logits — no
        # extra decode-row cost. Their token still lands at pass end. A
        # completer-only pass (prefill workers always; agg at C=1) therefore
        # adds no decode cost.
        decode_rows = [r for r in ready if r.generated >= 1]
        if not decode_rows:
            decode_ms = 0.0
        else:
            active_kv_tokens = self.kv.num_active_blocks * self.kv.block_size
            total_kv_tokens = self.kv.capacity * self.kv.block_size
            context_length = sum(r.seq_len() for r in decode_rows) // len(decode_rows)
            decode_ms = self.perf.decode_ms(len(decode_rows), active_kv_tokens, context_length, total_kv_tokens)
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
            if self.args.worker_type == "prefill":
                # disagg flow: the prefill worker produces the first token
                # (user-visible — it IS the TTFT token) and hands the KV
                # cache to a decode worker; its part of the request ends
                # here (the transfer itself is the driver's concern)
                completed = True
            req.token_times.append(decode_end)
            if completed:
                req.status = Status.DONE
                req.completed_ms = decode_end
                self.running.remove(req)
                if self.args.worker_type != "prefill":
                    req.free_all_blocks(self.kv)
                # prefill workers HOLD their KV blocks past completion: in
                # the pull model the decode side reads the KV remotely, so
                # the blocks stay resident until the transfer finishes —
                # slow transfers therefore squeeze prefill admission. The
                # disagg driver frees them when the handoff completes.
            emissions.append((req, completed))
        return decode_ms, emissions


# --------------------------------------------------------------------------
# Event-driven multi-worker driver
# --------------------------------------------------------------------------


class Simulator:
    """Virtual-clock event-driven driver: a worker executes a pass whenever
    it is idle and has work; the pass completion is a scheduled event at
    `end_ms`. Dispatch is round-robin. Modes:
      - trace: open loop, requests dispatched at their arrival_ms
      - concurrency: closed loop with an in-flight cap (the standard
        closed-loop benchmark-client convention: the next request is sent
        when one completes; TTFT is measured from actual dispatch)
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
            # completions processed in this drain) before driving workers:
            # the engine loop schedules whatever is queued when the next
            # iteration starts, so a same-instant arrival must not miss the
            # pass starting now and eat a full spurious pass of extra TTFT.
            while self._events and self._events[0][0] <= self.now:
                _, _, kind, payload = heapq.heappop(self._events)
                if kind == "arrival":
                    self._dispatch(payload)
                elif kind == "pass_done":
                    wid, emissions = payload
                    self.busy[wid] = False
                    for req, done in emissions:
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
    """P/D-disaggregated driver, following the disagg serving flow: the
    prefill pool computes the prompt and produces the FIRST token (that IS
    the TTFT token, streamed to the user), the KV cache is handed off to a
    decode worker, and the decode pool continues from token 2 — so the
    handoff shows up as the first ITL gap, not in TTFT.

    KV-transfer time is COMPUTED, not configured: each handoff is a flow on
    the ``TransferFabric``, where concurrent transfers share per-worker NIC
    bandwidth max-min fairly — a clump of completions fanning out of one
    prefill worker, or several prefill workers fanning in on one decode
    worker, slow each other down accordingly. The destination decode worker
    is chosen when the transfer STARTS (that is when the flow joins the
    ingress contention). Bandwidths come from a ``TransferSpec`` (see
    ``sysspec.transfer_spec_from_system`` for wiring the AIC system spec);
    ``transfer=None`` disables transfer modeling (zero-delay handoff).

    Dispatch is round-robin per pool (a degenerate kv_router: no affinity
    or queue-depth admission). Closed-loop concurrency only."""

    def __init__(
        self,
        num_prefill: int,
        num_decode: int,
        prefill_args: EngineArgs,
        decode_args: EngineArgs,
        perf_model=None,
        concurrency: int = 32,
        transfer: Optional[TransferSpec] = None,
    ):
        self.pools = {
            "prefill": [VllmSimCore(prefill_args, perf_model) for _ in range(num_prefill)],
            "decode": [VllmSimCore(decode_args, perf_model) for _ in range(num_decode)],
        }
        self.busy = {s: [False] * len(cores) for s, cores in self.pools.items()}
        self.stalled = {s: [False] * len(cores) for s, cores in self.pools.items()}
        self._rr = {"prefill": 0, "decode": 0}
        self.concurrency = concurrency
        self.fabric = TransferFabric(transfer) if transfer and transfer.kv_bytes_per_token > 0 else None
        self._bytes_per_token = transfer.kv_bytes_per_token if transfer else 0
        self._events: list = []
        self._seq = 0
        self.now = 0.0

    def _push(self, t: float, kind: str, payload) -> None:
        heapq.heappush(self._events, (t, self._seq, kind, payload))
        self._seq += 1

    def _dispatch(self, stage: str, req: Request) -> int:
        wid = self._rr[stage] % len(self.pools[stage])
        self._rr[stage] += 1
        self._dispatch_to(stage, wid, req)
        return wid

    def _dispatch_to(self, stage: str, wid: int, req: Request) -> None:
        self.pools[stage][wid].receive(req)
        self.stalled[stage][wid] = False

    def _release_prefill_kv(self, src_wid: int, req: Request) -> None:
        """Free the prefill-side KV once the decode side has pulled it, and
        wake the worker: it may have stalled on block allocation."""
        req.free_all_blocks(self.pools["prefill"][src_wid].kv)
        self.stalled["prefill"][src_wid] = False

    def _handoff(self, src_wid: int, req: Request) -> None:
        """Start the KV handoff of `req` from prefill worker `src_wid`.
        The decode-side KV-connector jump re-registers the transferred KV,
        so computed resets here."""
        req.computed = 0
        if self.fabric is None:
            self._release_prefill_kv(src_wid, req)  # zero-delay pull
            self._dispatch("decode", req)
            return
        dst = self._rr["decode"] % len(self.pools["decode"])
        self._rr["decode"] += 1
        self.fabric.submit(src_wid, dst, req.isl * self._bytes_per_token, self.now, (src_wid, dst, req))

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
                if kind == "xfer_tick":
                    pass  # wake-up only; completions are drained below
                elif kind == "pass_done":
                    stage, wid, emissions = payload
                    self.busy[stage][wid] = False
                    for req, done in emissions:
                        if not done:
                            continue
                        if stage == "prefill" and req.generated < req.osl:
                            # first token already emitted by the prefill
                            # worker; the decode worker continues the same
                            # sequence from the transferred KV
                            self._handoff(wid, req)
                        else:
                            # decode-side completion, or osl == 1 finishing
                            # on the prefill worker outright (nothing to
                            # transfer -> release its KV here)
                            if stage == "prefill":
                                self._release_prefill_kv(wid, req)
                            completed += 1
                            if backlog:
                                nxt = backlog.pop(0)
                                nxt.dispatch_ms = self.now
                                self._dispatch("prefill", nxt)
            if self.fabric is not None:
                for _t_done, (src, dst, req) in self.fabric.pop_completed(self.now):
                    self._release_prefill_kv(src, req)
                    self._dispatch_to("decode", dst, req)
                t = self.fabric.next_completion_ms()
                if t is not None:
                    self._push(t, "xfer_tick", None)
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
