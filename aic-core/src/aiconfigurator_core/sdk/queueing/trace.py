# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Trace adapters: turn recorded serving traces into queueing-model inputs.

Consolidates what previously lived as per-experiment scripts (the Mooncake
and InferenceX cc-traces validation pipelines): a token-level prefix-cache
oracle plus loaders that produce both consumption paths of the W-tier
contract —

- exact replay: ``arrival_trace`` tuples ``(arrival_ms, isl, prefix, osl)``
  for ``evaluate_open_loop(..., arrival_trace=...)`` (highest fidelity,
  preserves arrival<->shape pairing and ordering);
- W3 stream: a ``WorkloadSpec`` with joint ``shape_tuples`` and empirical
  ``arrival_quantiles`` (sweepable: rate/config can vary while marginals
  and first-order burst structure are kept).

Identity contract: prefixes are computed on the TRACE'S OWN token identity
(hash blocks are content ids — equal hash means equal tokens), which models
the system that produced the trace. Replaying a trace through a real server
with synthetic prompts adds a tokenizer round-trip that shifts token
boundaries (~5% length inflation measured); that is a replay artifact and
is deliberately NOT part of this adapter — replay harnesses must handle it
on their side (see the validation experiments).

Oracle semantics (mirrors vLLM/TRT-LLM paged prefix caching, verified
against live-engine ``cached_tokens`` self-reporting to +14/+15 tokens mean
at 34-93% reuse): pages of ``page_tokens`` tokens, LEADING full-page hits
only, over ``capacity_pages``. Two eviction semantics (``prefix_hit_tokens``):
page-granular LRU ("lru" — the original oracle) and radix-tree leaf-first
("leaf-lru" — TRT-LLM block reuse: chains shrink tail-to-root). Below
eviction pressure they agree; under pressure they differ decisively —
flat LRU hole-punches chains mid-prefix and collapses leading runs the
engine still serves fully cached (measured, cc window at 132k pool:
lru 58.4% vs engine 62.7% with whole-chain misses; leaf-lru at
capacity = pool − expected in-flight KV lands at 60.3%, per-turn mean
+257 tokens). ``prefix_hit_tokens`` additionally models partial-tail
reuse (``enable_partial_reuse``) and counts TOKENS. The oracle is acausal
w.r.t. in-flight computation (a block is reusable the moment an
earlier-ordered request lists it) — no material difference against engine
accounting below pressure (Mooncake replay, appendix 9); under pressure
feed the reusable share (pool minus in-flight) as capacity.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass
from heapq import heappop, heappush
from itertools import pairwise
from typing import Optional

from .spec import WorkloadSpec, stratified_quantiles, stratified_shape_tuples


def prefix_hits(page_streams: Iterable[list], capacity_pages: int) -> list:
    """Leading-hit page counts per request under a global LRU page cache.

    ``page_streams``: per request (in SERVICE order — order is load-bearing),
    the sequence of hashable page ids covering its prompt. Returns, per
    request, the number of LEADING pages already resident (the engine reuses
    only an unbroken leading run). Every page is inserted/refreshed
    regardless, and the cache evicts least-recently-used beyond
    ``capacity_pages``.
    """
    if capacity_pages < 1:
        raise ValueError("capacity_pages must be >= 1")
    pool = _FlatLruPool(capacity_pages)
    hits = []
    for pages in page_streams:
        h, _ = _consume_request(pool, pages, None)
        hits.append(h)
    return hits


class _FlatLruPool:
    """Page pool with page-granular LRU eviction: ANY page can be evicted
    independently, so pressure punches holes mid-chain and the leading run
    of a later request breaks at the first hole."""

    def __init__(self, capacity_pages: int):
        self.cache: OrderedDict = OrderedDict()
        self.cap = capacity_pages

    def hit(self, p) -> bool:
        if p in self.cache:
            self.cache.move_to_end(p)
            return True
        return False

    def insert(self, p, parent) -> None:
        self.cache[p] = None
        if len(self.cache) > self.cap:
            self.cache.popitem(last=False)


class _TreeLruPool:
    """Page pool with radix-tree leaf-first eviction, LRU tiebreak — the
    TRT-LLM block-reuse semantics: eviction removes childless blocks only,
    so cached chains shrink from the TAIL toward the root and a hot chain's
    leading prefix survives pressure that a flat LRU would hole-punch
    (measured on the cc window: flat-LRU leading hits collapse to 0 on
    turns the engine serves at ~100% cached). A page's parent is fixed at
    first insertion (content-hash chains make cross-context page reuse
    negligible in every trace seen)."""

    def __init__(self, capacity_pages: int):
        self.cap = capacity_pages
        self.stamp: dict = {}  # page -> last-use counter
        self.parent: dict = {}
        self.nchild: dict = {}
        self._heap: list = []  # (stamp, page) lazy entries
        self._n = 0

    def _touch(self, p) -> None:
        self._n += 1
        self.stamp[p] = self._n
        if not self.nchild.get(p):
            heappush(self._heap, (self._n, p))

    def hit(self, p) -> bool:
        if p in self.stamp:
            self._touch(p)
            return True
        return False

    def insert(self, p, parent) -> None:
        if parent is not None and parent not in self.stamp:
            parent = None  # chain root within the pool
        self.parent[p] = parent
        self.nchild.setdefault(p, 0)
        if parent is not None:
            self.nchild[parent] = self.nchild.get(parent, 0) + 1
        self._touch(p)
        while len(self.stamp) > self.cap:
            if not self._evict_one():
                break

    def _evict_one(self) -> bool:
        while self._heap:
            stamp, p = heappop(self._heap)
            if self.stamp.get(p) != stamp or self.nchild.get(p, 0) > 0:
                continue  # stale entry or grew children since
            del self.stamp[p]
            del self.nchild[p]
            par = self.parent.pop(p, None)
            if par is not None and par in self.stamp:
                self.nchild[par] -= 1
                if self.nchild[par] == 0:
                    heappush(self._heap, (self.stamp[par], par))
            return True
        return False


def _consume_request(pool, pages, tail) -> tuple:
    """One request against a page pool: returns (leading full-page hits,
    tail tokens reused). ``tail`` is ``(page_id, tokens)`` for the prompt's
    trailing partial page or None; it reuses only when the full leading run
    hit AND its page is resident, and it is matched but never inserted (a
    page enters the pool only computed whole)."""
    h = 0
    counting = True
    prev = None
    for p in pages:
        if pool.hit(p):
            if counting:
                h += 1
        else:
            counting = False
            pool.insert(p, prev)
        prev = p
    tail_tokens = 0
    if tail is not None and counting and pool.hit(tail[0]):
        tail_tokens = int(tail[1])
    return h, tail_tokens


def prefix_hit_tokens(
    page_streams: Iterable[list],
    capacity_pages: int,
    page_tokens: int,
    tails: Optional[Iterable] = None,
    eviction: str = "lru",
) -> list:
    """Leading-hit TOKEN counts per request — ``prefix_hits`` extended with
    the engine's partial-tail reuse (TRT-LLM ``enable_partial_reuse``,
    default-ON): when every full page of the prompt hits and the trailing
    partial page's id is resident, the tail's tokens count as reused too
    (the engine copies the matched fraction of the cached block).

    In trace-hash space this is the only expressible partial reuse:
    divergence INSIDE a block changes the content hash, so block-interior
    partial matches are invisible by construction (documented oracle
    boundary; measured residual after this term: |p90| ~1 page on the cc
    window). ``tails``: per request, ``(page_id, tokens)`` of the trailing
    partial page or None — see ``TraceRecord.pages_and_tail``.

    ``eviction`` selects the pool semantics: ``"lru"`` (page-granular — the
    original oracle, holes mid-chain under pressure) or ``"leaf-lru"``
    (radix-tree leaf-first, LRU tiebreak — TRT-LLM block-reuse eviction:
    chains shrink tail-to-root, hot prefixes survive). Under eviction
    pressure the two differ decisively; measured (cc window, 132k pool,
    engine reuse 62.7%): flat LRU 58.4% with whole-chain misses the engine
    serves fully cached, leaf-lru tracks the engine.
    """
    if capacity_pages < 1:
        raise ValueError("capacity_pages must be >= 1")
    if eviction not in ("lru", "leaf-lru"):
        raise ValueError("eviction must be 'lru' or 'leaf-lru'")
    pool = _TreeLruPool(capacity_pages) if eviction == "leaf-lru" else _FlatLruPool(capacity_pages)
    out = []
    tails_it = iter(tails) if tails is not None else None
    for pages in page_streams:
        tail = next(tails_it) if tails_it is not None else None
        h, tail_tokens = _consume_request(pool, pages, tail)
        out.append(h * page_tokens + tail_tokens)
    return out


@dataclass
class TraceRecord:
    """One request of a recorded trace, on the trace's own token identity."""

    arrival_ms: float
    isl: int
    osl: int
    # content-identity block hashes covering the prompt (block_tokens each,
    # last block truncated by isl); None when the trace carries no prefix
    # information
    hash_ids: Optional[tuple] = None
    block_tokens: int = 512
    # session-structured traces (cc): lane identity and the recorded service
    # duration, which lets think gaps be reconstructed as
    # t_{k+1} - (t_k + api_time_k)
    session: Optional[str] = None
    api_time_ms: Optional[float] = None

    def pages(self, page_tokens: int, namespace=None) -> list:
        """Position-aligned page ids for the leading-hit oracle.

        Blocks are whole multiples of engine pages in every trace seen so
        far (512 or 64 tokens vs 32-token pages); a page id is
        ``(namespace, block_hash, page_index_within_block)``. The trailing
        partial page of the prompt is dropped here (full-page reuse only);
        ``pages_and_tail`` exposes it for the partial-reuse oracle.
        """
        return self.pages_and_tail(page_tokens, namespace)[0]

    def pages_and_tail(self, page_tokens: int, namespace=None) -> tuple:
        """``(pages, tail)`` where ``tail`` is ``(page_id, tokens)`` for the
        prompt's trailing partial page (None when the prompt ends on a page
        boundary or carries no hashes) — the input pair for
        ``prefix_hit_tokens``'s partial-tail reuse."""
        if self.hash_ids is None:
            return [], None
        if self.block_tokens % page_tokens:
            raise ValueError(f"block_tokens={self.block_tokens} is not a whole multiple of page_tokens={page_tokens}")
        out = []
        tail = None
        remaining = self.isl
        for h in self.hash_ids:
            take = min(self.block_tokens, remaining)
            for j in range(take // page_tokens):
                out.append((namespace, h, j))
            leftover = take % page_tokens
            if leftover and remaining == take:
                # the prompt ends inside this block: its final partial page
                tail = ((namespace, h, take // page_tokens), leftover)
            remaining -= take
            if remaining <= 0:
                break
        return out, tail


def load_mooncake_jsonl(
    path,
    limit: Optional[int] = None,
    max_total_tokens: Optional[int] = None,
) -> list:
    """Mooncake-format trace: one JSON object per line with ``timestamp``
    (ms, bucketed), ``input_length``, ``output_length``, ``hash_ids``
    (512-token content blocks). ``max_total_tokens`` drops requests whose
    isl+osl exceed a serving cap (record the dropped fraction — the filter
    changes the workload)."""
    records = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if max_total_tokens and r["input_length"] + r["output_length"] > max_total_tokens:
                continue
            records.append(
                TraceRecord(
                    arrival_ms=float(r["timestamp"]),
                    isl=int(r["input_length"]),
                    osl=max(1, int(r["output_length"])),
                    hash_ids=tuple(r["hash_ids"]),
                    block_tokens=512,
                )
            )
            if limit and len(records) >= limit:
                break
    return records


def load_cc_sessions_jsonl(
    path,
    limit_sessions: Optional[int] = None,
    max_total_tokens: Optional[int] = None,
) -> list:
    """InferenceX cc-traces: one JSON object per SESSION, with main turns and
    nested subagent groups; hashes are 64-token blocks scoped to the session
    (namespaced here so cross-session ids never collide). Returns a list of
    sessions, each a time-ordered list of TraceRecord (subagent inner
    requests flattened by their own timestamps). Sessions containing any
    request beyond ``max_total_tokens`` are dropped whole (a partial session
    breaks the multi-turn prefix chain)."""
    sessions = []
    with open(path) as f:
        for li, line in enumerate(f):
            s = json.loads(line)
            block = int(s.get("block_size", 64))
            flat = []
            for r in s["requests"]:
                flat.extend(r["requests"] if r.get("type") == "subagent" else [r])
            flat.sort(key=lambda r: r["t"])
            if max_total_tokens and any(r["in"] + r["out"] > max_total_tokens for r in flat):
                continue
            sid = s.get("id", str(li))
            sessions.append(
                [
                    TraceRecord(
                        arrival_ms=float(r["t"]) * 1000.0,
                        isl=int(r["in"]),
                        osl=max(1, int(r["out"])),
                        hash_ids=tuple(r["hash_ids"]),
                        block_tokens=block,
                        session=sid,
                        api_time_ms=float(r["api_time"]) * 1000.0 if r.get("api_time") else None,
                    )
                    for r in flat
                ]
            )
            if limit_sessions and len(sessions) >= limit_sessions:
                break
    return sessions


@dataclass
class TraceWorkload:
    """Both consumption paths for one trace window, plus audit numbers."""

    workload: WorkloadSpec  # W3 stream form (sweepable)
    arrival_trace: list  # exact-replay form [(ms, isl, prefix, osl), ...]
    prefix_tokens: list  # oracle prefix per request, trace order
    reuse_fraction: float  # sum(prefix)/sum(isl) over the window
    request_rate: float  # requests/s at the (scaled) replay clock


def workload_from_trace(
    records: list,
    kv_capacity_tokens: int,
    page_tokens: int = 32,
    time_scale: float = 1.0,
    k_shape: int = 64,
    k_arrival: int = 64,
    turnaround_ms: float = 0.0,
    ingest_us_per_token: float = 0.0,
    partial_tail_reuse: bool = False,
    eviction: str = "lru",
) -> TraceWorkload:
    """Build queueing-model inputs from trace records (service order).

    ``time_scale`` stretches (>1) or compresses (<1) the trace clock:
    replayed_gap = recorded_gap * time_scale. Prefixes come from the
    leading-hit page oracle at the deployment's KV capacity. The W3 stream
    form carries joint shape tuples and empirical inter-arrival quantiles
    (zeros preserved — batched/bucketed arrivals ARE the burst structure).
    ``partial_tail_reuse`` adds the engine's partial-tail block reuse
    (TRT-LLM ``enable_partial_reuse``, default-ON in 1.3) to the oracle —
    see ``prefix_hit_tokens``; False preserves the validated full-page
    accounting. ``eviction`` picks the pool semantics ("lru" | "leaf-lru");
    under eviction pressure use "leaf-lru" with capacity = pool minus the
    expected in-flight KV (the engine's reusable share) — see
    ``prefix_hit_tokens``.
    """
    if not records:
        raise ValueError("records is empty")
    capacity_pages = max(1, kv_capacity_tokens // page_tokens)
    ns = [r.session for r in records]
    if partial_tail_reuse:
        pt = [r.pages_and_tail(page_tokens, namespace=n) for r, n in zip(records, ns, strict=True)]
        prefix_tokens = prefix_hit_tokens(
            (pages for pages, _ in pt), capacity_pages, page_tokens, tails=[tail for _, tail in pt],
            eviction=eviction,
        )
    else:
        hits = prefix_hits(
            (r.pages(page_tokens, namespace=n) for r, n in zip(records, ns, strict=True)),
            capacity_pages,
        )
        prefix_tokens = [h * page_tokens for h in hits]

    t0 = records[0].arrival_ms
    arrival = [(r.arrival_ms - t0) * time_scale for r in records]
    trace = [
        (t, r.isl, min(px, max(0, r.isl - 1)), r.osl) for t, r, px in zip(arrival, records, prefix_tokens, strict=True)
    ]

    span_s = max(arrival[-1], 1e-9) / 1000.0
    rate = len(records) / span_s if span_s > 0 else float("nan")
    gaps = [b - a for a, b in pairwise(arrival)]
    # fully-cached prompts (oracle prefix == isl) still compute >= 1 token
    # (the engine always runs the last token to produce logits), hence the
    # same isl-1 clamp the exact-replay tuples use
    shape_records = [(r.isl, min(px, max(0, r.isl - 1)), r.osl) for r, px in zip(records, prefix_tokens, strict=True)]
    wl = WorkloadSpec(
        isl=max(1, int(sum(r.isl for r in records) / len(records))),
        osl=max(1, int(sum(r.osl for r in records) / len(records))),
        request_rate=rate,
        shape_tuples=stratified_shape_tuples(shape_records, k_shape),
        arrival_quantiles=stratified_quantiles(gaps, k_arrival) if len(gaps) >= 2 else None,
        turnaround_ms=turnaround_ms,
        ingest_us_per_token=ingest_us_per_token,
    )
    total_isl = sum(r.isl for r in records)
    return TraceWorkload(
        workload=wl,
        arrival_trace=trace,
        prefix_tokens=prefix_tokens,
        reuse_fraction=sum(prefix_tokens) / total_isl if total_isl else 0.0,
        request_rate=rate,
    )
