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
only, global LRU over ``capacity_pages`` (the engine's page pool). The
oracle is acausal w.r.t. in-flight computation (a block is reusable the
moment an earlier-ordered request lists it) — measured against engine
accounting this makes no material difference (validated on the Mooncake
replay, appendix 9).
"""

from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Iterable, Optional

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
    cache: OrderedDict = OrderedDict()
    hits = []
    for pages in page_streams:
        h = 0
        counting = True
        for p in pages:
            if p in cache:
                cache.move_to_end(p)
                if counting:
                    h += 1
            else:
                counting = False
                cache[p] = None
                if len(cache) > capacity_pages:
                    cache.popitem(last=False)
        hits.append(h)
    return hits


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
        partial page of the prompt is never reusable and is dropped, like
        the engine does.
        """
        if self.hash_ids is None:
            return []
        if self.block_tokens % page_tokens:
            raise ValueError(
                f"block_tokens={self.block_tokens} is not a whole multiple "
                f"of page_tokens={page_tokens}"
            )
        per_block = self.block_tokens // page_tokens
        out = []
        remaining = self.isl
        for h in self.hash_ids:
            take = min(self.block_tokens, remaining)
            for j in range(take // page_tokens):
                out.append((namespace, h, j))
            remaining -= take
            if remaining <= 0:
                break
        return out


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
    for line in open(path):
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
    for li, line in enumerate(open(path)):
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
) -> TraceWorkload:
    """Build queueing-model inputs from trace records (service order).

    ``time_scale`` stretches (>1) or compresses (<1) the trace clock:
    replayed_gap = recorded_gap * time_scale. Prefixes come from the
    leading-hit page oracle at the deployment's KV capacity. The W3 stream
    form carries joint shape tuples and empirical inter-arrival quantiles
    (zeros preserved — batched/bucketed arrivals ARE the burst structure).
    """
    if not records:
        raise ValueError("records is empty")
    capacity_pages = max(1, kv_capacity_tokens // page_tokens)
    ns = [r.session for r in records]
    hits = prefix_hits(
        (r.pages(page_tokens, namespace=n) for r, n in zip(records, ns)),
        capacity_pages,
    )
    prefix_tokens = [h * page_tokens for h in hits]

    t0 = records[0].arrival_ms
    arrival = [(r.arrival_ms - t0) * time_scale for r in records]
    trace = [
        (t, r.isl, min(px, max(0, r.isl - 1)), r.osl)
        for t, r, px in zip(arrival, records, prefix_tokens)
    ]

    span_s = max(arrival[-1], 1e-9) / 1000.0
    rate = len(records) / span_s if span_s > 0 else float("nan")
    gaps = [b - a for a, b in zip(arrival, arrival[1:])]
    # fully-cached prompts (oracle prefix == isl) still compute >= 1 token
    # (the engine always runs the last token to produce logits), hence the
    # same isl-1 clamp the exact-replay tuples use
    shape_records = [
        (r.isl, min(px, max(0, r.isl - 1)), r.osl)
        for r, px in zip(records, prefix_tokens)
    ]
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
