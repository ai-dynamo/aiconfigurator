# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Trace adapters: prefix-cache oracle + trace loaders + workload builder."""

import json

import pytest

from aiconfigurator.sdk.queueing import (
    EngineSpec,
    TraceRecord,
    evaluate_open_loop,
    load_cc_sessions_jsonl,
    load_mooncake_jsonl,
    prefix_hits,
    workload_from_trace,
)

pytestmark = pytest.mark.unit


class SyntheticTiming:
    def prefill_ms(self, batch_size, mean_isl, mean_prefix):
        return 10.0 + 0.02 * batch_size * max(0, mean_isl - mean_prefix)

    def decode_ms(self, batch_size, context_len):
        return max(1.0, 2.0 + 0.05 * batch_size)


class TestPrefixHitsOracle:
    def test_repeat_request_hits_fully(self):
        s = [["a", "b", "c"], ["a", "b", "c"]]
        assert prefix_hits(s, capacity_pages=10) == [0, 3]

    def test_leading_only_semantics(self):
        # second request diverges at page 2: later matching pages don't count
        s = [["a", "b", "c"], ["a", "x", "c"]]
        assert prefix_hits(s, capacity_pages=10) == [0, 1]

    def test_divergent_pages_still_cached_for_later(self):
        s = [["a", "x"], ["a", "b"], ["a", "b"]]
        assert prefix_hits(s, capacity_pages=10) == [0, 1, 2]

    def test_capacity_eviction_drops_hits(self):
        # capacity 2: request 1's pages are evicted by request 2 before repeat
        s = [["a", "b"], ["c", "d"], ["a", "b"]]
        assert prefix_hits(s, capacity_pages=2) == [0, 0, 0]
        # ample capacity keeps them
        assert prefix_hits(s, capacity_pages=4) == [0, 0, 2]

    def test_lru_refresh_on_hit(self):
        # "a" is refreshed by request 2, so request 4 still hits it while
        # "b" (older) was evicted
        s = [["a", "b"], ["a"], ["c"], ["a"]]
        assert prefix_hits(s, capacity_pages=2) == [0, 1, 0, 1]


class TestTraceRecordPages:
    def test_pages_split_blocks_and_truncate_tail(self):
        r = TraceRecord(arrival_ms=0, isl=96 + 20, osl=1, hash_ids=(7, 9), block_tokens=96)
        # 96-token blocks over 32-token pages: block 7 -> 3 pages, block 9
        # covers the 20-token tail -> 0 full pages (dropped like the engine)
        assert r.pages(32) == [(None, 7, 0), (None, 7, 1), (None, 7, 2)]

    def test_namespace_separates_sessions(self):
        r = TraceRecord(arrival_ms=0, isl=64, osl=1, hash_ids=(1,), block_tokens=64)
        assert r.pages(32, namespace="s1") != r.pages(32, namespace="s2")

    def test_rejects_misaligned_block(self):
        r = TraceRecord(arrival_ms=0, isl=50, osl=1, hash_ids=(1,), block_tokens=50)
        with pytest.raises(ValueError):
            r.pages(32)


class TestLoaders:
    def test_mooncake_loader_filters_and_limits(self, tmp_path):
        p = tmp_path / "t.jsonl"
        rows = [
            {"timestamp": 0, "input_length": 1024, "output_length": 10, "hash_ids": [0, 1]},
            {"timestamp": 3000, "input_length": 90000, "output_length": 10, "hash_ids": [2]},
            {"timestamp": 3000, "input_length": 512, "output_length": 4, "hash_ids": [0]},
        ]
        p.write_text("\n".join(json.dumps(r) for r in rows))
        recs = load_mooncake_jsonl(p, max_total_tokens=24576)
        assert [r.isl for r in recs] == [1024, 512]  # 90k row filtered
        assert recs[0].block_tokens == 512
        assert load_mooncake_jsonl(p, limit=1)[0].isl == 1024

    def test_cc_loader_flattens_subagents_by_time(self, tmp_path):
        p = tmp_path / "cc.jsonl"
        sess = {
            "id": "s0",
            "block_size": 64,
            "requests": [
                {"t": 0.0, "type": "s", "in": 640, "out": 20, "hash_ids": [0], "api_time": 2.0},
                {
                    "t": 5.0,
                    "type": "subagent",
                    "requests": [
                        {"t": 6.0, "type": "n", "in": 128, "out": 5, "hash_ids": [1], "api_time": 1.0},
                    ],
                },
                {"t": 3.0, "type": "n", "in": 256, "out": 8, "hash_ids": [2], "api_time": 1.0},
            ],
        }
        p.write_text(json.dumps(sess))
        sessions = load_cc_sessions_jsonl(p)
        assert len(sessions) == 1
        assert [r.isl for r in sessions[0]] == [640, 256, 128]  # sorted by t
        assert all(r.session == "s0" for r in sessions[0])
        assert sessions[0][0].block_tokens == 64

    def test_cc_loader_drops_oversized_sessions_whole(self, tmp_path):
        p = tmp_path / "cc.jsonl"
        sess = {
            "id": "big",
            "requests": [
                {"t": 0.0, "type": "s", "in": 640, "out": 20, "hash_ids": [0]},
                {"t": 1.0, "type": "s", "in": 999999, "out": 20, "hash_ids": [1]},
            ],
        }
        p.write_text(json.dumps(sess))
        assert load_cc_sessions_jsonl(p, max_total_tokens=32768) == []


class TestWorkloadFromTrace:
    def _records(self):
        # request 2 fully re-sends request 0's prompt as its prefix
        return [
            TraceRecord(arrival_ms=0.0, isl=1024, osl=16, hash_ids=(0, 1), block_tokens=512),
            TraceRecord(arrival_ms=1000.0, isl=512, osl=8, hash_ids=(9,), block_tokens=512),
            TraceRecord(arrival_ms=2000.0, isl=1536, osl=16, hash_ids=(0, 1, 2), block_tokens=512),
        ]

    def test_prefix_and_alignment(self):
        tw = workload_from_trace(self._records(), kv_capacity_tokens=65536)
        assert tw.prefix_tokens == [0, 0, 1024]
        assert len(tw.arrival_trace) == 3
        t, isl, px, osl = tw.arrival_trace[2]
        assert (isl, px, osl) == (1536, 1024, 16)
        assert 0 < tw.reuse_fraction < 1
        assert tw.workload.shape_tuples  # W3 stream form populated
        assert tw.workload.request_rate == pytest.approx(1.5)

    def test_time_scale_stretches_clock(self):
        tw = workload_from_trace(self._records(), kv_capacity_tokens=65536, time_scale=2.0)
        assert tw.arrival_trace[2][0] == pytest.approx(4000.0)
        assert tw.request_rate == pytest.approx(0.75)

    def test_replay_smoke_through_evaluator(self):
        tw = workload_from_trace(self._records(), kv_capacity_tokens=65536, time_scale=0.5)
        rep = evaluate_open_loop(
            tw.workload,
            EngineSpec(max_num_batched_tokens=4096),
            SyntheticTiming(),
            backend="vllm",
            warmup_requests=0,
            arrival_trace=tw.arrival_trace,
        )
        assert rep.per_request and len(rep.per_request) == 3
        assert all(x["ttft_ms"] is not None for x in rep.per_request)


class TestPartialTailReuse:
    """prefix_hit_tokens: TRT-LLM enable_partial_reuse's trace-visible arm —
    the prompt's trailing partial page reuses when the full leading run hit
    and its page is resident; matched but never inserted."""

    def test_tail_counts_only_after_full_leading_hit(self):
        from aiconfigurator.sdk.queueing import prefix_hit_tokens

        # request 1 computes pages a0,a1 whole; request 2 = same block cut
        # mid-page-2: full pages [a0] hit + tail (a1, 12 tokens) resident
        streams = [["a0", "a1"], ["a0"]]
        tails = [None, ("a1", 12)]
        assert prefix_hit_tokens(streams, 10, 32, tails=tails) == [0, 32 + 12]

    def test_tail_ignored_when_leading_run_breaks(self):
        from aiconfigurator.sdk.queueing import prefix_hit_tokens

        streams = [["a0", "a1", "a2"], ["a0", "x1"]]
        tails = [None, ("a2", 20)]  # diverged at page 2: tail unreachable
        assert prefix_hit_tokens(streams, 10, 32, tails=tails) == [0, 32]

    def test_tail_pages_are_not_inserted(self):
        from aiconfigurator.sdk.queueing import prefix_hit_tokens

        # request 1's tail (b0, 10) is matched-only: request 2 needing the
        # full b0 page must MISS it (the pool holds pages computed whole)
        streams = [[], ["b0"]]
        tails = [("b0", 10), None]
        assert prefix_hit_tokens(streams, 10, 32, tails=tails) == [0, 0]

    def test_matches_prefix_hits_when_no_tails(self):
        from aiconfigurator.sdk.queueing import prefix_hit_tokens, prefix_hits

        s = [["a", "b"], ["a", "b"], ["c"], ["a", "b"]]
        assert prefix_hit_tokens((list(x) for x in s), 3, 32) == [h * 32 for h in prefix_hits(s, 3)]

    def test_pages_and_tail_shape(self):
        from aiconfigurator.sdk.queueing import TraceRecord

        r = TraceRecord(arrival_ms=0, isl=64 + 32 + 7, osl=1, hash_ids=(7, 9), block_tokens=64)
        pages, tail = r.pages_and_tail(32, namespace="s")
        assert pages == [("s", 7, 0), ("s", 7, 1), ("s", 9, 0)]
        assert tail == (("s", 9, 1), 7)
        # boundary-aligned prompt has no tail
        r2 = TraceRecord(arrival_ms=0, isl=128, osl=1, hash_ids=(7, 9), block_tokens=64)
        assert r2.pages_and_tail(32, namespace="s")[1] is None

    def test_workload_from_trace_partial_tail_raises_reuse(self):
        from aiconfigurator.sdk.queueing import TraceRecord, workload_from_trace

        recs = [
            TraceRecord(arrival_ms=0.0, isl=64, osl=4, hash_ids=(1,), block_tokens=64),
            TraceRecord(arrival_ms=1000.0, isl=64 + 20, osl=4, hash_ids=(1, 2), block_tokens=64),
        ]
        base = workload_from_trace(list(recs), kv_capacity_tokens=4096)
        pt = workload_from_trace(list(recs), kv_capacity_tokens=4096, partial_tail_reuse=True)
        # request 2's tail (20 tok into block 2) was never computed whole
        # by anyone -> no partial hit; both oracles agree
        assert base.prefix_tokens == pt.prefix_tokens
        recs3 = recs + [TraceRecord(arrival_ms=2000.0, isl=64 + 20, osl=4, hash_ids=(1, 2), block_tokens=64)]
        base3 = workload_from_trace(list(recs3), kv_capacity_tokens=4096)
        pt3 = workload_from_trace(list(recs3), kv_capacity_tokens=4096, partial_tail_reuse=True)
        # request 3 repeats request 2: full-page oracle stops at 64; the
        # partial-tail oracle... block 2 page 0 was never computed whole, so
        # the tail (20 tok into block 2) still misses -> equal here too
        assert base3.prefix_tokens[2] == 64
        assert pt3.prefix_tokens[2] == 64
        # a whole-block repeat WITH a mid-page cut gains the tail
        recs4 = [
            TraceRecord(arrival_ms=0.0, isl=128, osl=4, hash_ids=(5, 6), block_tokens=64),
            TraceRecord(arrival_ms=1000.0, isl=96 + 10, osl=4, hash_ids=(5, 6), block_tokens=64),
        ]
        pt4 = workload_from_trace(list(recs4), kv_capacity_tokens=4096, partial_tail_reuse=True)
        b4 = workload_from_trace(list(recs4), kv_capacity_tokens=4096)
        assert b4.prefix_tokens[1] == 96
        assert pt4.prefix_tokens[1] == 96 + 10


class TestLeafLruEviction:
    """leaf-lru pool: radix-tree leaf-first eviction (TRT-LLM block reuse) —
    chains shrink tail-to-root, so a hot chain's prefix survives pressure
    that hole-punches the flat LRU mid-chain."""

    def test_chain_shrinks_from_tail_not_holes(self):
        from aiconfigurator.sdk.queueing import prefix_hit_tokens

        # chain a0..a3 cached, then an unrelated chain overflows capacity 6:
        # flat LRU evicts a0,a1 (oldest pages -> hole at the ROOT kills the
        # leading run); leaf-lru evicts a3 then a2 (leaves first), keeping
        # the prefix a0,a1 alive for the repeat request
        streams = [["a0", "a1", "a2", "a3"], ["b0", "b1", "b2", "b3"], ["a0", "a1", "a2", "a3"]]
        flat = prefix_hit_tokens([list(s) for s in streams], 6, 32, eviction="lru")
        tree = prefix_hit_tokens([list(s) for s in streams], 6, 32, eviction="leaf-lru")
        assert flat[2] == 0  # root evicted -> leading run dead
        assert tree[2] == 2 * 32  # tail shrank, prefix a0,a1 survived

    def test_equivalent_without_pressure(self):
        from aiconfigurator.sdk.queueing import prefix_hit_tokens

        streams = [["a", "b"], ["a", "b", "c"], ["x"], ["a", "b", "c"]]
        assert prefix_hit_tokens([list(s) for s in streams], 100, 32, eviction="lru") == prefix_hit_tokens(
            [list(s) for s in streams], 100, 32, eviction="leaf-lru"
        )

    def test_rejects_unknown_eviction(self):
        from aiconfigurator.sdk.queueing import prefix_hit_tokens

        with pytest.raises(ValueError):
            prefix_hit_tokens([["a"]], 4, 32, eviction="mru")
