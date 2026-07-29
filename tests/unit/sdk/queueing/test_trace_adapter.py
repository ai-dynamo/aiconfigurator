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
