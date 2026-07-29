# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Session-loop evaluator (W4): endogenous arrivals from lane causality."""

import pytest

from aiconfigurator.sdk.queueing import EngineSpec, SessionTurn, evaluate_sessions

pytestmark = pytest.mark.unit


class SyntheticTiming:
    def prefill_ms(self, batch_size, mean_isl, mean_prefix):
        return 10.0 + 0.02 * batch_size * max(0, mean_isl - mean_prefix)

    def decode_ms(self, batch_size, context_len):
        return 5.0


TIMING = SyntheticTiming()
ENG = EngineSpec(max_num_batched_tokens=4096, max_num_seqs=8)


class TestSessionLoop:
    def test_turn_causality_and_think_gap(self):
        lane = [
            SessionTurn(isl=1000, prefix=0, osl=4),
            SessionTurn(isl=1200, prefix=999, osl=4, think_ms=500.0),
        ]
        rep = evaluate_sessions([lane], ENG, TIMING)
        a, b = rep.per_request
        assert (a["turn"], b["turn"]) == (0, 1)
        # turn 1 dispatches exactly at turn 0's completion + think
        t0_end = a["arrival_ms"] + a["e2e_ms"]
        assert b["arrival_ms"] == pytest.approx(t0_end + 500.0)
        # fidelity string declares the tier
        assert rep.workload_fidelity.startswith("W4")

    def test_prefix_hot_followup_is_fast(self):
        cold = [SessionTurn(isl=2000, prefix=0, osl=2), SessionTurn(isl=2200, prefix=0, osl=2)]
        hot = [SessionTurn(isl=2000, prefix=0, osl=2), SessionTurn(isl=2200, prefix=1999, osl=2)]
        r_cold = evaluate_sessions([cold], ENG, TIMING)
        r_hot = evaluate_sessions([hot], ENG, TIMING)
        assert r_hot.per_request[1]["ttft_ms"] < r_cold.per_request[1]["ttft_ms"]

    def test_lanes_contend_for_budget(self):
        lane = [SessionTurn(isl=4000, prefix=0, osl=2)]
        solo = evaluate_sessions([lane], ENG, TIMING)
        crowd = evaluate_sessions([list(lane) for _ in range(6)], ENG, TIMING, stagger_ms=1.0)
        # six simultaneous-ish first turns share one 4096 budget: later lanes queue
        t_solo = solo.per_request[0]["ttft_ms"]
        t_last = max(x["ttft_ms"] for x in crowd.per_request)
        assert t_last > t_solo * 2

    def test_arrival_plane_applies_to_lanes(self):
        lanes = [[SessionTurn(isl=4000, prefix=0, osl=2)], [SessionTurn(isl=200, prefix=0, osl=2)]]
        rep = evaluate_sessions(lanes, ENG, TIMING, ingest_us_per_token=3.6, stagger_ms=0.0)
        big, small = rep.per_request[0], rep.per_request[1]
        # same-instant dispatch: the small prompt ingests first and is served first
        assert small["ttft_ms"] < big["ttft_ms"]

    def test_empty_sessions_rejected(self):
        with pytest.raises(ValueError):
            evaluate_sessions([], ENG, TIMING)

    def test_session_start_offsets(self):
        lanes = [[SessionTurn(isl=500, prefix=0, osl=2)], [SessionTurn(isl=500, prefix=0, osl=2)]]
        rep = evaluate_sessions(lanes, ENG, TIMING, session_start_ms=[0.0, 9000.0])
        assert rep.per_request[1]["arrival_ms"] == pytest.approx(9000.0)


class TestSessionLoopDisagg:
    """W4 lanes over the disagg tandem: same lane causality, tandem serving
    flow (first token prefill-side, per-turn KV handoff on the fabric)."""

    def _spec(self, **kw):
        from aiconfigurator.sdk.queueing import DisaggSpec

        base = dict(
            num_prefill_workers=1,
            num_decode_workers=1,
            kv_bytes_per_token=100_000,
            egress_bytes_per_s=1e9,
            ingress_bytes_per_s=1e9,
            bw_efficiency=1.0,
        )
        base.update(kw)
        return DisaggSpec(**base)

    def test_turn_causality_think_gap_and_handoff(self):
        from aiconfigurator.sdk.queueing import evaluate_sessions_disagg

        lane = [
            SessionTurn(isl=1000, prefix=0, osl=4),
            SessionTurn(isl=1200, prefix=999, osl=4, think_ms=500.0),
        ]
        rep = evaluate_sessions_disagg([lane], ENG, ENG, TIMING, TIMING, self._spec())
        assert rep.mode == "disagg"
        assert rep.workload_fidelity.startswith("W4")
        a, b = rep.per_request
        assert (a["turn"], b["turn"]) == (0, 1)
        # turn 1 dispatches exactly at turn 0's completion + think
        assert b["arrival_ms"] == pytest.approx(a["arrival_ms"] + a["e2e_ms"] + 500.0)
        # every decoding turn paid its own KV handoff, priced by its isl
        assert a["xfer_ms"] >= 1000 * 100_000 / 1e9 * 1000.0 * 0.999
        assert b["xfer_ms"] >= 1200 * 100_000 / 1e9 * 1000.0 * 0.999
        # solo lane, turnaround 0: turn 0 TTFT is exactly its own prefill
        assert a["ttft_ms"] == pytest.approx(TIMING.prefill_ms(1, 1000, 0))
        # prefix-hot follow-up prefills only the effective prompt
        assert b["ttft_ms"] == pytest.approx(TIMING.prefill_ms(1, 1200, 999))

    def test_slow_serving_delays_lane_dispatches(self):
        """The W4 point: arrivals are outputs. A slower prefill stage must
        push every later turn's dispatch instant back."""
        from aiconfigurator.sdk.queueing import evaluate_sessions_disagg

        class SlowPrefill(SyntheticTiming):
            def prefill_ms(self, batch_size, mean_isl, mean_prefix):
                return 5 * super().prefill_ms(batch_size, mean_isl, mean_prefix)

        lanes = [[SessionTurn(isl=2000, prefix=0, osl=4), SessionTurn(isl=2000, prefix=0, osl=4, think_ms=100.0)]]
        fast = evaluate_sessions_disagg(lanes, ENG, ENG, TIMING, TIMING, self._spec())
        slow = evaluate_sessions_disagg(lanes, ENG, ENG, SlowPrefill(), TIMING, self._spec())
        assert slow.per_request[1]["arrival_ms"] > fast.per_request[1]["arrival_ms"]

    def test_lanes_contend_for_prefill_budget(self):
        from aiconfigurator.sdk.queueing import evaluate_sessions_disagg

        lane = [SessionTurn(isl=4000, prefix=0, osl=2)]
        solo = evaluate_sessions_disagg([lane], ENG, ENG, TIMING, TIMING, self._spec())
        crowd = evaluate_sessions_disagg(
            [list(lane) for _ in range(6)], ENG, ENG, TIMING, TIMING, self._spec(), stagger_ms=1.0
        )
        assert max(x["ttft_ms"] for x in crowd.per_request) > solo.per_request[0]["ttft_ms"] * 2

    def test_rejects_kv_pressure_inputs(self):
        from aiconfigurator.sdk.queueing import evaluate_sessions_disagg

        lanes = [[SessionTurn(isl=500, prefix=0, osl=2)]]
        gne = EngineSpec(guaranteed_no_evict=True, kv_capacity_tokens=4096)
        with pytest.raises(ValueError, match="KV-pressure"):
            evaluate_sessions_disagg(lanes, gne, ENG, TIMING, TIMING, self._spec())
