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
