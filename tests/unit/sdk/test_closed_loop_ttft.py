# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Minimal closed-loop pass-calendar estimators (post-processing tier)."""

import pandas as pd
import pytest

from aiconfigurator.sdk.closed_loop_ttft import (
    estimate_closed_loop_latency,
    estimate_disagg_closed_loop_latency,
    filter_closed_loop_sla,
    pick_under_closed_loop_sla,
    refine_closed_loop_latency,
)

pytestmark = pytest.mark.unit


def prefill_ms(b, mean_isl, mean_prefix):
    return 10.0 + 0.02 * b * max(0, mean_isl - mean_prefix)


def decode_ms(b, ctx):
    return max(1.0, 2.0 + 0.05 * b + 0.001 * ctx)


class TestAggEstimator:
    def test_c1_ttft_is_one_own_prefill(self):
        r = estimate_closed_loop_latency(1, 2048, 16, prefill_ms, decode_ms)
        assert r["ttft_steady_mean"] == pytest.approx(prefill_ms(1, 2048, 0))

    def test_littles_law_identity_holds(self):
        # C/X == TTFT + (osl-1)*TPOT: outputs are one consistent timeline
        # (the turnaround wait is part of TTFT, measured from dispatch)
        r = estimate_closed_loop_latency(8, 2048, 64, prefill_ms, decode_ms, turnaround_ms=15.0)
        cycle = 8 / r["throughput_rps"] * 1000.0
        assert cycle == pytest.approx(r["ttft_steady_mean"] + 63 * r["tpot_mean"], rel=0.02)

    def test_queueing_grows_with_concurrency(self):
        t = [estimate_closed_loop_latency(c, 4096, 64, prefill_ms, decode_ms)["ttft_steady_mean"] for c in (1, 4, 16)]
        assert t[0] < t[1] < t[2]

    def test_prefix_reduces_ttft(self):
        cold = estimate_closed_loop_latency(4, 4096, 32, prefill_ms, decode_ms)
        warm = estimate_closed_loop_latency(4, 4096, 32, prefill_ms, decode_ms, prefix=3072)
        assert warm["ttft_steady_mean"] < cold["ttft_steady_mean"]

    def test_chunked_off_admits_whole_prompts_only(self):
        on = estimate_closed_loop_latency(8, 6000, 16, prefill_ms, decode_ms, max_num_batched_tokens=8192)
        off = estimate_closed_loop_latency(
            8, 6000, 16, prefill_ms, decode_ms, max_num_batched_tokens=8192, enable_chunked_prefill=False
        )
        assert on["ttft_steady_mean"] != off["ttft_steady_mean"]

    def test_rejects_invalid(self):
        with pytest.raises(ValueError):
            estimate_closed_loop_latency(0, 128, 8, prefill_ms, decode_ms)
        with pytest.raises(ValueError):
            estimate_closed_loop_latency(1, 128, 8, prefill_ms, decode_ms, prefix=128)


class TestDisaggTandemEstimator:
    def test_c1_ttft_is_prefill_plus_handoff(self):
        r = estimate_disagg_closed_loop_latency(1, 4096, 8, 1, 1, 2000.0, 21.0, handoff_ms=5.0)
        assert r["ttft_steady_mean"] == pytest.approx(2005.0)

    def test_prefill_bound_imbalance_lands_in_ttft_not_tpot(self):
        # D-surplus regime: deeper closed loop piles prefill queueing into
        # TTFT while decode-side TPOT stays put (the disagg signature)
        shallow = estimate_disagg_closed_loop_latency(2, 4096, 256, 1, 1, 2000.0, 21.0, decode_max_seqs=64)
        deep = estimate_disagg_closed_loop_latency(16, 4096, 256, 1, 1, 2000.0, 21.0, decode_max_seqs=64)
        assert deep["ttft_steady_mean"] > shallow["ttft_steady_mean"] * 4
        assert deep["tpot_mean"] == pytest.approx(shallow["tpot_mean"], rel=0.05)

    def test_decode_bound_imbalance_lands_in_tpot_not_ttft(self):
        # P-surplus regime: decode slots saturate; TPOT inflates, TTFT holds
        ok = estimate_disagg_closed_loop_latency(4, 1024, 256, 2, 1, 500.0, 21.0, decode_max_seqs=4)
        sat = estimate_disagg_closed_loop_latency(12, 1024, 256, 2, 1, 500.0, 21.0, decode_max_seqs=4)
        assert sat["tpot_mean"] > ok["tpot_mean"] * 2
        assert sat["ttft_steady_mean"] < ok["ttft_steady_mean"] * 3

    def test_adding_prefill_worker_relieves_ttft(self):
        one = estimate_disagg_closed_loop_latency(8, 4096, 256, 1, 1, 2000.0, 21.0)
        two = estimate_disagg_closed_loop_latency(8, 4096, 256, 2, 1, 2000.0, 21.0)
        assert two["ttft_steady_mean"] < one["ttft_steady_mean"]

    def test_rejects_invalid(self):
        with pytest.raises(ValueError):
            estimate_disagg_closed_loop_latency(1, 128, 8, 0, 1, 100.0, 10.0)


class TestRefineDataFrame:
    def test_agg_rows_additive_columns_and_bad_rows_nan(self):
        step = prefill_ms(1, 2048, 0)
        df = pd.DataFrame(
            [
                {
                    "bs": 4,
                    "isl": 2048,
                    "osl": 32,
                    "ctx_tokens": 8192,
                    "prefix": 0,
                    "ttft": 123.0,
                    "prefill_step_ms": step,
                    "genonly_step_ms": 5.0,
                    "mix_step_ms": step + 5.0,
                    "num_mix_steps": 1,
                    "num_genonly_steps": 31,
                },
                {
                    "bs": 0,
                    "isl": 2048,
                    "osl": 32,
                    "ctx_tokens": 8192,
                    "prefix": 0,
                    "ttft": 456.0,
                    "prefill_step_ms": step,
                    "genonly_step_ms": 5.0,
                    "mix_step_ms": step + 5.0,
                    "num_mix_steps": 1,
                    "num_genonly_steps": 31,
                },
            ]
        )
        out = refine_closed_loop_latency(df)
        assert out["ttft"].tolist() == [123.0, 456.0]  # legacy untouched
        assert out["ttft_refined"].iloc[0] > 0
        assert pd.isna(out["ttft_refined"].iloc[1])  # invalid row -> NaN
        # refined triple is jointly consistent (closed-loop identity)
        cycle = 4 / out["throughput_refined"].iloc[0] * 1000.0
        assert cycle == pytest.approx(out["ttft_refined"].iloc[0] + 31 * out["tpot_refined"].iloc[0], rel=0.02)

    def test_sla_post_filter_drops_on_refined_keeps_nan(self):
        step = prefill_ms(1, 2048, 0)
        row = {
            "bs": 4,
            "isl": 2048,
            "osl": 32,
            "ctx_tokens": 8192,
            "prefix": 0,
            "ttft": 123.0,
            "prefill_step_ms": step,
            "genonly_step_ms": 5.0,
            "mix_step_ms": step + 5.0,
            "num_mix_steps": 1,
            "num_genonly_steps": 31,
        }
        bad = dict(row, bs=0, ttft=456.0)  # unpriceable -> refined NaN
        df = pd.DataFrame([row, bad])
        refined = refine_closed_loop_latency(df)["ttft_refined"].iloc[0]

        tight = filter_closed_loop_sla(df, ttft_ms=refined - 1.0)
        assert tight["ttft"].tolist() == [456.0]  # violator dropped, NaN row kept
        loose = filter_closed_loop_sla(df, ttft_ms=refined + 1.0, tpot_ms=1e9)
        assert loose["ttft"].tolist() == [123.0, 456.0]
        assert "ttft_refined" in loose.columns  # returns the refined copy
        assert "ttft_refined" not in df.columns  # input untouched

    def test_repick_recovers_lower_concurrency_point(self):
        step = prefill_ms(1, 2048, 0)
        ladder = []
        for bs in (2, 16):  # same deployment, two operating points
            ladder.append(
                {
                    "model": "m",
                    "parallel": "tp4",
                    "bs": bs,
                    "isl": 2048,
                    "osl": 32,
                    "ctx_tokens": 8192,
                    "prefix": 0,
                    "ttft": 100.0,
                    "tokens/s": 100.0 * bs,
                    "prefill_step_ms": step,
                    "genonly_step_ms": 5.0,
                    "mix_step_ms": step + 5.0,
                    "num_mix_steps": 1,
                    "num_genonly_steps": 31,
                }
            )
        df = pd.DataFrame(ladder)
        refined = refine_closed_loop_latency(df)["ttft_refined"]
        assert refined.iloc[1] > refined.iloc[0]  # deeper point queues more
        target = (refined.iloc[0] + refined.iloc[1]) / 2.0

        # naive post-filter of the PICKED row (the bs=16 one) yields nothing
        picked = df.iloc[[1]]
        assert filter_closed_loop_sla(picked, ttft_ms=target).empty
        # re-pick over the full ladder recovers the compliant bs=2 row
        best = pick_under_closed_loop_sla(df, ttft_ms=target)
        assert best["bs"].tolist() == [2]
        assert best["ttft_refined"].iloc[0] <= target

    def test_disagg_rows_use_tandem(self):
        df = pd.DataFrame(
            [
                {
                    "concurrency": 8,
                    "isl": 4096,
                    "osl": 64,
                    "(p)workers": 1,
                    "(d)workers": 1,
                    "(p)bs": 1,
                    "(d)bs": 64,
                    "(p)prefill_step_ms": 2000.0,
                    "tpot": 21.0,
                    "ttft": 3600.0,
                }
            ]
        )
        out = refine_closed_loop_latency(df)
        assert out["ttft_refined"].iloc[0] > 2000.0  # queueing beyond solo
        assert out["tpot_refined"].iloc[0] == pytest.approx(21.0, rel=0.05)
