# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Minimal closed-loop pass-calendar estimator (post-processing tier)."""

import pandas as pd
import pytest

from aiconfigurator.sdk.closed_loop_ttft import estimate_closed_loop_latency, refine_closed_loop_ttft

pytestmark = pytest.mark.unit


def prefill_ms(b, mean_isl, mean_prefix):
    return 10.0 + 0.02 * b * max(0, mean_isl - mean_prefix)


def decode_ms(b, ctx):
    return max(1.0, 2.0 + 0.05 * b + 0.001 * ctx)


class TestEstimateClosedLoopLatency:
    def test_c1_ttft_is_one_own_prefill(self):
        r = estimate_closed_loop_latency(1, 2048, 16, prefill_ms, decode_ms)
        assert r["ttft_steady_mean"] == pytest.approx(prefill_ms(1, 2048, 0))

    def test_littles_law_identity_holds(self):
        # C/X == TTFT + (osl-1)*TPOT: the recursion satisfies the
        # closed-loop accounting identity by construction (the turnaround
        # wait is part of TTFT — measured from the dispatch instant)
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


class TestRefineDataFrame:
    def test_additive_columns_and_bad_rows_nan(self):
        df = pd.DataFrame(
            [
                {"bs": 4, "isl": 2048, "osl": 32, "ctx_tokens": 8192, "prefix": 0, "ttft": 123.0},
                {"bs": 0, "isl": 2048, "osl": 32, "ctx_tokens": 8192, "prefix": 0, "ttft": 456.0},
            ]
        )
        out = refine_closed_loop_ttft(df, prefill_ms, decode_ms)
        assert list(df.columns) == ["bs", "isl", "osl", "ctx_tokens", "prefix", "ttft"]  # input untouched
        assert out["ttft"].tolist() == [123.0, 456.0]  # legacy column preserved
        assert out["ttft_refined"].iloc[0] > 0
        assert pd.isna(out["ttft_refined"].iloc[1])  # invalid row -> NaN, no raise
