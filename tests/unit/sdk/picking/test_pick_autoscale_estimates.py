# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Estimated-provenance propagation through autoscale disagg composition."""

import pandas as pd
import pytest

from aiconfigurator.sdk import common
from aiconfigurator.sdk.picking import pick_autoscale

pytestmark = pytest.mark.unit


def _candidate(*, estimated: bool) -> dict:
    return {
        "model": "test-model",
        "isl": 128,
        "osl": 16,
        "prefix": 0,
        "concurrency": 1,
        "request_rate": 10.0,
        "bs": 1,
        "global_bs": 1,
        "ttft": 10.0,
        "tpot": 2.0,
        "seq/s": 10.0,
        "seq/s/gpu": 10.0,
        "tokens/s": 160.0,
        "tokens/s/gpu": 160.0,
        "tokens/s/user": 8.0,
        "request_latency": 40.0,
        "num_total_gpus": 1,
        "tp": 1,
        "pp": 1,
        "dp": 1,
        "moe_tp": 1,
        "moe_ep": 1,
        "parallel": "tp1_pp1_dp1",
        "gemm": "bfloat16",
        "kvcache": "bfloat16",
        "fmha": "bfloat16",
        "moe": "none",
        "comm": "half",
        "memory": 1.0,
        "backend": "sglang",
        "version": "0.5.12",
        "system": "h200_sxm",
        "power_w": 300.0,
        common.ESTIMATED_COLUMN: estimated,
    }


@pytest.mark.parametrize(
    "prefill_estimated,decode_estimated,expected",
    [(True, False, True), (False, True, True), (False, False, False)],
)
def test_pick_autoscale_ors_worker_estimate_flags(prefill_estimated, decode_estimated, expected):
    result = pick_autoscale(
        pd.DataFrame([_candidate(estimated=prefill_estimated)]),
        pd.DataFrame([_candidate(estimated=decode_estimated)]),
        target_ttft=100.0,
        target_tpot=10.0,
        top_n=1,
        ttft_correction_factor=1.0,
    )

    best = result["best_config_df"]
    assert common.ESTIMATED_COLUMN in best.columns
    assert bool(best.iloc[0][common.ESTIMATED_COLUMN]) is expected
