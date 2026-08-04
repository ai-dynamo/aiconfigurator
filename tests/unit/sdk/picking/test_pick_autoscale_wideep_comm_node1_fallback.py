# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""WideEP communication node-1 fallback propagation through autoscale disagg composition."""

import pandas as pd
import pytest

from aiconfigurator.sdk import common
from aiconfigurator.sdk.picking import pick_autoscale

pytestmark = pytest.mark.unit


def _candidate(*, uses_wideep_comm_node1_fallback: bool) -> dict:
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
        common.WIDEEP_COMM_NODE1_FALLBACK_COLUMN: uses_wideep_comm_node1_fallback,
    }


@pytest.mark.parametrize(
    "prefill_fallback,decode_fallback,expected",
    [(True, False, True), (False, True, True), (False, False, False)],
)
def test_pick_autoscale_ors_worker_wideep_fallback_flags(prefill_fallback, decode_fallback, expected):
    result = pick_autoscale(
        pd.DataFrame([_candidate(uses_wideep_comm_node1_fallback=prefill_fallback)]),
        pd.DataFrame([_candidate(uses_wideep_comm_node1_fallback=decode_fallback)]),
        target_ttft=100.0,
        target_tpot=10.0,
        top_n=1,
        ttft_correction_factor=1.0,
    )

    best = result["best_config_df"]
    assert common.WIDEEP_COMM_NODE1_FALLBACK_COLUMN in best.columns
    assert bool(best.iloc[0][common.WIDEEP_COMM_NODE1_FALLBACK_COLUMN]) is expected
