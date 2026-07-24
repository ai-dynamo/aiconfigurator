# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Parity test: sweep._rate_match_dict vs picking._build_disagg_summary_dict.

The new sweep.py inlines the rate-matching math that lives in
picking._build_disagg_summary_dict (private helper).  This test locks
the two implementations to identical output so they cannot drift.

If this ever fails, either both implementations need to be updated in
sync, or one of them has acquired a bug.  Do not "fix" by tweaking only
one side.
"""

import pytest

from aiconfigurator.sdk.picking import _build_disagg_summary_dict
from aiconfigurator.sdk.sweep import _rate_match_dict

pytestmark = pytest.mark.unit


def _make_prefill_dict(**overrides) -> dict:
    """A row as it would appear in a ColumnsStatic DataFrame for prefill."""
    base = {
        "model": "test-model",
        "isl": 4000,
        "osl": 500,
        "prefix": 0,
        "concurrency": 1,
        "bs": 1,
        "global_bs": 1,
        "tp": 4,
        "pp": 1,
        "dp": 1,
        "moe_tp": 1,
        "moe_ep": 1,
        "parallel": "tp4pp1dp1",
        "ttft": 80.0,
        "tpot": 0.0,
        "seq/s": 10.0,
        "tokens/s/user": 0.0,
        "gemm": "fp8",
        "kvcache": "fp8",
        "fmha": "fp8",
        "moe": "fp8",
        "comm": "half",
        "memory": 12.3,
        "backend": "trtllm",
        "version": "1.3.0",
        "system": "h200_sxm",
        "power_w": 500.0,
    }
    base.update(overrides)
    return base


def _make_decode_dict(**overrides) -> dict:
    """A row as it would appear in a ColumnsStatic DataFrame for decode."""
    base = {
        "model": "test-model",
        "isl": 4000,
        "osl": 500,
        "prefix": 0,
        "concurrency": 64,
        "bs": 64,
        "global_bs": 64,
        "tp": 2,
        "pp": 1,
        "dp": 1,
        "moe_tp": 1,
        "moe_ep": 1,
        "parallel": "tp2pp1dp1",
        "ttft": 0.0,
        "tpot": 25.0,
        "seq/s": 5.0,
        "tokens/s/user": 40.0,
        "gemm": "fp8",
        "kvcache": "fp8",
        "fmha": "fp8",
        "moe": "fp8",
        "comm": "half",
        "memory": 18.7,
        "backend": "trtllm",
        "version": "1.3.0",
        "system": "h200_sxm",
        "power_w": 700.0,
    }
    base.update(overrides)
    return base


@pytest.mark.parametrize(
    "num_p, num_d",
    [
        (1, 1),
        (2, 4),
        (4, 8),
        (8, 16),
        (1, 32),
        (16, 1),
    ],
)
def test_rate_match_dict_matches_picking(num_p, num_d):
    p = _make_prefill_dict()
    d = _make_decode_dict()

    new_result = _rate_match_dict(p, num_p, d, num_d)
    old_result = _build_disagg_summary_dict(p, num_p, d, num_d)

    assert set(new_result.keys()) == set(old_result.keys()), (
        f"Key set differs.\nIn new but not old: {set(new_result) - set(old_result)}\n"
        f"In old but not new: {set(old_result) - set(new_result)}"
    )
    for key in new_result:
        assert new_result[key] == old_result[key], (
            f"Field {key!r} differs: new={new_result[key]} vs old={old_result[key]}"
        )


def test_rate_match_with_custom_degradation_factors():
    p = _make_prefill_dict()
    d = _make_decode_dict()
    custom_prefill = 0.85
    custom_decode = 0.88

    new_result = _rate_match_dict(p, 2, d, 4, prefill_degradation=custom_prefill, decode_degradation=custom_decode)
    old_result = _build_disagg_summary_dict(
        p,
        2,
        d,
        4,
        prefill_degradation_factor=custom_prefill,
        decode_degradation_factor=custom_decode,
    )

    for key in new_result:
        assert new_result[key] == old_result[key], (
            f"Field {key!r} differs with custom degradation: new={new_result[key]} vs old={old_result[key]}"
        )


def test_rate_match_prefill_ttft_capacity_can_bound_throughput():
    p = _make_prefill_dict(bs=1, ttft=500.0, **{"seq/s": 100.0})
    d = _make_decode_dict(**{"seq/s": 100.0})

    new_result = _rate_match_dict(p, 2, d, 2)
    old_result = _build_disagg_summary_dict(p, 2, d, 2)

    assert new_result["seq/s"] == 4.0
    for key in new_result:
        assert new_result[key] == old_result[key]


def test_rate_match_zero_osl_does_not_divide_by_zero():
    """request_latency uses max(osl - 1, 0); osl=1 keeps decode_time=0."""
    p = _make_prefill_dict(osl=1)
    d = _make_decode_dict(osl=1)
    new_result = _rate_match_dict(p, 1, d, 1)
    old_result = _build_disagg_summary_dict(p, 1, d, 1)
    for key in new_result:
        assert new_result[key] == old_result[key]


def test_rate_match_missing_power_w_defaults_to_zero():
    p = _make_prefill_dict()
    del p["power_w"]
    d = _make_decode_dict()
    del d["power_w"]
    new_result = _rate_match_dict(p, 1, d, 1)
    old_result = _build_disagg_summary_dict(p, 1, d, 1)
    for key in new_result:
        assert new_result[key] == old_result[key]
