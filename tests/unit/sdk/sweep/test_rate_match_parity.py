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


def test_rate_match_counts_cp_in_gpu_accounting():
    """#1476: a cp8 prefill worker occupies tp*pp*dp*cp GPUs, not tp*pp*dp.

    Before the fix a cp8 worker was accounted as one GPU, inflating
    num_total_gpus-normalized metrics ~8x and producing replica math the
    cluster cannot satisfy.
    """
    p = _make_prefill_dict(tp=1, pp=1, dp=1, cp=8, parallel="tp1pp1dp1cp8")
    d = _make_decode_dict(tp=1, pp=1, dp=1, parallel="tp1pp1dp1")

    for result in (_rate_match_dict(p, 2, d, 4), _build_disagg_summary_dict(p, 2, d, 4)):
        # 2 prefill workers x 8 GPUs (cp8) + 4 decode workers x 1 GPU
        assert result["num_total_gpus"] == 2 * 8 + 4 * 1
        assert result["(p)cp"] == 8
        seq_s = min(p["seq/s"] * 2 * 0.9, d["seq/s"] * 4 * 0.92)
        assert result["tokens/s/gpu"] == pytest.approx(seq_s * p["osl"] / 20)


def test_rate_match_missing_cp_key_defaults_to_one():
    """Rows predating the cp column (older CSVs, partial dicts) keep working."""
    p = _make_prefill_dict()  # no "cp" key
    d = _make_decode_dict()
    for result in (_rate_match_dict(p, 1, d, 1), _build_disagg_summary_dict(p, 1, d, 1)):
        assert result["num_total_gpus"] == 4 + 2  # tp4 prefill + tp2 decode
        assert result["(p)cp"] == 1


def test_rate_match_parity_holds_with_cp():
    p = _make_prefill_dict(cp=4, parallel="tp4pp1dp1cp4")
    d = _make_decode_dict()
    new_result = _rate_match_dict(p, 3, d, 5)
    old_result = _build_disagg_summary_dict(p, 3, d, 5)
    for key in new_result:
        assert new_result[key] == old_result[key], (
            f"Field {key!r} differs with cp: new={new_result[key]} vs old={old_result[key]}"
        )


class TestWorkerGpus:
    """worker_gpus: num_total_gpus is authoritative, dims are the fallback."""

    def test_prefers_authoritative_num_total_gpus(self):
        from aiconfigurator.sdk.picking import worker_gpus

        # Dims say 1 GPU, but the backend stamped 8 (e.g. a dimension this
        # helper's fallback list does not know about yet) — trust the stamp.
        assert worker_gpus({"tp": 1, "pp": 1, "dp": 1, "num_total_gpus": 8}) == 8

    def test_falls_back_to_dims_product_including_cp(self):
        from aiconfigurator.sdk.picking import worker_gpus

        assert worker_gpus({"tp": 2, "pp": 2, "dp": 1, "cp": 4}) == 16

    def test_missing_dims_default_to_one(self):
        from aiconfigurator.sdk.picking import worker_gpus

        assert worker_gpus({"tp": 4}) == 4

    def test_nan_num_total_gpus_falls_back(self):
        from aiconfigurator.sdk.picking import worker_gpus

        assert worker_gpus({"tp": 2, "pp": 1, "dp": 1, "num_total_gpus": float("nan")}) == 2
