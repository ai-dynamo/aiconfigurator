# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for exposed rate-match degradation factors in pick_autoscale.

``pick_autoscale`` used to rely on ``_build_disagg_summary_dict`` defaults for
prefill/decode degradation. They are now keyword args so a caller can calibrate
the analytical disagg point against measured silicon.
"""

import pandas as pd
import pytest

from aiconfigurator.sdk.performance_result import MOE_COMM_FALLBACKS_COLUMN, MoECommFallback
from aiconfigurator.sdk.picking import (
    _RATE_MATCHING_DECODE_DEGRADATION_FACTOR,
    _RATE_MATCHING_PREFILL_DEGRADATION_FACTOR,
    pick_autoscale,
)

pytestmark = pytest.mark.unit


def _prefill_dict(**overrides) -> dict:
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
        "cp": 1,
        "parallel": "tp4",
        "ttft": 80.0,
        "tpot": 0.0,
        "seq/s": 10.0,
        "seq/s/gpu": 2.5,
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
        "encoder_latency": 0.0,
        "encoder_memory": 0.0,
    }
    base.update(overrides)
    return base


def _decode_dict(**overrides) -> dict:
    base = {
        "model": "test-model",
        "isl": 4000,
        "osl": 500,
        "prefix": 0,
        "concurrency": 32,
        "bs": 32,
        "global_bs": 32,
        "tp": 4,
        "pp": 1,
        "dp": 1,
        "moe_tp": 1,
        "moe_ep": 1,
        "cp": 1,
        "parallel": "tp4",
        "ttft": 0.0,
        "tpot": 10.0,
        "seq/s": 3.0,
        "seq/s/gpu": 0.75,
        "tokens/s/user": 100.0,
        "gemm": "fp8",
        "kvcache": "fp8",
        "fmha": "fp8",
        "moe": "fp8",
        "comm": "half",
        "memory": 40.0,
        "backend": "trtllm",
        "version": "1.3.0",
        "system": "h200_sxm",
        "power_w": 600.0,
    }
    base.update(overrides)
    return base


def _run(*, prefill_seq_s: float = 10.0, decode_seq_s: float = 3.0, **kwargs):
    prefill_df = pd.DataFrame([_prefill_dict(**{"seq/s": prefill_seq_s, "seq/s/gpu": prefill_seq_s / 4})])
    decode_df = pd.DataFrame([_decode_dict(**{"seq/s": decode_seq_s, "seq/s/gpu": decode_seq_s / 4})])
    # Loose SLAs so both single candidates always survive filtering.
    return pick_autoscale(prefill_df, decode_df, target_ttft=1e9, target_tpot=1e9, **kwargs)


def test_default_factors_match_constants() -> None:
    """Omitting the factors reproduces the built-in constants exactly."""
    default = _run()
    explicit = _run(
        prefill_degradation_factor=_RATE_MATCHING_PREFILL_DEGRADATION_FACTOR,
        decode_degradation_factor=_RATE_MATCHING_DECODE_DEGRADATION_FACTOR,
    )
    row_d = default["best_config_df"].iloc[0]
    row_e = explicit["best_config_df"].iloc[0]
    assert row_d["seq/s"] == pytest.approx(row_e["seq/s"])


def test_decode_bound_seq_s_scales_with_decode_factor() -> None:
    """With decode as the rate-match bottleneck, seq/s tracks the decode factor.

    prefill seq/s (10) * 0.9 = 9 vs decode seq/s (3) * factor. Decode binds, so
    system seq/s == 3 * decode_degradation_factor.
    """
    r_hi = _run(decode_degradation_factor=0.92)
    r_lo = _run(decode_degradation_factor=0.5)
    seq_hi = r_hi["best_config_df"].iloc[0]["seq/s"]
    seq_lo = r_lo["best_config_df"].iloc[0]["seq/s"]
    assert seq_hi == pytest.approx(3.0 * 0.92)
    assert seq_lo == pytest.approx(3.0 * 0.5)
    assert seq_lo < seq_hi


def test_prefill_bound_seq_s_scales_with_prefill_factor() -> None:
    """With prefill as the bottleneck, seq/s tracks the prefill factor."""
    r_hi = _run(prefill_seq_s=2.0, decode_seq_s=10.0, prefill_degradation_factor=0.9)
    r_lo = _run(prefill_seq_s=2.0, decode_seq_s=10.0, prefill_degradation_factor=0.5)
    seq_hi = r_hi["best_config_df"].iloc[0]["seq/s"]
    seq_lo = r_lo["best_config_df"].iloc[0]["seq/s"]
    assert seq_hi == pytest.approx(2.0 * 0.9)
    assert seq_lo == pytest.approx(2.0 * 0.5)
    assert seq_lo < seq_hi


def test_full_utilization_recovers_undegraded_rate() -> None:
    """Factors of 1.0 remove the degradation entirely (decode-bound -> 3.0)."""
    r = _run(prefill_degradation_factor=1.0, decode_degradation_factor=1.0)
    assert r["best_config_df"].iloc[0]["seq/s"] == pytest.approx(3.0)


def test_autoscale_preserves_and_dedupes_role_fallback_provenance() -> None:
    context = MoECommFallback("context", "deepep_ht", 32, 8, 8, 1)
    generation = MoECommFallback("generation", "deepep_ll", 32, 8, 8, 1)
    prefill_df = pd.DataFrame([_prefill_dict(**{MOE_COMM_FALLBACKS_COLUMN: (context,)})])
    decode_df = pd.DataFrame([_decode_dict(**{MOE_COMM_FALLBACKS_COLUMN: (generation, context)})])

    result = pick_autoscale(prefill_df, decode_df, target_ttft=1e9, target_tpot=1e9)

    assert result["best_config_df"].iloc[0][MOE_COMM_FALLBACKS_COLUMN] == (context, generation)
