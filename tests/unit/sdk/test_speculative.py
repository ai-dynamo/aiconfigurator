# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pandas as pd
import pytest

from aiconfigurator.sdk.config import RuntimeConfig
from aiconfigurator.sdk.inference_summary import InferenceSummary
from aiconfigurator.sdk.speculative import (
    SpeculativeDecodingProfile,
    normalize_speculative_decoding,
)

pytestmark = pytest.mark.unit


def _summary() -> InferenceSummary:
    summary = InferenceSummary(RuntimeConfig(isl=128, osl=9))
    row = {
        "isl": 128,
        "osl": 9,
        "ttft": 20.0,
        "tpot": 10.0,
        "request_latency": 100.0,
        "request_rate": 5.0,
        "seq/s": 5.0,
        "seq/s/gpu": 1.25,
        "tokens/s": 45.0,
        "tokens/s/gpu": 11.25,
        "tokens/s/user": 100.0,
        "generation_latency": 80.0,
        "memory": 42.0,
    }
    summary.set_summary_df(pd.DataFrame([row]))
    summary.set_result_dict(dict(row))
    summary.set_generation_latency_dict({"generation_qkv": 80.0})
    return summary


def test_active_mtp_requires_explicit_acceptance_above_core():
    with pytest.raises(ValueError, match="requires 'nextn_accepted'"):
        normalize_speculative_decoding(2, None)


def test_acceptance_is_ignored_when_mtp_is_disabled():
    profile = SpeculativeDecodingProfile.from_inputs(0, 0.9)
    assert profile.expected_accepted_tokens == 0.0


@pytest.mark.parametrize("accepted", [-0.1, 2.1, float("inf"), float("nan")])
def test_acceptance_range_is_validated_by_upper_layer(accepted):
    with pytest.raises(ValueError, match="nextn_accepted"):
        normalize_speculative_decoding(2, accepted)


@pytest.mark.parametrize("accepted", [-0.1, float("inf"), float("nan")])
def test_direct_profile_rejects_invalid_acceptance(accepted):
    with pytest.raises(ValueError, match="finite and non-negative"):
        SpeculativeDecodingProfile(accepted)


def test_expected_progress_projects_service_metrics_not_core_breakdown():
    original = _summary()
    projected = SpeculativeDecodingProfile(1.0).project_summary(original, role="agg")
    row = projected.get_result_dict()

    assert row["ttft"] == 20.0
    assert row["tpot"] == 5.0
    assert row["request_latency"] == 60.0
    assert row["seq/s"] == 10.0
    assert row["tokens/s"] == 90.0
    assert row["tokens/s/user"] == 200.0
    assert row["generation_latency"] == 40.0
    assert row["memory"] == 42.0

    # The raw per-operation iteration cost from aic-core remains available and the
    # cached backend summary is not mutated by the projection.
    assert projected.get_generation_latency_dict() == {"generation_qkv": 80.0}
    assert original.get_result_dict()["tpot"] == 10.0


def test_aggregate_projection_does_not_double_apply_scheduler_progress():
    original = _summary()
    original.set_step_estimates(
        {
            "scheduling": {
                "decode_tokens_per_iteration": 2.0,
                "decode_iterations": 5.0,
            }
        }
    )

    projected = SpeculativeDecodingProfile(1.0).project_summary(original, role="agg")

    assert projected is not original
    assert projected.get_result_dict()["tpot"] == 10.0
    assert projected.get_result_dict()["tokens/s"] == 45.0


def test_aggregate_projection_reapplies_vllm_little_law_cap():
    original = _summary()
    frame = original.get_summary_df().copy()
    frame["backend"] = "vllm"
    frame["concurrency"] = 1
    frame["request_rate"] = 10.0
    frame["seq/s"] = 10.0
    frame["seq/s/gpu"] = 2.5
    frame["tokens/s"] = 80.0
    frame["tokens/s/gpu"] = 20.0
    original.set_summary_df(frame)
    original.set_result_dict(frame.iloc[0].to_dict())

    projected = SpeculativeDecodingProfile(1.0).project_summary(original, role="agg")
    row = projected.get_result_dict()

    # Projected request latency is 60 ms, so one concurrent request caps the
    # request rate at 1000 / 60 rather than the naive 10 * 2 = 20 seq/s.
    assert row["request_latency"] == 60.0
    assert row["seq/s"] == pytest.approx(16.667)
    assert row["request_rate"] == pytest.approx(16.667)
    assert row["tokens/s"] == pytest.approx(133.333)
    assert row["tokens/s/gpu"] == pytest.approx(33.333)


def test_prefill_metrics_are_not_projected():
    summary = _summary()
    assert SpeculativeDecodingProfile(1.0).project_summary(summary, role="prefill") is summary
