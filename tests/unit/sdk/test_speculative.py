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


def test_aggregate_projection_applies_when_scheduler_saw_no_explicit_progress():
    """Legacy flow: run_agg was called without decode_tokens_per_iteration, so
    its scheduling metadata carries no progress marker and the post-hoc scalar
    projection must still apply (regression: treating the scheduler's implicit
    1.0 default as 'already projected' silently froze TPOT at baseline)."""
    original = _summary()
    original.set_step_estimates({"scheduling": {"decode_iterations": 9.0}})

    projected = SpeculativeDecodingProfile(1.0).project_summary(original, role="agg")

    assert projected.get_result_dict()["tpot"] == 5.0
    assert projected.get_result_dict()["tokens/s"] == 90.0


def test_aggregate_projection_never_stacks_on_mismatched_scheduler_progress(caplog):
    """If run_agg already applied a different progress, the scheduler value is
    authoritative: re-scaling on top would compound two different speedups."""
    original = _summary()
    original.set_step_estimates({"scheduling": {"decode_tokens_per_iteration": 1.5}})

    with caplog.at_level("WARNING", logger="aiconfigurator.sdk.speculative"):
        projected = SpeculativeDecodingProfile(1.0).project_summary(original, role="agg")

    assert projected.get_result_dict()["tpot"] == 10.0
    assert projected.get_result_dict()["tokens/s"] == 45.0
    assert any("decode_tokens_per_iteration" in record.message for record in caplog.records)


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


class TestResolveDsparkNextn:
    """Unit tests for resolve_dspark_nextn in config_builders."""

    def test_non_dspark_returns_none(self, monkeypatch):
        from aiconfigurator_core.sdk.config_builders import resolve_dspark_nextn

        monkeypatch.setattr(
            "aiconfigurator_core.sdk.utils.get_model_config_from_model_path",
            lambda _: {"architecture": "LlamaForCausalLM"},
        )
        assert resolve_dspark_nextn("meta-llama/Llama-3.1-8B-Instruct") is None

    def test_kimi_k3_returns_block_size(self, monkeypatch):
        from aiconfigurator_core.sdk.config_builders import resolve_dspark_nextn

        monkeypatch.setattr(
            "aiconfigurator_core.sdk.utils.get_model_config_from_model_path",
            lambda _: {"architecture": "KimiK3ForConditionalGeneration"},
        )
        assert resolve_dspark_nextn("moonshotai/Kimi-K3") == 7

    def test_empty_model_path_raises(self):
        from aiconfigurator_core.sdk.config_builders import resolve_dspark_nextn

        with pytest.raises(ValueError, match="requires a model path"):
            resolve_dspark_nextn("")

    def test_expected_fetch_failure_warns_and_returns_none(self, monkeypatch, caplog):
        from aiconfigurator_core.sdk.config_builders import resolve_dspark_nextn
        from aiconfigurator_core.sdk.utils import HuggingFaceDownloadError

        monkeypatch.setattr(
            "aiconfigurator_core.sdk.utils.get_model_config_from_model_path",
            lambda _: (_ for _ in ()).throw(HuggingFaceDownloadError("network error")),
        )
        with caplog.at_level("WARNING", logger="aiconfigurator_core.sdk.config_builders"):
            assert resolve_dspark_nextn("some/model") is None
        assert "some/model" in caplog.text
        assert "network error" in caplog.text

    def test_unexpected_failure_propagates(self, monkeypatch):
        from aiconfigurator_core.sdk.config_builders import resolve_dspark_nextn

        monkeypatch.setattr(
            "aiconfigurator_core.sdk.utils.get_model_config_from_model_path",
            lambda _: (_ for _ in ()).throw(RuntimeError("programming error")),
        )
        with pytest.raises(RuntimeError, match="programming error"):
            resolve_dspark_nextn("some/model")

    def test_malformed_metadata_propagates(self, monkeypatch):
        from aiconfigurator_core.sdk.config_builders import resolve_dspark_nextn

        monkeypatch.setattr(
            "aiconfigurator_core.sdk.utils.get_model_config_from_model_path",
            lambda _: {},
        )
        with pytest.raises(KeyError, match="architecture"):
            resolve_dspark_nextn("some/model")
