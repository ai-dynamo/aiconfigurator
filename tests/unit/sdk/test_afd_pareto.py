# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the AFD pareto sweep helpers and AFD picking."""

import copy
import math

import pandas as pd
import pytest

from aiconfigurator.sdk import common
from aiconfigurator.sdk.pareto_analysis import (
    _AFD_DECODE_DEGRADATION,
    _AFD_PREFILL_DEGRADATION,
    _AFD_TTFT_CORRECTION_FACTOR,
    _combine_afd_row_with_static_prefill,
    _enumerate_afd_prefill_options,
)
from aiconfigurator.sdk.picking import pick_default

pytestmark = pytest.mark.unit


def _afd_decode_row(**overrides) -> dict:
    row = {
        "model": "test-model",
        "phase": "decode",
        "isl": 4000,
        "osl": 1000,
        "(a)nodes": 1,
        "(a)tp": 4,
        "(a)bs": 128,
        "(a)workers": 2,
        "(f)nodes": 1,
        "(f)tp": 8,
        "(f)ep": 1,
        "(f)workers": 8,
        "ttft": 0.0,
        "tpot": 10.0,
        "request_latency": 9990.0,
        "seq/s": 8.0,
        "request_rate": 8.0,
        "tokens/s": 8000.0,
        "tokens/s/gpu": 500.0,
        "tokens/s/user": 100.0,
        "concurrency": 256,
        "parallel": "a1n-tp4+f1n-ep1",
        "num_total_gpus": 16,
        "memory": 60.0,
        "power_w": 0.0,
        "backend": "trtllm",
        "version": "test",
        "system": "h200_sxm",
    }
    row.update(overrides)
    return row


class TestCombineAfdWithStaticPrefill:
    def test_rate_matched_combination(self):
        row = _afd_decode_row()
        options = [{"tp": 2, "num_gpus": 2, "ttft": 500.0, "seq_s": 2.0, "memory": 40.0, "power_w": 0.0}]
        combined = _combine_afd_row_with_static_prefill(row, options, target_ttft=2000.0)

        effective_per_worker = 2.0 * _AFD_PREFILL_DEGRADATION
        effective_decode_seq_s = 8.0 * _AFD_DECODE_DEGRADATION
        expected_workers = max(1, math.ceil(effective_decode_seq_s / effective_per_worker))
        expected_ttft = 500.0 * _AFD_TTFT_CORRECTION_FACTOR
        expected_seq_s = min(effective_decode_seq_s, expected_workers * effective_per_worker)
        assert combined["(p)workers"] == expected_workers
        assert combined["(p)tp"] == 2
        assert combined["ttft"] == pytest.approx(expected_ttft)
        assert combined["tpot"] == 10.0
        assert combined["request_latency"] == pytest.approx(expected_ttft + 10.0 * 999)
        # GPU budget merges the AFD pool and the prefill pool.
        assert combined["num_total_gpus"] == 16 + expected_workers * 2
        # Rate-matched throughput includes AFD's symmetric degradation factors.
        assert combined["seq/s"] == pytest.approx(expected_seq_s)
        assert combined["tokens/s"] == pytest.approx(expected_seq_s * 1000)
        assert combined["(p)impl"] == "static_ctx"
        assert combined["(d)impl"] == "afd"

    def test_prefers_ttft_compliant_option(self):
        row = _afd_decode_row()
        options = [
            # cheaper but violates the TTFT target (3000 * 1.8 = 5400 > 2000)
            {"tp": 1, "num_gpus": 1, "ttft": 3000.0, "seq_s": 8.0, "memory": 40.0, "power_w": 0.0},
            # compliant (800 * 1.8 = 1440 < 2000)
            {"tp": 4, "num_gpus": 4, "ttft": 800.0, "seq_s": 8.0, "memory": 40.0, "power_w": 0.0},
        ]
        combined = _combine_afd_row_with_static_prefill(row, options, target_ttft=2000.0)
        assert combined["(p)tp"] == 4
        assert combined["ttft"] == pytest.approx(800.0 * _AFD_TTFT_CORRECTION_FACTOR)

    def test_filters_options_that_violate_ttft(self):
        row = _afd_decode_row()
        options = [{"tp": 1, "num_gpus": 1, "ttft": 3000.0, "seq_s": 8.0, "memory": 40.0, "power_w": 0.0}]
        assert _combine_afd_row_with_static_prefill(row, options, target_ttft=2000.0) is None

    def test_prefers_min_gpu_option_that_satisfies_request_latency(self):
        row = _afd_decode_row()
        options = [
            # Would be cheapest, but 1000 * 1.8 + 9990 = 11790 > 11500.
            {"tp": 1, "num_gpus": 1, "ttft": 1000.0, "seq_s": 8.0, "memory": 40.0, "power_w": 0.0},
            # Feasible and cheaper than the last option.
            {"tp": 4, "num_gpus": 4, "ttft": 700.0, "seq_s": 8.0, "memory": 40.0, "power_w": 0.0},
            {"tp": 8, "num_gpus": 8, "ttft": 600.0, "seq_s": 8.0, "memory": 40.0, "power_w": 0.0},
        ]

        combined = _combine_afd_row_with_static_prefill(
            row,
            options,
            target_ttft=2000.0,
            target_request_latency=11500.0,
        )

        assert combined["(p)tp"] == 4
        assert combined["request_latency"] <= 11500.0

    def test_returns_none_without_options(self):
        assert _combine_afd_row_with_static_prefill(_afd_decode_row(), [], target_ttft=None) is None

    def test_respects_remaining_total_gpu_budget(self):
        row = _afd_decode_row(num_total_gpus=16)
        options = [
            {
                "tp": 8,
                "pp": 1,
                "dp": 1,
                "moe_tp": 8,
                "moe_ep": 1,
                "batch_size": 8,
                "num_gpus": 8,
                "ttft": 400.0,
                "seq_s": 8.0,
                "memory": 40.0,
                "power_w": 0.0,
                "system": "h200_sxm",
                "backend": "trtllm",
                "version": "too-large",
            },
            {
                "tp": 2,
                "pp": 2,
                "dp": 1,
                "moe_tp": 2,
                "moe_ep": 1,
                "batch_size": 4,
                "num_gpus": 4,
                "ttft": 600.0,
                "seq_s": 16.0,
                "memory": 40.0,
                "power_w": 0.0,
                "system": "h200_sxm",
                "backend": "trtllm",
                "version": "fits",
            },
        ]

        combined = _combine_afd_row_with_static_prefill(row, options, target_ttft=2000.0, total_gpus=20)

        assert combined["num_total_gpus"] == 20
        assert combined["(p)tp"] == 2
        assert combined["(p)pp"] == 2
        assert combined["(p)bs"] == 4
        assert combined["(p)version"] == "fits"


class TestEnumerateAfdPrefillOptions:
    def test_enumerates_parallel_configs_and_batch_sizes(self, monkeypatch):
        captured = []

        class FakeBackend:
            pass

        class FakeSummary:
            def __init__(self, model_config, batch_size):
                self._model_config = model_config
                self._batch_size = batch_size

            def check_oom(self):
                return False

            def get_result_dict(self):
                num_gpus = (
                    self._model_config.tp_size
                    * self._model_config.pp_size
                    * self._model_config.attention_dp_size
                    * self._model_config.cp_size
                )
                return {
                    "ttft": 100.0 + self._batch_size,
                    "seq/s": 10.0 * self._batch_size,
                    "num_total_gpus": num_gpus,
                    "memory": 20.0,
                    "power_w": 30.0,
                    "system": "prefill-system",
                    "backend": "prefill-backend",
                    "version": "prefill-version",
                }

        class FakeInferenceSession:
            def __init__(self, *, model, **_kwargs):
                self._model = model

            def run_static(self, runtime_config, **_kwargs):
                model_config = self._model.model_config
                captured.append(
                    (
                        model_config.tp_size,
                        model_config.pp_size,
                        model_config.attention_dp_size,
                        model_config.moe_tp_size,
                        model_config.moe_ep_size,
                        model_config.cp_size,
                        runtime_config.batch_size,
                    )
                )
                return FakeSummary(model_config, runtime_config.batch_size)

        monkeypatch.setattr("aiconfigurator.sdk.pareto_analysis.get_backend", lambda _name: FakeBackend())
        monkeypatch.setattr(
            "aiconfigurator.sdk.pareto_analysis.get_model",
            lambda _model_path, model_config, _backend_name: type(
                "FakeModel",
                (),
                {"model_config": copy.deepcopy(model_config)},
            )(),
        )
        monkeypatch.setattr("aiconfigurator.sdk.pareto_analysis.InferenceSession", FakeInferenceSession)

        options = _enumerate_afd_prefill_options(
            model_path="Qwen/Qwen3-32B",
            runtime_config=type("RuntimeConfig", (), {"batch_size": 1})(),
            database=object(),
            backend_name="trtllm",
            gpus_per_node=8,
            prefill_parallel_config_list=[(2, 2, 1, 2, 1), (4, 1, 2, 4, 1, 2)],
            prefill_batch_size_list=[1, 8],
            prefill_system_name="prefill-system",
            prefill_backend_name="prefill-backend",
            prefill_backend_version="prefill-version",
            max_candidates=8,
        )

        assert captured == [
            (2, 2, 1, 2, 1, 1, 1),
            (2, 2, 1, 2, 1, 1, 8),
            (4, 1, 2, 4, 1, 2, 1),
            (4, 1, 2, 4, 1, 2, 8),
        ]
        assert [(option["tp"], option["pp"], option["dp"], option["batch_size"]) for option in options] == [
            (2, 2, 1, 1),
            (2, 2, 1, 8),
            (4, 1, 2, 1),
            (4, 1, 2, 8),
        ]
        assert options[0]["num_gpus"] == 4
        assert options[-1]["num_gpus"] == 16
        assert options[-1]["cp"] == 2

    def test_prefill_candidate_overflow_errors(self):
        with pytest.raises(ValueError, match="afd_config.prefill_search.max_candidates=1"):
            _enumerate_afd_prefill_options(
                model_path="Qwen/Qwen3-32B",
                runtime_config=type("RuntimeConfig", (), {"batch_size": 1})(),
                database=object(),
                backend_name="trtllm",
                gpus_per_node=8,
                prefill_parallel_config_list=[(1, 1, 1, 1, 1)],
                prefill_batch_size_list=[1, 2],
                max_candidates=1,
            )


class TestPickDefaultAfd:
    def test_pick_default_groups_by_parallel(self):
        rows = [
            _afd_decode_row(parallel="a1n-tp4+f1n-ep1", tpot=10.0, **{}),
            _afd_decode_row(parallel="a1n-tp4+f1n-ep1", tpot=12.0, **{"tokens/s/gpu": 450.0}),
            _afd_decode_row(parallel="a2n-tp8+f1n-ep1", tpot=8.0, **{"tokens/s/gpu": 400.0, "num_total_gpus": 24}),
        ]
        df = pd.DataFrame(rows, columns=common.ColumnsAFD)
        result = pick_default(
            pareto_df=df,
            total_gpus=48,
            serving_mode="afd",
            target_tpot=30.0,
            top_n=5,
        )
        best_df = result["best_config_df"]
        assert not best_df.empty
        # one row per parallel strategy after grouping
        assert best_df["parallel"].nunique() == len(best_df)
        assert result["best_throughput"] > 0
