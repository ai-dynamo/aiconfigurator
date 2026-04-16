# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for pick_optimization_type in sdk/picking.py."""

import pandas as pd
import pytest

from aiconfigurator.sdk.picking import pick_optimization_type


def _make_agg_df(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal ColumnsAgg-shaped DataFrame for testing."""
    base = {
        "model": "test-model",
        "isl": 4000,
        "osl": 1000,
        "prefix": 0,
        "concurrency": 10,
        "request_rate": 5.0,
        "bs": 8,
        "global_bs": 8,
        "request_latency": 0.0,
        "seq/s": 1.0,
        "seq/s/gpu": 0.5,
        "tokens/s": 1000.0,
        "tokens/s/user": 100.0,
        "num_total_gpus": 1,
        "tp": 1,
        "pp": 1,
        "dp": 1,
        "moe_tp": 1,
        "moe_ep": 1,
        "parallel": "tp1pp1dp1",
        "gemm": "",
        "kvcache": "",
        "fmha": "",
        "moe": "",
        "comm": "",
        "memory": "",
        "balance_score": 0.0,
        "num_ctx_reqs": 0,
        "num_gen_reqs": 0,
        "num_tokens": 0,
        "backend": "vllm",
        "version": "0.1",
        "system": "h200_sxm",
        "power_w": 0.0,
    }
    records = []
    for r in rows:
        record = dict(base)
        record.update(r)
        # Derive request_latency if not set explicitly
        if "request_latency" not in r:
            record["request_latency"] = record["ttft"] + record["tpot"] * (record["osl"] - 1)
        records.append(record)
    return pd.DataFrame(records)


@pytest.mark.unit
class TestPickOptimizationType:
    """Verify pick_optimization_type sorts by the right metric."""

    def test_throughput_selects_highest_tokens_per_gpu(self):
        df = _make_agg_df(
            [
                {"tokens/s/gpu": 100, "tpot": 20, "ttft": 50, "parallel": "tp1"},
                {"tokens/s/gpu": 200, "tpot": 30, "ttft": 60, "parallel": "tp2"},
                {"tokens/s/gpu": 150, "tpot": 25, "ttft": 55, "parallel": "tp4"},
            ]
        )
        result = pick_optimization_type(
            pareto_df=df,
            optimization_type="throughput",
            total_gpus=8,
            serving_mode="agg",
        )
        best = result["best_config_df"].iloc[0]
        assert best["tokens/s/gpu"] == 200

    def test_latency_selects_lowest_tpot(self):
        df = _make_agg_df(
            [
                {"tokens/s/gpu": 100, "tpot": 20, "ttft": 50, "parallel": "tp1"},
                {"tokens/s/gpu": 200, "tpot": 30, "ttft": 60, "parallel": "tp2"},
                {"tokens/s/gpu": 150, "tpot": 10, "ttft": 45, "parallel": "tp4"},
            ]
        )
        result = pick_optimization_type(
            pareto_df=df,
            optimization_type="latency",
            total_gpus=8,
            serving_mode="agg",
        )
        best = result["best_config_df"].iloc[0]
        assert best["tpot"] == 10

    def test_returns_best_latencies(self):
        df = _make_agg_df(
            [
                {"tokens/s/gpu": 100, "tpot": 15, "ttft": 40, "parallel": "tp1"},
            ]
        )
        result = pick_optimization_type(
            pareto_df=df,
            optimization_type="throughput",
            total_gpus=8,
            serving_mode="agg",
        )
        lat = result["best_latencies"]
        assert lat["ttft"] == 40
        assert lat["tpot"] == 15
        assert lat["request_latency"] > 0

    def test_returns_best_throughput(self):
        df = _make_agg_df(
            [
                {"tokens/s/gpu": 100, "tpot": 20, "ttft": 50, "parallel": "tp1", "num_total_gpus": 2},
            ]
        )
        result = pick_optimization_type(
            pareto_df=df,
            optimization_type="throughput",
            total_gpus=8,
            serving_mode="agg",
        )
        assert result["best_throughput"] > 0

    def test_empty_df_returns_empty_result(self):
        result = pick_optimization_type(
            pareto_df=pd.DataFrame(),
            optimization_type="throughput",
            total_gpus=8,
            serving_mode="agg",
        )
        assert result["best_config_df"].empty
        assert result["best_throughput"] == 0.0
        assert result["best_latencies"]["ttft"] == 0.0

    def test_none_df_returns_empty_result(self):
        result = pick_optimization_type(
            pareto_df=None,
            optimization_type="latency",
            total_gpus=8,
            serving_mode="agg",
        )
        assert result["best_config_df"].empty

    def test_top_n_limits_output(self):
        rows = [{"tokens/s/gpu": i * 10, "tpot": 50 - i, "ttft": 40, "parallel": f"tp{i}"} for i in range(1, 10)]
        df = _make_agg_df(rows)
        result = pick_optimization_type(
            pareto_df=df,
            optimization_type="throughput",
            total_gpus=8,
            serving_mode="agg",
            top_n=3,
        )
        assert len(result["best_config_df"]) == 3

    def test_pareto_frontier_returned(self):
        df = _make_agg_df(
            [
                {"tokens/s/gpu": 100, "tpot": 20, "ttft": 50, "parallel": "tp1"},
                {"tokens/s/gpu": 200, "tpot": 30, "ttft": 60, "parallel": "tp2"},
            ]
        )
        result = pick_optimization_type(
            pareto_df=df,
            optimization_type="throughput",
            total_gpus=8,
            serving_mode="agg",
        )
        assert "pareto_frontier_df" in result
        assert not result["pareto_frontier_df"].empty

    def test_dedup_by_parallel_key(self):
        """When two rows share the same 'parallel' value, only the best is kept."""
        df = _make_agg_df(
            [
                {"tokens/s/gpu": 100, "tpot": 20, "ttft": 50, "parallel": "tp1"},
                {"tokens/s/gpu": 200, "tpot": 30, "ttft": 60, "parallel": "tp1"},
            ]
        )
        result = pick_optimization_type(
            pareto_df=df,
            optimization_type="throughput",
            total_gpus=8,
            serving_mode="agg",
        )
        assert len(result["best_config_df"]) == 1
        # The kept row should be the better one (higher throughput)
        assert result["best_config_df"].iloc[0]["tokens/s/gpu"] == 200

    def test_throughput_tiebreak_by_tpot(self):
        """When throughput is tied, lower tpot should win."""
        df = _make_agg_df(
            [
                {"tokens/s/gpu": 100, "tpot": 30, "ttft": 50, "parallel": "tp1", "num_total_gpus": 1},
                {"tokens/s/gpu": 100, "tpot": 10, "ttft": 50, "parallel": "tp2", "num_total_gpus": 1},
            ]
        )
        result = pick_optimization_type(
            pareto_df=df,
            optimization_type="throughput",
            total_gpus=8,
            serving_mode="agg",
        )
        # With same throughput, lower tpot should rank first
        assert result["best_config_df"].iloc[0]["tpot"] == 10

    def test_latency_tiebreak_by_throughput(self):
        """When tpot is tied, higher throughput should win."""
        df = _make_agg_df(
            [
                {"tokens/s/gpu": 100, "tpot": 10, "ttft": 50, "parallel": "tp1", "num_total_gpus": 1},
                {"tokens/s/gpu": 200, "tpot": 10, "ttft": 50, "parallel": "tp2", "num_total_gpus": 1},
            ]
        )
        result = pick_optimization_type(
            pareto_df=df,
            optimization_type="latency",
            total_gpus=8,
            serving_mode="agg",
        )
        assert result["best_config_df"].iloc[0]["tokens/s/gpu"] == 200
