# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for SLA constraint filtering in the picking layer.

Covers:
- TPOT constraint filtering excludes configs that violate the target
- --strict-sla pre-filters the full pareto_df (TTFT + TPOT) before the
  Pareto frontier is computed, so pareto.csv only shows SLA-compliant data
- Default (non-strict) preserves the full Pareto frontier
"""

import pandas as pd
import pytest

from aiconfigurator.sdk.pareto_analysis import (
    get_best_configs_under_tpot_constraint,
)
from aiconfigurator.sdk.picking import pick_default

pytestmark = pytest.mark.unit


def _make_pareto_df(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal pareto-style DataFrame for testing picking logic."""
    defaults = {
        "model": "test-model",
        "isl": 4000,
        "osl": 1000,
        "prefix": 0,
        "concurrency": 1,
        "request_rate": 1.0,
        "bs": 1,
        "global_bs": 1,
        "seq/s": 10.0,
        "seq/s/gpu": 5.0,
        "tokens/s": 1000.0,
        "tokens/s/user": 100.0,
        "num_total_gpus": 1,
        "tp": 1,
        "pp": 1,
        "dp": 1,
        "moe_tp": 0,
        "moe_ep": 0,
        "parallel": "tp1_pp1_dp1",
        "gemm": "fp16",
        "kvcache": "fp16",
        "fmha": "fp16",
        "moe": "none",
        "comm": "none",
        "memory": 0.5,
        "balance_score": 1.0,
        "num_ctx_reqs": 1,
        "num_gen_reqs": 1,
        "num_tokens": 1000,
        "ctx_tokens": 500,
        "gen_tokens": 500,
        "backend": "trtllm",
        "version": "1.0.0",
        "system": "h100_sxm",
        "power_w": 300.0,
        "request_latency": 0.0,
    }
    records = []
    for row in rows:
        record = {**defaults, **row}
        records.append(record)
    return pd.DataFrame(records)


class TestTpotConstraintFiltering:
    """Configs violating the user's --tpot must not appear in best_config_topn."""

    def test_only_sla_compliant_configs_returned(self):
        """Configs with tpot > target should be excluded."""
        df = _make_pareto_df(
            [
                {"tpot": 10.0, "ttft": 500.0, "tokens/s/gpu": 800.0},
                {"tpot": 20.0, "ttft": 500.0, "tokens/s/gpu": 1200.0},  # violates --tpot 15
                {"tpot": 45.0, "ttft": 500.0, "tokens/s/gpu": 1500.0},  # violates --tpot 15
            ]
        )
        result = get_best_configs_under_tpot_constraint(
            total_gpus=8,
            pareto_df=df,
            target_tpot=15.0,
            top_n=5,
        )
        assert not result.empty
        assert (result["tpot"] <= 15.0).all(), "All returned configs must meet the TPOT SLA"

    def test_fallback_when_nothing_meets_sla(self):
        """When no config meets the SLA, fallback returns closest violators."""
        df = _make_pareto_df(
            [
                {"tpot": 20.0, "ttft": 500.0, "tokens/s/gpu": 800.0},
                {"tpot": 30.0, "ttft": 500.0, "tokens/s/gpu": 1200.0},
            ]
        )
        result = get_best_configs_under_tpot_constraint(
            total_gpus=8,
            pareto_df=df,
            target_tpot=5.0,
            top_n=5,
        )
        # Should return something (fallback), sorted by closest to target
        assert not result.empty
        assert result.iloc[0]["tpot"] == 20.0, "Fallback should return the config closest to the SLA target"


class TestStrictSlaFiltering:
    """--strict-sla pre-filters pareto_df on TPOT so the Pareto frontier is SLA-clean."""

    def test_strict_filters_pareto_frontier_on_tpot(self):
        """With strict_sla=True, TPOT violators are removed from pareto_df
        *before* the Pareto frontier is computed."""
        df = _make_pareto_df(
            [
                {"tpot": 10.0, "ttft": 500.0, "tokens/s/gpu": 800.0, "tokens/s/user": 100.0},
                {"tpot": 50.0, "ttft": 500.0, "tokens/s/gpu": 1500.0, "tokens/s/user": 300.0},  # TPOT violation
                {"tpot": 14.0, "ttft": 400.0, "tokens/s/gpu": 600.0, "tokens/s/user": 80.0},  # meets SLA
            ]
        )
        result = pick_default(
            pareto_df=df,
            total_gpus=8,
            serving_mode="agg",
            target_tpot=15.0,
            top_n=5,
            strict_sla=True,
        )
        frontier = result["pareto_frontier_df"]
        best = result["best_config_df"]

        # Pareto frontier should only contain TPOT-compliant configs
        assert not frontier.empty
        assert (frontier["tpot"] <= 15.0).all(), "Pareto frontier must be TPOT-clean"

        # Best config should also be clean
        assert not best.empty
        assert (best["tpot"] <= 15.0).all()

    def test_strict_returns_empty_when_nothing_meets_tpot(self):
        """When no config meets --tpot, strict filtering returns empty results."""
        df = _make_pareto_df(
            [
                {"tpot": 20.0, "ttft": 500.0, "tokens/s/gpu": 800.0},
                {"tpot": 30.0, "ttft": 500.0, "tokens/s/gpu": 1200.0},
            ]
        )
        result = pick_default(
            pareto_df=df,
            total_gpus=8,
            serving_mode="agg",
            target_tpot=15.0,
            top_n=5,
            strict_sla=True,
        )
        assert result["pareto_frontier_df"].empty
        assert result["best_config_df"].empty

    def test_strict_filters_request_latency_path(self):
        """Strict filtering works when using request-latency constraint."""
        df = _make_pareto_df(
            [
                {"tpot": 10.0, "ttft": 500.0, "tokens/s/gpu": 800.0, "request_latency": 8000.0},
                {"tpot": 12.0, "ttft": 500.0, "tokens/s/gpu": 1200.0, "request_latency": 9000.0},
                {"tpot": 8.0, "ttft": 400.0, "tokens/s/gpu": 600.0, "request_latency": 15000.0},  # req_lat violation
            ]
        )
        result = pick_default(
            pareto_df=df,
            total_gpus=8,
            serving_mode="agg",
            target_request_latency=10000.0,
            top_n=5,
            strict_sla=True,
        )
        frontier = result["pareto_frontier_df"]
        assert not frontier.empty
        assert (frontier["request_latency"] <= 10000.0).all()


class TestNonStrictPreservesFullPareto:
    """Without --strict-sla, the full Pareto frontier is preserved."""

    def test_full_frontier_preserved(self):
        """Without strict_sla, all configs remain in the Pareto frontier."""
        # Two non-dominated points: one wins on tokens/s/user, the other on tokens/s/gpu.
        # Both are on the Pareto frontier.  The second violates --tpot 15
        # but should still appear in the frontier when strict mode is off.
        df = _make_pareto_df(
            [
                {"tpot": 10.0, "ttft": 500.0, "tokens/s/gpu": 2000.0, "tokens/s/user": 100.0},
                {"tpot": 200.0, "ttft": 500.0, "tokens/s/gpu": 800.0, "tokens/s/user": 400.0},
            ]
        )
        result = pick_default(
            pareto_df=df,
            total_gpus=8,
            serving_mode="agg",
            target_tpot=15.0,
            top_n=5,
        )
        frontier = result["pareto_frontier_df"]
        # Full frontier preserved — includes configs that violate TPOT
        assert len(frontier) == 2, "Non-strict mode must preserve the full Pareto frontier"

    def test_default_behavior_high_tpot_wins(self):
        """Without strict_sla, high-TPOT config can be in frontier if it has best throughput."""
        df = _make_pareto_df(
            [
                {"tpot": 10.0, "ttft": 500.0, "tokens/s/gpu": 800.0, "tokens/s/user": 300.0},  # meets TPOT
                {"tpot": 50.0, "ttft": 500.0, "tokens/s/gpu": 1500.0, "tokens/s/user": 100.0},  # violates TPOT
            ]
        )
        result = pick_default(
            pareto_df=df,
            total_gpus=8,
            serving_mode="agg",
            target_tpot=15.0,
            top_n=5,
        )
        frontier = result["pareto_frontier_df"]
        # Both should be in the frontier (non-strict doesn't filter)
        assert len(frontier) == 2, "Non-strict mode preserves TPOT-violating configs in frontier"
        # But best_config_topn is still TPOT-filtered by get_best_configs_under_tpot_constraint
        best = result["best_config_df"]
        assert (best["tpot"] <= 15.0).all(), "Best config is always TPOT-filtered"

    def test_strict_excludes_tpot_violators_from_frontier(self):
        """With strict_sla, TPOT violators are excluded from the Pareto frontier."""
        df = _make_pareto_df(
            [
                {"tpot": 10.0, "ttft": 500.0, "tokens/s/gpu": 800.0},  # meets TPOT
                {"tpot": 50.0, "ttft": 500.0, "tokens/s/gpu": 1500.0},  # violates TPOT
            ]
        )
        result = pick_default(
            pareto_df=df,
            total_gpus=8,
            serving_mode="agg",
            target_tpot=15.0,
            top_n=5,
            strict_sla=True,
        )
        frontier = result["pareto_frontier_df"]
        assert len(frontier) == 1, "Strict mode removes TPOT violators from frontier"
        assert (frontier["tpot"] <= 15.0).all()
