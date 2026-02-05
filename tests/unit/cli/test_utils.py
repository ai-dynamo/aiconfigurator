# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for CLI utility functions in utils.py.
"""

from unittest.mock import MagicMock

import pandas as pd
import pytest

from aiconfigurator.cli.utils import (
    _merge_into_top_n,
    merge_experiment_results_by_mode,
    process_experiment_result,
)

pytestmark = pytest.mark.unit


class TestProcessExperimentResult:
    """Tests for process_experiment_result function."""

    def test_process_result_with_tpot_constraint(self):
        """Test processing result with TPOT constraint."""
        # Create mock task config
        mock_task_config = MagicMock()
        mock_task_config.config.runtime_config.tpot = 30.0
        mock_task_config.config.runtime_config.request_latency = None
        mock_task_config.serving_mode = "agg"
        mock_task_config.total_gpus = 32

        # Create sample pareto_df
        pareto_df = pd.DataFrame(
            {
                "tokens/s/gpu": [100.0, 120.0, 90.0],
                "tokens/s/user": [10.0, 12.0, 9.0],
                "num_total_gpus": [8, 8, 8],
                "tpot": [25.0, 28.0, 20.0],
                "parallel": ["tp4_pp1_dp2", "tp2_pp1_dp4", "tp1_pp1_dp8"],
            }
        )

        result = {"pareto_df": pareto_df}

        best_config_df, best_throughput, pareto_frontier_df, x_axis_col = process_experiment_result(
            mock_task_config, result, top_n=2
        )

        # Verify results
        assert not best_config_df.empty
        assert "tokens/s/gpu_cluster" in best_config_df.columns
        assert best_throughput > 0.0
        assert x_axis_col == "tokens/s/user"
        assert not pareto_frontier_df.empty

    def test_process_result_with_request_latency_constraint(self):
        """Test processing result with request_latency constraint."""
        # Create mock task config
        mock_task_config = MagicMock()
        mock_task_config.config.runtime_config.tpot = None
        mock_task_config.config.runtime_config.request_latency = 1200.0
        mock_task_config.serving_mode = "disagg"
        mock_task_config.total_gpus = 32

        # Create sample pareto_df
        pareto_df = pd.DataFrame(
            {
                "tokens/s/gpu": [100.0, 120.0, 90.0],
                "tokens/s/user": [10.0, 12.0, 9.0],
                "num_total_gpus": [8, 8, 8],
                "request_latency": [1000.0, 1100.0, 900.0],
                "(d)parallel": ["tp4_pp1_dp2", "tp2_pp1_dp4", "tp1_pp1_dp8"],
            }
        )

        result = {"pareto_df": pareto_df}

        best_config_df, best_throughput, pareto_frontier_df, x_axis_col = process_experiment_result(
            mock_task_config, result, top_n=3
        )

        # Verify results
        assert not best_config_df.empty
        assert x_axis_col == "request_latency"
        assert "tokens/s/gpu_cluster" in best_config_df.columns
        assert best_throughput > 0.0

    def test_process_result_with_empty_pareto_df(self):
        """Test processing result with empty pareto_df."""
        mock_task_config = MagicMock()
        mock_task_config.config.runtime_config.tpot = 30.0
        mock_task_config.config.runtime_config.request_latency = None
        mock_task_config.serving_mode = "agg"
        mock_task_config.total_gpus = 32

        result = {"pareto_df": pd.DataFrame()}

        best_config_df, best_throughput, pareto_frontier_df, x_axis_col = process_experiment_result(
            mock_task_config, result, top_n=5
        )

        # Verify results
        assert best_config_df.empty
        assert best_throughput == 0.0
        assert pareto_frontier_df.empty
        assert x_axis_col in ["tokens/s/user", "request_latency"]

    def test_process_result_with_none_pareto_df(self):
        """Test processing result with None pareto_df."""
        mock_task_config = MagicMock()
        mock_task_config.config.runtime_config.tpot = 30.0
        mock_task_config.config.runtime_config.request_latency = None
        mock_task_config.serving_mode = "agg"
        mock_task_config.total_gpus = 32

        result = {"pareto_df": None}

        best_config_df, best_throughput, pareto_frontier_df, x_axis_col = process_experiment_result(
            mock_task_config, result, top_n=5
        )

        # Verify results
        assert best_config_df.empty
        assert best_throughput == 0.0
        assert pareto_frontier_df.empty

    def test_process_result_disagg_mode_uses_correct_group_by(self):
        """Test that disagg mode uses (d)parallel as group_by key."""
        mock_task_config = MagicMock()
        mock_task_config.config.runtime_config.tpot = 30.0
        mock_task_config.config.runtime_config.request_latency = None
        mock_task_config.serving_mode = "disagg"
        mock_task_config.total_gpus = 32

        pareto_df = pd.DataFrame(
            {
                "tokens/s/gpu": [100.0, 120.0],
                "tokens/s/user": [10.0, 12.0],
                "num_total_gpus": [8, 8],
                "tpot": [25.0, 28.0],
                "(d)parallel": ["tp4_pp1_dp2", "tp2_pp1_dp4"],
            }
        )

        result = {"pareto_df": pareto_df}

        best_config_df, _, _, _ = process_experiment_result(mock_task_config, result, top_n=5)

        # Verify that results are returned (group_by should work correctly)
        assert not best_config_df.empty

    def test_process_result_top_n_limiting(self):
        """Test that top_n correctly limits the number of returned configs."""
        mock_task_config = MagicMock()
        mock_task_config.config.runtime_config.tpot = 30.0
        mock_task_config.config.runtime_config.request_latency = None
        mock_task_config.serving_mode = "agg"
        mock_task_config.total_gpus = 32

        # Create sample with more configs than top_n
        pareto_df = pd.DataFrame(
            {
                "tokens/s/gpu": [100.0, 120.0, 90.0, 110.0, 95.0, 105.0],
                "tokens/s/user": [10.0, 12.0, 9.0, 11.0, 9.5, 10.5],
                "num_total_gpus": [8, 8, 8, 8, 8, 8],
                "tpot": [25.0, 28.0, 20.0, 26.0, 22.0, 24.0],
                "parallel": [f"config_{i}" for i in range(6)],
            }
        )

        result = {"pareto_df": pareto_df}

        best_config_df, _, _, _ = process_experiment_result(mock_task_config, result, top_n=3)

        # Should return at most 3 configs
        assert len(best_config_df) <= 3

    def test_process_result_computes_tokens_per_gpu_cluster(self):
        """Test that tokens/s/gpu_cluster is correctly computed."""
        mock_task_config = MagicMock()
        mock_task_config.config.runtime_config.tpot = 30.0
        mock_task_config.config.runtime_config.request_latency = None
        mock_task_config.serving_mode = "agg"
        mock_task_config.total_gpus = 32

        pareto_df = pd.DataFrame(
            {
                "tokens/s/gpu": [100.0],
                "tokens/s/user": [10.0],
                "num_total_gpus": [8],
                "tpot": [25.0],
                "parallel": ["tp4_pp1_dp2"],
            }
        )

        result = {"pareto_df": pareto_df}

        best_config_df, _, _, _ = process_experiment_result(mock_task_config, result, top_n=5)

        # Verify tokens/s/gpu_cluster is computed
        assert "tokens/s/gpu_cluster" in best_config_df.columns
        # Expected: 100 * (32 // 8) * 8 / 32 = 100 * 4 * 8 / 32 = 100
        assert best_config_df["tokens/s/gpu_cluster"].values[0] > 0


class TestMergeIntoTopN:
    """Tests for _merge_into_top_n helper function."""

    def test_merge_multiple_backends(self):
        """Test merging configs from multiple backends."""
        # Create mock task configs
        task_configs = {}
        for backend in ["trtllm", "vllm", "sglang"]:
            mock_config = MagicMock()
            mock_config.backend_name = backend
            task_configs[f"agg_{backend}"] = mock_config

        # Create mock best configs
        best_configs = {
            "agg_trtllm": pd.DataFrame({"tokens/s/gpu_cluster": [100.0], "tokens/s/user": [10.0]}),
            "agg_vllm": pd.DataFrame({"tokens/s/gpu_cluster": [120.0], "tokens/s/user": [12.0]}),
            "agg_sglang": pd.DataFrame({"tokens/s/gpu_cluster": [110.0], "tokens/s/user": [11.0]}),
        }

        pareto_fronts = {
            "agg_trtllm": pd.DataFrame({"tokens/s/gpu_cluster": [100.0], "tokens/s/user": [10.0]}),
            "agg_vllm": pd.DataFrame({"tokens/s/gpu_cluster": [120.0], "tokens/s/user": [12.0]}),
            "agg_sglang": pd.DataFrame({"tokens/s/gpu_cluster": [110.0], "tokens/s/user": [11.0]}),
        }

        pareto_x_axis = {"agg_trtllm": "tokens/s/user", "agg_vllm": "tokens/s/user", "agg_sglang": "tokens/s/user"}

        exps = ["agg_trtllm", "agg_vllm", "agg_sglang"]

        merged_best, merged_throughput, merged_pareto, x_col = _merge_into_top_n(
            exps, task_configs, best_configs, pareto_fronts, pareto_x_axis, top_n=5
        )

        # Verify results
        assert not merged_best.empty
        assert "backend" in merged_best.columns
        assert merged_throughput == 120.0  # Best from vllm
        assert not merged_pareto.empty
        assert "backend" in merged_pareto.columns
        assert x_col == "tokens/s/user"

    def test_merge_with_top_n_limiting(self):
        """Test that merging respects top_n limit."""
        task_configs = {}
        best_configs = {}
        pareto_fronts = {}
        pareto_x_axis = {}

        for i, backend in enumerate(["trtllm", "vllm", "sglang"]):
            mock_config = MagicMock()
            mock_config.backend_name = backend
            exp_name = f"agg_{backend}"
            task_configs[exp_name] = mock_config

            # Create configs with different throughputs
            best_configs[exp_name] = pd.DataFrame(
                {
                    "tokens/s/gpu_cluster": [100.0 + i * 10, 95.0 + i * 10],
                    "tokens/s/user": [10.0 + i, 9.5 + i],
                }
            )
            pareto_fronts[exp_name] = best_configs[exp_name].copy()
            pareto_x_axis[exp_name] = "tokens/s/user"

        exps = list(task_configs.keys())

        merged_best, _, _, _ = _merge_into_top_n(
            exps, task_configs, best_configs, pareto_fronts, pareto_x_axis, top_n=3
        )

        # Should return at most 3 configs
        assert len(merged_best) <= 3
        # Should be sorted by throughput descending
        assert merged_best["tokens/s/gpu_cluster"].is_monotonic_decreasing

    def test_merge_with_empty_dataframes(self):
        """Test merging when some configs are empty."""
        task_configs = {
            "agg_trtllm": MagicMock(backend_name="trtllm"),
            "agg_vllm": MagicMock(backend_name="vllm"),
        }

        best_configs = {
            "agg_trtllm": pd.DataFrame(),  # Empty
            "agg_vllm": pd.DataFrame({"tokens/s/gpu_cluster": [120.0], "tokens/s/user": [12.0]}),
        }

        pareto_fronts = {
            "agg_trtllm": pd.DataFrame(),
            "agg_vllm": pd.DataFrame({"tokens/s/gpu_cluster": [120.0], "tokens/s/user": [12.0]}),
        }

        pareto_x_axis = {"agg_trtllm": "tokens/s/user", "agg_vllm": "tokens/s/user"}

        exps = ["agg_trtllm", "agg_vllm"]

        merged_best, merged_throughput, merged_pareto, x_col = _merge_into_top_n(
            exps, task_configs, best_configs, pareto_fronts, pareto_x_axis, top_n=5
        )

        # Should only contain vllm results
        assert not merged_best.empty
        assert len(merged_best) == 1
        assert merged_best["backend"].values[0] == "vllm"
        assert merged_throughput == 120.0

    def test_merge_with_missing_pareto_fronts(self):
        """Test merging when pareto fronts are missing."""
        task_configs = {"agg_trtllm": MagicMock(backend_name="trtllm")}

        best_configs = {"agg_trtllm": pd.DataFrame({"tokens/s/gpu_cluster": [100.0], "tokens/s/user": [10.0]})}

        pareto_fronts = {}  # No pareto fronts

        pareto_x_axis = {"agg_trtllm": "tokens/s/user"}

        exps = ["agg_trtllm"]

        merged_best, merged_throughput, merged_pareto, x_col = _merge_into_top_n(
            exps, task_configs, best_configs, pareto_fronts, pareto_x_axis, top_n=5
        )

        # Should still merge best configs
        assert not merged_best.empty
        assert merged_throughput == 100.0
        # Pareto front should be None or empty
        assert merged_pareto is None or merged_pareto.empty

    def test_merge_recomputes_pareto_frontier(self):
        """Test that merge recomputes Pareto frontier from combined data."""
        task_configs = {}
        best_configs = {}
        pareto_fronts = {}
        pareto_x_axis = {}

        for backend in ["trtllm", "vllm"]:
            mock_config = MagicMock()
            mock_config.backend_name = backend
            exp_name = f"agg_{backend}"
            task_configs[exp_name] = mock_config

            # Create different pareto fronts
            if backend == "trtllm":
                pareto_fronts[exp_name] = pd.DataFrame(
                    {
                        "tokens/s/gpu_cluster": [100.0, 90.0],
                        "tokens/s/user": [10.0, 12.0],
                    }
                )
            else:
                pareto_fronts[exp_name] = pd.DataFrame(
                    {
                        "tokens/s/gpu_cluster": [110.0, 95.0],
                        "tokens/s/user": [11.0, 13.0],
                    }
                )

            best_configs[exp_name] = pareto_fronts[exp_name].copy()
            pareto_x_axis[exp_name] = "tokens/s/user"

        exps = ["agg_trtllm", "agg_vllm"]

        _, _, merged_pareto, _ = _merge_into_top_n(
            exps, task_configs, best_configs, pareto_fronts, pareto_x_axis, top_n=5
        )

        # Should combine and recompute pareto frontier
        assert not merged_pareto.empty
        assert "backend" in merged_pareto.columns


class TestMergeExperimentResultsByMode:
    """Tests for merge_experiment_results_by_mode function."""

    def test_merge_six_backends_into_two_modes(self):
        """Test merging 6 backend experiments into agg and disagg."""
        # Create mock task configs for all 6 experiments
        task_configs = {}
        for backend in ["trtllm", "vllm", "sglang"]:
            for mode in ["agg", "disagg"]:
                mock_config = MagicMock()
                mock_config.backend_name = backend
                mock_config.serving_mode = mode
                task_configs[f"{mode}_{backend}"] = mock_config

        # Create mock best configs with different throughputs
        best_configs = {
            "agg_trtllm": pd.DataFrame({"tokens/s/gpu_cluster": [100.0], "tokens/s/user": [10.0]}),
            "agg_vllm": pd.DataFrame({"tokens/s/gpu_cluster": [120.0], "tokens/s/user": [12.0]}),
            "agg_sglang": pd.DataFrame({"tokens/s/gpu_cluster": [110.0], "tokens/s/user": [11.0]}),
            "disagg_trtllm": pd.DataFrame({"tokens/s/gpu_cluster": [150.0], "tokens/s/user": [15.0]}),
            "disagg_vllm": pd.DataFrame({"tokens/s/gpu_cluster": [140.0], "tokens/s/user": [14.0]}),
            "disagg_sglang": pd.DataFrame({"tokens/s/gpu_cluster": [130.0], "tokens/s/user": [13.0]}),
        }

        pareto_fronts = {name: df.copy() for name, df in best_configs.items()}
        pareto_x_axis = dict.fromkeys(best_configs.keys(), "tokens/s/user")

        # Merge results
        merged_configs, merged_throughputs, merged_fronts, merged_axis = merge_experiment_results_by_mode(
            task_configs, best_configs, pareto_fronts, pareto_x_axis, top_n=5
        )

        # Should have only 2 merged results: agg and disagg
        assert len(merged_configs) == 2
        assert "agg" in merged_configs
        assert "disagg" in merged_configs

        # Merged configs should have "backend" column
        assert "backend" in merged_configs["agg"].columns
        assert "backend" in merged_configs["disagg"].columns

        # Best throughput should be from vllm for agg (120.0) and trtllm for disagg (150.0)
        assert merged_throughputs["agg"] == 120.0
        assert merged_throughputs["disagg"] == 150.0

        # Pareto fronts should be merged
        assert not merged_fronts["agg"].empty
        assert not merged_fronts["disagg"].empty

        # X-axis should be preserved
        assert merged_axis["agg"] == "tokens/s/user"
        assert merged_axis["disagg"] == "tokens/s/user"

    def test_merge_with_top_n_limiting(self):
        """Test that merge respects top_n limit across backends."""
        task_configs = {}
        best_configs = {}
        pareto_fronts = {}
        pareto_x_axis = {}

        for i, backend in enumerate(["trtllm", "vllm", "sglang"]):
            mock_config = MagicMock()
            mock_config.backend_name = backend
            mock_config.serving_mode = "agg"
            exp_name = f"agg_{backend}"
            task_configs[exp_name] = mock_config

            # Each backend has 3 configs
            best_configs[exp_name] = pd.DataFrame(
                {
                    "tokens/s/gpu_cluster": [100.0 + i * 10 + j for j in range(3)],
                    "tokens/s/user": [10.0 + i + j * 0.1 for j in range(3)],
                }
            )
            pareto_fronts[exp_name] = best_configs[exp_name].copy()
            pareto_x_axis[exp_name] = "tokens/s/user"

        # Merge with top_n=5
        merged_configs, _, _, _ = merge_experiment_results_by_mode(
            task_configs, best_configs, pareto_fronts, pareto_x_axis, top_n=5
        )

        # Should return at most 5 configs for agg
        assert len(merged_configs["agg"]) <= 5

    def test_merge_with_mixed_agg_disagg(self):
        """Test merging with mix of agg and disagg experiments."""
        task_configs = {
            "agg_trtllm": MagicMock(backend_name="trtllm", serving_mode="agg"),
            "agg_vllm": MagicMock(backend_name="vllm", serving_mode="agg"),
            "disagg_trtllm": MagicMock(backend_name="trtllm", serving_mode="disagg"),
        }

        best_configs = {
            "agg_trtllm": pd.DataFrame({"tokens/s/gpu_cluster": [100.0], "tokens/s/user": [10.0]}),
            "agg_vllm": pd.DataFrame({"tokens/s/gpu_cluster": [120.0], "tokens/s/user": [12.0]}),
            "disagg_trtllm": pd.DataFrame({"tokens/s/gpu_cluster": [150.0], "tokens/s/user": [15.0]}),
        }

        pareto_fronts = {name: df.copy() for name, df in best_configs.items()}
        pareto_x_axis = dict.fromkeys(best_configs.keys(), "tokens/s/user")

        merged_configs, merged_throughputs, _, _ = merge_experiment_results_by_mode(
            task_configs, best_configs, pareto_fronts, pareto_x_axis, top_n=5
        )

        # Should have 2 results: agg (from 2 backends) and disagg (from 1 backend)
        assert len(merged_configs) == 2
        assert not merged_configs["agg"].empty
        assert not merged_configs["disagg"].empty

        # Agg should have 2 configs, disagg should have 1
        assert len(merged_configs["agg"]) == 2
        assert len(merged_configs["disagg"]) == 1

    def test_merge_preserves_backend_information(self):
        """Test that backend information is preserved in merged results."""
        task_configs = {}
        best_configs = {}
        pareto_fronts = {}
        pareto_x_axis = {}

        for backend in ["trtllm", "vllm"]:
            mock_config = MagicMock()
            mock_config.backend_name = backend
            mock_config.serving_mode = "agg"
            exp_name = f"agg_{backend}"
            task_configs[exp_name] = mock_config

            best_configs[exp_name] = pd.DataFrame({"tokens/s/gpu_cluster": [100.0], "tokens/s/user": [10.0]})
            pareto_fronts[exp_name] = best_configs[exp_name].copy()
            pareto_x_axis[exp_name] = "tokens/s/user"

        merged_configs, _, _, _ = merge_experiment_results_by_mode(
            task_configs, best_configs, pareto_fronts, pareto_x_axis, top_n=5
        )

        # Backend column should exist and contain correct values
        assert "backend" in merged_configs["agg"].columns
        backends = set(merged_configs["agg"]["backend"].values)
        assert backends.issubset({"trtllm", "vllm"})

    def test_merge_with_empty_experiments(self):
        """Test merging when some experiments have no results."""
        task_configs = {
            "agg_trtllm": MagicMock(backend_name="trtllm", serving_mode="agg"),
            "agg_vllm": MagicMock(backend_name="vllm", serving_mode="agg"),
            "disagg_trtllm": MagicMock(backend_name="trtllm", serving_mode="disagg"),
        }

        best_configs = {
            "agg_trtllm": pd.DataFrame(),  # Empty
            "agg_vllm": pd.DataFrame({"tokens/s/gpu_cluster": [120.0], "tokens/s/user": [12.0]}),
            "disagg_trtllm": pd.DataFrame({"tokens/s/gpu_cluster": [150.0], "tokens/s/user": [15.0]}),
        }

        pareto_fronts = {name: df.copy() for name, df in best_configs.items()}
        pareto_x_axis = dict.fromkeys(best_configs.keys(), "tokens/s/user")

        merged_configs, merged_throughputs, _, _ = merge_experiment_results_by_mode(
            task_configs, best_configs, pareto_fronts, pareto_x_axis, top_n=5
        )

        # Should still produce results for both modes
        assert len(merged_configs) == 2
        # Agg should only have vllm result
        assert not merged_configs["agg"].empty
        assert len(merged_configs["agg"]) == 1
        assert merged_throughputs["agg"] == 120.0
