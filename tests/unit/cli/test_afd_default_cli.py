# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for AFD wiring in the default CLI mode (v2 Task path)."""

from types import SimpleNamespace

import pandas as pd
import pytest

from aiconfigurator.cli.utils import merge_experiment_results_by_mode
from aiconfigurator.sdk import common
from aiconfigurator.sdk import task_v2 as task_v2_module
from aiconfigurator.sdk.task_v2 import Task

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _stub_latest_db_version(monkeypatch):
    """Avoid touching the on-disk perf database when resolving versions."""
    monkeypatch.setattr(task_v2_module, "get_latest_database_version", lambda **_: "test-version")


class TestServingModeArgument:
    def test_serving_mode_choices_and_default(self, cli_parser):
        subparser_action = next(action for action in cli_parser._actions if action.dest == "mode")
        default_parser = subparser_action.choices["default"]
        serving_mode_action = next(action for action in default_parser._actions if action.dest == "serving_mode")
        assert set(serving_mode_action.choices) == {"auto", "all", "agg", "disagg", "afd"}
        assert serving_mode_action.default == "auto"


class TestV2AfdTask:
    def test_prefix_defaults_to_zero(self):
        task = Task(
            serving_mode="afd",
            model_path="Qwen/Qwen3-32B",
            system_name="h200_sxm",
            backend_name="trtllm",
            total_gpus=32,
        )
        assert task.prefix == 0

    def test_prefix_is_preserved(self):
        task = Task(
            serving_mode="afd",
            model_path="Qwen/Qwen3-32B",
            system_name="h200_sxm",
            backend_name="trtllm",
            total_gpus=32,
            prefix=128,
        )
        assert task.prefix == 128

    def test_serving_mode_is_afd(self):
        task = Task(
            serving_mode="afd",
            model_path="Qwen/Qwen3-32B",
            system_name="h200_sxm",
            backend_name="trtllm",
            total_gpus=32,
        )
        assert task.serving_mode == "afd"
        assert task.primary_model_path == "Qwen/Qwen3-32B"
        assert task.primary_system_name == "h200_sxm"

    def test_afd_search_space_resolved(self):
        task = Task(
            serving_mode="afd",
            model_path="Qwen/Qwen3-32B",
            system_name="h200_sxm",
            backend_name="trtllm",
            total_gpus=32,
        )
        assert task._afd_parallel_config_list
        assert task._afd_gpus_per_node == 8
        for n_a, n_f, tp_a, _f_ep, _mb, _pipe in task._afd_parallel_config_list:
            assert (n_a + n_f) * 8 <= 32
            assert 8 % tp_a == 0

    def test_insufficient_nodes_raises(self):
        with pytest.raises(ValueError, match="at least 2 nodes"):
            Task(
                serving_mode="afd",
                model_path="Qwen/Qwen3-32B",
                system_name="h200_sxm",
                backend_name="trtllm",
                total_gpus=8,
            )

    def test_to_yaml_contains_afd_fields(self):
        task = Task(
            serving_mode="afd",
            model_path="Qwen/Qwen3-32B",
            system_name="h200_sxm",
            backend_name="trtllm",
            total_gpus=32,
        )
        text = task.to_yaml()
        assert "serving_mode: afd" in text

    def test_quant_modes_resolved_for_afd(self):
        task = Task(
            serving_mode="afd",
            model_path="Qwen/Qwen3-32B",
            system_name="h200_sxm",
            backend_name="trtllm",
            total_gpus=32,
            gemm_quant_mode=common.GEMMQuantMode.fp8,
            moe_quant_mode=common.MoEQuantMode.fp8,
            kvcache_quant_mode=common.KVCacheQuantMode.fp8,
        )
        assert task.gemm_quant_mode == common.GEMMQuantMode.fp8
        assert task.moe_quant_mode == common.MoEQuantMode.fp8
        assert task.kvcache_quant_mode == common.KVCacheQuantMode.fp8


class TestMergeByModeIncludesAfd:
    def test_afd_bucket_is_merged_separately(self):
        afd_df = pd.DataFrame(
            [
                {
                    "parallel": "a1n-tp4+f1n-ep1",
                    "tokens/s/gpu_cluster": 100.0,
                    "tokens/s/user": 50.0,
                }
            ]
        )
        agg_df = pd.DataFrame(
            [
                {
                    "parallel": "tp4pp1",
                    "tokens/s/gpu_cluster": 80.0,
                    "tokens/s/user": 40.0,
                }
            ]
        )
        task_configs = {
            "agg_trtllm": SimpleNamespace(
                serving_mode="agg",
                backend_name="trtllm",
                primary_backend_name="trtllm",
            ),
            "afd_trtllm": SimpleNamespace(
                serving_mode="afd",
                backend_name="trtllm",
                primary_backend_name="trtllm",
            ),
        }
        best_configs = {"agg_trtllm": agg_df, "afd_trtllm": afd_df}
        pareto_fronts = {"agg_trtllm": agg_df, "afd_trtllm": afd_df}
        pareto_x_axis = {"agg_trtllm": "tokens/s/user", "afd_trtllm": "tokens/s/user"}

        merged_best, merged_tputs, merged_fronts, _ = merge_experiment_results_by_mode(
            task_configs, best_configs, pareto_fronts, pareto_x_axis, top_n=5
        )

        assert "afd" in merged_best and not merged_best["afd"].empty
        assert merged_tputs["afd"] == pytest.approx(100.0)
        assert "disagg" not in merged_best  # no disagg experiments -> no empty bucket
        assert not merged_fronts["afd"].empty
        # schemas are not cross-contaminated
        assert "(a)nodes" not in merged_best["agg"].columns

    def test_columns_afd_contains_new_fields(self):
        for col in (
            "parallel",
            "request_rate",
            "(p)workers",
            "(p)tp",
            "(p)pp",
            "(p)dp",
            "(p)moe_tp",
            "(p)ep",
            "(p)bs",
            "(p)num_gpus",
            "(p)system",
            "(p)backend",
            "(p)version",
            "(p)impl",
            "(d)impl",
        ):
            assert col in common.ColumnsAFD


class TestExpLoaderAcceptsAfd:
    """exp-mode YAML loader must build AFD task configs via v2 Task."""

    def test_build_experiment_task_configs_afd(self, monkeypatch):
        from aiconfigurator.cli.main import build_experiment_tasks

        config = {
            "afd_exp": {
                "serving_mode": "afd",
                "model_path": "Qwen/Qwen3-32B",
                "system_name": "h200_sxm",
                "backend_name": "trtllm",
                "total_gpus": 32,
                "database_mode": "HYBRID",
                "isl": 4000,
                "osl": 1000,
            }
        }
        task_configs = build_experiment_tasks(config=config)
        assert "afd_exp" in task_configs
        assert task_configs["afd_exp"].serving_mode == "afd"
        assert task_configs["afd_exp"]._afd_parallel_config_list

    def test_build_experiment_task_configs_afd_default_batch(self, monkeypatch):
        from aiconfigurator.cli.main import build_experiment_tasks

        config = {
            "afd_default": {
                "serving_mode": "afd",
                "model_path": "Qwen/Qwen3-32B",
                "system_name": "h200_sxm",
                "backend_name": "trtllm",
                "total_gpus": 32,
                "database_mode": "HYBRID",
            }
        }

        task_configs = build_experiment_tasks(config=config)
        task_config = task_configs["afd_default"]

        assert task_config._afd_parallel_config_list
        # afd_total_batch_size defaults to None (KV-capacity derived)
        assert task_config.afd_total_batch_size is None
