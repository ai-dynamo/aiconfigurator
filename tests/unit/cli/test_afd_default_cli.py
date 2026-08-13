# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for AFD wiring in the default CLI mode (v2 Task path)."""

import logging
from types import SimpleNamespace

import pandas as pd
import pytest

from aiconfigurator.cli.report_and_save import (
    _auto_result_tasks,
    _plot_worker_setup_table,
    _task_for_result_row,
    save_results,
)
from aiconfigurator.cli.utils import merge_experiment_results_by_mode
from aiconfigurator.sdk import common
from aiconfigurator.sdk import task_v2 as task_v2_module
from aiconfigurator.sdk.errors import NoFeasibleConfigError
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

    def test_afd_max_a_batch_size_default_and_override(self, cli_parser):
        subparser_action = next(action for action in cli_parser._actions if action.dest == "mode")
        default_parser = subparser_action.choices["default"]
        max_batch_action = next(action for action in default_parser._actions if action.dest == "afd_max_a_batch_size")
        assert max_batch_action.default == 1024

        args = cli_parser.parse_args(
            [
                "default",
                "--model-path",
                "Qwen/Qwen3-32B",
                "--total-gpus",
                "16",
                "--system",
                "h200_sxm",
                "--afd-max-a-batch-size",
                "1536",
            ]
        )
        assert args.afd_max_a_batch_size == 1536

    def test_afd_max_a_batch_size_rejects_values_below_search_floor(self, cli_parser):
        with pytest.raises(SystemExit):
            cli_parser.parse_args(["default", "--afd-max-a-batch-size", "31"])

    def test_afd_candidate_limit_defaults_and_overrides(self, cli_parser):
        subparser_action = next(action for action in cli_parser._actions if action.dest == "mode")
        default_parser = subparser_action.choices["default"]
        max_candidates_action = next(
            action for action in default_parser._actions if action.dest == "afd_max_candidates"
        )
        overflow_action = next(action for action in default_parser._actions if action.dest == "afd_candidate_overflow")

        assert max_candidates_action.default == 10_000
        assert overflow_action.default == "error"

        args = cli_parser.parse_args(
            [
                "default",
                "--model-path",
                "Qwen/Qwen3-32B",
                "--total-gpus",
                "16",
                "--system",
                "h200_sxm",
                "--afd-max-candidates",
                "500",
                "--afd-candidate-overflow",
                "truncate",
            ]
        )
        assert args.afd_max_candidates == 500
        assert args.afd_candidate_overflow == "truncate"

    def test_afd_candidate_limit_rejects_non_positive_values(self, cli_parser):
        with pytest.raises(SystemExit):
            cli_parser.parse_args(["default", "--afd-max-candidates", "0"])


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

    def test_afd_max_a_batch_size_is_forwarded_to_sweep(self):
        task = Task(
            serving_mode="afd",
            model_path="Qwen/Qwen3-32B",
            system_name="h200_sxm",
            backend_name="trtllm",
            total_gpus=32,
            afd_combined_with_pd=False,
            afd_max_a_batch_size=1536,
        )

        assert task.sweep_afd_kwargs(database=object())["max_a_batch_size"] == 1536

    def test_afd_max_a_batch_size_validation(self):
        task = Task(
            serving_mode="afd",
            model_path="Qwen/Qwen3-32B",
            system_name="h200_sxm",
            backend_name="trtllm",
            total_gpus=32,
            afd_max_a_batch_size=31,
        )

        with pytest.raises(ValueError, match="afd_max_a_batch_size must be an integer >= 32"):
            task.validate()

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

    def test_afd_total_gpus_overrides_total_gpus_across_search_and_sweep(self):
        task = Task(
            serving_mode="afd",
            model_path="Qwen/Qwen3-32B",
            system_name="h200_sxm",
            backend_name="trtllm",
            total_gpus=16,
            afd_total_gpus=32,
            afd_combined_with_pd=False,
        )

        assert task.effective_total_gpus == 32
        assert task.sweep_afd_kwargs(database=object())["total_gpus"] == 32
        assert any((n_a + n_f) * 8 > 16 for n_a, n_f, *_ in task._afd_parallel_config_list)

    def test_afd_total_gpus_is_sufficient_without_total_gpus(self):
        task = Task(
            serving_mode="afd",
            model_path="Qwen/Qwen3-32B",
            system_name="h200_sxm",
            backend_name="trtllm",
            afd_total_gpus=16,
            afd_combined_with_pd=False,
        )

        task.validate()
        assert task.effective_total_gpus == 16

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

    def test_pinned_topology_constructs_afd_config_without_derived_field_argument(self, monkeypatch):
        import aiconfigurator.sdk.backends.factory as backend_factory
        import aiconfigurator.sdk.inference_session as inference_session

        captured = {}

        class _Summary:
            @staticmethod
            def get_summary_df():
                return pd.DataFrame([{"request_rate": 1.0}])

        class _Session:
            def __init__(self, **kwargs):
                captured.update(kwargs)

            @staticmethod
            def run_afd(*args, **kwargs):
                return _Summary()

        task = Task(
            serving_mode="afd",
            model_path="Qwen/Qwen3-32B",
            system_name="h200_sxm",
            backend_name="trtllm",
            total_gpus=16,
            afd_n_a_nodes=1,
            afd_n_f_nodes=1,
            afd_tp_a=4,
        )
        monkeypatch.setattr(backend_factory, "get_backend", lambda _: object())
        monkeypatch.setattr(inference_session, "AFDInferenceSession", _Session)
        monkeypatch.setattr(task, "build_model_config", lambda **_: SimpleNamespace())

        result = task._run_afd_single_point(database=object())

        assert task._afd_topology_pinned is True
        assert task._afd_parallel_config_list == []
        assert captured["afd_config"].n_a_workers == 2
        assert result["request_rate"].tolist() == [1.0]

    def test_pinned_topology_uses_afd_total_gpus_budget(self):
        task = Task(
            serving_mode="afd",
            model_path="Qwen/Qwen3-32B",
            system_name="h200_sxm",
            backend_name="trtllm",
            total_gpus=16,
            afd_total_gpus=32,
            afd_n_a_nodes=2,
            afd_n_f_nodes=2,
            afd_tp_a=4,
        )

        assert task._afd_topology_pinned is True
        assert task.effective_total_gpus == 32

    def test_pinned_moe_topology_rejects_ep_that_does_not_divide_experts(self):
        with pytest.raises(ValueError, match=r"f_moe_ep_size=24.*256 experts"):
            Task(
                serving_mode="afd",
                model_path="deepseek-ai/DeepSeek-V3",
                system_name="h200_sxm",
                backend_name="trtllm",
                total_gpus=32,
                afd_n_a_nodes=1,
                afd_n_f_nodes=3,
                afd_tp_a=4,
            )

    def test_pinned_moe_topology_rejects_ep_larger_than_expert_count(self):
        with pytest.raises(ValueError, match=r"f_moe_ep_size=264.*256 experts"):
            Task(
                serving_mode="afd",
                model_path="deepseek-ai/DeepSeek-V3",
                system_name="h200_sxm",
                backend_name="trtllm",
                total_gpus=272,
                afd_n_a_nodes=1,
                afd_n_f_nodes=33,
                afd_tp_a=4,
            )

    def test_pinned_moe_topology_accepts_ep_dividing_experts_and_f_ranks(self):
        task = Task(
            serving_mode="afd",
            model_path="deepseek-ai/DeepSeek-V3",
            system_name="h200_sxm",
            backend_name="trtllm",
            total_gpus=24,
            afd_n_a_nodes=1,
            afd_n_f_nodes=2,
            afd_tp_a=4,
        )

        assert task._afd_topology_pinned is True

    def test_partial_pinned_topology_is_rejected(self):
        with pytest.raises(ValueError, match="requires afd_n_a_nodes, afd_n_f_nodes, and afd_tp_a together"):
            Task(
                serving_mode="afd",
                model_path="Qwen/Qwen3-32B",
                system_name="h200_sxm",
                backend_name="trtllm",
                total_gpus=16,
                afd_n_a_nodes=1,
            )

    def test_invalid_pinned_tp_is_rejected(self):
        with pytest.raises(ValueError, match="positive divisor"):
            Task(
                serving_mode="afd",
                model_path="Qwen/Qwen3-32B",
                system_name="h200_sxm",
                backend_name="trtllm",
                total_gpus=16,
                afd_n_a_nodes=1,
                afd_n_f_nodes=1,
                afd_tp_a=3,
            )

    def test_empty_unpinned_search_is_not_misclassified_as_pinned(self):
        with pytest.raises(NoFeasibleConfigError, match="no valid topology candidates"):
            Task(
                serving_mode="afd",
                model_path="Qwen/Qwen3-32B",
                system_name="h200_sxm",
                backend_name="trtllm",
                total_gpus=16,
                afd_tp_a_candidates=[3],
            )

    def test_combined_prefill_inherits_backend_wideep_and_preserves_candidates(self):
        task = Task(
            serving_mode="afd",
            model_path="Qwen/Qwen3-32B",
            system_name="h200_sxm",
            backend_name="sglang",
            backend_version="test-version",
            enable_wideep=True,
            total_gpus=32,
            prefill_num_gpu_candidates=[8],
            prefill_tp_candidates=[4],
        )

        assert task.prefill_model_path == task.model_path
        assert task.prefill_system_name == task.system_name
        assert task.prefill_backend_name == "sglang"
        assert task.prefill_backend_version == "test-version"
        assert task.prefill_enable_wideep is True
        assert task.prefill_num_gpu_candidates == [8]
        assert task.prefill_tp_candidates == [4]


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
        assert merged_best["afd"]["_task_key"].tolist() == ["afd_trtllm"]

    def test_columns_afd_contains_new_fields(self):
        for col in (
            "parallel",
            "nextn",
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


def test_auto_result_tasks_are_scoped_to_mode_and_preserve_variant_identity():
    tasks = {
        "agg_trtllm_fast": SimpleNamespace(serving_mode="agg", primary_backend_name="trtllm"),
        "agg_trtllm_safe": SimpleNamespace(serving_mode="agg", primary_backend_name="trtllm"),
        "disagg_trtllm": SimpleNamespace(serving_mode="disagg", primary_backend_name="trtllm"),
    }
    result_df = pd.DataFrame(
        [
            {
                "backend": "trtllm",
                "_task_key": "agg_trtllm_safe",
            }
        ]
    )

    exp_tasks = _auto_result_tasks("agg", tasks, result_df)
    row_task = _task_for_result_row("agg", result_df.iloc[0], exp_tasks)

    assert list(exp_tasks) == ["agg_trtllm_safe"]
    assert row_task is tasks["agg_trtllm_safe"]


def test_afd_worker_table_uses_prefill_num_gpus_per_worker():
    config_df = pd.DataFrame(
        [
            {
                "backend": "trtllm",
                "tokens/s/gpu": 10.0,
                "tokens/s/user": 1.0,
                "request_rate": 1.0,
                "ttft": 10.0,
                "tpot": 10.0,
                "request_latency": 20.0,
                "concurrency": 1,
                "num_total_gpus": 24,
                "(a)nodes": 1,
                "(a)tp": 4,
                "(a)bs": 8,
                "(a)workers": 2,
                "(f)nodes": 1,
                "(f)ep": 1,
                "(f)workers": 8,
                "(p)workers": 2,
                "(p)tp": 1,
                "(p)dp": 4,
                "(p)num_gpus": 4,
            }
        ]
    )

    table = _plot_worker_setup_table(
        "afd",
        config_df,
        total_gpus=24,
        tpot_target=50.0,
        top=1,
        is_moe=False,
        request_latency_target=None,
        show_power=False,
    )

    assert "24 (=A8+F8+P8)" in table


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
        assert task_config.afd_max_a_batch_size == 1024

    def test_build_experiment_task_configs_afd_custom_max_a_batch_size(self):
        from aiconfigurator.cli.main import build_experiment_tasks

        config = {
            "afd_custom_max_batch": {
                "serving_mode": "afd",
                "model_path": "Qwen/Qwen3-32B",
                "system_name": "h200_sxm",
                "backend_name": "trtllm",
                "total_gpus": 32,
                "database_mode": "HYBRID",
                "afd_max_a_batch_size": 1536,
            }
        }

        task_configs = build_experiment_tasks(config=config)

        assert task_configs["afd_custom_max_batch"].afd_max_a_batch_size == 1536

    def test_build_experiment_preserves_afd_fields(self):
        from aiconfigurator.cli.main import build_experiment_tasks

        config = {
            "afd_fields": {
                "serving_mode": "afd",
                "model_path": "Qwen/Qwen3-32B",
                "system_name": "h200_sxm",
                "backend_name": "trtllm",
                "total_gpus": 32,
                "database_mode": "HYBRID",
                "nextn": 2,
                "nextn_accepted": 1.35,
                "gemm_quant_mode": "fp8",
                "afd_combined_with_pd": False,
                "afd_tp_a_candidates": [4],
                "afd_microbatch_candidates": [3],
                "afd_pipeline_model_candidates": ["conservative"],
                "afd_max_candidates": 100,
                "afd_candidate_overflow": "truncate",
            }
        }

        task = build_experiment_tasks(config=config)["afd_fields"]

        assert task.nextn == 2
        assert task.nextn_accepted == pytest.approx(1.35)
        assert task.gemm_quant_mode == common.GEMMQuantMode.fp8
        assert task.afd_combined_with_pd is False
        assert task.afd_tp_a_candidates == [4]
        assert task.afd_microbatch_candidates == [3]
        assert task.afd_pipeline_model_candidates == ["conservative"]
        assert task.afd_max_candidates == 100
        assert task.afd_candidate_overflow == "truncate"

    def test_build_experiment_preserves_pinned_topology(self):
        from aiconfigurator.cli.main import build_experiment_tasks

        config = {
            "afd_pinned": {
                "serving_mode": "afd",
                "model_path": "Qwen/Qwen3-32B",
                "system_name": "h200_sxm",
                "backend_name": "trtllm",
                "total_gpus": 16,
                "database_mode": "HYBRID",
                "afd_n_a_nodes": 1,
                "afd_n_f_nodes": 1,
                "afd_tp_a": 4,
                "afd_a_batch_size": 64,
            }
        }

        task = build_experiment_tasks(config=config)["afd_pinned"]

        assert task._afd_topology_pinned is True
        assert task.afd_n_a_nodes == 1
        assert task.afd_n_f_nodes == 1
        assert task.afd_tp_a == 4
        assert task.afd_a_batch_size == 64

    def test_build_experiment_rejects_unknown_afd_fields(self, caplog):
        from aiconfigurator.cli.main import build_experiment_tasks

        config = {
            "afd_typo": {
                "serving_mode": "afd",
                "model_path": "Qwen/Qwen3-32B",
                "system_name": "h200_sxm",
                "total_gpus": 16,
                "database_mode": "HYBRID",
                "afd_tp_a_canditates": [4],
            }
        }

        with caplog.at_level(logging.ERROR):
            tasks = build_experiment_tasks(config=config)

        assert tasks == {}
        assert "unknown key(s): 'afd_tp_a_canditates'" in caplog.text


def test_hybrid_auto_mode_skips_missing_decode_backend(monkeypatch):
    import aiconfigurator.cli.main as cli_main

    monkeypatch.setattr(
        cli_main.perf_database,
        "get_supported_databases",
        lambda: {
            "h200_sxm": {
                "trtllm": ["test-version"],
                "sglang": ["test-version"],
            },
            "h100_sxm": {
                "trtllm": ["test-version"],
            },
        },
    )
    monkeypatch.setattr(cli_main, "check_is_moe", lambda _: False)
    monkeypatch.setattr(cli_main, "Task", lambda **kwargs: SimpleNamespace(**kwargs))

    tasks = cli_main.build_default_tasks(
        model_path="Qwen/Qwen3-32B",
        total_gpus=8,
        system="h200_sxm",
        decode_system="h100_sxm",
        backend="auto",
        backend_version="test-version",
        database_mode="HYBRID",
    )

    assert set(tasks) == {"agg_trtllm", "disagg_trtllm", "agg_sglang"}


def test_save_results_skips_afd_deployment_artifacts(tmp_path, monkeypatch, caplog):
    import aiconfigurator.cli.report_and_save as report_and_save

    task = Task(
        serving_mode="afd",
        model_path="Qwen/Qwen3-32B",
        system_name="h200_sxm",
        backend_name="trtllm",
        backend_version="test-version",
        total_gpus=16,
    )
    result = pd.DataFrame([{"tokens/s/user": 10.0, "tokens/s/gpu": 2.0, "power_w": float("nan")}])
    monkeypatch.setattr(
        report_and_save,
        "task_config_to_generator_config",
        lambda **_: pytest.fail("AFD must not enter the P/D generator bridge"),
    )

    with caplog.at_level(logging.WARNING):
        save_results(
            SimpleNamespace(inclusive_tpot=False),
            best_configs={"afd": result},
            pareto_fronts={"afd": result},
            tasks={"afd": task},
            save_dir=str(tmp_path),
            generated_backend_version="test-version",
            backend="trtllm",
        )

    result_dir = next(tmp_path.rglob("exp_config.yaml")).parent
    assert (result_dir / "best_config_topn.csv").is_file()
    assert pd.isna(pd.read_csv(result_dir / "best_config_topn.csv").loc[0, "power_w"])
    assert not (result_dir / "top1").exists()
    assert "Skipping deployment artifact generation for AFD experiment 'afd'" in caplog.text


def test_save_results_propagates_partial_video_rejection_before_writing(tmp_path):
    task = SimpleNamespace(
        num_videos_per_request=0,
        video_height=0,
        video_width=0,
        video_frames=8,
        num_video_tokens=0,
    )

    with pytest.raises(NotImplementedError, match="Saved deployment artifacts do not support video"):
        save_results(
            SimpleNamespace(inclusive_tpot=False),
            best_configs={},
            pareto_fronts={},
            tasks={"agg": task},
            save_dir=str(tmp_path),
        )

    assert list(tmp_path.iterdir()) == []


def test_save_results_auto_handles_asymmetric_backends_by_mode(tmp_path):
    tasks = {
        "agg_trtllm": Task(
            serving_mode="agg",
            model_path="Qwen/Qwen3-32B",
            system_name="h200_sxm",
            backend_name="trtllm",
            backend_version="test-version",
            total_gpus=16,
        ),
        "agg_vllm": Task(
            serving_mode="agg",
            model_path="Qwen/Qwen3-32B",
            system_name="h200_sxm",
            backend_name="vllm",
            backend_version="test-version",
            total_gpus=16,
        ),
        "disagg_trtllm": Task(
            serving_mode="disagg",
            prefill_model_path="Qwen/Qwen3-32B",
            prefill_system_name="h200_sxm",
            prefill_backend_name="trtllm",
            prefill_backend_version="test-version",
            decode_model_path="Qwen/Qwen3-32B",
            decode_system_name="h200_sxm",
            decode_backend_name="trtllm",
            decode_backend_version="test-version",
            total_gpus=16,
        ),
        "afd_trtllm": Task(
            serving_mode="afd",
            model_path="Qwen/Qwen3-32B",
            system_name="h200_sxm",
            backend_name="trtllm",
            backend_version="test-version",
            total_gpus=16,
            afd_combined_with_pd=False,
        ),
    }
    empty_results = {mode: pd.DataFrame() for mode in ("agg", "disagg", "afd")}

    save_results(
        SimpleNamespace(inclusive_tpot=False),
        best_configs=empty_results,
        pareto_fronts=empty_results,
        tasks=tasks,
        save_dir=str(tmp_path),
        backend="auto",
    )

    result_dir = next(tmp_path.rglob("pareto_frontier.png")).parent
    assert (result_dir / "agg" / "trtllm_exp_config.yaml").is_file()
    assert (result_dir / "agg" / "vllm_exp_config.yaml").is_file()
    assert (result_dir / "disagg" / "trtllm_exp_config.yaml").is_file()
    assert not (result_dir / "disagg" / "vllm_exp_config.yaml").exists()
    assert (result_dir / "afd" / "trtllm_exp_config.yaml").is_file()
    assert not (result_dir / "afd" / "vllm_exp_config.yaml").exists()
