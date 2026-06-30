# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unit-level workflow tests for CLI wiring.

These tests validate the wiring between CLI entrypoints and internal builders/executors,
while keeping heavy computation mocked out.
"""

import argparse
import json
import logging
import sys
import types
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import yaml

from aiconfigurator.cli.main import (
    _execute_tasks,
    _resolve_cli_log_level,
    build_default_tasks,
    build_experiment_tasks,
    configure_parser,
)
from aiconfigurator.cli.main import main as cli_main
from aiconfigurator.cli.report_and_save import _apply_inclusive_tpot
from aiconfigurator.cli.spica.helper import (
    _build_spica_thorough_config_data,
    _build_spica_trace_result_bundle,
    _build_spica_trace_search_space,
    _save_spica_trace_artifacts,
    _spica_candidates_to_result_df,
    _spica_extra_input_lines,
    _spica_generated_backend_version,
    _spica_generator_overrides,
    _spica_kvbm_config,
    _spica_role_worker_overrides,
    _spica_router_enabled,
    _SpicaReplayEvaluatorCompat,
    _validate_spica_artifact_target,
)
from aiconfigurator.sdk.errors import NoFeasibleConfigError

pytestmark = pytest.mark.unit


class TestCLILogLevelResolution:
    def test_defaults_to_info(self, monkeypatch) -> None:
        monkeypatch.delenv("AICONFIGURATOR_LOG_LEVEL", raising=False)
        args = argparse.Namespace(log_level=None)
        assert _resolve_cli_log_level(args) == logging.INFO

    def test_env_var_controls_level(self, monkeypatch) -> None:
        monkeypatch.setenv("AICONFIGURATOR_LOG_LEVEL", "debug")
        args = argparse.Namespace(log_level=None)
        assert _resolve_cli_log_level(args) == logging.DEBUG

    def test_cli_flag_overrides_env_var(self, monkeypatch) -> None:
        monkeypatch.setenv("AICONFIGURATOR_LOG_LEVEL", "warning")
        args = argparse.Namespace(log_level="DEBUG")
        assert _resolve_cli_log_level(args) == logging.DEBUG

    def test_invalid_env_var_falls_back_to_info(self, monkeypatch) -> None:
        monkeypatch.setenv("AICONFIGURATOR_LOG_LEVEL", "not-a-level")
        args = argparse.Namespace(log_level=None)
        assert _resolve_cli_log_level(args) == logging.INFO


class TestCLIIntegration:
    """Workflow tests for the CLI orchestration layer (builders/executor/save)."""

    @patch("aiconfigurator.cli.main._execute_tasks")
    @patch("aiconfigurator.cli.main.build_default_tasks")
    def test_cli_main_success_flow(self, mock_build_default, mock_execute, sample_cli_args_with_save_dir):
        """Test successful CLI main execution flow for default mode."""
        mock_task_config = MagicMock(name="TaskConfig")
        mock_build_default.return_value = {"agg": mock_task_config}

        mock_results_df = MagicMock(name="ResultsDF")
        mock_best_configs = {"agg": MagicMock(name="BestConfigDF")}
        mock_best_throughputs = {"agg": 123.4}
        mock_execute.return_value = (
            "agg",
            mock_best_configs,
            {"agg": mock_results_df},
            mock_best_throughputs,
            {"agg": {"ttft": 100.0, "tpot": 10.0, "request_latency": 1000.0}},
        )

        with patch("aiconfigurator.cli.main.save_results") as mock_save:
            cli_main(sample_cli_args_with_save_dir)

        mock_build_default.assert_called_once()
        mock_execute.assert_called_once()
        builder_args, builder_mode = mock_execute.call_args.args
        assert builder_args == {"agg": mock_task_config}
        assert builder_mode == "default"

        mock_save.assert_called_once()
        save_kwargs = mock_save.call_args.kwargs
        assert save_kwargs["args"] == sample_cli_args_with_save_dir
        assert save_kwargs["best_configs"] == mock_best_configs
        assert save_kwargs["pareto_fronts"] == {"agg": mock_results_df}
        assert save_kwargs["tasks"] == {"agg": mock_task_config}
        assert save_kwargs["save_dir"] == sample_cli_args_with_save_dir.save_dir

    @patch("aiconfigurator.cli.main.save_results")
    @patch("aiconfigurator.cli.main._execute_tasks")
    @patch("aiconfigurator.cli.main.build_experiment_tasks")
    def test_cli_main_success_flow_exp_mode(
        self,
        mock_build_exp,
        mock_execute,
        mock_save,
        cli_args_factory,
        mock_exp_yaml_path,
    ):
        """Test successful CLI main execution flow for exp mode."""
        mock_task_config = MagicMock(name="TaskConfig")
        mock_build_exp.return_value = {"my_exp": mock_task_config}
        mock_results_df = MagicMock(name="ResultsDF")
        mock_best_configs = {"my_exp": MagicMock(name="BestConfigDF")}
        mock_best_throughputs = {"my_exp": 123.4}
        mock_execute.return_value = (
            "my_exp",
            mock_best_configs,
            {"my_exp": mock_results_df},
            mock_best_throughputs,
            {"my_exp": {"ttft": 100.0, "tpot": 10.0, "request_latency": 1000.0}},
        )

        args = cli_args_factory(
            mode="exp",
            extra_args=["--yaml-path", str(mock_exp_yaml_path)],
            save_dir=str(mock_exp_yaml_path.parent),
        )

        cli_main(args)

        mock_build_exp.assert_called_once()
        mock_execute.assert_called_once()
        builder_args, builder_mode = mock_execute.call_args.args
        assert builder_args == {"my_exp": mock_task_config}
        assert builder_mode == "exp"

        mock_save.assert_called_once()
        save_kwargs = mock_save.call_args.kwargs
        assert save_kwargs["args"] == args
        assert save_kwargs["best_configs"] == mock_best_configs
        assert save_kwargs["pareto_fronts"] == {"my_exp": mock_results_df}
        assert save_kwargs["tasks"] == {"my_exp": mock_task_config}
        assert save_kwargs["save_dir"] == str(mock_exp_yaml_path.parent)

    @pytest.mark.parametrize(
        "mode,build_patch",
        [
            ("default", "aiconfigurator.cli.main.build_default_tasks"),
            ("exp", "aiconfigurator.cli.main.build_experiment_tasks"),
        ],
    )
    @patch("aiconfigurator.cli.main._execute_tasks")
    def test_cli_main_build_dispatch(self, mock_execute, mode, build_patch, cli_args_factory, mock_exp_yaml_path):
        """Main should dispatch to the correct builder based on CLI mode."""
        mock_execute.return_value = ("agg", {}, {}, {}, {})
        mock_task_config = MagicMock(name="TaskConfig")

        with patch(build_patch) as mock_builder:
            mock_builder.return_value = {"agg": mock_task_config}
            if mode == "exp":
                args = cli_args_factory(mode=mode, extra_args=["--yaml-path", str(mock_exp_yaml_path)])
            else:
                args = cli_args_factory(mode=mode)
            cli_main(args)

        mock_builder.assert_called_once()
        mock_execute.assert_called_once_with({"agg": mock_task_config}, mode, top_n=5)

    @patch("aiconfigurator.cli.main.run_spica_thorough_default")
    @patch("aiconfigurator.cli.main._execute_tasks")
    @patch("aiconfigurator.cli.main.build_default_tasks")
    def test_cli_default_thorough_sweep_dispatches_to_spica(
        self,
        mock_build_default,
        mock_execute,
        mock_run_spica,
        cli_args_factory,
    ):
        """default --thorough-sweep should use Spica with CLI-derived config."""
        args = cli_args_factory(mode="default", thorough_sweep=True)

        cli_main(args)

        mock_run_spica.assert_called_once_with(args)
        mock_build_default.assert_not_called()
        mock_execute.assert_not_called()

    @patch("aiconfigurator.cli.main.run_spica_thorough_default")
    @patch("aiconfigurator.cli.main._execute_tasks")
    @patch("aiconfigurator.cli.main.build_default_tasks")
    def test_cli_default_thorough_config_dispatches_without_default_inputs(
        self,
        mock_build_default,
        mock_execute,
        mock_run_spica,
        tmp_path,
    ):
        """default --thorough-config should not require dummy model/system/GPU args."""
        parser = argparse.ArgumentParser()
        configure_parser(parser)
        config_path = tmp_path / "spica.yaml"
        config_path.write_text("search_space: {}\n", encoding="utf-8")
        args = parser.parse_args(["default", "--thorough-config", str(config_path)])

        cli_main(args)

        mock_run_spica.assert_called_once_with(args)
        mock_build_default.assert_not_called()
        mock_execute.assert_not_called()

    def test_cli_default_missing_required_inputs_raises(self):
        """Legacy default still needs model/system/GPU when no native Spica config is provided."""
        parser = argparse.ArgumentParser()
        configure_parser(parser)
        args = parser.parse_args(["default"])

        with pytest.raises(SystemExit, match="default mode requires"):
            cli_main(args)

    def test_spica_default_search_space_collapses_single_gpu_noops(self, cli_args_factory):
        """Single-GPU Spica default sweeps should avoid routing/planner choices that cannot help."""
        args = cli_args_factory(
            mode="default",
            total_gpus=1,
            max_seq_len=8192,
        )

        search_space = _build_spica_trace_search_space(args, ["trtllm"])

        assert search_space["gpu_budget"] == 1
        assert search_space["context_length"] == 8192
        assert search_space["deployment_mode"] == ["agg"]
        assert search_space["router_mode"] == ["round_robin"]
        assert search_space["planner_scaling_policy"] == ["disabled"]

    def test_spica_thorough_config_data_uses_cli_inputs(self, cli_args_factory):
        """CLI-derived thorough mode should build a Spica config-shaped payload."""
        args = cli_args_factory(
            mode="default",
            thorough_sweep=True,
            model_path="meta-llama/Meta-Llama-3.1-8B",
            total_gpus=4,
            system="gb200",
            backend="trtllm",
            isl=2048,
            osl=512,
            prefix=256,
            max_seq_len=8192,
            nextn=1,
            ttft=8000.0,
            tpot=200.0,
        )

        config_data = _build_spica_thorough_config_data(args, ["trtllm"])

        assert config_data["search_space"]["model_name"] == "meta-llama/Meta-Llama-3.1-8B"
        assert config_data["search_space"]["hardware_sku"] == "gb200"
        assert config_data["search_space"]["gpu_budget"] == 4
        assert config_data["search_space"]["backend"] == ["trtllm"]
        assert config_data["search_space"]["context_length"] == 8192
        assert config_data["search_space"]["aic_nextn"] == 1
        assert config_data["search_space"]["router_mode"] == ["round_robin"]
        assert config_data["search_space"]["planner_scaling_policy"] == ["disabled"]
        assert config_data["search_space"]["planner_fpm_sampling"] == ["default"]
        assert config_data["search_space"]["planner_load_sensitivity"] == ["default"]
        assert config_data["workload"]["isl"] == 2048
        assert config_data["workload"]["osl"] == 512
        assert config_data["workload"]["concurrency"] == 512
        assert config_data["workload"]["num_request_ratio"] == pytest.approx(1000 / 512)
        assert "request_count" not in config_data["workload"]
        assert config_data["workload"]["shared_prefix_ratio"] == 0.125
        assert config_data["workload"]["num_prefix_groups"] == 1
        assert config_data["goal"] == {"target": "goodput_per_gpu", "sla": {"ttft_ms": 8000.0, "itl_ms": 200.0}}
        assert config_data["sweep"]["max_rounds"] == 3
        assert config_data["sweep"]["parallel_evals"] == 16

    def test_spica_replay_evaluator_compat_forwards_concurrency_override(self, monkeypatch):
        """Spica pareto sweeps pass a per-trial concurrency override into the evaluator."""
        captured: dict[str, object] = {}

        class FakeReplayEvaluator:
            def __init__(self, workload, goal):
                captured["workload"] = workload
                captured["goal"] = goal

            def evaluate(self, plan, *, concurrency_override=None):
                captured["plan"] = plan
                captured["concurrency_override"] = concurrency_override
                return {"goodput_output_throughput_tok_s": 123.0}

        fake_spica = types.ModuleType("spica")
        fake_spica.__path__ = []
        fake_evaluator = types.ModuleType("spica.evaluator")
        fake_evaluator.ReplayEvaluator = FakeReplayEvaluator
        fake_spica.evaluator = fake_evaluator
        monkeypatch.setitem(sys.modules, "spica", fake_spica)
        monkeypatch.setitem(sys.modules, "spica.evaluator", fake_evaluator)

        evaluator = _SpicaReplayEvaluatorCompat(workload="workload", goal="goal")
        assert evaluator.evaluate("plan", concurrency_override=7) == {"goodput_output_throughput_tok_s": 123.0}
        assert captured == {
            "workload": "workload",
            "goal": "goal",
            "plan": "plan",
            "concurrency_override": 7,
        }

    def test_spica_trace_results_use_default_result_shape(self, cli_args_factory):
        """Spica trace candidates should be adapted to the legacy default-mode result bundle."""
        candidates = [
            {
                "config": {
                    "deployment_mode": "disagg",
                    "backend": "trtllm",
                    "model_name": "Qwen/Qwen3-32B-FP8",
                    "prefill_replicas": 2,
                    "prefill_tp": 2,
                    "prefill_pp": 1,
                    "prefill_attention_dp": 1,
                    "prefill_moe_tp": 1,
                    "prefill_moe_ep": 1,
                    "prefill_max_num_batched_tokens": 8192,
                    "prefill_max_num_seqs": 64,
                    "decode_replicas": 4,
                    "decode_tp": 1,
                    "decode_pp": 1,
                    "decode_attention_dp": 1,
                    "decode_moe_tp": 1,
                    "decode_moe_ep": 1,
                    "decode_max_num_batched_tokens": 4096,
                    "decode_max_num_seqs": 128,
                    "router_mode": "kv_router",
                    "planner_scaling_policy": "predictive",
                    "enable_throughput_scaling": True,
                    "enable_load_scaling": False,
                },
                "used_gpus": 8,
                "score": 321.5,
                "metrics": {
                    "goodput_output_throughput_tok_s": 2572.0,
                    "output_throughput_tok_s": 3000.0,
                    "mean_ttft_ms": 120.0,
                    "mean_tpot_ms": 8.0,
                    "mean_e2e_latency_ms": 900.0,
                },
            },
            {
                "config": {
                    "deployment_mode": "agg",
                    "backend": "vllm",
                    "model_name": "Qwen/Qwen3-32B-FP8",
                    "replicas": 4,
                    "tp": 2,
                    "pp": 1,
                    "attention_dp": 1,
                    "moe_tp": 1,
                    "moe_ep": 1,
                    "agg_max_num_batched_tokens": 4096,
                    "agg_max_num_seqs": 128,
                    "router_mode": "round_robin",
                    "enable_throughput_scaling": False,
                    "enable_load_scaling": False,
                },
                "used_gpus": 8,
                "score": 280.0,
                "metrics": {
                    "goodput_output_throughput_tok_s": 2240.0,
                    "output_throughput_tok_s": 2500.0,
                    "mean_ttft_ms": 150.0,
                    "mean_tpot_ms": 9.0,
                    "mean_e2e_latency_ms": 950.0,
                },
            },
        ]

        args = cli_args_factory(
            mode="default",
            model_path="Qwen/Qwen3-32B-FP8",
            total_gpus=16,
            top_n=5,
            ttft=2000.0,
            tpot=30.0,
        )
        config = argparse.Namespace(workload=argparse.Namespace(trace_path="/data/replay/traffic.jsonl"))
        result_bundle = _build_spica_trace_result_bundle(candidates, args, config=config)

        assert result_bundle.chosen_exp == "disagg"
        assert result_bundle.trace_path == "/data/replay/traffic.jsonl"
        assert set(result_bundle.tasks) == {"agg", "disagg"}
        assert result_bundle.tasks["disagg"].serving_mode == "disagg"
        assert result_bundle.tasks["disagg"].primary_model_path == "Qwen/Qwen3-32B-FP8"
        assert result_bundle.best_throughputs["disagg"] == pytest.approx(321.5)
        assert result_bundle.pareto_x_axis == {"disagg": "tokens/s/user", "agg": "tokens/s/user"}

        disagg_best = result_bundle.best_configs["disagg"].iloc[0]
        assert disagg_best["tokens/s/gpu"] == pytest.approx(321.5)
        assert disagg_best["tokens/s/gpu_cluster"] == pytest.approx(321.5)
        assert disagg_best["tokens/s/user"] == pytest.approx(125.0)
        assert disagg_best["(p)parallel"] == "tp2_pp1"
        assert disagg_best["(d)parallel"] == "tp1_pp1"
        assert disagg_best["router"] == "kv_router"

        agg_best = result_bundle.best_configs["agg"].iloc[0]
        assert "(p)tp" not in result_bundle.best_configs["agg"].columns
        assert agg_best["parallel"] == "tp2_pp1"
        assert agg_best["tokens/s/gpu"] == pytest.approx(280.0)

    def test_spica_thorough_config_result_bundle_uses_candidate_scalars(self):
        """Native Spica configs can use multi-value search-space fields; reports use evaluated candidate values."""
        config = argparse.Namespace(
            search_space=argparse.Namespace(
                model_name=["Qwen/Qwen3-32B-FP8"],
                hardware_sku=["gb200"],
                gpu_budget=[1, 4],
                backend=["trtllm", "vllm"],
            ),
            workload=argparse.Namespace(isl=128, osl=16),
            goal=argparse.Namespace(
                target="goodput_per_gpu",
                sla=argparse.Namespace(ttft_ms=8000.0, itl_ms=200.0, e2e_ms=None),
            ),
        )
        args = argparse.Namespace(
            backend="auto",
            model_path="Wrong/CLI-Model",
            system="wrong_cli_system",
            total_gpus=99,
            top_n=1,
            strict_sla=False,
            ttft=1.0,
            tpot=2.0,
        )
        candidates = [
            {
                "config": {
                    "deployment_mode": "agg",
                    "backend": "trtllm",
                    "model_name": "Qwen/Qwen3-32B-FP8",
                    "hardware_sku": "gb200",
                    "tp": 4,
                    "pp": 1,
                    "attention_dp": 1,
                    "replicas": 1,
                    "router_mode": "round_robin",
                    "planner_scaling_policy": "disabled",
                },
                "used_gpus": 4,
                "score": 250.0,
                "metrics": {
                    "goodput_output_throughput_tok_s": 1000.0,
                    "output_throughput_tok_s": 1100.0,
                    "mean_ttft_ms": 100.0,
                    "mean_tpot_ms": 10.0,
                    "mean_e2e_latency_ms": 260.0,
                },
            }
        ]

        result_bundle = _build_spica_trace_result_bundle(
            candidates,
            args,
            config=config,
            config_path="/tmp/spica.yaml",
        )

        task = result_bundle.tasks["agg"]
        assert task.primary_model_path == "Qwen/Qwen3-32B-FP8"
        assert task.primary_system_name == "gb200"
        assert task.primary_backend_name == "trtllm"
        assert task.total_gpus == 4
        assert task.isl == 128
        assert task.osl == 16
        assert task.ttft == 8000.0
        assert task.tpot == 200.0
        assert result_bundle.workload_label == "config"

    def test_native_throughput_goal_owns_ranking_and_does_not_inherit_cli_sla(self):
        config = argparse.Namespace(
            search_space=argparse.Namespace(
                model_name="model",
                hardware_sku="h200_sxm",
                gpu_budget=8,
                min_gpu_budget=4,
                min_endpoint=2,
                context_length=32768,
            ),
            workload=argparse.Namespace(isl=128, osl=32),
            goal=argparse.Namespace(target="throughput", sla=None),
        )
        args = argparse.Namespace(
            backend="trtllm",
            model_path="wrong",
            system="wrong",
            total_gpus=99,
            top_n=1,
            strict_sla=True,
            ttft=2000.0,
            tpot=30.0,
        )
        candidates = [
            {
                "config": {"deployment_mode": "agg", "backend": "trtllm", "used_gpus": 1},
                "used_gpus": 1,
                "score": 500.0,
                "metrics": {
                    "output_throughput_tok_s": 500.0,
                    "mean_tpot_ms": 20.0,
                    "mean_e2e_latency_ms": 100.0,
                },
            },
            {
                "config": {"deployment_mode": "agg", "backend": "trtllm", "used_gpus": 4},
                "used_gpus": 4,
                "score": 1000.0,
                "metrics": {
                    "output_throughput_tok_s": 1000.0,
                    "mean_tpot_ms": 40.0,
                    "mean_e2e_latency_ms": 200.0,
                },
            },
        ]

        bundle = _build_spica_trace_result_bundle(candidates, args, config=config, config_path="/tmp/native.yaml")

        best = bundle.best_configs["agg"].iloc[0]
        assert best["spica_candidate_id"] == 1
        assert best["objective_target"] == "throughput"
        assert best["objective_score"] == pytest.approx(1000.0)
        assert best["objective_value"] == pytest.approx(1000.0)
        assert best["tokens/s/gpu"] == pytest.approx(250.0)  # physical throughput/GPU, not score
        assert bundle.best_throughputs["agg"] == pytest.approx(250.0)
        assert bundle.best_objective_scores["agg"] == pytest.approx(1000.0)
        assert bundle.tasks["agg"].ttft == 0.0
        assert bundle.tasks["agg"].tpot == 0.0
        assert bundle.pareto_x_axis["agg"] == "tokens/s/user"
        assert bundle.pareto_y_axis["agg"] == "objective_value"
        saved_config = bundle.candidates[1]["config"]
        assert saved_config["context_length"] == 32768
        assert saved_config["gpu_budget"] == 8
        assert saved_config["min_gpu_budget"] == 4
        assert saved_config["min_endpoint"] == 2
        assert saved_config["planner_optimization_target"] == "throughput"
        assert saved_config["planner_ttft_ms"] is None
        assert saved_config["planner_itl_ms"] is None

    def test_native_latency_and_custom_pareto_use_raw_objectives(self):
        latency_goal = argparse.Namespace(target="e2e_latency", sla=None)
        latency_df = _spica_candidates_to_result_df(
            [
                {
                    "config": {"deployment_mode": "agg"},
                    "used_gpus": 2,
                    "score": -125.0,
                    "metrics": {"output_throughput_tok_s": 400.0, "mean_e2e_latency_ms": 125.0},
                }
            ],
            goal=latency_goal,
        )
        assert latency_df.loc[0, "objective_score"] == pytest.approx(-125.0)
        assert latency_df.loc[0, "objective_value"] == pytest.approx(125.0)
        assert latency_df.loc[0, "tokens/s/gpu"] == pytest.approx(200.0)

        per_user_df = _spica_candidates_to_result_df(
            [
                {
                    "config": {"deployment_mode": "agg"},
                    "used_gpus": 1,
                    "score": 0.0,
                    "metrics": {"mean_tpot_ms": 10.0},
                }
            ],
            goal=argparse.Namespace(target="throughput_per_user", sla=None),
        )
        assert per_user_df.loc[0, "tokens/s/user"] == pytest.approx(100.0)
        assert per_user_df.loc[0, "objective_value"] == pytest.approx(0.0)

        pareto_goal = argparse.Namespace(
            target="pareto",
            sla=None,
            resolved_pareto_objectives=["throughput", "e2e_latency"],
        )
        config = argparse.Namespace(
            search_space=argparse.Namespace(model_name="model", hardware_sku="h200_sxm", gpu_budget=8),
            workload=argparse.Namespace(isl=128, osl=32),
            goal=pareto_goal,
        )
        args = argparse.Namespace(
            backend="trtllm",
            model_path="wrong",
            system="wrong",
            total_gpus=99,
            top_n=2,
            strict_sla=False,
            ttft=2000.0,
            tpot=30.0,
        )
        candidates = [
            {
                "config": {"deployment_mode": "agg"},
                "used_gpus": 1,
                "score": 100.0,
                "objectives": {"throughput": 100.0, "e2e_latency": 100.0},
                "metrics": {"output_throughput_tok_s": 100.0, "mean_e2e_latency_ms": 100.0},
            },
            {
                "config": {"deployment_mode": "agg"},
                "used_gpus": 1,
                "score": 90.0,
                "objectives": {"throughput": 90.0, "e2e_latency": 90.0},
                "metrics": {"output_throughput_tok_s": 90.0, "mean_e2e_latency_ms": 90.0},
            },
            {
                "config": {"deployment_mode": "agg"},
                "used_gpus": 1,
                "score": 80.0,
                "objectives": {"throughput": 80.0, "e2e_latency": 120.0},
                "metrics": {"output_throughput_tok_s": 80.0, "mean_e2e_latency_ms": 120.0},
            },
        ]

        bundle = _build_spica_trace_result_bundle(candidates, args, config=config, config_path="/tmp/pareto.yaml")
        assert bundle.pareto_x_axis["agg"] == "e2e_latency"
        assert bundle.pareto_y_axis["agg"] == "throughput"
        assert bundle.pareto_fronts["agg"]["spica_candidate_id"].tolist() == [1, 0]
        assert bundle.best_configs["agg"]["spica_candidate_id"].tolist() == [0, 1]
        assert bundle.best_objective_scores["agg"] == pytest.approx(100.0)

    def test_spica_result_uses_candidate_replay_concurrency(self):
        goal = argparse.Namespace(
            target="pareto",
            resolved_pareto_objectives=["throughput_per_gpu", "throughput_per_user"],
        )
        candidate = {
            "config": {
                "deployment_mode": "agg",
                "backend": "sglang",
                "concurrency": 11940,
                "kv_load_ratio": 0.59,
            },
            "used_gpus": 64,
            "score": 0.0,
            "objectives": {"throughput_per_gpu": 1788.4, "throughput_per_user": 23.56},
            "metrics": {
                "output_throughput_tok_s": 114457.6,
                "mean_output_token_throughput_per_user": 23.56,
            },
        }

        result = _spica_candidates_to_result_df([candidate], goal=goal)

        assert result.loc[0, "concurrency"] == 11940
        assert result.loc[0, "kv_load_ratio"] == pytest.approx(0.59)

    @pytest.mark.parametrize(
        ("load", "expected"),
        [
            ({"concurrency": 16}, "concurrency=16"),
            ({"request_rate": 2.5}, "request_rate=2.5"),
            ({"kv_load_ratio": [0.0, 1.0]}, "kv_load_ratio=[0.0, 1.0]"),
        ],
    )
    def test_spica_input_summary_reports_active_load_mode(self, load, expected):
        workload = argparse.Namespace(
            trace_path=None,
            isl=128,
            osl=16,
            num_request_ratio=10.0,
            concurrency=load.get("concurrency"),
            request_rate=load.get("request_rate"),
            kv_load_ratio=load.get("kv_load_ratio"),
        )
        config = argparse.Namespace(workload=workload, goal=argparse.Namespace(sla=None))

        lines = _spica_extra_input_lines(config, None)

        assert lines == [f"Synthetic Workload: ISL=128, OSL=16, {expected}, num_request_ratio=10.0"]

    def test_spica_generator_overrides_prefer_task_over_cli_for_planner(self):
        """Generated planner config should use evaluated Spica config values, not stale default CLI args."""
        args = argparse.Namespace(
            model_path="Wrong/CLI-Model",
            total_gpus=99,
            system="wrong_cli_system",
            ttft=1.0,
            tpot=2.0,
            generator_config=None,
            generator_set=None,
            generator_dynamo_version=None,
            namespace=None,
            transport=None,
            image_pull_secret=None,
            model_cache=None,
        )
        task = argparse.Namespace(primary_model_path="Qwen/Qwen3-32B-FP8", ttft=8000.0, tpot=200.0)
        row = pd.Series(
            {
                "deployment_mode": "disagg",
                "backend": "trtllm",
                "model_name": "Qwen/Qwen3-32B-FP8",
                "enable_throughput_scaling": True,
                "enable_load_scaling": True,
                "prefill_tp": 1,
                "prefill_pp": 1,
                "prefill_attention_dp": 1,
                "decode_tp": 2,
                "decode_pp": 1,
                "decode_attention_dp": 1,
            }
        )

        planner_config = _spica_generator_overrides(args, row, task)["DynConfig"]["planner_config"]

        assert planner_config["model_name"] == "Qwen/Qwen3-32B-FP8"
        assert planner_config["ttft_ms"] == pytest.approx(8000.0)
        assert planner_config["itl_ms"] == pytest.approx(200.0)
        assert planner_config["prefill_engine_num_gpu"] == 1
        assert planner_config["decode_engine_num_gpu"] == 2

    @pytest.mark.parametrize("deployment_target", ["dynamo-python", "llm-d-helm", "llm-d-kustomize"])
    def test_spica_artifacts_fail_closed_when_target_drops_native_features(self, deployment_target):
        generator_config = {
            "DynConfig": {
                "enable_router": True,
                "router_mode": "kv",
                "router_config": {"host_cache_hit_weight": 0.75},
                "planner_config": {"optimization_target": "throughput"},
                "kvbm_config": {"cpu_cache_override_num_blocks": 128},
            }
        }

        with pytest.raises(
            RuntimeError,
            match=rf"cannot faithfully emit router, planner, KVBM.*'{deployment_target}'",
        ):
            _validate_spica_artifact_target(generator_config, deployment_target)

        _validate_spica_artifact_target(generator_config, "dynamo-j2")

    def test_native_generator_overrides_preserve_planner_router_and_kvbm_contract(self):
        args = argparse.Namespace(
            model_path="Wrong/CLI-Model",
            backend="trtllm",
            ttft=2000.0,
            tpot=30.0,
            max_seq_len=None,
            generator_config=None,
            generator_set=None,
            generator_dynamo_version=None,
            namespace=None,
            transport=None,
            image_pull_secret=None,
            model_cache=None,
        )
        task = argparse.Namespace(
            primary_model_path="model",
            planner_optimization_target="throughput",
            ttft=0.0,
            tpot=0.0,
            min_gpu_budget=4,
            min_endpoint=2,
        )
        row = pd.Series(
            {
                "deployment_mode": "disagg",
                "backend": "trtllm",
                "model_name": "model",
                "planner_optimization_target": "throughput",
                "planner_ttft_ms": None,
                "planner_itl_ms": None,
                "enable_throughput_scaling": False,
                "enable_load_scaling": True,
                "gpu_budget": 16,
                "min_gpu_budget": 4,
                "min_endpoint": 2,
                "prefill_tp": 1,
                "prefill_pp": 1,
                "prefill_attention_dp": 1,
                "decode_tp": 2,
                "decode_pp": 1,
                "decode_attention_dp": 1,
                "decode_block_size": 64,
                "router_mode": "kv_router",
                "host_cache_hit_weight": 0.75,
                "disk_cache_hit_weight": 0.25,
                "active_prefill_tokens_threshold": 10000.0,
                "num_g2_blocks": 4096,
                "offload_batch_size": 4,
            }
        )

        dyn_config = _spica_generator_overrides(args, row, task)["DynConfig"]
        planner = dyn_config["planner_config"]
        assert planner["optimization_target"] == "throughput"
        assert planner["max_gpu_budget"] == 16
        assert planner["min_gpu_budget"] == 4
        assert planner["min_endpoint"] == 2
        assert "ttft_ms" not in planner and "itl_ms" not in planner
        router = dyn_config["router_config"]
        assert router["host_cache_hit_weight"] == pytest.approx(0.75)
        assert router["disk_cache_hit_weight"] == pytest.approx(0.25)
        assert router["active_prefill_tokens_threshold"] == 10000
        assert isinstance(router["active_prefill_tokens_threshold"], int)
        assert dyn_config["kvbm_config"] == {
            "cpu_cache_override_num_blocks": 4096,
            "max_transfer_batch_size": 4,
            "max_concurrent_transfers": 4,
        }

    def test_kvbm_batch_size_cannot_enable_offload_without_g2_blocks(self):
        assert _spica_kvbm_config(pd.Series({"num_g2_blocks": 0, "offload_batch_size": 4})) is None
        assert _spica_kvbm_config(pd.Series({"offload_batch_size": 4})) is None

    def test_spica_artifacts_prefer_the_backend_version_used_by_replay(self):
        args = argparse.Namespace(generated_config_version="1.3.0rc14")

        assert _spica_generated_backend_version(args, "trtllm", "1.3.0rc10") == "1.3.0rc10"

    def test_spica_round_robin_router_mode_still_enables_router(self):
        row = pd.Series(
            {
                "deployment_mode": "agg",
                "router_mode": "round_robin",
                "agg_block_size": 64,
            }
        )
        args = argparse.Namespace(
            generator_config=None,
            generator_set=None,
            generator_dynamo_version=None,
            namespace=None,
            transport=None,
            image_pull_secret=None,
            model_cache=None,
        )
        task = argparse.Namespace(primary_model_path="Qwen/Qwen3-32B-FP8", ttft=8000.0, tpot=200.0)

        overrides = _spica_generator_overrides(args, row, task)

        assert _spica_router_enabled(row) is True
        assert overrides["DynConfig"]["enable_router"] is True

    @pytest.mark.parametrize(
        ("row_data", "expected_message"),
        [
            (
                {"decode_block_size": 64, "decode_max_num_batched_tokens": 4100, "context_length": 8192},
                "decode_max_num_batched_tokens=4100 must be divisible by decode_block_size=64",
            ),
            (
                {"decode_block_size": 64, "decode_max_num_batched_tokens": 4096, "context_length": 8200},
                "decode cache_transceiver_max_tokens_in_buffer=8200 must be divisible by decode_block_size=64",
            ),
        ],
    )
    def test_spica_worker_overrides_reject_unaligned_token_limits(self, row_data, expected_message):
        row = pd.Series(row_data)

        with pytest.raises(ValueError, match=expected_message):
            _spica_role_worker_overrides(row, "decode", argparse.Namespace(max_seq_len=None))

    def test_spica_trace_artifacts_include_pareto_outputs(self, tmp_path, cli_args_factory):
        candidates = [
            {
                "config": {
                    "deployment_mode": "agg",
                    "backend": "trtllm",
                    "model_name": "Qwen/Qwen3-32B-FP8",
                    "tp": 1,
                    "pp": 1,
                    "attention_dp": 1,
                    "replicas": 1,
                    "aic_nextn": 2,
                    "agg_max_num_batched_tokens": 4096,
                    "agg_max_num_seqs": 128,
                    "agg_block_size": 64,
                    "agg_gpu_memory_utilization": 0.91,
                    "agg_enable_prefix_caching": True,
                    "context_length": 8192,
                    "router_mode": "round_robin",
                    "planner_scaling_policy": "disabled",
                },
                "used_gpus": 1,
                "score": 100.0,
                "metrics": {
                    "goodput_output_throughput_tok_s": 100.0,
                    "output_throughput_tok_s": 110.0,
                    "mean_ttft_ms": 100.0,
                    "mean_tpot_ms": 10.0,
                    "mean_e2e_latency_ms": 500.0,
                },
            },
            {
                "config": {
                    "deployment_mode": "agg",
                    "backend": "trtllm",
                    "model_name": "Qwen/Qwen3-32B-FP8",
                    "tp": 2,
                    "pp": 1,
                    "attention_dp": 1,
                    "replicas": 1,
                    "agg_max_num_batched_tokens": 2048,
                    "agg_max_num_seqs": 64,
                    "agg_block_size": 32,
                    "agg_gpu_memory_utilization": 0.82,
                    "agg_enable_prefix_caching": False,
                    "router_mode": "round_robin",
                    "planner_scaling_policy": "disabled",
                },
                "used_gpus": 2,
                "score": 80.0,
                "metrics": {
                    "goodput_output_throughput_tok_s": 80.0,
                    "output_throughput_tok_s": 90.0,
                    "mean_ttft_ms": 90.0,
                    "mean_tpot_ms": 20.0,
                    "mean_e2e_latency_ms": 700.0,
                },
            },
            {
                "config": {
                    "deployment_mode": "disagg",
                    "backend": "trtllm",
                    "model_name": "Qwen/Qwen3-32B-FP8",
                    "prefill_replicas": 1,
                    "prefill_tp": 1,
                    "prefill_pp": 1,
                    "prefill_attention_dp": 1,
                    "prefill_max_num_batched_tokens": 8192,
                    "prefill_max_num_seqs": 64,
                    "prefill_block_size": 64,
                    "prefill_gpu_memory_utilization": 0.88,
                    "prefill_enable_prefix_caching": True,
                    "decode_replicas": 1,
                    "decode_tp": 1,
                    "decode_pp": 1,
                    "decode_attention_dp": 2,
                    "decode_max_num_batched_tokens": 4096,
                    "decode_max_num_seqs": 128,
                    "decode_block_size": 32,
                    "decode_gpu_memory_utilization": 0.86,
                    "decode_enable_prefix_caching": False,
                    "context_length": 8192,
                    "router_mode": "kv_router",
                    "overlap_score_credit": 0.5,
                    "prefill_load_scale": 2.0,
                    "router_temperature": 0.2,
                    "active_decode_blocks_threshold": 0.75,
                    "active_prefill_tokens_threshold": 10000,
                    "active_prefill_tokens_threshold_frac": 2.0,
                    "no_admission_control": False,
                    "planner_scaling_policy": "predictive",
                    "enable_throughput_scaling": True,
                    "enable_load_scaling": True,
                    "throughput_adjustment_interval_seconds": 180,
                    "load_adjustment_interval_seconds": 5,
                    "max_num_fpm_samples": 25,
                    "fpm_sample_bucket_size": 4,
                    "load_scaling_down_sensitivity": 20,
                    "load_min_observations": 6,
                    "num_g2_blocks": 4096,
                    "offload_batch_size": 32,
                },
                "used_gpus": 3,
                "score": 120.0,
                "metrics": {
                    "goodput_output_throughput_tok_s": 240.0,
                    "output_throughput_tok_s": 260.0,
                    "mean_ttft_ms": 80.0,
                    "mean_tpot_ms": 12.5,
                    "mean_e2e_latency_ms": 600.0,
                },
            },
        ]

        pareto_df = _spica_candidates_to_result_df(candidates)
        assert pareto_df["tokens/s/user"].tolist() == pytest.approx([100.0, 50.0, 80.0])
        assert pareto_df["tokens/s/gpu"].tolist() == pytest.approx([100.0, 80.0, 120.0])
        assert pareto_df["goodput/s/gpu"].tolist() == pytest.approx([100.0, 80.0, 120.0])

        args = cli_args_factory(
            mode="default",
            model_path="Qwen/Qwen3-32B-FP8",
            total_gpus=3,
            top_n=2,
            max_seq_len=8192,
        )
        config = argparse.Namespace(
            workload=argparse.Namespace(trace_path="/tmp/traffic.jsonl", trace_format="mooncake")
        )
        result_bundle = _build_spica_trace_result_bundle(candidates, args, config=config)
        written_paths = _save_spica_trace_artifacts(result_bundle, str(tmp_path))
        result_dir = next(tmp_path.iterdir())
        written_names = {str(path).replace(f"{result_dir}/", "") for path in written_paths}

        assert {
            "spica_candidates.yaml",
            "spica_candidates.csv",
            "pareto.csv",
            "pareto_frontier.png",
            "agg/exp_config.yaml",
            "agg/pareto.csv",
            "agg/best_config_topn.csv",
            "agg/top1/agg_config.yaml",
            "agg/top1/bench_run.sh",
            "agg/top1/generator_config.yaml",
            "agg/top1/k8s_bench.yaml",
            "agg/top1/k8s_deploy.yaml",
            "agg/top1/run_0.sh",
            "agg/top1/sflow.yaml",
            "agg/top1/spica_candidate.yaml",
            "agg/top2/agg_config.yaml",
            "agg/top2/generator_config.yaml",
            "agg/top2/sflow.yaml",
            "agg/top2/spica_candidate.yaml",
            "disagg/exp_config.yaml",
            "disagg/pareto.csv",
            "disagg/best_config_topn.csv",
            "disagg/top1/bench_run.sh",
            "disagg/top1/decode_config.yaml",
            "disagg/top1/generator_config.yaml",
            "disagg/top1/k8s_bench.yaml",
            "disagg/top1/k8s_deploy.yaml",
            "disagg/top1/prefill_config.yaml",
            "disagg/top1/run_0.sh",
            "disagg/top1/sflow.yaml",
            "disagg/top1/spica_candidate.yaml",
        }.issubset(written_names)
        assert result_dir.parent == tmp_path
        assert result_dir.name.startswith("Qwen_Qwen3-32B-FP8_h200_sxm_trtllm_trace_traffic_ttft2000_tpot30_")

        combined_pareto = pd.read_csv(result_dir / "pareto.csv")
        assert set(combined_pareto["deployment_mode"]) == {"agg", "disagg"}
        agg_best = pd.read_csv(result_dir / "agg" / "best_config_topn.csv")
        assert agg_best.loc[0, "tokens/s/gpu"] == pytest.approx(100.0)
        agg_generator_config = yaml.safe_load((result_dir / "agg" / "top1" / "generator_config.yaml").read_text())
        assert agg_generator_config["WorkerConfig"]["agg_workers"] == 1
        assert agg_generator_config["params"]["agg"]["max_batch_size"] == 128
        assert agg_generator_config["params"]["agg"]["max_num_tokens"] == 4096
        assert agg_generator_config["params"]["agg"]["tokens_per_block"] == 64
        assert agg_generator_config["params"]["agg"]["kv_cache_free_gpu_memory_fraction"] == pytest.approx(0.91)
        assert agg_generator_config["params"]["agg"]["disable_prefix_cache"] is False
        assert agg_generator_config["params"]["agg"]["max_seq_len"] == 8192
        assert agg_generator_config["params"]["agg"]["extra_engine_args"]["max_num_tokens"] == 4096
        assert agg_generator_config["ModelConfig"]["nextn"] == 2
        agg_engine_config = yaml.safe_load((result_dir / "agg" / "top1" / "agg_config.yaml").read_text())
        assert agg_engine_config["max_num_tokens"] == 4096
        assert agg_engine_config["max_seq_len"] == 8192
        assert agg_engine_config["kv_cache_config"]["free_gpu_memory_fraction"] == pytest.approx(0.91)
        assert agg_engine_config["kv_cache_config"]["tokens_per_block"] == 64
        assert agg_engine_config["kv_cache_config"]["enable_block_reuse"] is True
        assert agg_engine_config["speculative_config"]["num_nextn_predict_layers"] == 2
        agg_top2_generator_config = yaml.safe_load((result_dir / "agg" / "top2" / "generator_config.yaml").read_text())
        assert agg_top2_generator_config["params"]["agg"]["max_batch_size"] == 64
        assert agg_top2_generator_config["params"]["agg"]["disable_prefix_cache"] is True
        disagg_generator_config = yaml.safe_load((result_dir / "disagg" / "top1" / "generator_config.yaml").read_text())
        assert disagg_generator_config["DynConfig"]["enable_router"] is True
        assert disagg_generator_config["DynConfig"]["router_mode"] == "kv"
        router_config = disagg_generator_config["DynConfig"]["router_config"]
        assert router_config["kv_cache_block_size"] == 32
        assert router_config["overlap_score_credit"] == pytest.approx(0.5)
        assert router_config["prefill_load_scale"] == pytest.approx(2.0)
        assert router_config["router_temperature"] == pytest.approx(0.2)
        assert router_config["admission_control"] == "token-capacity"
        assert router_config["active_decode_blocks_threshold"] == pytest.approx(0.75)
        assert router_config["active_prefill_tokens_threshold"] == 10000
        assert router_config["active_prefill_tokens_threshold_frac"] == pytest.approx(2.0)
        planner_config = disagg_generator_config["DynConfig"]["planner_config"]
        assert planner_config["enable_throughput_scaling"] is True
        assert planner_config["enable_load_scaling"] is True
        assert planner_config["throughput_adjustment_interval_seconds"] == 180
        assert planner_config["load_adjustment_interval_seconds"] == 5
        assert planner_config["max_num_fpm_samples"] == 25
        assert planner_config["fpm_sample_bucket_size"] == 4
        assert planner_config["load_scaling_down_sensitivity"] == 20
        assert planner_config["load_min_observations"] == 6
        assert planner_config["prefill_engine_num_gpu"] == 1
        assert planner_config["decode_engine_num_gpu"] == 2
        assert planner_config["ttft_ms"] == pytest.approx(2000.0)
        assert planner_config["itl_ms"] == pytest.approx(30.0)
        assert disagg_generator_config["DynConfig"]["kvbm_config"]["cpu_cache_override_num_blocks"] == 4096
        assert disagg_generator_config["DynConfig"]["kvbm_config"]["max_transfer_batch_size"] == 32
        assert disagg_generator_config["params"]["prefill"]["max_num_tokens"] == 8192
        assert disagg_generator_config["params"]["decode"]["max_num_tokens"] == 4096
        assert disagg_generator_config["params"]["decode"]["enable_attention_dp"] is True
        assert disagg_generator_config["params"]["decode"]["cache_transceiver_max_tokens_in_buffer"] == 8192
        assert disagg_generator_config["params"]["decode"]["extra_engine_args"]["max_num_tokens"] == 4096
        prefill_engine_config = yaml.safe_load((result_dir / "disagg" / "top1" / "prefill_config.yaml").read_text())
        decode_engine_config = yaml.safe_load((result_dir / "disagg" / "top1" / "decode_config.yaml").read_text())
        assert prefill_engine_config["max_num_tokens"] == 8192
        assert prefill_engine_config["cache_transceiver_config"]["max_tokens_in_buffer"] == 8192
        assert prefill_engine_config["kv_cache_config"]["enable_block_reuse"] is True
        assert decode_engine_config["max_num_tokens"] == 4096
        assert decode_engine_config["enable_attention_dp"] is True
        assert decode_engine_config["cache_transceiver_config"]["max_tokens_in_buffer"] == 8192
        assert decode_engine_config["kv_cache_config"]["enable_block_reuse"] is False
        k8s_deploy = yaml.safe_load((result_dir / "disagg" / "top1" / "k8s_deploy.yaml").read_text())
        services = k8s_deploy["spec"]["services"]
        frontend = services["Frontend"]["extraPodSpec"]["mainContainer"]
        assert frontend["command"] == ["python3", "-m", "dynamo.frontend"]
        assert frontend["args"][:4] == ["--http-port", "8000", "--router-mode", "kv"]
        assert "--router-kv-overlap-score-credit" in frontend["args"]
        assert "0.5" in frontend["args"]
        assert "--router-prefill-load-scale" in frontend["args"]
        assert "2.0" in frontend["args"]
        assert "--router-temperature" in frontend["args"]
        assert "0.2" in frontend["args"]
        assert "--admission-control" in frontend["args"]
        assert "token-capacity" in frontend["args"]
        planner = services["Planner"]["extraPodSpec"]["mainContainer"]
        assert planner["command"] == ["python3", "-m", "dynamo.planner"]
        planner_payload = json.loads(planner["args"][planner["args"].index("--config") + 1])
        assert planner_payload["enable_throughput_scaling"] is True
        assert planner_payload["enable_load_scaling"] is True
        assert planner_payload["prefill_engine_num_gpu"] == 1
        assert planner_payload["decode_engine_num_gpu"] == 2
        prefill_envs = {item["name"]: item["value"] for item in services["TRTLLMPrefillWorker"]["envs"]}
        assert prefill_envs["DYN_KVBM_CPU_CACHE_OVERRIDE_NUM_BLOCKS"] == "4096"
        assert prefill_envs["DYN_KVBM_MAX_TRANSFER_BATCH_SIZE"] == "32"
        decode_script = k8s_deploy["spec"]["services"]["TRTLLMDecodeWorker"]["extraPodSpec"]["mainContainer"]["args"][0]
        assert "max_num_tokens: 4096" in decode_script
        assert "enable_attention_dp: true" in decode_script
        assert "max_tokens_in_buffer: 8192" in decode_script
        sflow_yaml = (result_dir / "disagg" / "top1" / "sflow.yaml").read_text()
        assert "--router-kv-overlap-score-credit 0.5" in sflow_yaml
        assert "export DYN_KVBM_CPU_CACHE_OVERRIDE_NUM_BLOCKS=4096" in sflow_yaml
        assert "max_num_tokens: 4096" in sflow_yaml
        assert "max_tokens_in_buffer: 8192" in sflow_yaml
        run_script = (result_dir / "disagg" / "top1" / "run_0.sh").read_text()
        assert "--router-kv-overlap-score-credit 0.5" in run_script
        assert "export DYN_KVBM_CPU_CACHE_OVERRIDE_NUM_BLOCKS=4096" in run_script
        disagg_candidate = yaml.safe_load((result_dir / "disagg" / "top1" / "spica_candidate.yaml").read_text())
        assert disagg_candidate["config"]["deployment_mode"] == "disagg"

    @pytest.mark.parametrize(
        "builder_patch",
        [
            "aiconfigurator.cli.main.build_default_tasks",
            "aiconfigurator.cli.main.build_experiment_tasks",
        ],
    )
    def test_cli_main_unsupported_mode_raises(self, builder_patch, cli_args_factory):
        """Unsupported mode should cause SystemExit through argparse validation."""
        with patch(builder_patch) as mock_builder:
            mock_builder.return_value = {}
            parser = argparse.ArgumentParser()
            configure_parser(parser)
            with pytest.raises(SystemExit):
                parser.parse_args(["invalid"])
            mock_builder.assert_not_called()

    @pytest.mark.parametrize(
        "builder_patch",
        [
            "aiconfigurator.cli.main.build_default_tasks",
            "aiconfigurator.cli.main.build_experiment_tasks",
        ],
    )
    @patch("aiconfigurator.cli.main._execute_tasks")
    def test_cli_main_runtime_failure(self, mock_execute, builder_patch, cli_args_factory, tmp_path):
        """Execution errors propagate as RuntimeError for visibility."""
        mock_execute.side_effect = RuntimeError("failed")
        mock_task_config = MagicMock(name="TaskConfig")

        with patch(builder_patch) as mock_builder:
            mock_builder.return_value = {"agg": mock_task_config}

            if "default" in builder_patch:
                args = cli_args_factory(mode="default")
            else:
                yaml_file = tmp_path / "exp.yaml"
                yaml_file.write_text("exps: []")
                args = cli_args_factory(mode="exp", extra_args=["--yaml-path", str(yaml_file)])

            with pytest.raises(RuntimeError):
                cli_main(args)

        mock_builder.assert_called_once()
        mock_execute.assert_called_once()

    @pytest.mark.parametrize("database_mode", ["SILICON", "HYBRID", "EMPIRICAL"])
    def test_cli_default_mode_with_database_mode(self, cli_args_factory, database_mode):
        """Test that database_mode is correctly parsed and passed through in default mode."""
        args = cli_args_factory(
            mode="default",
            extra_args=["--database-mode", database_mode],
        )
        assert args.database_mode == database_mode

    def test_execute_tasks_no_feasible_config_logs_without_traceback(
        self,
        caplog,
    ):
        """Strict-SLA no-match should produce a controlled report, not a traceback."""
        mock_task_config = MagicMock(name="Task")
        mock_task_config.to_yaml.return_value = "serving_mode: agg"
        mock_task_config.database_mode = None
        mock_task_config.run.side_effect = NoFeasibleConfigError(
            "No configuration satisfied the TTFT/TPOT constraints."
        )

        with caplog.at_level(logging.WARNING), pytest.raises(SystemExit) as exc_info:
            _execute_tasks({"agg": mock_task_config}, mode="default", strict_sla=True)

        assert exc_info.value.code == 1
        assert "Experiment agg found no SLA-feasible configuration" in caplog.text
        assert "No successful experiment runs to compare." in caplog.text
        assert "Traceback" not in caplog.text
        assert all(record.exc_info is None for record in caplog.records)

    @patch("aiconfigurator.cli.main._execute_tasks")
    @patch("aiconfigurator.cli.main.build_experiment_tasks")
    def test_cli_exp_mode_with_database_mode_in_yaml(self, mock_build_exp, mock_execute, tmp_path):
        """Test that database_mode from YAML is correctly parsed in exp mode."""
        yaml_content = """
exp_with_db_mode:
    serving_mode: "agg"
    model_path: "Qwen/Qwen3-32B"
    system_name: "h200_sxm"
    total_gpus: 8
    database_mode: "HYBRID"
"""
        yaml_file = tmp_path / "exp_db_mode.yaml"
        yaml_file.write_text(yaml_content)

        mock_task_config = MagicMock(name="TaskConfig")
        mock_build_exp.return_value = {"exp_with_db_mode": mock_task_config}
        mock_execute.return_value = ("exp_with_db_mode", {}, {}, {}, {})

        parser = argparse.ArgumentParser()
        configure_parser(parser)
        args = parser.parse_args(["exp", "--yaml-path", str(yaml_file)])

        cli_main(args)

        mock_build_exp.assert_called_once()
        mock_execute.assert_called_once()

    @patch("aiconfigurator.cli.main._execute_tasks")
    @patch("aiconfigurator.cli.main.build_experiment_tasks")
    def test_cli_exp_mode_passes_global_engine_step_backend(
        self,
        mock_build_exp,
        mock_execute,
        cli_args_factory,
        mock_exp_yaml_path,
    ):
        """The shared --engine-step-backend flag should apply to exp mode."""
        mock_build_exp.return_value = {"my_exp": MagicMock(name="TaskConfig")}
        mock_execute.return_value = ("my_exp", {}, {}, {}, {})

        args = cli_args_factory(
            mode="exp",
            extra_args=[
                "--yaml-path",
                str(mock_exp_yaml_path),
                "--engine-step-backend",
                "rust",
            ],
        )

        cli_main(args)

        mock_build_exp.assert_called_once_with(
            yaml_path=str(mock_exp_yaml_path),
            engine_step_backend="rust",
        )
        mock_execute.assert_called_once()


class TestBuildDefaultTaskConfigs:
    """Tests for build_default_tasks function."""

    @patch("aiconfigurator.cli.main.Task")
    def test_skips_disagg_when_total_gpus_less_than_2(self, mock_task_config):
        """Disagg config should be skipped when total_gpus < 2."""
        mock_task_config.return_value = MagicMock(name="MockTaskConfig")

        result = build_default_tasks(
            model_path="Qwen/Qwen3-32B",
            total_gpus=1,
            system="h200_sxm",
        )

        # Should only have agg config, no disagg
        assert "agg" in result
        assert "disagg" not in result
        # TaskConfig should only be called once (for agg)
        assert mock_task_config.call_count == 1

    @patch("aiconfigurator.cli.main.Task")
    def test_includes_disagg_when_total_gpus_at_least_2(self, mock_task_config):
        """Disagg config should be included when total_gpus >= 2."""
        mock_task_config.return_value = MagicMock(name="MockTaskConfig")

        result = build_default_tasks(
            model_path="Qwen/Qwen3-32B",
            total_gpus=2,
            system="h200_sxm",
        )

        # Should have both agg and disagg configs
        assert "agg" in result
        assert "disagg" in result
        # TaskConfig should be called twice (agg + disagg)
        assert mock_task_config.call_count == 2

    @patch("aiconfigurator.cli.main.Task")
    @patch("aiconfigurator.cli.main.perf_database.get_supported_databases")
    def test_silicon_mode_allows_declared_explicit_version_for_shared_layer(
        self,
        mock_supported_databases,
        mock_task_config,
    ):
        """A marker-only version dir is enough to declare shared-layer reuse."""
        mock_supported_databases.return_value = {"b200_sxm": {"sglang": ["0.5.10", "0.5.12"]}}
        mock_task_config.return_value = MagicMock(name="MockTaskConfig")

        result = build_default_tasks(
            model_path="Qwen/Qwen3-0.6B",
            total_gpus=4,
            system="b200_sxm",
            backend="sglang",
            backend_version="0.5.12",
            database_mode="SILICON",
        )

        assert set(result) == {"agg", "disagg"}
        assert mock_task_config.call_count == 2
        assert mock_task_config.call_args_list[0].kwargs["backend_version"] == "0.5.12"
        assert mock_task_config.call_args_list[1].kwargs["prefill_backend_version"] == "0.5.12"
        assert mock_task_config.call_args_list[1].kwargs["decode_backend_version"] == "0.5.12"

    @patch("aiconfigurator.cli.main.Task")
    @patch("aiconfigurator.cli.main.perf_database.get_supported_databases")
    def test_silicon_mode_rejects_undeclared_explicit_version_for_shared_layer(
        self,
        mock_supported_databases,
        mock_task_config,
    ):
        """Sibling data does not make an arbitrary framework version supported."""
        mock_supported_databases.return_value = {"b200_sxm": {"sglang": ["0.5.10"]}}
        mock_task_config.return_value = MagicMock(name="MockTaskConfig")

        with pytest.raises(SystemExit):
            build_default_tasks(
                model_path="Qwen/Qwen3-0.6B",
                total_gpus=4,
                system="b200_sxm",
                backend="sglang",
                backend_version="0.5.12",
                database_mode="SILICON",
            )

        mock_task_config.assert_not_called()

    @patch("aiconfigurator.cli.main.Task")
    @patch("aiconfigurator.cli.main.perf_database.get_supported_databases")
    def test_auto_hybrid_mode_filters_to_declared_shared_layer_versions(
        self,
        mock_supported_databases,
        mock_task_config,
    ):
        """Auto backend sweeps in HYBRID should only include declared framework versions."""
        mock_supported_databases.return_value = {
            "b200_sxm": {
                "trtllm": ["1.3.0rc10"],
                "sglang": ["0.5.10", "0.5.12"],
                "vllm": ["0.19.0"],
            }
        }
        mock_task_config.return_value = MagicMock(name="MockTaskConfig")

        result = build_default_tasks(
            model_path="Qwen/Qwen3-0.6B",
            total_gpus=4,
            system="b200_sxm",
            backend="auto",
            backend_version="0.5.12",
            database_mode="HYBRID",
        )

        assert set(result) == {"agg_sglang", "disagg_sglang"}
        assert mock_task_config.call_count == 2
        assert mock_task_config.call_args_list[0].kwargs["backend_version"] == "0.5.12"
        assert mock_task_config.call_args_list[1].kwargs["prefill_backend_version"] == "0.5.12"
        assert mock_task_config.call_args_list[1].kwargs["decode_backend_version"] == "0.5.12"

    @patch("aiconfigurator.cli.main.Task")
    @patch("aiconfigurator.cli.main.perf_database.get_supported_databases")
    def test_hybrid_mode_rejects_undeclared_explicit_version_for_shared_layer(
        self,
        mock_supported_databases,
        mock_task_config,
    ):
        """HYBRID also uses shared-layer data, so arbitrary framework versions are rejected."""
        mock_supported_databases.return_value = {"b200_sxm": {"sglang": ["0.5.10"]}}
        mock_task_config.return_value = MagicMock(name="MockTaskConfig")

        with pytest.raises(SystemExit):
            build_default_tasks(
                model_path="Qwen/Qwen3-0.6B",
                total_gpus=4,
                system="b200_sxm",
                backend="sglang",
                backend_version="0.5.12",
                database_mode="HYBRID",
            )

        mock_task_config.assert_not_called()

    @patch("aiconfigurator.cli.main.Task")
    @patch("aiconfigurator.cli.main.perf_database.get_supported_databases")
    def test_auto_megamoe_sweeps_only_sglang(self, mock_supported_databases, mock_task_config):
        """The SGLang-only MegaMoE override must not be passed to TRT-LLM or vLLM."""
        mock_supported_databases.return_value = {
            "gb200": {
                "trtllm": ["0.5.10"],
                "sglang": ["0.5.10"],
                "vllm": ["0.5.10"],
            }
        }
        mock_task_config.return_value = MagicMock(name="MockTaskConfig")

        result = build_default_tasks(
            model_path="deepseek-ai/DeepSeek-V4-Pro",
            total_gpus=2,
            system="gb200",
            backend="auto",
            backend_version="0.5.10",
            moe_backend="megamoe",
        )

        assert set(result) == {"agg_sglang", "disagg_sglang"}
        assert mock_task_config.call_count == 2
        for call in mock_task_config.call_args_list:
            # agg passes backend_name; disagg fans out to prefill_/decode_backend_name.
            backend = call.kwargs.get("backend_name") or call.kwargs.get("prefill_backend_name")
            assert backend == "sglang"
            assert call.kwargs["moe_backend"] == "megamoe"

    @patch("aiconfigurator.cli.main.check_is_moe", return_value=True)
    @patch("aiconfigurator.cli.main.Task")
    def test_skips_optional_sglang_deepep_when_perf_data_missing(
        self,
        mock_task_config,
        _mock_check_is_moe,
        tmp_path,
        caplog,
    ):
        """Optional SGLang DeepEP sweeps should not be scheduled without DeepEP op data."""
        mock_task_config.return_value = MagicMock(name="MockTaskConfig")
        caplog.set_level(logging.INFO, logger="aiconfigurator.cli.main")

        with patch("aiconfigurator.cli.main._get_backend_data_path", return_value=str(tmp_path)):
            result = build_default_tasks(
                model_path="deepseek-ai/DeepSeek-R1",
                total_gpus=8,
                system="b200_sxm",
                backend="sglang",
                backend_version="0.5.10",
                database_mode="HYBRID",
            )

        assert set(result) == {"agg", "disagg"}
        assert mock_task_config.call_count == 2
        assert "Skipping SGLang DeepEP agg sweep" in caplog.text
        assert "Skipping SGLang DeepEP disagg sweep" in caplog.text
        assert "wideep_deepep_normal_perf.parquet" in caplog.text

    @patch("aiconfigurator.cli.main.check_is_moe", return_value=True)
    @patch("aiconfigurator.cli.main.Task")
    def test_includes_optional_sglang_deepep_when_perf_data_exists(
        self,
        mock_task_config,
        _mock_check_is_moe,
        tmp_path,
    ):
        """SGLang DeepEP sweeps remain available when required DeepEP op data exists."""
        mock_task_config.return_value = MagicMock(name="MockTaskConfig")
        for filename in ("wideep_deepep_normal_perf.parquet", "wideep_deepep_ll_perf.parquet"):
            (tmp_path / filename).write_text("header\n", encoding="utf-8")

        with patch("aiconfigurator.cli.main._get_backend_data_path", return_value=str(tmp_path)):
            result = build_default_tasks(
                model_path="deepseek-ai/DeepSeek-R1",
                total_gpus=8,
                system="h100_sxm",
                backend="sglang",
                backend_version="0.5.6.post2",
            )

        assert set(result) == {"agg", "agg_deepep", "disagg", "disagg_deepep"}
        assert mock_task_config.call_count == 4


class TestBuildExperimentTaskConfigs:
    """Tests for experiment config construction."""

    @patch("aiconfigurator.cli.main.Task")
    def test_global_engine_step_backend_applies_unless_exp_overrides(self, mock_task):
        config = {
            "global_backend": {
                "serving_mode": "agg",
                "model_path": "Qwen/Qwen3-32B",
                "system_name": "h200_sxm",
                "total_gpus": 8,
            },
            "exp_backend": {
                "serving_mode": "agg",
                "model_path": "Qwen/Qwen3-32B",
                "system_name": "h200_sxm",
                "total_gpus": 8,
                "engine_step_backend": "python",
            },
        }

        build_experiment_tasks(config=config, engine_step_backend="rust")

        # build_experiment calls Task.from_yaml(exp_config, **overrides): the global
        # default arrives via the overrides kwarg, a per-exp value stays in exp_config.
        by_backend = {}
        for call in mock_task.from_yaml.call_args_list:
            yaml_data = call.args[0]
            esb = call.kwargs.get("engine_step_backend", yaml_data.get("engine_step_backend"))
            by_backend[esb] = yaml_data
        assert set(by_backend) == {"rust", "python"}
        assert by_backend["rust"]["model_path"] == "Qwen/Qwen3-32B"
        assert by_backend["python"]["model_path"] == "Qwen/Qwen3-32B"

    @patch("aiconfigurator.cli.main.Task")
    def test_default_database_mode_is_passed_to_task_yaml(self, mock_task):
        config = {
            "default_mode": {
                "serving_mode": "agg",
                "model_path": "Qwen/Qwen3-32B",
                "system_name": "h200_sxm",
                "total_gpus": 8,
            }
        }

        build_experiment_tasks(config=config)

        yaml_data = mock_task.from_yaml.call_args.args[0]
        assert yaml_data["database_mode"] == "SILICON"


class TestInclusiveTpot:
    """Unit tests for _apply_inclusive_tpot output transformation."""

    def _make_df(self, ttft, tpot, osl):
        return pd.DataFrame([{"ttft": ttft, "tpot": tpot, "osl": osl}])

    def test_formula(self):
        df = self._make_df(ttft=500.0, tpot=20.0, osl=30)
        result = _apply_inclusive_tpot(df)
        expected = (500.0 + 20.0 * 29) / 30
        assert abs(result["tpot"].iloc[0] - expected) < 1e-9

    def test_does_not_mutate_input(self):
        df = self._make_df(ttft=500.0, tpot=20.0, osl=30)
        original_tpot = df["tpot"].iloc[0]
        _apply_inclusive_tpot(df)
        assert df["tpot"].iloc[0] == original_tpot

    def test_ttft_unchanged(self):
        df = self._make_df(ttft=500.0, tpot=20.0, osl=30)
        result = _apply_inclusive_tpot(df)
        assert result["ttft"].iloc[0] == 500.0

    def test_missing_columns_is_noop(self):
        df = pd.DataFrame([{"tpot": 20.0, "osl": 30}])  # no ttft column
        result = _apply_inclusive_tpot(df)
        assert result["tpot"].iloc[0] == 20.0

    def test_osl_one_equals_ttft(self):
        # osl=1: zero decode tokens, inclusive TPOT collapses to TTFT/1 = TTFT
        df = self._make_df(ttft=500.0, tpot=20.0, osl=1)
        result = _apply_inclusive_tpot(df)
        assert abs(result["tpot"].iloc[0] - 500.0) < 1e-9
