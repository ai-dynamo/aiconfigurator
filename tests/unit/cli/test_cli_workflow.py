# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unit-level workflow tests for CLI wiring.

These tests validate the wiring between CLI entrypoints and internal builders/executors,
while keeping heavy computation mocked out.
"""

import argparse
import logging
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from aiconfigurator.cli.main import (
    _build_spica_trace_search_space,
    _execute_tasks,
    _format_spica_trace_summary,
    _resolve_cli_log_level,
    _save_spica_trace_artifacts,
    _spica_candidates_to_pareto_df,
    build_default_tasks,
    build_experiment_tasks,
    configure_parser,
)
from aiconfigurator.cli.main import main as cli_main
from aiconfigurator.cli.report_and_save import _apply_inclusive_tpot
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

    @patch("aiconfigurator.cli.main._run_spica_trace_default")
    @patch("aiconfigurator.cli.main._execute_tasks")
    @patch("aiconfigurator.cli.main.build_default_tasks")
    def test_cli_default_trace_path_dispatches_to_spica(
        self,
        mock_build_default,
        mock_execute,
        mock_run_spica,
        cli_args_factory,
    ):
        """default --trace-path should bypass legacy AIC tasks and run Spica."""
        args = cli_args_factory(mode="default", trace_path="/tmp/traffic.jsonl")

        cli_main(args)

        mock_run_spica.assert_called_once_with(args)
        mock_build_default.assert_not_called()
        mock_execute.assert_not_called()

    def test_spica_trace_search_space_collapses_single_gpu_noops(self, cli_args_factory):
        """Single-GPU trace sweeps should avoid routing/planner choices that cannot help."""
        args = cli_args_factory(
            mode="default",
            total_gpus=1,
            trace_path="/tmp/traffic.jsonl",
            max_seq_len=8192,
        )

        search_space = _build_spica_trace_search_space(args, ["trtllm"])

        assert search_space["gpu_budget"] == 1
        assert search_space["context_length"] == 8192
        assert search_space["deployment_mode"] == ["agg"]
        assert search_space["router_mode"] == ["round_robin"]
        assert search_space["planner_scaling_policy"] == ["disabled"]

    def test_spica_trace_summary_uses_default_result_shape(self):
        """Spica trace output should resemble the existing default-mode final summary."""
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

        summary = _format_spica_trace_summary(
            candidates,
            top_n=5,
            model_path="Qwen/Qwen3-32B-FP8",
            total_gpus=16,
            trace_path="/data/replay/traffic.jsonl",
            ttft=2000.0,
            tpot=30.0,
        )

        assert "AIConfigurator Final Results" in summary
        assert "Input Configuration & SLA Target" in summary
        assert "Overall Best Configuration" in summary
        assert "Pareto Frontier:" in summary
        assert "Spica Trace Pareto Frontier" in summary
        assert "Deployment Details" in summary
        assert "Trace: /data/replay/traffic.jsonl (Mooncake JSONL)" in summary
        assert "Best Experiment Chosen:" in summary
        assert "disagg at 321.50 goodput/s/gpu" in summary
        assert "agg Top Configurations: (Sorted by goodput/s/gpu)" in summary
        assert "disagg Top Configurations: (Sorted by goodput/s/gpu)" in summary
        assert "goodput/s/gpu" in summary
        assert "(p)parallel" in summary
        assert "kv_router" in summary

    def test_spica_trace_artifacts_include_pareto_outputs(self, tmp_path):
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
                    "agg_max_num_batched_tokens": 4096,
                    "agg_max_num_seqs": 128,
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
                    "tp": 1,
                    "pp": 1,
                    "attention_dp": 1,
                    "replicas": 1,
                    "agg_max_num_batched_tokens": 2048,
                    "agg_max_num_seqs": 64,
                    "router_mode": "round_robin",
                    "planner_scaling_policy": "disabled",
                },
                "used_gpus": 1,
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
                    "decode_replicas": 1,
                    "decode_tp": 1,
                    "decode_pp": 1,
                    "decode_attention_dp": 1,
                    "router_mode": "kv_router",
                    "planner_scaling_policy": "predictive",
                    "enable_load_scaling": True,
                },
                "used_gpus": 2,
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

        pareto_df = _spica_candidates_to_pareto_df(candidates)
        assert pareto_df["tokens/s/user"].tolist() == pytest.approx([100.0, 50.0, 80.0])
        assert pareto_df["goodput/s/gpu"].tolist() == pytest.approx([100.0, 80.0, 120.0])

        written_paths = _save_spica_trace_artifacts(candidates, str(tmp_path), top_n=1)
        written_names = {str(path).replace(f"{tmp_path}/", "") for path in written_paths}

        assert {
            "spica_candidates.yaml",
            "spica_candidates.csv",
            "pareto.csv",
            "pareto_frontier.png",
            "agg/pareto.csv",
            "agg/best_config_topn.csv",
            "disagg/pareto.csv",
            "disagg/best_config_topn.csv",
        }.issubset(written_names)

        combined_pareto = pd.read_csv(tmp_path / "pareto.csv")
        assert set(combined_pareto["deployment_mode"]) == {"agg", "disagg"}
        agg_best = pd.read_csv(tmp_path / "agg" / "best_config_topn.csv")
        assert agg_best.loc[0, "goodput/s/gpu"] == pytest.approx(100.0)

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
