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
    _execute_task_configs,
    _load_default_yaml_config,
    build_default_task_configs,
    build_experiment_task_configs,
    configure_parser,
)
from aiconfigurator.cli.main import main as cli_main
from aiconfigurator.cli.report_and_save import _apply_inclusive_tpot
from aiconfigurator.sdk.errors import NoFeasibleConfigError

pytestmark = pytest.mark.unit


class TestCLIIntegration:
    """Workflow tests for the CLI orchestration layer (builders/executor/save)."""

    @patch("aiconfigurator.cli.main._execute_task_configs")
    @patch("aiconfigurator.cli.main.build_default_task_configs")
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
        assert save_kwargs["task_configs"] == {"agg": mock_task_config}
        assert save_kwargs["save_dir"] == sample_cli_args_with_save_dir.save_dir

    @patch("aiconfigurator.cli.main.save_results")
    @patch("aiconfigurator.cli.main._execute_task_configs")
    @patch("aiconfigurator.cli.main.build_experiment_task_configs")
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
        assert save_kwargs["task_configs"] == {"my_exp": mock_task_config}
        assert save_kwargs["save_dir"] == str(mock_exp_yaml_path.parent)

    @pytest.mark.parametrize(
        "mode,build_patch",
        [
            ("default", "aiconfigurator.cli.main.build_default_task_configs"),
            ("exp", "aiconfigurator.cli.main.build_experiment_task_configs"),
        ],
    )
    @patch("aiconfigurator.cli.main._execute_task_configs")
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

    @pytest.mark.parametrize(
        "builder_patch",
        [
            "aiconfigurator.cli.main.build_default_task_configs",
            "aiconfigurator.cli.main.build_experiment_task_configs",
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
            "aiconfigurator.cli.main.build_default_task_configs",
            "aiconfigurator.cli.main.build_experiment_task_configs",
        ],
    )
    @patch("aiconfigurator.cli.main._execute_task_configs")
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

    @patch("aiconfigurator.cli.main.TaskRunner")
    def test_execute_task_configs_no_feasible_config_logs_without_traceback(
        self,
        mock_task_runner_class,
        caplog,
    ):
        """Strict-SLA no-match should produce a controlled report, not a traceback."""
        mock_runner = mock_task_runner_class.return_value
        mock_runner.run.side_effect = NoFeasibleConfigError("No configuration satisfied the TTFT/TPOT constraints.")
        mock_task_config = MagicMock(name="TaskConfig")
        mock_task_config.to_yaml.return_value = "serving_mode: agg"
        mock_task_config.database_mode = None

        with caplog.at_level(logging.WARNING), pytest.raises(SystemExit) as exc_info:
            _execute_task_configs({"agg": mock_task_config}, mode="default", strict_sla=True)

        assert exc_info.value.code == 1
        assert "Experiment agg found no SLA-feasible configuration" in caplog.text
        assert "No successful experiment runs to compare." in caplog.text
        assert "Traceback" not in caplog.text
        assert all(record.exc_info is None for record in caplog.records)

    @patch("aiconfigurator.cli.main._execute_task_configs")
    @patch("aiconfigurator.cli.main.build_experiment_task_configs")
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

    @patch("aiconfigurator.cli.main._execute_task_configs")
    @patch("aiconfigurator.cli.main.build_experiment_task_configs")
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
    """Tests for build_default_task_configs function."""

    def test_inline_default_config_yaml_is_loaded(self):
        """Inline YAML/JSON patches should load like config files."""
        loaded = _load_default_yaml_config(
            None,
            '{"mode":"patch","config":{"worker_config":{"pp_list":[2]}}}',
        )

        assert loaded == {"mode": "patch", "config": {"worker_config": {"pp_list": [2]}}}

    def test_default_config_yaml_rejects_path_and_inline(self, tmp_path):
        """Default-mode config input should be unambiguous."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("config: {}\n")

        with pytest.raises(ValueError, match="Use only one"):
            _load_default_yaml_config(str(config_path), "{}")

    @patch("aiconfigurator.cli.main.TaskConfig")
    def test_default_config_yaml_is_applied_to_agg_and_disagg(self, mock_task_config):
        """Default-mode config YAML patches should flow into generated TaskConfigs."""
        mock_task_config.return_value = MagicMock(name="MockTaskConfig")
        yaml_config = {"mode": "patch", "config": {"worker_config": {"pp_list": [2]}}}

        build_default_task_configs(
            model_path="zai-org/GLM-5",
            total_gpus=128,
            system="b200_sxm",
            yaml_config=yaml_config,
        )

        assert mock_task_config.call_count == 2
        for call in mock_task_config.call_args_list:
            assert call.kwargs["yaml_config"] == yaml_config

    @patch("aiconfigurator.cli.main.TaskConfig")
    def test_skips_disagg_when_total_gpus_less_than_2(self, mock_task_config):
        """Disagg config should be skipped when total_gpus < 2."""
        mock_task_config.return_value = MagicMock(name="MockTaskConfig")

        result = build_default_task_configs(
            model_path="Qwen/Qwen3-32B",
            total_gpus=1,
            system="h200_sxm",
        )

        # Should only have agg config, no disagg
        assert "agg" in result
        assert "disagg" not in result
        # TaskConfig should only be called once (for agg)
        assert mock_task_config.call_count == 1

    @patch("aiconfigurator.cli.main.TaskConfig")
    def test_includes_disagg_when_total_gpus_at_least_2(self, mock_task_config):
        """Disagg config should be included when total_gpus >= 2."""
        mock_task_config.return_value = MagicMock(name="MockTaskConfig")

        result = build_default_task_configs(
            model_path="Qwen/Qwen3-32B",
            total_gpus=2,
            system="h200_sxm",
        )

        # Should have both agg and disagg configs
        assert "agg" in result
        assert "disagg" in result
        # TaskConfig should be called twice (agg + disagg)
        assert mock_task_config.call_count == 2

    @patch("aiconfigurator.cli.main.TaskConfig")
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

        result = build_default_task_configs(
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
            assert call.kwargs["backend_name"] == "sglang"
            assert call.kwargs["moe_backend"] == "megamoe"

    @patch("aiconfigurator.cli.main.check_is_moe", return_value=True)
    @patch("aiconfigurator.cli.main.TaskConfig")
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
            result = build_default_task_configs(
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
    @patch("aiconfigurator.cli.main.TaskConfig")
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
            result = build_default_task_configs(
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

    @patch("aiconfigurator.cli.main.TaskConfig")
    def test_global_engine_step_backend_applies_unless_exp_overrides(self, mock_task_config):
        mock_task_config.side_effect = lambda **kwargs: MagicMock(**kwargs)
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

        build_experiment_task_configs(config=config, engine_step_backend="rust")

        kwargs_by_backend = {
            call.kwargs["engine_step_backend"]: call.kwargs for call in mock_task_config.call_args_list
        }
        assert set(kwargs_by_backend) == {"rust", "python"}
        assert kwargs_by_backend["rust"]["model_path"] == "Qwen/Qwen3-32B"
        assert kwargs_by_backend["python"]["model_path"] == "Qwen/Qwen3-32B"


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
