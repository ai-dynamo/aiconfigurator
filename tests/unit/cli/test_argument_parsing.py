# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for CLI argument parsing functionality.

Tests CLI argument validation, choices, and default values.
"""

import typing

import pytest

from aiconfigurator.sdk import common

pytestmark = pytest.mark.unit


class TestCLIArgumentParsing:
    """Test CLI argument parsing and validation."""

    def test_default_mode_core_args_are_conditionally_required(self, cli_parser):
        """Default core args are validated after parsing so --thorough-config can stand alone."""
        subparsers = [action for action in cli_parser._actions if action.dest == "mode"]
        assert len(subparsers) == 1

        subparser_action = subparsers[0]
        default_parser = subparser_action.choices["default"]

        required_actions = [action for action in default_parser._actions if getattr(action, "required", False)]
        required_args = [action.dest for action in required_actions]

        assert "model_path" not in required_args
        assert "total_gpus" not in required_args
        assert "system" not in required_args

    def test_exp_mode_required_args(self, cli_parser):
        """Test that exp mode requires the yaml_path argument."""
        subparsers = [action for action in cli_parser._actions if action.dest == "mode"]
        assert len(subparsers) == 1

        subparser_action = subparsers[0]
        exp_parser = subparser_action.choices["exp"]

        required_actions = [action for action in exp_parser._actions if getattr(action, "required", False)]
        required_args = [action.dest for action in required_actions]

        assert "yaml_path" in required_args

    def test_mode_choices(self, cli_parser):
        """Ensure supported CLI modes are exposed."""
        action = next(action for action in cli_parser._actions if action.dest == "mode")
        assert set(action.choices.keys()) == {"default", "exp", "generate", "support", "estimate"}

    def test_generate_mode_required_args(self, cli_parser):
        """Test that generate mode requires the correct arguments."""
        subparsers = [action for action in cli_parser._actions if action.dest == "mode"]
        assert len(subparsers) == 1

        subparser_action = subparsers[0]
        generate_parser = subparser_action.choices["generate"]

        required_actions = [action for action in generate_parser._actions if getattr(action, "required", False)]
        required_args = [action.dest for action in required_actions]

        assert "model_path" in required_args
        assert "total_gpus" in required_args
        assert "system" in required_args

    def test_generate_mode_defaults(self, cli_parser):
        """Test that generate mode has correct defaults."""
        args = cli_parser.parse_args(
            ["generate", "--model-path", "Qwen/Qwen3-32B", "--total-gpus", "8", "--system", "h200_sxm"]
        )
        assert args.mode == "generate"
        assert args.model_path == "Qwen/Qwen3-32B"
        assert args.backend == common.BackendName.trtllm.value

    def test_generate_mode_model_path(self, cli_parser):
        """Test that generate mode accepts model_path."""
        args = cli_parser.parse_args(
            ["generate", "--model-path", "Qwen/Qwen3-8B", "--total-gpus", "8", "--system", "h200_sxm"]
        )
        assert args.model_path == "Qwen/Qwen3-8B"

    def test_backend_choices_validation(self, cli_parser):
        """Test that backend argument validates against supported choices."""
        subparser_action = next(action for action in cli_parser._actions if action.dest == "mode")
        default_parser = subparser_action.choices["default"]
        action = next(action for action in default_parser._actions if action.dest == "backend")
        expected_choices = [backend.value for backend in common.BackendName] + ["auto"]
        assert sorted(action.choices) == sorted(expected_choices)

    @pytest.mark.parametrize("system_value", ["h200_sxm", "b200_sxm", "gb200"])
    def test_supported_systems_parse_successfully(self, cli_parser, system_value):
        """System flag should accept supported platforms including b200 and gb200."""
        args = cli_parser.parse_args(
            ["default", "--model-path", "Qwen/Qwen3-32B", "--total-gpus", "16", "--system", system_value]
        )

        assert args.system == system_value

    def test_default_values_are_set(self, cli_parser):
        """Test that default values are properly set for optional arguments."""
        args = cli_parser.parse_args(
            ["default", "--model-path", "Qwen/Qwen3-32B", "--total-gpus", "8", "--system", "h200_sxm"]
        )

        assert args.backend == common.BackendName.trtllm.value
        assert args.backend_version is None
        assert args.database_mode == common.DatabaseMode.SILICON.name
        assert args.log_level is None
        assert args.decode_system is None
        assert args.generated_config_version is None
        assert args.generator_dynamo_version is None
        assert args.isl == 4000
        assert args.osl == 1000
        assert args.save_dir is None
        assert args.ttft == 2000.0
        assert args.tpot == 30.0
        assert args.request_latency is None
        assert args.thorough_sweep is False
        assert args.thorough_config is None
        assert args.inclusive_tpot is False
        assert args.prefix == 0
        assert args.engine_step_backend is None

    def test_default_trace_path_is_not_public_cli(self, cli_parser):
        """Trace replay should be configured through --thorough-config, not a single-format CLI shortcut."""
        with pytest.raises(SystemExit):
            cli_parser.parse_args(
                [
                    "default",
                    "--model-path",
                    "Qwen/Qwen3-32B",
                    "--total-gpus",
                    "8",
                    "--system",
                    "h200_sxm",
                    "--trace-path",
                    "/tmp/traffic.jsonl",
                ]
            )

    def test_default_thorough_sweep_parses(self, cli_parser):
        """--thorough-sweep selects Spica while still using regular default inputs."""
        args = cli_parser.parse_args(
            [
                "default",
                "--model-path",
                "Qwen/Qwen3-32B",
                "--total-gpus",
                "8",
                "--system",
                "h200_sxm",
                "--thorough-sweep",
            ]
        )

        assert args.thorough_sweep is True
        assert args.thorough_config is None

    def test_default_thorough_config_parses_without_legacy_required_args(self, cli_parser):
        """A native Spica config owns model/system/GPU inputs."""
        args = cli_parser.parse_args(["default", "--thorough-config", "/tmp/spica.yaml"])

        assert args.thorough_config == "/tmp/spica.yaml"
        assert args.model_path is None
        assert args.total_gpus is None
        assert args.system is None

    @pytest.mark.parametrize("flag", ["--trace-sweep-rounds", "--trace-parallel-evals"])
    def test_default_trace_tuning_flags_are_not_public_cli(self, cli_parser, flag):
        """Trace sweep tuning is intentionally configured by Spica config or internal env defaults."""
        with pytest.raises(SystemExit):
            cli_parser.parse_args(
                [
                    "default",
                    "--thorough-config",
                    "/tmp/spica.yaml",
                    flag,
                    "5",
                ]
            )

    def test_inclusive_tpot_default_false_in_exp_mode(self, cli_parser, mock_exp_yaml_path):
        """--inclusive-tpot defaults to False in exp mode."""
        args = cli_parser.parse_args(["exp", "--yaml-path", str(mock_exp_yaml_path)])
        assert args.inclusive_tpot is False

    def test_inclusive_tpot_enabled_in_exp_mode(self, cli_parser, mock_exp_yaml_path):
        """--inclusive-tpot can be set in exp mode."""
        args = cli_parser.parse_args(["exp", "--yaml-path", str(mock_exp_yaml_path), "--inclusive-tpot"])
        assert args.inclusive_tpot is True

    def test_log_level_flag(self, cli_parser):
        """Test that --log-level is parsed and normalized."""
        args = cli_parser.parse_args(
            [
                "default",
                "--model-path",
                "Qwen/Qwen3-32B",
                "--total-gpus",
                "8",
                "--system",
                "h200_sxm",
                "--log-level",
                "debug",
            ]
        )

        assert args.log_level == "DEBUG"

    def test_legacy_debug_flag(self, cli_parser):
        """Legacy --debug is still accepted for backward compatibility."""
        args = cli_parser.parse_args(
            [
                "default",
                "--model-path",
                "Qwen/Qwen3-32B",
                "--total-gpus",
                "8",
                "--system",
                "h200_sxm",
                "--debug",
            ]
        )

        assert args.debug is True

    def test_engine_step_backend_flag(self, cli_parser):
        """Test that the experimental engine step backend can be selected."""
        args = cli_parser.parse_args(
            [
                "default",
                "--model-path",
                "Qwen/Qwen3-32B",
                "--total-gpus",
                "8",
                "--system",
                "h200_sxm",
                "--engine-step-backend",
                "rust",
            ]
        )

        assert args.engine_step_backend == "rust"

    def test_save_directory_argument(self, cli_parser):
        """Test that save directory can be specified."""
        args = cli_parser.parse_args(
            [
                "default",
                "--model-path",
                "Qwen/Qwen3-32B",
                "--total-gpus",
                "8",
                "--system",
                "h200_sxm",
                "--save-dir",
                "/tmp/test",
            ]
        )

        assert args.save_dir == "/tmp/test"

    @pytest.mark.parametrize(
        "optional_param,value,expected_type",
        [
            ("isl", "8000", int),
            ("osl", "2048", int),
            ("ttft", "300.0", float),
            ("tpot", "10.0", float),
            ("request_latency", "1200.0", float),
            ("prefix", "128", int),
            ("nextn", "3", int),
        ],
    )
    def test_optional_parameters(self, cli_parser, optional_param, value, expected_type):
        """Test that optional parameters can be set and have correct types."""
        args = cli_parser.parse_args(
            [
                "default",
                "--model-path",
                "Qwen/Qwen3-32B",
                "--total-gpus",
                "8",
                "--system",
                "h200_sxm",
                f"--{optional_param.replace('_', '-')}",
                value,
            ]
        )

        param_value = getattr(args, optional_param)
        assert isinstance(param_value, expected_type)
        assert param_value == expected_type(value)

    def test_decode_system_defaults_to_system(self, cli_parser):
        """Decode system defaults to system when omitted and can be overridden."""
        args = cli_parser.parse_args(
            ["default", "--model-path", "Qwen/Qwen3-32B", "--total-gpus", "8", "--system", "h200_sxm"]
        )
        assert args.decode_system is None

        args_with_decode = cli_parser.parse_args(
            [
                "default",
                "--model-path",
                "Qwen/Qwen3-32B",
                "--total-gpus",
                "8",
                "--system",
                "h200_sxm",
                "--decode-system",
                "gb200",
            ]
        )
        assert args_with_decode.decode_system == "gb200"

    def test_model_path_accepts_huggingface_id(self, cli_parser):
        """Test that --model-path accepts a HuggingFace model ID."""
        args = cli_parser.parse_args(
            [
                "default",
                "--model-path",
                "Qwen/Qwen3-8B",
                "--total-gpus",
                "8",
                "--system",
                "h200_sxm",
            ]
        )
        assert args.model_path == "Qwen/Qwen3-8B"

    @pytest.mark.parametrize(
        "database_mode_value",
        ["SILICON", "HYBRID", "EMPIRICAL", "SOL"],
    )
    def test_database_mode_values_parse_successfully(self, cli_parser, database_mode_value):
        """Database mode flag should accept all supported mode values."""
        args = cli_parser.parse_args(
            [
                "default",
                "--model-path",
                "Qwen/Qwen3-32B",
                "--total-gpus",
                "8",
                "--system",
                "h200_sxm",
                "--database-mode",
                database_mode_value,
            ]
        )
        assert args.database_mode == database_mode_value

    def test_database_mode_invalid_value_raises(self, cli_parser):
        """Test that invalid database_mode value raises an error."""
        with pytest.raises(SystemExit):
            cli_parser.parse_args(
                [
                    "default",
                    "--model-path",
                    "Qwen/Qwen3-32B",
                    "--total-gpus",
                    "8",
                    "--system",
                    "h200_sxm",
                    "--database-mode",
                    "INVALID_MODE",
                ]
            )

    def test_database_mode_choices_validation(self, cli_parser):
        """Test that database_mode argument validates against supported choices."""
        subparser_action = next(action for action in cli_parser._actions if action.dest == "mode")
        default_parser = subparser_action.choices["default"]
        action = next(action for action in default_parser._actions if action.dest == "database_mode")
        expected_choices = [mode.name for mode in common.DatabaseMode if mode != common.DatabaseMode.SOL_FULL]
        assert sorted(action.choices) == sorted(expected_choices)

    def test_nextn_default_value(self, cli_parser):
        """Test that --nextn defaults to 0."""
        args = cli_parser.parse_args(
            ["default", "--model-path", "Qwen/Qwen3-32B", "--total-gpus", "8", "--system", "h200_sxm"]
        )
        assert args.nextn == 0

    def test_nextn_accept_rates_default_value(self, cli_parser):
        """Test that --nextn-accept-rates defaults to '0.85,0.3,0,0,0'."""
        args = cli_parser.parse_args(
            ["default", "--model-path", "Qwen/Qwen3-32B", "--total-gpus", "8", "--system", "h200_sxm"]
        )
        assert args.nextn_accept_rates == "0.85,0.3,0,0,0"

    def test_nextn_can_be_set(self, cli_parser):
        """Test that --nextn can be set to a custom value."""
        args = cli_parser.parse_args(
            [
                "default",
                "--model-path",
                "Qwen/Qwen3-32B",
                "--total-gpus",
                "8",
                "--system",
                "h200_sxm",
                "--nextn",
                "3",
            ]
        )
        assert args.nextn == 3

    def test_nextn_accept_rates_can_be_set(self, cli_parser):
        """Test that --nextn-accept-rates can be set to custom values."""
        args = cli_parser.parse_args(
            [
                "default",
                "--model-path",
                "Qwen/Qwen3-32B",
                "--total-gpus",
                "8",
                "--system",
                "h200_sxm",
                "--nextn-accept-rates",
                "0.9,0.5,0.2,0.1,0",
            ]
        )
        assert args.nextn_accept_rates == "0.9,0.5,0.2,0.1,0"


class TestPercentileSlaArguments:
    """Percentile SLA flags (queueing model): parsing, defaults, rejection."""

    _BASE: typing.ClassVar[list] = [
        "default",
        "--model-path",
        "Qwen/Qwen3-32B",
        "--total-gpus",
        "8",
        "--system",
        "h200_sxm",
    ]

    def test_defaults_keep_legacy_filter_semantics(self, cli_parser):
        """All percentile args default to None (and --sla-refine to False):
        main() derives sla_percentile from their presence, so bare
        invocations must stay on the legacy avg filter."""
        args = cli_parser.parse_args(self._BASE)
        assert args.itl is None
        assert args.ttft_percentile is None
        assert args.tpot_percentile is None
        assert args.itl_percentile is None
        assert args.request_latency_percentile is None
        assert args.sla_refine is False

    def test_percentile_labels_parse_to_floats(self, cli_parser):
        args = cli_parser.parse_args(
            self._BASE + ["--ttft-percentile", "p99", "--itl", "100", "--itl-percentile", "p50"]
        )
        assert args.ttft_percentile == 0.99
        assert args.itl == 100.0
        assert args.itl_percentile == 0.5

    def test_percentile_labels_case_insensitive(self, cli_parser):
        args = cli_parser.parse_args(self._BASE + ["--ttft-percentile", "P95"])
        assert args.ttft_percentile == 0.95

    def test_invalid_percentile_rejected(self, cli_parser):
        with pytest.raises(SystemExit):
            cli_parser.parse_args(self._BASE + ["--ttft-percentile", "p42"])

    def test_sla_refine_flag(self, cli_parser):
        args = cli_parser.parse_args(self._BASE + ["--sla-refine"])
        assert args.sla_refine is True
