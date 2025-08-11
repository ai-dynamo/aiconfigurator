# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for CLI argument parsing functionality.

Tests CLI argument validation, choices, and default values.
"""

import pytest
from aiconfigurator.sdk import common


class TestCLIArgumentParsing:
    """Test CLI argument parsing and validation."""

    def test_configure_parser_creates_valid_parser(self, cli_parser):
        """Test that configure_parser creates a valid argument parser."""
        required_actions = [action for action in cli_parser._actions if action.required]
        required_args = [action.dest for action in required_actions]
        
        assert 'model_name' in required_args
        assert 'total_gpus' in required_args
        assert 'system' in required_args

    @pytest.mark.parametrize("param_name,expected_choices", [
        ("model_name", list(common.SupportedModels.keys())),
        ("system", ['h100_sxm', 'h200_sxm']),
        ("backend", [backend.value for backend in common.BackendName])
    ])
    def test_argument_choices_validation(self, cli_parser, param_name, expected_choices):
        """Test that arguments validate against supported choices."""
        action = next(action for action in cli_parser._actions if action.dest == param_name)
        assert list(action.choices) == expected_choices

    def test_default_values_are_set(self, cli_parser):
        """Test that default values are properly set for optional arguments."""
        args = cli_parser.parse_args([
            '--model_name', 'QWEN3_32B',
            '--total_gpus', '8',
            '--system', 'h200_sxm'
        ])
        
        assert args.backend == common.BackendName.trtllm.value
        assert args.version == '0.20.0'
        assert args.save_dir is None
        assert args.debug is False

    def test_debug_mode_flag(self, cli_parser):
        """Test that debug mode can be enabled."""
        args = cli_parser.parse_args([
            '--model_name', 'QWEN3_32B',
            '--total_gpus', '8',
            '--system', 'h200_sxm',
            '--debug'
        ])
        
        assert args.debug is True

    def test_save_directory_argument(self, cli_parser):
        """Test that save directory can be specified."""
        args = cli_parser.parse_args([
            '--model_name', 'QWEN3_32B',
            '--total_gpus', '8',
            '--system', 'h200_sxm',
            '--save_dir', '/tmp/test'
        ])
        
        assert args.save_dir == '/tmp/test'

    @pytest.mark.parametrize("optional_param,value,expected_type", [
        ("isl", "4000", int),
        ("osl", "500", int),
        ("ttft", "300.0", float),
        ("tpot", "10.0", float),
    ])
    def test_optional_parameters(self, cli_parser, optional_param, value, expected_type):
        """Test that optional parameters can be set and have correct types."""
        args = cli_parser.parse_args([
            '--model_name', 'QWEN3_32B',
            '--total_gpus', '8',
            '--system', 'h200_sxm',
            f'--{optional_param}', value
        ])
        
        param_value = getattr(args, optional_param)
        assert isinstance(param_value, expected_type)
        assert param_value == expected_type(value) 