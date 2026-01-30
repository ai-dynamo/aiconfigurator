# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for the 'any' backend feature.

Tests that the 'any' backend option correctly expands into all concrete backend combinations.
"""

import pytest

from aiconfigurator.cli.main import build_default_task_configs
from aiconfigurator.sdk import common

pytestmark = pytest.mark.unit


class TestAnyBackendEnum:
    """Test that 'any' is a valid BackendName."""

    def test_any_backend_exists(self):
        """Test that 'any' is a valid BackendName enum value."""
        assert hasattr(common.BackendName, "any")
        assert common.BackendName.any.value == "any"

    def test_all_backends_excludes_any(self):
        """Test that all_backends() does not include 'any'."""
        concrete = common.BackendName.all_backends()
        assert common.BackendName.any.value not in concrete
        assert len(concrete) == 3
        assert common.BackendName.trtllm.value in concrete
        assert common.BackendName.sglang.value in concrete
        assert common.BackendName.vllm.value in concrete


class TestGetBackendsToCheck:
    """Test the resolve_backends classmethod."""

    def test_any_expands_to_all_backends(self):
        """Test that 'any' expands to all concrete backends."""
        backends = common.BackendName.resolve_backends("any")
        assert len(backends) == 3
        assert "trtllm" in backends
        assert "sglang" in backends
        assert "vllm" in backends

    def test_single_backend_returns_single_list(self):
        """Test that a single backend returns a single-element list."""
        assert common.BackendName.resolve_backends("trtllm") == ["trtllm"]
        assert common.BackendName.resolve_backends("sglang") == ["sglang"]
        assert common.BackendName.resolve_backends("vllm") == ["vllm"]


class TestBuildDefaultTaskConfigsWithAny:
    """Test build_default_task_configs with 'any' backend."""

    def test_any_backend_creates_12_configs(self):
        """Test that 'any' backend creates 12 task configs (3 agg + 9 disagg)."""
        task_configs = build_default_task_configs(
            model_path="Qwen/Qwen3-32B",
            total_gpus=8,
            system="h200_sxm",
            backend="any",
        )

        # Should have 3 agg configs + 9 disagg configs = 12 total
        assert len(task_configs) == 12

        # Check agg configs exist
        assert "agg_trtllm" in task_configs
        assert "agg_sglang" in task_configs
        assert "agg_vllm" in task_configs

        # Check disagg configs exist (all 9 combinations)
        for prefill in ["trtllm", "sglang", "vllm"]:
            for decode in ["trtllm", "sglang", "vllm"]:
                assert f"disagg_{prefill}_{decode}" in task_configs

    def test_single_backend_creates_2_configs(self):
        """Test that a single backend creates 2 task configs (1 agg + 1 disagg)."""
        task_configs = build_default_task_configs(
            model_path="Qwen/Qwen3-32B",
            total_gpus=8,
            system="h200_sxm",
            backend="trtllm",
        )

        # Should have 1 agg + 1 disagg = 2 total
        # Uses same naming pattern as 'any' mode for consistency
        assert len(task_configs) == 2
        assert "agg_trtllm" in task_configs
        assert "disagg_trtllm_trtllm" in task_configs

    def test_any_backend_agg_configs_have_correct_backend(self):
        """Test that agg configs have the correct backend name."""
        task_configs = build_default_task_configs(
            model_path="Qwen/Qwen3-32B",
            total_gpus=8,
            system="h200_sxm",
            backend="any",
        )

        assert task_configs["agg_trtllm"].config.worker_config.backend_name == "trtllm"
        assert task_configs["agg_sglang"].config.worker_config.backend_name == "sglang"
        assert task_configs["agg_vllm"].config.worker_config.backend_name == "vllm"

    def test_any_backend_disagg_configs_have_correct_backends(self):
        """Test that disagg configs have correct prefill/decode backend names."""
        task_configs = build_default_task_configs(
            model_path="Qwen/Qwen3-32B",
            total_gpus=8,
            system="h200_sxm",
            backend="any",
        )

        # Check a few disagg configs
        disagg_trtllm_trtllm = task_configs["disagg_trtllm_trtllm"]
        assert disagg_trtllm_trtllm.config.prefill_worker_config.backend_name == "trtllm"
        assert disagg_trtllm_trtllm.config.decode_worker_config.backend_name == "trtllm"

        disagg_trtllm_sglang = task_configs["disagg_trtllm_sglang"]
        assert disagg_trtllm_sglang.config.prefill_worker_config.backend_name == "trtllm"
        assert disagg_trtllm_sglang.config.decode_worker_config.backend_name == "sglang"

        disagg_sglang_vllm = task_configs["disagg_sglang_vllm"]
        assert disagg_sglang_vllm.config.prefill_worker_config.backend_name == "sglang"
        assert disagg_sglang_vllm.config.decode_worker_config.backend_name == "vllm"

    def test_explicit_heterogeneous_backends(self):
        """Test specifying different prefill and decode backends explicitly."""
        task_configs = build_default_task_configs(
            model_path="Qwen/Qwen3-32B",
            total_gpus=8,
            system="h200_sxm",
            backend="sglang",
            decode_backend="trtllm",
        )

        # Should have 1 agg config (sglang) and 1 disagg config (sglang prefill, trtllm decode)
        assert len(task_configs) == 2
        assert "agg_sglang" in task_configs
        assert "disagg_sglang_trtllm" in task_configs

        disagg = task_configs["disagg_sglang_trtllm"]
        assert disagg.config.prefill_worker_config.backend_name == "sglang"
        assert disagg.config.decode_worker_config.backend_name == "trtllm"


class TestBackendChoicesIncludesAny:
    """Test that CLI argument parsing includes 'any' as a valid choice."""

    def test_backend_choices_includes_any(self, cli_parser):
        """Test that backend argument includes 'any' as a valid choice."""
        subparser_action = next(action for action in cli_parser._actions if action.dest == "mode")
        default_parser = subparser_action.choices["default"]
        action = next(action for action in default_parser._actions if action.dest == "backend")
        assert "any" in action.choices

    def test_any_backend_parses_successfully(self, cli_parser):
        """Test that --backend any parses successfully."""
        args = cli_parser.parse_args(
            [
                "default",
                "--model_path",
                "Qwen/Qwen3-32B",
                "--total_gpus",
                "8",
                "--system",
                "h200_sxm",
                "--backend",
                "any",
            ]
        )
        assert args.backend == "any"
