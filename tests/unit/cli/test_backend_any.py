# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for --backend auto functionality.
"""

import pytest

from aiconfigurator.cli.main import build_default_tasks
from aiconfigurator.sdk.common import BackendName

pytestmark = pytest.mark.unit


class TestBackendAny:
    """Tests for --backend auto functionality."""

    def test_build_default_tasks_single_backend(self):
        """Single backend should create 2 task configs (agg, disagg)."""
        tasks = build_default_tasks(
            model_path="Qwen/Qwen3-32B",
            total_gpus=8,
            system="h200_sxm",
            backend="trtllm",
        )

        assert len(tasks) == 2
        assert "agg" in tasks
        assert "disagg" in tasks
        assert tasks["agg"].primary_backend_name == "trtllm"
        assert tasks["disagg"].primary_backend_name == "trtllm"

    def test_build_default_tasks_any_backend(self):
        """Backend 'auto' should create 6 internal task configs but they will be merged later."""
        tasks = build_default_tasks(
            model_path="Qwen/Qwen3-32B",
            total_gpus=8,
            system="h200_sxm",
            backend="auto",
        )

        # Should have 6 internal configs: agg_trtllm, agg_vllm, agg_sglang, disagg_trtllm, disagg_vllm, disagg_sglang
        # These will be merged in _execute_tasks to produce 2 results (agg, disagg)
        assert len(tasks) == 6

        # Check all expected experiment names exist
        expected_names = {
            "agg_trtllm",
            "agg_vllm",
            "agg_sglang",
            "disagg_trtllm",
            "disagg_vllm",
            "disagg_sglang",
        }
        assert set(tasks.keys()) == expected_names

        # Verify each config has the correct backend
        for backend in BackendName:
            backend_name = backend.value
            assert tasks[f"agg_{backend_name}"].primary_backend_name == backend_name
            assert tasks[f"disagg_{backend_name}"].primary_backend_name == backend_name
            assert tasks[f"agg_{backend_name}"].serving_mode == "agg"
            assert tasks[f"disagg_{backend_name}"].serving_mode == "disagg"

    def test_build_default_tasks_any_backend_parameters(self):
        """Backend 'auto' should preserve all parameters for each backend config."""
        tasks = build_default_tasks(
            model_path="Qwen/Qwen3-32B",
            total_gpus=32,
            system="h200_sxm",
            backend="auto",
            isl=4000,
            osl=1000,
            ttft=2000.0,
            tpot=30.0,
            prefix=500,
        )

        # Check that all configs have the same parameters (except backend_name)
        for exp_name, task_config in tasks.items():
            assert task_config.primary_model_path == "Qwen/Qwen3-32B"
            assert task_config.total_gpus == 32
            assert task_config.primary_system_name == "h200_sxm"
            assert task_config.isl == 4000
            assert task_config.osl == 1000
            assert task_config.ttft == 2000.0
            assert task_config.tpot == 30.0
            assert task_config.prefix == 500

            # Disagg configs should have decode_system set (defaults to system)
            if exp_name.startswith("disagg"):
                assert task_config.decode_system_name == "h200_sxm"

    def test_build_default_tasks_with_nextn(self):
        """Test that nextn and nextn_accepted are passed to TaskConfig when specified."""
        tasks = build_default_tasks(
            model_path="Qwen/Qwen3-32B",
            total_gpus=8,
            system="h200_sxm",
            backend="trtllm",
            nextn=3,
            nextn_accepted=1.5,
        )

        assert len(tasks) == 2

        # Verify nextn is set in config
        for task_config in tasks.values():
            assert task_config.nextn == 3
            assert task_config.nextn_accepted == 1.5

    def test_build_default_tasks_nextn_default_zero(self):
        """Test that nextn defaults to 0 (MTP disabled) when not specified."""
        tasks = build_default_tasks(
            model_path="Qwen/Qwen3-32B",
            total_gpus=8,
            system="h200_sxm",
            backend="trtllm",
        )

        assert len(tasks) == 2

        # Verify nextn defaults to 0 (and no acceptance assumption exists)
        for task_config in tasks.values():
            assert task_config.nextn == 0
            assert task_config.nextn_accepted is None
