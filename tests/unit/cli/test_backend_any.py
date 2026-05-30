# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for --backend auto functionality.
"""

import pytest

from aiconfigurator.cli.main import build_default_task_configs
from aiconfigurator.sdk.common import BackendName

pytestmark = pytest.mark.unit


class TestBackendAny:
    """Tests for --backend auto functionality."""

    def test_build_default_task_configs_single_backend(self):
        """Single backend should create 2 task configs (agg, disagg)."""
        task_configs = build_default_task_configs(
            model_path="Qwen/Qwen3-32B",
            total_gpus=8,
            system="h200_sxm",
            backend="trtllm",
        )

        assert len(task_configs) == 2
        assert "agg" in task_configs
        assert "disagg" in task_configs
        assert task_configs["agg"].backend_name == "trtllm"
        # Disagg: backend lives on per-role fields; the agg-side flat field
        # stays at the dataclass default.
        assert task_configs["disagg"].prefill_backend_name == "trtllm"
        assert task_configs["disagg"].decode_backend_name == "trtllm"

    def test_build_default_task_configs_any_backend(self):
        """Backend 'auto' should create 6 internal task configs but they will be merged later."""
        task_configs = build_default_task_configs(
            model_path="Qwen/Qwen3-32B",
            total_gpus=8,
            system="h200_sxm",
            backend="auto",
        )

        # Should have 6 internal configs: agg_trtllm, agg_vllm, agg_sglang, disagg_trtllm, disagg_vllm, disagg_sglang
        # These will be merged in _execute_task_configs to produce 2 results (agg, disagg)
        assert len(task_configs) == 6

        # Check all expected experiment names exist
        expected_names = {
            "agg_trtllm",
            "agg_vllm",
            "agg_sglang",
            "disagg_trtllm",
            "disagg_vllm",
            "disagg_sglang",
        }
        assert set(task_configs.keys()) == expected_names

        # Verify each config has the correct backend
        for backend in BackendName:
            backend_name = backend.value
            agg_task = task_configs[f"agg_{backend_name}"]
            disagg_task = task_configs[f"disagg_{backend_name}"]
            assert agg_task.backend_name == backend_name
            assert disagg_task.prefill_backend_name == backend_name
            assert disagg_task.decode_backend_name == backend_name
            assert agg_task.serving_mode == "agg"
            assert disagg_task.serving_mode == "disagg"

    def test_build_default_task_configs_any_backend_parameters(self):
        """Backend 'auto' should preserve all parameters for each backend config."""
        task_configs = build_default_task_configs(
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

        # Check that all configs have the same workload parameters
        for exp_name, task_config in task_configs.items():
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

    def test_build_default_task_configs_with_nextn(self):
        """Test that nextn and nextn_accept_rates are passed to Task when specified."""
        task_configs = build_default_task_configs(
            model_path="Qwen/Qwen3-32B",
            total_gpus=8,
            system="h200_sxm",
            backend="trtllm",
            nextn=3,
            nextn_accept_rates=[0.9, 0.5, 0.2, 0.1, 0.0],
        )

        assert len(task_configs) == 2

        for task_config in task_configs.values():
            assert task_config.nextn == 3
            assert task_config.nextn_accept_rates == [0.9, 0.5, 0.2, 0.1, 0.0]

    def test_build_default_task_configs_nextn_default_zero(self):
        """Test that nextn resolves to 0 (MTP disabled) when CLI didn't request it.

        Task.__post_init__ pulls the model's HF ``num_nextn_predict_layers`` when
        nextn is unset; Qwen3-32B is a dense model (no MTP), so it resolves to 0.
        """
        task_configs = build_default_task_configs(
            model_path="Qwen/Qwen3-32B",
            total_gpus=8,
            system="h200_sxm",
            backend="trtllm",
        )

        assert len(task_configs) == 2

        for task_config in task_configs.values():
            assert task_config.nextn == 0
            assert task_config.nextn_accept_rates is not None
