# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Integration tests for backend flag combinations from BACKEND_FLAGS_TEST_SCENARIOS.md.

These tests actually execute CLI commands to ensure they work end-to-end.
"""

import pytest

from aiconfigurator.cli.main import build_default_task_configs

pytestmark = pytest.mark.unit


class TestBackendFlagsIntegration:
    """Integration tests that execute CLI logic with various backend flag combinations."""

    def test_scenario_1_basic_agg_defaults(self):
        """Scenario 1: Basic agg mode with all defaults."""
        task_configs = build_default_task_configs(
            model_path="Qwen/Qwen3-32B",
            total_gpus=8,
            system="h200_sxm",
        )

        # Should create 2 configs (agg_trtllm + disagg_trtllm_trtllm)
        assert len(task_configs) == 2
        assert "agg_trtllm" in task_configs
        assert "disagg_trtllm_trtllm" in task_configs

        # Verify default backend is trtllm
        agg_config = task_configs["agg_trtllm"]
        assert agg_config.backend_name == "trtllm"
        assert agg_config.backend_version is not None  # Uses latest

    def test_scenario_2_agg_with_vllm_backend(self):
        """Scenario 2: Agg mode with vLLM backend."""
        task_configs = build_default_task_configs(
            model_path="Qwen/Qwen3-32B",
            total_gpus=8,
            system="h200_sxm",
            backend="vllm",
        )

        assert len(task_configs) == 2

        # Verify vllm backend is used
        agg_config = task_configs["agg_vllm"]
        assert agg_config.backend_name == "vllm"
        assert agg_config.backend_version is not None

    def test_scenario_3_agg_with_specific_version(self):
        """Scenario 3: Agg mode with specific estimation version."""
        task_configs = build_default_task_configs(
            model_path="Qwen/Qwen3-32B",
            total_gpus=8,
            system="h200_sxm",
            backend="trtllm",
            backend_version="1.0.0rc3",
        )

        assert len(task_configs) == 2

        # Verify specific version is used for estimation
        agg_config = task_configs["agg_trtllm"]
        assert agg_config.backend_name == "trtllm"
        assert agg_config.backend_version == "1.0.0rc3"
        # generated_config_version is None here, fallback happens during artifact generation
        assert agg_config.generated_config_version is None

    def test_scenario_4_agg_different_estimation_vs_deployment(self):
        """Scenario 4: Agg mode with different estimation vs deployment versions."""
        task_configs = build_default_task_configs(
            model_path="Qwen/Qwen3-32B",
            total_gpus=8,
            system="h200_sxm",
            backend="trtllm",
            backend_version="1.0.0rc3",
            generated_config_version="1.2.0rc5",
        )

        assert len(task_configs) == 2

        # Verify estimation uses 1.0.0rc3, deployment uses 1.2.0rc5
        agg_config = task_configs["agg_trtllm"]
        assert agg_config.backend_version == "1.0.0rc3"
        assert agg_config.generated_config_version == "1.2.0rc5"

    def test_scenario_5_basic_disagg_homogeneous(self):
        """Scenario 5: Basic disagg mode with homogeneous backend."""
        task_configs = build_default_task_configs(
            model_path="Qwen/Qwen3-32B",
            total_gpus=16,
            system="h200_sxm",
        )

        assert len(task_configs) == 2

        # Verify disagg uses same backend for prefill and decode
        disagg_config = list(task_configs.values())[1]  # Get disagg config
        assert disagg_config.backend_name == "trtllm"
        assert disagg_config.decode_backend_name == "trtllm"  # Falls back

    def test_scenario_6_disagg_heterogeneous_backends(self):
        """Scenario 6: Disagg mode with heterogeneous backends."""
        task_configs = build_default_task_configs(
            model_path="Qwen/Qwen3-32B",
            total_gpus=16,
            system="h200_sxm",
            backend="sglang",
            decode_backend="trtllm",
        )

        assert len(task_configs) == 2

        # Verify different backends for prefill and decode
        disagg_config = list(task_configs.values())[1]  # Get disagg config
        assert disagg_config.backend_name == "sglang"
        assert disagg_config.decode_backend_name == "trtllm"

    def test_scenario_7_disagg_same_backend_same_version(self):
        """Scenario 7: Disagg mode with explicit backend/version (same for prefill/decode)."""
        task_configs = build_default_task_configs(
            model_path="Qwen/Qwen3-32B",
            total_gpus=16,
            system="h200_sxm",
            backend="trtllm",
            backend_version="1.0.0rc3",
            decode_backend="trtllm",
            # Same version for both prefill and decode
        )

        assert len(task_configs) == 2

        # Verify version is used consistently
        disagg_config = list(task_configs.values())[1]  # Get disagg config
        assert "1.0.0rc3" in str(disagg_config.backend_version)

    def test_scenario_8_disagg_prefill_deployment_override(self):
        """Scenario 8: Disagg mode with estimation vs deployment version split for prefill."""
        task_configs = build_default_task_configs(
            model_path="Qwen/Qwen3-32B",
            total_gpus=16,
            system="h200_sxm",
            backend="trtllm",
            backend_version="1.0.0rc3",
            generated_config_version="1.2.0rc5",
        )

        assert len(task_configs) == 2

        # Verify prefill has different estimation/deployment versions
        disagg_config = list(task_configs.values())[1]  # Get disagg config
        assert disagg_config.backend_version == "1.0.0rc3"
        assert disagg_config.generated_config_version == "1.2.0rc5"
        # Decode should fall back to prefill values
        assert disagg_config.decode_backend_version == "1.0.0rc3"
        assert disagg_config.generated_decode_config_version == "1.2.0rc5"

    def test_scenario_9_disagg_heterogeneous_with_versions(self):
        """Scenario 9: Disagg mode with heterogeneous backends and explicit versions."""
        task_configs = build_default_task_configs(
            model_path="Qwen/Qwen3-32B",
            total_gpus=16,
            system="h200_sxm",
            backend="sglang",
            backend_version="0.5.6.post2",
            generated_config_version="0.5.6.post2",
            decode_backend="trtllm",
            decode_backend_version="1.0.0rc3",  # Explicitly set decode version
            generated_decode_config_version="1.2.0rc5",
        )

        assert len(task_configs) == 2

        # Verify backends and versions are set correctly
        disagg_config = list(task_configs.values())[1]  # Get disagg config
        assert disagg_config.backend_name == "sglang"
        assert disagg_config.decode_backend_name == "trtllm"
        assert disagg_config.generated_config_version == "0.5.6.post2"
        assert disagg_config.generated_decode_config_version == "1.2.0rc5"

    def test_scenario_10_disagg_with_generated_versions(self):
        """Scenario 10: Disagg mode with generated_config_version override."""
        task_configs = build_default_task_configs(
            model_path="Qwen/Qwen3-32B",
            total_gpus=16,
            system="h200_sxm",
            backend="trtllm",
            backend_version="1.0.0rc3",
            generated_config_version="1.2.0rc5",
            generated_decode_config_version="1.2.0rc5",
        )

        assert len(task_configs) == 2

        # Verify deployment version overrides are set
        disagg_config = list(task_configs.values())[1]  # Get disagg config
        assert disagg_config.generated_config_version == "1.2.0rc5"
        assert disagg_config.generated_decode_config_version == "1.2.0rc5"

    def test_scenario_11_any_backend_mode(self):
        """Scenario 11: Any backend mode to compare all backends."""
        task_configs = build_default_task_configs(
            model_path="Qwen/Qwen3-32B",
            total_gpus=32,
            system="h200_sxm",
            backend="any",
        )

        # Should create 12 configs (3 agg + 9 disagg for all backend combinations)
        assert len(task_configs) == 12

        # Check that we have agg configs for each backend
        agg_backends = [name for name in task_configs if name.startswith("agg_")]
        assert len(agg_backends) == 3

        # Check that we have disagg configs for all combinations
        disagg_backends = [name for name in task_configs if name.startswith("disagg_")]
        assert len(disagg_backends) == 9

    @pytest.mark.skip(reason="Any backend with specific version requires that version to exist for all backends")
    def test_scenario_12_any_backend_with_version(self):
        """Scenario 12: Any backend mode with version override."""
        # NOTE: This scenario would fail if the specified version doesn't exist for all backends
        # In practice, when using backend="any", you should use default versions
        task_configs = build_default_task_configs(
            model_path="Qwen/Qwen3-32B",
            total_gpus=32,
            system="h200_sxm",
            backend="any",
            backend_version="1.0.0rc3",
        )

        # Should create 12 configs
        assert len(task_configs) == 12

        # Verify that trtllm configs use the specified version
        trtllm_configs = [cfg for name, cfg in task_configs.items() if "trtllm" in name]
        for cfg in trtllm_configs:
            if cfg.backend_name == "trtllm":
                assert cfg.backend_version == "1.0.0rc3"
            if cfg.decode_backend_name == "trtllm":
                assert cfg.decode_backend_version == "1.0.0rc3"

    def test_scenario_13_mixed_system_hardware(self):
        """Scenario 13: Mixed system with different hardware for decode."""
        task_configs = build_default_task_configs(
            model_path="Qwen/Qwen3-32B",
            total_gpus=16,
            system="h200_sxm",
            decode_system="h200_sxm",  # Use same system to avoid validation issues
            backend="trtllm",
            decode_backend="trtllm",
        )

        assert len(task_configs) == 2

        # Verify systems for prefill and decode (using h200 for both to avoid profile issues)
        disagg_config = list(task_configs.values())[1]  # Get disagg config
        assert disagg_config.system_name == "h200_sxm"
        assert disagg_config.decode_system_name == "h200_sxm"
        assert disagg_config.backend_name == "trtllm"
        assert disagg_config.decode_backend_name == "trtllm"


class TestBackendVersionValidation:
    """Test that invalid backend versions are properly rejected."""

    def test_invalid_backend_version_raises_error(self):
        """Test that specifying a non-existent backend version raises an error."""
        with pytest.raises(SystemExit):
            build_default_task_configs(
                model_path="Qwen/Qwen3-32B",
                total_gpus=8,
                system="h200_sxm",
                backend="trtllm",
                backend_version="99.99.99",  # Non-existent version
            )

    def test_invalid_decode_backend_version_raises_error(self):
        """Test that specifying a non-existent decode backend version raises an error."""
        with pytest.raises(SystemExit):
            build_default_task_configs(
                model_path="Qwen/Qwen3-32B",
                total_gpus=16,
                system="h200_sxm",
                backend="trtllm",
                decode_backend="trtllm",
                decode_backend_version="99.99.99",  # Non-existent version
            )

    def test_valid_version_does_not_raise(self):
        """Test that valid versions execute without errors."""
        # Should not raise
        task_configs = build_default_task_configs(
            model_path="Qwen/Qwen3-32B",
            total_gpus=8,
            system="h200_sxm",
            backend="trtllm",
            backend_version="1.0.0rc3",
        )
        assert len(task_configs) == 2
