# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for CLI configuration loading functionality.

Tests YAML configuration loading and validation.
"""

import pytest
from unittest.mock import patch

from aiconfigurator.cli.main import load_config_from_yaml


class TestAIConfiguratorConfigLoading:
    """Test configuration loading from YAML templates."""

    def test_load_config_validates_total_gpus_minimum(self):
        """Test that config loading validates minimum total GPUs."""
        with patch('aiconfigurator.cli.main.yaml.safe_load') as mock_yaml_load:
            with patch('builtins.open'):
                mock_yaml_load.return_value = {'isl': 4000, 'osl': 500, 'ttft': 300.0, 'tpot': 10.0}
                
                with pytest.raises(AssertionError, match="Total GPUs .* is less than 2"):
                    load_config_from_yaml(
                        model_name='QWEN3_32B',
                        system_name='h200_sxm',
                        backend_name='trtllm',
                        version='0.20.0',
                        total_gpus=1,  # Should fail
                        isl=4000,
                        osl=500,
                        ttft=300.0,
                        tpot=10.0
                    )

    @pytest.mark.parametrize("invalid_value", [0, -1, -10])
    def test_load_config_validates_positive_values(self, invalid_value):
        """Test that config loading validates positive values for ISL, OSL, TTFT, TPOT."""
        with patch('aiconfigurator.cli.main.yaml.safe_load') as mock_yaml_load:
            with patch('builtins.open'):
                mock_yaml_load.return_value = {
                    'isl': 4000, 'osl': 500, 'ttft': 300.0, 'tpot': 10.0
                }
                
                with pytest.raises(AssertionError, match="must be greater than 0"):
                    load_config_from_yaml(
                        model_name='QWEN3_32B',
                        system_name='h200_sxm',
                        backend_name='trtllm',
                        version='0.20.0',
                        total_gpus=8,
                        isl=invalid_value,  # Should fail
                        osl=500,
                        ttft=300.0,
                        tpot=10.0
                    ) 