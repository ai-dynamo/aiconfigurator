# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for CLI integration functionality.

Tests the full CLI workflow and integration between components.
"""

import pytest
from unittest.mock import patch, MagicMock

from aiconfigurator.cli.main import main as cli_main


class TestCLIIntegration:
    """Integration tests for the full CLI workflow."""

    @patch('aiconfigurator.cli.main.AIConfigurator')
    @patch('aiconfigurator.cli.main.load_config_from_yaml')
    def test_cli_main_success_flow(self, mock_load_config, mock_aiconfigurator_class, sample_cli_args):
        """Test successful CLI main execution flow."""
        # Setup mocks
        mock_config = MagicMock()
        mock_load_config.return_value = mock_config
        
        mock_aiconfigurator = MagicMock()
        mock_result = MagicMock()
        mock_aiconfigurator.run.return_value = mock_result
        mock_aiconfigurator_class.return_value = mock_aiconfigurator
        
        # Execute
        cli_main(sample_cli_args)
        
        # Verify calls
        mock_load_config.assert_called_once_with(
            'QWEN3_32B', 'h200_sxm', 'trtllm', '0.20.0', 8, None, None, None, None
        )
        mock_aiconfigurator.run.assert_called_once_with(mock_config)
        mock_aiconfigurator.log_final_summary.assert_called_once()

    @pytest.mark.parametrize("error_scenario,setup_error", [
        ("config_loading_error", lambda mocks: setattr(mocks['load_config'], 'side_effect', Exception("Config loading failed"))),
        ("execution_error", lambda mocks: setattr(mocks['aiconfigurator'].run, 'side_effect', Exception("Execution failed")))
    ])
    @patch('aiconfigurator.cli.main.AIConfigurator')
    @patch('aiconfigurator.cli.main.load_config_from_yaml')
    def test_cli_main_error_handling(self, mock_load_config, mock_aiconfigurator_class, 
                                   sample_cli_args, error_scenario, setup_error):
        """Test CLI main handles various errors gracefully."""
        # Setup mocks
        mock_config = MagicMock()
        mock_load_config.return_value = mock_config
        
        mock_aiconfigurator = MagicMock()
        mock_aiconfigurator_class.return_value = mock_aiconfigurator
        
        # Setup specific error scenario
        mocks = {
            'load_config': mock_load_config,
            'aiconfigurator': mock_aiconfigurator
        }
        setup_error(mocks)
        
        # Should call exit(1)
        with patch('aiconfigurator.cli.main.exit') as mock_exit:
            cli_main(sample_cli_args)
            mock_exit.assert_called_with(1) 