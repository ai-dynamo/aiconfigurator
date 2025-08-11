# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import argparse
from unittest.mock import MagicMock

from aiconfigurator.cli.main import configure_parser as configure_cli_parser
from aiconfigurator.sdk import common


@pytest.fixture
def cli_parser():
    """Pre-configured CLI parser for testing."""
    parser = argparse.ArgumentParser()
    configure_cli_parser(parser)
    return parser


@pytest.fixture
def sample_cli_args():
    """Sample CLI arguments for testing."""
    args = MagicMock()
    args.model_name = 'QWEN3_32B'
    args.system = 'h200_sxm'
    args.backend = 'trtllm'
    args.version = '0.20.0'
    args.total_gpus = 8
    args.isl = None
    args.osl = None
    args.ttft = None
    args.tpot = None
    args.save_dir = None
    args.debug = False
    return args


@pytest.fixture
def mock_perf_database():
    """Mock performance database for CLI testing."""
    mock_db = MagicMock()
    mock_db.system = 'h200_sxm'
    mock_db.backend = 'trtllm'
    mock_db.version = '0.20.0'
    return mock_db 