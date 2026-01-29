# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for common SDK configurations.

Tests supported systems, model families, and other common configurations.
"""

from pathlib import Path

import pytest

from aiconfigurator.sdk import common

pytestmark = pytest.mark.unit


def _find_repo_root(start: Path) -> Path:
    """Find repository root.

    In the Docker test image we copy `src/` and `tests/` into `/workspace/` but do
    not copy `pyproject.toml`, so we detect the repo root via `src/aiconfigurator/`.
    """
    start = start.resolve()
    for parent in [start, *start.parents]:
        if (parent / "src" / "aiconfigurator").is_dir():
            return parent
    raise RuntimeError("Cannot find repository root (expected src/aiconfigurator/)")


class TestSupportedSystems:
    """Test supported systems configuration."""

    def test_supported_systems_exists(self):
        """Test that SupportedSystems set exists and has content."""
        assert hasattr(common, "SupportedSystems")
        assert isinstance(common.SupportedSystems, set)
        assert len(common.SupportedSystems) > 0

    def test_supported_systems_matches_yaml_files_and_folders(self):
        """Test that SupportedSystems set matches the YAML files and data folders in systems directory."""
        repo_root = _find_repo_root(Path(__file__))
        systems_dir = repo_root / "src" / "aiconfigurator" / "systems"
        data_dir = systems_dir / "data"

        # Get all YAML files in the systems directory (excluding subdirectories)
        yaml_files = list(systems_dir.glob("*.yaml"))

        # Extract system names from YAML filenames (without .yaml extension)
        yaml_system_names = {f.stem for f in yaml_files}

        # Get all folders in the data directory
        data_folders = [f for f in data_dir.iterdir() if f.is_dir()]
        data_folder_names = {f.name for f in data_folders}

        # Assert that the YAML files match SupportedSystems
        assert common.SupportedSystems.issubset(yaml_system_names), (
            "SupportedSystems set does not match YAML files in systems directory.\n"
        )

        # Assert that the data folders match SupportedSystems
        assert common.SupportedSystems.issubset(data_folder_names), (
            "SupportedSystems set does not match data folders in systems/data directory.\n"
        )


class TestSupportMatrix:
    """Test support matrix functionality."""

    def test_get_support_matrix(self):
        """Test that get_support_matrix returns a list of dictionaries."""
        matrix = common.get_support_matrix()
        assert isinstance(matrix, list)
        assert len(matrix) > 0
        assert isinstance(matrix[0], dict)
        assert "HuggingFaceID" in matrix[0]
        assert "System" in matrix[0]
        assert "Mode" in matrix[0]
        assert "Status" in matrix[0]

    def test_check_support(self):
        """Test check_support function."""
        # Test a known supported combination (Qwen3-32B on H200)
        # These are expected to be PASS in the current support_matrix.csv
        agg, disagg = common.check_support("Qwen/Qwen3-32B", "h200_sxm")
        assert agg is True
        assert disagg is True

        # Test architecture-based support for a model not in the matrix
        # Must pass architecture explicitly since model isn't in the matrix
        agg, disagg = common.check_support(
            "Qwen/Qwen3-235B-A22B-Thinking-2507", "h200_sxm", architecture="Qwen3ForCausalLM"
        )
        assert agg is True
        assert disagg is True

        # Test with a specific version that should pass
        # Pick one that is likely to be there, e.g., 1.2.0rc5 for Qwen3-32B on H200
        agg, disagg = common.check_support("Qwen/Qwen3-32B", "h200_sxm", backend="trtllm", version="1.2.0rc5")
        assert agg is True
        assert disagg is True

        # Test an unsupported model
        agg, disagg = common.check_support("non-existent-model", "h100_sxm")
        assert agg is False
        assert disagg is False

        # Test an unsupported system
        agg, disagg = common.check_support("Qwen/Qwen3-32B", "non-existent-system")
        assert agg is False
        assert disagg is False
