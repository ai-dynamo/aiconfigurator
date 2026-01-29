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

    @pytest.mark.parametrize(
        "model,system,backend,version,architecture,expected_agg,expected_disagg",
        [
            # Known supported combination (Qwen3-32B on H200)
            ("Qwen/Qwen3-32B", "h200_sxm", None, None, None, True, True),
            # Architecture-based support for a model not in the matrix
            ("Qwen/Qwen3-235B-A22B-Thinking-2507", "h200_sxm", None, None, "Qwen3ForCausalLM", True, True),
            # Specific backend and version that should pass
            ("Qwen/Qwen3-32B", "h200_sxm", "trtllm", "1.2.0rc5", None, True, True),
            # Unsupported model
            ("non-existent-model", "h100_sxm", None, None, None, False, False),
            # Unsupported system
            ("Qwen/Qwen3-32B", "non-existent-system", None, None, None, False, False),
        ],
    )
    def test_check_support(self, model, system, backend, version, architecture, expected_agg, expected_disagg):
        """Test check_support function with various model/system combinations."""
        agg, disagg = common.check_support(model, system, backend, version, architecture)
        assert agg is expected_agg
        assert disagg is expected_disagg
