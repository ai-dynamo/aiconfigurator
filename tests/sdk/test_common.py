# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for common SDK configurations.

Tests supported systems, model families, and other common configurations.
"""

from pathlib import Path

from aiconfigurator.sdk import common


class TestSupportedSystems:
    """Test supported systems configuration."""

    def test_supported_systems_exists(self):
        """Test that SupportedSystems set exists and has content."""
        assert hasattr(common, "SupportedSystems")
        assert isinstance(common.SupportedSystems, set)
        assert len(common.SupportedSystems) > 0

    def test_supported_systems_matches_yaml_files_and_folders(self):
        """Test that SupportedSystems set matches the YAML files and data folders in systems directory."""
        # Get the path to the systems directory
        # Assuming the test is run from the repository root
        systems_dir = Path(__file__).parent.parent.parent / "src" / "aiconfigurator" / "systems"
        data_dir = systems_dir / "data"

        # Get all YAML files in the systems directory (excluding subdirectories)
        yaml_files = list(systems_dir.glob("*.yaml"))

        # Extract system names from YAML filenames (without .yaml extension)
        yaml_system_names = {f.stem for f in yaml_files}

        # Get all folders in the data directory
        data_folders = [f for f in data_dir.iterdir() if f.is_dir()]
        data_folder_names = {f.name for f in data_folders}

        # Assert that the YAML files match SupportedSystems
        assert common.SupportedSystems == yaml_system_names, (
            f"SupportedSystems set does not match YAML files in systems directory.\n"
            f"In SupportedSystems but not in YAML files: {common.SupportedSystems - yaml_system_names}\n"
            f"In YAML files but not in SupportedSystems: {yaml_system_names - common.SupportedSystems}"
        )

        # Assert that the data folders match SupportedSystems
        assert common.SupportedSystems == data_folder_names, (
            f"SupportedSystems set does not match data folders in systems/data directory.\n"
            f"In SupportedSystems but not in data folders: {common.SupportedSystems - data_folder_names}\n"
            f"In data folders but not in SupportedSystems: {data_folder_names - common.SupportedSystems}"
        )
