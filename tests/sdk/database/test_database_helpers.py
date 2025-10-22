# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import importlib
import sys
from pathlib import Path

import pytest
import yaml


@pytest.fixture(scope="module")
def perf_database():
    """
    Import the local aiconfigurator.sdk.perf_database module from src/,
    ensuring it takes precedence over any installed package.
    """
    project_root = Path(__file__).resolve().parents[3]
    src_path = project_root / "src"
    sys.path.insert(0, str(src_path))

    saved_aiconfigurator_modules = {}

    # Purge already-imported site-packages version if present
    for key in list(sys.modules.keys()):
        if key == "aiconfigurator" or key.startswith("aiconfigurator."):
            saved_aiconfigurator_modules[key] = sys.modules.pop(key)

    import aiconfigurator.sdk.perf_database as perf_database

    importlib.reload(perf_database)
    yield perf_database

    # Restore aiconfigurator modules after the tests in this file finish.
    sys.modules.update(saved_aiconfigurator_modules)


@pytest.fixture
def temp_systems_dir(tmp_path: Path) -> Path:
    return tmp_path


def setup_mock_filesystem(systems_dir: Path, layout: dict) -> None:
    for system_name, backends in layout.items():
        data_dir_name = f"data_{system_name}"
        data_dir = systems_dir / data_dir_name
        data_dir.mkdir(parents=True, exist_ok=True)

        (systems_dir / f"{system_name}.yaml").write_text(yaml.safe_dump({"data_dir": data_dir_name}))

        for backend_name, versions in backends.items():
            backend_dir = data_dir / backend_name
            backend_dir.mkdir(exist_ok=True)
            for version in versions:
                # Hidden dirs should be ignored by implementation
                if version.startswith("."):
                    (backend_dir / version).mkdir(exist_ok=True)
                else:
                    (backend_dir / version).mkdir(exist_ok=True)


# ----------------------------- get_supported_databases -----------------------------


def test_get_supported_databases_basic(temp_systems_dir: Path, perf_database):
    setup_mock_filesystem(
        temp_systems_dir,
        {
            "h100": {"trtllm": ["1.0.0", "1.1.0", ".hidden"], "vllm": ["0.5.0"]},
            "h200": {"trtllm": ["1.2.0", "1.3.0rc2"]},
        },
    )

    result = perf_database.get_supported_databases(str(temp_systems_dir))

    assert "h100" in result and "h200" in result
    assert result["h100"]["trtllm"] == ["1.0.0", "1.1.0"]
    assert result["h100"]["vllm"] == ["0.5.0"]
    assert result["h200"]["trtllm"] == ["1.2.0", "1.3.0rc2"]


def test_get_supported_databases_empty_dir(temp_systems_dir: Path, perf_database):
    result = perf_database.get_supported_databases(str(temp_systems_dir))
    # defaultdict, but empty
    assert isinstance(result, dict)
    assert len(result) == 0


def test_get_supported_databases_edge_cases(temp_systems_dir: Path, perf_database):
    """Tests that get_supported_databases handles various edge cases gracefully."""
    # Case 1: Empty directory
    assert perf_database.get_supported_databases(str(temp_systems_dir)) == {}

    # Case 2: Invalid YAML file should be skipped
    (temp_systems_dir / "invalid.yaml").write_text("key: [")
    setup_mock_filesystem(temp_systems_dir, {"h100": {"trtllm": ["1.0.0"]}})
    result = perf_database.get_supported_databases(str(temp_systems_dir))
    assert "invalid" not in result
    assert "h100" in result  # Valid system should still be processed

    # Case 3: System YAML pointing to a non-existent data_dir
    (temp_systems_dir / "bad_path.yaml").write_text(yaml.safe_dump({"data_dir": "nonexistent_dir"}))
    result = perf_database.get_supported_databases(str(temp_systems_dir))
    assert "bad_path" not in result


# ----------------------------- get_latest_database_version -----------------------------


def test_get_latest_database_version_prefers_stable_over_rc(temp_systems_dir: Path, perf_database):
    # With the corrected logic, 2.0.0rc1 is correctly considered newer than 1.1.0
    mock_supported = {"h100": {"trtllm": ["1.1.0", "2.0.0rc1"]}}

    original = perf_database.get_supported_databases
    try:
        perf_database.get_supported_databases = lambda: mock_supported
        latest = perf_database.get_latest_database_version("h100", "trtllm")
        assert latest == "2.0.0rc1"
    finally:
        perf_database.get_supported_databases = original


def test_get_latest_database_version_rc_only(temp_systems_dir: Path, perf_database):
    mock_supported = {"h100": {"trtllm": ["1.0.0rc1", "1.0.0rc2", "1.1.0rc1"]}}

    original = perf_database.get_supported_databases
    try:
        perf_database.get_supported_databases = lambda: mock_supported
        latest = perf_database.get_latest_database_version("h100", "trtllm")
        assert latest == "1.1.0rc1"
    finally:
        perf_database.get_supported_databases = original


def test_get_latest_database_version_nonexistent_returns_none(temp_systems_dir: Path, perf_database):
    mock_supported = {"h100": {"trtllm": ["1.0.0"]}}

    original = perf_database.get_supported_databases
    try:
        perf_database.get_supported_databases = lambda: mock_supported
        assert perf_database.get_latest_database_version("nonexistent", "trtllm") is None
        assert perf_database.get_latest_database_version("h100", "nonexistent") is None
    finally:
        perf_database.get_supported_databases = original


def test_get_latest_database_version_unparseable_versions(temp_systems_dir: Path, perf_database):
    """Tests that it falls back gracefully when versions are unparseable."""
    mock_supported = {"h100": {"trtllm": ["foo", "bar"]}}

    original = perf_database.get_supported_databases
    try:
        perf_database.get_supported_databases = lambda: mock_supported
        # It should return the 'max' of the unparseable strings as a fallback
        latest = perf_database.get_latest_database_version("h100", "trtllm")
        assert latest == "foo"
    finally:
        perf_database.get_supported_databases = original


def test_get_latest_database_version_major_version_rc_is_newer(temp_systems_dir: Path, perf_database):
    """Tests that a v1.0 RC is correctly considered newer than a v0.20 stable."""
    mock_supported = {"h200": {"trtllm": ["0.20.0", "1.0.0rc3"]}}

    original = perf_database.get_supported_databases
    try:
        perf_database.get_supported_databases = lambda: mock_supported
        latest = perf_database.get_latest_database_version("h200", "trtllm")
        assert latest == "1.0.0rc3"
    finally:
        perf_database.get_supported_databases = original
