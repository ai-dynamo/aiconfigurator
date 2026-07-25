# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import importlib
import sys
from pathlib import Path

import pytest
import yaml

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


@pytest.fixture(scope="module")
def perf_database():
    """
    Import the local aiconfigurator.sdk.perf_database module from src/,
    ensuring it takes precedence over any installed package.
    """
    project_root = _find_repo_root(Path(__file__))
    src_path = project_root / "src"
    sys.path.insert(0, str(src_path))

    saved_aiconfigurator_modules = {}

    # Purge already-imported site-packages version if present
    for key in list(sys.modules.keys()):
        if key in {"aiconfigurator", "aiconfigurator_core"} or key.startswith(
            ("aiconfigurator.", "aiconfigurator_core.")
        ):
            saved_aiconfigurator_modules[key] = sys.modules.pop(key)

    import aiconfigurator.sdk.perf_database as perf_database

    importlib.reload(perf_database)
    yield perf_database

    # Drop the isolated legacy/canonical module pair, then restore both module
    # graphs together so imported class references in other test modules do not
    # point at a reloaded canonical implementation.
    for key in list(sys.modules.keys()):
        if key in {"aiconfigurator", "aiconfigurator_core"} or key.startswith(
            ("aiconfigurator.", "aiconfigurator_core.")
        ):
            sys.modules.pop(key)
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
                    version_dir = backend_dir / version
                    version_dir.mkdir(exist_ok=True)
                    (version_dir / "gemm_perf.parquet").write_text("placeholder\n", encoding="utf-8")


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


def test_get_supported_databases_skips_incomplete_versions(temp_systems_dir: Path, perf_database):
    setup_mock_filesystem(temp_systems_dir, {"h100": {"vllm": ["0.14.0", "0.22.0"]}})
    incomplete_path = temp_systems_dir / "data_h100" / "vllm" / "0.22.0" / "INCOMPLETE.txt"
    incomplete_path.write_text("not enough profiling coverage\n", encoding="utf-8")

    result = perf_database.get_supported_databases(str(temp_systems_dir))

    assert result["h100"]["vllm"] == ["0.14.0"]
    assert perf_database.get_latest_database_version("h100", "vllm", systems_paths=str(temp_systems_dir)) == "0.14.0"


def test_get_supported_databases_includes_shared_layer_marker_versions(temp_systems_dir: Path, perf_database):
    setup_mock_filesystem(temp_systems_dir, {"h100": {"vllm": ["0.19.0"]}})
    marker_path = temp_systems_dir / "data_h100" / "vllm" / "0.22.0" / perf_database.SHARED_LAYER_REUSE_MARKER
    marker_path.parent.mkdir(parents=True)
    marker_path.write_text("declared shared-layer reuse\n", encoding="utf-8")
    (temp_systems_dir / "data_h100" / "vllm" / "0.23.0").mkdir(parents=True)

    result = perf_database.get_supported_databases(str(temp_systems_dir))

    assert result["h100"]["vllm"] == ["0.19.0", "0.22.0"]


def test_get_supported_databases_skips_incomplete_shared_layer_marker_versions(temp_systems_dir: Path, perf_database):
    setup_mock_filesystem(temp_systems_dir, {"h100": {"vllm": ["0.19.0"]}})
    marker_path = temp_systems_dir / "data_h100" / "vllm" / "0.22.0" / perf_database.SHARED_LAYER_REUSE_MARKER
    marker_path.parent.mkdir(parents=True)
    marker_path.write_text("declared shared-layer reuse\n", encoding="utf-8")
    (marker_path.parent / "INCOMPLETE.txt").write_text("not enough profiling coverage\n", encoding="utf-8")

    result = perf_database.get_supported_databases(str(temp_systems_dir))

    assert result["h100"]["vllm"] == ["0.19.0"]


def test_get_latest_database_version_skips_marker_only_versions_by_default(temp_systems_dir: Path, perf_database):
    setup_mock_filesystem(temp_systems_dir, {"h100": {"vllm": ["0.19.0"]}})
    marker_path = temp_systems_dir / "data_h100" / "vllm" / "0.22.0" / perf_database.SHARED_LAYER_REUSE_MARKER
    marker_path.parent.mkdir(parents=True)
    marker_path.write_text("declared shared-layer reuse\n", encoding="utf-8")

    assert perf_database.get_latest_database_version("h100", "vllm", systems_paths=str(temp_systems_dir)) == "0.19.0"
    assert (
        perf_database.get_latest_database_version(
            "h100",
            "vllm",
            systems_paths=str(temp_systems_dir),
            include_shared_layer_marker_versions=True,
        )
        == "0.22.0"
    )


def test_get_latest_database_version_uses_marked_versions_with_perf_data_by_default(
    temp_systems_dir: Path, perf_database
):
    setup_mock_filesystem(temp_systems_dir, {"h100": {"vllm": ["0.19.0"]}})
    marker_path = temp_systems_dir / "data_h100" / "vllm" / "0.22.0" / perf_database.SHARED_LAYER_REUSE_MARKER
    marker_path.parent.mkdir(parents=True)
    marker_path.write_text("declared shared-layer reuse\n", encoding="utf-8")
    (marker_path.parent / "generation_attention_perf.parquet").write_text("placeholder\n", encoding="utf-8")

    assert perf_database.get_latest_database_version("h100", "vllm", systems_paths=str(temp_systems_dir)) == "0.22.0"


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


def test_get_supported_databases_multiple_paths(temp_systems_dir: Path, perf_database):
    setup_mock_filesystem(temp_systems_dir, {"h100": {"trtllm": ["1.0.0"]}})
    other_dir = temp_systems_dir / "other"
    other_dir.mkdir()
    setup_mock_filesystem(other_dir, {"h200": {"vllm": ["0.6.0"]}})

    result = perf_database.get_supported_databases([str(temp_systems_dir), str(other_dir)])

    assert result["h100"]["trtllm"] == ["1.0.0"]
    assert result["h200"]["vllm"] == ["0.6.0"]


def test_build_no_databases_message_with_missing_path(temp_systems_dir: Path, perf_database):
    custom_systems_dir = temp_systems_dir / "custom_systems"
    custom_systems_dir.mkdir()
    previous_paths = perf_database.get_systems_paths()
    try:
        perf_database.set_systems_paths(str(custom_systems_dir))
        message = perf_database.build_no_databases_message()
        assert "No loadable performance databases found under --systems-paths" in message
        assert "Configured systems paths:" in message
        assert str(custom_systems_dir) in message
        assert "adding `default`" in message
    finally:
        perf_database.set_systems_paths(previous_paths)


def test_build_no_databases_message_with_existing_empty_path(temp_systems_dir: Path, perf_database):
    previous_paths = perf_database.get_systems_paths()
    try:
        perf_database.set_systems_paths("default")
        message = perf_database.build_no_databases_message()
        assert "No loadable performance databases found under --systems-paths" in message
        assert "Configured systems paths:" in message
        assert "already included" in message
    finally:
        perf_database.set_systems_paths(previous_paths)


def test_set_systems_paths_invalid_entry_raises(temp_systems_dir: Path, perf_database):
    missing_path = temp_systems_dir / "missing_systems_path"
    with pytest.raises(ValueError, match="Invalid --systems-paths"):
        perf_database.set_systems_paths(str(missing_path))


def test_estimate_only_database_can_load_without_perf_files(perf_database):
    """SOL/EMPIRICAL modes can instantiate from system specs without silicon files."""
    from aiconfigurator.sdk import common

    db = perf_database.get_database("h100_pcie", "trtllm", "estimate", allow_missing_data=True)

    assert db is not None
    db.set_default_database_mode(common.DatabaseMode.SOL)
    result = db.query_mem_op(1024)
    assert float(result) > 0


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


def test_get_latest_database_version_preserves_suffix_order(perf_database):
    mock_supported = {"h100": {"trtllm": ["v0.20_fix0719", "v0.20_fix0720"]}}

    original = perf_database.get_supported_databases
    try:
        perf_database.get_supported_databases = lambda: mock_supported
        latest = perf_database.get_latest_database_version("h100", "trtllm")
        assert latest == "v0.20_fix0720"
    finally:
        perf_database.get_supported_databases = original


def test_get_latest_database_version_two_part_rc_does_not_outrank_stable(perf_database):
    mock_supported = {"h100": {"trtllm": ["1.2.0", "1.2rc1"]}}

    original = perf_database.get_supported_databases
    try:
        perf_database.get_supported_databases = lambda systems_paths=None: mock_supported
        latest = perf_database.get_latest_database_version("h100", "trtllm")
        assert latest == "1.2.0"
    finally:
        perf_database.get_supported_databases = original


def test_get_database_prefers_measured_later_path_over_estimate_fallback(tmp_path: Path, perf_database, monkeypatch):
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    for root in (first_root, second_root):
        root.mkdir()
        (root / "data_sys").mkdir()
        (root / "sys.yaml").write_text(yaml.safe_dump({"data_dir": "data_sys"}), encoding="utf-8")
    (second_root / "data_sys" / "trtllm" / "1.0.0").mkdir(parents=True)

    class FakePerfDatabase:
        def __init__(self, system, backend, version, systems_root, database_mode=None):
            self.systems_root = systems_root

    monkeypatch.setattr(perf_database, "PerfDatabase", FakePerfDatabase)
    cache_snapshot = dict(perf_database.databases_cache)
    try:
        perf_database.databases_cache.clear()

        db = perf_database.get_database(
            "sys",
            "trtllm",
            "1.0.0",
            systems_paths=[str(first_root), str(second_root)],
            allow_missing_data=True,
        )

        assert db.systems_root == str(second_root)
    finally:
        perf_database.databases_cache.clear()
        perf_database.databases_cache.update(cache_snapshot)


def test_get_database_skips_incomplete_version_directory(tmp_path: Path, perf_database, monkeypatch):
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    for root in (first_root, second_root):
        root.mkdir()
        (root / "data_sys").mkdir()
        (root / "sys.yaml").write_text(yaml.safe_dump({"data_dir": "data_sys"}), encoding="utf-8")
    incomplete_path = first_root / "data_sys" / "trtllm" / "1.0.0"
    incomplete_path.mkdir(parents=True)
    (incomplete_path / "INCOMPLETE.txt").write_text("collection did not finish", encoding="utf-8")
    (second_root / "data_sys" / "trtllm" / "1.0.0").mkdir(parents=True)

    class FakePerfDatabase:
        def __init__(self, system, backend, version, systems_root, database_mode=None):
            self.systems_root = systems_root

    monkeypatch.setattr(perf_database, "PerfDatabase", FakePerfDatabase)
    cache_snapshot = dict(perf_database.databases_cache)
    try:
        perf_database.databases_cache.clear()

        db = perf_database.get_database(
            "sys",
            "trtllm",
            "1.0.0",
            systems_paths=[str(first_root), str(second_root)],
        )

        assert db.systems_root == str(second_root)
    finally:
        perf_database.databases_cache.clear()
        perf_database.databases_cache.update(cache_snapshot)


def test_perf_database_clear_runtime_caches_clears_interpolation_and_lru_state(perf_database):
    from aiconfigurator.sdk.perf_interp import engine as perf_interp_engine

    db = object.__new__(perf_database.PerfDatabase)
    cache_clear_calls = []

    def fake_cached_method():
        return None

    fake_cached_method.cache_clear = lambda: cache_clear_calls.append("cleared")
    db.fake_cached_method = fake_cached_method
    perf_interp_engine._SITE_INDEX_CACHE[id(db)] = (db, None)

    db.clear_runtime_caches()

    assert not perf_interp_engine._SITE_INDEX_CACHE
    assert cache_clear_calls == ["cleared"]


def test_empirical_and_silicon_views_use_distinct_shared_layer_templates(perf_database):
    """Formula-only views must not inherit sibling SILICON rows."""
    empirical = perf_database.get_database_view("b200_sxm", "trtllm", "1.3.0rc10", database_mode="EMPIRICAL")
    silicon = perf_database.get_database_view("b200_sxm", "trtllm", "1.3.0rc10", database_mode="SILICON")

    assert empirical._root_database_template is not silicon._root_database_template
    assert empirical.enable_shared_layer is False
    assert silicon.enable_shared_layer is True


def test_database_view_configuration_is_isolated_and_same_key_is_reused(perf_database):
    from aiconfigurator.sdk import common

    template = perf_database.get_database("b200_sxm", "trtllm", "1.3.0rc10", database_mode="SILICON")
    template.set_default_database_mode(common.DatabaseMode.SILICON)
    template.set_transfer_policy(None)
    template.clear_runtime_caches()
    shared_table = {"large": object()}
    template._test_loaded_table = shared_table

    try:
        view = perf_database.get_database_view(
            "b200_sxm",
            "trtllm",
            "1.3.0rc10",
            database_mode="HYBRID",
            transfer_policy="off",
        )
        same_view = perf_database.get_database_view(
            "b200_sxm",
            "trtllm",
            "1.3.0rc10",
            database_mode="HYBRID",
            transfer_policy="off",
        )
        aggressive = perf_database.get_database_view(
            "b200_sxm",
            "trtllm",
            "1.3.0rc10",
            database_mode="HYBRID",
            transfer_policy="aggressive",
        )

        assert view is same_view
        assert aggressive is not view
        assert view is not template
        assert view._root_database_template is template
        assert view._test_loaded_table is shared_table
        assert view.supported_quant_mode._database is view
        assert view.get_default_database_mode() is common.DatabaseMode.HYBRID
        assert view.transfer_policy == frozenset()
        assert aggressive.transfer_policy == common.ALL_TRANSFERS
        assert template.get_default_database_mode() is common.DatabaseMode.SILICON
        assert template.transfer_policy == common.ALL_TRANSFERS

        with pytest.raises(RuntimeError, match="immutable mode/policy"):
            view.set_transfer_policy(None)
        with pytest.raises(RuntimeError, match="immutable mode/policy"):
            view.set_default_database_mode(common.DatabaseMode.SILICON)
    finally:
        del template._test_loaded_table
        template.clear_runtime_caches()


def test_configured_view_cache_normalizes_keys_and_separates_roots(perf_database):
    import copy

    from aiconfigurator.sdk import common

    template = perf_database.get_database("b200_sxm", "trtllm", "1.3.0rc10", database_mode="SILICON")
    template.clear_runtime_caches()
    other_template = copy.copy(template)
    other_template._is_query_view = False

    view = perf_database._get_configured_database_view(template, "hybrid", None)
    enum_view = perf_database._get_configured_database_view(
        template,
        common.DatabaseMode.HYBRID,
        common.ALL_TRANSFERS,
    )
    list_view = perf_database._get_configured_database_view(
        template,
        common.DatabaseMode.HYBRID,
        list(common.TransferKind),
    )
    other_view = perf_database._get_configured_database_view(other_template, "HYBRID", "aggressive")

    assert view is enum_view is list_view
    assert other_view is not view
    assert view._root_database_template is template
    assert other_view._root_database_template is other_template
    template.clear_runtime_caches()


def test_clearing_template_runtime_caches_refreshes_configured_copy(perf_database):
    from aiconfigurator.sdk import common

    template = perf_database.get_database("b200_sxm", "trtllm", "1.3.0rc10", database_mode="SILICON")
    template.clear_runtime_caches()
    old_marker = object()
    new_marker = object()
    template._configured_view_marker = old_marker

    try:
        first = perf_database._get_configured_database_view(template, common.DatabaseMode.HYBRID, "off")
        template._configured_view_marker = new_marker
        cached = perf_database._get_configured_database_view(template, common.DatabaseMode.HYBRID, "off")

        assert cached is first
        assert cached._configured_view_marker is old_marker

        template.clear_runtime_caches()
        refreshed = perf_database._get_configured_database_view(template, common.DatabaseMode.HYBRID, "off")

        assert refreshed is not first
        assert refreshed._configured_view_marker is new_marker
    finally:
        del template._configured_view_marker
        template.clear_runtime_caches()


def test_configured_view_rejects_incompatible_shared_layer_template(perf_database):
    from aiconfigurator.sdk import common

    silicon_template = perf_database.get_database("b200_sxm", "trtllm", "1.3.0rc10", database_mode="SILICON")

    with pytest.raises(ValueError, match="use get_database_view"):
        perf_database._get_configured_database_view(silicon_template, common.DatabaseMode.EMPIRICAL)


def test_transfer_policy_and_mode_change_clear_global_grid_cache(perf_database):
    """In-place mode/policy mutation eagerly drops stale/unreachable grids,
    while no-op setter calls preserve the cache."""
    from aiconfigurator.sdk import common
    from aiconfigurator.sdk.operations import util_empirical

    db = perf_database.get_database("b200_sxm", "trtllm", "1.3.0rc10", database_mode="HYBRID")

    util_empirical._GRID_CACHE["__s1__"] = object()
    db.set_transfer_policy("conservative")  # != default ALL -> clears
    assert "__s1__" not in util_empirical._GRID_CACHE

    util_empirical._GRID_CACHE["__s2__"] = object()
    db.set_transfer_policy("conservative")  # no-op -> preserved
    assert "__s2__" in util_empirical._GRID_CACHE

    util_empirical._GRID_CACHE["__s3__"] = object()
    db.set_default_database_mode(common.DatabaseMode.EMPIRICAL)  # != HYBRID -> clears
    assert "__s3__" not in util_empirical._GRID_CACHE
    util_empirical._GRID_CACHE.clear()


def test_clear_database_runtime_caches_clears_matching_cached_database_once(perf_database):
    class FakeDatabase:
        def __init__(self):
            self.clear_count = 0

        def clear_runtime_caches(self):
            self.clear_count += 1

    keep_db = FakeDatabase()
    clear_db = FakeDatabase()
    cache_snapshot = dict(perf_database.databases_cache)
    try:
        perf_database.databases_cache.clear()
        perf_database.databases_cache[("root", "system", False, False)]["backend"]["1.0.0"] = clear_db
        perf_database.databases_cache[("root", "system", True, False)]["backend"]["1.0.0"] = clear_db
        perf_database.databases_cache[("root", "system", False, False)]["backend"]["2.0.0"] = keep_db
        perf_database.databases_cache[("root", "other-system", False, False)]["backend"]["1.0.0"] = keep_db

        perf_database.clear_database_runtime_caches("system", "backend", "1.0.0")

        assert clear_db.clear_count == 1
        assert keep_db.clear_count == 0
    finally:
        perf_database.databases_cache.clear()
        perf_database.databases_cache.update(cache_snapshot)


def test_unload_database_removes_matching_database_and_clears_runtime_cache(perf_database):
    class FakeDatabase:
        def __init__(self):
            self.clear_count = 0

        def clear_runtime_caches(self):
            self.clear_count += 1

    keep_db = FakeDatabase()
    unload_db = FakeDatabase()
    cache_snapshot = dict(perf_database.databases_cache)
    try:
        perf_database.databases_cache.clear()
        perf_database.databases_cache[("root", "system", False, False)]["backend"]["1.0.0"] = unload_db
        perf_database.databases_cache[("root", "system", False, False)]["backend"]["2.0.0"] = keep_db
        perf_database.databases_cache[("root", "other-system", False, False)]["backend"]["1.0.0"] = keep_db

        perf_database.unload_database("system", "backend", "1.0.0")

        assert unload_db.clear_count == 1
        assert perf_database.databases_cache[("root", "system", False, False)]["backend"] == {"2.0.0": keep_db}
        assert perf_database.databases_cache[("root", "other-system", False, False)]["backend"]["1.0.0"] is keep_db
    finally:
        perf_database.databases_cache.clear()
        perf_database.databases_cache.update(cache_snapshot)
