# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for strict provenance mode (Collector V3 PR 4 task 2, AIC-1503,
design §5/§7.4).

``PerfDatabase(strict_provenance=...)`` resolves ``None`` from the
``AIC_STRICT_PROVENANCE`` env var. The actual load-time validation runs
inside ``get_database()`` (the loader entry point), scoped to dirs the
REQUESTED (system, backend, version) actually touches: its own family-layout
version dir(s) (primary) plus any donor dir their ``reuse.yaml`` declares
(channel 2, design §6.3). Constructing a bare ``PerfDatabase`` directly (as
the rest of this test package does against synthetic trees) performs no
validation of its own -- see ``test_bare_construction_never_validates``.

Strict mode raises ``ValueError`` (naming the offending dir/table) on: a
parquet-holding version dir with no ``collection_meta.yaml``; a table not
listed in its dir's sidecar; a malformed ``reuse.yaml``/``collection_meta.yaml``.
Non-strict warns instead of raising for all of the above. A
``provenance: legacy`` sidecar is graced -- warns, never raises -- in BOTH
modes. Legacy-layout (``<backend>/<version>``, no family segment) version
dirs predate the V3 metadata regime and are exempt from strict checking
entirely (mirrors ``_op_file_family_from_path``'s legacy-layout carve-out
from Task 1).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pytest
import yaml

from aiconfigurator.sdk import common
from aiconfigurator.sdk.operations.base import resolve_op_data_path
from aiconfigurator.sdk.perf_database import (
    _STRICT_VALIDATED_REQUESTS,
    PerfDatabase,
    _new_database_dict,
    _store_loaded_database,
    _strict_provenance_enabled,
    databases_cache,
    get_database,
    get_database_view,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers (mirrors tests/unit/sdk/database/test_reuse_ordering.py)
# ---------------------------------------------------------------------------

PARQUET_STUB = b"PAR1stub"


def _write(root: Path, rel: str, data: bytes = PARQUET_STUB) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)


def _write_yaml(root: Path, rel: str, doc: dict) -> None:
    _write(root, rel, yaml.safe_dump(doc).encode("utf-8"))


def _write_text(root: Path, rel: str, text: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def _reuse_entry(table: str, from_version: str, reason: str = "test donor") -> dict:
    return {"table": table, "from_version": from_version, "reason": reason, "approved_by": "yimingl"}


@pytest.fixture
def systems_root(tmp_path: Path) -> Path:
    """An ``h100_sxm`` systems tree with just a system YAML. Each test adds
    whatever data/sidecar files it needs under ``data/h100_sxm/``."""
    root = tmp_path / "systems"
    root.mkdir()
    (root / "h100_sxm.yaml").write_text("data_dir: data/h100_sxm\n", encoding="utf-8")
    return root


@pytest.fixture(autouse=True)
def _clear_databases_cache():
    databases_cache.clear()
    _STRICT_VALIDATED_REQUESTS.clear()
    yield
    databases_cache.clear()
    _STRICT_VALIDATED_REQUESTS.clear()


def _get_db(systems_root: Path, *, backend: str = "trtllm", version: str = "1.0.0", **kwargs):
    return get_database("h100_sxm", backend, version, systems_paths=str(systems_root), **kwargs)


# ---------------------------------------------------------------------------
# Env-var resolution
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", ["1", "true", "True", "TRUE"])
def test_env_var_truthy_values_enable_strict(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("AIC_STRICT_PROVENANCE", value)
    assert _strict_provenance_enabled(None) is True


@pytest.mark.parametrize("value", ["0", "false", "False", "", "no"])
def test_env_var_falsy_values_disable_strict(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("AIC_STRICT_PROVENANCE", value)
    assert _strict_provenance_enabled(None) is False


def test_env_var_unset_defaults_non_strict(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AIC_STRICT_PROVENANCE", raising=False)
    assert _strict_provenance_enabled(None) is False


def test_explicit_bool_overrides_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AIC_STRICT_PROVENANCE", "1")
    assert _strict_provenance_enabled(False) is False
    monkeypatch.setenv("AIC_STRICT_PROVENANCE", "0")
    assert _strict_provenance_enabled(True) is True


def test_perf_database_stores_resolved_flag(systems_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AIC_STRICT_PROVENANCE", raising=False)
    db_default = PerfDatabase("h100_sxm", "trtllm", "1.0.0", str(systems_root))
    assert db_default.strict_provenance is False

    db_true = PerfDatabase("h100_sxm", "trtllm", "1.0.0", str(systems_root), strict_provenance=True)
    assert db_true.strict_provenance is True

    monkeypatch.setenv("AIC_STRICT_PROVENANCE", "1")
    db_env = PerfDatabase("h100_sxm", "trtllm", "1.0.0", str(systems_root))
    assert db_env.strict_provenance is True


# ---------------------------------------------------------------------------
# Bare PerfDatabase() construction never validates (scope decision)
# ---------------------------------------------------------------------------


def test_bare_construction_never_validates(systems_root: Path) -> None:
    """Constructing a ``PerfDatabase`` directly against a tree with a
    parquet-holding dir and no ``collection_meta.yaml`` must NOT raise, even
    with ``strict_provenance=True`` -- the fail-closed validation is a
    ``get_database()`` (loader entry point) concern, not a constructor
    concern. This is deliberate: the rest of this test package (and the
    wider unit suite) constructs ``PerfDatabase`` directly against synthetic
    trees that predate ``collection_meta.yaml`` sidecars, and must keep
    working unchanged under CI's ``AIC_STRICT_PROVENANCE=1``.
    """
    _write(systems_root, "data/h100_sxm/gemm/trtllm/1.0.0/gemm_perf.parquet")

    db = PerfDatabase("h100_sxm", "trtllm", "1.0.0", str(systems_root), strict_provenance=True)
    assert db.strict_provenance is True  # flag is stored...
    # ...but construction itself did not validate anything (no raise above).


# ---------------------------------------------------------------------------
# Missing collection_meta.yaml sidecar
# ---------------------------------------------------------------------------


def test_strict_raises_on_missing_sidecar(systems_root: Path) -> None:
    _write(systems_root, "data/h100_sxm/gemm/trtllm/1.0.0/gemm_perf.parquet")

    with pytest.raises(ValueError, match=r"no collection_meta\.yaml"):
        _get_db(systems_root, strict_provenance=True)


def test_non_strict_warns_on_missing_sidecar(systems_root: Path, caplog: pytest.LogCaptureFixture) -> None:
    _write(systems_root, "data/h100_sxm/gemm/trtllm/1.0.0/gemm_perf.parquet")

    with caplog.at_level(logging.WARNING):
        db = _get_db(systems_root, strict_provenance=False)

    assert db is not None
    assert any("no collection_meta.yaml" in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------------------
# Table present but uncovered by the dir's sidecar
# ---------------------------------------------------------------------------


def test_strict_raises_on_uncovered_table(systems_root: Path) -> None:
    _write(systems_root, "data/h100_sxm/attention/trtllm/1.0.0/context_attention_perf.parquet")
    _write(systems_root, "data/h100_sxm/attention/trtllm/1.0.0/generation_attention_perf.parquet")
    _write_yaml(
        systems_root,
        "data/h100_sxm/attention/trtllm/1.0.0/collection_meta.yaml",
        {
            "schema_version": 1,
            "runtime": {"framework": "trtllm", "version": "1.0.0"},
            "tables": {"context_attention_perf": {"status": "complete"}},
        },
    )

    with pytest.raises(ValueError, match="generation_attention_perf"):
        _get_db(systems_root, strict_provenance=True)


def test_non_strict_warns_on_uncovered_table(systems_root: Path, caplog: pytest.LogCaptureFixture) -> None:
    _write(systems_root, "data/h100_sxm/attention/trtllm/1.0.0/context_attention_perf.parquet")
    _write(systems_root, "data/h100_sxm/attention/trtllm/1.0.0/generation_attention_perf.parquet")
    _write_yaml(
        systems_root,
        "data/h100_sxm/attention/trtllm/1.0.0/collection_meta.yaml",
        {
            "schema_version": 1,
            "runtime": {"framework": "trtllm", "version": "1.0.0"},
            "tables": {"context_attention_perf": {"status": "complete"}},
        },
    )

    with caplog.at_level(logging.WARNING):
        db = _get_db(systems_root, strict_provenance=False)

    assert db is not None
    assert any(
        "generation_attention_perf" in r.getMessage() and "not covered" in r.getMessage() for r in caplog.records
    )


def test_fully_covered_sidecar_raises_nothing_in_strict(systems_root: Path) -> None:
    _write(systems_root, "data/h100_sxm/gemm/trtllm/1.0.0/gemm_perf.parquet")
    _write_yaml(
        systems_root,
        "data/h100_sxm/gemm/trtllm/1.0.0/collection_meta.yaml",
        {
            "schema_version": 1,
            "runtime": {"framework": "trtllm", "version": "1.0.0"},
            "tables": {"gemm_perf": {"status": "complete"}},
        },
    )

    db = _get_db(systems_root, strict_provenance=True)
    assert db is not None


# ---------------------------------------------------------------------------
# Malformed sidecars: strict surfaces the existing ValueError, non-strict warns
# ---------------------------------------------------------------------------


def test_strict_raises_on_malformed_collection_meta(systems_root: Path) -> None:
    _write(systems_root, "data/h100_sxm/gemm/trtllm/1.0.0/gemm_perf.parquet")
    _write_text(systems_root, "data/h100_sxm/gemm/trtllm/1.0.0/collection_meta.yaml", "not a mapping\n- oops\n")

    with pytest.raises(ValueError, match=r"collection_meta\.yaml"):
        _get_db(systems_root, strict_provenance=True)


def test_non_strict_warns_on_malformed_collection_meta(systems_root: Path, caplog: pytest.LogCaptureFixture) -> None:
    _write(systems_root, "data/h100_sxm/gemm/trtllm/1.0.0/gemm_perf.parquet")
    _write_text(systems_root, "data/h100_sxm/gemm/trtllm/1.0.0/collection_meta.yaml", "not a mapping\n- oops\n")

    with caplog.at_level(logging.WARNING):
        db = _get_db(systems_root, strict_provenance=False)

    assert db is not None
    assert any("collection_meta.yaml" in r.getMessage() for r in caplog.records)


def test_strict_raises_on_malformed_reuse_yaml(systems_root: Path) -> None:
    _write(systems_root, "data/h100_sxm/gemm/trtllm/1.0.0/gemm_perf.parquet")
    _write_yaml(
        systems_root,
        "data/h100_sxm/gemm/trtllm/1.0.0/collection_meta.yaml",
        {
            "schema_version": 1,
            "runtime": {"framework": "trtllm", "version": "1.0.0"},
            "tables": {"gemm_perf": {"status": "complete"}},
        },
    )
    _write_yaml(
        systems_root,
        "data/h100_sxm/gemm/trtllm/1.0.0/reuse.yaml",
        {"reuse": [{"table": "gemm_perf"}]},  # missing from_version/reason/approved_by
    )

    with pytest.raises(ValueError, match="missing required key"):
        _get_db(systems_root, strict_provenance=True)


def test_non_strict_warns_on_malformed_reuse_yaml(systems_root: Path, caplog: pytest.LogCaptureFixture) -> None:
    _write(systems_root, "data/h100_sxm/gemm/trtllm/1.0.0/gemm_perf.parquet")
    _write_yaml(
        systems_root,
        "data/h100_sxm/gemm/trtllm/1.0.0/collection_meta.yaml",
        {
            "schema_version": 1,
            "runtime": {"framework": "trtllm", "version": "1.0.0"},
            "tables": {"gemm_perf": {"status": "complete"}},
        },
    )
    _write_yaml(
        systems_root,
        "data/h100_sxm/gemm/trtllm/1.0.0/reuse.yaml",
        {"reuse": [{"table": "gemm_perf"}]},
    )

    with caplog.at_level(logging.WARNING):
        db = _get_db(systems_root, strict_provenance=False)

    assert db is not None
    assert any("missing required key" in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------------------
# Malformed reuse.yaml: load-time warn must not leave a query-time crash
# behind (AIC-1503 PR4 task 5, FIX 1). Before this fix, ``get_database()``'s
# own load-time walk (above) warned and returned a usable non-strict
# database, but ``_build_op_sources`` -> ``_requested_version_reuse_entries``
# parsed the SAME malformed reuse.yaml with no strict/non-strict distinction
# and always raised -- so the first real op query against that database
# crashed anyway.
# ---------------------------------------------------------------------------


def test_non_strict_query_after_load_does_not_crash_on_malformed_reuse_yaml(
    systems_root: Path, caplog: pytest.LogCaptureFixture
) -> None:
    requested, earlier = "1.0.0", "0.9.0"
    _write(systems_root, f"data/h100_sxm/gemm/trtllm/{requested}/gemm_perf.parquet")
    _write(systems_root, f"data/h100_sxm/gemm/trtllm/{earlier}/gemm_perf.parquet")
    _write_yaml(
        systems_root,
        f"data/h100_sxm/gemm/trtllm/{requested}/reuse.yaml",
        {"reuse": [{"table": "gemm_perf"}]},  # missing from_version/reason/approved_by
    )

    system_data_root = str(systems_root / "data" / "h100_sxm")
    primary_path = resolve_op_data_path(system_data_root, "trtllm", requested, "gemm_perf.parquet")

    with caplog.at_level(logging.WARNING):
        db = _get_db(systems_root, version=requested, strict_provenance=False)
        assert db is not None
        # Simulates the first real op query against the now-loaded database --
        # must not raise, and the nearest-earlier fallback channel (design
        # §6.2) must still admit rows despite the declared-reuse channel
        # being unusable.
        sources = db._build_op_sources(common.PerfDataFilename.gemm, primary_path, system_data_root)

    assert [entry["channel"] for entry in db.data_provenance["gemm_perf.parquet"]] == ["primary", "fallback"]
    assert sources[-1][0].endswith(f"gemm/trtllm/{earlier}/gemm_perf.parquet")
    assert any("missing required key" in r.getMessage() for r in caplog.records)


def test_strict_query_raises_on_malformed_reuse_yaml_even_with_bare_construction(systems_root: Path) -> None:
    """The strict check inside ``_requested_version_reuse_entries`` itself
    raises too, not just ``get_database()``'s separate load-time walk --
    proven by constructing ``PerfDatabase`` directly (bare construction never
    validates, see ``test_bare_construction_never_validates``) so
    ``get_database()``'s own pre-check is bypassed entirely."""
    requested, earlier = "1.0.0", "0.9.0"
    _write(systems_root, f"data/h100_sxm/gemm/trtllm/{requested}/gemm_perf.parquet")
    _write(systems_root, f"data/h100_sxm/gemm/trtllm/{earlier}/gemm_perf.parquet")
    _write_yaml(
        systems_root,
        f"data/h100_sxm/gemm/trtllm/{requested}/reuse.yaml",
        {"reuse": [{"table": "gemm_perf"}]},
    )

    db = PerfDatabase("h100_sxm", "trtllm", requested, str(systems_root), strict_provenance=True)
    system_data_root = str(systems_root / "data" / "h100_sxm")
    primary_path = resolve_op_data_path(system_data_root, "trtllm", requested, "gemm_perf.parquet")

    with pytest.raises(ValueError, match="missing required key"):
        db._build_op_sources(common.PerfDataFilename.gemm, primary_path, system_data_root)


# ---------------------------------------------------------------------------
# Declared-reuse donor dirs (channel 2, design §6.3) are admitted sources too
# ---------------------------------------------------------------------------


def test_strict_raises_on_declared_donor_uncovered_table(systems_root: Path) -> None:
    requested, donor = "1.0.0", "0.9.0"
    _write(systems_root, f"data/h100_sxm/moe/trtllm/{requested}/moe_perf.parquet")
    _write_yaml(
        systems_root,
        f"data/h100_sxm/moe/trtllm/{requested}/collection_meta.yaml",
        {
            "schema_version": 1,
            "runtime": {"framework": "trtllm", "version": requested},
            "tables": {"moe_perf": {"status": "complete"}},
        },
    )
    _write_yaml(
        systems_root,
        f"data/h100_sxm/moe/trtllm/{requested}/reuse.yaml",
        {"reuse": [_reuse_entry("wideep_moe_perf", donor)]},
    )
    # Donor exists physically but its own sidecar does not cover wideep_moe_perf.
    _write(systems_root, f"data/h100_sxm/moe/trtllm/{donor}/wideep_moe_perf.parquet")
    _write_yaml(
        systems_root,
        f"data/h100_sxm/moe/trtllm/{donor}/collection_meta.yaml",
        {
            "schema_version": 1,
            "runtime": {"framework": "trtllm", "version": donor},
            "tables": {},
        },
    )

    with pytest.raises(ValueError, match="wideep_moe_perf"):
        _get_db(systems_root, version=requested, strict_provenance=True)


def test_non_strict_warns_on_declared_donor_missing_sidecar(
    systems_root: Path, caplog: pytest.LogCaptureFixture
) -> None:
    requested, donor = "1.0.0", "0.9.0"
    _write(systems_root, f"data/h100_sxm/moe/trtllm/{requested}/moe_perf.parquet")
    _write_yaml(
        systems_root,
        f"data/h100_sxm/moe/trtllm/{requested}/collection_meta.yaml",
        {
            "schema_version": 1,
            "runtime": {"framework": "trtllm", "version": requested},
            "tables": {"moe_perf": {"status": "complete"}},
        },
    )
    _write_yaml(
        systems_root,
        f"data/h100_sxm/moe/trtllm/{requested}/reuse.yaml",
        {"reuse": [_reuse_entry("wideep_moe_perf", donor)]},
    )
    # Donor exists physically with no collection_meta.yaml at all.
    _write(systems_root, f"data/h100_sxm/moe/trtllm/{donor}/wideep_moe_perf.parquet")

    with caplog.at_level(logging.WARNING):
        db = _get_db(systems_root, version=requested, strict_provenance=False)

    assert db is not None
    assert any(
        "wideep_moe_perf" in r.getMessage() and "no collection_meta.yaml" in r.getMessage() for r in caplog.records
    )


# ---------------------------------------------------------------------------
# provenance: legacy grace (design §5) -- warns, never raises, in BOTH modes
# ---------------------------------------------------------------------------


def test_legacy_provenance_uncovered_table_warns_not_raises_in_strict(
    systems_root: Path, caplog: pytest.LogCaptureFixture
) -> None:
    _write(systems_root, "data/h100_sxm/gemm/trtllm/1.0.0/gemm_perf.parquet")
    _write_yaml(
        systems_root,
        "data/h100_sxm/gemm/trtllm/1.0.0/collection_meta.yaml",
        {
            "schema_version": 1,
            "provenance": "legacy",
            "runtime": {"framework": "trtllm", "version": "1.0.0"},
            "tables": {},  # does not list gemm_perf
        },
    )

    with caplog.at_level(logging.WARNING):
        db = _get_db(systems_root, strict_provenance=True)  # must NOT raise

    assert db is not None
    assert any("graced" in r.getMessage() for r in caplog.records)


def test_legacy_provenance_uncovered_table_warns_in_non_strict_too(
    systems_root: Path, caplog: pytest.LogCaptureFixture
) -> None:
    _write(systems_root, "data/h100_sxm/gemm/trtllm/1.0.0/gemm_perf.parquet")
    _write_yaml(
        systems_root,
        "data/h100_sxm/gemm/trtllm/1.0.0/collection_meta.yaml",
        {
            "schema_version": 1,
            "provenance": "legacy",
            "runtime": {"framework": "trtllm", "version": "1.0.0"},
            "tables": {},
        },
    )

    with caplog.at_level(logging.WARNING):
        db = _get_db(systems_root, strict_provenance=False)

    assert db is not None
    assert any("graced" in r.getMessage() for r in caplog.records)


def test_legacy_provenance_grace_does_not_extend_to_missing_sidecar(systems_root: Path) -> None:
    """Grace applies to a sidecar that EXISTS and says legacy; a MISSING
    sidecar is not a "legacy sidecar" and still raises in strict mode."""
    _write(systems_root, "data/h100_sxm/gemm/trtllm/1.0.0/gemm_perf.parquet")

    with pytest.raises(ValueError, match=r"no collection_meta\.yaml"):
        _get_db(systems_root, strict_provenance=True)


# ---------------------------------------------------------------------------
# Legacy tree LAYOUT (no family segment) is exempt from strict checking
# ---------------------------------------------------------------------------


def test_legacy_layout_primary_dir_exempt_from_strict(systems_root: Path) -> None:
    """<data_dir>/<backend>/<version> (no family dir) predates the V3
    metadata regime entirely; strict mode must not require a
    collection_meta.yaml there (mirrors test_perf_database_shared_layer.py's
    synthetic trees, which use this exact layout)."""
    _write(systems_root, "data/h100_sxm/trtllm/1.0.0/gemm_perf.parquet")

    db = _get_db(systems_root, strict_provenance=True)
    assert db is not None


# ---------------------------------------------------------------------------
# get_database_view threads strict_provenance through to get_database
# ---------------------------------------------------------------------------


def test_get_database_view_threads_strict_provenance(systems_root: Path) -> None:
    _write(systems_root, "data/h100_sxm/gemm/trtllm/1.0.0/gemm_perf.parquet")

    with pytest.raises(ValueError, match=r"no collection_meta\.yaml"):
        get_database_view("h100_sxm", "trtllm", "1.0.0", systems_paths=str(systems_root), strict_provenance=True)

    view = get_database_view("h100_sxm", "trtllm", "1.0.0", systems_paths=str(systems_root), strict_provenance=False)
    assert view is not None


# ---------------------------------------------------------------------------
# Cache keying: strict_provenance must not collide with non-strict entries
# (AIC-1503 review defect: databases_cache was keyed by
# (systems_root, system, shared_flag) only -- not by strict_provenance -- so
# a strict call for the same (system, backend, version) could silently
# return an earlier, unvalidated non-strict instance instead of
# re-validating and raising.)
# ---------------------------------------------------------------------------


def test_non_strict_load_then_strict_call_does_not_reuse_stale_cache_and_raises(
    systems_root: Path, caplog: pytest.LogCaptureFixture
) -> None:
    _write(systems_root, "data/h100_sxm/gemm/trtllm/1.0.0/gemm_perf.parquet")

    with caplog.at_level(logging.WARNING):
        non_strict_db = _get_db(systems_root, strict_provenance=False)
    assert non_strict_db is not None
    assert non_strict_db.strict_provenance is False

    # Same (system, backend, version) triple, strict this time: must
    # re-validate against the coverage-gap tree and raise -- not return the
    # cached non-strict instance.
    with pytest.raises(ValueError, match=r"no collection_meta\.yaml"):
        _get_db(systems_root, strict_provenance=True)


def test_strict_load_then_non_strict_load_are_distinct_cache_entries(systems_root: Path) -> None:
    """Control: a fully-covered tree loads cleanly under both modes; each
    resolves to its own cache entry and both remain independently usable."""
    _write(systems_root, "data/h100_sxm/gemm/trtllm/1.0.0/gemm_perf.parquet")
    _write_yaml(
        systems_root,
        "data/h100_sxm/gemm/trtllm/1.0.0/collection_meta.yaml",
        {
            "schema_version": 1,
            "runtime": {"framework": "trtllm", "version": "1.0.0"},
            "tables": {"gemm_perf": {"status": "complete"}},
        },
    )

    strict_db = _get_db(systems_root, strict_provenance=True)
    non_strict_db = _get_db(systems_root, strict_provenance=False)

    assert strict_db is not None
    assert non_strict_db is not None
    assert strict_db is not non_strict_db
    assert strict_db.strict_provenance is True
    assert non_strict_db.strict_provenance is False

    # Re-requesting each mode returns its own cached instance, not the
    # other mode's -- no eviction/overwrite across the strict boundary.
    assert _get_db(systems_root, strict_provenance=True) is strict_db
    assert _get_db(systems_root, strict_provenance=False) is non_strict_db


# ---------------------------------------------------------------------------
# Cache-hit path still validates under strict mode (worker-imported entries)
# ---------------------------------------------------------------------------


def test_strict_cache_hit_validates_worker_imported_unvalidated_entry(systems_root: Path) -> None:
    """``get_all_databases()`` imports worker-created databases via
    ``_store_loaded_database`` under a strict cache key WITHOUT any request
    validation (direct construction validates nothing, see
    ``test_bare_construction_never_validates``). A later strict
    ``get_database()`` call that hits that cache entry must still run
    ``_check_strict_provenance_for_request`` and raise on the coverage gap,
    not hand back the unvalidated instance."""
    _write(systems_root, "data/h100_sxm/gemm/trtllm/1.0.0/gemm_perf.parquet")  # no sidecar

    db = PerfDatabase("h100_sxm", "trtllm", "1.0.0", str(systems_root), strict_provenance=True)
    _store_loaded_database(_new_database_dict(), ("h100_sxm", "trtllm", "1.0.0", str(systems_root)), db)

    with pytest.raises(ValueError, match=r"no collection_meta\.yaml"):
        _get_db(systems_root, strict_provenance=True)


# ---------------------------------------------------------------------------
# Unreadable version dir: strict fails closed instead of silently accepting
# an empty file list (the pre-fix _version_dir_data_filenames swallowed
# os.listdir failures entirely)
# ---------------------------------------------------------------------------


def _chmod_000_or_skip(path: Path) -> None:
    if os.geteuid() == 0:
        pytest.skip("chmod 000 does not restrict root")
    path.chmod(0o000)


def test_strict_raises_on_unreadable_version_dir(systems_root: Path) -> None:
    _write(systems_root, "data/h100_sxm/gemm/trtllm/1.0.0/gemm_perf.parquet")
    version_dir = systems_root / "data/h100_sxm/gemm/trtllm/1.0.0"
    _chmod_000_or_skip(version_dir)
    try:
        with pytest.raises(ValueError, match="cannot inspect perf-data files"):
            _get_db(systems_root, strict_provenance=True)
    finally:
        version_dir.chmod(0o755)


def test_non_strict_warns_on_unreadable_version_dir(systems_root: Path, caplog: pytest.LogCaptureFixture) -> None:
    _write(systems_root, "data/h100_sxm/gemm/trtllm/1.0.0/gemm_perf.parquet")
    version_dir = systems_root / "data/h100_sxm/gemm/trtllm/1.0.0"
    _chmod_000_or_skip(version_dir)
    try:
        with caplog.at_level(logging.WARNING):
            db = _get_db(systems_root, strict_provenance=False)
    finally:
        version_dir.chmod(0o755)

    assert db is not None
    assert any("cannot inspect perf-data files" in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------------------
# Real-tree smoke: post-PR3 tree is fully covered, strict ON loads clean
# ---------------------------------------------------------------------------


def test_real_tree_smoke_strict_on() -> None:
    databases_cache.clear()
    try:
        db = get_database("h200_sxm", "trtllm", "1.3.0rc10", strict_provenance=True)
        assert db is not None
        assert db.strict_provenance is True
    finally:
        databases_cache.clear()
