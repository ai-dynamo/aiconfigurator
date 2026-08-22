# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Queryable-version slots (systems/query_versions.yaml).

Each (system, framework) exposes at most three queryable versions:
authored current/previous plus a fleet-derived next (highest data-backed
version newer than current; override systems are frozen baselines with no
next). The literal aliases current/previous/next resolve anywhere a version is
requested; anything else fails loudly unless the caller (a test fixture
pinning data coordinates) passes allow_unlisted_version=True. Absent slots
file = gate off (synthetic/external trees unchanged).
"""

import os

import pytest
import yaml

from aiconfigurator_core.sdk import perf_database as pdb

pytestmark = pytest.mark.unit

_DOC = {
    "schema_version": 1,
    "defaults": {
        "trtllm": {"current": "1.3.0rc20", "previous": "1.2.0rc5"},
        "sglang": {"current": "0.5.14"},
    },
    "overrides": {
        "a100_sxm": {"trtllm": {"current": "1.0.0"}},
    },
}


def _touch_data(root, system, family, backend, version, table="gemm_perf.parquet"):
    d = root / "data" / system / family / backend / version
    os.makedirs(d, exist_ok=True)
    (d / table).write_bytes(b"")


@pytest.fixture
def systems_root(tmp_path):
    root = tmp_path / "systems"
    root.mkdir()
    for system in ("h200_sxm", "a100_sxm"):
        (root / f"{system}.yaml").write_text(f"data_dir: data/{system}\n")
    (root / "query_versions.yaml").write_text(yaml.dump(_DOC))
    # current + previous data on h200; a development drop at a newer version
    _touch_data(root, "h200_sxm", "gemm", "trtllm", "1.3.0rc20")
    _touch_data(root, "h200_sxm", "gemm", "trtllm", "1.2.0rc5")
    _touch_data(root, "h200_sxm", "moe", "trtllm", "1.3.0rc23", table="moe_perf.parquet")
    # a marker-only newer dir must NOT count as next
    d = root / "data" / "h200_sxm" / "mla" / "trtllm" / "1.4.0rc1"
    os.makedirs(d, exist_ok=True)
    (d / "reuse.yaml").write_text("schema_version: 1\nreuse: []\n")
    # a100 is an override baseline
    _touch_data(root, "a100_sxm", "gemm", "trtllm", "1.0.0")
    pdb._load_query_slots_doc.cache_clear()
    pdb._derive_fleet_next.cache_clear()
    yield str(root)
    pdb._load_query_slots_doc.cache_clear()
    pdb._derive_fleet_next.cache_clear()


def test_slots_resolution_and_fleet_dev(systems_root):
    slots = pdb.get_version_slots("h200_sxm", "trtllm", systems_root)
    assert slots == {"current": "1.3.0rc20", "previous": "1.2.0rc5", "next": "1.3.0rc23"}


def test_marker_only_dirs_do_not_qualify_as_dev(systems_root):
    slots = pdb.get_version_slots("h200_sxm", "trtllm", systems_root)
    assert slots["next"] != "1.4.0rc1"


def test_override_baseline_has_no_dev(systems_root):
    assert pdb.get_version_slots("a100_sxm", "trtllm", systems_root) == {"current": "1.0.0"}


def test_aliases_resolve(systems_root):
    r = pdb.resolve_query_version
    assert r("h200_sxm", "trtllm", "current", systems_root) == "1.3.0rc20"
    assert r("h200_sxm", "trtllm", "previous", systems_root) == "1.2.0rc5"
    assert r("h200_sxm", "trtllm", "next", systems_root) == "1.3.0rc23"


def test_slot_values_pass_and_unlisted_raise(systems_root):
    assert pdb.resolve_query_version("h200_sxm", "trtllm", "1.3.0rc20", systems_root) == "1.3.0rc20"
    with pytest.raises(ValueError, match="not a queryable version"):
        pdb.resolve_query_version("h200_sxm", "trtllm", "1.3.0rc10", systems_root)
    assert (
        pdb.resolve_query_version("h200_sxm", "trtllm", "1.3.0rc10", systems_root, allow_unlisted=True) == "1.3.0rc10"
    )


def test_missing_alias_raises_with_slots_listed(systems_root):
    with pytest.raises(ValueError, match="has no 'next'"):
        pdb.resolve_query_version("a100_sxm", "trtllm", "next", systems_root)


def test_system_without_data_keeps_gate_off(systems_root):
    # sglang defaults exist but neither system holds sglang data -> gate off,
    # estimate-only style requests pass through.
    assert pdb.get_version_slots("h200_sxm", "sglang", systems_root) is None
    assert pdb.resolve_query_version("h200_sxm", "sglang", "anything", systems_root) == "anything"


def test_absent_slots_file_disables_gate(tmp_path):
    root = tmp_path / "bare"
    root.mkdir()
    (root / "h200_sxm.yaml").write_text("data_dir: data/h200_sxm\n")
    pdb._load_query_slots_doc.cache_clear()
    try:
        assert pdb.get_version_slots("h200_sxm", "trtllm", str(root)) is None
        assert pdb.resolve_query_version("h200_sxm", "trtllm", "9.9.9", str(root)) == "9.9.9"
    finally:
        pdb._load_query_slots_doc.cache_clear()


def test_supported_databases_enumerate_slots(systems_root):
    sup = pdb.get_supported_databases(systems_paths=systems_root)
    assert sup["h200_sxm"]["trtllm"] == ["1.2.0rc5", "1.3.0rc20", "1.3.0rc23"]
    assert sup["a100_sxm"]["trtllm"] == ["1.0.0"]


def test_latest_is_current_not_dev(systems_root):
    assert pdb.get_latest_database_version("h200_sxm", "trtllm", systems_paths=systems_root) == "1.3.0rc20"
