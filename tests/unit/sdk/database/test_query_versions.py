# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Declared queryable-version list (query_versions.yaml) resolution.

The list — not the set of version directories — defines queryability:
listed versions pass; unlisted versions with a surviving directory pass
under a deprecation warning; unlisted versions with no directory soft-map
to the platform primary instead of failing.
"""

import os

import pytest
import yaml

from aiconfigurator_core.sdk import perf_database as pdb

pytestmark = pytest.mark.unit

_DOC = {
    "schema_version": 1,
    "defaults": {
        "trtllm": ["1.3.0rc20", "1.3.0rc23"],
        "sglang": ["0.5.14", "0.5.16"],
        "vllm": ["0.24.0"],
    },
    "overrides": {
        "a100_sxm": {"trtllm": ["1.0.0"], "sglang": ["0.5.10"], "vllm": ["0.14.0"]},
        "b60": {"vllm": ["0.20.0"]},
    },
}


@pytest.fixture
def systems_root(tmp_path):
    """A minimal systems root: one system spec, a data tree, and the list."""
    root = tmp_path / "systems"
    root.mkdir()
    (root / "h200_sxm.yaml").write_text("data_dir: data/h200_sxm\n")
    (root / "a100_sxm.yaml").write_text("data_dir: data/a100_sxm\n")
    (root / "query_versions.yaml").write_text(yaml.dump(_DOC))
    # a surviving unlisted version dir (grace case)
    os.makedirs(root / "data" / "h200_sxm" / "moe" / "trtllm" / "1.3.0rc10")
    pdb._load_query_versions.cache_clear()
    yield str(root)
    pdb._load_query_versions.cache_clear()


def test_listed_version_passes_unchanged(systems_root):
    assert pdb.resolve_query_version("h200_sxm", "trtllm", "1.3.0rc20", systems_root) == "1.3.0rc20"
    assert pdb.resolve_query_version("h200_sxm", "trtllm", "1.3.0rc23", systems_root) == "1.3.0rc23"


def test_override_beats_default(systems_root):
    assert pdb.get_query_versions("a100_sxm", "trtllm", systems_root) == ["1.0.0"]
    assert pdb.get_query_versions("h200_sxm", "trtllm", systems_root) == ["1.3.0rc20", "1.3.0rc23"]
    # a100's own primary passes; the global primary is NOT in a100's list but
    # has no dir either -> maps to a100's primary
    assert pdb.resolve_query_version("a100_sxm", "trtllm", "1.0.0", systems_root) == "1.0.0"
    assert pdb.resolve_query_version("a100_sxm", "trtllm", "1.2.0rc5", systems_root) == "1.0.0"


def test_unlisted_with_surviving_dir_passes_with_warning(systems_root, caplog):
    with caplog.at_level("WARNING"):
        got = pdb.resolve_query_version("h200_sxm", "trtllm", "1.3.0rc10", systems_root)
    assert got == "1.3.0rc10"
    assert any("not a declared queryable version" in r.message for r in caplog.records)


def test_unlisted_without_dir_maps_to_primary(systems_root, caplog):
    with caplog.at_level("WARNING"):
        got = pdb.resolve_query_version("h200_sxm", "trtllm", "0.0.0rc999", systems_root)
    assert got == "1.3.0rc20"
    assert any("mapping to the platform primary" in r.message for r in caplog.records)


def test_absent_list_disables_mapping(tmp_path):
    root = tmp_path / "bare"
    root.mkdir()
    (root / "h200_sxm.yaml").write_text("data_dir: data/h200_sxm\n")
    pdb._load_query_versions.cache_clear()
    try:
        assert pdb.get_query_versions("h200_sxm", "trtllm", str(root)) is None
        assert pdb.resolve_query_version("h200_sxm", "trtllm", "anything", str(root)) == "anything"
    finally:
        pdb._load_query_versions.cache_clear()


def test_shipped_list_matches_shipped_primaries():
    """The checked-in list's primaries must hold data in the shipped tree."""
    versions = pdb.get_query_versions("h200_sxm", "trtllm")
    assert versions and versions[0] == "1.3.0rc20"
    db = pdb.get_database("h200_sxm", "trtllm", versions[0])
    assert db is not None
