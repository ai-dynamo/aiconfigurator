# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for collector version resolver and registry infrastructure."""

from typing import ClassVar

import pytest

from collector.version_resolver import (
    _check_compat,
    _normalize_version,
    build_collections,
    resolve_module,
)


# ---------------------------------------------------------------------------
# _normalize_version
# ---------------------------------------------------------------------------
@pytest.mark.unit
class TestNormalizeVersion:
    """Verify PEP 440-aware version tuple generation."""

    @pytest.mark.parametrize(
        "version_str,expected",
        [
            ("1.2.0", (1, 2, 0, 0, 0)),
            ("0.20.0", (0, 20, 0, 0, 0)),
            ("0.5.5", (0, 5, 5, 0, 0)),
        ],
    )
    def test_final_release(self, version_str, expected):
        assert _normalize_version(version_str) == expected

    @pytest.mark.parametrize(
        "version_str,expected",
        [
            ("1.2.0dev1", (1, 2, 0, -4, 1)),
            ("1.2.0a2", (1, 2, 0, -3, 2)),
            ("1.2.0b1", (1, 2, 0, -2, 1)),
            ("1.2.0rc2", (1, 2, 0, -1, 2)),
        ],
    )
    def test_pre_release(self, version_str, expected):
        assert _normalize_version(version_str) == expected

    @pytest.mark.parametrize(
        "version_str,expected",
        [
            ("0.5.5.post2", (0, 5, 5, 1, 2)),
            ("1.1.0.post1", (1, 1, 0, 1, 1)),
        ],
    )
    def test_post_release(self, version_str, expected):
        assert _normalize_version(version_str) == expected

    def test_local_metadata_ignored(self):
        assert _normalize_version("0.20.0+cu124") == _normalize_version("0.20.0")

    def test_ordering_dev_to_post(self):
        ordered = ["1.2.0dev1", "1.2.0a2", "1.2.0b1", "1.2.0rc2", "1.2.0", "1.2.0.post2"]
        tuples = [_normalize_version(v) for v in ordered]
        assert tuples == sorted(tuples)

    def test_rc_less_than_release(self):
        assert _normalize_version("1.1.0rc5") < _normalize_version("1.1.0")

    def test_post_greater_than_release(self):
        assert _normalize_version("0.5.5.post2") > _normalize_version("0.5.5")


# ---------------------------------------------------------------------------
# _check_compat
# ---------------------------------------------------------------------------
@pytest.mark.unit
class TestCheckCompat:
    """Verify __compat__ constraint evaluation."""

    @pytest.mark.parametrize(
        "compat,runtime,expected",
        [
            # Simple lower bound
            ("trtllm>=1.1.0", "1.1.0", True),
            ("trtllm>=1.1.0", "1.3.0", True),
            ("trtllm>=1.1.0", "1.0.0", False),
            # rc is pre-release, does not satisfy >=release
            ("trtllm>=1.1.0", "1.1.0rc2", False),
            # post-release satisfies >= release
            ("trtllm>=1.1.0", "1.1.0.post1", True),
            # Range constraint
            ("trtllm>=0.21.0,<1.1.0", "1.0.0", True),
            ("trtllm>=0.21.0,<1.1.0", "1.1.0", False),
            ("trtllm>=0.21.0,<1.1.0", "0.20.0", False),
            # rc falls below upper bound
            ("trtllm<1.1.0", "1.1.0rc2", True),
            # Tight range
            ("trtllm>=0.20.0,<0.21.0", "0.20.0", True),
            ("trtllm>=0.20.0,<0.21.0", "0.21.0", False),
            # SGLang post versions
            ("sglang>=0.5.5", "0.5.5.post2", True),
            ("sglang>=0.5.5", "0.5.4", False),
            # vLLM
            ("vllm>=0.11.0", "0.14.0", True),
            ("vllm>=0.11.0", "0.10.0", False),
        ],
    )
    def test_compat_constraints(self, compat, runtime, expected):
        assert _check_compat(compat, runtime) == expected


# ---------------------------------------------------------------------------
# resolve_module
# ---------------------------------------------------------------------------
@pytest.mark.unit
class TestResolveModule:
    """Verify version-based module resolution from registry entries."""

    VERSIONED_ENTRY: ClassVar[dict] = {
        "op": "moe",
        "get_func": "get_moe_test_cases",
        "run_func": "run_moe_torch",
        "versions": [
            ("1.1.0", "collector.trtllm.collect_moe_v3"),
            ("0.21.0", "collector.trtllm.collect_moe_v2"),
            ("0.20.0", "collector.trtllm.collect_moe_v1"),
        ],
    }

    UNVERSIONED_ENTRY: ClassVar[dict] = {
        "op": "gemm",
        "module": "collector.trtllm.collect_gemm",
        "get_func": "get_gemm_test_cases",
        "run_func": "run_gemm",
    }

    def test_unversioned_returns_module_directly(self):
        assert resolve_module(self.UNVERSIONED_ENTRY, "999.0.0") == "collector.trtllm.collect_gemm"

    @pytest.mark.parametrize(
        "runtime,expected_module",
        [
            ("1.3.0", "collector.trtllm.collect_moe_v3"),
            ("1.1.0", "collector.trtllm.collect_moe_v3"),
            ("1.0.0", "collector.trtllm.collect_moe_v2"),
            ("0.21.0", "collector.trtllm.collect_moe_v2"),
            ("0.20.0", "collector.trtllm.collect_moe_v1"),
        ],
    )
    def test_versioned_routing(self, runtime, expected_module):
        assert resolve_module(self.VERSIONED_ENTRY, runtime) == expected_module

    def test_unsupported_version_returns_none(self):
        assert resolve_module(self.VERSIONED_ENTRY, "0.19.0") is None

    def test_rc_routes_to_previous(self):
        """1.1.0rc2 < 1.1.0, so it should fall through to v2 (>= 0.21.0)."""
        assert resolve_module(self.VERSIONED_ENTRY, "1.1.0rc2") == "collector.trtllm.collect_moe_v2"

    def test_post_routes_to_current(self):
        """0.21.0.post1 >= 0.21.0, so it should match v2."""
        assert resolve_module(self.VERSIONED_ENTRY, "0.21.0.post1") == "collector.trtllm.collect_moe_v2"


# ---------------------------------------------------------------------------
# build_collections
# ---------------------------------------------------------------------------
@pytest.mark.unit
class TestBuildCollections:
    """Verify collection list building from registry."""

    SAMPLE_REGISTRY: ClassVar[list] = [
        {
            "op": "gemm",
            "module": "collector.trtllm.collect_gemm",
            "get_func": "get_gemm_test_cases",
            "run_func": "run_gemm",
        },
        {
            "op": "moe",
            "get_func": "get_moe_test_cases",
            "run_func": "run_moe_torch",
            "versions": [
                ("1.1.0", "collector.trtllm.collect_moe_v3"),
                ("0.20.0", "collector.trtllm.collect_moe_v1"),
            ],
        },
    ]

    def test_all_ops_included(self):
        colls = build_collections(self.SAMPLE_REGISTRY, "trtllm", "1.1.0")
        op_names = [c["type"] for c in colls]
        assert "gemm" in op_names
        assert "moe" in op_names

    def test_ops_filter(self):
        colls = build_collections(self.SAMPLE_REGISTRY, "trtllm", "1.1.0", ops=["gemm"])
        assert len(colls) == 1
        assert colls[0]["type"] == "gemm"

    def test_unsupported_version_skipped(self):
        colls = build_collections(self.SAMPLE_REGISTRY, "trtllm", "0.19.0")
        op_names = [c["type"] for c in colls]
        assert "gemm" in op_names
        assert "moe" not in op_names

    def test_resolved_module_in_output(self):
        colls = build_collections(self.SAMPLE_REGISTRY, "trtllm", "1.1.0", ops=["moe"])
        assert colls[0]["module"] == "collector.trtllm.collect_moe_v3"

    def test_resolved_module_old_version(self):
        colls = build_collections(self.SAMPLE_REGISTRY, "trtllm", "0.20.0", ops=["moe"])
        assert colls[0]["module"] == "collector.trtllm.collect_moe_v1"

    def test_output_dict_shape(self):
        colls = build_collections(self.SAMPLE_REGISTRY, "trtllm", "1.1.0", ops=["gemm"])
        c = colls[0]
        assert set(c.keys()) == {"name", "type", "module", "get_func", "run_func"}
        assert c["name"] == "trtllm"


# ---------------------------------------------------------------------------
# Registry integrity
# ---------------------------------------------------------------------------
@pytest.mark.unit
class TestRegistryIntegrity:
    """Validate structural invariants of all backend registries."""

    @pytest.fixture(
        params=["trtllm", "vllm", "sglang"],
    )
    def registry(self, request):
        mod = __import__(f"collector.{request.param}.registry", fromlist=["REGISTRY"])
        return mod.REGISTRY, request.param

    def test_every_entry_has_required_keys(self, registry):
        reg, backend = registry
        for entry in reg:
            assert "op" in entry, f"{backend}: entry missing 'op'"
            assert "get_func" in entry, f"{backend}/{entry.get('op')}: missing 'get_func'"
            assert "run_func" in entry, f"{backend}/{entry.get('op')}: missing 'run_func'"
            assert "module" in entry or "versions" in entry, (
                f"{backend}/{entry['op']}: must have 'module' or 'versions'"
            )

    def test_versions_descending(self, registry):
        """version tuples must be in descending min_version order."""
        reg, backend = registry
        for entry in reg:
            if "versions" not in entry:
                continue
            min_vers = [_normalize_version(v) for v, _ in entry["versions"]]
            assert min_vers == sorted(min_vers, reverse=True), (
                f"{backend}/{entry['op']}: versions not in descending order"
            )

    def test_no_duplicate_ops(self, registry):
        reg, backend = registry
        ops = [e["op"] for e in reg]
        assert len(ops) == len(set(ops)), f"{backend}: duplicate op names found"
