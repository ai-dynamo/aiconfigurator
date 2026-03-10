# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for collector version resolver and registry infrastructure."""

import ast
import re
from pathlib import Path
from typing import ClassVar

import pytest
from packaging.version import Version

from collector.registry_types import OpEntry, VersionRoute
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
    """Verify PEP 440-aware version normalization."""

    @pytest.mark.parametrize(
        "version_str,expected",
        [
            ("1.2.0", Version("1.2.0")),
            ("0.20.0", Version("0.20.0")),
            ("0.5.5", Version("0.5.5")),
        ],
    )
    def test_final_release(self, version_str, expected):
        assert _normalize_version(version_str) == expected

    @pytest.mark.parametrize(
        "version_str,expected",
        [
            ("1.2.0dev1", Version("1.2.0dev1")),
            ("1.2.0a2", Version("1.2.0a2")),
            ("1.2.0b1", Version("1.2.0b1")),
            ("1.2.0rc2", Version("1.2.0rc2")),
        ],
    )
    def test_pre_release(self, version_str, expected):
        assert _normalize_version(version_str) == expected

    @pytest.mark.parametrize(
        "version_str,expected",
        [
            ("0.5.5.post2", Version("0.5.5.post2")),
            ("1.1.0.post1", Version("1.1.0.post1")),
        ],
    )
    def test_post_release(self, version_str, expected):
        assert _normalize_version(version_str) == expected

    def test_local_metadata_ignored(self):
        assert _normalize_version("0.20.0+cu124") == _normalize_version("0.20.0")

    def test_ordering_dev_to_post(self):
        ordered = ["1.2.0dev1", "1.2.0a2", "1.2.0b1", "1.2.0rc2", "1.2.0", "1.2.0.post2"]
        versions = [_normalize_version(v) for v in ordered]
        assert versions == sorted(versions)

    def test_rc_less_than_release(self):
        assert _normalize_version("1.1.0rc5") < _normalize_version("1.1.0")

    def test_post_greater_than_release(self):
        assert _normalize_version("0.5.5.post2") > _normalize_version("0.5.5")

    def test_short_release_equals_patch_zero(self):
        assert _normalize_version("1.1") == _normalize_version("1.1.0")


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

    @pytest.mark.parametrize(
        "compat",
        [
            "trtllm",
            "trtllm>=",
            "trtllm>=1.1.0,<<2.0.0",
        ],
    )
    def test_invalid_compat_raises(self, compat):
        with pytest.raises(ValueError, match="Invalid __compat__"):
            _check_compat(compat, "1.1.0")


# ---------------------------------------------------------------------------
# resolve_module
# ---------------------------------------------------------------------------
@pytest.mark.unit
class TestResolveModule:
    """Verify version-based module resolution from registry entries."""

    VERSIONED_ENTRY: ClassVar[OpEntry] = OpEntry(
        op="moe",
        get_func="get_moe_test_cases",
        run_func="run_moe_torch",
        versions=(
            VersionRoute("1.1.0", "collector.trtllm.collect_moe_v3"),
            VersionRoute("0.21.0", "collector.trtllm.collect_moe_v2"),
            VersionRoute("0.20.0", "collector.trtllm.collect_moe_v1"),
        ),
    )

    UNVERSIONED_ENTRY: ClassVar[OpEntry] = OpEntry(
        op="gemm",
        module="collector.trtllm.collect_gemm",
        get_func="get_gemm_test_cases",
        run_func="run_gemm",
    )

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

    def test_short_version_routes_to_current(self):
        """1.1 should be treated as 1.1.0."""
        assert resolve_module(self.VERSIONED_ENTRY, "1.1") == "collector.trtllm.collect_moe_v3"


# ---------------------------------------------------------------------------
# build_collections
# ---------------------------------------------------------------------------
@pytest.mark.unit
class TestBuildCollections:
    """Verify collection list building from registry."""

    SAMPLE_REGISTRY: ClassVar[list[OpEntry]] = [
        OpEntry(
            op="gemm",
            module="collector.trtllm.collect_gemm",
            get_func="get_gemm_test_cases",
            run_func="run_gemm",
        ),
        OpEntry(
            op="moe",
            get_func="get_moe_test_cases",
            run_func="run_moe_torch",
            versions=(
                VersionRoute("1.1.0", "collector.trtllm.collect_moe_v3"),
                VersionRoute("0.20.0", "collector.trtllm.collect_moe_v1"),
            ),
        ),
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
# OpEntry validation
# ---------------------------------------------------------------------------
@pytest.mark.unit
class TestOpEntryValidation:
    """Verify OpEntry construction-time invariants."""

    def test_rejects_missing_module_and_versions(self):
        with pytest.raises(ValueError, match="must specify 'module' or 'versions'"):
            OpEntry(op="bad", get_func="f", run_func="r")

    def test_rejects_both_module_and_versions(self):
        with pytest.raises(ValueError, match="cannot specify both"):
            OpEntry(
                op="bad",
                get_func="f",
                run_func="r",
                module="some.module",
                versions=(VersionRoute("1.0.0", "other.module"),),
            )

    def test_accepts_module_only(self):
        entry = OpEntry(op="ok", get_func="f", run_func="r", module="some.module")
        assert entry.module == "some.module"
        assert entry.versions == ()

    def test_accepts_versions_only(self):
        routes = (VersionRoute("1.0.0", "some.module"),)
        entry = OpEntry(op="ok", get_func="f", run_func="r", versions=routes)
        assert entry.module is None
        assert entry.versions == routes

    def test_frozen(self):
        entry = OpEntry(op="ok", get_func="f", run_func="r", module="m")
        with pytest.raises(AttributeError):
            entry.op = "changed"


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

    def test_every_entry_is_opentry(self, registry):
        reg, backend = registry
        for entry in reg:
            assert isinstance(entry, OpEntry), f"{backend}: entry {entry!r} is not an OpEntry"

    def test_versions_descending(self, registry):
        """version tuples must be in descending min_version order."""
        reg, backend = registry
        for entry in reg:
            if not entry.versions:
                continue
            min_vers = [_normalize_version(r.min_version) for r in entry.versions]
            assert min_vers == sorted(min_vers, reverse=True), f"{backend}/{entry.op}: versions not in descending order"

    def test_no_duplicate_ops(self, registry):
        reg, backend = registry
        ops = [e.op for e in reg]
        assert len(ops) == len(set(ops)), f"{backend}: duplicate op names found"

    # -----------------------------------------------------------------------
    # Module / function existence
    # -----------------------------------------------------------------------

    @staticmethod
    def _module_top_level_names(module_dotpath: str) -> set[str]:
        """Parse a collector module source via AST and return top-level def/class names.

        Uses AST so that heavy runtime dependencies (torch, tensorrt_llm, …)
        do NOT need to be installed.
        """
        parts = module_dotpath.split(".")
        source_file = Path(*parts).with_suffix(".py")
        if not source_file.exists():
            return set()
        tree = ast.parse(source_file.read_text(), filename=str(source_file))
        return {
            node.name
            for node in ast.iter_child_nodes(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        }

    def test_module_exports_declared_functions(self, registry):
        """Every module referenced in the registry must define the declared
        get_func and run_func at the module's top level.

        Uses AST parsing so the test works without GPU/framework dependencies.
        """
        reg, backend = registry
        # Deduplicate (module, get_func, run_func) to avoid redundant checks
        seen: set[tuple[str, str, str]] = set()
        for entry in reg:
            if entry.versions:
                modules = [r.module for r in entry.versions]
            else:
                modules = [entry.module]
            for mod_path in modules:
                key = (mod_path, entry.get_func, entry.run_func)
                if key in seen:
                    continue
                seen.add(key)

                names = self._module_top_level_names(mod_path)
                assert names, f"{backend}/{entry.op}: source file for module {mod_path!r} not found or empty"
                for attr_label, func_name in [("get_func", entry.get_func), ("run_func", entry.run_func)]:
                    assert func_name in names, (
                        f"{backend}/{entry.op}: module {mod_path!r} "
                        f"does not define {func_name!r} (declared as {attr_label})"
                    )

    # -----------------------------------------------------------------------
    # File naming convention enforcement
    # -----------------------------------------------------------------------
    _VN_SUFFIX_RE = re.compile(r"_v(\d+)$")

    def test_versioned_modules_use_vn_suffix(self, registry):
        """All module paths inside a 'versions' tuple must end with _vN."""
        reg, backend = registry
        for entry in reg:
            if not entry.versions:
                continue
            for route in entry.versions:
                assert self._VN_SUFFIX_RE.search(route.module), (
                    f"{backend}/{entry.op}: versioned module {route.module!r} "
                    f"(min_version={route.min_version}) is missing a _vN suffix"
                )

    def test_unversioned_modules_no_vn_suffix(self, registry):
        """Unversioned entries (single 'module') must NOT have a _vN suffix."""
        reg, backend = registry
        for entry in reg:
            if entry.versions:
                continue
            assert not self._VN_SUFFIX_RE.search(entry.module), (
                f"{backend}/{entry.op}: unversioned module {entry.module!r} should not have a _vN suffix"
            )

    def test_version_numbers_sequential_from_one(self, registry):
        """For each module base name, _vN suffixes must be v1, v2, ..., vN with no gaps.

        Checked across the entire registry because different ops may share
        the same versioned module (e.g. moe and moe_eplb both use collect_moe_v3).
        """
        reg, backend = registry
        # Collect all version numbers grouped by module base name
        base_versions: dict[str, set[int]] = {}
        for entry in reg:
            if not entry.versions:
                continue
            for route in entry.versions:
                m = self._VN_SUFFIX_RE.search(route.module)
                if not m:
                    continue
                base = self._VN_SUFFIX_RE.sub("", route.module)
                base_versions.setdefault(base, set()).add(int(m.group(1)))

        for base, nums in base_versions.items():
            sorted_nums = sorted(nums)
            expected = list(range(1, max(sorted_nums) + 1))
            assert sorted_nums == expected, (
                f"{backend}: module base {base!r} has version suffixes {sorted_nums}, expected sequential {expected}"
            )
