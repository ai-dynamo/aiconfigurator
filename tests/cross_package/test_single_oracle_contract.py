# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single-oracle contract: per-op performance values come ONLY from the
compiled Rust engine.

PR-5 of #1357 deleted the Python per-call query stack (the per-family
``_query_*_table`` math, ``perf_interp``, the empirical-utilization math in
``util_empirical``) and left the public surface as engine-routed deprecation
shims. This test freezes that end state the same way
``test_import_contract.py`` freezes the module map: re-growing a Python-side
performance-math path — a new ``query_*`` method, an op-level ``query``
override, an interpolation helper — REQUIRES editing the whitelists below,
which makes the regression deliberate and visible in review instead of
accidental.

If you are here because this test failed: per-op performance math belongs in
``aic-core/rust/aiconfigurator-core`` (one oracle, cross-checked by the
frozen parity goldens). Python owns model/topology composition and data
loading, not per-op latency values. See
``aic-core/rust/aiconfigurator-core/docs/python-dedup-plan.md``
(post-PR-5 invariant section).
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import pkgutil
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

OPERATIONS_DIR = Path(__file__).resolve().parents[2] / "aic-core" / "src" / "aiconfigurator_core" / "sdk" / "operations"
PERF_DATABASE_PATH = OPERATIONS_DIR.parent / "perf_database.py"

# Operation subclasses allowed to override ``query`` — ORCHESTRATION bodies
# whose per-message values still come from the engine (they compose standard
# comm/gemm twins via the single-op evaluation plumbing):
#   - the AFD comm ops: A/F topology math (send probability, link volumes)
#   - Mamba2: deprecated composite kept for the public-SDK window (the
#     deprecation-cleanup PR removes it); its five sub-ops are engine-evaluated twins
QUERY_OVERRIDE_WHITELIST = {
    "AFDTransfer",
    "AFDFAllGather",
    "AFDFReduceScatter",
    "AFDCombine",
    "Mamba2",
}

# The frozen public per-call surface on PerfDatabase: every entry is a
# deprecated engine-routed shim (or an explicit tombstone that raises), all
# removed together in the deprecation-cleanup PR. Adding a NEW query_* method to PerfDatabase is a
# single-oracle violation — route callers through the op-list FFI instead.
PERF_DATABASE_QUERY_SHIMS = {
    "query_gemm",
    "query_compute_scale",
    "query_scale_matrix",
    "query_context_attention",
    "query_encoder_attention",
    "query_generation_attention",
    "query_context_mla",
    "query_generation_mla",
    "query_context_mla_module",
    "query_generation_mla_module",
    "query_wideep_generation_mla",
    "query_wideep_context_mla",
    "query_custom_allreduce",
    "query_nccl",
    "query_moe",
    "query_mla_bmm",
    "query_mem_op",
    "query_mamba2",
    "query_gdn",
    "query_p2p",
    "query_wideep_deepep_ll",
    "query_wideep_deepep_normal",
    "query_wideep_moe_compute",
    "query_trtllm_alltoall",
    "query_moe_a2a",
    "query_moe_expert_compute",
    "query_context_dsa_module",
    "query_generation_dsa_module",
    "query_mhc_module",
    "query_context_deepseek_v4_attention_module",
    "query_generation_deepseek_v4_attention_module",
    "query_dsv4_megamoe_module",
}

# util_empirical's surviving public surface: the provenance pipeline (the
# compiled engine reports its empirical tier back through it). The
# grid/estimate/transfer MATH is gone — its oracle is
# aic-core/rust/aiconfigurator-core/src/operators/util_empirical.rs.
UTIL_EMPIRICAL_PUBLIC_SURFACE = {
    "PROVENANCE_ORDER",
    "note_provenance",
    "capture_provenance",
    "worst_provenance",
    "clear_grid_cache",
    # (memory, compute) profile classification — the admission-table key the
    # task_v2 validate gate consults; metadata, not estimation math.
    "quant_profile",
}


def test_perf_interp_is_gone():
    assert importlib.util.find_spec("aiconfigurator_core.sdk.perf_interp") is None, (
        "sdk.perf_interp was retired in PR-5 of #1357: per-op interpolation lives in the "
        "compiled engine (aiconfigurator-core/src/perf_database + operators). Do not reintroduce "
        "a Python interpolation layer."
    )


def test_util_empirical_is_provenance_only():
    module = importlib.import_module("aiconfigurator_core.sdk.operations.util_empirical")
    public = {
        name
        for name in vars(module)
        if not name.startswith("_") and name != "annotations" and not _is_import(module, name)
    }
    unexpected = public - UTIL_EMPIRICAL_PUBLIC_SURFACE
    assert not unexpected, (
        f"util_empirical grew beyond the provenance pipeline: {sorted(unexpected)}. Empirical "
        "utilization math belongs in the Rust engine (operators/util_empirical.rs)."
    )


def _is_import(module, name):
    import types

    return isinstance(getattr(module, name), types.ModuleType)


def _operation_defs(tree):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield node


# Banned def-name shapes for Python-side per-op estimation math. Name-based
# guards cannot catch a determined rename (the behavioral guard is the
# CodeRabbit path instruction + human review); they DO catch the shapes this
# codebase has actually grown: `_query_*` lookup/dispatch bodies (including
# non-`_table` variants like the retired `_query_cp`), `_lookup_*`
# interpolators, and `get_sol`/`get_empirical` closures.
_BANNED_DEF_EXACT = frozenset({"get_sol", "get_empirical"})
_BANNED_DEF_PREFIXES = ("_query_", "_lookup_")


def _offending_defs(source_text: str, filename: str = "<memory>") -> list[str]:
    offenders = []
    for node in _operation_defs(ast.parse(source_text)):
        name = node.name
        if name in _BANNED_DEF_EXACT or name.startswith(_BANNED_DEF_PREFIXES):
            offenders.append(f"{filename}:{node.lineno} def {name}")
    return offenders


def test_no_query_table_math_in_operations():
    assert OPERATIONS_DIR.is_dir(), f"source layout expected at {OPERATIONS_DIR} (scan must not pass vacuously)"
    offenders = []
    for path in sorted(OPERATIONS_DIR.glob("*.py")):
        offenders.extend(_offending_defs(path.read_text(encoding="utf-8"), path.name))
    assert not offenders, (
        "Python-side per-op query/roofline math reappeared (single-oracle violation, #1357 PR-5): "
        + "; ".join(offenders)
    )


def test_math_def_scanner_catches_offenders():
    """Negative fixture: the scanner itself must flag every banned shape —
    including the non-`_table` `_query_*` variant that hid the retired
    `_query_cp` cluster from the first version of this guard."""
    fixture = (
        "class Op:\n"
        "    def _query_cp(self):\n"
        "        pass\n"
        "    def _query_gemm_table(self):\n"
        "        pass\n"
        "    @staticmethod\n"
        "    def _lookup_2d(table):\n"
        "        pass\n"
        "def outer():\n"
        "    def get_sol():\n"
        "        pass\n"
        "    def get_empirical():\n"
        "        pass\n"
        "def _engine_query_plan(self):\n"
        "    pass\n"
    )
    flagged = {entry.split(" def ")[1] for entry in _offending_defs(fixture)}
    assert flagged == {"_query_cp", "_query_gemm_table", "_lookup_2d", "get_sol", "get_empirical"}


def test_operation_query_overrides_are_whitelisted():
    operations = importlib.import_module("aiconfigurator_core.sdk.operations")
    for info in pkgutil.iter_modules(operations.__path__):
        importlib.import_module(f"aiconfigurator_core.sdk.operations.{info.name}")
    from aiconfigurator_core.sdk.operations.base import Operation, _all_operation_subclasses

    offenders = {
        cls.__name__
        for cls in _all_operation_subclasses(Operation)
        # Only classes DEFINED in the operations package are the contract
        # surface — test suites legitimately define local Operation stubs.
        if cls.__module__.startswith("aiconfigurator_core.sdk.operations")
        and "query" in cls.__dict__
        and cls.__name__ not in QUERY_OVERRIDE_WHITELIST
    }
    assert not offenders, (
        f"Operation subclasses override query() outside the orchestration whitelist: {sorted(offenders)}. "
        "Per-op values come from the engine — declare _ENGINE_QUERY_SHAPE (base shim) or use the op-list FFI."
    )


def test_perf_database_query_surface_is_frozen():
    from aiconfigurator_core.sdk.perf_database import PerfDatabase

    live = {name for name in dir(PerfDatabase) if name.startswith("query_")}
    added = live - PERF_DATABASE_QUERY_SHIMS
    removed = PERF_DATABASE_QUERY_SHIMS - live
    assert not added, (
        f"PerfDatabase grew new query_* methods: {sorted(added)}. The per-call surface is a frozen "
        "set of deprecated shims (removed in the deprecation-cleanup PR); new per-op access goes through "
        "EngineHandle.evaluate_ops_json."
    )
    assert not removed, (
        f"query_* shims disappeared before their deprecation window closed: {sorted(removed)} "
        "(update this contract deliberately if the deprecation-cleanup PR is executing the removal)."
    )


def test_no_perf_interp_references_in_operations():
    assert OPERATIONS_DIR.is_dir() and PERF_DATABASE_PATH.is_file(), (
        f"source layout expected at {OPERATIONS_DIR} (scan must not pass vacuously)"
    )
    offenders = []
    for path in sorted(OPERATIONS_DIR.glob("*.py")) + [PERF_DATABASE_PATH]:
        text = path.read_text(encoding="utf-8")
        if "perf_interp" in text:
            offenders.append(path.name)
    assert not offenders, f"perf_interp references reappeared in: {offenders}"
