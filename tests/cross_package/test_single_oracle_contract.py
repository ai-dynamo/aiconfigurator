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
loading, not per-op latency values. The policy, including where the correct
home is for what you were trying to add, is ``.claude/rules/rust-core/parity.md``
Rule 2 (the single-oracle invariant).
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
QUERY_OVERRIDE_WHITELIST = {
    "AFDTransfer",
    "AFDFAllGather",
    "AFDFReduceScatter",
    "AFDCombine",
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


# Complete QUALIFIED def inventory of operations/*.py: every function at its
# lexical path (module function, `Class.method`, nested closures as
# `outer.inner`). Qualification defeats the shadow-class bypass a plain
# name set allows (a new class defining only pre-existing NAMES like
# `__init__`/`get_weights` still adds new qualified paths). ANY added or
# removed def requires editing this frozen inventory — the reviewable
# declaration point for anything that could be estimation math under an
# innocent name.
OPERATIONS_DEF_INVENTORY = {
    "__init__.py": frozenset(),
    "afd_transfer.py": frozenset(
        {
            "AFDCombine.__init__",
            "AFDCombine.get_weights",
            "AFDCombine.query",
            "AFDFAllGather.__init__",
            "AFDFAllGather.f_gpus_in_node",
            "AFDFAllGather.get_weights",
            "AFDFAllGather.num_f_nodes",
            "AFDFAllGather.query",
            "AFDFReduceScatter.__init__",
            "AFDFReduceScatter.f_gpus_in_node",
            "AFDFReduceScatter.get_weights",
            "AFDFReduceScatter.num_f_nodes",
            "AFDFReduceScatter.query",
            "AFDTransfer.__init__",
            "AFDTransfer.direction",
            "AFDTransfer.get_weights",
            "AFDTransfer.num_f_nodes",
            "AFDTransfer.query",
            "_afd_send_prob",
            "_engine_comm_query",
        }
    ),
    "attention.py": frozenset(
        {
            "ContextAttention._cache_key",
            "ContextAttention.clear_cache",
            "ContextAttention.load_data",
            "EncoderAttention._cache_key",
            "EncoderAttention.clear_cache",
            "EncoderAttention.load_data",
            "GenerationAttention._cache_key",
            "GenerationAttention.clear_cache",
            "GenerationAttention.load_data",
            "_cache_key",
            # AIC-1715/1716: attention kernel-lane PRECEDENCE resolution — which
            # kernel_source lane a query should prefer, not a per-op VALUE. Reads
            # a YAML map (attention_lanes.resolve_attention_lane_order) and ranks
            # a loaded table's own lanes by measured coverage; never computes a
            # latency/energy/SOL number itself, so it stays outside the single
            # per-op-value-in-Rust-only rule.
            "_lane_order_cached",
            "_source_tiered_lane_walk_order",
            "resolve_lane_order",
            "lane_walk_order",
            "lane_walk_order._rank",
            # Same rationale: called from engine.py::_resolve_attention_lane_orders
            # once a database is bound to a built model's op lists (models are
            # constructed WITHOUT a database — pure shape graphs — since the
            # pyo3 op unification); composes the two entries above, never
            # computes a latency/energy/SOL number itself.
            "resolved_lane_order_for_op",
        }
    ),
    "base.py": frozenset(
        {
            "OpShellKit._engine_query",
            "OpShellKit._engine_query_is_context",
            "OpShellKit._engine_query_plan",
            "OpShellKit._record_load",
            "OpShellKit.clear_cache",
            "OpShellKit.load_data",
            "OpShellKit.supported_quant_modes",
            "PythonOperation.__init__",
            "PythonOperation.get_weights",
            "_all_operation_subclasses",
            "_resolve_perf_data_path",
            "_version_dir_is_partial",
            "_version_dir_is_unusable",
            "clear_all_op_caches",
            "resolve_op_data_path",
            "warm_all_op_data",
        }
    ),
    "communication.py": frozenset(
        {
            "CustomAllReduce._cache_key",
            "CustomAllReduce.clear_cache",
            "CustomAllReduce.load_data",
            "NCCL._cache_key",
            "NCCL.clear_cache",
            "NCCL.load_data",
            "_cache_key",
        }
    ),
    "dsa.py": frozenset(
        {
            "ContextDSAModule._cache_key",
            "ContextDSAModule.clear_cache",
            "ContextDSAModule.load_data",
            "GenerationDSAModule._cache_key",
            "GenerationDSAModule.clear_cache",
            "GenerationDSAModule.load_data",
            "_cache_key",
            "_normalize_projection_quant_modes",
        }
    ),
    "dsv4.py": frozenset(
        {
            "ContextDeepSeekV4AttentionModule._cache_key",
            "ContextDeepSeekV4AttentionModule.clear_cache",
            "ContextDeepSeekV4AttentionModule.load_data",
            "ContextDeepSeekV4AttentionModule.load_data._load_sparse",
            "ContextDeepSeekV4AttentionModule.load_data._primary",
            "DeepSeekV4MHCModule._cache_key",
            "DeepSeekV4MHCModule.clear_cache",
            "DeepSeekV4MHCModule.load_data",
            "DeepSeekV4MegaMoEModule._cache_key",
            "DeepSeekV4MegaMoEModule.clear_cache",
            "DeepSeekV4MegaMoEModule.load_data",
            "GenerationDeepSeekV4AttentionModule._cache_key",
            "GenerationDeepSeekV4AttentionModule.clear_cache",
            "GenerationDeepSeekV4AttentionModule.load_data",
            "_cache_key",
        }
    ),
    "elementwise.py": frozenset(),
    "embedding.py": frozenset(),
    "fpm_forward.py": frozenset(
        {
            "FPMForwardOp.__init__",
            "FPMForwardOp.get_weights",
            "_norm_backend_request",
            "_norm_identity",
        }
    ),
    "gemm.py": frozenset(
        {
            "GEMM._cache_key",
            "GEMM.clear_cache",
            "GEMM.load_data",
            "GEMM.supported_quant_modes",
            "xprofile_util_level_known",
        }
    ),
    "mamba.py": frozenset(
        {
            "GDNKernel._cache_key",
            "GDNKernel.clear_cache",
            "GDNKernel.load_data",
            "KDAKernel._cache_key",
            "KDAKernel.load_data",
            "Mamba2Kernel._cache_key",
            "Mamba2Kernel.clear_cache",
            "Mamba2Kernel.load_data",
            "_cache_key",
        }
    ),
    "mla.py": frozenset(
        {
            "ContextMLA._cache_key",
            "ContextMLA.clear_cache",
            "ContextMLA.load_data",
            "GenerationMLA._cache_key",
            "GenerationMLA.clear_cache",
            "GenerationMLA.load_data",
            "MLABmm._cache_key",
            "MLABmm._engine_query_plan",
            "MLABmm.clear_cache",
            "MLABmm.load_data",
            "MLAModule._cache_key",
            "MLAModule.clear_cache",
            "MLAModule.load_data",
            "WideEPContextMLA._cache_key",
            "WideEPContextMLA.clear_cache",
            "WideEPContextMLA.load_data",
            "WideEPGenerationMLA._cache_key",
            "WideEPGenerationMLA.clear_cache",
            "WideEPGenerationMLA.load_data",
            "_cache_key",
        }
    ),
    "moe.py": frozenset(
        {
            "MoE._cache_key",
            "MoE._seq_split",
            "MoE._seq_split.setter",
            "MoE.clear_cache",
            "MoE.load_data",
            "MoEDispatch.__init__",
            "MoEDispatch._cache_key",
            "MoEDispatch._quant_mode",
            "MoEDispatch.clear_cache",
            "MoEDispatch.load_data",
            "_cache_key",
            "xprofile_util_level_known",
        }
    ),
    "moe_comm.py": frozenset(
        {
            "MoEAllToAll.__init__",
            "MoEAllToAll._cache_key",
            "MoEAllToAll.clear_cache",
            "MoEAllToAll.load_data",
            "MoECommBackendSpec.feasible",
            "MoEExpertCompute.__init__",
            "MoEExpertCompute._cache_key",
            "MoEExpertCompute.clear_cache",
            "MoEExpertCompute.load_data",
            "_cache_key",
            "communication_dtype_for",
            "_validate_a2a_request",
            "_validate_ep_phase",
            "nodes_for",
        }
    ),
    "msa.py": frozenset(),
    "overlap.py": frozenset(
        {
            "FallbackOp._engine_query_is_context",
            "FallbackOp._engine_query_plan",
            "FallbackOp._seq_split",
            "FallbackOp._seq_split.setter",
            "OverlapOp._engine_query_is_context",
            "OverlapOp._engine_query_plan",
            "OverlapOp._seq_split",
            "OverlapOp._seq_split.setter",
            "_has_leaves",
            "_infer_phase",
        }
    ),
    "util_empirical.py": frozenset(
        {
            "capture_provenance",
            "clear_grid_cache",
            "note_provenance",
            "quant_profile",
            "worst_provenance",
        }
    ),
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


def test_perf_database_has_no_per_call_query_surface():
    """The deprecated ``query_*`` shim window closed with the
    deprecation-cleanup PR: PerfDatabase exposes NO per-call query surface.
    New per-op access goes through EngineHandle.evaluate_ops_json /
    evaluate_ops_sol_json, the per-phase surface, or whole runs."""
    from aiconfigurator_core.sdk.perf_database import PerfDatabase

    live = {name for name in dir(PerfDatabase) if name.startswith("query_")}
    assert not live, (
        f"PerfDatabase grew query_* methods: {sorted(live)}. The per-call surface was removed "
        "after its deprecation window; new per-op access goes through EngineHandle.evaluate_ops_json."
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


def _file_def_names(source_text: str) -> list[str]:
    """Every def at its qualified lexical path, in occurrence order (a list,
    not a set, so duplicate qualified redefinitions surface as duplicates).
    Property setter/deleter re-defs share the property's name by construction;
    they are qualified with the decorator role so a getter/setter pair is not
    flagged as an ambiguous redefinition."""
    out: list[str] = []

    def role_suffix(node) -> str:
        for dec in getattr(node, "decorator_list", []):
            if isinstance(dec, ast.Attribute) and dec.attr in ("setter", "deleter"):
                return f".{dec.attr}"
        return ""

    def walk(node, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qualified = f"{prefix}{child.name}{role_suffix(child)}"
                out.append(qualified)
                walk(child, qualified + ".")
            elif isinstance(child, ast.ClassDef):
                walk(child, f"{prefix}{child.name}.")
            else:
                walk(child, prefix)

    walk(ast.parse(source_text), "")
    return out


def test_operations_def_inventory_is_frozen():
    """Every function definition in operations/ is enumerated above. Adding a
    def (whatever its name) fails here until the inventory is deliberately
    edited — the reviewable declaration point for anything that could be
    estimation math under an innocent name."""
    assert OPERATIONS_DIR.is_dir(), f"source layout expected at {OPERATIONS_DIR} (scan must not pass vacuously)"
    live_lists = {path.name: _file_def_names(path.read_text(encoding="utf-8")) for path in OPERATIONS_DIR.glob("*.py")}
    for fname, qualified in sorted(live_lists.items()):
        duplicated = sorted({q for q in qualified if qualified.count(q) > 1})
        assert not duplicated, f"{fname}: duplicate qualified defs (ambiguous redefinition): {duplicated}"
    live = {fname: frozenset(qualified) for fname, qualified in live_lists.items()}
    assert set(live) == set(OPERATIONS_DEF_INVENTORY), (
        f"operations module set drifted: files added {sorted(set(live) - set(OPERATIONS_DEF_INVENTORY))}, "
        f"removed {sorted(set(OPERATIONS_DEF_INVENTORY) - set(live))} — update the inventory AND "
        "test_import_contract.py deliberately."
    )
    problems = []
    for fname in sorted(live):
        added = live[fname] - OPERATIONS_DEF_INVENTORY[fname]
        removed = OPERATIONS_DEF_INVENTORY[fname] - live[fname]
        if added:
            problems.append(f"{fname}: added defs {sorted(added)}")
        if removed:
            problems.append(f"{fname}: removed defs {sorted(removed)}")
    assert not problems, (
        "operations/ def inventory drifted — declare the change deliberately in "
        "OPERATIONS_DEF_INVENTORY (and justify any new function that computes performance values): "
        + "; ".join(problems)
    )


def test_def_inventory_catches_innocently_named_oracle():
    """Negative fixture for the rename gap the banned prefixes cannot cover:
    an estimator named `estimate_latency` / `table_lookup` / `_interpolate_2d`
    matches no banned prefix, but it is a NEW def, so the frozen inventory
    flags it."""
    fixture = (
        "def estimate_latency(shape, table):\n"
        "    return table[shape] * 1.05\n"
        "def table_lookup(table, key):\n"
        "    return table[key]\n"
        "def _interpolate_2d(grid, x, y):\n"
        "    return grid[x][y]\n"
    )
    new_names = frozenset(_file_def_names(fixture))
    assert _offending_defs(fixture) == []  # the prefix guard alone is blind here...
    for fname, frozen in OPERATIONS_DEF_INVENTORY.items():
        assert not (new_names & frozen), f"fixture names collide with {fname}"
    # ...but none of these names exists in any frozen per-file inventory, so
    # introducing them into ANY operations module trips
    # test_operations_def_inventory_is_frozen.


def test_def_inventory_catches_shadow_class_with_existing_names():
    """Negative fixture for the shadow-class bypass: a NEW class that defines
    only names already present in a module (``__init__``/``get_weights``/
    ``load_data``) adds no new PLAIN names — but its qualified paths are new,
    so the frozen inventory still flags it."""
    fixture = (
        "class ShadowOp:\n"
        "    def __init__(self):\n"
        "        pass\n"
        "    def get_weights(self):\n"
        "        return 0\n"
        "    def load_data(self):\n"
        "        pass\n"
    )
    qualified = frozenset(_file_def_names(fixture))
    assert qualified == {"ShadowOp.__init__", "ShadowOp.get_weights", "ShadowOp.load_data"}
    plain = {q.rpartition(".")[2] for q in qualified}
    for fname, frozen in OPERATIONS_DEF_INVENTORY.items():
        frozen_plain = {q.rpartition(".")[2] for q in frozen}
        if plain <= frozen_plain:
            # the plain names all pre-exist in this file (the bypass a flat
            # name set would allow)...
            assert not (qualified & frozen), f"fixture qualified names collide with {fname}"
            break
    else:  # pragma: no cover — operations/ always has such a file
        raise AssertionError("no inventory file contains all fixture plain names")
    # ...but none of the QUALIFIED paths exists anywhere, so introducing the
    # shadow class trips test_operations_def_inventory_is_frozen.
