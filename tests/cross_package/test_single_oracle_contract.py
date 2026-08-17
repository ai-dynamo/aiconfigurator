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


# Complete def-name inventory of operations/*.py (module, class, and nested
# functions alike). ANY added or removed def requires editing this frozen
# inventory — the import-contract-style deliberate friction that catches an
# innocently NAMED Python oracle (e.g. `estimate_latency`) that the banned
# prefixes above cannot: new estimation math cannot appear without a
# reviewable one-line diff here declaring the new function.
OPERATIONS_DEF_INVENTORY = {
    "__init__.py": frozenset(),
    "afd_transfer.py": frozenset(
        {
            "__init__",
            "_afd_send_prob",
            "_engine_comm_query",
            "direction",
            "f_gpus_in_node",
            "get_weights",
            "num_f_nodes",
            "query",
        }
    ),
    "attention.py": frozenset(
        {
            "__init__",
            "_cache_key",
            "_log_attention_row_conflict",
            "clear_cache",
            "generation_attn_flops",
            "generation_attn_mode",
            "get_weights",
            "load_context_attention_data",
            "load_data",
            "load_encoder_attention_data",
            "load_generation_attention_data",
        }
    ),
    "base.py": frozenset(
        {
            "__init__",
            "_all_operation_subclasses",
            "_engine_query",
            "_engine_query_is_context",
            "_engine_query_plan",
            "_read_filtered_rows",
            "_read_perf_rows",
            "_record_load",
            "_resolve_perf_data_path",
            "_version_dir_is_partial",
            "_version_dir_is_unusable",
            "clear_all_op_caches",
            "clear_cache",
            "get_weights",
            "load_data",
            "query",
            "resolve_op_data_path",
            "supported_quant_modes",
            "warm_all_op_data",
        }
    ),
    "communication.py": frozenset(
        {
            "__init__",
            "_cache_key",
            "clear_cache",
            "get_weights",
            "load_custom_allreduce_data",
            "load_data",
            "load_nccl_data",
        }
    ),
    "dsa.py": frozenset(
        {
            "__init__",
            "_b",
            "_cache_key",
            "_dsa_kernel_source_buckets",
            "_format_dsa_unavailable_message",
            "_nest",
            "_read_dsa_row_sources",
            "clear_cache",
            "dsa_block_weights_bytes",
            "get_weights",
            "load_context_dsa_module_data",
            "load_data",
            "load_generation_dsa_module_data",
        }
    ),
    "dsv4.py": frozenset(
        {
            "__init__",
            "_cache_key",
            "_coerce",
            "_deep_merge_dsv4_dicts",
            "_dsv4_normalize_dtype",
            "_engine_query_plan",
            "_estimate_weights",
            "_is_bad_key",
            "_load",
            "_load_dsv4_split",
            "_load_sparse",
            "_make_nested",
            "_normalize_distribution",
            "_put_nested",
            "_row_phase",
            "_to_bool",
            "_validate_dsv4_local_head_semantics",
            "clear_cache",
            "get_weights",
            "load_context_dsv4_kind_module_data",
            "load_data",
            "load_dsv4_megamoe_module_data",
            "load_dsv4_sparse_kernel_data",
            "load_dsv4_sparse_op_data",
            "load_generation_dsv4_kind_module_data",
            "load_mhc_module_data",
        }
    ),
    "elementwise.py": frozenset(
        {
            "__init__",
            "get_weights",
        }
    ),
    "embedding.py": frozenset(
        {
            "__init__",
            "get_weights",
        }
    ),
    "fpm_forward.py": frozenset(
        {
            "__init__",
            "_norm_backend_request",
            "_norm_identity",
            "get_weights",
        }
    ),
    "gemm.py": frozenset(
        {
            "__init__",
            "_cache_key",
            "_engine_query_plan",
            "_load",
            "clear_cache",
            "get_weights",
            "load_compute_scale_data",
            "load_data",
            "load_gemm_data",
            "load_scale_matrix_data",
            "supported_quant_modes",
            "xprofile_util_level_known",
        }
    ),
    "mamba.py": frozenset(
        {
            "__init__",
            "_cache_key",
            "_gemm_value",
            "_mem_value",
            "clear_cache",
            "get_weights",
            "load_data",
            "load_gdn_data",
            "load_kda_data",
            "load_mamba2_data",
            "query",
        }
    ),
    "mla.py": frozenset(
        {
            "__init__",
            "_cache_key",
            "_engine_query_plan",
            "_mla_module_native_heads",
            "clear_cache",
            "get_weights",
            "load_context_mla_data",
            "load_context_mla_module_data",
            "load_data",
            "load_generation_mla_data",
            "load_generation_mla_module_data",
            "load_mla_bmm_data",
            "load_wideep_context_mla_data",
            "load_wideep_generation_mla_data",
        }
    ),
    "moe.py": frozenset(
        {
            "__init__",
            "_cache_key",
            "_engine_query_plan",
            "_normalize_quant_mode_for_table",
            "_select_alltoall_kernel",
            "_select_kernel",
            "clear_cache",
            "get_weights",
            "load_data",
            "load_moe_data",
            "load_trtllm_alltoall_data",
            "load_wideep_context_moe_data",
            "load_wideep_deepep_ll_data",
            "load_wideep_deepep_normal_data",
            "load_wideep_generation_moe_data",
            "load_wideep_moe_compute_data",
            "xprofile_util_level_known",
        }
    ),
    "moe_comm.py": frozenset(
        {
            "__init__",
            "_adapt_legacy_deepep",
            "_adapt_legacy_deepep_ll",
            "_adapt_legacy_deepep_normal",
            "_adapt_legacy_sglang_context_moe",
            "_adapt_legacy_sglang_generation_moe",
            "_adapt_legacy_sglang_wideep_moe",
            "_adapt_legacy_trtllm_alltoall",
            "_adapt_legacy_trtllm_wideep_moe",
            "_cache_key",
            "_engine_query_plan",
            "_load_legacy_a2a",
            "_load_legacy_ep",
            "_moe_a2a_store",
            "_moe_ep_store",
            "_normalize_sms",
            "_require_latency",
            "_resolve_kernel_source",
            "_row_power",
            "_store_a2a_leaf",
            "_store_ep_leaf",
            "_validate_a2a_request",
            "_validate_ep_phase",
            "clear_cache",
            "feasible",
            "get_weights",
            "load_data",
            "load_moe_a2a_data",
            "load_moe_expert_compute_data",
            "nodes_for",
        }
    ),
    "msa.py": frozenset(
        {
            "__init__",
            "get_weights",
            "load_data",
        }
    ),
    "overlap.py": frozenset(
        {
            "__init__",
            "_engine_query_is_context",
            "_engine_query_plan",
            "_infer_phase",
            "get_weights",
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


def _file_def_names(source_text: str) -> frozenset[str]:
    return frozenset(
        node.name
        for node in ast.walk(ast.parse(source_text))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    )


def test_operations_def_inventory_is_frozen():
    """Every function definition in operations/ is enumerated above. Adding a
    def (whatever its name) fails here until the inventory is deliberately
    edited — the reviewable declaration point for anything that could be
    estimation math under an innocent name."""
    assert OPERATIONS_DIR.is_dir(), f"source layout expected at {OPERATIONS_DIR} (scan must not pass vacuously)"
    live = {path.name: _file_def_names(path.read_text(encoding="utf-8")) for path in OPERATIONS_DIR.glob("*.py")}
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
    new_names = _file_def_names(fixture)
    assert _offending_defs(fixture) == []  # the prefix guard alone is blind here...
    for fname, frozen in OPERATIONS_DEF_INVENTORY.items():
        assert not (new_names & frozen), f"fixture names collide with {fname}"
    # ...but none of these names exists in any frozen per-file inventory, so
    # introducing them into ANY operations module trips
    # test_operations_def_inventory_is_frozen.
