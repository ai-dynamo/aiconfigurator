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
            "ContextAttention.__init__",
            "ContextAttention._cache_key",
            "ContextAttention.clear_cache",
            "ContextAttention.load_data",
            "EncoderAttention.__init__",
            "EncoderAttention._cache_key",
            "EncoderAttention.clear_cache",
            "EncoderAttention.load_data",
            "GenerationAttention.__init__",
            "GenerationAttention._cache_key",
            "GenerationAttention.clear_cache",
            "GenerationAttention.load_data",
            "_cache_key",
        }
    ),
    "base.py": frozenset(
        {
            "Operation.__init__",
            "Operation._engine_query",
            "Operation._engine_query_is_context",
            "Operation._engine_query_plan",
            "Operation._record_load",
            "Operation.clear_cache",
            "Operation.get_weights",
            "Operation.load_data",
            "Operation.query",
            "Operation.supported_quant_modes",
            "_all_operation_subclasses",
            "_read_filtered_rows",
            "_read_perf_rows",
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
            "CustomAllReduce.__init__",
            "CustomAllReduce._cache_key",
            "CustomAllReduce.clear_cache",
            "CustomAllReduce.load_data",
            "NCCL.__init__",
            "NCCL._cache_key",
            "NCCL.clear_cache",
            "NCCL.load_data",
            "P2P.__init__",
            "_cache_key",
        }
    ),
    "dsa.py": frozenset(
        {
            "ContextDSAModule.__init__",
            "ContextDSAModule._cache_key",
            "ContextDSAModule.clear_cache",
            "ContextDSAModule.load_data",
            "GenerationDSAModule.__init__",
            "GenerationDSAModule._cache_key",
            "GenerationDSAModule.clear_cache",
            "GenerationDSAModule.load_data",
            "_cache_key",
            "_dsa_kernel_source_buckets",
            "_format_dsa_unavailable_message",
            "_read_dsa_row_sources",
            "load_context_dsa_module_data",
            "load_context_dsa_module_data._nest",
            "load_generation_dsa_module_data",
            "load_generation_dsa_module_data._nest",
        }
    ),
    "dsv4.py": frozenset(
        {
            "ContextDeepSeekV4AttentionModule._cache_key",
            "ContextDeepSeekV4AttentionModule.clear_cache",
            "ContextDeepSeekV4AttentionModule.load_data",
            "ContextDeepSeekV4AttentionModule.load_data._load_sparse",
            "ContextDeepSeekV4AttentionModule.load_data._primary",
            "DeepSeekV4MHCModule.__init__",
            "DeepSeekV4MHCModule._cache_key",
            "DeepSeekV4MHCModule.clear_cache",
            "DeepSeekV4MHCModule.load_data",
            "DeepSeekV4MegaMoEModule.__init__",
            "DeepSeekV4MegaMoEModule._cache_key",
            "DeepSeekV4MegaMoEModule._engine_query_plan",
            "DeepSeekV4MegaMoEModule._normalize_distribution",
            "DeepSeekV4MegaMoEModule.clear_cache",
            "DeepSeekV4MegaMoEModule.load_data",
            "GenerationDeepSeekV4AttentionModule._cache_key",
            "GenerationDeepSeekV4AttentionModule.clear_cache",
            "GenerationDeepSeekV4AttentionModule.load_data",
            "_BaseDeepSeekV4AttentionModule.__init__",
            "_cache_key",
            "_deep_merge_dsv4_dicts",
            "_dsv4_normalize_dtype",
            "_load_dsv4_split",
            "_validate_dsv4_local_head_semantics",
            "load_context_dsv4_kind_module_data",
            "load_context_dsv4_kind_module_data._make_nested",
            "load_dsv4_sparse_kernel_data",
            "load_dsv4_sparse_op_data",
            "load_dsv4_sparse_op_data._coerce",
            "load_dsv4_sparse_op_data._is_bad_key",
            "load_generation_dsv4_kind_module_data",
            "load_generation_dsv4_kind_module_data._make_nested",
            "load_mhc_module_data",
        }
    ),
    "elementwise.py": frozenset(
        {
            "ElementWise.__init__",
        }
    ),
    "embedding.py": frozenset(
        {
            "Embedding.__init__",
        }
    ),
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
            "GEMM.__init__",
            "GEMM._cache_key",
            "GEMM._engine_query_plan",
            "GEMM.clear_cache",
            "GEMM.load_data",
            "GEMM.supported_quant_modes",
            "xprofile_util_level_known",
        }
    ),
    "mamba.py": frozenset(
        {
            "GDNKernel.__init__",
            "GDNKernel._cache_key",
            "GDNKernel.clear_cache",
            "GDNKernel.load_data",
            "KDAKernel.__init__",
            "KDAKernel.load_data",
            "Mamba2.__init__",
            "Mamba2.get_weights",
            "Mamba2.query",
            "Mamba2.query._gemm_value",
            "Mamba2.query._mem_value",
            "Mamba2Kernel.__init__",
            "Mamba2Kernel._cache_key",
            "Mamba2Kernel.clear_cache",
            "Mamba2Kernel.load_data",
            "_cache_key",
        }
    ),
    "mla.py": frozenset(
        {
            "ContextMLA.__init__",
            "ContextMLA._cache_key",
            "ContextMLA.clear_cache",
            "ContextMLA.load_data",
            "GenerationMLA.__init__",
            "GenerationMLA._cache_key",
            "GenerationMLA.clear_cache",
            "GenerationMLA.load_data",
            "MLABmm.__init__",
            "MLABmm._cache_key",
            "MLABmm._engine_query_plan",
            "MLABmm.clear_cache",
            "MLABmm.load_data",
            "MLAModule.__init__",
            "MLAModule._cache_key",
            "MLAModule.clear_cache",
            "MLAModule.load_data",
            "WideEPContextMLA.__init__",
            "WideEPContextMLA._cache_key",
            "WideEPContextMLA.clear_cache",
            "WideEPContextMLA.load_data",
            "WideEPGenerationMLA.__init__",
            "WideEPGenerationMLA._cache_key",
            "WideEPGenerationMLA.clear_cache",
            "WideEPGenerationMLA.load_data",
            "_cache_key",
        }
    ),
    "moe.py": frozenset(
        {
            "MoE.__init__",
            "MoE._cache_key",
            "MoE._engine_query_plan",
            "MoE.clear_cache",
            "MoE.load_data",
            "MoEDispatch.__init__",
            "MoEDispatch._cache_key",
            "MoEDispatch.clear_cache",
            "MoEDispatch.get_weights",
            "MoEDispatch.load_data",
            "TrtLLMWideEPMoE.__init__",
            "TrtLLMWideEPMoE._cache_key",
            "TrtLLMWideEPMoE._engine_query_plan",
            "TrtLLMWideEPMoE._select_kernel",
            "TrtLLMWideEPMoE.clear_cache",
            "TrtLLMWideEPMoE.get_weights",
            "TrtLLMWideEPMoE.load_data",
            "TrtLLMWideEPMoEDispatch.__init__",
            "TrtLLMWideEPMoEDispatch._cache_key",
            "TrtLLMWideEPMoEDispatch._engine_query_plan",
            "TrtLLMWideEPMoEDispatch._normalize_quant_mode_for_table",
            "TrtLLMWideEPMoEDispatch._select_alltoall_kernel",
            "TrtLLMWideEPMoEDispatch.clear_cache",
            "TrtLLMWideEPMoEDispatch.get_weights",
            "TrtLLMWideEPMoEDispatch.load_data",
            "_cache_key",
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
            "MoEAllToAll.__init__",
            "MoEAllToAll._cache_key",
            "MoEAllToAll.clear_cache",
            "MoEAllToAll.load_data",
            "MoECommBackendSpec.feasible",
            "MoEExpertCompute.__init__",
            "MoEExpertCompute._cache_key",
            "MoEExpertCompute._engine_query_plan",
            "MoEExpertCompute._resolve_kernel_source",
            "MoEExpertCompute.clear_cache",
            "MoEExpertCompute.load_data",
            "_adapt_legacy_deepep",
            "_adapt_legacy_deepep_ll",
            "_adapt_legacy_deepep_normal",
            "_adapt_legacy_sglang_context_moe",
            "_adapt_legacy_sglang_generation_moe",
            "_adapt_legacy_sglang_wideep_moe",
            "_adapt_legacy_trtllm_alltoall",
            "_adapt_legacy_trtllm_wideep_moe",
            "_cache_key",
            "_load_legacy_a2a",
            "_load_legacy_ep",
            "_moe_a2a_store",
            "_moe_ep_store",
            "_normalize_sms",
            "_require_latency",
            "_row_power",
            "_store_a2a_leaf",
            "_store_ep_leaf",
            "_validate_a2a_request",
            "_validate_ep_phase",
            "load_moe_a2a_data",
            "load_moe_expert_compute_data",
            "nodes_for",
        }
    ),
    "msa.py": frozenset(
        {
            "_BaseMSAModule.__init__",
            "_BaseMSAModule.load_data",
        }
    ),
    "overlap.py": frozenset(
        {
            "FallbackOp.__init__",
            "FallbackOp._engine_query_is_context",
            "FallbackOp._engine_query_plan",
            "OverlapOp.__init__",
            "OverlapOp._engine_query_is_context",
            "OverlapOp._engine_query_plan",
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


def _file_def_names(source_text: str) -> list[str]:
    """Every def at its qualified lexical path, in occurrence order (a list,
    not a set, so duplicate qualified redefinitions surface as duplicates)."""
    out: list[str] = []

    def walk(node, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qualified = f"{prefix}{child.name}"
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


def test_surviving_parsers_have_no_production_callers():
    """The moe/moe_comm/dsa/dsv4 ``load_*_data`` parsers survive PR-6 only as
    TEST-ONLY schema-contract fixtures (the collector suite's format
    handshake); every production ``PerfDatabase._<family>_data`` attribute is
    served by the engine table view. A production call site re-appearing for
    any of them is the Python data plane growing back — fail here so the
    revival is a deliberate, reviewed decision.
    """
    import re

    sdk_root = OPERATIONS_DIR.parents[2]  # aic-core/src/aiconfigurator_core
    surviving = sorted(
        {
            match
            for path in OPERATIONS_DIR.glob("*.py")
            for match in re.findall(r"^def (load_\w+_data)\b", path.read_text(encoding="utf-8"), re.MULTILINE)
        }
    )
    # The inventory itself is load-bearing: if the cleanup PR deletes the
    # parsers, this test retires with them (empty set would pass vacuously).
    assert surviving, "no surviving parsers found — retire this test together with the parser deletion"

    offenders = []
    for path in sdk_root.rglob("*.py"):
        rel = path.relative_to(sdk_root).as_posix()
        if "/operations/" in f"/{rel}":
            continue  # definitions and their module-internal use
        text = path.read_text(encoding="utf-8")
        for name in surviving:
            for match in re.finditer(rf"\b{name}\b", text):
                line = text[: match.start()].count("\n") + 1
                snippet = text.splitlines()[line - 1].strip()
                if snippet.startswith("#"):
                    continue  # commentary, not a call site
                # perf_database's legacy re-export block is the one allowed
                # reference (kept for the collector suite's import path).
                if rel.endswith("perf_database.py") and re.match(rf"^{name},?$", snippet):
                    continue
                offenders.append(f"{rel}:{line}: {snippet[:100]}")
    assert not offenders, "surviving test-only parsers gained production references:\n" + "\n".join(offenders)
