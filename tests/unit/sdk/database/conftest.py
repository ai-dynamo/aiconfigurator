# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import atexit
import os
import shutil
import tempfile
from collections import defaultdict
from unittest.mock import patch

import pytest
import yaml

from aiconfigurator.sdk import common
from aiconfigurator.sdk.operations import warm_all_op_data as _warm_lazy_op_caches
from aiconfigurator.sdk.operations.base import Operation
from aiconfigurator.sdk.perf_database import PerfDatabase

# ``_warm_lazy_op_caches`` is a thin alias for ``warm_all_op_data`` —
# the public helper that walks every ``Operation`` subclass and calls
# ``load_data(database)`` on it. Used by the fixtures below to warm
# every op's class cache while loader patches are still active so
# subsequent test queries hit the cache instead of opening real CSVs.


@pytest.fixture(autouse=True)
def _reset_op_load_counts():
    """Reset ``Operation._load_data_call_count`` between tests.

    Data caches stay intact (the comprehensive_perf_db singleton relies on
    them); only the instrumentation counter is reset so
    ``test_load_data_counts.py`` and any future lazy-load assertions start
    from a clean slate."""
    Operation._load_data_call_count.clear()
    yield
    Operation._load_data_call_count.clear()


# ---------------------------------------------------------------------------
# Synthetic-table injection (PR-6): op ``load_data`` binds ENGINE table views
# (``engine_table_view.fetch_table_view``), so tests stub that single choke
# point instead of the retired per-module loader functions. Overrides are
# keyed by the PerfDatabase attribute name and hold the REHYDRATED table
# shape (enum/int keys — exactly what the old loaders returned); anything not
# overridden fetches as ``None`` ("no source files"), which the binding wraps
# as an unloaded ``LoadedOpData`` (or ``None`` where the loader did).
# ---------------------------------------------------------------------------
_FETCH_VIEW_TARGET = "aiconfigurator_core.sdk.engine_table_view.fetch_table_view"


def _fake_fetch_table_view(overrides: dict[str, object]):
    def _fetch(database, attribute):
        if _COMPREHENSIVE_OVERRIDES is not None and getattr(database, "system", None) == "test_system":
            return _COMPREHENSIVE_OVERRIDES.get(attribute)
        return overrides.get(attribute)

    return _fetch


# ---------------------------------------------------------------------------
# Module-level view router for the comprehensive singleton's fake system.
# Installed ONCE at conftest import — i.e. before any test's monkeypatch can
# run — so scoped fetch patches always layer ON TOP of it and their teardown
# restores back TO it. The synthetic tables arrive later via
# _COMPREHENSIVE_OVERRIDES (set when the singleton is built); until then the
# router is a transparent pass-through.
# ---------------------------------------------------------------------------
from aiconfigurator_core.sdk import engine_table_view as _etv

_COMPREHENSIVE_OVERRIDES: dict[str, object] | None = None
_REAL_FETCH_TABLE_VIEW = _etv.fetch_table_view


def _routed_fetch(database, attribute):
    if _COMPREHENSIVE_OVERRIDES is not None and getattr(database, "system", None) == "test_system":
        return _COMPREHENSIVE_OVERRIDES.get(attribute)
    return _REAL_FETCH_TABLE_VIEW(database, attribute)


_etv.fetch_table_view = _routed_fetch


def _patch_all_loaders_and_yaml(monkeypatch) -> None:
    """
    Patch yaml + every data-loading function (so no real files are required).

    This keeps unit tests deterministic and avoids depending on large data files.
    """
    # Patch yaml.load so that PerfDatabase.__init__ sees a valid system_spec.
    dummy_system_spec = {
        # PerfDatabase uses a legacy-shaped logical lookup key, then resolves
        # physical files under data/<family>/<backend>/<version>.
        "data_dir": "data",
        "misc": {"nccl_version": "v1"},
        "gpu": {
            # These values are used in many "SOL"-mode formulas. The per-dtype
            # entries keep the historical 1/2/4 ratios so numeric assertions
            # written against the old bf16-scaled fallback stay valid.
            "bfloat16_tc_flops": 1_000.0,
            "int8_tc_flops": 2_000.0,
            "fp8_tc_flops": 2_000.0,
            "fp4_tc_flops": 4_000.0,
            "sm_version": 90,  # fp8-capable: the per-dtype entries above assume fp8/fp4 MMA exists
            "mem_bw": 100.0,
            # For query_nccl SILICON branch:
            "mem_empirical_constant_latency": 1.0,
        },
        "node": {
            # Used by query_custom_allreduce, query_nccl, query_p2p:
            "inter_node_bw": 100.0,
            "intra_node_bw": 100.0,
            "num_gpus_per_node": 8,
            "p2p_latency": 0.000001,
        },
    }
    monkeypatch.setattr(yaml, "load", lambda stream, Loader=None: dummy_system_spec)  # noqa: N803

    # Patch load_gemm_data to return a minimal nested dict keyed by
    # common.GEMMQuantMode.bfloat16 with multiple entries to avoid extrapolation errors.
    # Each entry includes {"latency": ..., "power": ..., "energy": ...}.
    dummy_gemm_data = {
        common.GEMMQuantMode.bfloat16: {
            64: {
                128: {
                    256: {"latency": 10.0, "power": 5.0, "energy": 50.0},  # at (m=64, n=128, k=256)
                    512: {"latency": 20.0, "power": 6.0, "energy": 120.0},
                },
                256: {
                    256: {"latency": 15.0, "power": 5.5, "energy": 82.5},
                    512: {"latency": 25.0, "power": 6.5, "energy": 162.5},
                },
            },
            128: {
                128: {
                    256: {"latency": 12.0, "power": 5.2, "energy": 62.4},
                    512: {"latency": 22.0, "power": 6.2, "energy": 136.4},
                },
                256: {
                    256: {"latency": 17.0, "power": 5.7, "energy": 96.9},
                    512: {"latency": 27.0, "power": 6.7, "energy": 180.9},
                },
            },
        }
    }

    # Patch load_custom_allreduce_data to return proper structure.
    # Structure: { 'bfloat16': { 2: { 'AUTO': { 1024:  5.0 } } } }
    dummy_custom_allreduce_data = {
        "bfloat16": {
            2: {"AUTO": {1024: 5.0, 2048: 15.0}},
            4: {"AUTO": {1024: 10.0, 2048: 20.0}},
            8: {"AUTO": {1024: 15.0, 2048: 30.0}},
        }
    }

    # Per-attribute view overrides for stub_perf_db; everything else fetches
    # as None. The full/skip DSA split is two distinct attributes now, so an
    # op_kind-swallowing stub cannot mask skip/full mis-wiring anymore.
    overrides = {
        "_gemm_data": dummy_gemm_data,
        "_custom_allreduce_data": dummy_custom_allreduce_data,
        "_moe_data": {},
        "_moe_low_latency_data": {},
    }
    monkeypatch.setattr(_FETCH_VIEW_TARGET, _fake_fetch_table_view(overrides))


@pytest.fixture
def stub_perf_db(tmp_path, monkeypatch):
    """
    Instantiate a PerfDatabase with "dummy" system/backend/version and patched loaders.
    No actual performance data files are needed.
    """
    _patch_all_loaders_and_yaml(monkeypatch)
    system = "any_system"
    backend = "any_backend"
    version = "v1"
    systems_dir = str(tmp_path)  # path is never actually read because of our patches

    # Create the yaml file to avoid FileNotFoundError
    yaml_file = tmp_path / f"{system}.yaml"
    yaml_file.write_text("dummy: data")  # Content doesn't matter because yaml.load is patched

    db = PerfDatabase(system, backend, version, systems_dir)
    _warm_lazy_op_caches(db)
    return db


def _build_comprehensive_test_data():
    """Build all the dummy data dicts for comprehensive_perf_db."""
    dummy_system_spec = {
        "data_dir": "data",
        "misc": {"nccl_version": "v1"},
        "gpu": {
            # Per-dtype entries keep the historical 1/2/4 ratios so numeric
            # assertions written against the old bf16-scaled fallback stay valid.
            "bfloat16_tc_flops": 1_000_000_000_000.0,  # 1 TFLOPS
            "int8_tc_flops": 2_000_000_000_000.0,
            "fp8_tc_flops": 2_000_000_000_000.0,
            "fp4_tc_flops": 4_000_000_000_000.0,
            "sm_version": 90,  # fp8-capable: the per-dtype entries above assume fp8/fp4 MMA exists
            "mem_bw": 1_000_000_000_000.0,  # 1 TB/s
            "mem_bw_empirical_scaling_factor": 0.8,
            "mem_empirical_constant_latency": 0.001,  # 1 us
        },
        "node": {
            "inter_node_bw": 100_000_000_000.0,  # 100 GB/s
            "intra_node_bw": 200_000_000_000.0,  # 200 GB/s
            "num_gpus_per_node": 8,
            "p2p_latency": 0.000001,  # 1 us
        },
    }

    # Comprehensive GEMM data with energy
    dummy_gemm_data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict())))
    for quant_mode in [common.GEMMQuantMode.bfloat16, common.GEMMQuantMode.fp8]:
        for m in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
            for n in [128, 256, 512, 1024]:
                for k in [128, 256, 512, 1024]:
                    latency = 0.1 + m * 0.001 + n * 0.0001 + k * 0.00001
                    power = 5.0 + m * 0.01  # Dummy power value
                    energy = power * latency
                    dummy_gemm_data[quant_mode][m][n][k] = {
                        "latency": latency,
                        "power": power,
                        "energy": energy,
                    }

    # Context attention data
    dummy_context_attention_data = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(
                    lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict())))
                )
            )
        )
    )
    for quant_mode in [common.FMHAQuantMode.bfloat16, common.FMHAQuantMode.fp8]:
        for kv_cache_dtype in [common.KVCacheQuantMode.bfloat16, common.KVCacheQuantMode.fp8]:
            for kv_n in [0, 1, 2, 4, 8]:  # 0 means MHA
                for head_size in [64, 128]:
                    for window_size in [0, 128]:
                        for n in [4, 8, 16, 32]:
                            for s in [16, 32, 64, 128, 256]:
                                for b in [1, 2, 4, 8]:
                                    dummy_context_attention_data[quant_mode][kv_cache_dtype][kv_n][head_size][
                                        window_size
                                    ][n][s][b] = 0.01 * (n * s * b) / 1000.0

    # Encoder (non-causal) attention data: [fmha_quant][head_size][n][s][b]
    # MHA only, no kv_cache_dtype, no window_size.
    dummy_encoder_attention_data = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict())))
    )
    for quant_mode in [common.FMHAQuantMode.bfloat16]:
        for head_size in [64, 72, 80]:
            for n in [4, 8, 16, 32]:
                for s in [16, 32, 64, 128, 256]:
                    for b in [1, 2, 4, 8]:
                        dummy_encoder_attention_data[quant_mode][head_size][n][s][b] = {
                            "latency": 0.02 * (n * s * b) / 1000.0,
                            "power": 0.0,
                            "energy": 0.0,
                        }

    # Generation attention data
    dummy_generation_attention_data = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict()))))
        )
    )
    for kv_cache_dtype in [common.KVCacheQuantMode.bfloat16, common.KVCacheQuantMode.fp8]:
        for kv_n in [0, 1, 2, 4, 8]:
            for head_size in [64, 128]:
                for window_size in [0, 128]:
                    for n in [4, 8, 16, 32]:
                        # Only create data where kv_n <= n to satisfy the assertion
                        if kv_n <= n:
                            for b in [1, 2, 4, 8, 16]:
                                for s in [1, 16, 32, 64, 128, 256, 512, 1024]:
                                    dummy_generation_attention_data[kv_cache_dtype][kv_n][head_size][window_size][n][b][
                                        s
                                    ] = 0.001 * (n * b * s) / 1000.0

    # MoE data
    dummy_moe_data = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(
                    lambda: defaultdict(
                        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict())))
                    )
                )
            )
        )
    )
    for quant_mode in [common.MoEQuantMode.bfloat16, common.MoEQuantMode.fp8]:
        for workload in ["uniform", "imbalanced"]:
            for topk in [1, 2]:
                for num_experts in [8, 16]:
                    for hidden_size in [1024, 2048, 4096]:
                        for inter_size in [4096, 8192]:
                            for moe_tp in [1, 2]:
                                for moe_ep in [1, 2]:
                                    for num_tokens in [1, 2, 4, 8, 16, 32]:
                                        dummy_moe_data[quant_mode][workload][topk][num_experts][hidden_size][
                                            inter_size
                                        ][moe_tp][moe_ep][num_tokens] = 0.1 * num_tokens

    # Context MLA data
    dummy_context_mla_data = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict()))))
    )
    for quant_mode in [common.FMHAQuantMode.bfloat16]:
        for kv_cache_dtype in [common.KVCacheQuantMode.bfloat16]:
            for num_heads in [16, 32, 64, 128]:
                for s in [16, 32, 64, 128]:
                    for b in [1, 2, 4, 8]:
                        dummy_context_mla_data[quant_mode][kv_cache_dtype][num_heads][s][b] = 0.0001 * s * b * num_heads

    # Generation MLA data
    dummy_generation_mla_data = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict())))
    )
    for kv_cache_dtype in [common.KVCacheQuantMode.bfloat16]:
        for num_heads in [16, 32, 64, 128]:
            for b in [1, 2, 4, 8]:
                for s in [1, 16, 32, 64, 128]:
                    dummy_generation_mla_data[kv_cache_dtype][num_heads][b][s] = 0.00001 * b * s * num_heads

    # MLA BMM data
    dummy_mla_bmm_data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict())))
    for quant_mode in [common.GEMMQuantMode.bfloat16, common.GEMMQuantMode.fp8]:
        for op_name in ["mla_gen_pre", "mla_gen_post"]:
            for num_heads in [1, 2, 4, 8]:
                for num_tokens in [1, 2, 4, 8, 16, 32]:
                    dummy_mla_bmm_data[quant_mode][op_name][num_heads][num_tokens] = 0.01 * num_heads * num_tokens

    # Custom allreduce data
    dummy_custom_allreduce_data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict())))
    for dtype in [common.CommQuantMode.half]:
        for tp_size in [1, 2, 4, 8]:
            for strategy in ["AUTO"]:
                for msg_size in [512, 1024, 2048, 4096, 8192]:
                    dummy_custom_allreduce_data[dtype][tp_size][strategy][msg_size] = 0.001 * msg_size * tp_size

    # NCCL data
    dummy_nccl_data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict())))
    for dtype in [common.CommQuantMode.half, common.CommQuantMode.int8]:  # Use enum objects as keys
        for operation in ["all_gather", "alltoall", "reduce_scatter"]:
            for num_gpus in [1, 2, 4, 8]:
                for msg_size in [512, 1024, 2048, 4096]:
                    dummy_nccl_data[dtype][operation][num_gpus][msg_size] = 0.001 * msg_size * num_gpus

    return {
        "system_spec": dummy_system_spec,
        "gemm_data": dummy_gemm_data,
        # Both attention loaders key on kernel_source first (AIC-1715); rows
        # without one land in lane "default", which is what these fixtures
        # impersonate.
        "context_attention_data": {"default": dummy_context_attention_data},
        "encoder_attention_data": dummy_encoder_attention_data,
        "generation_attention_data": {"default": dummy_generation_attention_data},
        "moe_data": dummy_moe_data,
        "context_mla_data": dummy_context_mla_data,
        "generation_mla_data": dummy_generation_mla_data,
        "mla_bmm_data": dummy_mla_bmm_data,
        "custom_allreduce_data": dummy_custom_allreduce_data,
        "nccl_data": dummy_nccl_data,
    }


# Module-level singleton: the PerfDatabase is built once (including the expensive
# _correct_data + _extrapolate_data_grid passes in __init__) and reused by ALL tests.
_comprehensive_db_singleton: PerfDatabase | None = None


# Generation of ``engine._PROBE_CACHE_GENERATION`` the singleton was built
# under. Recorded explicitly at build time: this database never reaches the
# real ``fetch_table_view`` (the module router answers its fetches from
# ``_COMPREHENSIVE_OVERRIDES``), so the engine-side probe-spec memo a
# database normally carries is never written for it and cannot be used as
# the eviction signal.
_comprehensive_db_generation: int | None = None


def _get_comprehensive_db_singleton() -> PerfDatabase:
    """Build (or rebuild after an eviction lever fired) the shared singleton.

    ``clear_all_op_caches()`` and the other documented eviction levers wipe
    the per-op class caches and advance ``engine._PROBE_CACHE_GENERATION``.
    Tests that fire a lever while a scoped stub fetch is active (e.g. the
    router-survival test in ``test_data_loaders.py``) leave the shared class
    caches re-populated from the stub under this singleton's cache key, so
    whichever test on the xdist worker touches the singleton next reads
    stub-shaped tables and fails order-dependently (AIC-1777:
    TestSupportedQuantModes / TestUpdateSupportMatrix). Rebuilding whenever
    the generation moved since the build cures that: the rebuild happens at
    the NEXT fixture request, which in those orders is outside the stub
    window, and steady-state accesses (no lever fired) keep the documented
    same-object singleton contract.
    """
    global _comprehensive_db_singleton, _comprehensive_db_generation
    from aiconfigurator_core.sdk import engine as _engine

    if _comprehensive_db_singleton is not None and _comprehensive_db_generation == _engine._PROBE_CACHE_GENERATION:
        return _comprehensive_db_singleton
    _comprehensive_db_singleton = None

    cached = _build_comprehensive_test_data()
    system_spec = cached["system_spec"]

    tmp_dir = tempfile.mkdtemp()
    # Superseded singleton dirs must survive until process exit: deepcopied
    # databases (mutable_comprehensive_perf_db) keep referencing the old
    # systems path, and a post-eviction engine re-fetch through such a copy
    # would turn "empty tables" into a missing-directory error if the dir
    # were removed eagerly on rebuild. Clean them all up at exit instead.
    atexit.register(shutil.rmtree, tmp_dir, ignore_errors=True)
    yaml_file = os.path.join(tmp_dir, "test_system.yaml")
    with open(yaml_file, "w") as f:
        yaml.dump(system_spec, f)

    # Per-attribute view overrides — anything not listed fetches as None.
    # (The old tuple-returning load_moe_data fed BOTH moe tables the same
    # dict; keep that shape.)
    overrides = {
        "_gemm_data": cached["gemm_data"],
        "_context_attention_data": cached["context_attention_data"],
        "_generation_attention_data": cached["generation_attention_data"],
        "_encoder_attention_data": cached["encoder_attention_data"],
        "_moe_data": cached["moe_data"],
        "_moe_low_latency_data": cached["moe_data"],
        "_custom_allreduce_data": cached["custom_allreduce_data"],
        "_nccl_data": cached["nccl_data"],
        "_context_mla_data": cached["context_mla_data"],
        "_generation_mla_data": cached["generation_mla_data"],
        "_mla_bmm_data": cached["mla_bmm_data"],
    }

    # Publish the synthetic tables to the MODULE-LEVEL router (installed at
    # conftest import, before any monkeypatch can run — see _routed_fetch
    # below the stub helpers). Publishing state instead of swapping the
    # function here keeps two properties: the routing survives any later
    # ``clear_all_op_caches()`` (a re-load of the singleton or a deepcopy
    # must keep resolving to the synthetic tables), AND a scoped
    # ``stub_perf_db`` monkeypatch that happens to be active while this
    # singleton is first built cannot be captured as the "real" fetch and
    # then torn down out from under the router.
    global _COMPREHENSIVE_OVERRIDES
    _COMPREHENSIVE_OVERRIDES = overrides

    yaml_patch = patch("yaml.load", side_effect=lambda stream, Loader=None: system_spec)  # noqa: N803
    yaml_patch.start()
    try:
        _comprehensive_db_singleton = PerfDatabase("test_system", "trtllm", "v1", tmp_dir)
        _warm_lazy_op_caches(_comprehensive_db_singleton)
        _comprehensive_db_generation = _engine._PROBE_CACHE_GENERATION
    finally:
        yaml_patch.stop()

    return _comprehensive_db_singleton


@pytest.fixture
def comprehensive_perf_db():
    """
    Return a **shared, read-only** PerfDatabase with comprehensive test data.

    This is a singleton — the same object is returned to every test.
    This avoids the expensive __init__ (data loading + correction + extrapolation)
    that previously ran per-test and dominated the test suite runtime.

    DO NOT mutate the returned object. If a test needs to modify the db,
    use the ``mutable_comprehensive_perf_db`` fixture instead.
    """
    return _get_comprehensive_db_singleton()


@pytest.fixture
def mutable_comprehensive_perf_db():
    """
    Return an independent deep copy of the comprehensive PerfDatabase.

    Use this instead of ``comprehensive_perf_db`` when the test needs to mutate
    the database (e.g. replacing data dicts, modifying system_spec, calling
    _correct_data). Each invocation returns a fresh copy so mutations cannot
    leak between tests.

    Slower than the shared fixture (~1s for deepcopy) but much faster than
    re-running PerfDatabase.__init__ from scratch.
    """
    import copy

    return copy.deepcopy(_get_comprehensive_db_singleton())
