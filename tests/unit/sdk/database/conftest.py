# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import tempfile
from collections import defaultdict
from unittest.mock import patch

import pytest
import yaml

from aiconfigurator.sdk import common
from aiconfigurator.sdk.perf_database import PerfDatabase

# ---------------------------------------------------------------------------
# Single source of truth: every loader function that PerfDatabase.__init__
# can call via its internal _load_op_data / func_map.
#
# Mapping: function_name -> default stub return value.
#   * Most loaders return None when the perf file is absent.
#   * load_moe_data is special: it returns a *tuple* of two dicts.
#   * load_nccl_data covers both nccl and oneccl keys in func_map.
#
# When PerfDatabase adds a new loader, add a line here.  Both
# _patch_all_loaders_and_yaml() and _get_comprehensive_db_singleton()
# derive their patch lists from this dict, so they cannot drift.
# ---------------------------------------------------------------------------
_LOADER_STUBS: dict[str, object] = {
    "load_gemm_data": None,
    "load_context_attention_data": None,
    "load_generation_attention_data": None,
    "load_moe_data": (None, None),  # returns tuple
    "load_custom_allreduce_data": None,
    "load_nccl_data": None,  # also used for oneccl
    "load_context_mla_data": None,
    "load_generation_mla_data": None,
    "load_mla_bmm_data": None,
    "load_mamba2_data": None,
    "load_gdn_data": None,
    "load_compute_scale_data": None,
    "load_scale_matrix_data": None,
    "load_wideep_context_moe_data": None,
    "load_wideep_generation_moe_data": None,
    "load_wideep_context_mla_data": None,
    "load_wideep_generation_mla_data": None,
    "load_wideep_deepep_normal_data": None,
    "load_wideep_deepep_ll_data": None,
    "load_wideep_moe_compute_data": None,
    "load_trtllm_alltoall_data": None,
    "load_context_mla_module_data": None,
    "load_generation_mla_module_data": None,
    "load_context_dsa_module_data": None,
    "load_generation_dsa_module_data": None,
    "load_mhc_module_data": None,
    "load_context_deepseek_v4_attention_module_data": None,
    "load_generation_deepseek_v4_attention_module_data": None,
}


def _patch_all_loaders_and_yaml(monkeypatch) -> None:
    """
    Patch yaml + every data-loading function (so no real files are required).

    This keeps unit tests deterministic and avoids depending on large data files.
    """
    # Patch yaml.load so that PerfDatabase.__init__ sees a valid system_spec.
    dummy_system_spec = {
        "data_dir": "data",  # PerfDatabase will look under systems_dir/data/<backend>/<version>
        "misc": {"nccl_version": "v1"},
        "gpu": {
            # These two values are used in many "SOL"-mode formulas:
            "bfloat16_tc_flops": 1_000.0,
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

    # Per-loader overrides for stub_perf_db (most stay at the default from _LOADER_STUBS)
    overrides = {
        "load_gemm_data": dummy_gemm_data,
        "load_custom_allreduce_data": dummy_custom_allreduce_data,
        "load_moe_data": ({}, {}),
    }

    for name, default_value in _LOADER_STUBS.items():
        ret = overrides.get(name, default_value)
        monkeypatch.setattr(f"aiconfigurator.sdk.perf_database.{name}", lambda path, _r=ret: _r)


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

    return PerfDatabase(system, backend, version, systems_dir)


def _build_comprehensive_test_data():
    """Build all the dummy data dicts for comprehensive_perf_db."""
    dummy_system_spec = {
        "data_dir": "data",
        "misc": {"nccl_version": "v1"},
        "gpu": {
            "bfloat16_tc_flops": 1_000_000_000_000.0,  # 1 TFLOPS
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
        "context_attention_data": dummy_context_attention_data,
        "generation_attention_data": dummy_generation_attention_data,
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


def _get_comprehensive_db_singleton() -> PerfDatabase:
    """Build and cache a fully-initialized PerfDatabase singleton."""
    global _comprehensive_db_singleton
    if _comprehensive_db_singleton is not None:
        return _comprehensive_db_singleton

    cached = _build_comprehensive_test_data()
    system_spec = cached["system_spec"]

    tmp_dir = tempfile.mkdtemp()
    yaml_file = os.path.join(tmp_dir, "test_system.yaml")
    with open(yaml_file, "w") as f:
        yaml.dump(system_spec, f)

    # Use unittest.mock.patch (not monkeypatch) so we can call this outside a fixture.
    # Per-loader overrides — anything not listed here falls through to
    # the default in _LOADER_STUBS (None for most, (None, None) for moe).
    overrides = {
        "load_gemm_data": cached["gemm_data"],
        "load_context_attention_data": cached["context_attention_data"],
        "load_generation_attention_data": cached["generation_attention_data"],
        "load_moe_data": (cached["moe_data"], cached["moe_data"]),
        "load_custom_allreduce_data": cached["custom_allreduce_data"],
        "load_nccl_data": cached["nccl_data"],
        "load_context_mla_data": cached["context_mla_data"],
        "load_generation_mla_data": cached["generation_mla_data"],
        "load_mla_bmm_data": cached["mla_bmm_data"],
    }
    patches = [
        patch("yaml.load", side_effect=lambda stream, Loader=None: system_spec),  # noqa: N803
    ]
    for name, default_value in _LOADER_STUBS.items():
        ret = overrides.get(name, default_value)
        patches.append(patch(f"aiconfigurator.sdk.perf_database.{name}", return_value=ret))

    for p in patches:
        p.start()
    try:
        _comprehensive_db_singleton = PerfDatabase("test_system", "trtllm", "v1", tmp_dir)
    finally:
        for p in patches:
            p.stop()

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
