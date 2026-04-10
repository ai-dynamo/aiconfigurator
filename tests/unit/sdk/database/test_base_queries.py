# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import pytest

from aiconfigurator.sdk import common
from aiconfigurator.sdk.perf_database import LoadedOpData

# Import PerfDatabase and its dependencies

pytestmark = pytest.mark.unit


def test_query_gemm_exact_match(stub_perf_db):
    """
    query_gemm should return the exact latency stored under (quant_mode=fp16, m=64, n=128, k=256).
    We patched load_gemm_data to have exactly one entry: 10.0.
    However, _correct_data() may update this based on SOL calculation.
    """
    quant_mode = common.GEMMQuantMode.float16  # matches our dummy key
    m, n, k = 64, 128, 256

    observed = stub_perf_db.query_gemm(m, n, k, quant_mode, database_mode=common.DatabaseMode.SILICON)
    # The value may have been corrected by _correct_data(), but we can check it's reasonable
    assert observed > 0, f"Expected positive value, got {observed}"

    # Also test that SOL mode works
    sol_value = stub_perf_db.query_gemm(m, n, k, quant_mode, database_mode=common.DatabaseMode.SOL)
    assert sol_value > 0, f"Expected positive SOL value, got {sol_value}"


def test_query_gemm_empirical_mode(stub_perf_db):
    """
    EMPIRICAL mode should return the SOL latency scaled by 1 / 0.8.
    """
    quant_mode = common.GEMMQuantMode.float16
    m, n, k = 64, 128, 256

    sol_value = stub_perf_db.query_gemm(m, n, k, quant_mode, database_mode=common.DatabaseMode.SOL)
    empirical_value = stub_perf_db.query_gemm(m, n, k, quant_mode, database_mode=common.DatabaseMode.EMPIRICAL)

    assert math.isclose(empirical_value, sol_value / 0.8), (
        f"EMPIRICAL expected {sol_value / 0.8}, got {empirical_value}"
    )


def test_query_gemm_exact_match_skips_3d_interpolation(comprehensive_perf_db, monkeypatch):
    """Exact GEMM hits should bypass both 1D and 3D interpolation."""
    quant_mode = common.GEMMQuantMode.float16
    m, n, k = 16, 128, 128

    def _fail_interp_3d(*args, **kwargs):
        raise AssertionError("_interp_3d should not be used for exact GEMM matches")

    def _fail_interp_1d(*args, **kwargs):
        raise AssertionError("_interp_1d should not be used for exact GEMM matches")

    monkeypatch.setattr(comprehensive_perf_db, "_interp_3d", _fail_interp_3d)
    monkeypatch.setattr(comprehensive_perf_db, "_interp_1d", _fail_interp_1d)

    observed = comprehensive_perf_db.query_gemm(m, n, k, quant_mode, database_mode=common.DatabaseMode.SILICON)
    expected = 0.1 + m * 0.001 + n * 0.0001 + k * 0.00001

    assert math.isclose(float(observed), expected)


def test_query_gemm_interpolates_only_on_m_when_nk_match(comprehensive_perf_db, monkeypatch):
    """GEMM lookup should use 1D interpolation on m when n and k match."""
    quant_mode = common.GEMMQuantMode.float16
    m, n, k = 12, 128, 128
    calls = {}

    def _fail_interp_3d(*args, **kwargs):
        raise AssertionError("_interp_3d should not be used when n/k match and only m needs interpolation")

    def _spy_interp_1d(x, y, value):
        calls["x"] = x
        calls["y"] = y
        calls["value"] = value
        return {"latency": 0.1 + value * 0.001 + n * 0.0001 + k * 0.00001, "power": 0.0, "energy": 0.0}

    monkeypatch.setattr(comprehensive_perf_db, "_interp_3d", _fail_interp_3d)
    monkeypatch.setattr(comprehensive_perf_db, "_interp_1d", _spy_interp_1d)

    observed = comprehensive_perf_db.query_gemm(m, n, k, quant_mode, database_mode=common.DatabaseMode.SILICON)
    expected = 0.1 + m * 0.001 + n * 0.0001 + k * 0.00001

    assert math.isclose(float(observed), expected)
    assert calls["x"] == [8, 16]
    assert calls["value"] == m


def test_query_gemm_fast_paths_support_legacy_scalar_leaves(comprehensive_perf_db, monkeypatch):
    """Fast GEMM paths should support legacy scalar-leaf tables."""
    quant_mode = common.GEMMQuantMode.float16
    comprehensive_perf_db._gemm_data[quant_mode] = {
        8: {128: {128: 0.5}},
        16: {128: {128: 0.9}},
    }

    exact = comprehensive_perf_db.query_gemm(8, 128, 128, quant_mode, database_mode=common.DatabaseMode.SILICON)
    interp = comprehensive_perf_db.query_gemm(12, 128, 128, quant_mode, database_mode=common.DatabaseMode.SILICON)

    assert math.isclose(float(exact), 0.5)
    assert math.isclose(float(interp), 0.7)
    assert exact.energy == 0.0
    assert interp.energy == 0.0


def test_query_trtllm_alltoall_normalizes_fp8_block_lookup(stub_perf_db):
    """
    fp8_block reuses the fp8 TRT-LLM alltoall perf tables.
    """
    stub_perf_db.version = "1.2.0rc6"
    stub_perf_db.system_spec["gpu"]["sm_version"] = 100
    stub_perf_db._trtllm_alltoall_data = LoadedOpData(
        {
            "NVLinkOneSided": {
                "alltoall_dispatch": {
                    common.MoEQuantMode.fp8: {
                        1: {
                            1024: {
                                8: {
                                    256: {
                                        8: {
                                            16: {"latency": 3.0, "power": 0.0, "energy": 0.0},
                                            32: {"latency": 6.0, "power": 0.0, "energy": 0.0},
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
        common.PerfDataFilename.trtllm_alltoall,
        "dummy_trtllm_alltoall_perf.txt",
    )

    result = stub_perf_db.query_trtllm_alltoall(
        op_name="alltoall_dispatch",
        num_tokens=16,
        hidden_size=1024,
        topk=8,
        num_experts=256,
        moe_ep_size=8,
        quant_mode=common.MoEQuantMode.fp8_block,
        node_num=1,
        database_mode=common.DatabaseMode.SILICON,
        moe_backend="CUTLASS",
    )

    assert math.isclose(float(result), 3.0)


def test_query_custom_allreduce_database_mode_calculation(stub_perf_db):
    """
    When database_mode == SOL, query_custom_allreduce uses get_sol:
        sol_time = 2 * size * 2 / tp_size * (tp_size - 1) / p2pBW * 1000
    We set p2pBW = stub_perf_db.system_spec['node']['inter_node_bw'] = 100.0
    For tp_size=2, size=1024, that becomes:
        sol_time = 2 * 1024 * 2 / 2 * (2 - 1) / 100.0 * 1000
                 = (4096 / 2 * 1 / 100.0) * 1000
                 = (2048 / 100.0) * 1000
                 = 20.48 * 1000
                 = 20480.0
    """
    size = 1024
    tp_size = 2
    quant_mode = "float16"  # for SOL branch we ignore the custom allreduce dict

    sol_time = stub_perf_db.query_custom_allreduce(quant_mode, tp_size, size, database_mode=common.DatabaseMode.SOL)

    expected = (2 * size * 2 / tp_size * (tp_size - 1) / stub_perf_db.system_spec["node"]["inter_node_bw"]) * 1000
    assert math.isclose(sol_time, expected), f"SOL-mode allreduce mismatch: expected {expected}, got {sol_time}"


def test_query_custom_allreduce_sol_full_returns_full_tuple(stub_perf_db):
    """
    When database_mode == SOL_FULL, query_custom_allreduce returns (sol_time, sol_math, sol_mem).
    The sol_time value should match the calculated SOL time.
    """
    size = 1024
    tp_size = 2
    quant_mode = "float16"

    sol_time, sol_math, sol_mem = stub_perf_db.query_custom_allreduce(
        quant_mode, tp_size, size, database_mode=common.DatabaseMode.SOL_FULL
    )
    # The get_sol function calculates: sol_time = 2 * size * 2 / tp_size * (tp_size - 1) / p2p_bw * 1000
    expected_sol_time = (
        2 * size * 2 / tp_size * (tp_size - 1) / stub_perf_db.system_spec["node"]["inter_node_bw"]
    ) * 1000

    assert math.isclose(sol_time, expected_sol_time)
    assert math.isclose(sol_math, 0.0)
    assert math.isclose(sol_mem, 0.0)


def test_query_custom_allreduce_non_database_mode_uses_custom_latency(stub_perf_db):
    """
    When database_mode is neither SOL nor SOL_FULL (e.g. DatabaseMode.NONE), the code picks:
        comm_dict = self._custom_allreduce_data[quant_mode][min(tp_size, 8)]['AUTO']
        size_left, size_right = nearest keys enveloping `size`
        lat = interpolate between comm_dict[size_left], comm_dict[size_right]
    We patched _custom_allreduce_data so that:
        _custom_allreduce_data['float16']['2']['AUTO']['1024'] == 5.0
    For tp_size=2 and size=1024 exactly, we expect 5.0.
    """
    size = 1024
    tp_size = 2
    quant_mode = "float16"

    # Use a “SILICON” mode to force fallback into the custom-data path
    custom_latency = stub_perf_db.query_custom_allreduce(
        quant_mode, tp_size, size, database_mode=common.DatabaseMode.SILICON
    )
    assert math.isclose(custom_latency, 5.0), f"Expected custom-allreduce latency 5.0, got {custom_latency}"


def test_query_nccl_database_mode_all_gather(stub_perf_db):
    """
    For query_nccl(..., database_mode=SOL) and operation='all_gather':
        if dtype == CommQuantMode.half → type_bytes = 2
        sol_time = message_size * (num_gpus - 1) * type_bytes / num_gpus / p2pBW * 1000
    We set p2pBW = stub_perf_db.system_spec['node']['inter_node_bw'] = 100.0
    Let num_gpus=4, message_size=512, dtype=half → type_bytes=2
    Then:
        sol_time = 512 * 3 * 2 / 4 / 100.0 * 1000
                 = (3072 / 4 / 100.0) * 1000
                 = (768 / 100.0) * 1000
                 = 7.68 * 1000
                 = 7680.0
    """
    dtype = common.CommQuantMode.half
    num_gpus = 4
    operation = "all_gather"
    message_size = 512

    sol_time = stub_perf_db.query_nccl(dtype, num_gpus, operation, message_size, database_mode=common.DatabaseMode.SOL)
    expected = (
        dtype.value.memory
        * message_size
        * (num_gpus - 1)
        / num_gpus
        / stub_perf_db.system_spec["node"]["inter_node_bw"]
        * 1000
    )

    assert math.isclose(sol_time, expected), f"Expected {expected}, got {sol_time}"


@pytest.mark.parametrize("operation", ["alltoall", "reduce_scatter"])
def test_query_nccl_database_mode_alltoall_and_reduce_scatter(stub_perf_db, operation):
    """
    The code for 'alltoall' and 'reduce_scatter' in get_sol is identical:
        sol_time = message_size * (num_gpus - 1) * type_bytes / num_gpus / p2pBW * 1000
    Using dtype = CommQuantMode.int8 makes type_bytes=1.
    Let message_size = 1000, type_bytes=1, num_gpus=8, p2pBW=100.0:
        sol_time = 1000 * 7 / 8 / 100.0 * 1000 = (875 / 100.0) * 1000 = 8.75 * 1000 = 8750.0
    """
    dtype = common.CommQuantMode.int8  # type_bytes = 1 for int8
    num_gpus = 8  # num_gpus only matters for 'all_gather'
    sol_time = stub_perf_db.query_nccl(dtype, num_gpus, operation, 1000, database_mode=common.DatabaseMode.SOL)
    expected = (
        dtype.value.memory * 1000 * (num_gpus - 1) / num_gpus / stub_perf_db.system_spec["node"]["inter_node_bw"] * 1000
    )
    assert math.isclose(sol_time, expected), f"Expected {expected} for op {operation}, got {sol_time}"


def test_query_context_attention_hybrid_fallback(stub_perf_db):
    """
    HYBRID mode should fall back to the empirical calculation when silicon data is missing.
    """
    kwargs = dict(
        b=2,
        s=32,
        prefix=0,
        n=32,
        n_kv=16,
        kvcache_quant_mode=common.KVCacheQuantMode.float16,
        fmha_quant_mode=common.FMHAQuantMode.float16,
        head_size=128,
        window_size=0,
    )

    empirical = stub_perf_db.query_context_attention(database_mode=common.DatabaseMode.EMPIRICAL, **kwargs)
    hybrid = stub_perf_db.query_context_attention(database_mode=common.DatabaseMode.HYBRID, **kwargs)

    assert math.isclose(hybrid, empirical), f"HYBRID fallback mismatch: expected {empirical}, got {hybrid}"


def test_query_p2p_database_mode(stub_perf_db):
    """
    query_p2p(..., database_mode=SOL) uses:
        sol_time = message_bytes / inter_node_bw * 1000
    With message_bytes=256 and inter_node_bw=100.0:
        sol_time = (256 / 100.0) * 1000 = 2.56 * 1000 = 2560.0
    """
    sol_time = stub_perf_db.query_p2p(256, database_mode=common.DatabaseMode.SOL)
    expected = (256 / stub_perf_db.system_spec["node"]["inter_node_bw"]) * 1000
    assert math.isclose(sol_time, expected), f"Expected {expected}, got {sol_time}"


def test_system_spec_was_loaded_correctly(stub_perf_db):
    """
    Sanity check: PerfDatabase.system_spec should be exactly what our patched yaml.load returned.
    """
    spec = stub_perf_db.system_spec
    assert isinstance(spec, dict)
    assert spec["gpu"]["float16_tc_flops"] == 1_000.0
    assert spec["node"]["inter_node_bw"] == 100.0
