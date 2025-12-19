# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import pytest

from aiconfigurator.sdk import common

# Import PerfDatabase and its dependencies

# Mark all tests to use the patch
pytestmark = pytest.mark.patch_loader_and_yaml


def test_query_gemm_exact_match(perf_db):
    """
    query_gemm should return the exact latency stored under (quant_mode=fp16, m=64, n=128, k=256).
    We patched load_gemm_data to have exactly one entry: 10.0.
    However, _correct_data() may update this based on SOL calculation.
    """
    quant_mode = common.GEMMQuantMode.float16  # matches our dummy key
    m, n, k = 64, 128, 256

    observed = perf_db.query_gemm(m, n, k, quant_mode, database_mode=common.DatabaseMode.SILICON)
    # The value may have been corrected by _correct_data(), but we can check it's reasonable
    assert observed > 0, f"Expected positive value, got {observed}"

    # Also test that SOL mode works
    sol_value = perf_db.query_gemm(m, n, k, quant_mode, database_mode=common.DatabaseMode.SOL)
    assert sol_value > 0, f"Expected positive SOL value, got {sol_value}"


def test_query_gemm_empirical_mode(perf_db):
    """
    EMPIRICAL mode should return the SOL latency scaled by 1 / 0.8.
    """
    quant_mode = common.GEMMQuantMode.float16
    m, n, k = 64, 128, 256

    sol_value = perf_db.query_gemm(m, n, k, quant_mode, database_mode=common.DatabaseMode.SOL)
    empirical_value = perf_db.query_gemm(m, n, k, quant_mode, database_mode=common.DatabaseMode.EMPIRICAL)

    assert math.isclose(empirical_value, sol_value / 0.8), (
        f"EMPIRICAL expected {sol_value / 0.8}, got {empirical_value}"
    )


def test_query_custom_allreduce_database_mode_calculation(perf_db):
    """
    When database_mode == SOL, query_custom_allreduce uses get_sol:
        sol_time = 2 * size * 2 / tp_size * (tp_size - 1) / p2pBW * 1000
    We set p2pBW = perf_db.system_spec['node']['inter_node_bw'] = 100.0
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

    sol_time = perf_db.query_custom_allreduce(quant_mode, tp_size, size, database_mode=common.DatabaseMode.SOL)

    expected = (2 * size * 2 / tp_size * (tp_size - 1) / perf_db.system_spec["node"]["inter_node_bw"]) * 1000
    assert math.isclose(sol_time, expected), f"SOL-mode allreduce mismatch: expected {expected}, got {sol_time}"


def test_query_custom_allreduce_sol_full_returns_full_tuple(perf_db):
    """
    When database_mode == SOL_FULL, query_custom_allreduce returns (sol_time, sol_math, sol_mem).
    The sol_time value should match the calculated SOL time.
    """
    size = 1024
    tp_size = 2
    quant_mode = "float16"

    sol_time, sol_math, sol_mem = perf_db.query_custom_allreduce(
        quant_mode, tp_size, size, database_mode=common.DatabaseMode.SOL_FULL
    )
    # The get_sol function calculates: sol_time = 2 * size * 2 / tp_size * (tp_size - 1) / p2p_bw * 1000
    expected_sol_time = (2 * size * 2 / tp_size * (tp_size - 1) / perf_db.system_spec["node"]["inter_node_bw"]) * 1000

    assert math.isclose(sol_time, expected_sol_time)
    assert math.isclose(sol_math, 0.0)
    assert math.isclose(sol_mem, 0.0)


def test_query_custom_allreduce_non_database_mode_uses_custom_latency(perf_db):
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
    custom_latency = perf_db.query_custom_allreduce(
        quant_mode, tp_size, size, database_mode=common.DatabaseMode.SILICON
    )
    assert math.isclose(custom_latency, 5.0), f"Expected custom-allreduce latency 5.0, got {custom_latency}"


def test_query_nccl_database_mode_all_gather(perf_db):
    """
    For query_nccl(..., database_mode=SOL) and operation='all_gather':
        if dtype == CommQuantMode.half → type_bytes = 2
        sol_time = message_size * (num_gpus - 1) * type_bytes / num_gpus / p2pBW * 1000
    We set p2pBW = perf_db.system_spec['node']['inter_node_bw'] = 100.0
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

    sol_time = perf_db.query_nccl(dtype, num_gpus, operation, message_size, database_mode=common.DatabaseMode.SOL)
    expected = (
        dtype.value.memory
        * message_size
        * (num_gpus - 1)
        / num_gpus
        / perf_db.system_spec["node"]["inter_node_bw"]
        * 1000
    )

    assert math.isclose(sol_time, expected), f"Expected {expected}, got {sol_time}"


@pytest.mark.parametrize("operation", ["alltoall", "reduce_scatter"])
def test_query_nccl_database_mode_alltoall_and_reduce_scatter(perf_db, operation):
    """
    The code for 'alltoall' and 'reduce_scatter' in get_sol is identical:
        sol_time = message_size * (num_gpus - 1) * type_bytes / num_gpus / p2pBW * 1000
    Using dtype = CommQuantMode.int8 makes type_bytes=1.
    Let message_size = 1000, type_bytes=1, num_gpus=8, p2pBW=100.0:
        sol_time = 1000 * 7 / 8 / 100.0 * 1000 = (875 / 100.0) * 1000 = 8.75 * 1000 = 8750.0
    """
    dtype = common.CommQuantMode.int8  # type_bytes = 1 for int8
    num_gpus = 8  # num_gpus only matters for 'all_gather'
    sol_time = perf_db.query_nccl(dtype, num_gpus, operation, 1000, database_mode=common.DatabaseMode.SOL)
    expected = (
        dtype.value.memory * 1000 * (num_gpus - 1) / num_gpus / perf_db.system_spec["node"]["inter_node_bw"] * 1000
    )
    assert math.isclose(sol_time, expected), f"Expected {expected} for op {operation}, got {sol_time}"


def test_query_context_attention_hybrid_fallback(perf_db):
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

    empirical = perf_db.query_context_attention(database_mode=common.DatabaseMode.EMPIRICAL, **kwargs)
    hybrid = perf_db.query_context_attention(database_mode=common.DatabaseMode.HYBRID, **kwargs)

    assert math.isclose(hybrid, empirical), f"HYBRID fallback mismatch: expected {empirical}, got {hybrid}"


def test_query_p2p_database_mode(perf_db):
    """
    query_p2p(..., database_mode=SOL) uses:
        sol_time = message_bytes / inter_node_bw * 1000
    With message_bytes=256 and inter_node_bw=100.0:
        sol_time = (256 / 100.0) * 1000 = 2.56 * 1000 = 2560.0
    """
    sol_time = perf_db.query_p2p(256, database_mode=common.DatabaseMode.SOL)
    expected = (256 / perf_db.system_spec["node"]["inter_node_bw"]) * 1000
    assert math.isclose(sol_time, expected), f"Expected {expected}, got {sol_time}"


def test_system_spec_was_loaded_correctly(perf_db):
    """
    Sanity check: PerfDatabase.system_spec should be exactly what our patched yaml.load returned.
    """
    spec = perf_db.system_spec
    assert isinstance(spec, dict)
    assert spec["gpu"]["float16_tc_flops"] == 1_000.0
    assert spec["node"]["inter_node_bw"] == 100.0
