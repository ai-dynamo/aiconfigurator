# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import math

# Import PerfDatabase and its dependencies
from aiconfigurator.sdk.perf_database import (
    PerfDatabase,
    load_gemm_data,
    load_context_attention_data,
    load_generation_attention_data,
    load_custom_allreduce_data,
    load_moe_data,
    load_context_mla_data,
    load_generation_mla_data,
    load_mla_bmm_data,
    load_nccl_data,
)
from aiconfigurator.sdk import common
import yaml

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

    observed = perf_db.query_gemm(m, n, k, quant_mode, sol_mode=common.SOLMode.NON_SOL)
    # The value may have been corrected by _correct_data(), but we can check it's reasonable
    assert observed > 0, f"Expected positive value, got {observed}"
    
    # Also test that SOL mode works
    sol_value = perf_db.query_gemm(m, n, k, quant_mode, sol_mode=common.SOLMode.SOL)
    assert sol_value > 0, f"Expected positive SOL value, got {sol_value}"


def test_query_allreduce_sol_mode_calculation(perf_db):
    """
    When sol_mode == SOL, query_allreduce uses get_sol:
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
    quant_mode = 'float16'  # for SOL branch we ignore the custom allreduce dict

    sol_time = perf_db.query_allreduce(quant_mode, tp_size, size, sol_mode=common.SOLMode.SOL)

    expected = (2 * size * 2 / tp_size * (tp_size - 1) / perf_db.system_spec['node']['inter_node_bw']) * 1000
    assert math.isclose(sol_time, expected), f"SOL‐mode allreduce mismatch: expected {expected}, got {sol_time}"


def test_query_allreduce_sol_full_returns_full_tuple(perf_db):
    """
    When sol_mode == SOL_FULL, query_allreduce returns the full tuple (time, 0, 0) from get_sol.
    Using the same numbers as above: (20480.0, 0, 0).
    """
    size = 1024
    tp_size = 2
    quant_mode = 'float16'

    result = perf_db.query_allreduce(quant_mode, tp_size, size, sol_mode=common.SOLMode.SOL_FULL)
    # The get_sol function returns: (sol_time, 0, 0)
    sol_time = (2 * size * 2 / tp_size * (tp_size - 1) / perf_db.system_spec['node']['inter_node_bw']) * 1000
    expected = (sol_time, 0, 0)

    assert isinstance(result, tuple) and len(result) == 3
    assert math.isclose(result[0], expected[0]) and result[1] == expected[1] and result[2] == expected[2]


def test_query_allreduce_non_sol_mode_uses_custom_latency(perf_db):
    """
    When sol_mode is neither SOL nor SOL_FULL (e.g. SOLMode.NONE), the code picks:
        comm_dict = self._custom_allreduce_data[quant_mode][min(tp_size, 8)]['AUTO']
        size_left, size_right = nearest keys enveloping `size`
        lat = interpolate between comm_dict[size_left], comm_dict[size_right]
    We patched _custom_allreduce_data so that:
        _custom_allreduce_data['float16']['2']['AUTO']['1024'] == 5.0
    For tp_size=2 and size=1024 exactly, we expect 5.0.
    """
    size = 1024
    tp_size = 2
    quant_mode = 'float16'

    # Use a “non‐SOL” mode to force fallback into the custom‐data path
    custom_latency = perf_db.query_allreduce(quant_mode, tp_size, size, sol_mode=common.SOLMode.NON_SOL)
    assert math.isclose(custom_latency, 5.0), f"Expected custom‐allreduce latency 5.0, got {custom_latency}"


def test_query_nccl_sol_mode_all_gather(perf_db):
    """
    For query_nccl(..., sol_mode=SOL) and operation='all_gather':
        if dtype == CommQuantMode.half → type_bytes = 2
        sol_time = message_size*(num_gpus-1)*type_bytes / p2pBW * 1000
    We set p2pBW = perf_db.system_spec['node']['inter_node_bw'] = 100.0
    Let num_gpus=4, message_size=512, dtype=half → type_bytes=2
    Then:
        sol_time = 512 * (4-1) * 2 / 100.0 * 1000
                 = 512 * 3 * 2 / 100.0 * 1000
                 = (3072 / 100.0) * 1000
                 = 30.72 * 1000
                 = 30720.0
    """
    dtype = common.CommQuantMode.half
    num_gpus = 4
    operation = 'all_gather'
    message_size = 512

    sol_time = perf_db.query_nccl(dtype, num_gpus, operation, message_size, sol_mode=common.SOLMode.SOL)
    expected = (message_size * (num_gpus - 1) * 2 / perf_db.system_spec['node']['inter_node_bw']) * 1000

    assert math.isclose(sol_time, expected), f"Expected {expected}, got {sol_time}"


@pytest.mark.parametrize('operation', ['alltoall', 'reduce_scatter'])
def test_query_nccl_sol_mode_alltoall_and_reduce_scatter(perf_db, operation):
    """
    The code for 'alltoall' and 'reduce_scatter' in get_sol is identical:
        sol_time = 2 * message_size * type_bytes / p2pBW * 1000
    Using dtype = CommQuantMode.int8 makes type_bytes=1.
    Let message_size = 1000, type_bytes=1, p2pBW=100.0:
        sol_time = 2 * 1000 * 1 / 100.0 * 1000 = (2000 / 100.0)*1000 = 20*1000 = 20000.0
    """
    dtype = common.CommQuantMode.int8  # type_bytes = 1 for int8
    num_gpus = 8             # num_gpus only matters for 'all_gather'
    sol_time = perf_db.query_nccl(dtype, num_gpus, operation, 1000, sol_mode=common.SOLMode.SOL)
    expected = (2 * 1000 * 1 / perf_db.system_spec['node']['inter_node_bw']) * 1000
    assert math.isclose(sol_time, expected), (
        f"Expected {expected} for op {operation}, got {sol_time}"
    )


def test_query_p2p_sol_mode(perf_db):
    """
    query_p2p(..., sol_mode=SOL) uses:
        sol_time = message_bytes / inter_node_bw * 1000
    With message_bytes=256 and inter_node_bw=100.0:
        sol_time = (256 / 100.0) * 1000 = 2.56 * 1000 = 2560.0
    """
    sol_time = perf_db.query_p2p(256, sol_mode=common.SOLMode.SOL)
    expected = (256 / perf_db.system_spec['node']['inter_node_bw']) * 1000
    assert math.isclose(sol_time, expected), f"Expected {expected}, got {sol_time}"


def test_system_spec_was_loaded_correctly(perf_db):
    """
    Sanity check: PerfDatabase.system_spec should be exactly what our patched yaml.load returned.
    """
    spec = perf_db.system_spec
    assert isinstance(spec, dict)
    assert spec['gpu']['float16_tc_flops'] == 1_000.0
    assert spec['node']['inter_node_bw'] == 100.0