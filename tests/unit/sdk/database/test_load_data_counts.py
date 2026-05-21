# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sanity assertions for the lazy-load instrumentation counter.

The class-level cache + idempotent ``load_data`` design (Pattern A) means
each op class loads its CSV-backed data exactly once per ``(systems_root,
system, backend, version, enable_shared_layer)`` tuple regardless of how
many queries fire after that. Demonstrated here for GEMM; later issues
extend this file as more ops migrate.
"""

from __future__ import annotations

from aiconfigurator.sdk import common
from aiconfigurator.sdk.operations.base import Operation
from aiconfigurator.sdk.operations.gemm import GEMM


def test_gemm_loads_exactly_once_across_many_queries(stub_perf_db):
    """A single GEMM.load_data call covers an arbitrary number of queries."""
    # The autouse ``_reset_op_load_counts`` fixture cleared the counter
    # before this test ran. ``stub_perf_db`` already triggered one eager
    # load during construction.
    initial_count = Operation._load_data_call_count[GEMM]
    assert initial_count <= 1, "GEMM should load at most once during db construction"

    quant_mode = common.GEMMQuantMode.bfloat16
    for m, n, k in [(64, 128, 256), (64, 128, 512), (64, 256, 256), (128, 128, 256), (128, 256, 512)]:
        stub_perf_db.query_gemm(m, n, k, quant_mode)

    # Many queries → still loaded exactly once.
    assert Operation._load_data_call_count[GEMM] == initial_count


def test_gemm_load_count_unaffected_by_repeated_load_data_calls(stub_perf_db):
    """Calling ``GEMM.load_data`` directly multiple times must not increment
    the counter beyond the initial load."""
    initial_count = Operation._load_data_call_count[GEMM]
    for _ in range(5):
        GEMM.load_data(stub_perf_db)
    assert Operation._load_data_call_count[GEMM] == initial_count


def test_clear_all_op_caches_resets_counter_and_class_cache():
    """``clear_all_op_caches`` clears both the data cache and the counter,
    so a fresh load is needed afterward. Production webapps can use this
    as a manual eviction lever.

    The save/restore around the call keeps the comprehensive_perf_db
    singleton's cache entry alive for sibling tests — clearing it would
    force a fresh disk load with no loader patches active."""
    from aiconfigurator.sdk.operations.base import clear_all_op_caches

    saved = {
        "data": dict(GEMM._data_cache),
        "compute_scale": dict(GEMM._compute_scale_cache),
        "scale_matrix": dict(GEMM._scale_matrix_cache),
    }
    try:
        sentinel_key = ("/dev/null", "test", "test", "test", False)
        GEMM._data_cache[sentinel_key] = "sentinel"
        Operation._load_data_call_count[GEMM] = 99

        clear_all_op_caches()

        assert sentinel_key not in GEMM._data_cache
        assert Operation._load_data_call_count[GEMM] == 0
    finally:
        GEMM._data_cache.update(saved["data"])
        GEMM._compute_scale_cache.update(saved["compute_scale"])
        GEMM._scale_matrix_cache.update(saved["scale_matrix"])
