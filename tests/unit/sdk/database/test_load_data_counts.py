# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sanity assertions for the lazy-load instrumentation counter.

The class-level cache + idempotent ``load_data`` design (lazy per-op data ownership) means
each op class loads its CSV-backed data exactly once per ``(systems_root,
system, backend, version, enable_shared_layer)`` tuple regardless of how
many queries fire after that. Demonstrated here for GEMM; later issues
extend this file as more ops migrate.
"""

from __future__ import annotations

from aiconfigurator.sdk.operations.base import Operation
from aiconfigurator.sdk.operations.gemm import GEMM


def test_perf_database_init_opens_no_csvs(tmp_path, monkeypatch):
    """The headline guarantee of lazy per-op data ownership:
    ``PerfDatabase()`` triggers no ``load_data`` calls. Op data is
    loaded only when the matrix is read or a query fires.

    Regression: the previous implementation warmed all 22 ops eagerly in
    ``__init__`` to accommodate an old test-fixture loader-patch
    pattern. Test fixtures now own the warm-up explicitly, so production
    code can be fully lazy."""
    import yaml

    from aiconfigurator.sdk.operations.base import Operation
    from aiconfigurator.sdk.perf_database import PerfDatabase

    # Patch yaml + every loader so ``__init__`` doesn't try to read real CSVs
    # (it shouldn't try anyway — that's what this test asserts).
    monkeypatch.setattr(
        yaml,
        "load",
        lambda stream, Loader=None: {  # noqa: N803
            "data_dir": "data",
            "misc": {"nccl_version": "v1"},
            "gpu": {"bfloat16_tc_flops": 1.0, "mem_bw": 1.0, "mem_empirical_constant_latency": 1.0},
            "node": {"inter_node_bw": 1.0, "intra_node_bw": 1.0, "num_gpus_per_node": 8, "p2p_latency": 1.0},
        },
    )

    yaml_file = tmp_path / "any_system.yaml"
    yaml_file.write_text("dummy: data")  # content irrelevant — yaml.load is patched

    Operation._load_data_call_count.clear()
    PerfDatabase("any_system", "any_backend", "v1", str(tmp_path))
    assert dict(Operation._load_data_call_count) == {}, (
        "PerfDatabase.__init__ must not trigger any OpClass.load_data — "
        "lazy-first-query is the headline contract of the operations package"
    )


# ``test_gemm_loads_exactly_once_across_many_queries`` retired with #1357
# PR-5: the per-call query shims route through the compiled engine (which
# loads its own tables from disk) and never touch ``OpClass.load_data``, so
# "many queries -> one Python load" is no longer a live relationship. The
# load-once idempotence contract itself is pinned directly below.


def test_gemm_load_count_unaffected_by_repeated_load_data_calls(stub_perf_db):
    """Calling ``GEMM.load_data`` directly multiple times must not increment
    the counter beyond the initial load."""
    initial_count = Operation._load_data_call_count.get(GEMM, 0)
    for _ in range(5):
        GEMM.load_data(stub_perf_db)
    assert Operation._load_data_call_count.get(GEMM, 0) == initial_count


def test_clear_all_op_caches_resets_counter_and_class_cache():
    """``clear_all_op_caches`` clears both the data cache and the counter,
    so a fresh load is needed afterward. Production webapps can use this
    as a manual eviction lever.

    The save/restore around the call keeps the comprehensive_perf_db
    singleton's cache entries alive for sibling tests — clearing them
    would force fresh disk loads with no loader patches active. (The
    util-grid cache this test also seeded retired with the estimation math
    in #1357 PR-5; ``clear_grid_cache`` survives as a compat no-op inside
    ``clear_all_op_caches``.)"""
    from aiconfigurator.sdk.operations.base import Operation as _OpBase
    from aiconfigurator.sdk.operations.base import _all_operation_subclasses, clear_all_op_caches

    # Snapshot every class-level dict cache on every Operation subclass so
    # ``clear_all_op_caches()`` doesn't permanently evict the
    # comprehensive_perf_db singleton's entries for sibling tests.
    saved: list[tuple[type, str, dict]] = []
    for cls in _all_operation_subclasses(_OpBase):
        for attr_name in list(cls.__dict__):
            if attr_name.endswith("_cache") and isinstance(cls.__dict__[attr_name], dict):
                saved.append((cls, attr_name, dict(cls.__dict__[attr_name])))

    try:
        sentinel_key = ("/dev/null", "test", "test", "test", False)
        GEMM._data_cache[sentinel_key] = "sentinel"
        Operation._load_data_call_count[GEMM] = 99

        # Seed the other two GEMM caches too so we can verify the contract
        # clears all three per-class caches, not just ``_data_cache``.
        GEMM._compute_scale_cache[sentinel_key] = "compute-scale-sentinel"
        GEMM._scale_matrix_cache[sentinel_key] = "scale-matrix-sentinel"

        clear_all_op_caches()

        assert sentinel_key not in GEMM._data_cache
        assert sentinel_key not in GEMM._compute_scale_cache
        assert sentinel_key not in GEMM._scale_matrix_cache
        assert Operation._load_data_call_count.get(GEMM, 0) == 0
    finally:
        for cls, attr_name, contents in saved:
            getattr(cls, attr_name).update(contents)
