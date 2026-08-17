# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Consistency anchors for the sanity-check engine reference (PR-4/#1357).

``tools/sanity_check/engine_reference.EngineReference`` sources the
validation notebook's per-op reference values through the compiled Rust
engine's ad-hoc op-list FFI. Since PR-5 retired the Python per-call math,
the ``PerfDatabase.query_*`` expected side of these tests is itself an
engine-routed deprecation shim — so what this suite pins today is that the
SHIM's op construction and the reference's op construction agree exactly
(including typed data-miss errors), across full quant grids on two shipped
systems. The against-the-legacy-math anchoring moved to the pinned capture
in ``tests/cross_package/test_query_shim_baseline.py``; this file retires
together with the shims (the deprecation-cleanup PR).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

from aiconfigurator.sdk import common
from aiconfigurator.sdk.common import DatabaseMode
from aiconfigurator.sdk.perf_database import (
    PerfDataNotAvailableError,
    get_database,
    get_latest_database_version,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools" / "sanity_check"))
from engine_reference import EngineReference

pytestmark = pytest.mark.unit

# One Hopper and one Blackwell combo: together they cover every re-oracled
# family plus the fp8_static gemm op model (collected on b200 trtllm).
COMBOS = [("h200_sxm", "trtllm"), ("b200_sxm", "trtllm")]


@pytest.fixture(scope="module", params=COMBOS, ids=lambda c: f"{c[0]}-{c[1]}")
def db_and_reference(request):
    system, backend = request.param
    version = get_latest_database_version(system, backend)
    if version is None:
        pytest.skip(f"no database for {system}/{backend}")
    database = get_database(system=system, backend=backend, version=version)
    if database is None:
        pytest.skip(f"database load failed for {system}/{backend}/{version}")
    return database, EngineReference(database)


def _assert_matches(py_value, engine_value):
    """SILICON facade values act as floats; SOL_FULL values are raw triples."""
    if isinstance(py_value, tuple):
        assert isinstance(engine_value, tuple) and len(engine_value) == 3
        for expected, got in zip(py_value, engine_value, strict=True):
            assert math.isclose(expected, got, rel_tol=1e-9, abs_tol=1e-12), (py_value, engine_value)
    else:
        assert math.isclose(float(py_value), float(engine_value), rel_tol=1e-9, abs_tol=1e-12), (
            float(py_value),
            engine_value,
        )


def _check_both_modes(query_facade, query_engine, **kwargs):
    """Compare SILICON and SOL_FULL. A facade data miss must reproduce as the
    SAME typed error on the engine side (probe-and-skip keeps working)."""
    checked = 0
    for mode in (DatabaseMode.SILICON, DatabaseMode.SOL_FULL):
        try:
            expected = query_facade(**kwargs, database_mode=mode)
        except (PerfDataNotAvailableError, ValueError) as exc:
            with pytest.raises(type(exc)):
                query_engine(**kwargs, database_mode=mode)
            continue
        _assert_matches(expected, query_engine(**kwargs, database_mode=mode))
        checked += 1
    return checked


def test_gemm_matches_per_call_facade(db_and_reference):
    database, reference = db_and_reference
    from aiconfigurator.sdk.operations import GEMM

    GEMM.load_data(database)
    checked = 0
    for quant_mode in database._gemm_data:
        for m, n, k in ((7, 4096, 4096), (256, 8192, 1024), (4096, 1024, 8192)):
            checked += _check_both_modes(
                database.query_gemm, reference.query_gemm, m=m, n=n, k=k, quant_mode=quant_mode
            )
    assert checked


def test_context_attention_matches_per_call_facade(db_and_reference):
    """Since PR-5 the facade shim serves the same op-level estimate (table +
    fused rope/kv-write extras) the reference charts, so both modes compare
    directly. The extras-vs-raw-table decomposition identity this test used
    to assert is anchored historically by the pinned pre-retirement baseline
    (``expected_from="engine"`` ctx-attn cases)."""
    database, reference = db_and_reference
    from aiconfigurator.sdk.operations import ContextAttention

    ContextAttention.load_data(database)
    checked = 0
    for fmha_quant_mode in database._context_attention_data:
        for kvcache_quant_mode in database._context_attention_data[fmha_quant_mode]:
            for s, prefix in ((2048, 0), (1434, 614)):
                checked += _check_both_modes(
                    database.query_context_attention,
                    reference.query_context_attention,
                    b=1,
                    s=s,
                    prefix=prefix,
                    n=32,
                    n_kv=8,
                    kvcache_quant_mode=kvcache_quant_mode,
                    fmha_quant_mode=fmha_quant_mode,
                )
    assert checked


def test_generation_attention_matches_per_call_facade(db_and_reference):
    database, reference = db_and_reference
    from aiconfigurator.sdk.operations import GenerationAttention

    GenerationAttention.load_data(database)
    checked = 0
    for kvcache_quant_mode in database._generation_attention_data:
        for b, s in ((64, 4096), (1, 128)):
            checked += _check_both_modes(
                database.query_generation_attention,
                reference.query_generation_attention,
                b=b,
                s=s,
                n=32,
                n_kv=4,
                kvcache_quant_mode=kvcache_quant_mode,
            )
    assert checked


def test_mla_matches_per_call_facade(db_and_reference):
    database, reference = db_and_reference
    from aiconfigurator.sdk.operations import ContextMLA, GenerationMLA

    ContextMLA.load_data(database)
    GenerationMLA.load_data(database)
    checked = 0
    for fmha_quant_mode in database._context_mla_data:
        for kvcache_quant_mode in database._context_mla_data[fmha_quant_mode]:
            checked += _check_both_modes(
                database.query_context_mla,
                reference.query_context_mla,
                b=8,
                s=1434,
                prefix=614,
                num_heads=16,
                kvcache_quant_mode=kvcache_quant_mode,
                fmha_quant_mode=fmha_quant_mode,
            )
    for kvcache_quant_mode in database._generation_mla_data:
        checked += _check_both_modes(
            database.query_generation_mla,
            reference.query_generation_mla,
            b=64,
            s=4096,
            num_heads=16,
            kvcache_quant_mode=kvcache_quant_mode,
        )
    assert checked


def test_dsa_module_matches_per_call_facade(db_and_reference):
    database, reference = db_and_reference
    checked = 0
    checked += _check_both_modes(
        database.query_context_dsa_module,
        reference.query_context_dsa_module,
        b=1,
        s=4096,
        prefix=0,
        num_heads=128,
        kvcache_quant_mode=common.KVCacheQuantMode.bfloat16,
        fmha_quant_mode=common.FMHAQuantMode.bfloat16,
    )
    checked += _check_both_modes(
        database.query_generation_dsa_module,
        reference.query_generation_dsa_module,
        b=1,
        s=8192,
        num_heads=128,
        kv_cache_dtype=common.KVCacheQuantMode.bfloat16,
    )
    assert checked


def test_moe_matches_per_call_facade(db_and_reference):
    database, reference = db_and_reference
    from aiconfigurator.sdk.operations import MoE

    MoE.load_data(database)
    checked = 0
    for quant_mode in database._moe_data:
        workload_distribution = next(iter(database._moe_data[quant_mode].keys()))
        for num_tokens in (7, 512, 16384):
            checked += _check_both_modes(
                database.query_moe,
                reference.query_moe,
                num_tokens=num_tokens,
                hidden_size=7168,
                inter_size=2048,
                topk=8,
                num_experts=256,
                moe_tp_size=1,
                moe_ep_size=8,
                quant_mode=quant_mode,
                workload_distribution=workload_distribution,
            )
    assert checked


def test_comm_matches_per_call_facade(db_and_reference):
    database, reference = db_and_reference
    checked = 0
    for tp_size in (2, 8):
        # message sizes chosen to exercise the u32 factorization (2**32 needs
        # the per-token split) alongside ordinary sizes.
        for size in (2**10, 2**24, 2**32):
            checked += _check_both_modes(
                database.query_custom_allreduce,
                reference.query_custom_allreduce,
                quant_mode=common.CommQuantMode.half,
                tp_size=tp_size,
                size=size,
            )
    for operation in ("all_gather", "all_reduce", "alltoall", "reduce_scatter"):
        for size in (2**10, 2**20, 2**30):
            checked += _check_both_modes(
                lambda database_mode, **kw: database.query_nccl(
                    kw["quant_mode"], kw["num_gpus"], kw["operation"], kw["size"], database_mode=database_mode
                ),
                reference.query_nccl,
                quant_mode=common.CommQuantMode.half,
                num_gpus=16,
                operation=operation,
                size=size,
            )
    assert checked
