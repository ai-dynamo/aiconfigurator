# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Alignment anchors for the sanity-check engine re-oracle (PR-4 of #1357).

``tools/sanity_check/engine_reference.EngineReference`` re-sources the
validation notebook's per-op reference values through the compiled Rust
engine's ad-hoc op-list FFI. These tests pin the new surface against the
Python per-call stack — the facade where the op is a 1:1 wrapper (SILICON
latencies AND the SOL_FULL ``(sol_time, sol_math, sol_mem)`` triples), and
the Python op level where the chart deliberately moved to op semantics
(context attention's fused extras; gemm fp8_static's overhead model) — per
family, on real shipped databases. This is the cross-language alignment
evidence PR-5 (the per-call query-stack removal) builds on: when the Python
side goes, the expected side of these tests moves to pinned values or is
retired with it.
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
            if quant_mode == common.GEMMQuantMode.fp8_static:
                # The engine charts the OP model for fp8_static (dynamic-fp8
                # row minus overhead tables); the facade normalized its
                # lookup to the fp8 table. Pin SILICON op-vs-op (both
                # languages' op level — the surface PR-5 deletes) ...
                expected = float(GEMM("gemm", 1.0, n, k, quant_mode).query(database, x=m))
                got = reference.query_gemm(m=m, n=n, k=k, quant_mode=quant_mode, database_mode=DatabaseMode.SILICON)
                _assert_matches(expected, got)
                # ... while the SOL triple still equals the facade's raw
                # roofline exactly: the op's SOL floor always fires (the
                # subtrahends are SOL values taken off that same roofline).
                _assert_matches(
                    database.query_gemm(m=m, n=n, k=k, quant_mode=quant_mode, database_mode=DatabaseMode.SOL_FULL),
                    reference.query_gemm(m=m, n=n, k=k, quant_mode=quant_mode, database_mode=DatabaseMode.SOL_FULL),
                )
                checked += 2
                continue
            checked += _check_both_modes(
                database.query_gemm, reference.query_gemm, m=m, n=n, k=k, quant_mode=quant_mode
            )
    assert checked


def test_context_attention_matches_python_op(db_and_reference):
    """Context attention charts the OP-level estimate (table + fused
    rope/kv-write extras), so the SILICON anchor is op-vs-op. The SOL triple
    is pinned against the facade through the extras decomposition identity:
    the extras are memory-only, so sol_math matches the raw-table facade
    exactly and the SAME delta lands on sol_time and sol_mem."""
    database, reference = db_and_reference
    from aiconfigurator.sdk.operations import ContextAttention

    ContextAttention.load_data(database)
    checked = 0
    for fmha_quant_mode in database._context_attention_data:
        for kvcache_quant_mode in database._context_attention_data[fmha_quant_mode]:
            for s, prefix in ((2048, 0), (1434, 614)):
                op = ContextAttention("context_attention", 1.0, 32, 8, kvcache_quant_mode, fmha_quant_mode)
                kwargs = dict(
                    b=1,
                    s=s,
                    n=32,
                    n_kv=8,
                    kvcache_quant_mode=kvcache_quant_mode,
                    fmha_quant_mode=fmha_quant_mode,
                    prefix=prefix,
                )
                try:
                    expected = float(op.query(database, batch_size=1, s=s, prefix=prefix))
                except (PerfDataNotAvailableError, ValueError) as exc:
                    with pytest.raises(type(exc)):
                        reference.query_context_attention(**kwargs, database_mode=DatabaseMode.SILICON)
                    continue
                _assert_matches(
                    expected,
                    reference.query_context_attention(**kwargs, database_mode=DatabaseMode.SILICON),
                )

                facade_time, facade_math, facade_mem = database.query_context_attention(
                    **kwargs, database_mode=DatabaseMode.SOL_FULL
                )
                sol_time, sol_math, sol_mem = reference.query_context_attention(
                    **kwargs, database_mode=DatabaseMode.SOL_FULL
                )
                assert math.isclose(sol_math, facade_math, rel_tol=1e-9, abs_tol=1e-12)
                extras_on_time = sol_time - facade_time
                extras_on_mem = sol_mem - facade_mem
                assert extras_on_time > 0 and math.isclose(
                    extras_on_time, extras_on_mem, rel_tol=1e-9, abs_tol=1e-12
                ), (extras_on_time, extras_on_mem)
                checked += 1
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
