# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aiconfigurator.sdk import common
from aiconfigurator.sdk import operations as ops
from aiconfigurator.sdk.performance_result import PerformanceResult

pytestmark = pytest.mark.unit


def test_trtllm_supported_quant_modes_include_fp8_static(comprehensive_perf_db):
    modes = comprehensive_perf_db.supported_quant_mode["gemm"]
    assert common.GEMMQuantMode.fp8.name in modes
    assert common.GEMMQuantMode.fp8_static.name in modes
    assert modes.count(common.GEMMQuantMode.fp8_static.name) == 1


def test_query_gemm_fp8_static_reuses_fp8_table(comprehensive_perf_db):
    m, n, k = 32, 256, 512
    fp8_result = comprehensive_perf_db.query_gemm(m, n, k, common.GEMMQuantMode.fp8)
    static_result = comprehensive_perf_db.query_gemm(m, n, k, common.GEMMQuantMode.fp8_static)

    assert float(static_result) == pytest.approx(float(fp8_result))
    assert static_result.energy == pytest.approx(fp8_result.energy)


def test_query_compute_scale_fp8_static_reuses_fp8_table(comprehensive_perf_db):
    # Provide enough points for 2D interpolation (>=2 keys in each axis).
    comprehensive_perf_db._compute_scale_data = {
        common.GEMMQuantMode.fp8: {
            64: {
                256: {"latency": 1.0, "energy": 10.0},
                512: {"latency": 2.0, "energy": 20.0},
            },
            128: {
                256: {"latency": 1.5, "energy": 15.0},
                512: {"latency": 2.5, "energy": 25.0},
            },
        }
    }

    # Query an interior point so we avoid any boundary corner cases.
    m, k = 96, 384
    fp8_result = comprehensive_perf_db.query_compute_scale(m, k, common.GEMMQuantMode.fp8)
    static_result = comprehensive_perf_db.query_compute_scale(m, k, common.GEMMQuantMode.fp8_static)

    assert float(static_result) == pytest.approx(float(fp8_result))
    assert static_result.energy == pytest.approx(fp8_result.energy)

    # Out-of-range m should be clamped to the table range (avoid hard failure in SILICON mode).
    clamped = comprehensive_perf_db.query_compute_scale(10_000, k, common.GEMMQuantMode.fp8_static)
    fp8_max_m = comprehensive_perf_db.query_compute_scale(128, k, common.GEMMQuantMode.fp8)
    assert float(clamped) == pytest.approx(float(fp8_max_m))
    assert clamped.energy == pytest.approx(fp8_max_m.energy)


def test_query_scale_matrix_fp8_static_reuses_fp8_table(comprehensive_perf_db):
    comprehensive_perf_db._scale_matrix_data = {
        common.GEMMQuantMode.fp8: {
            64: {
                256: {"latency": 3.0, "energy": 30.0},
                512: {"latency": 4.0, "energy": 40.0},
            },
            128: {
                256: {"latency": 3.5, "energy": 35.0},
                512: {"latency": 4.5, "energy": 45.0},
            },
        }
    }

    m, k = 96, 384
    fp8_result = comprehensive_perf_db.query_scale_matrix(m, k, common.GEMMQuantMode.fp8)
    static_result = comprehensive_perf_db.query_scale_matrix(m, k, common.GEMMQuantMode.fp8_static)

    assert float(static_result) == pytest.approx(float(fp8_result))
    assert static_result.energy == pytest.approx(fp8_result.energy)

    # Out-of-range m should be clamped to the table range (avoid hard failure in SILICON mode).
    clamped = comprehensive_perf_db.query_scale_matrix(10_000, k, common.GEMMQuantMode.fp8_static)
    fp8_max_m = comprehensive_perf_db.query_scale_matrix(128, k, common.GEMMQuantMode.fp8)
    assert float(clamped) == pytest.approx(float(fp8_max_m))
    assert clamped.energy == pytest.approx(fp8_max_m.energy)


def test_gemm_query_subtracts_overheads_only_for_fp8_static_and_qwen_proj_fc2():
    class FakeDatabase:
        def __init__(self):
            self.calls: list[tuple[str, common.GEMMQuantMode]] = []

        def query_gemm(self, m, n, k, quant_mode, database_mode=None):
            self.calls.append(("gemm", quant_mode))
            return PerformanceResult(10.0, energy=100.0)

        def query_compute_scale(self, m, k, quant_mode, database_mode=None):
            self.calls.append(("compute_scale", quant_mode))
            return PerformanceResult(1.0, energy=10.0)

        def query_scale_matrix(self, m, k, quant_mode, database_mode=None):
            self.calls.append(("scale_matrix", quant_mode))
            return PerformanceResult(2.0, energy=20.0)

    db = FakeDatabase()

    # Qwen proj GEMM: subtract compute_scale + scale_matrix
    op = ops.GEMM(
        "context_proj_gemm",
        1.0,
        n=128,
        k=256,
        quant_mode=common.GEMMQuantMode.fp8_static,
        low_precision_input=True,
    )
    result = op.query(db, x=64, model_name="Qwen/Qwen3-32B")
    assert float(result) == pytest.approx(7.0)
    assert result.energy == pytest.approx(70.0)
    assert db.calls == [
        ("gemm", common.GEMMQuantMode.fp8_static),
        ("compute_scale", common.GEMMQuantMode.fp8_static),
        ("scale_matrix", common.GEMMQuantMode.fp8_static),
    ]

    # Non-proj GEMM: subtract compute_scale only (no scale_matrix)
    db2 = FakeDatabase()
    op2 = ops.GEMM(
        "context_q_b_proj_gemm",
        1.0,
        n=128,
        k=256,
        quant_mode=common.GEMMQuantMode.fp8_static,
    )
    result2 = op2.query(db2, x=64, model_name="Qwen/Qwen3-32B")
    assert float(result2) == pytest.approx(9.0)
    assert result2.energy == pytest.approx(90.0)
    assert ("scale_matrix", common.GEMMQuantMode.fp8_static) not in db2.calls

    # fp8 (non-static): no overhead subtraction
    db3 = FakeDatabase()
    op3 = ops.GEMM(
        "context_proj_gemm",
        1.0,
        n=128,
        k=256,
        quant_mode=common.GEMMQuantMode.fp8,
    )
    result3 = op3.query(db3, x=64, model_name="Qwen/Qwen3-32B")
    assert float(result3) == pytest.approx(10.0)
    assert result3.energy == pytest.approx(100.0)
    assert db3.calls == [("gemm", common.GEMMQuantMode.fp8)]
