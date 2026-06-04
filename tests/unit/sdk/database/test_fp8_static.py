# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aiconfigurator.sdk import common
from aiconfigurator.sdk import operations as ops
from aiconfigurator.sdk.perf_database import LoadedOpData, PerfDataNotAvailableError
from aiconfigurator.sdk.performance_result import PerformanceResult

pytestmark = pytest.mark.unit


def _gemm_loaded_data(*quant_modes: common.GEMMQuantMode) -> LoadedOpData:
    return LoadedOpData(
        {
            quant_mode: {
                32: {
                    256: {
                        512: {"latency": 1.0, "energy": 10.0},
                    }
                }
            }
            for quant_mode in quant_modes
        },
        common.PerfDataFilename.gemm,
        "dummy_path",
    )


def _overhead_loaded_data(filename: common.PerfDataFilename) -> LoadedOpData:
    return LoadedOpData(
        {
            common.GEMMQuantMode.fp8: {
                32: {
                    512: {"latency": 1.0, "energy": 10.0},
                }
            }
        },
        filename,
        "dummy_path",
    )


@pytest.mark.parametrize("backend", [common.BackendName.trtllm.value, common.BackendName.vllm.value])
def test_supported_quant_modes_include_fp8_static_only_when_base_and_overhead_data_exist(
    mutable_comprehensive_perf_db, backend
):
    db = mutable_comprehensive_perf_db
    db.backend = backend
    db._gemm_data = _gemm_loaded_data(common.GEMMQuantMode.fp8)
    db._update_support_matrix()

    modes = db.supported_quant_mode["gemm"]
    assert common.GEMMQuantMode.fp8.name in modes
    assert common.GEMMQuantMode.fp8_static.name not in modes

    db._gemm_data = _gemm_loaded_data(common.GEMMQuantMode.fp8_static)
    db._update_support_matrix()
    modes = db.supported_quant_mode["gemm"]
    assert common.GEMMQuantMode.fp8_static.name not in modes

    db._gemm_data = _gemm_loaded_data(common.GEMMQuantMode.fp8)
    db._compute_scale_data = _overhead_loaded_data(common.PerfDataFilename.compute_scale)
    db._scale_matrix_data = _overhead_loaded_data(common.PerfDataFilename.scale_matrix)
    db._update_support_matrix()

    modes = db.supported_quant_mode["gemm"]
    assert common.GEMMQuantMode.fp8.name in modes
    assert common.GEMMQuantMode.fp8_static.name in modes
    assert modes.count(common.GEMMQuantMode.fp8_static.name) == 1


def test_sglang_supported_quant_modes_include_fp8_static_only_when_base_and_overhead_data_exist(
    mutable_comprehensive_perf_db,
):
    db = mutable_comprehensive_perf_db
    db.backend = common.BackendName.sglang.value
    db._gemm_data = LoadedOpData(
        {
            common.GEMMQuantMode.fp8: {
                32: {
                    256: {
                        512: {"latency": 1.0, "energy": 10.0},
                    }
                }
            }
        },
        common.PerfDataFilename.gemm,
        "dummy_path",
    )
    db._update_support_matrix()

    modes = db.supported_quant_mode["gemm"]
    assert common.GEMMQuantMode.fp8.name in modes
    assert common.GEMMQuantMode.fp8_static.name not in modes

    db._gemm_data = _gemm_loaded_data(common.GEMMQuantMode.fp8_static)
    db._update_support_matrix()
    modes = db.supported_quant_mode["gemm"]
    assert common.GEMMQuantMode.fp8_static.name not in modes

    db._gemm_data = _gemm_loaded_data(common.GEMMQuantMode.fp8)
    db._compute_scale_data = _overhead_loaded_data(common.PerfDataFilename.compute_scale)
    db._scale_matrix_data = _overhead_loaded_data(common.PerfDataFilename.scale_matrix)
    db._update_support_matrix()

    modes = db.supported_quant_mode["gemm"]
    assert common.GEMMQuantMode.fp8.name in modes
    assert common.GEMMQuantMode.fp8_static.name in modes
    assert modes.count(common.GEMMQuantMode.fp8_static.name) == 1


@pytest.mark.parametrize(
    "backend",
    [common.BackendName.trtllm.value, common.BackendName.vllm.value, common.BackendName.sglang.value],
)
def test_query_gemm_fp8_static_uses_dynamic_fp8_table(mutable_comprehensive_perf_db, backend):
    db = mutable_comprehensive_perf_db
    db.backend = backend
    db._gemm_data = LoadedOpData(
        {
            common.GEMMQuantMode.fp8: {
                32: {
                    256: {
                        512: {"latency": 1.0, "energy": 10.0},
                    }
                }
            },
            common.GEMMQuantMode.fp8_static: {
                32: {
                    256: {
                        512: {"latency": 2.0, "energy": 20.0},
                    }
                }
            },
        },
        common.PerfDataFilename.gemm,
        "dummy_path",
    )
    db.query_gemm.cache_clear()

    result = db.query_gemm(
        32,
        256,
        512,
        common.GEMMQuantMode.fp8_static,
        database_mode=common.DatabaseMode.SILICON,
    )

    assert float(result) == pytest.approx(1.0)
    assert result.energy == pytest.approx(10.0)


@pytest.mark.parametrize(
    "backend",
    [common.BackendName.trtllm.value, common.BackendName.vllm.value, common.BackendName.sglang.value],
)
def test_query_gemm_fp8_static_requires_dynamic_fp8_table(mutable_comprehensive_perf_db, backend):
    db = mutable_comprehensive_perf_db
    db.backend = backend
    db._gemm_data = LoadedOpData(
        {
            common.GEMMQuantMode.fp8_static: {
                33: {
                    272: {
                        544: {"latency": 1.0, "energy": 10.0},
                    }
                }
            },
        },
        common.PerfDataFilename.gemm,
        "dummy_path",
    )
    db.query_gemm.cache_clear()

    with pytest.raises(PerfDataNotAvailableError, match="fp8_static"):
        db.query_gemm(
            33,
            272,
            544,
            common.GEMMQuantMode.fp8_static,
            database_mode=common.DatabaseMode.SILICON,
        )


def test_query_gemm_fp8_static_sparse_shape_miss_is_structured(mutable_comprehensive_perf_db):
    db = mutable_comprehensive_perf_db
    db.backend = common.BackendName.sglang.value
    db._gemm_data = LoadedOpData(
        {
            common.GEMMQuantMode.fp8: {
                1: {
                    4096: {
                        5120: {"latency": 1.0, "energy": 10.0},
                    }
                }
            },
        },
        common.PerfDataFilename.gemm,
        "dummy_path",
    )
    db.query_gemm.cache_clear()

    with pytest.raises(PerfDataNotAvailableError, match=r"GEMM perf data not available.*fp8_static"):
        db.query_gemm(
            1,
            4096,
            4096,
            common.GEMMQuantMode.fp8_static,
            database_mode=common.DatabaseMode.SILICON,
        )


def test_query_compute_scale_fp8_static_reuses_fp8_table(mutable_comprehensive_perf_db):
    db = mutable_comprehensive_perf_db
    # Provide enough points for 2D interpolation (>=2 keys in each axis).
    compute_scale_data_dict = {
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
    db._compute_scale_data = LoadedOpData(compute_scale_data_dict, common.PerfDataFilename.compute_scale, "dummy_path")

    # Query an interior point so we avoid any boundary corner cases.
    m, k = 96, 384
    fp8_result = db.query_compute_scale(m, k, common.GEMMQuantMode.fp8)
    static_result = db.query_compute_scale(m, k, common.GEMMQuantMode.fp8_static)

    assert float(static_result) == pytest.approx(float(fp8_result))
    assert static_result.energy == pytest.approx(fp8_result.energy)

    # Out-of-range m should be clamped to the table range (avoid hard failure in SILICON mode).
    clamped = db.query_compute_scale(10_000, k, common.GEMMQuantMode.fp8_static)
    fp8_max_m = db.query_compute_scale(128, k, common.GEMMQuantMode.fp8)
    assert float(clamped) == pytest.approx(float(fp8_max_m))
    assert clamped.energy == pytest.approx(fp8_max_m.energy)


def test_query_scale_matrix_fp8_static_reuses_fp8_table(mutable_comprehensive_perf_db):
    db = mutable_comprehensive_perf_db
    scale_matrix_data_dict = {
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
    db._scale_matrix_data = LoadedOpData(scale_matrix_data_dict, common.PerfDataFilename.scale_matrix, "dummy_path")

    m, k = 96, 384
    fp8_result = db.query_scale_matrix(m, k, common.GEMMQuantMode.fp8)
    static_result = db.query_scale_matrix(m, k, common.GEMMQuantMode.fp8_static)

    assert float(static_result) == pytest.approx(float(fp8_result))
    assert static_result.energy == pytest.approx(fp8_result.energy)

    # Out-of-range m should be clamped to the table range (avoid hard failure in SILICON mode).
    clamped = db.query_scale_matrix(10_000, k, common.GEMMQuantMode.fp8_static)
    fp8_max_m = db.query_scale_matrix(128, k, common.GEMMQuantMode.fp8)
    assert float(clamped) == pytest.approx(float(fp8_max_m))
    assert clamped.energy == pytest.approx(fp8_max_m.energy)


def test_gemm_query_subtracts_overheads_for_fp8_static():
    class FakeDatabase:
        def __init__(self):
            self.backend = common.BackendName.sglang.value
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
    assert result.source == "estimated"
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
    assert result2.source == "estimated"
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
    assert result3.source == "silicon"
    assert db3.calls == [("gemm", common.GEMMQuantMode.fp8)]
