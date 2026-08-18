# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import pytest

from aiconfigurator.sdk import common
from aiconfigurator.sdk.perf_database import LoadedOpData

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


# ---------------------------------------------------------------------------
# Retired with #1357 PR-5 (single oracle = the compiled engine):
# - query_gemm fp8_static -> dynamic-fp8 table reuse, the structured miss, and
#   the nearest-site transfer (quant normalization + interpolation internals);
# - query_compute_scale / query_scale_matrix (now tombstoned facades);
# - the GEMM.query fp8_static overhead-subtraction closure and its SOL floor —
#   fp8_static is one of the two approved semantic moves: the engine now
#   prices it through the op model rather than the Python subtraction chain.
# Value behaviour is anchored by tests/cross_package/test_query_shim_baseline.py
# and the frozen parity goldens. The support-matrix admission tests above stay:
# they pin what LOADED data admits fp8_static, which is Python-owned.
# ---------------------------------------------------------------------------
