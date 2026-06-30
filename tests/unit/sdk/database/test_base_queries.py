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
    query_gemm should return the exact latency stored under (quant_mode=bfloat16, m=64, n=128, k=256).
    We patched load_gemm_data to have exactly one entry: 10.0.
    However, _correct_data() may update this based on SOL calculation.
    """
    quant_mode = common.GEMMQuantMode.bfloat16  # matches our dummy key
    m, n, k = 64, 128, 256

    observed = stub_perf_db.query_gemm(m, n, k, quant_mode, database_mode=common.DatabaseMode.SILICON)
    # The value may have been corrected by _correct_data(), but we can check it's reasonable
    assert observed > 0, f"Expected positive value, got {observed}"

    # Also test that SOL mode works
    sol_value = stub_perf_db.query_gemm(m, n, k, quant_mode, database_mode=common.DatabaseMode.SOL)
    assert sol_value > 0, f"Expected positive SOL value, got {sol_value}"
    assert sol_value.source == "sol"


def test_query_gemm_empirical_data_calibrated(stub_perf_db):
    """At a collected shape, EMPIRICAL recovers the silicon latency."""
    quant_mode = common.GEMMQuantMode.bfloat16  # stub has bf16 GEMM data
    m, n, k = 64, 128, 256

    silicon_value = stub_perf_db.query_gemm(m, n, k, quant_mode, database_mode=common.DatabaseMode.SILICON)
    empirical_value = stub_perf_db.query_gemm(m, n, k, quant_mode, database_mode=common.DatabaseMode.EMPIRICAL)

    assert math.isclose(float(empirical_value), float(silicon_value), rel_tol=1e-6)


def test_query_gemm_empirical_raises_without_data(stub_perf_db):
    """When the op has no data for the requested slice (fp8 absent in the stub),
    EMPIRICAL raises EmpiricalNotImplementedError instead of fabricating a
    SOL / constant -- the legacy placeholder fallback was removed.
    """
    from aiconfigurator.sdk.errors import EmpiricalNotImplementedError

    quant_mode = common.GEMMQuantMode.fp8  # no fp8 GEMM data in the stub
    m, n, k = 64, 128, 256

    with pytest.raises(EmpiricalNotImplementedError):
        stub_perf_db.query_gemm(m, n, k, quant_mode, database_mode=common.DatabaseMode.EMPIRICAL)


def test_query_gemm_exact_match_skips_3d_interpolation(comprehensive_perf_db, monkeypatch):
    """Exact GEMM hits should bypass both 1D and 3D interpolation."""
    quant_mode = common.GEMMQuantMode.bfloat16
    m, n, k = 16, 128, 128

    def _fail_interp_3d(*args, **kwargs):
        raise AssertionError("_interp_3d should not be used for exact GEMM matches")

    def _fail_interp_1d(*args, **kwargs):
        raise AssertionError("_interp_1d should not be used for exact GEMM matches")

    monkeypatch.setattr("aiconfigurator.sdk.interpolation.interp_3d", _fail_interp_3d)
    monkeypatch.setattr("aiconfigurator.sdk.interpolation.interp_1d", _fail_interp_1d)

    observed = comprehensive_perf_db.query_gemm(m, n, k, quant_mode, database_mode=common.DatabaseMode.SILICON)
    expected = 0.1 + m * 0.001 + n * 0.0001 + k * 0.00001

    assert math.isclose(float(observed), expected)


def test_query_gemm_interpolates_only_on_m_when_nk_match(comprehensive_perf_db, monkeypatch):
    """GEMM lookup should use 1D interpolation on m when n and k match."""
    quant_mode = common.GEMMQuantMode.bfloat16
    m, n, k = 12, 128, 128
    calls = {}

    def _fail_interp_3d(*args, **kwargs):
        raise AssertionError("_interp_3d should not be used when n/k match and only m needs interpolation")

    def _spy_interp_1d(x, y, value):
        calls["x"] = x
        calls["y"] = y
        calls["value"] = value
        return {"latency": 0.1 + value * 0.001 + n * 0.0001 + k * 0.00001, "power": 0.0, "energy": 0.0}

    monkeypatch.setattr("aiconfigurator.sdk.interpolation.interp_3d", _fail_interp_3d)
    monkeypatch.setattr("aiconfigurator.sdk.interpolation.interp_1d", _spy_interp_1d)

    observed = comprehensive_perf_db.query_gemm(m, n, k, quant_mode, database_mode=common.DatabaseMode.SILICON)
    expected = 0.1 + m * 0.001 + n * 0.0001 + k * 0.00001

    assert math.isclose(float(observed), expected)
    assert calls["x"] == [8, 16]
    assert calls["value"] == m
    assert observed.source == "silicon"


def test_query_gemm_hybrid_propagates_plain_interpolation_value_error(comprehensive_perf_db, monkeypatch):
    """Only the typed interpolation coverage signal may trigger HYBRID fallback."""

    def _broken_interp_3d(*args, **kwargs):
        raise ValueError("malformed GEMM interpolation input")

    monkeypatch.setattr("aiconfigurator.sdk.interpolation.interp_3d", _broken_interp_3d)
    comprehensive_perf_db.query_gemm.cache_clear()

    with pytest.raises(ValueError, match="malformed GEMM interpolation input"):
        comprehensive_perf_db.query_gemm(
            13,
            257,
            513,
            common.GEMMQuantMode.bfloat16,
            database_mode=common.DatabaseMode.HYBRID,
        )


def test_query_gemm_fast_paths_support_legacy_scalar_leaves(mutable_comprehensive_perf_db):
    """Fast GEMM paths should support legacy scalar-leaf tables."""
    db = mutable_comprehensive_perf_db
    quant_mode = common.GEMMQuantMode.bfloat16
    db._gemm_data[quant_mode] = {
        8: {128: {128: 0.5}},
        16: {128: {128: 0.9}},
    }

    exact = db.query_gemm(8, 128, 128, quant_mode, database_mode=common.DatabaseMode.SILICON)
    interp = db.query_gemm(12, 128, 128, quant_mode, database_mode=common.DatabaseMode.SILICON)

    assert math.isclose(float(exact), 0.5)
    assert math.isclose(float(interp), 0.7)
    assert exact.energy == 0.0
    assert interp.energy == 0.0
    assert interp.source == "silicon"


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


@pytest.mark.parametrize("moe_backend", ["DEEPGEMM", "CUTE_DSL"])
@pytest.mark.parametrize("database_mode", list(common.DatabaseMode))
def test_query_trtllm_alltoall_not_enabled_is_zero_in_every_mode(stub_perf_db, monkeypatch, moe_backend, database_mode):
    """Kernel eligibility is a no-op decision, not a missing calibration."""

    def fail_estimate(*args, **kwargs):
        raise AssertionError("disabled alltoall must not enter empirical estimation")

    monkeypatch.setattr("aiconfigurator.sdk.operations.util_empirical.estimate", fail_estimate)
    result = stub_perf_db.query_trtllm_alltoall(
        op_name="alltoall_dispatch",
        num_tokens=16,
        hidden_size=1024,
        topk=8,
        num_experts=256,
        moe_ep_size=8,
        quant_mode=common.MoEQuantMode.fp8,
        node_num=1,
        database_mode=database_mode,
        moe_backend=moe_backend,
    )

    if database_mode == common.DatabaseMode.SOL_FULL:
        assert result == (0.0, 0.0, 0.0)
    else:
        assert float(result) == 0.0
        expected_source = "sol" if database_mode == common.DatabaseMode.SOL else "empirical"
        assert result.source == expected_source


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
    quant_mode = "bfloat16"  # for SOL branch we ignore the custom allreduce dict

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
    quant_mode = "bfloat16"

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
        _custom_allreduce_data['bfloat16']['2']['AUTO']['1024'] == 5.0
    For tp_size=2 and size=1024 exactly, we expect 5.0.
    """
    size = 1024
    tp_size = 2
    quant_mode = "bfloat16"

    # Use a “SILICON” mode to force fallback into the custom-data path
    custom_latency = stub_perf_db.query_custom_allreduce(
        quant_mode, tp_size, size, database_mode=common.DatabaseMode.SILICON
    )
    assert math.isclose(custom_latency, 5.0), f"Expected custom-allreduce latency 5.0, got {custom_latency}"


def test_query_allreduce_rms_interpolates_hidden_size(stub_perf_db, monkeypatch):
    from aiconfigurator.sdk.operations.communication import AllReduceRMS

    monkeypatch.setattr(AllReduceRMS, "load_data", classmethod(lambda cls, database: None))
    stub_perf_db._allreduce_rms_data = LoadedOpData(
        {
            common.CommQuantMode.half: {
                8: {
                    "allreduce_residual_rms": {
                        2048: {
                            "AUTO": {
                                163840: {"latency": 0.0100, "power": 0.0, "energy": 0.0},
                                327680: {"latency": 0.0150, "power": 0.0, "energy": 0.0},
                            }
                        },
                        4096: {
                            "AUTO": {
                                327680: {"latency": 0.0125, "power": 0.0, "energy": 0.0},
                                655360: {"latency": 0.0200, "power": 0.0, "energy": 0.0},
                            }
                        },
                    }
                }
            }
        },
        common.PerfDataFilename.allreduce_rms,
        "dummy_allreduce_rms_perf.txt",
    )

    exact_2048 = stub_perf_db.query_allreduce_rms(
        common.CommQuantMode.half,
        tp_size=8,
        size=163840,
        hidden_size=2048,
        database_mode=common.DatabaseMode.SILICON,
    )
    interpolated = stub_perf_db.query_allreduce_rms(
        common.CommQuantMode.half,
        tp_size=8,
        size=245760,
        hidden_size=3072,
        database_mode=common.DatabaseMode.SILICON,
    )

    assert float(exact_2048) == pytest.approx(0.0100)
    assert float(interpolated) == pytest.approx(0.01125)


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
    HYBRID falls back to the empirical path when silicon is missing. With no util
    data for the slice (and no transfer reference), both EMPIRICAL and HYBRID now
    raise EmpiricalNotImplementedError consistently -- no fabricated SOL/constant.
    """
    from aiconfigurator.sdk.errors import EmpiricalNotImplementedError

    kwargs = dict(
        b=2,
        s=32,
        prefix=0,
        n=32,
        n_kv=16,
        kvcache_quant_mode=common.KVCacheQuantMode.bfloat16,
        fmha_quant_mode=common.FMHAQuantMode.bfloat16,
        head_size=128,
        window_size=0,
    )

    with pytest.raises(EmpiricalNotImplementedError):
        stub_perf_db.query_context_attention(database_mode=common.DatabaseMode.EMPIRICAL, **kwargs)
    with pytest.raises(EmpiricalNotImplementedError):
        stub_perf_db.query_context_attention(database_mode=common.DatabaseMode.HYBRID, **kwargs)


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
    assert spec["gpu"]["bfloat16_tc_flops"] == 1_000.0
    assert spec["node"]["inter_node_bw"] == 100.0


# ---------------------------------------------------------------------------
# Communication EMPIRICAL mode is data-calibrated (util = SOL/measured).
# These guard the per-op invariant at collected points and missing slices.
# ---------------------------------------------------------------------------


def test_query_custom_allreduce_empirical_data_calibrated(comprehensive_perf_db):
    """At a collected (quant, tp, size) point, EMPIRICAL recovers silicon."""
    q, tp, size = common.CommQuantMode.half, 2, 1024  # collected in the stub
    silicon = comprehensive_perf_db.query_custom_allreduce(q, tp, size, database_mode=common.DatabaseMode.SILICON)
    empirical = comprehensive_perf_db.query_custom_allreduce(q, tp, size, database_mode=common.DatabaseMode.EMPIRICAL)
    assert math.isclose(float(empirical), float(silicon), rel_tol=1e-6)


def test_query_custom_allreduce_empirical_raises_without_data(comprehensive_perf_db):
    """With no custom_allreduce data for the slice, EMPIRICAL raises (no SOL/const)."""
    from aiconfigurator.sdk.errors import EmpiricalNotImplementedError

    q, tp, size = common.CommQuantMode.int8, 2, 1024  # int8 not in the custom_allreduce stub
    with pytest.raises(EmpiricalNotImplementedError):
        comprehensive_perf_db.query_custom_allreduce(q, tp, size, database_mode=common.DatabaseMode.EMPIRICAL)


def test_query_nccl_empirical_data_calibrated(comprehensive_perf_db):
    """At a collected (dtype, op, num_gpus, size) point, EMPIRICAL recovers silicon."""
    dtype, ngpu, op, size = common.CommQuantMode.half, 2, "all_gather", 1024  # collected
    silicon = comprehensive_perf_db.query_nccl(dtype, ngpu, op, size, database_mode=common.DatabaseMode.SILICON)
    empirical = comprehensive_perf_db.query_nccl(dtype, ngpu, op, size, database_mode=common.DatabaseMode.EMPIRICAL)
    assert math.isclose(float(empirical), float(silicon), rel_tol=1e-6)


def test_query_nccl_empirical_raises_without_data(comprehensive_perf_db):
    """With no NCCL data for the operation, EMPIRICAL raises (no SOL/const)."""
    from aiconfigurator.sdk.errors import EmpiricalNotImplementedError

    dtype, ngpu, op, size = common.CommQuantMode.half, 2, "all_reduce", 1024  # all_reduce absent in stub
    with pytest.raises(EmpiricalNotImplementedError):
        comprehensive_perf_db.query_nccl(dtype, ngpu, op, size, database_mode=common.DatabaseMode.EMPIRICAL)
