# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for NCCL/OneCCL fallback logic in query_nccl.

When NCCL perf data is not available (e.g. on XPU systems), query_nccl should
transparently fall back to OneCCL data if it has been loaded.
"""

import math
from collections import defaultdict

import pytest
import yaml

from aiconfigurator.sdk import common
from aiconfigurator.sdk.perf_database import PerfDatabase, PerfDataNotAvailableError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_comm_data(dtypes, operations, gpu_counts, msg_sizes, scale=0.001, power=0.0):
    """Build a nested dict matching the NCCL/OneCCL data layout.

    Args:
        power: Power value. OneCCL data has no power measurement, so defaults to 0.0.
    """
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for dtype in dtypes:
        for op in operations:
            for ng in gpu_counts:
                for ms in msg_sizes:
                    latency = scale * ms * ng
                    data[dtype][op][ng][ms] = {
                        "latency": latency,
                        "power": power,
                        "energy": latency * power,
                    }
    return dict(data)


_DTYPES = [common.CommQuantMode.half, common.CommQuantMode.int8]
_OPS = ["all_reduce", "all_gather", "reduce_scatter"]
_NCCL_GPUS = [2, 4, 8]
_ONECCL_GPUS = [2, 4]
_MSG_SIZES = [512, 1024, 2048, 4096]


@pytest.fixture
def _db_factory(tmp_path, monkeypatch):
    """
    Factory that returns a PerfDatabase whose _nccl_data and _oneccl_data
    can be independently controlled via keyword arguments.

    Usage:
        db = db_factory(nccl_data=<dict|None>, oneccl_data=<dict|None>,
                        has_oneccl_version=True)
    """
    dummy_spec_base = {
        "data_dir": "data",
        "misc": {"nccl_version": "v1"},
        "gpu": {
            "bfloat16_tc_flops": 1_000.0,
            "mem_bw": 100.0,
            "mem_empirical_constant_latency": 1.0,
        },
        "node": {
            "inter_node_bw": 100.0,
            "intra_node_bw": 100.0,
            "num_gpus_per_node": 8,
            "p2p_latency": 0.000001,
        },
    }

    def _factory(*, nccl_data=None, oneccl_data=None, has_oneccl_version=True):
        spec = dict(dummy_spec_base)
        spec["misc"] = dict(spec["misc"])
        if has_oneccl_version:
            spec["misc"]["oneccl_version"] = "v1"

        monkeypatch.setattr(yaml, "load", lambda stream, Loader=None: spec)  # noqa: N803

        # Track which path each loader was called with so we can return the right data.
        def _nccl_loader(path):
            return nccl_data

        def _oneccl_loader(path):
            return oneccl_data

        # We need load_nccl_data to differentiate between nccl and oneccl calls.
        # PerfDataFilename.nccl -> nccl_data_dir, PerfDataFilename.oneccl -> oneccl_data_dir.
        # Both map to load_nccl_data, but the path differs. We capture via closure.
        call_count = {"n": 0}

        def _nccl_load_dispatch(path):
            # First call is for NCCL, second for OneCCL (per __init__ ordering)
            call_count["n"] += 1
            if call_count["n"] == 1:
                return nccl_data
            return oneccl_data

        monkeypatch.setattr("aiconfigurator.sdk.perf_database.load_nccl_data", _nccl_load_dispatch)

        # Patch all other loaders to avoid file access
        monkeypatch.setattr("aiconfigurator.sdk.perf_database.load_gemm_data", lambda p: {})
        monkeypatch.setattr("aiconfigurator.sdk.perf_database.load_context_attention_data", lambda p: {})
        monkeypatch.setattr("aiconfigurator.sdk.perf_database.load_generation_attention_data", lambda p: {})
        monkeypatch.setattr("aiconfigurator.sdk.perf_database.load_moe_data", lambda p: ({}, {}))
        monkeypatch.setattr("aiconfigurator.sdk.perf_database.load_custom_allreduce_data", lambda p: {})
        monkeypatch.setattr("aiconfigurator.sdk.perf_database.load_context_mla_data", lambda p: {})
        monkeypatch.setattr("aiconfigurator.sdk.perf_database.load_generation_mla_data", lambda p: {})
        monkeypatch.setattr("aiconfigurator.sdk.perf_database.load_mla_bmm_data", lambda p: {})
        monkeypatch.setattr("aiconfigurator.sdk.perf_database.load_context_dsa_module_data", lambda p: None)
        monkeypatch.setattr("aiconfigurator.sdk.perf_database.load_generation_dsa_module_data", lambda p: None)

        yaml_file = tmp_path / "sys.yaml"
        yaml_file.write_text("dummy: data")
        return PerfDatabase("sys", "backend", "v1", str(tmp_path))

    return _factory


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestNcclOnecclFallback:
    """Verify the NCCL → OneCCL fallback logic in query_nccl (SILICON mode)."""

    def test_nccl_loaded_uses_nccl(self, _db_factory):
        """When NCCL data is loaded, query_nccl should use it directly."""
        nccl = _make_comm_data(_DTYPES, _OPS, _NCCL_GPUS, _MSG_SIZES, scale=0.001, power=5.0)
        oneccl = _make_comm_data(_DTYPES, _OPS, _ONECCL_GPUS, _MSG_SIZES, scale=0.999)
        db = _db_factory(nccl_data=nccl, oneccl_data=oneccl)

        result = db.query_nccl(
            common.CommQuantMode.half,
            4,
            "all_reduce",
            1024,
            database_mode=common.DatabaseMode.SILICON,
        )
        # Should match NCCL scale (0.001), not OneCCL scale (0.999)
        expected_latency = 0.001 * 1024 * 4
        assert math.isclose(float(result), expected_latency, rel_tol=1e-6), (
            f"Expected NCCL latency {expected_latency}, got {float(result)}"
        )

    def test_fallback_to_oneccl_when_nccl_not_loaded(self, _db_factory):
        """When NCCL data is not loaded but OneCCL is, query_nccl should use OneCCL."""
        oneccl = _make_comm_data(_DTYPES, _OPS, _ONECCL_GPUS, _MSG_SIZES, scale=0.002)
        db = _db_factory(nccl_data=None, oneccl_data=oneccl)

        result = db.query_nccl(
            common.CommQuantMode.half,
            4,
            "all_reduce",
            1024,
            database_mode=common.DatabaseMode.SILICON,
        )
        expected_latency = 0.002 * 1024 * 4
        assert math.isclose(float(result), expected_latency, rel_tol=1e-6), (
            f"Expected OneCCL fallback latency {expected_latency}, got {float(result)}"
        )

    def test_fallback_oneccl_has_no_power(self, _db_factory):
        """OneCCL data has no power measurement — energy should be 0.0."""
        oneccl = _make_comm_data(_DTYPES, _OPS, _ONECCL_GPUS, _MSG_SIZES, scale=0.002)  # power=0.0
        db = _db_factory(nccl_data=None, oneccl_data=oneccl)

        result = db.query_nccl(
            common.CommQuantMode.half,
            4,
            "all_gather",
            2048,
            database_mode=common.DatabaseMode.SILICON,
        )
        expected_latency = 0.002 * 2048 * 4
        assert math.isclose(float(result), expected_latency, rel_tol=1e-6)
        assert result.energy == 0.0, "OneCCL has no power data, so energy should be 0.0"

    def test_raises_when_neither_loaded(self, _db_factory):
        """When neither NCCL nor OneCCL data is loaded, raise PerfDataNotAvailableError."""
        db = _db_factory(nccl_data=None, oneccl_data=None)

        with pytest.raises(PerfDataNotAvailableError):
            db.query_nccl(
                common.CommQuantMode.half,
                4,
                "all_reduce",
                1024,
                database_mode=common.DatabaseMode.SILICON,
            )

    def test_raises_when_nccl_not_loaded_and_no_oneccl_version(self, _db_factory):
        """When NCCL is not loaded and system has no oneccl_version, should raise."""
        db = _db_factory(nccl_data=None, oneccl_data=None, has_oneccl_version=False)

        with pytest.raises(PerfDataNotAvailableError):
            db.query_nccl(
                common.CommQuantMode.half,
                4,
                "all_reduce",
                1024,
                database_mode=common.DatabaseMode.SILICON,
            )

    def test_hybrid_falls_back_to_empirical_when_neither_loaded(self, _db_factory):
        """In HYBRID mode, when neither backend has data, fall back to empirical."""
        db = _db_factory(nccl_data=None, oneccl_data=None)

        # HYBRID mode should not raise — it falls back to empirical
        result = db.query_nccl(
            common.CommQuantMode.half,
            4,
            "all_gather",
            1024,
            database_mode=common.DatabaseMode.HYBRID,
        )
        assert float(result) > 0, "HYBRID empirical fallback should return positive latency"

    def test_single_gpu_returns_zero(self, _db_factory):
        """num_gpus=1 is a fast-path returning zero latency regardless of data."""
        db = _db_factory(nccl_data=None, oneccl_data=None)

        result = db.query_nccl(
            common.CommQuantMode.half,
            1,
            "all_reduce",
            1024,
            database_mode=common.DatabaseMode.SILICON,
        )
        assert float(result) == 0.0
        assert result.energy == 0.0

    def test_fallback_multiple_operations(self, _db_factory):
        """Verify fallback works for all supported collective operations."""
        oneccl = _make_comm_data(_DTYPES, _OPS, _ONECCL_GPUS, _MSG_SIZES, scale=0.003)
        db = _db_factory(nccl_data=None, oneccl_data=oneccl)

        for op in ["all_reduce", "all_gather", "reduce_scatter"]:
            result = db.query_nccl(
                common.CommQuantMode.half,
                4,
                op,
                1024,
                database_mode=common.DatabaseMode.SILICON,
            )
            expected = 0.003 * 1024 * 4
            assert math.isclose(float(result), expected, rel_tol=1e-6), (
                f"Fallback failed for operation '{op}': expected {expected}, got {float(result)}"
            )
