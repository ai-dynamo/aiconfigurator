# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Regression tests for NVBUG-6095486.

In SILICON mode, query_custom_allreduce must raise a structured
PerfDataNotAvailableError when the loaded custom_allreduce data does not
contain a bucket for the requested (quant_mode, tp_size, strategy) tuple.
Previously it leaked an internal AssertionError from _nearest_1d_point_helper,
which upper layers could not distinguish from real invariant failures.

Additionally, _query_silicon_or_hybrid must not dump a full stack trace for
this expected-but-unavailable case — that traceback noise during otherwise
successful Pareto searches was the user-facing symptom of the bug.
"""

import logging
import math
from collections import defaultdict

import pytest
import yaml

from aiconfigurator.sdk import common
from aiconfigurator.sdk.perf_database import PerfDatabase, PerfDataNotAvailableError


def _make_defaultdict_custom_allreduce(tp_sizes):
    """
    Build a custom_allreduce dataset in the same 4-deep defaultdict shape
    that load_custom_allreduce_data produces, containing entries only for
    the requested tp_sizes under CommQuantMode.half / strategy "AUTO".
    """
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict())))
    for tp in tp_sizes:
        for msg_size in [1024, 2048, 4096]:
            latency = 0.001 * msg_size * tp
            data[common.CommQuantMode.half][tp]["AUTO"][msg_size] = {
                "latency": latency,
                "power": 0.0,
                "energy": 0.0,
            }
    return data


@pytest.fixture
def _db_factory(tmp_path, monkeypatch):
    """
    PerfDatabase factory whose custom_allreduce dataset is injected per call.
    """
    dummy_spec = {
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
    monkeypatch.setattr(yaml, "load", lambda stream, Loader=None: dummy_spec)  # noqa: N803

    def _factory(custom_allreduce_data):
        monkeypatch.setattr(
            "aiconfigurator.sdk.perf_database.load_custom_allreduce_data",
            lambda path: custom_allreduce_data,
        )
        # Minimal stubs for other loaders — we only exercise custom_allreduce.
        monkeypatch.setattr("aiconfigurator.sdk.perf_database.load_gemm_data", lambda p: {})
        monkeypatch.setattr("aiconfigurator.sdk.perf_database.load_context_attention_data", lambda p: {})
        monkeypatch.setattr("aiconfigurator.sdk.perf_database.load_generation_attention_data", lambda p: {})
        monkeypatch.setattr("aiconfigurator.sdk.perf_database.load_moe_data", lambda p: ({}, {}))
        monkeypatch.setattr("aiconfigurator.sdk.perf_database.load_nccl_data", lambda p: {})
        monkeypatch.setattr("aiconfigurator.sdk.perf_database.load_context_mla_data", lambda p: {})
        monkeypatch.setattr("aiconfigurator.sdk.perf_database.load_generation_mla_data", lambda p: {})
        monkeypatch.setattr("aiconfigurator.sdk.perf_database.load_mla_bmm_data", lambda p: {})
        monkeypatch.setattr("aiconfigurator.sdk.perf_database.load_context_dsa_module_data", lambda p: None)
        monkeypatch.setattr("aiconfigurator.sdk.perf_database.load_generation_dsa_module_data", lambda p: None)

        yaml_file = tmp_path / "sys.yaml"
        yaml_file.write_text("dummy: data")
        return PerfDatabase("sys", "backend", "v1", str(tmp_path))

    return _factory


class TestCustomAllreduceEmptyBucket:
    """SILICON mode must surface empty custom_allreduce buckets as structured errors."""

    def test_silicon_raises_structured_error_when_tp_bucket_missing(self, _db_factory):
        """
        Reproduces NVBUG-6095486: CSV has tp_size in {2, 4} only; a tp_size=8
        query must raise PerfDataNotAvailableError, not leak AssertionError
        from _nearest_1d_point_helper.
        """
        data = _make_defaultdict_custom_allreduce(tp_sizes=[2, 4])
        db = _db_factory(data)

        with pytest.raises(PerfDataNotAvailableError) as excinfo:
            db.query_custom_allreduce(
                common.CommQuantMode.half,
                tp_size=8,
                size=6291456,
                database_mode=common.DatabaseMode.SILICON,
            )

        msg = str(excinfo.value)
        assert "tp_size" in msg
        assert "HYBRID" in msg, "error message should point users at the HYBRID workaround"

    def test_silicon_raises_structured_error_when_quant_mode_missing(self, _db_factory):
        """Empty dataset (no quant_mode entries) must raise PerfDataNotAvailableError."""
        data = _make_defaultdict_custom_allreduce(tp_sizes=[])
        db = _db_factory(data)

        with pytest.raises(PerfDataNotAvailableError):
            db.query_custom_allreduce(
                common.CommQuantMode.half,
                tp_size=4,
                size=1024,
                database_mode=common.DatabaseMode.SILICON,
            )

    def test_hybrid_falls_back_to_empirical_when_tp_bucket_missing(self, _db_factory):
        """HYBRID mode must keep falling back cleanly on the same condition."""
        data = _make_defaultdict_custom_allreduce(tp_sizes=[2, 4])
        db = _db_factory(data)

        result = db.query_custom_allreduce(
            common.CommQuantMode.half,
            tp_size=8,
            size=6291456,
            database_mode=common.DatabaseMode.HYBRID,
        )
        assert float(result) > 0, "HYBRID empirical fallback should yield positive latency"

    def test_silicon_available_bucket_still_works(self, _db_factory):
        """Sanity check: a tp_size that IS present still returns the interpolated value."""
        data = _make_defaultdict_custom_allreduce(tp_sizes=[2, 4])
        db = _db_factory(data)

        result = db.query_custom_allreduce(
            common.CommQuantMode.half,
            tp_size=4,
            size=2048,
            database_mode=common.DatabaseMode.SILICON,
        )
        expected = 0.001 * 2048 * 4
        assert math.isclose(float(result), expected, rel_tol=1e-6)

    def test_silicon_missing_bucket_does_not_log_traceback(self, _db_factory, caplog):
        """
        Bullet 1 of NVBUG-6095486: successful searches that merely skip a
        candidate must not spam internal tracebacks. When the structured
        PerfDataNotAvailableError is raised, _query_silicon_or_hybrid must
        not emit an ERROR-level log record with exc_info attached.
        """
        data = _make_defaultdict_custom_allreduce(tp_sizes=[2, 4])
        db = _db_factory(data)

        caplog.set_level(logging.DEBUG, logger="aiconfigurator.sdk.perf_database")
        with pytest.raises(PerfDataNotAvailableError):
            db.query_custom_allreduce(
                common.CommQuantMode.half,
                tp_size=8,
                size=6291456,
                database_mode=common.DatabaseMode.SILICON,
            )

        records_with_traceback = [
            r
            for r in caplog.records
            if r.name == "aiconfigurator.sdk.perf_database" and r.levelno >= logging.ERROR and r.exc_info is not None
        ]
        assert not records_with_traceback, (
            "PerfDataNotAvailableError is a structured signal — it must not be logged "
            f"via logger.exception. Got: {[r.getMessage() for r in records_with_traceback]}"
        )
