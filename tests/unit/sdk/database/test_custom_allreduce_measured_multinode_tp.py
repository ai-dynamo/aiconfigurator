# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for measured multi-node custom-allreduce TP slices (issue #1416).

On NVL systems (GB200/GB300) a node holds 4 GPUs, so every serving-relevant
TP size (8/16/32) spans nodes. The query path used to cap every lookup at
``num_gpus_per_node`` and then synthesize the multi-node cost from a
p2p-bandwidth ratio, which meant collected TP8/TP16 rows were ignored even
when present.

These tests pin the policy:
  * an exactly-measured TP slice is used verbatim, with no bandwidth scaling;
  * an unmeasured TP count still falls back to the node-capped slice plus the
    bandwidth correction (issue #1260 compatibility behaviour).
"""

import math
from collections import defaultdict

import pytest
import yaml

from aiconfigurator.sdk import common
from aiconfigurator.sdk.operations import util_empirical
from aiconfigurator.sdk.operations.communication import CustomAllReduce
from aiconfigurator.sdk.perf_database import PerfDatabase

# Distinct per-tp latencies so a wrong slice cannot coincidentally match.
_LATENCY_BY_TP = {2: 1.0, 4: 2.0, 8: 5.0, 16: 9.0}
_MSG_SIZES = (1024, 2048, 4096)


def _dataset(tp_sizes):
    """Build the 4-deep defaultdict shape that load_custom_allreduce_data returns."""
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict())))
    for tp in tp_sizes:
        for msg_size in _MSG_SIZES:
            data[common.CommQuantMode.half][tp]["AUTO"][msg_size] = {
                "latency": _LATENCY_BY_TP[tp],
                "power": 0.0,
                "energy": 0.0,
            }
    return data


@pytest.fixture
def nvl_db_factory(tmp_path, monkeypatch):
    """PerfDatabase factory on a 4-GPU-per-node (NVL) system spec."""
    dummy_spec = {
        "data_dir": "data",
        "misc": {"nccl_version": "v1"},
        "gpu": {
            "bfloat16_tc_flops": 1_000.0,
            "mem_bw": 100.0,
            "mem_empirical_constant_latency": 1.0,
        },
        "node": {
            "inter_node_bw": 50.0,
            "intra_node_bw": 100.0,
            "num_gpus_per_node": 4,
            "p2p_latency": 0.000001,
        },
    }
    monkeypatch.setattr(yaml, "load", lambda stream, Loader=None: dummy_spec)  # noqa: N803

    def _factory(custom_allreduce_data):
        monkeypatch.setattr(
            "aiconfigurator.sdk.operations.communication.load_custom_allreduce_data",
            lambda path: custom_allreduce_data,
        )
        monkeypatch.setattr("aiconfigurator.sdk.operations.gemm.load_gemm_data", lambda p: {})
        monkeypatch.setattr("aiconfigurator.sdk.operations.attention.load_context_attention_data", lambda p: {})
        monkeypatch.setattr("aiconfigurator.sdk.operations.attention.load_generation_attention_data", lambda p: {})
        monkeypatch.setattr("aiconfigurator.sdk.operations.moe.load_moe_data", lambda p: ({}, {}))
        monkeypatch.setattr("aiconfigurator.sdk.operations.communication.load_nccl_data", lambda p: {})
        monkeypatch.setattr("aiconfigurator.sdk.operations.mla.load_context_mla_data", lambda p: {})
        monkeypatch.setattr("aiconfigurator.sdk.operations.mla.load_generation_mla_data", lambda p: {})
        monkeypatch.setattr("aiconfigurator.sdk.operations.mla.load_mla_bmm_data", lambda p: {})
        monkeypatch.setattr("aiconfigurator.sdk.operations.dsa.load_context_dsa_module_data", lambda p: None)
        monkeypatch.setattr("aiconfigurator.sdk.operations.dsa.load_generation_dsa_module_data", lambda p: None)

        # Both caches are process-lifetime and keyed by database identity,
        # which is constant here; clear them so each factory call really
        # loads its own dataset.
        CustomAllReduce.clear_cache()
        util_empirical.clear_grid_cache()

        (tmp_path / "sys.yaml").write_text("dummy: data")
        return PerfDatabase("sys", "backend", "v1", str(tmp_path))

    return _factory


@pytest.mark.unit
@pytest.mark.parametrize("tp_size", [8, 16])
def test_measured_multinode_tp_is_returned_verbatim(nvl_db_factory, tp_size):
    """A collected cross-node TP slice must be used as-is, not scaled.

    The measured curve already contains the real cross-node cost; applying the
    p2p-bandwidth correction on top would double-count the multi-node penalty.
    """
    db = nvl_db_factory(_dataset([2, 4, 8, 16]))

    result = db.query_custom_allreduce(
        common.CommQuantMode.half,
        tp_size=tp_size,
        size=2048,
        database_mode=common.DatabaseMode.SILICON,
    )

    assert math.isclose(float(result), _LATENCY_BY_TP[tp_size], rel_tol=1e-9)


@pytest.mark.unit
def test_unmeasured_multinode_tp_still_falls_back_to_node_slice(nvl_db_factory):
    """Without a measured TP8 slice, TP8 keeps the node-capped + scaled estimate.

    This is the issue #1260 compatibility path: it must stay reachable so
    multi-node configs on systems with no collected data remain runnable.
    """
    db = nvl_db_factory(_dataset([2, 4]))

    result = db.query_custom_allreduce(
        common.CommQuantMode.half,
        tp_size=8,
        size=2048,
        database_mode=common.DatabaseMode.SILICON,
    )

    # Derived from the TP4 slice, so it must differ from the raw TP4 latency
    # (the bandwidth correction is applied) and from any measured TP8 value.
    assert not math.isclose(float(result), _LATENCY_BY_TP[8], rel_tol=1e-9)
    assert float(result) > 0.0


@pytest.mark.unit
def test_measured_slice_beats_node_cap_for_same_query(nvl_db_factory):
    """The same TP16 query must change once measured TP16 rows exist."""
    without = float(
        nvl_db_factory(_dataset([2, 4])).query_custom_allreduce(
            common.CommQuantMode.half, tp_size=16, size=2048, database_mode=common.DatabaseMode.SILICON
        )
    )
    with_data = float(
        nvl_db_factory(_dataset([2, 4, 16])).query_custom_allreduce(
            common.CommQuantMode.half, tp_size=16, size=2048, database_mode=common.DatabaseMode.SILICON
        )
    )

    assert math.isclose(with_data, _LATENCY_BY_TP[16], rel_tol=1e-9)
    assert not math.isclose(without, with_data, rel_tol=1e-9)


@pytest.mark.unit
def test_within_node_tp_is_unaffected(nvl_db_factory):
    """TP <= num_gpus_per_node keeps its exact-slice behaviour."""
    db = nvl_db_factory(_dataset([2, 4, 8, 16]))

    for tp in (2, 4):
        result = db.query_custom_allreduce(
            common.CommQuantMode.half, tp_size=tp, size=2048, database_mode=common.DatabaseMode.SILICON
        )
        assert math.isclose(float(result), _LATENCY_BY_TP[tp], rel_tol=1e-9)
