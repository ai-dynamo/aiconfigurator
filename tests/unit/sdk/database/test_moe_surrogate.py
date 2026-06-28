# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from aiconfigurator.sdk import common
from aiconfigurator.sdk.operations.moe import (
    MoE,
    MoEDispatch,
    TrtLLMWideEPMoE,
    load_moe_data,
    load_trtllm_alltoall_data,
    load_wideep_context_moe_data,
    load_wideep_generation_moe_data,
    load_wideep_moe_compute_data,
)
from aiconfigurator.sdk.perf_database import PerfDataNotAvailableError
from aiconfigurator.sdk.performance_result import PerformanceResult


class _Data(dict):
    def raise_if_not_loaded(self) -> None:
        return None


class _Database:
    def __init__(self, backend: str) -> None:
        self.backend = backend
        self.system = "test"
        self.version = "test"
        self._default_database_mode = common.DatabaseMode.SILICON
        self.system_spec = {
            "gpu": {
                "bfloat16_tc_flops": 1e30,
                "mem_bw": 1e30,
                "sm_version": 100,
            },
            "node": {"num_gpus_per_node": 8},
        }

    @staticmethod
    def _interp_pr(latency, energy=0.0):
        return PerformanceResult(latency, energy=energy, source="silicon")

    @staticmethod
    def _query_silicon_or_hybrid(*, get_silicon, **_kwargs):
        return get_silicon()


def _metric(latency: float, power: float) -> dict[str, float]:
    return {"latency": latency, "power": power, "energy": latency * power}


def test_moe_token_curve_and_low_token_launch(monkeypatch) -> None:
    monkeypatch.setattr(MoE, "load_data", classmethod(lambda _cls, _database: None))
    db = _Database(common.BackendName.vllm.value)
    quant = common.MoEQuantMode.bfloat16
    curve = {4: _metric(10, 2), 8: _metric(14, 4)}
    db._moe_data = _Data({quant: {"uniform": {2: {8: {16: {32: {1: {1: curve}}}}}}}})

    def query(tokens: int) -> PerformanceResult:
        return MoE._query_moe_table(
            db,
            tokens,
            16,
            32,
            2,
            8,
            1,
            1,
            quant,
            "uniform",
            database_mode=common.DatabaseMode.SILICON,
        )

    interior = query(6)
    assert float(interior) == 12
    assert interior.energy == 36  # interpolated 3 W * 12 ms

    lower = query(2)
    assert float(lower) == 8  # launch=6, re-anchored to 10 at token=4
    assert lower.energy == 16


def test_regular_moe_gated_and_nongated_rows_do_not_leak(monkeypatch) -> None:
    row = {
        "moe_dtype": "bfloat16",
        "num_tokens": "4",
        "hidden_size": "16",
        "inter_size": "32",
        "topk": "2",
        "num_experts": "8",
        "moe_tp_size": "1",
        "moe_ep_size": "1",
        "distribution": "uniform",
    }
    monkeypatch.setattr(
        "aiconfigurator.sdk.operations.moe._read_filtered_rows",
        lambda _source: [
            {**row, "kernel_source": "moe_torch_flow", "latency": "10"},
            {**row, "kernel_source": "moe_torch_flow_nongated", "latency": "20"},
        ],
    )
    data, _ = load_moe_data("unused")
    monkeypatch.setattr(MoE, "load_data", classmethod(lambda _cls, _database: None))
    db = _Database(common.BackendName.vllm.value)
    db._moe_data = _Data(data)
    quant = common.MoEQuantMode.bfloat16
    assert set(data[quant]) == {"uniform"}

    def query(is_gated: bool) -> PerformanceResult:
        return MoE._query_moe_table(
            db,
            4,
            16,
            32,
            2,
            8,
            1,
            1,
            quant,
            "uniform",
            database_mode=common.DatabaseMode.SILICON,
            is_gated=is_gated,
        )

    assert float(query(True)) == 10
    assert float(query(False)) == 20


def test_source_priority_is_first_wins_for_wideep_loaders(monkeypatch) -> None:
    common_row = {
        "moe_dtype": "bfloat16",
        "num_tokens": "4",
        "hidden_size": "16",
        "inter_size": "32",
        "topk": "2",
        "num_experts": "8",
        "moe_tp_size": "1",
        "moe_ep_size": "1",
        "distribution": "uniform",
    }
    rows = [{**common_row, "latency": "10"}, {**common_row, "latency": "99"}]
    monkeypatch.setattr("aiconfigurator.sdk.operations.moe._read_filtered_rows", lambda _source: rows)
    quant = common.MoEQuantMode.bfloat16
    for loader in (load_wideep_context_moe_data, load_wideep_generation_moe_data):
        data = loader("unused")
        assert data[quant]["uniform"][2][8][16][32][1][1][4]["latency"] == 10

    compute_rows = [
        {**common_row, "kernel_source": "kernel", "num_slots": "8", "latency": "11"},
        {**common_row, "kernel_source": "kernel", "num_slots": "8", "latency": "98"},
    ]
    monkeypatch.setattr("aiconfigurator.sdk.operations.moe._read_filtered_rows", lambda _source: compute_rows)
    compute = load_wideep_moe_compute_data("unused")
    assert compute["kernel"][quant]["uniform"][2][8][16][32][8][1][1][4]["latency"] == 11

    alltoall_row = {
        "op_name": "alltoall_dispatch",
        "kernel_source": "NVLinkTwoSided",
        "moe_dtype": "bfloat16",
        "num_nodes": "1",
        "num_tokens": "4",
        "hidden_size": "16",
        "topk": "2",
        "num_experts": "8",
        "moe_ep_size": "4",
    }
    monkeypatch.setattr(
        "aiconfigurator.sdk.operations.moe._read_filtered_rows",
        lambda _source: [{**alltoall_row, "latency": "12"}, {**alltoall_row, "latency": "97"}],
    )
    alltoall = load_trtllm_alltoall_data("unused")
    assert alltoall["NVLinkTwoSided"]["alltoall_dispatch"][quant][1][16][2][8][4][4]["latency"] == 12


def test_deepep_independent_sms_token_mesh(monkeypatch) -> None:
    monkeypatch.setattr(MoEDispatch, "load_data", classmethod(lambda _cls, _database: None))
    db = _Database(common.BackendName.sglang.value)
    table = {
        10: {4: _metric(2, 2), 8: _metric(4, 4)},
        30: {4: _metric(6, 6), 8: _metric(8, 8)},
    }
    db._wideep_deepep_normal_data = _Data({2: {16: {2: {8: table}}}})

    result = MoEDispatch._query_wideep_deepep_normal_table(
        db,
        node_num=2,
        num_tokens=6,
        num_experts=8,
        topk=2,
        hidden_size=16,
        sms=20,
        database_mode=common.DatabaseMode.SILICON,
    )

    assert float(result) == 0.005
    assert result.energy == 0.025  # interpolated 5 W * 5 us, converted to ms


def test_wideep_moe_upper_boundary_carries_sol_ratio(monkeypatch) -> None:
    monkeypatch.setattr(TrtLLMWideEPMoE, "load_data", classmethod(lambda _cls, _database: None))
    monkeypatch.setattr(
        TrtLLMWideEPMoE,
        "_select_kernel",
        classmethod(lambda _cls, _database, _quant: "kernel"),
    )
    db = _Database(common.BackendName.trtllm.value)
    db.system_spec["gpu"]["bfloat16_tc_flops"] = 1e9
    db.system_spec["gpu"]["mem_bw"] = 1e9
    quant = common.MoEQuantMode.bfloat16
    curve = {4: _metric(8, 2), 8: _metric(10, 3)}
    db._wideep_moe_compute_data = _Data({"kernel": {quant: {"uniform": {2: {8: {16: {32: {8: {1: {1: curve}}}}}}}}}})

    def query(tokens: int, mode: common.DatabaseMode) -> PerformanceResult:
        return TrtLLMWideEPMoE._query_compute_table(
            db,
            tokens,
            16,
            32,
            2,
            8,
            8,
            1,
            1,
            quant,
            "uniform",
            database_mode=mode,
        )

    sol_at_8 = float(query(8, common.DatabaseMode.SOL))
    sol_at_16 = float(query(16, common.DatabaseMode.SOL))
    result = query(16, common.DatabaseMode.SILICON)
    expected = 10 * sol_at_16 / sol_at_8
    assert float(result) == expected
    assert result.energy == 3 * expected


def test_wideep_compute_rejects_non_gated_query(monkeypatch) -> None:
    monkeypatch.setattr(TrtLLMWideEPMoE, "load_data", classmethod(lambda _cls, _database: None))
    db = _Database(common.BackendName.trtllm.value)
    quant = common.MoEQuantMode.bfloat16
    legacy_curve = {4: _metric(7, 1)}
    db._wideep_moe_compute_data = _Data(
        {"kernel": {quant: {"uniform": {2: {8: {16: {32: {8: {1: {1: legacy_curve}}}}}}}}}}
    )

    with pytest.raises(PerfDataNotAvailableError, match="does not distinguish gated and non-gated"):
        TrtLLMWideEPMoE._query_compute_table(
            db,
            4,
            16,
            32,
            2,
            8,
            8,
            1,
            1,
            quant,
            "uniform",
            database_mode=common.DatabaseMode.SILICON,
            is_gated=False,
        )
