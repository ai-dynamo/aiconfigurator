# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import aiconfigurator.sdk.operations.mla as mla_module
from aiconfigurator.sdk import common
from aiconfigurator.sdk.operations.mla import ContextMLA, GenerationMLA, WideEPGenerationMLA
from aiconfigurator.sdk.performance_result import PerformanceResult


class _Data(dict):
    def raise_if_not_loaded(self) -> None:
        return None


class _Database:
    def __init__(self, backend: str = "trtllm") -> None:
        self.backend = backend
        self.system = "test"
        self.version = "test"
        self.systems_root = "/tmp"
        self.enable_shared_layer = False
        self._default_database_mode = common.DatabaseMode.SILICON
        self.system_spec = {
            "data_dir": "data",
            "gpu": {"bfloat16_tc_flops": 1e12, "mem_bw": 1e12},
        }

    @staticmethod
    def _build_op_sources(*_args):
        return []

    @staticmethod
    def _interp_pr(latency, energy=0.0):
        return PerformanceResult(latency, energy=energy, source="silicon")

    @staticmethod
    def _query_silicon_or_hybrid(*, get_silicon, **_kwargs):
        return get_silicon()


def _metric(latency: float, power: float) -> dict[str, float]:
    return {"latency": latency, "power": power, "energy": latency * power}


def test_generation_load_keeps_raw_sparse_coordinates(monkeypatch) -> None:
    kv = common.KVCacheQuantMode.bfloat16
    raw = {kv: {4: {1: {8: _metric(2, 3), 16: _metric(4, 3)}}}}
    monkeypatch.setattr(mla_module, "load_generation_mla_data", lambda _sources: raw)
    GenerationMLA.clear_cache()
    db = _Database()
    try:
        GenerationMLA.load_data(db)
        assert set(db._generation_mla_data[kv][4][1]) == {8, 16}
    finally:
        GenerationMLA.clear_cache()


def test_context_prefix_uses_full_sequence_exact_point(monkeypatch) -> None:
    monkeypatch.setattr(ContextMLA, "load_data", classmethod(lambda _cls, _db: None))
    db = _Database()
    fmha = common.FMHAQuantMode.bfloat16
    kv = common.KVCacheQuantMode.bfloat16
    db._context_mla_data = _Data({fmha: {kv: {4: {16: {1: _metric(8, 5)}}}}})

    result = ContextMLA._query_context_mla_table(db, 1, 8, 8, 4, kv, fmha, database_mode=common.DatabaseMode.SILICON)
    assert float(result) == 6  # exact full_s=16 multiplied by the 3/4 prefix correction
    assert result.energy == 30
    sol = ContextMLA._query_context_mla_table(db, 1, 8, 8, 4, kv, fmha, common.DatabaseMode.SOL)
    assert sol.source == "sol" and float(sol) > 0


def test_generation_off_grid_interpolates_power_consistent_energy(monkeypatch) -> None:
    monkeypatch.setattr(GenerationMLA, "load_data", classmethod(lambda _cls, _db: None))
    db = _Database()
    kv = common.KVCacheQuantMode.bfloat16
    table = {
        4: {
            1: {10: _metric(2, 2), 20: _metric(4, 4)},
            3: {10: _metric(6, 6), 20: _metric(8, 8)},
        }
    }
    db._generation_mla_data = _Data({kv: table})

    result = GenerationMLA._query_generation_mla_table(db, 2, 15, 4, kv, database_mode=common.DatabaseMode.SILICON)
    assert float(result) == 5
    assert result.energy == 25  # interpolated 5 W * predicted 5 ms
    assert set(table[4]) == {1, 3} and set(table[4][1]) == {10, 20}


def test_wideep_backend_categories_do_not_share_exact_values(monkeypatch) -> None:
    monkeypatch.setattr(WideEPGenerationMLA, "load_data", classmethod(lambda _cls, _db: None))
    db = _Database("sglang")
    kv = common.KVCacheQuantMode.bfloat16
    fmha = common.FMHAQuantMode.bfloat16
    db._wideep_generation_mla_data = _Data(
        {
            "flashinfer": {kv: {4: {1: {16: _metric(2, 2)}}}},
            "fa3": {kv: {4: {1: {16: _metric(9, 3)}}}},
        }
    )

    def query(backend: str) -> PerformanceResult:
        return WideEPGenerationMLA._query_wideep_generation_mla_table(
            db, 1, 16, 32, kv, fmha, attention_backend=backend, database_mode=common.DatabaseMode.SILICON
        )

    assert float(query("flashinfer")) == 2
    assert float(query("fa3")) == 9
