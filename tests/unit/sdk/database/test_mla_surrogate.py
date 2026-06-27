# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

import aiconfigurator.sdk.operations.mla as mla_module
from aiconfigurator.sdk import common
from aiconfigurator.sdk.operations.mla import (
    ContextMLA,
    GenerationMLA,
    MLABmm,
    WideEPContextMLA,
    WideEPGenerationMLA,
)
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


def test_mla_bmm_uses_sparse_curve_for_exact_interior_and_exterior(monkeypatch) -> None:
    def fail_legacy_interpolation(*_args, **_kwargs):
        raise AssertionError("legacy interpolation used")

    monkeypatch.setattr(MLABmm, "load_data", classmethod(lambda _cls, _db: None))
    monkeypatch.setattr("aiconfigurator.sdk.interpolation.nearest_1d_point_helper", fail_legacy_interpolation)
    monkeypatch.setattr("aiconfigurator.sdk.interpolation.interp_1d", fail_legacy_interpolation)
    db = _Database()
    quant = common.GEMMQuantMode.bfloat16
    table = {10: _metric(2, 3), 20: _metric(4, 5)}
    db._mla_bmm_data = _Data({quant: {"mla_gen_pre": {4: table}}})

    def query(tokens: int) -> PerformanceResult:
        return MLABmm._query_mla_bmm_table(db, tokens, 4, quant, if_pre=True, database_mode=common.DatabaseMode.SILICON)

    exact = query(10)
    assert float(exact) == 2
    assert exact.energy == 6

    interior = query(15)
    assert float(interior) == 3
    assert interior.energy == 12  # interpolated 4 W * predicted 3 ms

    for tokens, boundary_tokens, boundary_latency, boundary_power in ((5, 10, 2, 3), (30, 20, 4, 5)):
        boundary_sol = float(MLABmm._query_mla_bmm_table(db, boundary_tokens, 4, quant, True, common.DatabaseMode.SOL))
        query_sol = float(MLABmm._query_mla_bmm_table(db, tokens, 4, quant, True, common.DatabaseMode.SOL))
        exterior = query(tokens)
        assert float(exterior) == boundary_latency * query_sol / boundary_sol
        assert exterior.energy == boundary_power * float(exterior)


def test_wideep_backend_categories_do_not_share_exact_values(monkeypatch) -> None:
    monkeypatch.setattr(WideEPGenerationMLA, "load_data", classmethod(lambda _cls, _db: None))
    db = _Database("sglang")
    kv = common.KVCacheQuantMode.bfloat16
    fmha = common.FMHAQuantMode.bfloat16
    db._wideep_generation_mla_data = _Data(
        {
            "flashinfer": {kv: {fmha: {4: {1: {16: _metric(2, 2)}}}}},
            "fa3": {kv: {fmha: {4: {1: {16: _metric(9, 3)}}}}},
        }
    )

    def query(backend: str) -> PerformanceResult:
        return WideEPGenerationMLA._query_wideep_generation_mla_table(
            db, 1, 16, 32, kv, fmha, attention_backend=backend, database_mode=common.DatabaseMode.SILICON
        )

    assert float(query("flashinfer")) == 2
    assert float(query("fa3")) == 9


def test_mla_module_multisource_loaders_keep_active_rows(tmp_path) -> None:
    headers = (
        "framework,version,device,op_name,kernel_source,model,architecture,"
        "mla_dtype,kv_cache_dtype,gemm_type,num_heads,batch_size,isl,tp_size,step,latency\n"
    )
    context_active = tmp_path / "context_active.txt"
    context_sibling = tmp_path / "context_sibling.txt"
    context_row = "TRTLLM,1,NVIDIA,mla_context_module,default,m,A,bfloat16,bfloat16,bfloat16,4,1,16,1,0,{latency}\n"
    context_active.write_text(headers + context_row.format(latency=2.0))
    context_sibling.write_text(headers + context_row.format(latency=9.0))

    generation_active = tmp_path / "generation_active.txt"
    generation_sibling = tmp_path / "generation_sibling.txt"
    generation_row = (
        "TRTLLM,1,NVIDIA,mla_generation_module,default,m,A,bfloat16,bfloat16,bfloat16,4,1,8,1,8,{latency}\n"
    )
    generation_active.write_text(headers + generation_row.format(latency=3.0))
    generation_sibling.write_text(headers + generation_row.format(latency=10.0))

    def sources(active, sibling):
        return [(str(active), None), (str(sibling), {"default"})]

    context = mla_module.load_context_mla_module_data(sources(context_active, context_sibling))
    generation = mla_module.load_generation_mla_module_data(sources(generation_active, generation_sibling))
    fmha = common.FMHAQuantMode.bfloat16
    kv = common.KVCacheQuantMode.bfloat16
    gemm = common.GEMMQuantMode.bfloat16

    assert context[fmha][kv][gemm][4][16][1]["latency"] == 2
    assert generation[fmha][kv][gemm][4][1][16]["latency"] == 3


def test_wideep_generation_loader_and_query_isolate_fmha_dtype(tmp_path, monkeypatch) -> None:
    perf_file = tmp_path / "wideep_generation_mla_perf.txt"
    headers = (
        "framework,version,device,op_name,kernel_source,model,architecture,"
        "mla_dtype,kv_cache_dtype,gemm_type,num_heads,batch_size,isl,tp_size,step,latency\n"
    )
    rows = (
        "SGLANG,1,NVIDIA,wideep_generation_mla,flashinfer,m,A,"
        "bfloat16,fp8,bfloat16,4,1,8,32,8,2.0\n"
        "SGLANG,1,NVIDIA,wideep_generation_mla,flashinfer,m,A,"
        "fp8_block,fp8,bfloat16,4,1,8,32,8,9.0\n"
    )
    perf_file.write_text(headers + rows)
    data = mla_module.load_wideep_generation_mla_data(str(perf_file))
    kv = common.KVCacheQuantMode.fp8
    assert set(data["flashinfer"]) == {kv}
    assert set(data["flashinfer"][kv]) == {
        common.FMHAQuantMode.bfloat16,
        common.FMHAQuantMode.fp8_block,
    }

    monkeypatch.setattr(WideEPGenerationMLA, "load_data", classmethod(lambda _cls, _db: None))
    db = _Database("sglang")
    db._wideep_generation_mla_data = _Data(data)

    def query(fmha: common.FMHAQuantMode) -> PerformanceResult:
        return WideEPGenerationMLA._query_wideep_generation_mla_table(
            db, 1, 16, 32, kv, fmha, attention_backend="flashinfer", database_mode=common.DatabaseMode.SILICON
        )

    assert float(query(common.FMHAQuantMode.bfloat16)) == 2
    assert float(query(common.FMHAQuantMode.fp8_block)) == 9
    caller_keys = {key[0] for key in db._sparse_surrogate_cache}
    assert ("wideep-generation-mla", "flashinfer", common.FMHAQuantMode.bfloat16, kv) in caller_keys
    assert ("wideep-generation-mla", "flashinfer", common.FMHAQuantMode.fp8_block, kv) in caller_keys


def test_wideep_flashinfer_explicitly_maps_to_trtllm_mla(monkeypatch) -> None:
    monkeypatch.setattr(WideEPContextMLA, "load_data", classmethod(lambda _cls, _db: None))
    monkeypatch.setattr(WideEPGenerationMLA, "load_data", classmethod(lambda _cls, _db: None))
    db = _Database("sglang")
    fmha = common.FMHAQuantMode.fp8_block
    kv = common.KVCacheQuantMode.fp8
    db._wideep_context_mla_data = _Data({"trtllm_mla": {fmha: {kv: {4: {16: {1: _metric(6, 2)}}}}}})
    db._wideep_generation_mla_data = _Data({"trtllm_mla": {kv: {fmha: {4: {1: {16: _metric(7, 3)}}}}}})

    context = WideEPContextMLA._query_wideep_context_mla_table(
        db, 1, 16, 0, 32, kv, fmha, attention_backend="flashinfer", database_mode=common.DatabaseMode.SILICON
    )
    generation = WideEPGenerationMLA._query_wideep_generation_mla_table(
        db, 1, 16, 32, kv, fmha, attention_backend=None, database_mode=common.DatabaseMode.SILICON
    )
    explicit = WideEPGenerationMLA._query_wideep_generation_mla_table(
        db, 1, 16, 32, kv, fmha, attention_backend="trtllm_mla", database_mode=common.DatabaseMode.SILICON
    )

    assert float(context) == 6
    assert float(generation) == 7
    assert float(explicit) == 7
    with pytest.raises(KeyError, match="fa3"):
        WideEPGenerationMLA._query_wideep_generation_mla_table(
            db, 1, 16, 32, kv, fmha, attention_backend="fa3", database_mode=common.DatabaseMode.SILICON
        )
