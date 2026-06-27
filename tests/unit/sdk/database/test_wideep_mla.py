# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

import pytest

from aiconfigurator.sdk import common
from aiconfigurator.sdk.operations import mla as mla_ops
from aiconfigurator.sdk.performance_result import PerformanceResult

pytestmark = pytest.mark.unit


class _LoadedData(dict):
    def raise_if_not_loaded(self):
        return None


def _database():
    database = MagicMock()
    database._default_database_mode = common.DatabaseMode.SILICON
    database._extracted_metrics_cache = {}
    database._query_silicon_or_hybrid.side_effect = lambda **kwargs: kwargs["get_silicon"]()
    database._interp_pr.side_effect = lambda latency, energy=0.0: PerformanceResult(latency, energy=energy)
    return database


def test_wideep_context_accepts_blackwell_trtllm_mla_kernel(monkeypatch):
    database = _database()
    leaf = {32: {64: {2: {"latency": 1.25, "energy": 2.5}}}}
    database._wideep_context_mla_data = _LoadedData(
        {
            "trtllm_mla": {
                common.FMHAQuantMode.bfloat16: {
                    common.KVCacheQuantMode.fp8: leaf,
                }
            }
        }
    )
    monkeypatch.setattr(
        mla_ops.WideEPContextMLA,
        "load_data",
        classmethod(lambda cls, db: None),
    )

    def interp_3d(x, y, z, data, *_args, **_kwargs):
        assert (x, y, z) == (32, 64, 2)
        assert data is leaf
        return {"latency": 1.25, "energy": 2.5}

    monkeypatch.setattr(mla_ops.interpolation, "interp_3d", interp_3d)
    result = mla_ops.WideEPContextMLA._query_wideep_context_mla_table(
        database,
        b=2,
        s=64,
        prefix=0,
        tp_size=4,
        kvcache_quant_mode=common.KVCacheQuantMode.fp8,
        fmha_quant_mode=common.FMHAQuantMode.bfloat16,
        attention_backend="trtllm_mla",
        database_mode=common.DatabaseMode.SILICON,
    )

    assert float(result) == pytest.approx(1.25)
    assert result.energy == pytest.approx(2.5)


def test_wideep_generation_accepts_blackwell_trtllm_mla_kernel(monkeypatch):
    database = _database()
    leaf = {32: {2: {64: {"latency": 0.75, "energy": 1.5}}}}
    database._wideep_generation_mla_data = _LoadedData({"trtllm_mla": {common.KVCacheQuantMode.fp8: leaf}})
    monkeypatch.setattr(
        mla_ops.WideEPGenerationMLA,
        "load_data",
        classmethod(lambda cls, db: None),
    )

    def interp_3d(x, y, z, data, *_args, **_kwargs):
        assert (x, y, z) == (32, 2, 64)
        assert data is leaf
        return {"latency": 0.75, "energy": 1.5}

    monkeypatch.setattr(mla_ops.interpolation, "interp_3d", interp_3d)
    result = mla_ops.WideEPGenerationMLA._query_wideep_generation_mla_table(
        database,
        b=2,
        s=64,
        tp_size=4,
        kvcache_quant_mode=common.KVCacheQuantMode.fp8,
        fmha_quant_mode=common.FMHAQuantMode.bfloat16,
        attention_backend="trtllm_mla",
        database_mode=common.DatabaseMode.SILICON,
    )

    assert float(result) == pytest.approx(0.75)
    assert result.energy == pytest.approx(1.5)
