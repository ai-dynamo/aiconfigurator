# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract coverage for WideEP MLA backend-to-kernel-source resolution."""

import pytest

from aiconfigurator.sdk import common
from aiconfigurator.sdk.operations import mla

pytestmark = pytest.mark.unit


def _query_empirical(database, query_kind: str, attention_backend: str | None):
    common_args = (
        2,
        128,
    )
    if query_kind == "generation":
        return database.query_wideep_generation_mla(
            *common_args,
            8,
            common.KVCacheQuantMode.bfloat16,
            common.FMHAQuantMode.bfloat16,
            attention_backend=attention_backend,
            database_mode=common.DatabaseMode.EMPIRICAL,
        )
    return database.query_wideep_context_mla(
        *common_args,
        0,
        8,
        common.KVCacheQuantMode.bfloat16,
        common.FMHAQuantMode.bfloat16,
        attention_backend=attention_backend,
        database_mode=common.DatabaseMode.EMPIRICAL,
    )


@pytest.fixture
def blackwell_empirical_table(monkeypatch, comprehensive_perf_db):
    """Expose a trtllm_mla-only table and record the resolved grid key."""
    monkeypatch.setattr(comprehensive_perf_db, "_wideep_generation_mla_data", {"trtllm_mla": {}}, raising=False)
    monkeypatch.setattr(comprehensive_perf_db, "_wideep_context_mla_data", {"trtllm_mla": {}}, raising=False)
    monkeypatch.setattr(mla.WideEPGenerationMLA, "load_data", classmethod(lambda cls, database: None))
    monkeypatch.setattr(mla.WideEPContextMLA, "load_data", classmethod(lambda cls, database: None))

    resolved_sources: list[str] = []

    def grid_for(cache_key, *args, **kwargs):
        resolved_sources.append(cache_key[4])
        return object()

    monkeypatch.setattr(mla.util_empirical, "grid_for", grid_for)
    monkeypatch.setattr(mla.util_empirical, "estimate", lambda *args, **kwargs: (1.0, None))
    return comprehensive_perf_db, resolved_sources


def test_user_backend_order_matches_task_contract():
    assert mla._WIDEEP_MLA_ATTENTION_BACKENDS == ("flashinfer", "fa3")


def test_exact_measured_kernel_source_remains_valid():
    assert mla._resolve_wideep_mla_kernel_source({"custom_kernel": {}}, "custom_kernel") == "custom_kernel"


@pytest.mark.parametrize("query_kind", ["generation", "context"])
@pytest.mark.parametrize("attention_backend", [None, "flashinfer", "fa3", "trtllm_mla"])
def test_empirical_resolves_supported_backends_to_blackwell_kernel_source(
    blackwell_empirical_table, query_kind, attention_backend
):
    database, resolved_sources = blackwell_empirical_table

    result = _query_empirical(database, query_kind, attention_backend)

    assert float(result) == 1.0
    assert result.source == "empirical"
    assert resolved_sources == ["trtllm_mla"]


@pytest.mark.parametrize("query_kind", ["generation", "context"])
@pytest.mark.parametrize("attention_backend", ["", "torch"])
def test_empirical_rejects_unknown_backend_without_borrowing_blackwell_data(
    blackwell_empirical_table, query_kind, attention_backend
):
    database, resolved_sources = blackwell_empirical_table

    with pytest.raises(ValueError, match="match an available kernel_source"):
        _query_empirical(database, query_kind, attention_backend)

    assert resolved_sources == []
