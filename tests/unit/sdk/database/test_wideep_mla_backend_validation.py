# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validation coverage for WideEP MLA empirical table selection."""

import pytest

from aiconfigurator.sdk import common
from aiconfigurator.sdk.operations import mla

pytestmark = pytest.mark.unit


@pytest.fixture
def blackwell_empirical_table(monkeypatch, comprehensive_perf_db):
    """Expose a plausible trtllm_mla fallback without reading table files."""
    monkeypatch.setattr(comprehensive_perf_db, "_wideep_generation_mla_data", {"trtllm_mla": {}}, raising=False)
    monkeypatch.setattr(comprehensive_perf_db, "_wideep_context_mla_data", {"trtllm_mla": {}}, raising=False)
    monkeypatch.setattr(mla.WideEPGenerationMLA, "load_data", classmethod(lambda cls, database: None))
    monkeypatch.setattr(mla.WideEPContextMLA, "load_data", classmethod(lambda cls, database: None))
    monkeypatch.setattr(mla.util_empirical, "grid_for", lambda *args, **kwargs: object())
    monkeypatch.setattr(mla.util_empirical, "estimate", lambda *args, **kwargs: (1.0, None))
    return comprehensive_perf_db


def test_generation_empirical_rejects_unsupported_attention_backend(blackwell_empirical_table):
    with pytest.raises(ValueError, match="Unsupported attention backend: torch"):
        blackwell_empirical_table.query_wideep_generation_mla(
            2,
            128,
            8,
            common.KVCacheQuantMode.bfloat16,
            common.FMHAQuantMode.bfloat16,
            attention_backend="torch",
            database_mode=common.DatabaseMode.EMPIRICAL,
        )


def test_context_empirical_rejects_unsupported_attention_backend(blackwell_empirical_table):
    with pytest.raises(ValueError, match="Unsupported attention backend: torch"):
        blackwell_empirical_table.query_wideep_context_mla(
            2,
            128,
            0,
            8,
            common.KVCacheQuantMode.bfloat16,
            common.FMHAQuantMode.bfloat16,
            attention_backend="torch",
            database_mode=common.DatabaseMode.EMPIRICAL,
        )
