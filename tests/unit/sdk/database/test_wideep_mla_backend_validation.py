# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Python-shell coverage for WideEP MLA backend-to-kernel-source resolution."""

import pytest

from aiconfigurator.sdk import common
from aiconfigurator.sdk.operations import WideEPContextMLA, WideEPGenerationMLA
from aiconfigurator.sdk.perf_database import get_database_view

pytestmark = pytest.mark.unit

_DEFAULT_BACKEND = object()


@pytest.fixture(scope="module")
def blackwell_empirical_database():
    """Use the shipped B200 campaign whose WideEP MLA key is trtllm_mla."""
    database = get_database_view(
        "b200_sxm",
        "sglang",
        "0.5.10",
        database_mode=common.DatabaseMode.EMPIRICAL,
        shared_layer=False,
        allow_unlisted_version=True,
    )
    assert database is not None
    return database


def _operation(query_kind: str, attention_backend):
    operation_cls = WideEPGenerationMLA if query_kind == "generation" else WideEPContextMLA
    args = (
        f"wideep_{query_kind}_mla",
        1.0,
        8,
        common.KVCacheQuantMode.fp8,
        common.FMHAQuantMode.fp8_block,
    )
    if attention_backend is _DEFAULT_BACKEND:
        return operation_cls(*args)
    return operation_cls(*args, attention_backend)


def _query_empirical(database, query_kind: str, attention_backend):
    operation = _operation(query_kind, attention_backend)
    kwargs = {"batch_size": 2, "s": 4096}
    if query_kind == "context":
        kwargs["prefix"] = 0
    return operation._engine_query(database, **kwargs)


@pytest.mark.parametrize("query_kind", ["generation", "context"])
def test_default_and_aliases_resolve_to_blackwell_kernel_source(blackwell_empirical_database, query_kind):
    results = [
        _query_empirical(blackwell_empirical_database, query_kind, backend)
        for backend in (_DEFAULT_BACKEND, "flashinfer", "fa3", "trtllm_mla")
    ]

    assert all(result.source == "empirical" for result in results)
    assert [float(result) for result in results] == pytest.approx([float(results[0])] * len(results))


@pytest.mark.parametrize("query_kind", ["generation", "context"])
@pytest.mark.parametrize("attention_backend", ["", "torch"])
def test_unknown_backend_cannot_borrow_blackwell_data(blackwell_empirical_database, query_kind, attention_backend):
    with pytest.raises(ValueError, match="attention_backend"):
        _query_empirical(blackwell_empirical_database, query_kind, attention_backend)
