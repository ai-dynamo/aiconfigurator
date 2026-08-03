# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aiconfigurator.sdk import common
from aiconfigurator.sdk.errors import PerfDataNotAvailableError
from aiconfigurator.sdk.perf_database import PerfDatabase, get_systems_paths

pytestmark = pytest.mark.unit


def _query_wideep_mla(system: str, attention_backend: str | None, phase: str):
    database = PerfDatabase(system, "sglang", "0.5.10", systems_root=get_systems_paths()[0])
    kwargs = {
        "b": 1,
        "s": 1024 if phase == "context" else 1,
        "tp_size": 1,
        "kvcache_quant_mode": common.KVCacheQuantMode.fp8,
        "fmha_quant_mode": common.FMHAQuantMode.fp8_block,
        "attention_backend": attention_backend,
    }
    if phase == "context":
        return database.query_wideep_context_mla(prefix=0, **kwargs)
    return database.query_wideep_generation_mla(**kwargs)


@pytest.mark.parametrize("phase", ["context", "generation"])
def test_blackwell_auto_uses_trtllm_mla_measurements(phase):
    result = _query_wideep_mla("b200_sxm", None, phase)

    assert float(result) > 0


@pytest.mark.parametrize("phase", ["context", "generation"])
@pytest.mark.parametrize("requested", ["flashinfer", "fa3"])
def test_blackwell_explicit_backend_does_not_fall_back_to_trtllm_mla(requested, phase):
    with pytest.raises(PerfDataNotAvailableError):
        _query_wideep_mla("b200_sxm", requested, phase)


@pytest.mark.parametrize("phase", ["context", "generation"])
def test_hopper_auto_uses_fa3_measurements(phase):
    result = _query_wideep_mla("h200_sxm", None, phase)

    assert float(result) > 0
