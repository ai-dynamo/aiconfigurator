# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for KV-cache budget semantics (of_free vs of_total).

TRT-LLM's ``free_gpu_memory_fraction`` applies to the FREE pool after non-KV
allocations; vLLM's ``gpu_memory_utilization`` caps TOTAL device memory before
subtracting non-KV. Mixing them up passes memory-infeasible configs through
the sweep (ai-dynamo/aiconfigurator#1396: a decode config needing 3.3x the
actually-allocatable KV was selected as top1).
"""

import pytest

from aiconfigurator.sdk.backends.base_backend import BaseBackend
from aiconfigurator.sdk.backends.vllm_backend import (
    VLLM_DEFAULT_GPU_MEMORY_UTILIZATION,
    VLLMBackend,
)
from aiconfigurator.sdk.config import RuntimeConfig
from aiconfigurator.sdk.inference_summary import InferenceSummary
from aiconfigurator.sdk.memory import kv_cache_budget_bytes

pytestmark = pytest.mark.unit

_GIB = 1 << 30


def test_budget_semantics_differ():
    """of_free applies the fraction to (capacity - non_kv); of_total to capacity."""
    of_free = kv_cache_budget_bytes(capacity=100.0, non_kv=50.0, fraction=0.9, of_free=True)
    of_total = kv_cache_budget_bytes(capacity=100.0, non_kv=50.0, fraction=0.9, of_free=False)
    assert of_free == pytest.approx(45.0)
    assert of_total == pytest.approx(40.0)


def _summary_with(kv_gib: float, non_kv_gib: float, capacity_gib: float, fraction_of_free: bool) -> InferenceSummary:
    summary = InferenceSummary(RuntimeConfig(isl=1024, osl=1024))
    memory = {"kvcache": kv_gib, "total": kv_gib + non_kv_gib}
    summary.set_memory_and_check_oom(
        memory,
        int(capacity_gib * _GIB),
        free_gpu_memory_fraction=0.9,
        fraction_of_free=fraction_of_free,
    )
    return summary


def test_kv_oom_depends_on_fraction_semantics():
    """A KV size between the two budgets flags OOM only under of_total.

    capacity=100, non_kv=50, fraction=0.9: of_free budget = 45, of_total
    budget = 40. kv=42 sits between the two.
    """
    assert not _summary_with(42.0, 50.0, 100.0, fraction_of_free=True).check_kv_cache_oom()
    assert _summary_with(42.0, 50.0, 100.0, fraction_of_free=False).check_kv_cache_oom()


def test_vllm_backend_enables_static_kv_budget_check():
    """VLLMBackend provides of_total defaults so the static path checks KV budgets."""
    backend = VLLMBackend()
    assert backend.get_default_free_gpu_memory_fraction() == VLLM_DEFAULT_GPU_MEMORY_UTILIZATION
    assert backend.memory_fraction_of_free() is False
    kwargs = backend._static_oom_check_kwargs()
    assert kwargs["free_gpu_memory_fraction"] == VLLM_DEFAULT_GPU_MEMORY_UTILIZATION
    assert kwargs["fraction_of_free"] is False


def test_base_backend_static_check_defaults_off():
    """Backends without a default fraction keep the pre-existing behavior (no budget check)."""

    class _NoDefault(BaseBackend):
        pass

    assert _NoDefault()._static_oom_check_kwargs() == {}
