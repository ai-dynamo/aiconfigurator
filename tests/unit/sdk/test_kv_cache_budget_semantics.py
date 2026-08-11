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


def test_sglang_backend_derives_capacity_tiered_fraction():
    """SGLang's mem_fraction_static follows the framework's capacity-tiered derivation."""
    from aiconfigurator.sdk.backends.sglang_backend import (
        SGLANG_FALLBACK_MEM_FRACTION_STATIC,
        SGLANGBackend,
        derive_sglang_mem_fraction_static,
    )

    backend = SGLANGBackend()
    assert backend.memory_fraction_of_free() is False
    assert backend.get_default_free_gpu_memory_fraction() == SGLANG_FALLBACK_MEM_FRACTION_STATIC

    # b200_sxm (180 GiB), tp1: >160 GB tier, chunked_prefill 16384, graph max_bs 512
    # -> reserved 512 + 24576 + 128 (tp*pp/8 GiB) + 1024 (graph) = 26240
    # -> (184320 - 26240) / 184320 = 0.858
    b200_capacity = 193273528320
    assert derive_sglang_mem_fraction_static(b200_capacity) == pytest.approx(0.858, abs=0.001)
    kwargs = backend._static_oom_check_kwargs(b200_capacity)
    assert kwargs["free_gpu_memory_fraction"] == pytest.approx(0.858, abs=0.001)
    assert kwargs["fraction_of_free"] is False

    # DP attention adds max_bs*dp*3 (+ *1.5 above 300): DP=8 -> +18432 MB
    # -> reserved 44672 -> 0.758 (matches the framework-resolved ~0.76).
    assert derive_sglang_mem_fraction_static(b200_capacity, dp_size=8) == pytest.approx(0.758, abs=0.001)

    # h100_sxm (80 GiB), tp1: 60-160 GB tier, chunked_prefill 8192, max_bs 256
    # -> reserved max(512 + 12288 + 128 + 512, 10240) = 13440 -> 0.836
    assert derive_sglang_mem_fraction_static(80 * (1 << 30)) == pytest.approx(0.836, abs=0.001)

    # explicit user fraction always wins
    kwargs = backend._static_oom_check_kwargs(b200_capacity, free_gpu_memory_fraction=0.8)
    assert kwargs["free_gpu_memory_fraction"] == 0.8


def test_vllm_default_fraction_is_version_dependent():
    """vLLM's gpu_memory_utilization default changed 0.90 -> 0.92 at 0.22
    (vllm/config/cache.py @v0.14.0/v0.19.0 vs @v0.22.0)."""
    backend = VLLMBackend()
    assert backend.get_default_free_gpu_memory_fraction("0.19.0") == pytest.approx(0.90)
    assert backend.get_default_free_gpu_memory_fraction("0.22.0") == pytest.approx(0.92)
    assert backend.get_default_free_gpu_memory_fraction("0.24.0") == pytest.approx(0.92)
    assert backend.get_default_free_gpu_memory_fraction(None) == pytest.approx(0.92)


def test_explicit_fraction_overrides_backend_default():
    """A user-configured Task/estimate fraction wins over the backend default."""
    backend = VLLMBackend()
    kwargs = backend._static_oom_check_kwargs(180 * (1 << 30), free_gpu_memory_fraction=0.8)
    assert kwargs["free_gpu_memory_fraction"] == 0.8
    kwargs = backend._static_oom_check_kwargs(180 * (1 << 30), backend_version="0.19.0")
    assert kwargs["free_gpu_memory_fraction"] == pytest.approx(0.90)


def test_vllm_agg_resolver_carries_version_aware_fraction():
    """The agg resolver must not drop the fraction: run_agg budgets through
    ``_oom_check_kwargs(agg_extra)``, and before this resolver existed the
    base class returned {} — every aggregated vLLM run was budgeted at the
    0.92 constant regardless of database version or explicit override."""
    backend = VLLMBackend()
    assert backend._resolve_agg_kwargs({}, isl=1024, osl=1024, backend_version="0.19.0") == {
        "free_gpu_memory_fraction": pytest.approx(0.90)
    }
    assert backend._resolve_agg_kwargs({}, isl=1024, osl=1024, backend_version="0.22.0") == {
        "free_gpu_memory_fraction": pytest.approx(0.92)
    }
    # explicit override wins over the version default
    resolved = backend._resolve_agg_kwargs(
        {"free_gpu_memory_fraction": 0.75}, isl=1024, osl=1024, backend_version="0.22.0"
    )
    assert resolved == {"free_gpu_memory_fraction": 0.75}
    # idempotent: re-resolving forwarded kwargs returns the same values
    assert backend._resolve_agg_kwargs(resolved, isl=1024, osl=1024, backend_version="0.22.0") == resolved
    # and the resolved value is what the agg OOM check budgets with
    oom_kwargs = backend._oom_check_kwargs(
        backend._resolve_agg_kwargs({}, isl=1024, osl=1024, backend_version="0.19.0")
    )
    assert oom_kwargs["free_gpu_memory_fraction"] == pytest.approx(0.90)
    assert oom_kwargs["fraction_of_free"] is False


def test_sglang_agg_resolver_honors_explicit_fraction():
    """SGLang agg keeps its 0.88 fallback when unset but must honor an
    explicit user fraction (it was previously discarded the same way)."""
    from aiconfigurator.sdk.backends.sglang_backend import (
        SGLANG_FALLBACK_MEM_FRACTION_STATIC,
        SGLANGBackend,
    )

    backend = SGLANGBackend()
    resolved = backend._resolve_agg_kwargs({}, isl=1024, osl=1024)
    assert backend._oom_check_kwargs(resolved)["free_gpu_memory_fraction"] == SGLANG_FALLBACK_MEM_FRACTION_STATIC
    resolved = backend._resolve_agg_kwargs({"free_gpu_memory_fraction": 0.8}, isl=1024, osl=1024)
    assert backend._oom_check_kwargs(resolved)["free_gpu_memory_fraction"] == 0.8


def test_agg_cache_key_distinguishes_fraction():
    """The cached run_agg summary embeds the KV-budget OOM verdict, so
    summaries resolved under different fractions must not share an entry."""
    backend = VLLMBackend()
    key_090 = backend._make_agg_cache_key(
        1024, 1024, 8, 512, "python", backend._resolve_agg_kwargs({}, isl=1024, osl=1024, backend_version="0.19.0")
    )
    key_092 = backend._make_agg_cache_key(
        1024, 1024, 8, 512, "python", backend._resolve_agg_kwargs({}, isl=1024, osl=1024, backend_version="0.22.0")
    )
    assert key_090 != key_092


def test_base_backend_agg_resolver_defaults_off():
    """Backends without a default fraction resolve to {} (no budget check)."""

    class _NoDefault(BaseBackend):
        pass

    assert _NoDefault()._resolve_agg_kwargs({}, isl=1024, osl=1024) == {}
