# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging

import numpy as np

from aiconfigurator_core.sdk import common
from aiconfigurator_core.sdk.backends.base_backend import BaseBackend
from aiconfigurator_core.sdk.backends.trtllm_backend import TRTLLMBackend
from aiconfigurator_core.sdk.models import BaseModel

logger = logging.getLogger(__name__)

# vLLM's ``gpu_memory_utilization`` default: the engine only manages this
# fraction of TOTAL device memory (weights + activations + CUDA graphs + KV
# all fit inside it). Without this cap the sweep packs memory to 100% of the
# device and selects decode batch sizes whose KV cache cannot be allocated in
# a real deployment (see ai-dynamo/aiconfigurator#1396: predicted decode
# bs=96/rank vs 29/rank actually admitted on B200 for Kimi-K2.5-NVFP4).
# Values mirror the framework default (CacheConfig.gpu_memory_utilization,
# vllm/config/cache.py: 0.9 @v0.14.0/v0.19.0; 0.92 @v0.22.0, unchanged
# @v0.24.0/v0.25.x). Keyed by minor-version prefix; unknown versions fall
# through to the newest known default.
VLLM_GPU_MEMORY_UTILIZATION_BY_VERSION: dict[str, float] = {
    "0.14": 0.9,
    "0.19": 0.9,
    "0.22": 0.92,
    "0.24": 0.92,
}
VLLM_DEFAULT_GPU_MEMORY_UTILIZATION: float = 0.92
# Small safety margin for KV block/page granularity and memory-profiler
# variance between vLLM releases.
KV_CACHE_MEMORY_TOLERANCE: float = 0.02


class VLLMBackend(BaseBackend):
    """vLLM backend.

    Currently mirrors TRT-LLM's activation-memory model (the pre-refactor
    implementation literally delegated ``_get_memory_usage`` to TRTLLMBackend),
    reusing both TRT-LLM's per-family coefficient table and its
    ``_moe_workspace_width`` hook so estimates stay byte-identical with the
    old delegation.

    KV-cache OOM accounting: vLLM caps its whole footprint at
    ``gpu_memory_utilization`` of TOTAL device memory (``of_total``
    semantics, unlike TRT-LLM's ``free_gpu_memory_fraction`` which applies to
    the free pool after weights). The overrides below wire that budget into
    both the static (disagg component) and agg OOM checks.
    """

    # Reuse TRT-LLM's per-family activation coefficients until a vLLM-specific
    # tuning lands.
    ACTIVATION_COEFFICIENTS = TRTLLMBackend.ACTIVATION_COEFFICIENTS

    # Mirror TRT-LLM's MoE workspace accounting (raw h for DEEPSEEK family,
    # ``_hidden_size`` for GEMMA4MIX). Plain class-attribute alias to the
    # function object — Python binds it to the VLLMBackend instance at call
    # time; the function does not touch any TRTLLMBackend-specific state.
    _moe_workspace_width = TRTLLMBackend._moe_workspace_width

    def __init__(self):
        super().__init__()
        self.name = common.BackendName.vllm

    def get_default_free_gpu_memory_fraction(self, backend_version: str | None = None) -> float | None:
        if backend_version:
            for prefix, value in VLLM_GPU_MEMORY_UTILIZATION_BY_VERSION.items():
                if str(backend_version).startswith(prefix):
                    return value
        return VLLM_DEFAULT_GPU_MEMORY_UTILIZATION

    def get_kv_cache_memory_check_params(self) -> tuple[float, float]:
        return 0.0, KV_CACHE_MEMORY_TOLERANCE

    def memory_fraction_of_free(self) -> bool:
        # gpu_memory_utilization caps TOTAL device memory, not the free pool.
        return False

    def _oom_check_kwargs(self, agg_extra: dict) -> dict:
        fraction = agg_extra.get("free_gpu_memory_fraction")
        if fraction is None:
            fraction = VLLM_DEFAULT_GPU_MEMORY_UTILIZATION
        return {
            "free_gpu_memory_fraction": fraction,
            "kv_cache_reserved_fraction": 0.0,
            "kv_cache_tolerance": KV_CACHE_MEMORY_TOLERANCE,
            "fraction_of_free": False,
        }

    def _mix_step_efficiency(self, ctx_tokens: int, gen_tokens: int) -> float:
        # vLLM v1 serialises prefill (max_num_partial_prefills=1): each mix step
        # processes one request's full ISL alongside a handful of decode tokens
        # from other requests. With gen_frac = (b-1)/ISL ≈ 0.001 at typical
        # operating points, the base-class power-law formula extrapolates to
        # ~0.19 — an 80% reduction with no physical basis. Full-corpus analysis
        # (1928 vLLM agg entries) shows median implied efficiency of 1.115,
        # confirming the base-class formula is inapplicable to this regime.
        # Return 1.0: no correction applied for this backend.
        return 1.0

    def _mix_step_gen_tokens(self, b: int, ctx_tokens: int, isl: int, osl: int) -> int:
        # vLLM v1 scheduler sets max_num_partial_prefills=1 by default, meaning
        # exactly one request is in partial-prefill state per forward pass.
        # The remaining b - ceil(ctx_tokens/isl) requests are in decode phase.
        # This applies regardless of whether steps_to_finish_ctx >= osl or not,
        # giving a consistent formula across both scheduling regimes.
        # Source: vllm/v1/core/sched/scheduler.py, SchedulerConfig.max_num_partial_prefills
        return max(1, b - int(np.ceil(ctx_tokens / isl)))

    def _prefill_dispatch_overhead_ms(self, model: BaseModel) -> float:
        # CPU-side dispatch overhead scales with layer count and is not captured
        # in silicon benchmarks. Recalibrated at ~0.8ms/layer against the full
        # silicon corpus across hardware platforms and model families.
        return model._num_layers * 0.8

    def _ttft_queuing_factor(self, b: int, steps_to_finish_ctx: float) -> float:
        # vLLM v1 serialises prefill (max_num_partial_prefills=1): requests queue
        # behind the active prefill, so TTFT grows with concurrency. In steady
        # state, growth is sub-linear — calibrated to the silicon corpus
        # (tp_size-matched vLLM agg entries, b=1..64) as log_256(b), which
        # improves MAPE from 26.4% (no correction) to 18.0% overall.
        # Formula: 1 + log2(b)/8, capped at 2xT_prefill (saturates at b=256).
        # A principled M/D/1 treatment (requiring T_decode input) is a follow-on.
        if b <= 1:
            return 1.0
        return float(min(1.0 + np.log2(b) / 8.0, 2.0))

    def _throughput_cap(self, step_throughput: float, ttft: float, tpot: float, b: int, osl: int) -> float:
        # Cap throughput at the Little's Law limit: b concurrent requests each
        # taking (ttft + tpot*(osl-1)) ms cannot sustain more than
        # b*(osl-1)*1000 / request_latency_ms output tokens/s in steady state.
        request_latency_ms = ttft + tpot * max(osl - 1, 0)
        if request_latency_ms <= 0:
            return step_throughput
        ll_throughput = b * max(osl - 1, 0) * 1000.0 / request_latency_ms
        return min(step_throughput, ll_throughput)
