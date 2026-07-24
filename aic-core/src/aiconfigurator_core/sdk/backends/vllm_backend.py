# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging

import numpy as np

from aiconfigurator_core.sdk import common
from aiconfigurator_core.sdk.backends.base_backend import BaseBackend
from aiconfigurator_core.sdk.backends.trtllm_backend import TRTLLMBackend
from aiconfigurator_core.sdk.models import BaseModel

logger = logging.getLogger(__name__)


class VLLMBackend(BaseBackend):
    """vLLM backend.

    Currently mirrors TRT-LLM's activation-memory model (the pre-refactor
    implementation literally delegated ``_get_memory_usage`` to TRTLLMBackend),
    with no KV-cache-aware OOM accounting yet. We reuse both TRT-LLM's
    per-family coefficient table and its ``_moe_workspace_width`` hook so
    estimates stay byte-identical with the old delegation; the agg-pipeline
    hooks (``_resolve_agg_kwargs``, ``_oom_check_kwargs``, ...) remain at
    BaseBackend defaults — vLLM does not yet do KV-cache OOM probing.
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

    def _mix_step_gen_tokens(self, b: int, ctx_tokens: int, isl: int, decode_iterations: float) -> int:
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
