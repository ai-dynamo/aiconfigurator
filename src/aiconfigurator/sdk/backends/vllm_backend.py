# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging

import numpy as np

from aiconfigurator.sdk import common
from aiconfigurator.sdk.backends.base_backend import BaseBackend
from aiconfigurator.sdk.backends.trtllm_backend import TRTLLMBackend

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

    def _mix_step_gen_tokens(self, b: int, ctx_tokens: int, isl: int, osl: int) -> int:
        # vLLM v1 scheduler sets max_num_partial_prefills=1 by default, meaning
        # exactly one request is in partial-prefill state per forward pass.
        # The remaining b - ceil(ctx_tokens/isl) requests are in decode phase.
        # This applies regardless of whether steps_to_finish_ctx >= osl or not,
        # giving a consistent formula across both scheduling regimes.
        # Source: vllm/v1/core/sched/scheduler.py, SchedulerConfig.max_num_partial_prefills
        return max(1, b - int(np.ceil(ctx_tokens / isl)))
