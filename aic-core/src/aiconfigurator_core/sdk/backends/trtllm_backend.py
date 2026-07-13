# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import ClassVar

from aiconfigurator_core.sdk import common
from aiconfigurator_core.sdk.backends.base_backend import BaseBackend
from aiconfigurator_core.sdk.models import BaseModel

logger = logging.getLogger(__name__)

# Fraction of available KV cache memory assumed to be reserved by TRT-LLM
# for internal block-allocator overhead.  Applied in production to make the
# KV OOM check conservative; set to 0 in tests to validate the raw formula.
KV_CACHE_MEMORY_RESERVED_FRACTION: float = 0.015

# Acceptable formula error relative to real TRT-LLM benchmark measurements.
# Used as the %-based tolerance band in KV cache capacity tests.
KV_CACHE_MEMORY_TOLERANCE: float = 0.02

# Default fraction of free GPU memory that TRT-LLM allocates for KV cache.
TRTLLM_DEFAULT_FREE_GPU_MEMORY_FRACTION: float = 0.9

# Default max_num_tokens for TRT-LLM engine builds (BuildConfig.max_num_tokens).
# Determines activation memory pre-allocated at engine build time.
TRTLLM_DEFAULT_MAX_NUM_TOKENS: int = 8192


class TRTLLMBackend(BaseBackend):
    """TRTLLM backend.

    Adds KV-cache-aware OOM accounting on top of BaseBackend: ``run_agg`` and
    ``find_best_agg_result_under_constraints`` accept ``max_seq_len``,
    ``max_num_tokens``, and ``free_gpu_memory_fraction`` kwargs that flow into
    the cache key, the memory sizing, and the OOM probe.
    """

    # Per-family activation scaling coefficients tuned for TRT-LLM's
    # measurement workflow. FIXME: based on TRT-LLM measurement with
    # traditional MoE; fine-grained MoE may need re-study.
    ACTIVATION_COEFFICIENTS: ClassVar[dict[str, dict[int, float]]] = {
        "GPT": {1: 10, 2: 6, 4: 5, 8: 5},
        "LLAMA": {1: 11, 2: 6.5, 4: 5, 8: 5},
        "MOE": {1: 22, 2: 13, 4: 10, 8: 10},
        "GEMMA4MIX": {1: 22, 2: 13, 4: 10, 8: 10},
        "DEEPSEEK": {1: 22, 2: 13, 4: 10, 8: 10},
        "DEEPSEEKV32": {1: 22, 2: 13, 4: 10, 8: 10},
        "DEEPSEEKV4": {1: 22, 2: 13, 4: 10, 8: 10},
        "KIMIK25": {1: 22, 2: 13, 4: 10, 8: 10},
        # 4+6/TP, fp8 will have relatively low act, but ignore here. need more experiments
        "default": {1: 10, 2: 6, 4: 5, 8: 5},
    }

    def __init__(self):
        super().__init__()
        self.name = common.BackendName.trtllm

    def _moe_workspace_width(self, model: BaseModel, model_family: str, h: int) -> int:
        # TRT-LLM uses ``_hidden_size`` for GEMMA4MIX but keeps the raw ``h``
        # for the DEEPSEEK family — this is an existing TRT-LLM accounting
        # quirk predating DeepSeek-V4's attention expansion. Still an
        # improvement opportunity in trtllm to align with SGLang's accounting.
        if model_family == "GEMMA4MIX":
            return getattr(model, "_hidden_size", h)
        return h

    def _tpot_mix_steps(self, num_mix_steps: int) -> int:
        # TRT-LLM in-flight batching has a pipeline-drain latency at the
        # context/decode boundary: ~3 steps elapse before new requests can be
        # enqueued after the last prefill finishes. Empirical correction.
        return max(1, num_mix_steps - 3)

    def get_default_free_gpu_memory_fraction(self) -> float | None:
        return TRTLLM_DEFAULT_FREE_GPU_MEMORY_FRACTION

    def get_kv_cache_memory_check_params(self) -> tuple[float, float]:
        return KV_CACHE_MEMORY_RESERVED_FRACTION, KV_CACHE_MEMORY_TOLERANCE

    def _resolve_agg_kwargs(self, kwargs: dict, isl: int, osl: int) -> dict:
        # Use ``if x is None`` (rather than kwargs.get default) so that an
        # explicit None from the Python API still falls back to the constant.
        max_seq_len = kwargs.get("max_seq_len")
        if max_seq_len is None:
            # KV cache must hold isl + beam_width * osl tokens per slot; the
            # plain isl + osl default under-sizes the cache when beam_width > 1.
            try:
                beam_width = int(kwargs.get("beam_width", 1))
            except (TypeError, ValueError):
                beam_width = 1
            max_seq_len = isl + beam_width * osl
        free_gpu_memory_fraction = kwargs.get("free_gpu_memory_fraction")
        if free_gpu_memory_fraction is None:
            free_gpu_memory_fraction = TRTLLM_DEFAULT_FREE_GPU_MEMORY_FRACTION
        max_num_tokens = kwargs.get("max_num_tokens")
        if max_num_tokens is None:
            max_num_tokens = TRTLLM_DEFAULT_MAX_NUM_TOKENS
        return {
            "max_seq_len": max_seq_len,
            "free_gpu_memory_fraction": free_gpu_memory_fraction,
            "max_num_tokens": max_num_tokens,
        }

    def _make_agg_cache_key(
        self,
        isl: int,
        osl: int,
        b: int,
        ctx_tokens: int,
        engine_step_backend_key: str,
        agg_extra: dict,
    ) -> tuple:
        return (
            isl,
            osl,
            b,
            ctx_tokens,
            engine_step_backend_key,
            agg_extra["max_seq_len"],
            agg_extra["max_num_tokens"],
            agg_extra["free_gpu_memory_fraction"],
        )

    def _memory_usage_kwargs_for_agg(self, num_tokens: int, agg_extra: dict) -> dict:
        # Activation memory tracks BuildConfig.max_num_tokens, not the agg-derived
        # num_tokens. KV cache tracks max_seq_len per slot.
        return {
            "num_tokens": agg_extra["max_num_tokens"],
            "max_seq_len": agg_extra["max_seq_len"],
        }

    def _oom_check_kwargs(self, agg_extra: dict) -> dict:
        return {
            "free_gpu_memory_fraction": agg_extra["free_gpu_memory_fraction"],
            "kv_cache_reserved_fraction": KV_CACHE_MEMORY_RESERVED_FRACTION,
            "kv_cache_tolerance": KV_CACHE_MEMORY_TOLERANCE,
        }
