# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import ClassVar

from aiconfigurator_core.sdk import common
from aiconfigurator_core.sdk.backends.base_backend import BaseBackend

logger = logging.getLogger(__name__)

# SGLang derives ``mem_fraction_static`` from device capacity at startup
# (ServerArgs._handle_gpu_memory_settings, python/sglang/srt/server_args.py
# @0.5.16): reserved = 512 MB + 1.5 MB x chunked_prefill_size (itself tiered
# by capacity), plus parallel/cuda-graph/DeepEP slack, floored at 10 GiB on
# >60 GB GPUs; fraction = (capacity - reserved) / capacity. When capacity is
# unknown the framework falls back to 0.88 (same file). The fraction is
# of_total: it caps weights + KV pool against TOTAL device memory.
SGLANG_FALLBACK_MEM_FRACTION_STATIC: float = 0.88
KV_CACHE_MEMORY_TOLERANCE: float = 0.02


def derive_sglang_mem_fraction_static(
    mem_capacity_bytes: int,
    tp_size: int = 1,
    pp_size: int = 1,
    dp_size: int = 1,
) -> float:
    """SGLang's capacity- and config-tiered default ``mem_fraction_static``.

    Mirrors ServerArgs._handle_gpu_memory_settings + reserve_for_graph_mb
    (server_args.py @0.5.16): reserved = 512 MB + 1.5 MB x chunked_prefill
    (capacity tier) + tp*pp/8 GiB parallel slack + decode-graph max_bs x 2 MB,
    plus the DP-attention padding max_bs x dp x 3 MB (and a further x1.5 term
    when max_bs > 300) — at DP=8 on a 180 GiB device the DP terms alone are
    ~18 GiB, so they cannot be omitted. Floored at 10 GiB on >60 GB GPUs.
    DeepEP all-to-all reserves are moe-backend-specific and remain omitted.
    """
    capacity_mb = mem_capacity_bytes / (1 << 20)
    if capacity_mb < 20 * 1024:
        chunked_prefill, graph_max_bs = 2048, 8
    elif capacity_mb < 35 * 1024:
        chunked_prefill, graph_max_bs = 2048, (24 if tp_size < 4 else 80)
    elif capacity_mb < 60 * 1024:
        chunked_prefill, graph_max_bs = 4096, (32 if tp_size < 4 else 160)
    elif capacity_mb < 160 * 1024:
        chunked_prefill, graph_max_bs = 8192, (256 if tp_size < 4 else 512)
    else:
        chunked_prefill, graph_max_bs = 16384, 512
    reserved_mb = 512 + 1.5 * chunked_prefill
    reserved_mb += tp_size * pp_size / 8 * 1024
    reserved_mb += graph_max_bs * 2
    if dp_size > 1:
        # DP attention padding (reserve_for_graph_mb @0.5.16).
        reserved_mb += graph_max_bs * dp_size * 3
        if graph_max_bs > 300:
            reserved_mb += graph_max_bs * dp_size * 1.5
    if capacity_mb > 60 * 1024:
        reserved_mb = max(reserved_mb, 10 * 1024)
    return round((capacity_mb - reserved_mb) / capacity_mb, 3)


class SGLANGBackend(BaseBackend):
    """SGLANG backend.

    Carries higher activation/system overheads than TRT-LLM to reflect SGLANG's
    Python execution overhead, plus a larger minimum activation budget.
    """

    # Per-family activation scaling tuned for SGLANG (Python overhead +
    # dynamic execution => higher coefficients than TRT-LLM).
    ACTIVATION_COEFFICIENTS: ClassVar[dict[str, dict[int, float]]] = {
        "GPT": {1: 13, 2: 8, 4: 6.5, 8: 6.5},
        "LLAMA": {1: 14, 2: 8.5, 4: 6.5, 8: 6.5},
        "MOE": {1: 28, 2: 17, 4: 13, 8: 13},
        "GEMMA4MIX": {1: 28, 2: 17, 4: 13, 8: 13},
        "STEP3P7": {1: 28, 2: 17, 4: 13, 8: 13},
        "DEEPSEEK": {1: 28, 2: 17, 4: 13, 8: 13},
        "DEEPSEEKV32": {1: 28, 2: 17, 4: 13, 8: 13},
        "DEEPSEEKV4": {1: 28, 2: 17, 4: 13, 8: 13},
        "KIMIK25": {1: 28, 2: 17, 4: 13, 8: 13},
        "KIMIK3": {1: 28, 2: 17, 4: 13, 8: 13},
        "default": {1: 13, 2: 8, 4: 6.5, 8: 6.5},
    }
    MIN_ACTIVATION_BYTES = 90 * 1024 * 1024  # higher floor than TRT-LLM's 70 MiB
    ACTIVATION_OVERHEAD_FRAC = 0.15  # 15% additional activation overhead
    OTHERS_OVERHEAD_FRAC = 0.20  # 20% additional system overhead

    def __init__(self):
        super().__init__()
        self.name = common.BackendName.sglang

    def get_default_free_gpu_memory_fraction(self, backend_version: str | None = None) -> float | None:
        return SGLANG_FALLBACK_MEM_FRACTION_STATIC

    def get_kv_cache_memory_check_params(self) -> tuple[float, float]:
        return 0.0, KV_CACHE_MEMORY_TOLERANCE

    def memory_fraction_of_free(self) -> bool:
        # mem_fraction_static caps weights + KV against TOTAL device memory.
        # (Strictly it excludes activations from the capped pool; subtracting
        # the full non-KV footprint in the shared check is slightly
        # conservative for SGLang.)
        return False

    def _static_oom_check_kwargs(
        self,
        mem_capacity_bytes: int | None = None,
        free_gpu_memory_fraction: float | None = None,
        backend_version: str | None = None,
        model_config=None,
    ) -> dict:
        if free_gpu_memory_fraction is not None:
            fraction = free_gpu_memory_fraction
        elif mem_capacity_bytes:
            fraction = derive_sglang_mem_fraction_static(
                mem_capacity_bytes,
                tp_size=getattr(model_config, "tp_size", 1) or 1,
                pp_size=getattr(model_config, "pp_size", 1) or 1,
                dp_size=getattr(model_config, "attention_dp_size", 1) or 1,
            )
        else:
            fraction = SGLANG_FALLBACK_MEM_FRACTION_STATIC
        return {
            "free_gpu_memory_fraction": fraction,
            "kv_cache_reserved_fraction": 0.0,
            "kv_cache_tolerance": KV_CACHE_MEMORY_TOLERANCE,
            "fraction_of_free": False,
        }

    def _oom_check_kwargs(self, agg_extra: dict) -> dict:
        fraction = agg_extra.get("free_gpu_memory_fraction")
        if fraction is None:
            fraction = SGLANG_FALLBACK_MEM_FRACTION_STATIC
        return {
            "free_gpu_memory_fraction": fraction,
            "kv_cache_reserved_fraction": 0.0,
            "kv_cache_tolerance": KV_CACHE_MEMORY_TOLERANCE,
            "fraction_of_free": False,
        }

    def _tpot_mix_steps(self, num_mix_steps: int) -> int:
        # Same pipeline-drain correction as TRT-LLM: ~3 steps elapse before
        # new requests can be enqueued after the last prefill finishes.
        return max(1, num_mix_steps - 3)
