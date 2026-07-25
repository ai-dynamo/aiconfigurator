# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Base class and registry for the models package.

Each model family lives in its own module and registers itself via the
``@register_model("FAMILY")`` decorator. ``get_model()`` in the package's
``__init__.py`` does a registry lookup and dispatches to ``cls.create(...)``.

Adding a new model:
    1. Create ``models/<your_model>.py`` with::

        @register_model("YOUR_FAMILY")
        class YourModel(BaseModel):
            @classmethod
            def create(cls, model_info, model_config, backend_name):
                ...
            def __init__(self, ...):
                ...

    2. Register the architecture name(s) in
       ``aiconfigurator_core.sdk.common.ARCHITECTURE_TO_MODEL_FAMILY`` and add
       ``"YOUR_FAMILY"`` to ``ModelFamily``.

    No edits to ``models/__init__.py`` or ``get_model()`` are needed —
    auto-discovery imports every module in this package at import time.
"""

from __future__ import annotations

import logging
from typing import ClassVar

from aiconfigurator_core.sdk import config
from aiconfigurator_core.sdk.config_builders import normalize_nextn

logger = logging.getLogger(__name__)


_MODEL_REGISTRY: dict[str, type] = {}


def register_model(*families: str):
    """Decorator: register ``cls`` as the implementation of one or more families.

    Most classes register one family. Pass multiple when one model class
    handles several families with branching inside ``create()`` — e.g.
    ``DeepSeekModel`` is the entry point for both ``DEEPSEEK`` and
    ``KIMIK25``.

    Logs a warning if a family is already registered (catches typos where
    two files claim the same family).
    """
    if not families:
        raise ValueError("register_model requires at least one family name")

    def decorator(cls):
        for family in families:
            if family in _MODEL_REGISTRY:
                logger.warning(
                    "Overwriting model registration for family %r: %s -> %s",
                    family,
                    _MODEL_REGISTRY[family].__name__,
                    cls.__name__,
                )
            _MODEL_REGISTRY[family] = cls
        return cls

    return decorator


class BaseModel:
    """
    Base model class.
    """

    def __init__(
        self,
        model_path: str,
        model_family: str,
        architecture: str,
        num_layers: int,
        num_heads: int,
        num_kv_heads: int,
        head_size: int,
        hidden_size: int,
        inter_size: int,
        vocab_size: int,
        context_length: int,
        model_config: config.ModelConfig,
        extra_params=None,
    ) -> None:
        """Initialize base model metadata and derived runtime flags."""
        self.model_path = model_path
        self.model_family = model_family
        self.architecture = architecture
        self.config = model_config
        self.extra_params = extra_params
        self._use_qk_norm = bool(extra_params.get("use_qk_norm", False)) if isinstance(extra_params, dict) else False
        self.encoder_ops = []
        self.context_ops = []
        self.generation_ops = []

        # internal only
        self._num_layers = num_layers
        self._num_heads = num_heads
        self._num_kv_heads = num_kv_heads
        self._head_size = head_size
        self._hidden_size = hidden_size
        self._inter_size = inter_size
        self._vocab_size = vocab_size
        self._context_length = context_length
        self._num_kv_heads_per_gpu = (self._num_kv_heads + model_config.tp_size - 1) // model_config.tp_size

        if self._num_layers % model_config.pp_size != 0:
            logger.warning(
                f"num_layers {self._num_layers} is not divisible by pp_size "
                f"{model_config.pp_size}. this will introduce additional rounding error. "
                f"Currently we're nothing to correct this."
            )

        assert self._num_heads % model_config.tp_size == 0, (
            f"num_heads {self._num_heads} should be divisible by tp_size {model_config.tp_size} "
        )

        self._nextn = normalize_nextn(model_config.nextn)
        model_config.nextn = self._nextn

    @property
    def activation_hidden_size(self) -> int:
        return self._num_heads * self._head_size

    # ------------------------------------------------------------------
    # Context parallelism (CP) declaration + comm factory (1145-style).
    # GLM-5 DSA does NOT use these -- it handles CP inside ContextDSAModule.
    # Dense models opt in via supports_cp and splat _cp_attn_comm_ops at the
    # attention site; their FMHA cost is modeled by ContextAttention(cp_size=).
    # ------------------------------------------------------------------
    _BACKEND_CP_STYLE: ClassVar[dict] = {
        "sglang": "allgather",  # SGLang AllGather-of-KV variant
        "trtllm": "ring",  # Ring Attention (not yet wired)
    }

    @classmethod
    def supports_cp(cls, backend_name: str) -> bool:
        """Whether this (model, backend) combo supports context parallelism.

        Default False. CP-capable model classes override to declare which
        backends they support. ``get_model`` checks this BEFORE construction
        and raises a clear error rather than silently producing wrong numbers.
        """
        return False

    @classmethod
    def _resolve_cp_style(cls, backend_name: str) -> str:
        """Pick the CP variant for this (model, backend). Called only when cp_size>1."""
        return cls._BACKEND_CP_STYLE.get(backend_name, "none")

    def _cp_attn_comm_ops(self) -> list:
        """Per-layer CP cross-rank comm ops for this model's ``cp_style``.

        AllGather (sglang): one NCCL all-gather of the full KV, sized from
        ``get_kvcache_bytes_per_sequence(1) / num_layers`` (per-layer per-token
        KV bytes). Models splat this into ``context_ops`` adjacent to the
        attention op. Returns ``[]`` for cp_size<=1 or non-allgather styles.
        """
        import aiconfigurator_core.sdk.operations as ops

        cp_size = self.config.cp_size
        if cp_size <= 1:
            return []
        style = self.config.cp_style
        comm_bytes = self.config.comm_quant_mode.value.memory
        if style == "allgather":
            kv_bytes_per_token = self.get_kvcache_bytes_per_sequence(1) / self._num_layers
            return [
                ops.NCCL(
                    "context_cp_all_gather",
                    self._num_layers,
                    "all_gather",
                    num_elements_per_token=kv_bytes_per_token / comm_bytes,
                    num_gpus=cp_size,
                    comm_quant_mode=self.config.comm_quant_mode,
                )
            ]
        return []

    def _cp_kv_memory_divisor(self) -> int:
        """Per-rank persistent-KV divisor under CP (always 1: full KV per rank).

        Verified against sglang v0.5.13 that CP gives **no** per-rank KV-memory
        savings for any family -- each rank holds the FULL KV:

        - **Dense GQA**: prefill CP gathers + writes the full KV to every rank's
          pool (``cp_all_gather_rerange_kv_cache`` -> ``cp_allgather_and_save_kv_cache``,
          "write the full result into each rank's local memory pool").
        - **MLA / DSA** (DeepSeek V3/V3.2/V4, Kimi): the prefill gather is
          transient, but **decode does not run CP** (``*_use_prefill_cp`` require
          ``is_context_parallel_extend``) and reads the full KV resident in the
          local pool (decode page_table / ``cache_seqlens`` span the full seq_len
          with no gather) -- so the full KV must reside per rank.

        CP therefore saves prefill *compute*, not KV memory. Kept as a method so
        a future seq-sliced variant (e.g. Ring) can override.
        """
        return 1

    def get_kvcache_elements_per_token(self) -> int:
        """KV cache size per token (per GPU) summed over all layers, in elements.

        Multiply by ``kvcache_quant_mode.value.memory`` (bytes/elem) for byte size.

        - MLA models (DeepSeek V3/V3.2, Kimi K2/K2.5): the latent KV is shared
          across heads and not sharded by attention TP, so the per-GPU cost is
          ``num_layers * (kv_lora_rank + qk_rope_head_dim)``.
        - Otherwise (GQA/MHA): ``num_kv_heads_per_gpu * head_size * num_layers * 2``.
        """
        if self.model_family in ("DEEPSEEK", "DEEPSEEKV32", "KIMIK25"):
            kv_lora_rank, qk_rope_head_dim = 0, 0
            if isinstance(self.extra_params, dict):
                kv_lora_rank = self.extra_params.get("kv_lora_rank") or 0
                qk_rope_head_dim = self.extra_params.get("qk_rope_head_dim") or 0
            # Fallback to DeepSeek-V3 / Kimi K2 defaults if config didn't expose them.
            if kv_lora_rank == 0:
                kv_lora_rank = 512
            if qk_rope_head_dim == 0:
                qk_rope_head_dim = 64
            return self._num_layers * (kv_lora_rank + qk_rope_head_dim)

        num_kv_heads_per_gpu = (self._num_kv_heads + self.config.tp_size - 1) // self.config.tp_size
        return num_kv_heads_per_gpu * self._head_size * self._num_layers * 2

    def get_kvcache_bytes_per_sequence(self, seq_len: int) -> float:
        """KV cache bytes for one sequence on one GPU."""
        seq_len = max(0, seq_len)
        return seq_len * self.config.kvcache_quant_mode.value.memory * self.get_kvcache_elements_per_token()

    def get_kvcache_max_tokens(self, kv_budget_bytes: float) -> int:
        """Largest single-sequence length whose KV cache fits in ``kv_budget_bytes``.

        The capacity-sizing inverse of :meth:`get_kvcache_bytes_per_sequence`. The
        base model's KV grows linearly -- a constant number of bytes per token --
        so the inverse is exact floor-division by that per-token size.

        Models whose KV growth is non-linear -- hybrid sliding-window attention
        (SWA layers cap at the window while global layers keep growing) or
        compressed / sparse attention (the per-token rate drops past a window, plus
        fixed decode-state buffers) -- override :meth:`get_kvcache_bytes_per_sequence`
        and also override this method (delegating to
        :meth:`_binary_search_kvcache_max_tokens`) so capacity follows their true piecewise
        curve instead of extrapolating the ``seq_len=1`` slope.
        """
        budget = float(kv_budget_bytes)
        per_token = self.get_kvcache_bytes_per_sequence(1)
        if budget <= 0.0 or per_token <= 0.0:
            return 0
        return int(budget // per_token)

    def _binary_search_kvcache_max_tokens(self, kv_budget_bytes: float) -> int:
        """Monotonic-search inverse of :meth:`get_kvcache_bytes_per_sequence`.

        For non-linear-growth models, where a single per-token constant cannot
        describe the curve, so :meth:`get_kvcache_max_tokens` cannot floor-divide.
        ``get_kvcache_bytes_per_sequence`` is monotonic non-decreasing, so this
        doubles the trial length until its KV exceeds the budget, then binary
        searches for the largest length that still fits.
        """
        budget = float(kv_budget_bytes)
        if budget <= 0.0:
            return 0

        # The equal-size guard handles a model whose KV stops growing with length
        # (a cache fully capped by its window): once doubling the length leaves the
        # KV size unchanged it has saturated, every longer length fits too, and the
        # budget would never be exceeded -- so the loop would not otherwise stop.
        hi, hi_bytes = 1, self.get_kvcache_bytes_per_sequence(1)
        if hi_bytes <= 0.0:
            return 0
        while hi_bytes <= budget:
            nxt = hi * 2
            nxt_bytes = self.get_kvcache_bytes_per_sequence(nxt)
            if nxt_bytes == hi_bytes:
                # KV saturated (a fully window-capped cache): memory never binds, so
                # the real limit is the model's context length. Fall back to the
                # current step only when the context length is unknown.
                return int(self._context_length) if self._context_length > 0 else nxt
            hi, hi_bytes = nxt, nxt_bytes

        # bytes(hi // 2) <= budget < bytes(hi); binary search the boundary.
        lo = hi // 2
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if self.get_kvcache_bytes_per_sequence(mid) <= budget:
                lo = mid
            else:
                hi = mid
        return lo
