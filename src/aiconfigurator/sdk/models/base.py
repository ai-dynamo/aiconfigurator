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
       ``aiconfigurator.sdk.common.ARCHITECTURE_TO_MODEL_FAMILY`` and add
       ``"YOUR_FAMILY"`` to ``ModelFamily``.

    No edits to ``models/__init__.py`` or ``get_model()`` are needed —
    auto-discovery imports every module in this package at import time.
"""

from __future__ import annotations

import logging
from typing import ClassVar

from aiconfigurator.sdk import config

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

        self._nextn = model_config.nextn
        self._nextn_accept_rates = model_config.nextn_accept_rates

    @property
    def activation_hidden_size(self) -> int:
        return self._num_heads * self._head_size

    # ------------------------------------------------------------------
    # CP variant declaration
    # ------------------------------------------------------------------

    # Default mapping from backend name to CP variant. Backends without an
    # entry produce "none" -- ``get_model`` raises before construction.
    #
    # ``vllm`` is intentionally absent: vLLM's production CP is DCP
    # (decode-time, seq-sharded KV + Q AllGather + LSE combine), which is
    # a fundamentally different concept from this framework's prefill-time
    # sequence parallelism. vLLM's PCP (prefill CP) parameter exists in
    # upstream but the attention backends gate ``supports_pcp = False``.
    # The "ulysses" style code below is kept as scaffolding -- not used
    # by any current backend mapping.
    _BACKEND_CP_STYLE: ClassVar[dict[str, str]] = {
        "sglang": "allgather",  # SGLang AllGather-of-KV variant
        "trtllm": "ring",  # Ring Attention with P2P chain (not yet wired)
    }

    @classmethod
    def supports_cp(cls, backend_name: str) -> bool:
        """Whether this (model, backend) combo supports context parallelism.

        Default: False. Each model class that implements CP ops must
        override to declare which backends it supports.

        ``get_model`` checks this *before* construction; if cp_size>1 and
        this returns False, ``get_model`` raises with a clear error
        rather than silently producing wrong perf numbers.
        """
        return False

    @classmethod
    def _resolve_cp_style(cls, backend_name: str) -> str:
        """Pick the CP variant for this (model, backend) combo.

        Default: per-backend mapping (``_BACKEND_CP_STYLE``). Models can
        override to force a different variant -- e.g. a model that ships
        AllGather perf data on multiple backends.

        Called only when ``cp_size > 1``; ``supports_cp`` has already
        returned True by this point.
        """
        return cls._BACKEND_CP_STYLE.get(backend_name, "none")

    # ------------------------------------------------------------------
    # CP comm-op factory -- models inline at attention site
    # ------------------------------------------------------------------

    def _cp_attn_comm_ops(self) -> list:
        """Per-layer CP cross-rank comm ops for this model's ``cp_style``.

        Returns a list (possibly empty) of ops to splat into the context
        op pipeline at the attention site. Models call this inline, e.g.
        ``self.context_ops.extend([..., attn, *self._cp_attn_comm_ops(), ...])``.

        Order/position within ``context_ops`` doesn't affect simulated
        latency (sequential sum), so we return a single bundle that goes
        adjacent to the attention op rather than splitting into pre/post.

        - ``"none"`` / ``cp_size <= 1``: ``[]``
        - ``"allgather"``: one NCCL all-gather sized from
          ``get_kvcache_bytes_per_sequence(1) / num_layers``.
        - ``"ulysses"``: two NCCL all-to-all ops (pre/post attention).
        - ``"ring"``: P2P chain (not yet implemented -- returns ``[]``
          for now; the model would build this inline once Ring perf data
          lands).

        Models with heterogeneous KV (Gemma4 SWA+global, Hybrid SWA+global,
        DSV4 per-ratio) bypass this default and build their own per-type /
        per-ratio NCCL ops inline.
        """
        from aiconfigurator.sdk import operations as ops  # local import: avoid base->operations cycle

        cp_size = self.config.cp_size
        if cp_size <= 1:
            return []

        style = self.config.cp_style
        comm_bytes = self.config.comm_quant_mode.value.memory

        if style == "allgather":
            # Full KV from each rank, gathered across cp ranks. Sized from
            # the model's get_kvcache_bytes_per_sequence (per-rank-per-layer
            # via /num_layers; per-token via seq_len=1).
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

        if style == "ulysses":
            # DeepSpeed-Ulysses: A2A on QKV pre-attn (head <-> seq), A2A
            # on output post-attn (seq <-> head). Each A2A moves hidden
            # per token; total volume per layer = 2 * hidden * tokens.
            qkv_bytes_per_token = self._hidden_size * 3 * comm_bytes  # Q + K + V
            out_bytes_per_token = self._hidden_size * comm_bytes
            return [
                ops.NCCL(
                    "context_cp_a2a_qkv",
                    self._num_layers,
                    "all_to_all",
                    num_elements_per_token=qkv_bytes_per_token / comm_bytes,
                    num_gpus=cp_size,
                    comm_quant_mode=self.config.comm_quant_mode,
                ),
                ops.NCCL(
                    "context_cp_a2a_out",
                    self._num_layers,
                    "all_to_all",
                    num_elements_per_token=out_bytes_per_token / comm_bytes,
                    num_gpus=cp_size,
                    comm_quant_mode=self.config.comm_quant_mode,
                ),
            ]

        if style == "ring":
            # TODO(cp-ring): emit a P2P chain (cp-1 sends per layer, each
            # carrying 1/cp of the KV). Needs Ring-specific perf data
            # before we can put numbers on it; return [] for now so
            # supports_cp gates kick in earlier than this method.
            return []

        return []

    def get_kvcache_bytes_per_sequence(self, seq_len: int) -> float:
        """KV cache bytes for one sequence **on one GPU**, summed over all layers.

        This is the *single source of truth* for KV memory math per model.
        Semantics:

        - **Local / TP-sharded**: returns bytes that live on a single GPU
          *after* tensor-parallel head sharding. GQA/MHA layouts divide
          ``num_kv_heads`` by ``tp_size`` (ceil); MLA/DSA latents are
          replicated across TP ranks and are TP-agnostic; heterogeneous
          layouts (Gemma4 SWA+global, Hybrid) shard each layer type.
        - **CP-agnostic**: this function does *not* divide by ``cp_size``.
          Per-rank persistent KV depends on the CP *variant*, which the
          caller resolves:
            * AllGather (sglang): KV is gathered at prefill and the full
              cache is replicated on every rank for decode -- ``full``.
            * Ring (trtllm, not yet wired): seq-sliced -- ``full / cp``.
            * Ulysses (scaffolding only -- not used by any current backend
              mapping): head-sliced -- ``full / cp``.
          ``_get_memory_usage`` branches on ``cp_style`` to pick the
          right divisor; ``_cp_attn_comm_ops`` always uses the un-divided
          per-token value (AllGather payload is full-KV per rank, since
          each rank ends up holding the gathered result).
        - **Includes non-linear contributions**: window caps (Gemma4 SWA,
          Hybrid SWA), sparse compressed entries + windowed dense (DSV4)
          are folded in here so the result is the true memory footprint.

        Each model class must implement this -- KV layouts differ enough
        across families that a one-size base default was a frequent source
        of subtle bugs. Downstream consumers derive what they need (CP
        all-gather sizing pulls per-layer per-token via
        ``get_kvcache_bytes_per_sequence(1) / num_layers`` for linear-in-seq
        layouts).
        """
        raise NotImplementedError(f"{type(self).__name__} must implement get_kvcache_bytes_per_sequence")
