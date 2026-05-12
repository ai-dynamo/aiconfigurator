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
