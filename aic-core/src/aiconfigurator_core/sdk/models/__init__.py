# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Models package — one file per model family with a decorator-based registry.

Two mechanisms expose model classes:

1. **Registry** — populated automatically. ``pkgutil.iter_modules`` imports
   every ``.py`` file in this package (except ``base`` and ``helpers``) at
   package import time, which fires their ``@register_model(...)`` decorators
   and adds the class to ``_MODEL_REGISTRY``. ``get_model()`` reads from the
   registry. **Adding a new model only needs the new file** — no edits here.

2. **Re-exports** at the bottom of this file (``from .gpt import GPTModel``
   and the ``__all__`` list). These exist so callers can write
   ``from aiconfigurator_core.sdk.models import GPTModel`` directly, matching the
   pre-refactor monolithic-module import style. **Adding a new public class
   to this list IS a manual edit**, but only if the class needs to be
   importable by name from the package root. Skipping it has no functional
   impact — the registry lookup via ``get_model()`` will find the class
   either way.
"""

from __future__ import annotations

import importlib
import pkgutil

from aiconfigurator_core.sdk import config
from aiconfigurator_core.sdk.models.base import _MODEL_REGISTRY, BaseModel
from aiconfigurator_core.sdk.models.helpers import (
    _apply_model_quant_defaults,
    _architecture_to_model_family,
    _get_model_info,
    _infer_quant_modes_from_raw_config,
    attention_op_keys,
    check_is_moe,
    get_model_family,
    mtp_scale_factor,
    resolve_context_fmha_by_data,
    resolve_dsv4_moe_arch,
    resolve_dsv4_moe_arch_mode,
)

# Auto-import every other module in this package so ``@register_model``
# decorators populate ``_MODEL_REGISTRY``. New model files become discoverable
# without editing this __init__.
_SKIP = {"base", "helpers"}
for _, _name, _ in pkgutil.iter_modules(__path__):
    if _name not in _SKIP:
        importlib.import_module(f".{_name}", __name__)
del _SKIP


def get_model(
    model_path: str,
    model_config: config.ModelConfig,
    backend_name: str,
) -> BaseModel:
    """Build a model from a HuggingFace model path.

    Resolves the model family from the architecture, applies quantization
    defaults, then dispatches to the registered class's ``create()``
    classmethod. Per-family construction details (MoE prefix args, WideEP
    dispatch, post-construction hooks) live inside each model's
    ``create()``.
    """
    # Shallow-copy so mutations below don't poison the @cache'd original.
    model_info = dict(_get_model_info(model_path))
    raw_config = model_info.get("raw_config", {})
    architecture = model_info["architecture"]
    model_family = _architecture_to_model_family(architecture)

    _apply_model_quant_defaults(model_config, raw_config, architecture, backend_name)
    if check_is_moe(model_path, model_info=model_info):
        model_config.resolve_moe_parallelism()

    if model_config.overwrite_num_layers > 0:
        model_info["layers"] = model_config.overwrite_num_layers

    # Enrich model_info with derived fields so create() doesn't need to repeat the work.
    model_info["model_path"] = model_path
    model_info["model_family"] = model_family

    cls = _MODEL_REGISTRY.get(model_family)
    if cls is None:
        raise ValueError(
            f"Unknown model family: {model_family}. Registered families: {', '.join(sorted(_MODEL_REGISTRY.keys()))}"
        )

    # Gate context parallelism BEFORE construction. ``supports_cp`` defaults to
    # False; each CP-capable model class overrides it to declare which backends
    # it supports. GLM-5 DSA handles CP inside ContextDSAModule; dense models
    # use the 1145-style skeleton (seq_split + _cp_attn_comm_ops + zigzag FMHA).
    if model_config.cp_size > 1:
        if not cls.supports_cp(backend_name):
            raise NotImplementedError(
                f"Context parallelism (cp_size={model_config.cp_size}) is not supported for "
                f"model_family={model_family!r} on backend={backend_name!r}. The model class "
                f"must override ``supports_cp`` and implement CP in its op pipeline."
            )
        # sglang CP requires the attention side to be pure CP (no concurrent attn TP/DP).
        if backend_name == "sglang" and (model_config.tp_size != 1 or model_config.attention_dp_size != 1):
            raise ValueError(
                f"sglang CP requires tp_size=1 and attention_dp_size=1 when cp_size>1 "
                f"(CP and attention TP/DP are mutually exclusive on sglang). Got "
                f"tp_size={model_config.tp_size}, attention_dp_size={model_config.attention_dp_size}, "
                f"cp_size={model_config.cp_size}."
            )
        model_config.cp_style = cls._resolve_cp_style(backend_name)
    else:
        model_config.cp_style = "none"

    return cls.create(model_info, model_config, backend_name)


# Re-export concrete model classes for backward compatibility. Auto-discovery
# above already imported them; we list them here for static analysis / IDE
# support and so wildcard imports work.
from aiconfigurator_core.sdk.models.deepseek import (
    DeepSeekModel,
    TrtllmWideEPDeepSeekModel,
    WideEPDeepSeekModel,
)
from aiconfigurator_core.sdk.models.deepseek_v4 import DeepSeekV4Model
from aiconfigurator_core.sdk.models.deepseek_v32 import (
    DeepSeekV32Model,
    TrtllmWideEPDeepSeekV32Model,
    WideEPDeepSeekV32Model,
)
from aiconfigurator_core.sdk.models.gemma4 import Gemma4MixModel
from aiconfigurator_core.sdk.models.gpt import GPTModel
from aiconfigurator_core.sdk.models.hybrid_moe import HybridMoEModel
from aiconfigurator_core.sdk.models.llama import LLAMAModel
from aiconfigurator_core.sdk.models.moe import MOEModel, SGLangEPMOEModel
from aiconfigurator_core.sdk.models.nemotron_h import NemotronHModel
from aiconfigurator_core.sdk.models.nemotron_nas import NemotronNas
from aiconfigurator_core.sdk.models.qwen3vl import Qwen3VLModel, Qwen3VLMoEModel
from aiconfigurator_core.sdk.models.qwen35 import Qwen35Model

__all__ = [
    "BaseModel",
    "DeepSeekModel",
    "DeepSeekV4Model",
    "DeepSeekV32Model",
    "GPTModel",
    "Gemma4MixModel",
    "HybridMoEModel",
    "LLAMAModel",
    "MOEModel",
    "NemotronHModel",
    "NemotronNas",
    "Qwen3VLMoEModel",
    "Qwen3VLModel",
    "Qwen35Model",
    "SGLangEPMOEModel",
    "TrtllmWideEPDeepSeekModel",
    "TrtllmWideEPDeepSeekV32Model",
    "WideEPDeepSeekModel",
    "WideEPDeepSeekV32Model",
    "_apply_model_quant_defaults",
    "_architecture_to_model_family",
    "_get_model_info",
    "_infer_quant_modes_from_raw_config",
    "attention_op_keys",
    "check_is_moe",
    "get_model",
    "get_model_family",
    "mtp_scale_factor",
    "resolve_context_fmha_by_data",
    "resolve_dsv4_moe_arch",
    "resolve_dsv4_moe_arch_mode",
]
