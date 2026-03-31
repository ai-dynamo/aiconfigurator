# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Models package — one file per model class with a decorator-based registry.

All public names are re-exported here for backward compatibility:
    from aiconfigurator.sdk.models import get_model, BaseModel, GPTModel, ...
"""

from __future__ import annotations

import importlib
import logging
import pkgutil

from aiconfigurator.sdk import config
from aiconfigurator.sdk.models.base import _MODEL_REGISTRY, BaseModel
from aiconfigurator.sdk.models.helpers import (
    _apply_model_quant_defaults,
    _architecture_to_model_family,
    _get_model_info,
    _infer_quant_modes_from_raw_config,
    calc_expectation,
    check_is_moe,
    get_model_family,
)

# Auto-import all model modules to trigger @register_model decoration.
# Any .py file in this package (except base, helpers, __init__) is auto-imported.
_SKIP = {"base", "helpers"}
_name = None
for _, _name, _ in pkgutil.iter_modules(__path__):
    if _name not in _SKIP:
        importlib.import_module(f".{_name}", __name__)
del _SKIP, _name

logger = logging.getLogger(__name__)


def get_model(
    model_path: str,
    model_config: config.ModelConfig,
    backend_name: str,
) -> BaseModel:
    """
    Get model.
    """
    # Shallow-copy so mutations below don't poison the @cache'd original.
    model_info = dict(_get_model_info(model_path))
    raw_config = model_info.get("raw_config", {})
    architecture = model_info["architecture"]
    model_family = _architecture_to_model_family(architecture)

    _apply_model_quant_defaults(model_config, raw_config, architecture, backend_name)

    if model_config.overwrite_num_layers > 0:
        model_info["layers"] = model_config.overwrite_num_layers

    # Enrich model_info with derived fields for create()
    model_info["model_path"] = model_path
    model_info["model_family"] = model_family

    cls = _MODEL_REGISTRY.get(model_family)
    if cls is None:
        raise ValueError(
            f"Unknown model family: {model_family}. Registered families: {', '.join(sorted(_MODEL_REGISTRY.keys()))}"
        )
    return cls.create(model_info, model_config, backend_name)


# Re-export model classes for backward compatibility.
# These are available via auto-discovery above; explicit names here for static analysis / IDE support.
from aiconfigurator.sdk.models.deepseek import (
    DeepSeekModel,
    TrtllmWideEPDeepSeekModel,
    WideEPDeepSeekModel,
)
from aiconfigurator.sdk.models.gpt import GPTModel
from aiconfigurator.sdk.models.hybrid_moe import HybridMoEModel
from aiconfigurator.sdk.models.llama import LLAMAModel
from aiconfigurator.sdk.models.moe import MOEModel
from aiconfigurator.sdk.models.nemotron_h import NemotronHModel
from aiconfigurator.sdk.models.nemotron_nas import NemotronNas

__all__ = [
    "BaseModel",
    "DeepSeekModel",
    "GPTModel",
    "HybridMoEModel",
    "LLAMAModel",
    "MOEModel",
    "NemotronHModel",
    "NemotronNas",
    "TrtllmWideEPDeepSeekModel",
    "WideEPDeepSeekModel",
    "_apply_model_quant_defaults",
    "_architecture_to_model_family",
    "_get_model_info",
    "_infer_quant_modes_from_raw_config",
    "calc_expectation",
    "check_is_moe",
    "get_model",
    "get_model_family",
]
