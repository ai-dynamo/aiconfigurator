# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stable Python facade for AIConfigurator's estimator core.

The standalone :mod:`aiconfigurator_core` distribution owns these
implementations. The upper ``aiconfigurator`` distribution provides thin
compatibility modules for the historical ``aiconfigurator.sdk`` import paths.

The names in :data:`__all__` are the deliberately small, supported facade for
core-only consumers. They are resolved lazily so importing
``aiconfigurator_core.sdk`` does not eagerly load the model registry, perf
database, or native engine. Existing explicit module imports remain supported.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

__all__ = [
    "EngineHandle",
    "ModelConfig",
    "RuntimeConfig",
    "RustForwardPassPerfModel",
    "compile_engine",
    "estimate_kv_cache",
    "estimate_num_gpu_blocks",
]

_PUBLIC_EXPORTS = {
    "EngineHandle": ("aiconfigurator_core.sdk.engine", "EngineHandle"),
    "ModelConfig": ("aiconfigurator_core.sdk.config", "ModelConfig"),
    "RuntimeConfig": ("aiconfigurator_core.sdk.config", "RuntimeConfig"),
    "RustForwardPassPerfModel": (
        "aiconfigurator_core.sdk.rust_engine_step",
        "RustForwardPassPerfModel",
    ),
    "compile_engine": ("aiconfigurator_core.sdk.engine", "compile_engine"),
    "estimate_kv_cache": ("aiconfigurator_core.sdk.memory", "estimate_kv_cache"),
    "estimate_num_gpu_blocks": (
        "aiconfigurator_core.sdk.memory",
        "estimate_num_gpu_blocks",
    ),
}


def __getattr__(name: str) -> Any:
    """Resolve one stable facade export without eagerly importing the SDK."""
    try:
        module_name, attribute = _PUBLIC_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})


if TYPE_CHECKING:
    from aiconfigurator_core.sdk.config import ModelConfig, RuntimeConfig
    from aiconfigurator_core.sdk.engine import EngineHandle, compile_engine
    from aiconfigurator_core.sdk.memory import estimate_kv_cache, estimate_num_gpu_blocks
    from aiconfigurator_core.sdk.rust_engine_step import RustForwardPassPerfModel
