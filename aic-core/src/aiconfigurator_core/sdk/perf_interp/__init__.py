# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Perf-table interpolation/extrapolation engine (v2).

``config`` declares the per-op records; ``engine.query(cfg, data, x, y, z)``
resolves a query. See ``config.py`` for the full design.
"""

from aiconfigurator_core.sdk.perf_interp.config import (
    OP_CONFIG_FACTORIES,
    Grid,
    OpInterpConfig,
    ScatteredSites,
    ValueTransform,
    context_attention_config,
    context_grid_config,
    gemm_config,
    generation_attention_config,
    generation_grid_config,
)
from aiconfigurator_core.sdk.perf_interp.engine import clear_caches, get_value, query

__all__ = [
    "OP_CONFIG_FACTORIES",
    "Grid",
    "OpInterpConfig",
    "ScatteredSites",
    "ValueTransform",
    "clear_caches",
    "context_attention_config",
    "context_grid_config",
    "gemm_config",
    "generation_attention_config",
    "generation_grid_config",
    "get_value",
    "query",
]
