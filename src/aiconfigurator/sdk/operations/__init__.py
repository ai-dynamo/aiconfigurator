# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Operations package — one file per operation family with class-owned data
loading and querying.

This package replaces the prior monolithic ``operations.py``. Migration
proceeds family-by-family (ISSUE-04 through ISSUE-14); each issue moves a
group of op classes out of ``_legacy.py`` into a dedicated file. ``_legacy.py``
is deleted by ISSUE-15 once empty.

Public surface preserves the prior import pattern:

    from aiconfigurator.sdk.operations import GEMM, ContextAttention, ...

works whether the class lives in ``_legacy.py`` or in a per-family module.
"""

from __future__ import annotations

# Everything still in _legacy.py until its owning ISSUE migrates it.
# Re-export the legacy module's logger so tests that do
# ``mock.patch("aiconfigurator.sdk.operations.logger")`` keep working. The
# logger lives in ``_legacy.py``; per-family migrations don't redirect their
# log calls until their owning issue lands.
from aiconfigurator.sdk.operations._legacy import (
    NCCL,
    P2P,
    ContextDeepSeekV4AttentionModule,
    ContextDSAModule,
    ContextMLA,
    CustomAllReduce,
    DeepSeekV4MHCModule,
    FallbackOp,
    GDNKernel,
    GenerationDeepSeekV4AttentionModule,
    GenerationDSAModule,
    GenerationMLA,
    Mamba2,
    Mamba2Kernel,
    MLABmm,
    MLAModule,
    MoE,
    MoEDispatch,
    OverlapOp,
    TrtLLMWideEPMoE,
    TrtLLMWideEPMoEDispatch,
    WideEPContextMLA,
    WideEPGenerationMLA,
    _BaseDeepSeekV4AttentionModule,
    logger,  # noqa: F401
)

# Per-family modules (migrated out of _legacy.py).
from aiconfigurator.sdk.operations.attention import ContextAttention, GenerationAttention

# Shared base + utilities. These come first so per-family modules can import
# from .base without circular issues.
from aiconfigurator.sdk.operations.base import Operation, clear_all_op_caches
from aiconfigurator.sdk.operations.elementwise import ElementWise
from aiconfigurator.sdk.operations.embedding import Embedding
from aiconfigurator.sdk.operations.gemm import GEMM

# Re-export commonly-imported names that the prior monolithic operations.py
# exposed at module level. Some test files and external callers do
# ``from aiconfigurator.sdk.operations import PerformanceResult``.
from aiconfigurator.sdk.performance_result import PerformanceResult

__all__ = [
    "GEMM",
    "NCCL",
    "P2P",
    "ContextAttention",
    "ContextDSAModule",
    "ContextDeepSeekV4AttentionModule",
    "ContextMLA",
    "CustomAllReduce",
    "DeepSeekV4MHCModule",
    "ElementWise",
    "Embedding",
    "FallbackOp",
    "GDNKernel",
    "GenerationAttention",
    "GenerationDSAModule",
    "GenerationDeepSeekV4AttentionModule",
    "GenerationMLA",
    "MLABmm",
    "MLAModule",
    "Mamba2",
    "Mamba2Kernel",
    "MoE",
    "MoEDispatch",
    "Operation",
    "OverlapOp",
    "PerformanceResult",
    "TrtLLMWideEPMoE",
    "TrtLLMWideEPMoEDispatch",
    "WideEPContextMLA",
    "WideEPGenerationMLA",
    "_BaseDeepSeekV4AttentionModule",
    "clear_all_op_caches",
]
