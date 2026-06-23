# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Operations package — one file per operation family with class-owned data
loading and querying.

This package replaces the prior monolithic ``operations.py``. Each family
lives in its own module; ``Operation`` and ``clear_all_op_caches`` come
from ``base.py``.

Public surface preserves the prior import pattern:

    from aiconfigurator.sdk.operations import GEMM, ContextAttention, ...
"""

from __future__ import annotations

from aiconfigurator.sdk.operations.attention import ContextAttention, EncoderAttention, GenerationAttention
from aiconfigurator.sdk.operations.base import Operation, clear_all_op_caches, warm_all_op_data
from aiconfigurator.sdk.operations.communication import NCCL, P2P, CustomAllReduce
from aiconfigurator.sdk.operations.dsa import ContextDSAModule, GenerationDSAModule
from aiconfigurator.sdk.operations.dsv4 import (
    ContextDeepSeekV4AttentionModule,
    DeepSeekV4MegaMoEModule,
    DeepSeekV4MHCModule,
    GenerationDeepSeekV4AttentionModule,
    _BaseDeepSeekV4AttentionModule,
)
from aiconfigurator.sdk.operations.elementwise import ElementWise
from aiconfigurator.sdk.operations.embedding import Embedding
from aiconfigurator.sdk.operations.gemm import GEMM
from aiconfigurator.sdk.operations.layerwise import Layerwise
from aiconfigurator.sdk.operations.mamba import GDNKernel, Mamba2, Mamba2Kernel
from aiconfigurator.sdk.operations.mla import (
    ContextMLA,
    GenerationMLA,
    MLABmm,
    MLAModule,
    WideEPContextMLA,
    WideEPGenerationMLA,
)
from aiconfigurator.sdk.operations.moe import MoE, MoEDispatch, TrtLLMWideEPMoE, TrtLLMWideEPMoEDispatch
from aiconfigurator.sdk.operations.overlap import FallbackOp, OverlapOp

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
    "DeepSeekV4MegaMoEModule",
    "ElementWise",
    "Embedding",
    "EncoderAttention",
    "FallbackOp",
    "GDNKernel",
    "GenerationAttention",
    "GenerationDSAModule",
    "GenerationDeepSeekV4AttentionModule",
    "GenerationMLA",
    "Layerwise",
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
    "warm_all_op_data",
]
