# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Operations package — one file per operation family with class-owned data
loading and querying.

This package replaces the prior monolithic ``operations.py``. Each family
lives in its own module; ``Operation`` and ``clear_all_op_caches`` come
from ``base.py``.

Public surface preserves the prior import pattern:

    from aiconfigurator_core.sdk.operations import GEMM, ContextAttention, ...
"""

from __future__ import annotations

from aiconfigurator_core.sdk.operations.afd_transfer import (
    AFDCombine,
    AFDFAllGather,
    AFDFReduceScatter,
    AFDTransfer,
    _afd_send_prob,
)
from aiconfigurator_core.sdk.operations.attention import ContextAttention, EncoderAttention, GenerationAttention
from aiconfigurator_core.sdk.operations.base import Operation, clear_all_op_caches, warm_all_op_data
from aiconfigurator_core.sdk.operations.communication import NCCL, P2P, CustomAllReduce
from aiconfigurator_core.sdk.operations.dsa import ContextDSAModule, GenerationDSAModule
from aiconfigurator_core.sdk.operations.dsv4 import (
    ContextDeepSeekV4AttentionModule,
    DeepSeekV4MegaMoEModule,
    DeepSeekV4MHCModule,
    GenerationDeepSeekV4AttentionModule,
    _BaseDeepSeekV4AttentionModule,
)
from aiconfigurator_core.sdk.operations.elementwise import ElementWise
from aiconfigurator_core.sdk.operations.embedding import Embedding
from aiconfigurator_core.sdk.operations.gemm import GEMM
from aiconfigurator_core.sdk.operations.mamba import GDNKernel, Mamba2, Mamba2Kernel
from aiconfigurator_core.sdk.operations.mla import (
    ContextMLA,
    GenerationMLA,
    MLABmm,
    MLAModule,
    WideEPContextMLA,
    WideEPGenerationMLA,
)
from aiconfigurator_core.sdk.operations.moe import MoE, MoEDispatch, TrtLLMWideEPMoE, TrtLLMWideEPMoEDispatch
from aiconfigurator_core.sdk.operations.msa import ContextMSAModule, GenerationMSAModule
from aiconfigurator_core.sdk.operations.overlap import FallbackOp, OverlapOp

# Re-export commonly-imported names that the prior monolithic operations.py
# exposed at module level. Some test files and external callers do
# ``from aiconfigurator_core.sdk.operations import PerformanceResult``.
from aiconfigurator_core.sdk.performance_result import PerformanceResult

__all__ = [
    "GEMM",
    "NCCL",
    "P2P",
    "AFDCombine",
    "AFDFAllGather",
    "AFDFReduceScatter",
    "AFDTransfer",
    "ContextAttention",
    "ContextDSAModule",
    "ContextDeepSeekV4AttentionModule",
    "ContextMLA",
    "ContextMSAModule",
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
    "GenerationMSAModule",
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
    "_afd_send_prob",
    "clear_all_op_caches",
    "warm_all_op_data",
]
