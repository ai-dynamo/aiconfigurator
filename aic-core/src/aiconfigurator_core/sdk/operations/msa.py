# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MiniMax Sparse Attention (MSA) module ops for MiniMax-M3.

MSA (github.com/MiniMax-AI/MSA) is structurally a GQA version of DSA: an indexer
does a cheap per-block "dense proxy" pass to score KV blocks, the top-k blocks
are selected, and full attention runs over only the selected tokens. Versus DSA
the main attention is standard GQA (not MLA-compressed), and the indexer scores
per *block* (block_size tokens) rather than per token.

There is no collected MSA silicon data. These ops therefore run in HYBRID /
EMPIRICAL only: the SOL is derived below (same three-group split as DSA/DSV4 --
GEMM projections, FP8 indexer, sparse attention), and the empirical value is a
CROSS-OP TRANSFER from DSA's measured utilisation at the same workload, scaled
by a manual ``dsa_scale_k`` (util_scale hook): ``latency = SOL_msa /
(util_dsa * k)``. SOL only needs to capture the (b, s, prefix) shape trend; k
pulls the absolute level. Falls back to a constant when DSA data is absent.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from aiconfigurator_core.sdk import common
from aiconfigurator_core.sdk.operations.base import Operation

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class _BaseMSAModule(Operation):
    """Shared MSA op: SOL + cross-op-transfer empirical (no MSA silicon data)."""

    def __init__(
        self,
        name: str,
        scale_factor: float,
        num_heads: int,
        num_kv_heads: int,
        hidden_size: int,
        head_dim: int,
        v_head_dim: int,
        index_n_heads: int,
        index_head_dim: int,
        index_topk: int,
        block_size: int,
        kvcache_quant_mode: common.KVCacheQuantMode,
        fmha_quant_mode: common.FMHAQuantMode,
        gemm_quant_mode: common.GEMMQuantMode,
        dsa_architecture: str = "GlmMoeDsaForCausalLM",
        dsa_scale_k: float = 1.0,
    ) -> None:
        super().__init__(name, scale_factor)
        self._num_heads = num_heads
        self._num_kv_heads = num_kv_heads
        self._hidden_size = hidden_size
        self._head_dim = head_dim
        self._v_head_dim = v_head_dim
        self._index_n_heads = index_n_heads
        self._index_head_dim = index_head_dim
        self._index_topk = index_topk
        self._block_size = block_size
        self._kvcache_quant_mode = kvcache_quant_mode
        self._fmha_quant_mode = fmha_quant_mode
        self._gemm_quant_mode = gemm_quant_mode
        self._dsa_architecture = dsa_architecture
        self._dsa_scale_k = dsa_scale_k

    @classmethod
    def load_data(cls, database):  # no MSA silicon table
        pass


class ContextMSAModule(_BaseMSAModule):
    """Context (prefill) MSA. SILICON raises (no data); HYBRID/EMPIRICAL transfer from DSA."""

    _ENGINE_QUERY_SHAPE = "context"


class GenerationMSAModule(_BaseMSAModule):
    """Generation (decode) MSA. s = total kv length."""

    _ENGINE_QUERY_SHAPE = "generation"
