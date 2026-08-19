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

import aiconfigurator_core._aiconfigurator_core as _core
from aiconfigurator_core.sdk.operations.base import OpShellKit

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class ContextMSAModule(_core.ContextMSAModule, OpShellKit):
    """Context (prefill) MSA. SILICON raises (no data); HYBRID/EMPIRICAL transfer from DSA."""


class GenerationMSAModule(_core.GenerationMSAModule, OpShellKit):
    """Generation (decode) MSA. s = total kv length."""
