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
from aiconfigurator_core.sdk.errors import EmpiricalNotImplementedError
from aiconfigurator_core.sdk.operations.base import Operation
from aiconfigurator_core.sdk.operations.util_empirical import note_provenance
from aiconfigurator_core.sdk.performance_result import PerformanceResult

if TYPE_CHECKING:
    from aiconfigurator_core.sdk.perf_database import PerfDatabase

logger = logging.getLogger(__name__)


def _msa_attention_sol(
    database: PerfDatabase,
    *,
    is_context: bool,
    b: int,
    s: int,
    prefix: int,
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
) -> tuple[float, float, float]:
    """SOL for one MSA block. GQA projections + per-block FP8 indexer + sparse
    attention over the top-k (= index_topk) selected tokens.

    Mirrors DSA/DSV4's three-group structure (gemm / fp8 indexer / fmha attn);
    the attention group uses GQA dims (compute by num_heads, KV cache by
    num_kv_heads) and the top-k saturated causal pair count.
    """
    from aiconfigurator_core.sdk.operations.gemm import GEMM

    qk_head_dim = head_dim
    tokens = b * s if is_context else b
    # context: full prefill of `s` new tokens on top of `prefix` cached.
    # generation: 1 query token, kv_len = s - 1 cached.
    full_s = prefix + s if is_context else s
    kv_len = full_s if is_context else max(0, s - 1)

    # ── GEMM group (Q / GQA-KV / O / indexer-Q projections) ──────────────
    gemm_ops = (
        2 * tokens * hidden_size * (num_heads * qk_head_dim)  # Q
        + 2 * tokens * hidden_size * (2 * num_kv_heads * head_dim)  # K, V (GQA)
        + 2 * tokens * (num_heads * v_head_dim) * hidden_size  # O
        + 2 * tokens * hidden_size * (index_n_heads * index_head_dim)  # indexer Q
    )

    # ── sparse attention: top-k saturated causal (query, kv) pair count ──
    if is_context:
        if full_s <= index_topk:
            pairs = b * (full_s * (full_s + 1) - prefix * (prefix + 1)) // 2
        elif prefix >= index_topk:
            pairs = tokens * index_topk
        else:
            ramp = b * (index_topk * (index_topk + 1) - prefix * (prefix + 1)) // 2
            sat = b * (full_s - index_topk) * index_topk
            pairs = ramp + sat
        score_len = full_s
    else:
        pairs = tokens * min(kv_len, index_topk)
        score_len = kv_len
    effective_kv = min(kv_len, index_topk) if not is_context else min(full_s, index_topk)
    attention_ops = 2 * num_heads * (qk_head_dim + v_head_dim) * pairs  # QK^T + AV

    # ── indexer: per-block scoring (block_size tokens per block), FP8 ────
    num_blocks = (score_len + block_size - 1) // block_size if score_len > index_topk else 0
    indexer_ops = 2 * tokens * index_n_heads * index_head_dim * num_blocks

    # ── memory ───────────────────────────────────────────────────────────
    gemm_weight_bytes = (
        hidden_size * num_heads * qk_head_dim
        + hidden_size * 2 * num_kv_heads * head_dim
        + num_heads * v_head_dim * hidden_size
        + hidden_size * index_n_heads * index_head_dim
    ) * gemm_quant_mode.value.memory
    kv_cache_bytes = b * num_kv_heads * effective_kv * (qk_head_dim + v_head_dim) * kvcache_quant_mode.value.memory
    indexer_cache_bytes = b * num_blocks * index_n_heads * index_head_dim  # FP8 index keys, per block
    q_io_bytes = tokens * num_heads * qk_head_dim * fmha_quant_mode.value.memory * 2
    total_mem = gemm_weight_bytes + kv_cache_bytes + indexer_cache_bytes + q_io_bytes

    gemm_flops = GEMM._get_quant_tc_flops(database.system_spec, gemm_quant_mode)
    fp8_flops = GEMM._get_quant_tc_flops(database.system_spec, common.FMHAQuantMode.fp8)
    attn_flops = GEMM._get_quant_tc_flops(database.system_spec, fmha_quant_mode)

    sol_math = (gemm_ops / gemm_flops + indexer_ops / fp8_flops + attention_ops / attn_flops) * 1000
    sol_mem = total_mem / database.system_spec["gpu"]["mem_bw"] * 1000
    sol_time = max(sol_math, sol_mem)
    return sol_time, sol_math, sol_mem


def _dsa_context_util(
    database, *, b, s, prefix, num_heads, kvcache_quant_mode, fmha_quant_mode, gemm_quant_mode, architecture
):
    """DSA's measured utilisation (SOL/silicon) at the same workload, or None."""
    try:
        sol = float(
            database.query_context_dsa_module(
                b=b,
                s=s,
                prefix=prefix,
                num_heads=num_heads,
                kvcache_quant_mode=kvcache_quant_mode,
                fmha_quant_mode=fmha_quant_mode,
                gemm_quant_mode=gemm_quant_mode,
                architecture=architecture,
                database_mode=common.DatabaseMode.SOL,
            )
        )
        sil = float(
            database.query_context_dsa_module(
                b=b,
                s=s,
                prefix=prefix,
                num_heads=num_heads,
                kvcache_quant_mode=kvcache_quant_mode,
                fmha_quant_mode=fmha_quant_mode,
                gemm_quant_mode=gemm_quant_mode,
                architecture=architecture,
                database_mode=common.DatabaseMode.SILICON,
            )
        )
        return sol / sil if sol > 0 and sil > 0 else None
    except Exception:
        return None


def _dsa_generation_util(database, *, b, s, num_heads, kvcache_quant_mode, gemm_quant_mode, architecture):
    try:
        sol = float(
            database.query_generation_dsa_module(
                b,
                s,
                num_heads,
                kvcache_quant_mode,
                gemm_quant_mode,
                database_mode=common.DatabaseMode.SOL,
                architecture=architecture,
            )
        )
        sil = float(
            database.query_generation_dsa_module(
                b,
                s,
                num_heads,
                kvcache_quant_mode,
                gemm_quant_mode,
                database_mode=common.DatabaseMode.SILICON,
                architecture=architecture,
            )
        )
        return sol / sil if sol > 0 and sil > 0 else None
    except Exception:
        return None


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
        self._weights = 0.0

    @classmethod
    def load_data(cls, database):  # no MSA silicon table
        pass

    def _sol(self, database, b, s, prefix, is_context):
        return _msa_attention_sol(
            database,
            is_context=is_context,
            b=b,
            s=s,
            prefix=prefix,
            num_heads=self._num_heads,
            num_kv_heads=self._num_kv_heads,
            hidden_size=self._hidden_size,
            head_dim=self._head_dim,
            v_head_dim=self._v_head_dim,
            index_n_heads=self._index_n_heads,
            index_head_dim=self._index_head_dim,
            index_topk=self._index_topk,
            block_size=self._block_size,
            kvcache_quant_mode=self._kvcache_quant_mode,
            fmha_quant_mode=self._fmha_quant_mode,
            gemm_quant_mode=self._gemm_quant_mode,
        )[0]

    def get_weights(self, **kwargs):
        return self._weights * self._scale_factor


class ContextMSAModule(_BaseMSAModule):
    """Context (prefill) MSA. SILICON raises (no data); HYBRID/EMPIRICAL transfer from DSA."""

    def query(self, database, **kwargs):
        b = kwargs.get("batch_size")
        s = kwargs.get("s")
        prefix = kwargs.get("prefix", 0)
        mode = database._default_database_mode
        sol = self._sol(database, b, s, prefix, is_context=True)
        if mode in (common.DatabaseMode.SOL, common.DatabaseMode.SOL_FULL):
            return PerformanceResult(sol * self._scale_factor, energy=0.0, source="sol")
        if mode == common.DatabaseMode.SILICON:
            from aiconfigurator_core.sdk.perf_database import PerfDataNotAvailableError

            raise PerfDataNotAvailableError("MSA has no silicon data; use HYBRID or EMPIRICAL.")
        # EMPIRICAL / HYBRID: cross-op (XOP) transfer from DSA util * k. MSA has no
        # silicon data, so when XOP is disabled by the transfer policy there is nothing
        # to fall back on -> raise honestly.
        if common.TransferKind.XOP not in database.transfer_policy:
            raise EmpiricalNotImplementedError(
                "MSA context: cross-op transfer (xop) is disabled by the transfer policy "
                "and MSA has no own silicon data."
            )
        util = _dsa_context_util(
            database,
            b=b,
            s=s,
            prefix=prefix,
            num_heads=self._num_heads,
            kvcache_quant_mode=self._kvcache_quant_mode,
            fmha_quant_mode=self._fmha_quant_mode,
            gemm_quant_mode=self._gemm_quant_mode,
            architecture=self._dsa_architecture,
        )
        if not (util and util > 0):
            raise EmpiricalNotImplementedError(
                f"MSA context: no DSA util to transfer from (arch={self._dsa_architecture}, "
                f"b={b}, s={s}); collect DSA data or set msa_dsa_scale_k against an available quant."
            )
        note_provenance("xop")  # cross-op transfer from DSA
        lat = sol / (util * self._dsa_scale_k)
        return PerformanceResult(lat * self._scale_factor, energy=0.0, source="empirical")


class GenerationMSAModule(_BaseMSAModule):
    """Generation (decode) MSA. s = total kv length."""

    def query(self, database, **kwargs):
        b = kwargs.get("batch_size")
        s = kwargs.get("s")
        mode = database._default_database_mode
        sol = self._sol(database, b, s, 0, is_context=False)
        if mode in (common.DatabaseMode.SOL, common.DatabaseMode.SOL_FULL):
            return PerformanceResult(sol * self._scale_factor, energy=0.0, source="sol")
        if mode == common.DatabaseMode.SILICON:
            from aiconfigurator_core.sdk.perf_database import PerfDataNotAvailableError

            raise PerfDataNotAvailableError("MSA has no silicon data; use HYBRID or EMPIRICAL.")
        if common.TransferKind.XOP not in database.transfer_policy:
            raise EmpiricalNotImplementedError(
                "MSA generation: cross-op transfer (xop) is disabled by the transfer policy "
                "and MSA has no own silicon data."
            )
        util = _dsa_generation_util(
            database,
            b=b,
            s=s,
            num_heads=self._num_heads,
            kvcache_quant_mode=self._kvcache_quant_mode,
            gemm_quant_mode=self._gemm_quant_mode,
            architecture=self._dsa_architecture,
        )
        if not (util and util > 0):
            raise EmpiricalNotImplementedError(
                f"MSA generation: no DSA util to transfer from (arch={self._dsa_architecture}, "
                f"b={b}, s={s}); collect DSA data or set msa_dsa_scale_k against an available quant."
            )
        note_provenance("xop")  # cross-op transfer from DSA
        lat = sol / (util * self._dsa_scale_k)
        return PerformanceResult(lat * self._scale_factor, energy=0.0, source="empirical")
