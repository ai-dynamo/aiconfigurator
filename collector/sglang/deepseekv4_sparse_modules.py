# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V4-Flash sparse-attention kernel-level collector for SGLang.

Benchmarks the two past_kv-sensitive kernels that bench accurately at the
kernel level (paged_mqa_logits + hca_attn).  Inputs match upstream layouts
(DeepGEMM ``test_attention.py`` for the indexer GEMM, FlashMLA
``MODEL1_FP8Sparse`` quant for the FMLA kernel).

Kernels:

    1. ``deep_gemm.fp8_paged_mqa_logits``       (CSA indexer scoring)
    2. ``flash_mla.flash_mla_with_kvcache`` HCA  (HCA c128 sparse FMLA)

(``topk_512`` and ``csa_attn`` are modeled analytically in perf_database —
see KERNELS comment.)

Sweep dims (defaults):

    M        : new query tokens (mirrors current prefill ctx sweep)  # noqa: N803,N806
               [1, 8, 64, 256, 1024, 4096, 8192]
    past_kv  : 0 → ~1M     [0, 1024, 4096, 16384, 65536, 262144, 1048575-8192]

CSV schema matches existing aic dsv4_flash module CSVs (so loaders can be
shared): ``isl`` carries M, ``step`` carries past_kv, ``compress_ratio``
distinguishes CSA(=4) / HCA(=128).
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from collections.abc import Callable

import torch

try:
    from collector.sglang.helper import benchmark_with_power, log_perf
except ModuleNotFoundError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from helper import benchmark_with_power, log_perf

# Re-export test case generators from the centralised common_test_cases
# module so collect.py's registry can resolve them via getattr on this module.
try:
    from collector.common_test_cases import (
        _DSV4_FLASH_DEFAULT_MODEL as DEFAULT_MODEL,
    )
    from collector.common_test_cases import (
        _DSV4_FLASH_SPARSE_BS_LIST as DEFAULT_BS_LIST,
    )
    from collector.common_test_cases import (
        _DSV4_FLASH_SPARSE_ISL_LIST as DEFAULT_ISL_LIST,
    )
    from collector.common_test_cases import (
        _DSV4_FLASH_SPARSE_PAST_KV_LIST as DEFAULT_PAST_KV_LIST,
    )
    from collector.common_test_cases import (
        _DSV4_FLASH_SPARSE_TP_LIST_ATTN as DEFAULT_TP_LIST_ATTN,
    )
    from collector.common_test_cases import (
        DSV4_FLASH_SPARSE_KERNELS as KERNELS,
    )
    from collector.common_test_cases import (
        _build_dsv4_flash_sparse_test_cases as _build_sparse_test_cases,
    )
except ModuleNotFoundError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from common_test_cases import (
        _DSV4_FLASH_DEFAULT_MODEL as DEFAULT_MODEL,
    )
    from common_test_cases import (
        _DSV4_FLASH_SPARSE_BS_LIST as DEFAULT_BS_LIST,
    )
    from common_test_cases import (
        _DSV4_FLASH_SPARSE_ISL_LIST as DEFAULT_ISL_LIST,
    )
    from common_test_cases import (
        _DSV4_FLASH_SPARSE_PAST_KV_LIST as DEFAULT_PAST_KV_LIST,
    )
    from common_test_cases import (
        _DSV4_FLASH_SPARSE_TP_LIST_ATTN as DEFAULT_TP_LIST_ATTN,
    )
    from common_test_cases import (
        DSV4_FLASH_SPARSE_KERNELS as KERNELS,
    )
    from common_test_cases import (
        _build_dsv4_flash_sparse_test_cases as _build_sparse_test_cases,
    )


def get_dsv4_flash_paged_mqa_logits_test_cases():
    from collector.common_test_cases import get_dsv4_flash_paged_mqa_logits_test_cases as _impl

    return _impl()


def get_dsv4_flash_hca_attn_test_cases():
    from collector.common_test_cases import get_dsv4_flash_hca_attn_test_cases as _impl

    return _impl()


__all__ = [
    "DEFAULT_BS_LIST",
    "DEFAULT_ISL_LIST",
    "DEFAULT_MODEL",
    "DEFAULT_PAST_KV_LIST",
    "DEFAULT_TP_LIST_ATTN",
    "KERNELS",
    "_build_sparse_test_cases",
    "get_dsv4_flash_hca_attn_test_cases",
    "get_dsv4_flash_paged_mqa_logits_test_cases",
    "run_dsv4_sparse_kernel_worker",
]


# ═══════════════════════════════════════════════════════════════════════
# V4-Flash architectural constants
# ═══════════════════════════════════════════════════════════════════════

# Indexer
N_IDX_HEADS = 64
IDX_HEAD_DIM = 128

# Main attention (V4-Flash NSA -- MODEL1_FP8Sparse layout)
N_HEADS_Q = 64
V_HEAD_DIM = 512

# FlashMLA d_qk for V4-Flash NSA = 512 (NOT 576).  Layout MODEL1_FP8Sparse:
#   d_nope=448, d_rope=64, tile_size=64, num_tiles=7
# bytes_per_token = 448 + 64*2 + 7 + 1 = 584 (with 1 pad)
FMLA_D_QK = 512
FMLA_D_NOPE = 448
FMLA_D_ROPE = 64
FMLA_TILE_SIZE = 64
FMLA_NUM_TILES = 7

# Page sizes
PAGE_SIZE_C4 = 64  # paged_mqa_logits block_kv
PAGE_SIZE_FULL = 64  # FlashMLA paged block_size

NUM_SMS_H20 = 78
DEFAULT_ARCHITECTURE = "DeepseekV4ForCausalLM"


# Two kernels are benched at the kernel level:
#   - paged_mqa_logits: CSA indexer scoring (TP-independent, accurate)
#   - hca_attn:       HCA's flash_mla over c128 cache (TP-independent, accurate)
#                     (production V4-Flash at TP>1 also runs FlashMLA with
#                      h_q=64 — sglang pads Q to full 64 heads, then slices
#                      output back; see deepseek_v4.py:847.  So TP=1 data is
#                      valid for any deployment TP.)
#
# Two kernels are modeled ANALYTICALLY in perf_database (NOT benched):
#   - topk_512: pure memory-IO scan over fp32 logits (per-token causal).
#     Bench at random / chained logits gives Δ off by 60%; AIC IO formula
#     with eff≈0.1 gives Δ off by only ~8%.  Modeled as
#         Δ_bytes(M, past_kv) = M * past_kv   (causal-scan additional bytes)
#         Δ_time              = Δ_bytes / (mem_bw * 0.1) * 1000  (ms)
#   - csa_attn: cache scatter pattern of topk-selected indices is impossible
#     to reproduce with sequential indices in a kernel-level bench.
KERNEL_TO_OP_NAME = {
    "paged_mqa_logits": "dsv4_flash_paged_mqa_logits_module",
    "hca_attn": "dsv4_flash_hca_attn_module",
}

KERNEL_TO_KERNEL_SOURCE = {
    "paged_mqa_logits": "deep_gemm.fp8_paged_mqa_logits",
    "hca_attn": "flash_mla_with_kvcache",
}

# compress_ratio: 4 for indexer kernel, 128 for HCA's c128 attn
KERNEL_TO_COMPRESS_RATIO = {
    "paged_mqa_logits": 4,
    "hca_attn": 128,
}

KERNEL_TO_DEFAULT_FILENAME = {
    "paged_mqa_logits": "dsv4_flash_paged_mqa_logits_module_perf.txt",
    "hca_attn": "dsv4_flash_hca_attn_module_perf.txt",
}

# ═══════════════════════════════════════════════════════════════════════
# Bench helper
# ═══════════════════════════════════════════════════════════════════════


def _bench_cuda_graph(
    kernel_fn: Callable[[], None],
    *,
    num_warmup: int = 5,
    num_iterations: int = 20,
    graph_repeat: int = 4,
    device: str = "cuda:0",
) -> dict:
    """Benchmark a kernel via AIC's benchmark_with_power helper.

    benchmark_with_power handles warmup, CUDA-Graph capture/replay, optional
    power sampling, and graph-private-pool teardown. Capture failure is a
    hard error: ``allow_graph_fail=False`` and ``used_cuda_graph`` is
    checked explicitly. Returns ``{"latency_ms", "power_stats"}``.
    """
    if num_iterations < 3:
        raise ValueError("num_iterations must be at least 3")
    if graph_repeat < 1:
        raise ValueError("graph_repeat must be at least 1")

    def timed_kernel():
        with torch.no_grad():
            return kernel_fn()

    with benchmark_with_power(
        device=torch.device(device),
        kernel_func=timed_kernel,
        num_warmups=num_warmup,
        num_runs=num_iterations,
        repeat_n=graph_repeat,
        allow_graph_fail=False,
    ) as result:
        pass

    if not result.get("used_cuda_graph", False):
        raise RuntimeError("benchmark_with_power did not use CUDA Graph")

    return {
        "latency_ms": float(result["latency_ms"]),
        "power_stats": result.get("power_stats"),
    }


# ═══════════════════════════════════════════════════════════════════════
# Cache packing helpers (mirror DeepGEMM/FlashMLA test code)
# ═══════════════════════════════════════════════════════════════════════


def _kv_cache_cast_to_fp8_indexer(x: torch.Tensor) -> torch.Tensor:
    """DeepGEMM test_attention's kv_cache_cast_to_fp8.

    x: (num_blocks, block_size, 1, head_dim=128) bf16
    out: (num_blocks, block_size, 1, head_dim+4=132) packed fp8 + per-token fp32 sf
    """
    num_blocks, block_size, num_heads, head_dim = x.shape
    assert num_heads == 1
    x_amax = x.abs().float().amax(dim=3, keepdim=True).clamp(1e-4)
    sf = x_amax / 448.0
    x_scaled = (x * (1.0 / sf)).to(torch.float8_e4m3fn)

    out = torch.empty((num_blocks, block_size * (head_dim + 4)), device=x.device, dtype=torch.uint8)
    out[:, : block_size * head_dim] = x_scaled.view(num_blocks, block_size * head_dim).view(torch.uint8)
    out[:, block_size * head_dim :] = sf.view(num_blocks, block_size).view(torch.uint8)
    return out.view(num_blocks, block_size, 1, head_dim + 4)


def _quantize_k_cache_model1(k_bf16: torch.Tensor) -> torch.Tensor:
    """FlashMLA MODEL1_FP8Sparse pack (V4-Flash layout).

    k_bf16: (num_blocks, block_size, 1, d_qk=512) bf16
    out:    (num_blocks, block_size, 1, bytes_per_token=584) packed fp8

    Layout per token (inside the bytes_per_token slab):
        [d_nope=448 fp8 nope][d_rope*2=128 bf16 rope][num_tiles=7 fp8 e8m0 sf][1 pad]
    """
    num_blocks, block_size, num_heads, d_qk = k_bf16.shape
    assert num_heads == 1 and d_qk == FMLA_D_QK
    k = k_bf16.squeeze(2)  # (num_blocks, block_size, d_qk)

    bytes_per_token = FMLA_D_NOPE + 2 * FMLA_D_ROPE + FMLA_NUM_TILES + 1  # 584
    size_per_block_padded = (block_size * bytes_per_token + 576 - 1) // 576 * 576
    # Allocate padded (so memory layout matches kernel TMA expectations) but
    # SLICE back to exact ``block_size*bytes_per_token`` so the final view has
    # last-dim = bytes_per_token (kernel asserts on this).
    out = torch.empty((num_blocks, size_per_block_padded), dtype=torch.float8_e4m3fn, device=k_bf16.device)
    out_view = out[:, : block_size * bytes_per_token]

    # nope+rope region
    nope_rope_part = out_view[:, : block_size * (FMLA_D_NOPE + 2 * FMLA_D_ROPE)].view(
        num_blocks, block_size, FMLA_D_NOPE + 2 * FMLA_D_ROPE
    )
    nope = nope_rope_part[:, :, :FMLA_D_NOPE]
    rope = nope_rope_part[:, :, FMLA_D_NOPE:].view(k_bf16.dtype)

    sf = (
        out_view[:, block_size * (FMLA_D_NOPE + 2 * FMLA_D_ROPE) :]
        .view(num_blocks, block_size, 8)[:, :, :7]
        .view(torch.float8_e8m0fnu)
    )

    rope[:] = k[..., FMLA_D_NOPE:]
    for tile_idx in range(FMLA_NUM_TILES):
        s, e = tile_idx * FMLA_TILE_SIZE, (tile_idx + 1) * FMLA_TILE_SIZE
        scale_inv = (k[..., s:e].abs().float().amax(dim=-1).float() / 448.0).clamp_min(1e-4)
        scale_inv = torch.pow(2, scale_inv.log2().ceil()).to(torch.float32)
        sf[:, :, tile_idx] = scale_inv.to(torch.float8_e8m0fnu)

        scale_inv = scale_inv.unsqueeze(-1)
        nope[:, :, s:e] = (k[..., s:e].float() / scale_inv.float()).to(torch.float8_e4m3fn)

    # Reshape sliced (unpadded) view to (num_blocks, block_size, 1, bytes_per_token)
    return out_view.view(num_blocks, block_size, 1, bytes_per_token)


# ═══════════════════════════════════════════════════════════════════════
# Kernel 1: deep_gemm.fp8_paged_mqa_logits
# ═══════════════════════════════════════════════════════════════════════


def _bench_paged_mqa_logits(M: int, past_kv: int, *, batch_size: int = 1, device: str = "cuda:0") -> float:  # noqa: N803
    """Benchmark paged_mqa_logits.

    Note: the SM90 kernel imposes ``next_n ≤ 2`` (smem capacity); larger M
    must be split into multiple per-request entries.  For prefill chunks
    with many tokens we map ``M → batch_dim`` and keep ``next_n=1``.  Work
    is identical (M x full_c4 x n_idx_heads x idx_head_dim) — only the
    request grouping changes.  In real serving, sglang's
    ``fp8_paged_mqa_logits_chunked`` wraps the same idea (chunk along M).
    """
    from deep_gemm import fp8_paged_mqa_logits, get_paged_mqa_logits_metadata

    del batch_size  # ignored — we treat each new token as its own batch entry
    full_s = M + past_kv
    full_c4 = max(1, full_s // 4)
    block_kv = PAGE_SIZE_C4

    b = M
    next_n = 1

    # Q: (b, 1, num_heads, head_dim) → fp8
    q_bf16 = torch.randn(b, next_n, N_IDX_HEADS, IDX_HEAD_DIM, dtype=torch.bfloat16, device=device)
    q = q_bf16.to(torch.float8_e4m3fn)

    # KV cache: SHARED across all b "fake-batch" entries (avoid b-fold blowup
    # at long past_kv).  Different entries' block_tables all point at the
    # same physical blocks — kernel just reads the same KV M times.
    blocks_per_req = (full_c4 + block_kv - 1) // block_kv
    kv_bf16 = torch.randn(blocks_per_req, block_kv, 1, IDX_HEAD_DIM, dtype=torch.bfloat16, device=device)
    kv_in = _kv_cache_cast_to_fp8_indexer(kv_bf16)

    weights = torch.randn(b * next_n, N_IDX_HEADS, dtype=torch.float32, device=device)

    # Per-token causal context_lens — matches sglang's ``seq_lens_casual``
    # (deepseek_v4_backend_radix.py:1124).  Each new token i has effective
    # past+i+1 KV positions (causal); c4_seq_lens = (past+i+1) // 4.
    # Without this, each query scans the full c4 cache uniformly →
    # benchmark work is M x full_c4 instead of triangular M x full_c4 / 2.
    causal_seq = torch.arange(past_kv + 1, past_kv + M + 1, dtype=torch.int32, device=device)
    causal_c4 = (causal_seq // 4).clamp(min=1)  # min=1 to avoid empty scans
    context_lens = causal_c4.view(b, next_n)

    # All requests reuse the same block list [0..blocks_per_req-1]
    block_table = torch.arange(blocks_per_req, dtype=torch.int32, device=device)
    block_table = block_table.unsqueeze(0).expand(b, blocks_per_req).contiguous()

    schedule_meta = get_paged_mqa_logits_metadata(context_lens, block_kv, NUM_SMS_H20)

    def kernel_fn():
        return fp8_paged_mqa_logits(q, kv_in, weights, context_lens, block_table, schedule_meta, int(full_c4), False)

    return _bench_cuda_graph(kernel_fn, device=device)


# ═══════════════════════════════════════════════════════════════════════
# Helpers for FlashMLA HCA
# ═══════════════════════════════════════════════════════════════════════


def _build_flash_mla_inputs(
    M: int,  # noqa: N803
    past_kv: int,
    *,
    K_per_query: int,  # noqa: N803
    batch_size: int,
    n_local_heads: int,
    device: str,
) -> tuple[torch.Tensor, ...]:
    """Build fp8 paged K cache + Q + indices + scheduler metadata.

    Layout = MODEL1_FP8Sparse (V4-Flash NSA): d_qk=512 with 584-byte fp8 cache.
    """
    full_s = M + past_kv
    M_per_req = M // batch_size if batch_size > 1 else M  # noqa: N806
    K_per_query = max(min(K_per_query, full_s), 1)  # noqa: N806

    # Q: (batch, M_per_req, n_local_heads, FMLA_D_QK=512) bf16
    q = torch.randn(batch_size, M_per_req, n_local_heads, FMLA_D_QK, dtype=torch.bfloat16, device=device)

    # K cache: SHARED across batch entries to avoid b-fold blowup.
    blocks_per_req = (full_s + PAGE_SIZE_FULL - 1) // PAGE_SIZE_FULL
    k_bf16 = torch.randn(blocks_per_req, PAGE_SIZE_FULL, 1, FMLA_D_QK, dtype=torch.bfloat16, device=device)
    k_cache = _quantize_k_cache_model1(k_bf16)

    block_table = torch.arange(blocks_per_req, dtype=torch.int32, device=device)
    block_table = block_table.unsqueeze(0).expand(batch_size, blocks_per_req).contiguous()

    # indices_in_kvcache: (batch, M_per_req, K_per_query) int32 — first K_per_query positions per Q
    base = torch.arange(K_per_query, dtype=torch.int32, device=device)
    indices = base.view(1, 1, K_per_query).expand(batch_size, M_per_req, K_per_query).contiguous()

    # Pad indices to multiple of 64 (FlashMLA assertion)
    if K_per_query % 64 != 0:
        pad = 64 - K_per_query % 64
        pad_t = torch.full((batch_size, M_per_req, pad), -1, dtype=torch.int32, device=device)
        indices = torch.cat([indices, pad_t], dim=-1).contiguous()

    cache_seqlens = torch.full((batch_size,), full_s, dtype=torch.int32, device=device)

    return q, k_cache, block_table, indices, cache_seqlens


# ═══════════════════════════════════════════════════════════════════════
# Kernel 2 (HCA): flash_mla_with_kvcache
# ═══════════════════════════════════════════════════════════════════════


def _bench_flash_mla_sparse(
    M: int,  # noqa: N803
    past_kv: int,
    *,
    K_per_query: int,  # noqa: N803
    batch_size: int = 1,
    tp_size: int = 1,
    device: str = "cuda:0",
) -> float:
    """Benchmark sparse FlashMLA matching V4's V4 backend call shape.

    Real V4 HCA backend (deepseek_v4_backend_radix.py:1087+) passes:
      - main K cache + indices = SWA window (128 fixed positions)
      - extra K cache + indices = c128 (HCA) positions
    Total K attended per query = SWA_WINDOW + extra_K_per_query.

    TP zero-pad (mirrors ``sglang/srt/models/deepseek_v4.py:847``):
      1. Projection produces ``q_local`` of shape (..., 64//tp, d_qk) —
         the rank's actual computed heads.
      2. ``q_padded`` is allocated full (..., 64, d_qk) and the rank's
         ``tp_slice`` is filled from ``q_local``; other heads are zeros.
      3. FlashMLA always receives h_q=64 (kernel only supports {64, 128}).
    """
    from flash_mla import flash_mla_with_kvcache, get_mla_metadata

    # rank-local head count (what the upstream projection actually produces)
    n_local_heads = max(1, N_HEADS_Q // tp_size)
    _full_s = M + past_kv
    M_per_req = M // batch_size if batch_size > 1 else M  # noqa: N806

    # Build main K cache + ``q_local`` (per-rank Q at n_local_heads)
    q_local, k_cache_main, _, _, cache_seqlens = _build_flash_mla_inputs(
        M,
        past_kv,
        K_per_query=K_per_query,
        batch_size=batch_size,
        n_local_heads=n_local_heads,
        device=device,
    )

    # Zero-pad ``q_local`` to full N_HEADS_Q before the FlashMLA call —
    # this is what sglang's deepseek_v4.py:847 does at TP > 1, and at
    # TP = 1 it's a no-op since n_local_heads == N_HEADS_Q.  FlashMLA
    # only supports h_q ∈ {64, 128}, so passing the unpadded q_local
    # (e.g. h_q=8 at tp=8) trips ``Unsupported h_q`` regardless of TP.
    if n_local_heads == N_HEADS_Q:
        q = q_local
    else:
        q = torch.zeros(batch_size, M_per_req, N_HEADS_Q, FMLA_D_QK, dtype=torch.bfloat16, device=device)
        q[:, :, :n_local_heads, :] = q_local  # rank-0's tp_slice

    # Kernel always sees full h_q.
    n_local_heads = N_HEADS_Q
    swa_window = 128
    swa_indices = torch.arange(swa_window, dtype=torch.int32, device=device)
    swa_indices = swa_indices.view(1, 1, swa_window).expand(batch_size, M_per_req, swa_window).contiguous()
    swa_topk_lengths = torch.full((batch_size,), swa_window, dtype=torch.int32, device=device)

    # Build extra K cache (c128 or c4) + extra indices
    extra_blocks = (K_per_query + PAGE_SIZE_FULL - 1) // PAGE_SIZE_FULL
    extra_k_bf16 = torch.randn(max(1, extra_blocks), PAGE_SIZE_FULL, 1, FMLA_D_QK, dtype=torch.bfloat16, device=device)
    extra_k_cache = _quantize_k_cache_model1(extra_k_bf16)
    extra_K = max(64, ((K_per_query + 63) // 64) * 64)  # noqa: N806
    extra_base = torch.arange(extra_K, dtype=torch.int32, device=device)
    extra_indices = extra_base.view(1, 1, extra_K).expand(batch_size, M_per_req, extra_K).contiguous()
    extra_topk_lengths = torch.full((batch_size,), K_per_query, dtype=torch.int32, device=device)

    sched_meta, _ = get_mla_metadata(
        cache_seqlens=cache_seqlens,
        num_q_tokens_per_head_k=M_per_req * n_local_heads,
        num_heads_k=1,
        num_heads_q=n_local_heads,
        is_fp8_kvcache=True,
        topk=swa_indices.size(-1),
    )

    softmax_scale = 1.0 / (FMLA_D_QK**0.5)
    attn_sink = torch.zeros(n_local_heads, dtype=torch.float32, device=device)

    def kernel_fn():
        flash_mla_with_kvcache(
            q=q,
            k_cache=k_cache_main,
            block_table=None,
            cache_seqlens=None,
            head_dim_v=V_HEAD_DIM,
            tile_scheduler_metadata=sched_meta,
            num_splits=None,
            softmax_scale=softmax_scale,
            causal=False,
            is_fp8_kvcache=True,
            indices=swa_indices,
            topk_length=swa_topk_lengths,
            attn_sink=attn_sink,
            extra_k_cache=extra_k_cache,
            extra_indices_in_kvcache=extra_indices,
            extra_topk_length=extra_topk_lengths,
        )

    return _bench_cuda_graph(kernel_fn, device=device)


def _bench_hca_attn(M: int, past_kv: int, *, batch_size: int = 1, tp_size: int = 1, device: str = "cuda:0") -> float:  # noqa: N803
    """HCA: each Q attends to all c128 positions (no topk cap)."""
    full_s = M + past_kv
    K_per_query = max(1, full_s // 128)  # noqa: N806
    return _bench_flash_mla_sparse(
        M,
        past_kv,
        K_per_query=K_per_query,
        batch_size=batch_size,
        tp_size=tp_size,
        device=device,
    )


_BENCH_FN = {
    "paged_mqa_logits": _bench_paged_mqa_logits,
    "hca_attn": _bench_hca_attn,
}


# ═══════════════════════════════════════════════════════════════════════
# CSV write helper
# ═══════════════════════════════════════════════════════════════════════


def _make_perf_filename(kernel: str, output_path: str) -> str:
    if os.path.isdir(output_path) or not output_path.endswith(".txt"):
        return os.path.join(output_path, KERNEL_TO_DEFAULT_FILENAME[kernel])
    return output_path


def _write_row(
    perf_filename: str,
    *,
    kernel: str,
    bs: int,
    isl: int,
    past_kv: int,
    tp_size: int,
    latency_ms: float,
    device_name: str,
    model_path: str = DEFAULT_MODEL,
    architecture: str = DEFAULT_ARCHITECTURE,
    power_stats: dict | None = None,
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(perf_filename)) or ".", exist_ok=True)

    mla_dtype = "bfloat16" if kernel == "hca_attn" else "fp8_e4m3"
    kv_cache_dtype = "fp8_e4m3"
    gemm_type = "fp8_block"

    log_perf(
        item_list=[
            {
                "model": model_path,
                "architecture": architecture,
                "mla_dtype": mla_dtype,
                "kv_cache_dtype": kv_cache_dtype,
                "gemm_type": gemm_type,
                "num_heads": N_HEADS_Q,
                "batch_size": bs,
                "isl": isl,
                "tp_size": tp_size,
                "step": past_kv,
                "compress_ratio": KERNEL_TO_COMPRESS_RATIO[kernel],
                "latency": f"{latency_ms:.6f}",
            }
        ],
        framework="SGLang",
        version="kernel-level",
        device_name=device_name,
        op_name=KERNEL_TO_OP_NAME[kernel],
        kernel_source=KERNEL_TO_KERNEL_SOURCE[kernel],
        perf_filename=perf_filename,
        power_stats=power_stats,
    )


# ═══════════════════════════════════════════════════════════════════════
# Worker
# ═══════════════════════════════════════════════════════════════════════
# Test cases (``get_dsv4_flash_{paged_mqa_logits,hca_attn}_test_cases`` and
# ``_build_sparse_test_cases``) are imported from ``dsv4_flash_test_cases``
# at the top of this module — kept central so both collectors share the
# same sweep grid definitions.


def run_dsv4_sparse_kernel_worker(
    bs: int,
    isl: int,
    past_kv: int,
    tp_size: int,
    kernel: str,
    model_path: str,
    *,
    perf_filename: str,
    device: str = "cuda:0",
):
    """Worker invoked by collect.py.

    Tuple positional args come from get_func; ``perf_filename`` is bound by
    collect.py via functools.partial.

    Internal mapping to bench functions:
      - paged_mqa_logits : M = bs x isl  (work depends only on total new
        tokens; ``b=M, next_n=1`` workaround for SM90 smem limit on next_n)
      - hca_attn         : pass batch_size=bs and M=bsxisl directly
        (``_build_flash_mla_inputs`` derives M_per_req=isl).
    """
    if kernel not in _BENCH_FN:
        raise ValueError(f"unknown kernel={kernel}; expected one of {list(_BENCH_FN)}")

    # The OpEntry binds a single ``perf_filename`` (placeholder
    # ``dsv4_flash_sparse_module_perf.txt``) but we collect TWO kernels in
    # one op — always derive the directory from the bound path and dispatch
    # to ``dsv4_flash_{kernel}_module_perf.txt`` per the case's ``kernel``.
    output_dir = os.path.dirname(perf_filename) or os.getcwd()
    perf_path = _make_perf_filename(kernel, output_dir)

    M = bs * isl  # noqa: N806
    print(f"[dsv4-sparse {kernel}] bs={bs} isl={isl} past_kv={past_kv} tp={tp_size} (M={M}) → {perf_path}")

    bench_fn = _BENCH_FN[kernel]
    if kernel == "hca_attn":
        kwargs = dict(batch_size=bs, tp_size=tp_size, device=device)
    else:
        # paged_mqa_logits: bs at kernel level is flattened to b=M, next_n=1
        kwargs = dict(batch_size=1, device=device)

    try:
        bench_result = bench_fn(M, past_kv, **kwargs)
    except torch.cuda.OutOfMemoryError:
        print(f"  OOM at bs={bs} isl={isl} past_kv={past_kv}; skipping")
        torch.cuda.empty_cache()
        return
    except Exception:
        traceback.print_exc()
        print(f"  failed at bs={bs} isl={isl} past_kv={past_kv}; skipping")
        torch.cuda.empty_cache()
        return

    latency_ms = float(bench_result["latency_ms"])
    power_stats = bench_result.get("power_stats")
    device_name = torch.cuda.get_device_name(device)
    _write_row(
        perf_path,
        kernel=kernel,
        bs=bs,
        isl=isl,
        past_kv=past_kv,
        tp_size=tp_size,
        latency_ms=latency_ms,
        device_name=device_name,
        model_path=model_path,
        power_stats=power_stats,
    )
    power_str = f", power={power_stats['power']:.1f}W" if power_stats and power_stats.get("power") is not None else ""
    print(f"  latency={latency_ms:.4f} ms{power_str}")


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════


def _parse_int_list(value: str) -> list[int]:
    return [int(x) for x in value.split(",") if x.strip()]


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect DeepSeek-V4-Flash sparse-attention kernel-level latency.")
    parser.add_argument("--kernel", default="all", help=f"comma-separated subset of {KERNELS} (or 'all')")
    parser.add_argument("--bs-list", type=_parse_int_list, default=DEFAULT_BS_LIST)
    parser.add_argument("--isl-list", type=_parse_int_list, default=DEFAULT_ISL_LIST)
    parser.add_argument("--past-kv-list", type=_parse_int_list, default=DEFAULT_PAST_KV_LIST)
    parser.add_argument(
        "--tp-list-attn", type=_parse_int_list, default=DEFAULT_TP_LIST_ATTN, help="TP sizes for hca_attn"
    )
    parser.add_argument("--output-path", default=os.getcwd())
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    return parser


def main():
    args = _build_arg_parser().parse_args()

    if args.kernel == "all":
        kernels = list(KERNELS)
    else:
        kernels = [k.strip() for k in args.kernel.split(",") if k.strip()]
        for k in kernels:
            if k not in KERNELS:
                raise SystemExit(f"unknown kernel '{k}'; expected one of {KERNELS}")

    cases = _build_sparse_test_cases(
        kernels=kernels,
        bs_list=args.bs_list,
        isl_list=args.isl_list,
        past_kv_list=args.past_kv_list,
        tp_list_attn=args.tp_list_attn,
    )
    print(f"Running {len(cases)} sparse-kernel test cases on {args.device}")
    for case in cases:
        bs, isl, past_kv, tp, kernel = case[:5]
        perf_path = _make_perf_filename(kernel, args.output_path)
        run_dsv4_sparse_kernel_worker(
            bs,
            isl,
            past_kv,
            tp,
            kernel,
            args.model_path,
            perf_filename=perf_path,
            device=args.device,
        )


if __name__ == "__main__":
    main()
