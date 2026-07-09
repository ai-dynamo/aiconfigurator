# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V4 sparse-attention kernel-level collector for SGLang.

Collects the DeepSeek-V4 NSA sparse-attention sub-kernels at the kernel level.
DeepSeek-V4's primary AIC path uses prefix-aware full CSA/HCA attention-module
rows; paged_mqa_logits / csa_attn / hca_attn here are supporting kernel-level
data for prefix/past_kv correction and residual analysis, while the topk_calib
DELTA is actively consumed by perf_database's topK correction.

The four sub-kernels form ONE sparse-op family — the CSA/HCA sparse path in
order — all collected, none modeled analytically:

    1. ``deep_gemm.fp8_paged_mqa_logits``      CSA indexer scoring
    2. ``topk_transform`` (topk_512/_1024)     CSA selection — flat-vs-top_last
                                               DELTA calib (flat_ms/top_last_ms)
    3. ``flash_mla_with_kvcache`` (csa_attn)   CSA sparse FMLA over the
                                               topk-selected c4 positions
    4. ``flash_mla_with_kvcache`` (hca_attn)   HCA c128 sparse FMLA

Inputs are SAME-SOURCE: every sub-kernel derives its ``(prefix, isl, bs[, tp])``
shapes STRICTLY 1:1 from the CSA/HCA attention-module CSV it belongs to
(paged_mqa / topk / csa_attn ← CSA module; hca_attn ← HCA module) — no separate
sweep grid — benched at upstream layouts (DeepGEMM ``test_attention.py`` for the
indexer GEMM, FlashMLA ``MODEL1_FP8Sparse`` quant for the FMLA kernel).

CSV schema matches existing aic dsv4 module CSVs (so loaders can be shared):
``isl`` carries M, ``step`` carries past_kv, ``compress_ratio`` distinguishes
CSA(=4) / HCA(=128).
"""

# Requires an SGLang build with DeepSeek-V4 support. Stock lmsysorg/sglang:v*
# images may not include the required deepseek_v4 modules; use a
# deepseek-v4-blackwell/deepseek-v4-grace-blackwell image or matching Dynamo
# sglang-runtime:*deepseek-v4* image.
from __future__ import annotations

import functools
import importlib.util
import json
import os
import sys
import traceback
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import torch

try:
    from collector.sglang.helper import benchmark_with_power, log_perf
except ModuleNotFoundError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from helper import benchmark_with_power, log_perf

# Re-export test case generators from the centralized case generator
# module so collect.py's registry can resolve them via getattr on this module.
try:
    from collector.case_generator import _DSV4_DEFAULT_MODELS
except ModuleNotFoundError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from case_generator import _DSV4_DEFAULT_MODELS


def _dsv4_sparse_kernel_cases(kernel):
    # One task per (model, bs); collect.py spreads bs across GPU workers (no
    # single-worker cuda-graph buildup). All sparse kernels run a single fixed
    # head config: the FMLA pads to its required head count OUTSIDE the kernel
    # (model TP zero-pad), so the kernel is TP-independent -> one config per bs,
    # no tp sweep. Each task sweeps (isl, prefix) for its bs.
    try:
        from collector.case_generator import _selected_dsv4_models
    except ModuleNotFoundError:
        from case_generator import _selected_dsv4_models
    cases = []
    for m in _selected_dsv4_models():
        ctx = _dsv4_context_derived_shapes(m)
        dec = _dsv4_generation_derived_shapes(m)
        bss = sorted({b for (_p, _i, b) in ctx} | {b for (_p, _i, b) in dec})
        cases.extend([m, kernel, b] for b in bss)
    return cases


def get_dsv4_paged_mqa_logits_test_cases():
    if not _dsv4_sparse_kernel_supported("paged_mqa_logits"):
        return []
    return _dsv4_sparse_kernel_cases("paged_mqa_logits")


def get_dsv4_hca_attn_test_cases():
    if not _dsv4_sparse_kernel_supported("hca_attn"):
        return []
    return _dsv4_sparse_kernel_cases("hca_attn")


def get_dsv4_csa_attn_test_cases():
    if not _dsv4_sparse_kernel_supported("csa_attn"):
        return []
    return _dsv4_sparse_kernel_cases("csa_attn")


get_dsv4_flash_paged_mqa_logits_test_cases = get_dsv4_paged_mqa_logits_test_cases
get_dsv4_flash_hca_attn_test_cases = get_dsv4_hca_attn_test_cases


__all__ = [
    "DEFAULT_MODEL",
    "get_dsv4_csa_attn_test_cases",
    "get_dsv4_flash_hca_attn_test_cases",
    "get_dsv4_flash_paged_mqa_logits_test_cases",
    "get_dsv4_hca_attn_test_cases",
    "get_dsv4_paged_mqa_logits_test_cases",
    "get_dsv4_topk_calib_test_cases",
    "run_dsv4_sparse_kernel_worker",
]


DEFAULT_MODEL = _DSV4_DEFAULT_MODELS[0]
MODEL_CONFIGS_DIR = (
    Path(__file__).resolve().parents[2].parent / "aiconfigurator-core" / "src" / "aiconfigurator_core" / "model_configs"
)

# ═══════════════════════════════════════════════════════════════════════
# Model config (single source for all DSV4 model-specific shapes)
# ═══════════════════════════════════════════════════════════════════════


def _dsv4_model_config(model_path: str) -> dict:
    """Load a DSV4 model config json — the single source of model shapes.

    Accepts a local model directory (``<dir>/config.json``) or an HF-style id
    resolved under ``MODEL_CONFIGS_DIR``. Hard error if absent; these values
    MUST NOT be guessed/defaulted.
    """
    if model_path and os.path.isdir(model_path):
        path = Path(model_path) / "config.json"
    else:
        path = MODEL_CONFIGS_DIR / ((model_path or "").replace("/", "--") + "_config.json")
    if not path.is_file():
        raise FileNotFoundError(f"DSV4 model config not found at {path} for {model_path!r}")
    return json.loads(path.read_text())


def _dsv4_cfg_int(cfg: dict, field: str) -> int:
    """Required int field from a loaded config (hard error if missing/empty)."""
    if cfg.get(field) in (None, ""):
        raise KeyError(f"{field} missing in DSV4 model config")
    return int(cfg[field])


@functools.cache
def _dsv4_sparse_config(model_path: str) -> SimpleNamespace:
    """All sparse-attention kernel shapes, extracted from the model config the
    SAME way sglang's MQALayer / C4Indexer ``__init__`` read them
    (models/deepseek_v4.py) — the single source for every sparse-kernel shape,
    no scattered reads or magic numbers.

    NSA compress-ratio layer types are exactly {0, 4, 128} (sglang asserts this
    and selects the CSA/indexer path with the literal ``compress_ratio == 4``);
    the topk / paged-mqa indexer is that c4 path, so ``csa_compress_ratio`` is 4
    by architecture (validated against the config, not min()/heuristic).
    ``page_size`` and the fp8 quant tile are kernel-layout invariants (not in the
    config) and remain module constants below.
    """
    cfg = _dsv4_model_config(model_path)
    compress = {int(r) for r in (cfg.get("compress_ratios") or [])}
    if not compress <= {0, 4, 128}:
        raise ValueError(
            f"unexpected compress_ratios {sorted(compress)} for {model_path!r}; "
            "DSV4 NSA expects a subset of {0, 4, 128}"
        )
    if 4 not in compress:
        raise KeyError(f"no CSA layer (compress_ratio=4) in config for {model_path!r}; got {sorted(compress)}")
    head_dim = _dsv4_cfg_int(cfg, "head_dim")
    rope = _dsv4_cfg_int(cfg, "qk_rope_head_dim")
    return SimpleNamespace(
        head_dim=head_dim,  # FlashMLA d_qk / V_HEAD_DIM (512)
        d_rope=rope,  # FlashMLA d_rope (64)
        d_nope=head_dim - rope,  # FlashMLA d_nope (448)
        num_attention_heads=_dsv4_cfg_int(cfg, "num_attention_heads"),
        index_n_heads=_dsv4_cfg_int(cfg, "index_n_heads"),
        index_head_dim=_dsv4_cfg_int(cfg, "index_head_dim"),
        index_topk=_dsv4_cfg_int(cfg, "index_topk"),  # V4-Pro=1024, V4-Flash=512
        csa_compress_ratio=4,  # sglang: compress_ratio == 4 is the CSA/indexer path
        sliding_window=_dsv4_cfg_int(cfg, "sliding_window"),  # HCA SWA window (128)
    )


# ═══════════════════════════════════════════════════════════════════════
# DeepSeek-V4 sparse architectural constants
# ═══════════════════════════════════════════════════════════════════════
# Shape fields below are identical across DSV4 variants (Flash/Pro), so the
# default model config is authoritative; read them via the sparse config.
_DEFAULT_SC = _dsv4_sparse_config(DEFAULT_MODEL)
V_HEAD_DIM = _DEFAULT_SC.head_dim  # 512
FMLA_D_QK = V_HEAD_DIM  # FlashMLA d_qk == head_dim (NOT 576)
FMLA_D_ROPE = _DEFAULT_SC.d_rope  # 64
FMLA_D_NOPE = _DEFAULT_SC.d_nope  # 448

# Kernel-layout invariants — NOT in the model config. The FlashMLA
# MODEL1_FP8Sparse fp8 quant tile size and the deep_gemm / FlashMLA paged block
# size are hardcoded in the kernels themselves (deep_gemm blocksize 64;
# flashmla_backend.PAGE_SIZE = 64), so they are fixed constants here.
# bytes_per_token = d_nope + d_rope*2 + num_tiles + 1 pad = 448 + 128 + 7 + 1 = 584
FMLA_TILE_SIZE = 64
FMLA_NUM_TILES = FMLA_D_NOPE // FMLA_TILE_SIZE  # 448 / 64 = 7

DEFAULT_ARCHITECTURE = "DeepseekV4ForCausalLM"


@functools.lru_cache(maxsize=1)
def _dsv4_kv_page_size() -> int:
    """DSV4 KV-pool paged block size — single source from sglang.

    The deep_gemm indexer (paged_mqa_logits block_kv) and FlashMLA paged
    attention share ONE KV-pool page size: sglang's DSA indexer asserts
    get_token_to_kv_pool().page_size == 64 and FlashMLA uses the same value
    (flashmla_backend.PAGE_SIZE). Imported lazily — flashmla_backend pulls in
    the whole MLA attention backend, which must not run at collector
    module-import time.
    """
    from sglang.srt.layers.attention.flashmla_backend import PAGE_SIZE

    return int(PAGE_SIZE)


def _device_num_sms(device: str | torch.device) -> int:
    """Return the actual SM count of ``device`` (the kernel sizes
    ``schedule_meta`` from this and asserts ``_schedule_meta_size == num_sms + 1``)."""
    return torch.cuda.get_device_properties(device).multi_processor_count


# The DSV4 sparse-op family — all four collected by ONE worker
# (run_dsv4_sparse_kernel_worker), all driven by the KERNEL_* maps below, all
# same-source (shapes read 1:1 from the CSA/HCA attention-module CSV they belong
# to), all writing the SAME row schema (one ``latency`` per row):
#   - paged_mqa_logits: CSA indexer scoring (deep_gemm)            -> 1 row/shape
#   - topk:           CSA selection — benched under two score
#                     distributions, emitted as TWO rows/shape
#                     (score_mode=flat | top_last); perf_database
#                     takes DELTA = flat.latency - top_last.latency -> 2 rows/shape
#   - csa_attn:       CSA's flash_mla over the topk-selected c4
#                     positions (K_per_query = min(index_topk, full_s//4))
#   - hca_attn:       HCA's flash_mla over c128 cache.
#                     (production DSV4 at TP>1 also runs FlashMLA with
#                      the native h_q — sglang pads Q to full native heads,
#                      then slices output back; see deepseek_v4.py:847.  So
#                      TP=1 data is valid for any deployment TP.)
KERNEL_TO_OP_NAME = {
    "paged_mqa_logits": "dsv4_paged_mqa_logits_module",
    "topk": "dsv4_csa_topk_calib",
    "hca_attn": "dsv4_hca_attn_module",
    "csa_attn": "dsv4_csa_attn_module",
}

KERNEL_TO_KERNEL_SOURCE = {
    "paged_mqa_logits": "deep_gemm.fp8_paged_mqa_logits",
    "topk": "topk_transform_v2",
    "hca_attn": "flash_mla_with_kvcache",
    "csa_attn": "flash_mla_with_kvcache",
}

# compress_ratio: 4 for the CSA c4 path (indexer/topk/csa_attn), 128 for HCA c128.
KERNEL_TO_COMPRESS_RATIO = {
    "paged_mqa_logits": 4,
    "topk": 4,
    "hca_attn": 128,
    "csa_attn": 4,
}

# Each sparse sub-kernel derives its (prefix, isl, bs) shapes from its owning
# attention module's INPUT sweep (paged_mqa/topk/csa_attn <- CSA, hca_attn <-
# HCA) via _dsv4_context_derived_shapes / _dsv4_generation_derived_shapes —
# NOT from the already-collected module CSVs.
# ALL four sparse kernels are TP-independent — the indexer/FMLA latency does not
# change with tp (the module's per-tp rows differ only by the attention module's
# comms/sharding, not the sparse sub-kernel). So each is benched ONCE per unique
# (prefix, isl, bs) module shape and written one row per shape (tp_size=1),
# matching how the SDK consumes them (keyed by shape, not tp) — no per-tp
# expansion. (topk additionally emits two score_mode rows per shape.)


def _dsv4_sparse_kernel_supported(kernel: str) -> bool:
    """Return True when the active runtime can execute a DSV4 sparse kernel."""
    if os.environ.get("COLLECTOR_FORCE_DSV4_SPARSE") == "1":
        return True
    if kernel in ("hca_attn", "csa_attn"):
        return importlib.util.find_spec("flash_mla") is not None
    if kernel == "paged_mqa_logits":
        if importlib.util.find_spec("deep_gemm") is None:
            return False
        if torch.cuda.is_available():
            major, _minor = torch.cuda.get_device_capability()
            if major >= 12:
                return False
        return True
    raise ValueError(f"unknown DSV4 sparse kernel: {kernel}")


# ═══════════════════════════════════════════════════════════════════════
# Bench helper
# ═══════════════════════════════════════════════════════════════════════


def _bench_cuda_graph(
    kernel_fn: Callable[[], None],
    *,
    num_warmup: int = 5,
    num_iterations: int = 20,
    graph_repeat: int = 4,
    allow_graph_fail: bool = False,
    device: str = "cuda:0",
) -> dict:
    """Benchmark a kernel via AIC's benchmark_with_power helper (the single
    bench path for all DSV4 sparse kernels).

    benchmark_with_power handles warmup, CUDA-Graph capture/replay, optional
    power sampling, and graph-private-pool teardown. With ``allow_graph_fail``
    False, CUDA-graph capture is mandatory (used_cuda_graph is asserted); set it
    True for kernels whose graph capture may fall back to eager (e.g. the JIT
    topk_transform whose host-side planning is done outside the timed region).
    Returns ``{"latency_ms", "power_stats"}``.
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
        allow_graph_fail=allow_graph_fail,
    ) as result:
        pass

    if not allow_graph_fail and not result.get("used_cuda_graph", False):
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
    """FlashMLA MODEL1_FP8Sparse pack (DSV4 sparse layout).

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

    # Return a view whose stride(0) is the padded per-block size — FlashMLA's
    # MODEL1 path asserts ``k_cache.stride(0) % TMA_K_STRIDE == 0`` (with
    # TMA_K_STRIDE = D_NOPE + 2*D_ROPE = 576), and ``bytes_per_token * block_size``
    # = 37376 is *not* a multiple of 576, so we need the per-block padding to be
    # visible in the tensor's stride. ``.view`` collapses stride(0) to the
    # contiguous value when num_blocks == 1, breaking the assertion for small
    # shapes; ``as_strided`` lets us pin stride(0) to size_per_block_padded for
    # any num_blocks.
    return out.as_strided(
        size=(num_blocks, block_size, 1, bytes_per_token),
        stride=(size_per_block_padded, bytes_per_token, bytes_per_token, 1),
        storage_offset=0,
    )


def _expand_block_table(num_blocks: int, batch_size: int, device) -> torch.Tensor:
    """Block list [0..num_blocks) replicated to (batch_size, num_blocks) int32.
    All requests share the same physical blocks (KV cache is shared in-bench)."""
    row = torch.arange(num_blocks, dtype=torch.int32, device=device)
    return row.unsqueeze(0).expand(batch_size, num_blocks).contiguous()


def _expand_indices(k: int, batch_size: int, m: int, device) -> torch.Tensor:
    """Sequential KV indices [0..k) replicated to (batch_size, m, k) int32."""
    base = torch.arange(k, dtype=torch.int32, device=device)
    return base.view(1, 1, k).expand(batch_size, m, k).contiguous()


# ═══════════════════════════════════════════════════════════════════════
# Kernel 1: deep_gemm.fp8_paged_mqa_logits
# ═══════════════════════════════════════════════════════════════════════


def _bench_paged_mqa_logits(
    M: int,  # noqa: N803
    past_kv: int,
    *,
    index_n_heads: int,
    index_head_dim: int,
    batch_size: int = 1,
    device: str = "cuda:0",
) -> float:
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
    block_kv = _dsv4_kv_page_size()

    b = M
    next_n = 1

    # Q: (b, 1, index_n_heads, index_head_dim) → fp8
    q_bf16 = torch.randn(b, next_n, index_n_heads, index_head_dim, dtype=torch.bfloat16, device=device)
    q = q_bf16.to(torch.float8_e4m3fn)

    # KV cache: SHARED across all b "fake-batch" entries (avoid b-fold blowup
    # at long past_kv).  Different entries' block_tables all point at the
    # same physical blocks — kernel just reads the same KV M times.
    blocks_per_req = (full_c4 + block_kv - 1) // block_kv
    kv_bf16 = torch.randn(blocks_per_req, block_kv, 1, index_head_dim, dtype=torch.bfloat16, device=device)
    kv_in = _kv_cache_cast_to_fp8_indexer(kv_bf16)

    weights = torch.randn(b * next_n, index_n_heads, dtype=torch.float32, device=device)

    # Per-token causal context_lens — matches sglang's ``seq_lens_casual``
    # (deepseek_v4_backend_radix.py:1124).  Each new token i has effective
    # past+i+1 KV positions (causal); c4_seq_lens = (past+i+1) // 4.
    # Without this, each query scans the full c4 cache uniformly →
    # benchmark work is M x full_c4 instead of triangular M x full_c4 / 2.
    causal_seq = torch.arange(past_kv + 1, past_kv + M + 1, dtype=torch.int32, device=device)
    causal_c4 = (causal_seq // 4).clamp(min=1)  # min=1 to avoid empty scans
    context_lens = causal_c4.view(b, next_n)

    # All requests reuse the same block list [0..blocks_per_req-1]
    block_table = _expand_block_table(blocks_per_req, b, device)

    schedule_meta = get_paged_mqa_logits_metadata(context_lens, block_kv, _device_num_sms(device))

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

    Layout = MODEL1_FP8Sparse (DSV4 NSA): d_qk=512 with 584-byte fp8 cache.
    """
    M_per_req = M // batch_size if batch_size > 1 else M  # noqa: N806
    # Per-request sequence length: each of the ``batch_size`` requests has
    # ``M_per_req`` new query tokens over ``past_kv`` prefix (NOT the flattened
    # total M), so cache_seqlens / indices mirror the owning module
    # (prefix, isl, bs) row 1:1 instead of one giant bs*isl request.
    full_s = M_per_req + past_kv
    K_per_query = max(min(K_per_query, full_s), 1)  # noqa: N806

    # Q: (batch, M_per_req, n_local_heads, FMLA_D_QK=512) bf16
    q = torch.randn(batch_size, M_per_req, n_local_heads, FMLA_D_QK, dtype=torch.bfloat16, device=device)

    # K cache: SHARED across batch entries to avoid b-fold blowup.
    blocks_per_req = (full_s + _dsv4_kv_page_size() - 1) // _dsv4_kv_page_size()
    k_bf16 = torch.randn(blocks_per_req, _dsv4_kv_page_size(), 1, FMLA_D_QK, dtype=torch.bfloat16, device=device)
    k_cache = _quantize_k_cache_model1(k_bf16)

    block_table = _expand_block_table(blocks_per_req, batch_size, device)

    # indices_in_kvcache: (batch, M_per_req, K_per_query) int32 — first K_per_query positions per Q
    indices = _expand_indices(K_per_query, batch_size, M_per_req, device)

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
    native_heads: int,
    sliding_window: int,
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
      1. Projection produces ``q_local`` of shape (..., native_heads//tp, d_qk) —
         the rank's actual computed heads.
      2. ``q_padded`` is allocated full (..., native_heads, d_qk) and the rank's
         ``tp_slice`` is filled from ``q_local``; other heads are zeros.
      3. FlashMLA always receives the full native head count.
    """
    from flash_mla import flash_mla_with_kvcache, get_mla_metadata

    # rank-local head count (what the upstream projection actually produces)
    n_local_heads = max(1, native_heads // tp_size)
    M_per_req = M // batch_size if batch_size > 1 else M  # noqa: N806
    _full_s = M_per_req + past_kv  # per-request sequence length (bs requests, M_per_req new tokens each)

    # Build main K cache + ``q_local`` (per-rank Q at n_local_heads)
    q_local, k_cache_main, _, _, cache_seqlens = _build_flash_mla_inputs(
        M,
        past_kv,
        K_per_query=K_per_query,
        batch_size=batch_size,
        n_local_heads=n_local_heads,
        device=device,
    )

    # Zero-pad ``q_local`` to the full native head count before the FlashMLA
    # call. Passing the unpadded TP-local head count trips ``Unsupported h_q``.
    if n_local_heads == native_heads:
        q = q_local
    else:
        q = torch.zeros(batch_size, M_per_req, native_heads, FMLA_D_QK, dtype=torch.bfloat16, device=device)
        q[:, :, :n_local_heads, :] = q_local  # rank-0's tp_slice

    # Kernel always sees full h_q.
    n_local_heads = native_heads
    swa_window = sliding_window
    swa_indices = _expand_indices(swa_window, batch_size, M_per_req, device)
    swa_topk_lengths = torch.full((batch_size,), swa_window, dtype=torch.int32, device=device)

    # Build extra K cache (c128 or c4) + extra indices
    _pg = _dsv4_kv_page_size()
    extra_blocks = (K_per_query + _pg - 1) // _pg
    extra_k_bf16 = torch.randn(max(1, extra_blocks), _pg, 1, FMLA_D_QK, dtype=torch.bfloat16, device=device)
    extra_k_cache = _quantize_k_cache_model1(extra_k_bf16)
    extra_K = max(64, ((K_per_query + 63) // 64) * 64)  # noqa: N806
    extra_indices = _expand_indices(extra_K, batch_size, M_per_req, device)
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


# HCA and CSA are the SAME FlashMLA kernel (``_bench_flash_mla_sparse``) — they
# differ only in how many KV positions each query attends to:
#   - hca_attn: all c128 positions               -> full_s // 128
#   - csa_attn: the topk-selected c4 positions    -> min(index_topk, full_s // 4)
# (csa uses the same sequential-index approximation as hca; the real topk
# scatter pattern is not reproducible in a kernel-level bench.)
_FMLA_K_PER_QUERY = {
    "hca_attn": lambda full_s, sc: max(1, full_s // 128),
    "csa_attn": lambda full_s, sc: max(1, min(sc.index_topk, full_s // 4)),
}


# ═══════════════════════════════════════════════════════════════════════
# CSV write helper
# ═══════════════════════════════════════════════════════════════════════


def _make_perf_filename(kernel: str, output_path: str) -> str:
    # Default filename derives directly from the op name (op_name + "_perf.txt").
    if os.path.isdir(output_path) or not output_path.endswith(".txt"):
        return os.path.join(output_path, f"{KERNEL_TO_OP_NAME[kernel]}_perf.txt")
    return output_path


def _write_row(
    perf_filename: str,
    *,
    kernel: str,
    bs: int,
    isl: int,
    past_kv: int,
    tp_size: int,
    native_heads: int,
    latency_ms: float,
    device_name: str,
    model_path: str = DEFAULT_MODEL,
    architecture: str = DEFAULT_ARCHITECTURE,
    score_mode: str | None = None,
    power_stats: dict | None = None,
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(perf_filename)) or ".", exist_ok=True)

    mla_dtype = "bfloat16" if kernel in ("hca_attn", "csa_attn") else "fp8_e4m3"
    kv_cache_dtype = "fp8_e4m3"
    gemm_type = "fp8_block"

    item = {
        "model": model_path,
        "architecture": architecture,
        "mla_dtype": mla_dtype,
        "kv_cache_dtype": kv_cache_dtype,
        "gemm_type": gemm_type,
        "num_heads": native_heads,
        "batch_size": bs,
        "isl": isl,
        "tp_size": tp_size,
        "step": past_kv,
        "compress_ratio": KERNEL_TO_COMPRESS_RATIO[kernel],
        "latency": f"{latency_ms:.6f}",
    }
    # topk emits two rows per shape (flat vs top_last); the discriminator column
    # is only present for topk so the single-latency kernels keep their schema.
    if score_mode is not None:
        item["score_mode"] = score_mode

    log_perf(
        item_list=[item],
        framework="SGLang",
        version="kernel-level",
        device_name=device_name,
        op_name=KERNEL_TO_OP_NAME[kernel],
        kernel_source=KERNEL_TO_KERNEL_SOURCE[kernel],
        perf_filename=perf_filename,
        power_stats=power_stats,
    )


# ═══════════════════════════════════════════════════════════════════════
# Worker (invoked by collect.py via the registry; test-case generators live in
# case_generator.py so all collectors share one sweep-grid definition).
# ═══════════════════════════════════════════════════════════════════════


def _guarded_bench(bench_fn: Callable[[], object], label: str):
    """Run a per-shape bench, returning its result or ``None`` on OOM / any
    error (logged, with a CUDA cache flush), so a single bad shape skips rather
    than aborting the whole op."""
    try:
        return bench_fn()
    except torch.cuda.OutOfMemoryError:
        print(f"  OOM at {label}; skipping shape")
        torch.cuda.empty_cache()
        return None
    except Exception:
        traceback.print_exc()
        print(f"  failed at {label}; skipping shape")
        torch.cuda.empty_cache()
        return None


def _bench_sparse_kernel_shape(kernel, prefix, isl, bs, sc, device):
    """Bench one module ``(prefix, isl, bs)`` shape; return a list of
    ``(score_mode, latency_ms)`` rows. TP-independent (paged_mqa flattens bs→b=M;
    FMLA pads to full native heads), so this is computed once per shape.

    Single-latency kernels return one ``(None, latency)``; topk returns the
    flat-vs-representative pair ``[("flat", …), ("top_last", …)]`` whose DELTA
    perf_database consumes (flat.latency - top_last.latency)."""
    M = bs * isl  # noqa: N806
    if kernel in _FMLA_K_PER_QUERY:
        # FMLA mirrors the module row 1:1: bs requests, each isl new query tokens
        # over `prefix` past_kv. Pass batch_size=bs (so M_per_req=isl) and base
        # K_per_query on the PER-REQUEST sequence length (isl+prefix), not the
        # flattened total (bs*isl+prefix).
        full_s = isl + prefix
        lat = _bench_flash_mla_sparse(
            M,
            prefix,
            K_per_query=_FMLA_K_PER_QUERY[kernel](full_s, sc),
            native_heads=sc.num_attention_heads,
            sliding_window=sc.sliding_window,
            batch_size=bs,
            device=device,
        )["latency_ms"]
        return [(None, lat)]
    if kernel == "topk":
        return _bench_topk_shape(prefix, isl, bs, sc, device)
    # paged_mqa_logits: bs flattened to b=M, next_n=1
    lat = _bench_paged_mqa_logits(
        M,
        prefix,
        index_n_heads=sc.index_n_heads,
        index_head_dim=sc.index_head_dim,
        device=device,
    )["latency_ms"]
    return [(None, lat)]


def _derive_context_shapes(bs_list, seq_list, prefix_list, is_valid, env_filter=None):
    """Build unique ``(prefix, isl, bs)`` context shapes from a STATIC sweep grid
    (the owning attention module's INPUT grid), deduped + order-preserving.

    Shared by GLM5 (dsa) and DSV4 (csa/hca): the sparse sub-kernels reconstruct
    the module's intended input grid instead of reading back the rows the full
    module actually survived. The full module drops large isl at runtime (smem /
    KV-pool caps) that the cheap sparse indexer/topk kernels do not hit, so a
    1:1 read of the module CSV under-samples them.

    ``is_valid(bs, isl, prefix) -> bool`` encodes the module's context limits.
    ``env_filter`` (optional) takes the list of ``(bs, isl, prefix)`` tuples and
    returns a filtered list (e.g. an ``AIC_*_CONTEXT_*`` env pin); applied
    before dedup. Iteration is prefix-outer so a late long-prefix shape does not
    reorder the smaller-isl rows.
    """
    cases = [
        (bs, isl, prefix) for prefix in prefix_list for bs in bs_list for isl in seq_list if is_valid(bs, isl, prefix)
    ]
    if env_filter is not None:
        cases = env_filter(cases)
    shapes, seen = [], set()
    for bs, isl, prefix in cases:
        key = (int(prefix), int(isl), int(bs))
        if key not in seen:
            seen.add(key)
            shapes.append(key)
    return shapes


_CHUNKED_PREFILL = None


def _chunked_prefill_size_from_gpu_mem(gpu_mem):
    """Replicate ``ServerArgs._handle_gpu_memory_settings`` chunked_prefill_size
    tiering directly from device memory (MB). Thresholds are identical across the
    old (cuda_graph_max_bs) and new (cuda_graph_config) sglang server_args, so
    this is the same derivation sglang performs -- used as a forward-compatible
    fallback when the bare ServerArgs instance can't run the method itself.
    B200 (>=160GB) -> 16384, matching the GLM-5.2 model card launch command."""
    if gpu_mem < 60 * 1024:
        return 4096 if gpu_mem >= 35 * 1024 else 2048
    if gpu_mem < 160 * 1024:
        return 8192
    return 16384


def _sglang_chunked_prefill_size():
    # The csa/hca MODULE (collect_dsv4_attn) launches sglang with
    # chunked_prefill_size=None -> sglang DERIVES it from GPU memory. Mirror
    # that here (sglang's own GPU-mem tiering: get_device_memory_capacity +
    # ServerArgs._handle_gpu_memory_settings; 2k/4k/8k/16k by device mem),
    # model-free and weight-free. NOT a hardcoded 8192 and NOT a guessed
    # ServerArgs. chunked_prefill_size depends only on device memory.
    global _CHUNKED_PREFILL
    if _CHUNKED_PREFILL is None:
        try:
            from sglang.srt.server_args import ServerArgs, get_device_memory_capacity
        except ModuleNotFoundError:
            from srt.server_args import ServerArgs, get_device_memory_capacity
        gpu_mem = None
        try:
            gpu_mem = get_device_memory_capacity("cuda")
        except Exception:
            pass
        sa = ServerArgs.__new__(ServerArgs)
        sa.chunked_prefill_size = None
        sa.cuda_graph_max_bs = None
        sa.cuda_graph_bs = None
        sa.tp_size = 1
        sa.device = "cuda"
        try:
            sa._handle_gpu_memory_settings(gpu_mem)
        except Exception:
            pass  # chunked_prefill_size is set first, before any model-dependent step
        chunked = sa.chunked_prefill_size
        if chunked is None and gpu_mem is not None:
            # Newer sglang (0.0.0.dev / >=0.5.x) refactored cuda_graph_max_bs/_bs
            # into a cuda_graph_config object that _handle_gpu_memory_settings
            # dereferences at entry (self.cuda_graph_config.decode), so it
            # AttributeErrors on the bare __new__ instance before assigning
            # chunked_prefill_size. Fall back to sglang's OWN tiering (same
            # thresholds) computed directly from device memory -- still a
            # derivation from GPU memory, not a guess. Backward compatible:
            # on old sglang the method assigns it and this branch is skipped.
            chunked = _chunked_prefill_size_from_gpu_mem(gpu_mem)
        if chunked is None:
            # gpu_mem itself unavailable -> nothing to derive from. Fail loud
            # rather than int(None) or guessing a value.
            raise RuntimeError(
                "Could not derive sglang chunked_prefill_size from GPU memory "
                "(get_device_memory_capacity failed). Cannot derive DSV4 "
                "sparse-kernel context shapes."
            )
        _CHUNKED_PREFILL = int(chunked)
    return _CHUNKED_PREFILL


def _dsv4_context_derived_shapes(model_path):
    """DSV4 sparse-kernel context shapes derived from the csa/hca context MODULE
    INPUT sweep (same source as GLM5's dsa derive) — NOT read back from the
    module CSV. paged_mqa_logits / topk / csa_attn / hca_attn therefore cover
    the full-chunk isl the module drops at runtime; CSA and HCA share one grid
    (the module input validity does not depend on attn_kind).

    DSV4 differs from GLM5 in scale: context new-token budget is the
    chunked-prefill size (bs*isl), per-request context is capped by the model
    max_position_embeddings, and bs*(isl+prefix) is bounded by the KV pool.
    isl==1 (paged decode) shapes are still read from the owning generation CSV
    by the worker.
    """
    try:
        from collector.case_generator import (
            _DSV4_MODULE_BATCH_SIZES,
            _DSV4_MODULE_PAST_KV_LIST,
            _DSV4_MODULE_SEQ_LENGTHS,
            _dsv4_module_is_valid_shape,
        )
    except ModuleNotFoundError:
        from case_generator import (
            _DSV4_MODULE_BATCH_SIZES,
            _DSV4_MODULE_PAST_KV_LIST,
            _DSV4_MODULE_SEQ_LENGTHS,
            _dsv4_module_is_valid_shape,
        )
    cfg = _dsv4_model_config(model_path)
    max_pos = cfg.get("max_position_embeddings")
    try:
        from collector.sglang.runtime_limits import required_kv_tokens
    except ModuleNotFoundError:
        from runtime_limits import required_kv_tokens
    chunk = _sglang_chunked_prefill_size()  # sglang-derived (like the module), NOT hardcoded 8192

    def _valid(bs, isl, prefix):
        # Reuse the csa/hca MODULE's skip: chunk = sglang chunked_prefill_size
        # (GPU-derived, like the module), per-request max_pos, and the module's
        # _dsv4_module_is_valid_shape + required_kv_tokens (the shared KV-token fn
        # GLM5 uses via dsa_indexer_total_kv_tokens_supported). The 1M KV-pool cap
        # is the module's proxy -- DSV4 has no sglang-readable pool size offline.
        if bs <= 0 or isl <= 0 or prefix < 0:
            return False
        if bs * isl > chunk:
            return False
        if max_pos and prefix + isl > max_pos:
            return False
        return _dsv4_module_is_valid_shape("context", bs, isl) and (
            required_kv_tokens(bs, isl, prefix, is_prefill=True) <= 1_048_576
        )

    return _derive_context_shapes(
        _DSV4_MODULE_BATCH_SIZES,
        _DSV4_MODULE_SEQ_LENGTHS,
        _DSV4_MODULE_PAST_KV_LIST,
        _valid,
    )


def _dsv4_generation_derived_shapes(model_path):
    """Decode ``(kv_len, isl=1, bs)`` shapes from the csa/hca generation MODULE
    INPUT sweep (``_DSV4_MODULE_BATCH_SIZES`` x ``_DSV4_MODULE_SEQ_LENGTHS`` +
    generation validity), NOT read from the generation CSV. Decode sweeps
    batch x kv_cache_len with isl==1 (the perf ``step`` column is kv_len); CSA
    and HCA share one grid (validity does not depend on attn_kind). Reuses the
    shared _derive_context_shapes engine with seq_list=[1] and the kv-length
    sweep as the prefix dimension.
    """
    try:
        from collector.case_generator import (
            _DSV4_MODULE_BATCH_SIZES,
            _DSV4_MODULE_SEQ_LENGTHS,
            _dsv4_module_is_valid_shape,
        )
    except ModuleNotFoundError:
        from case_generator import (
            _DSV4_MODULE_BATCH_SIZES,
            _DSV4_MODULE_SEQ_LENGTHS,
            _dsv4_module_is_valid_shape,
        )
    cfg = _dsv4_model_config(model_path)
    max_pos = cfg.get("max_position_embeddings")

    def _valid(bs, isl, kv):
        if bs <= 0 or kv <= 0:
            return False
        if max_pos and kv >= max_pos:  # decode seq_len must be < max_position
            return False
        return _dsv4_module_is_valid_shape("generation", bs, kv)

    return _derive_context_shapes(_DSV4_MODULE_BATCH_SIZES, [1], _DSV4_MODULE_SEQ_LENGTHS, _valid)


def run_dsv4_sparse_kernel_worker(
    model_path: str,
    kernel: str,
    bs_only: int,
    *,
    perf_filename: str,
    device: str = "cuda:0",
):
    """Single worker for the whole DSV4 sparse-op family (invoked by collect.py,
    one case per model from get_func).

    Reads the kernel's owning module CSV(s) — paged_mqa_logits/topk/csa_attn ←
    CSA, hca_attn ← HCA — dedups to the unique ``(prefix, isl, bs)`` module shapes
    (the kernels are TP-independent), and benches each once.
    ``_bench_sparse_kernel_shape`` returns one or more ``(score_mode, latency)``
    rows per shape — one for the single-latency kernels, two for topk (flat +
    top_last) — each written as one row (tp_size=1)."""
    if kernel not in KERNEL_TO_OP_NAME:
        raise ValueError(f"unknown kernel={kernel}; expected one of {list(KERNEL_TO_OP_NAME)}")
    sc = _dsv4_sparse_config(model_path)
    output_dir = os.path.dirname(perf_filename) or os.getcwd()
    perf_path = _make_perf_filename(kernel, output_dir)

    # Both context and decode shapes are derived from the csa/hca module INPUT
    # sweeps (no module CSV read): context from the csa/hca context sweep,
    # decode (isl==1) from the generation sweep. TP-independent: one bench /
    # one row per unique (prefix, isl, bs); CSA/HCA share one grid.
    ctx_shapes = _dsv4_context_derived_shapes(model_path)
    dec_shapes = _dsv4_generation_derived_shapes(model_path)
    _seen = set(ctx_shapes)
    shapes = ctx_shapes + [sh for sh in dec_shapes if sh not in _seen]
    # this task owns one bs (collect.py distributes bs across GPU workers)
    shapes = [(prefix, isl, bs) for (prefix, isl, bs) in shapes if bs == bs_only]
    if not shapes:
        print(f"[dsv4-sparse {kernel} bs={bs_only}] no shapes; skipping.")
        return

    device_name = torch.cuda.get_device_name(device)
    print(f"[dsv4-sparse {kernel} bs={bs_only}] {len(shapes)} shapes -> {perf_path}")
    n_ok = 0
    for prefix, isl, bs in shapes:
        # Every module shape is benched (strict 1:1); tiny shapes run fine
        # (K/window clamp to full_s) and a shape that OOMs is skipped by
        # _guarded_bench rather than pre-filtered.
        results = _guarded_bench(
            lambda: _bench_sparse_kernel_shape(kernel, prefix, isl, bs, sc, device),
            f"bs={bs} isl={isl} past_kv={prefix}",
        )
        if results is None:
            continue
        n_ok += 1
        for score_mode, latency_ms in results:
            _write_row(
                perf_path,
                kernel=kernel,
                bs=bs,
                isl=isl,
                past_kv=prefix,
                tp_size=1,
                native_heads=sc.num_attention_heads,
                latency_ms=latency_ms,
                device_name=device_name,
                model_path=model_path,
                score_mode=score_mode,
            )
    print(f"  {kernel}: benched {n_ok}/{len(shapes)} unique shapes")


# ═══════════════════════════════════════════════════════════════════════
# topk_512 indexer DELTA calibration (kernels, fed by the shared worker above)
# ═══════════════════════════════════════════════════════════════════════
# The CSA indexer's topk kernel needs a CALIBRATION rather than a single latency:
# the CSA module CSV already contains the topK time measured on DEGENERATE scores
# (dummy weights -> near-constant logits -> the small O(n^2) tie-break path).
# ``_bench_topk_shape`` benches topk_transform under FLAT (worst-case degenerate)
# and TOP_LAST (representative: largest scores at the causal tail) distributions
# and returns the two as ``(score_mode, latency)`` rows; the shared worker writes
# them like any other sparse-kernel rows, and perf_database applies
# DELTA = flat.latency - top_last.latency to swap the degenerate cost for a
# representative one at query time (see _load_dsv4_topk_calib).
#
# topK is CSA-only (compress_ratio=4) and, like the other sparse kernels,
# TP-independent (per-token causal scan): one (flat, top_last) pair per shape.


def _dsv4_topk_kernel_supported() -> bool:
    """True when the SGLang build exposes the sm_100 topk_512 indexer kernel."""
    if os.environ.get("COLLECTOR_FORCE_DSV4_SPARSE") == "1":
        return True
    try:
        # find_spec raises (not returns None) when a PARENT package is absent.
        return importlib.util.find_spec("sglang.jit_kernel.dsv4.topk") is not None
    except ModuleNotFoundError:
        return False


def _make_topk_scores(mode: str, rows: int, seq: int, device: str, generator, topk_k: int) -> torch.Tensor:
    # score_stride must be a multiple of 4 (kernel TMA 16B alignment); allocate
    # padded width and let the kernel read [:, :seq].
    pad = ((seq + 3) // 4) * 4
    if mode == "flat":
        return torch.zeros(rows, pad, device=device)
    if mode == "top_last":
        s = -5.0 + 0.05 * torch.randn(rows, pad, device=device, generator=generator)
        s[:, seq - topk_k : seq] = 5.0 + torch.randn(rows, topk_k, device=device, generator=generator)
        return s.contiguous()
    raise ValueError(f"unknown topk score mode: {mode}")


def _bench_topk_512(rows: int, c4_len: int, mode: str, device: str, topk_k: int) -> float:
    """Time one topk_transform shape via the shared ``_bench_cuda_graph`` path
    (same bench helper as paged_mqa_logits/hca_attn).

    The topk_transform kernel runs inside sglang's decode CUDA graph in
    production, so we capture under CUDA graph too — but with
    ``allow_graph_fail=True`` (eager fallback) since this JIT kernel's capture is
    less robust. Host-side planning (``plan_topk_v2``) is done ONCE outside the
    timed region — only the kernel is captured/timed.
    """
    from sglang.jit_kernel.dsv4.topk import plan_topk_v2, topk_transform_512_v2

    generator = torch.Generator(device=device)
    generator.manual_seed(1234)
    seq_lens = torch.full((rows,), c4_len, dtype=torch.int32, device=device)
    meta = plan_topk_v2(seq_lens, 0)
    torch.cuda.synchronize(device)
    pages = (c4_len + _dsv4_kv_page_size() - 1) // _dsv4_kv_page_size()
    page_table = torch.arange(pages, dtype=torch.int32, device=device).unsqueeze(0).repeat(rows, 1)
    scores = _make_topk_scores(mode, rows, c4_len, device, generator, topk_k=topk_k)
    out = torch.empty((rows, topk_k), dtype=torch.int32, device=device)

    def kernel_func():
        topk_transform_512_v2(scores, seq_lens, page_table, out, _dsv4_kv_page_size(), meta)

    return _bench_cuda_graph(kernel_func, allow_graph_fail=True, device=device)["latency_ms"]


def get_dsv4_topk_calib_test_cases():
    """topk_512 DELTA calibration cases (gated on kernel availability)."""
    if not _dsv4_topk_kernel_supported():
        return []
    return _dsv4_sparse_kernel_cases("topk")


def _bench_topk_shape(prefix: int, isl: int, bs: int, sc, device: str) -> list:
    """topk DELTA calibration for one CSA ``(prefix, isl, bs)`` shape: bench
    topk_transform under FLAT (degenerate worst-case) and TOP_LAST
    (representative) score distributions and return them as two
    ``[("flat", latency), ("top_last", latency)]`` rows.

    Trivial when c4 <= K (no select stage, no degenerate tie-break) -> both 0
    (DELTA 0). ``rows`` = bs*isl is the per-token causal scan count."""
    ratio = sc.csa_compress_ratio  # CSA compress ratio (=4)
    topk_k = sc.index_topk  # V4-Pro=1024, V4-Flash=512
    rows = max(bs * isl, 1)
    c4_len = (prefix + isl) // ratio
    if c4_len <= topk_k:
        return [("flat", 0.0), ("top_last", 0.0)]
    return [
        (mode, round(_bench_topk_512(rows, c4_len, mode, device, topk_k=topk_k), 6)) for mode in ("flat", "top_last")
    ]
