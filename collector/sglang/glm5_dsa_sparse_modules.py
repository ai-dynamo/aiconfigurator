# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GLM-5 DSA sparse-attention kernel-level collector for SGLang.

GLM-5 analogue of ``deepseekv4_sparse_modules.py``. Collects the three GLM-5
NSA/DSA sparse sub-kernels at the kernel level, SAME-SOURCE: every sub-kernel
derives its ``(prefix, isl, bs)`` shapes STRICTLY 1:1 from the GLM-5 DSA
attention-module CSV (``dsa_context_module`` / ``dsa_generation_module``) — the
same mechanism DSV4 uses against its csa/hca module CSVs. Shapes only; the
kernel is benched standalone with synthetic fp8/bf16 inputs (no real weights).

Sub-kernels (GLM-5 DSA prefill path, in order):
    1. ``deep_gemm.fp8_mqa_logits``  (mqa)      indexer scoring, NON-paged ragged
                                                kv over the FULL context. 32 idx
                                                heads x head_dim 128.
    2. ``fast_topk_v2`` (sgl_kernel) (topk)     top-2048 over the mqa logits
                                                (FULL length, NOT /4).
    3. ``flash_mla_sparse_fwd``      (dsa_attn) sparse FMLA over topk-selected
                                                positions; d_qk=576 (kv_lora 512
                                                + rope 64), d_v=512.

GLM-5 is uniform DSA (no compress_ratios): one attention kind, full-context.
CSV schema matches the aic module CSVs (``isl``=M, ``step``=past_kv).
"""

from __future__ import annotations

import os

import torch

# Generic (kernel-agnostic) infra reused from the DSV4 sparse collector.
from collector.sglang.deepseekv4_sparse_modules import (
    _bench_cuda_graph,
    _derive_context_shapes,
    _guarded_bench,
)
from collector.sglang.deepseekv4_sparse_modules import (
    _dsv4_cfg_int as _cfg_int,
)
from collector.sglang.deepseekv4_sparse_modules import (
    _dsv4_model_config as _model_config,
)

try:
    from collector.sglang.helper import log_perf
except ModuleNotFoundError:
    from helper import log_perf

__all__ = [
    "get_glm5_dsa_attn_test_cases",
    "get_glm5_mqa_test_cases",
    "get_glm5_topk_test_cases",
    "run_glm5_dsa_sparse_kernel_worker",
]

GLM5_DEFAULT_MODEL = "nvidia/GLM-5-NVFP4"
GLM5_ARCHITECTURE = "GlmMoeDsaForCausalLM"


def _selected_glm5_models():
    """GLM model to collect sparse kernels for. The kernels (mqa/topk/dsa_attn)
    are bf16 and identical across GLM-5 variants, differing only in prefix range.
    When both GLM-5-NVFP4 and GLM-5.2-NVFP4 are configured, collect only
    GLM-5.2-NVFP4 (longest context — its range + the max_position ceiling in the
    derived shapes covers GLM-5); otherwise the default."""
    try:
        try:
            from collector.sglang.collect_mla_module import get_mla_module_model_specs
        except ModuleNotFoundError:
            from collect_mla_module import get_mla_module_model_specs
        paths = {s.model_path for s in get_mla_module_model_specs(attention_type="dsa")}
        if {"nvidia/GLM-5-NVFP4", "nvidia/GLM-5.2-NVFP4"} <= paths:
            return ["nvidia/GLM-5.2-NVFP4"]
    except Exception:
        pass
    return [GLM5_DEFAULT_MODEL]


def _glm5_sparse_config(model_path: str):
    from types import SimpleNamespace

    cfg = _model_config(model_path)
    kv_lora = _cfg_int(cfg, "kv_lora_rank")  # 512
    rope = _cfg_int(cfg, "qk_rope_head_dim")  # 64
    return SimpleNamespace(
        num_attention_heads=_cfg_int(cfg, "num_attention_heads"),  # 64
        index_n_heads=_cfg_int(cfg, "index_n_heads"),  # 32
        index_head_dim=_cfg_int(cfg, "index_head_dim"),  # 128
        index_topk=_cfg_int(cfg, "index_topk"),  # 2048
        kv_lora_rank=kv_lora,  # FMLA d_v = 512
        d_rope=rope,  # 64
        d_qk=kv_lora + rope,  # FMLA d_qk = 576
        compress_ratio=1,  # uniform DSA, full context
    )


KERNEL_TO_OP_NAME = {
    "mqa": "glm5_mqa_logits_module",
    "topk": "glm5_topk_module",
    "dsa_attn": "glm5_dsa_attn_module",
}
KERNEL_TO_KERNEL_SOURCE = {
    "mqa": "deep_gemm.fp8_mqa_logits",
    "topk": "fast_topk_v2",
    "dsa_attn": "flash_mla_sparse_fwd",
}


def _make_perf_filename(kernel: str, output_path: str, op_name_map: dict | None = None) -> str:
    if op_name_map is None:
        op_name_map = KERNEL_TO_OP_NAME
    if os.path.isdir(output_path) or not output_path.endswith(".txt"):
        return os.path.join(output_path, f"{op_name_map[kernel]}_perf.txt")
    return output_path


def _write_row(
    perf_filename,
    *,
    kernel,
    bs,
    isl,
    past_kv,
    tp_size,
    native_heads,
    latency_ms,
    device_name,
    model_path,
    score_mode=None,
    kernel_source=None,
    architecture: str = GLM5_ARCHITECTURE,
    op_name_map: dict | None = None,
):
    # ``architecture`` / ``op_name_map`` default to GLM-5 so existing GLM-5
    # callers are unchanged; DeepSeek-V3.2 reuses this with its own values
    # (DeepseekV32ForCausalLM + dsv32_* names) -- see dsv32_dsa_sparse_modules.
    if op_name_map is None:
        op_name_map = KERNEL_TO_OP_NAME
    os.makedirs(os.path.dirname(os.path.abspath(perf_filename)) or ".", exist_ok=True)
    mla_dtype = "bfloat16" if kernel == "dsa_attn" else "fp8_e4m3"
    item = {
        "model": model_path,
        "architecture": architecture,
        "mla_dtype": mla_dtype,
        "kv_cache_dtype": "fp8_e4m3",
        "gemm_type": "fp8_block",
        "num_heads": native_heads,
        "batch_size": bs,
        "isl": isl,
        "tp_size": tp_size,
        "step": past_kv,
        "compress_ratio": 1,
        "latency": f"{latency_ms:.6f}",
    }
    if score_mode is not None:
        item["score_mode"] = score_mode
    log_perf(
        item_list=[item],
        framework="SGLang",
        version="kernel-level",
        device_name=device_name,
        op_name=op_name_map[kernel],
        kernel_source=kernel_source or KERNEL_TO_KERNEL_SOURCE[kernel],
        perf_filename=perf_filename,
    )


# ═══════════════════════════════════════════════════════════════════════
# Kernel benches (standalone, synthetic inputs)
# ═══════════════════════════════════════════════════════════════════════
def _bench_glm5_mqa(M, past_kv, isl, *, index_n_heads, index_head_dim, device):  # noqa: N803
    """deep_gemm.fp8_mqa_logits — ragged batch of bs = M // isl requests.
    M = bs*isl query tokens over a CONCATENATED per-request KV cache (bs
    segments of past_kv + isl); ks/ke are absolute [start, end) into that
    cache, matching sglang's dsa_indexer (per-token k_start / k_end). Each
    request r local pos p scans causal [r*seg, r*seg + past_kv + p + 1)."""
    from deep_gemm import fp8_mqa_logits

    bs = max(1, M // isl)
    seg = past_kv + isl  # per-request KV length
    full_s = max(1, bs * seg)  # CONCATENATED kv: bs segments of (past_kv + isl)
    q = torch.randn(M, index_n_heads, index_head_dim, dtype=torch.bfloat16, device=device).to(torch.float8_e4m3fn)
    k_fp8 = torch.randn(full_s, index_head_dim, dtype=torch.bfloat16, device=device).to(torch.float8_e4m3fn)
    k_scale = torch.ones(full_s, dtype=torch.float32, device=device)
    weights = torch.randn(M, index_n_heads, dtype=torch.float32, device=device)
    # Absolute [ks, ke) into the concatenated kv (sglang dsa_indexer k_start/k_end):
    # token (request r, local pos p) attends its own segment
    # [r*seg, r*seg + past_kv + p + 1).
    seg_start = torch.repeat_interleave(torch.arange(bs, dtype=torch.int32, device=device) * seg, isl)
    causal = torch.arange(1, isl + 1, dtype=torch.int32, device=device).repeat(bs)
    ks = seg_start
    ke = (seg_start + past_kv + causal).clamp(max=full_s)

    def kernel_fn():
        return fp8_mqa_logits(q, (k_fp8, k_scale), weights, ks, ke, clean_logits=False)

    return _bench_cuda_graph(kernel_fn, allow_graph_fail=True, device=device)


def _glm5_fuse_topk_enabled() -> bool:
    """Whether SGLang runs the FUSED topk+index-transform (env default = True)."""
    try:
        from sglang.srt import environ as _environ

        return bool(_environ.envs.SGLANG_NSA_FUSE_TOPK.get())
    except Exception:
        return True  # sglang environ.py: SGLANG_NSA_FUSE_TOPK = EnvBool(True)


def _make_glm5_topk_scores(mode, rows, seq, device, generator, topk_k):
    """DSV4-style score distributions for the topk DELTA calibration.

    * ``flat``     — all zeros: degenerate worst case (every element ties, so
                     the kernel does maximal tie-break work).
    * ``top_last`` — background ~-5 with the last ``topk_k`` positions ~+5:
                     representative (clear winners, the common real case).

    Width padded to a multiple of 4 (kernel TMA 16B alignment); kernel reads
    ``[:, :seq]``.
    """
    pad = ((seq + 3) // 4) * 4
    if mode == "flat":
        return torch.zeros(rows, pad, dtype=torch.float32, device=device)
    if mode == "top_last":
        s = -5.0 + 0.05 * torch.randn(rows, pad, dtype=torch.float32, device=device, generator=generator)
        k = min(topk_k, seq)
        s[:, seq - k : seq] = 5.0 + torch.randn(rows, k, dtype=torch.float32, device=device, generator=generator)
        return s.contiguous()
    raise ValueError(f"unknown topk score mode: {mode}")


def _bench_glm5_topk(M, past_kv, isl, bs, *, topk, device):  # noqa: N803
    """GLM-5 indexer top-k — the kernel SGLang's NSA backend ACTUALLY runs,
    benched as a FLAT/TOP_LAST DELTA calibration (DSV4-style).

    Kernel selection (``SGLANG_NSA_FUSE_TOPK`` env default = True → fused):
      * prefill / context (isl > 1, ragged): ``fast_topk_transform_ragged_fused``
      * decode  / generation (isl == 1, paged): ``fast_topk_transform_fused``
      * fuse-topk disabled: plain ``fast_topk_v2``.
    All metadata (``cu_seqlens`` / ``topk_indices_offset`` / page table /
    lengths) is built from ``(bs, isl, past_kv)`` EXACTLY as the NSA backend
    does — no model weights — so the standalone call is the real kernel with
    real arg shapes.

    topk timing is DATA-DEPENDENT (measured 3-22% spread by score distribution),
    so instead of guessing the real logit distribution we bench two anchors —
    ``flat`` (degenerate worst-case) and ``top_last`` (representative) — and
    return both as ``[("flat", lat), ("top_last", lat)]``; the SDK applies the
    DELTA as the data-dependent correction.  Trivial when the full per-request
    context ``<= topk`` (select-all, no data-dependent cost) → both ``0``.

    Returns ``(results, kernel_source)``.
    """
    from sgl_kernel import (
        fast_topk_transform_fused,
        fast_topk_transform_ragged_fused,
        fast_topk_v2,
    )

    try:
        from sglang.srt.layers.attention.nsa_backend import compute_cu_seqlens
    except Exception:
        from sglang.srt.layers.attention.dsa.dsa_backend_mtp_precompute import compute_cu_seqlens

    fused = _glm5_fuse_topk_enabled()
    if not fused:
        kernel_src = "fast_topk_v2"
    elif isl == 1:
        kernel_src = "fast_topk_transform_fused"
    else:
        kernel_src = "fast_topk_transform_ragged_fused"

    # GLM-5 is uniform DSA (ratio=1): per-request full context = past_kv + isl.
    seq = max(1, past_kv + isl)
    if seq <= topk:
        # nothing to select -> no data-dependent cost; DELTA is 0.
        return [("flat", 0.0), ("top_last", 0.0)], kernel_src

    lengths = torch.full((M,), seq, dtype=torch.int32, device=device)
    ks = torch.zeros(M, dtype=torch.int32, device=device)
    generator = torch.Generator(device=device)
    generator.manual_seed(1234)

    if not fused:

        def make_fn(score):
            return lambda: fast_topk_v2(score, lengths, topk, row_starts=ks)
    elif isl == 1:
        cols = max(topk, 1)
        pt = torch.arange(cols, dtype=torch.int32, device=device).clamp(max=seq - 1)
        page_table_size_1 = pt.view(1, cols).repeat(M, 1).contiguous()
        cu_seqlens_q = compute_cu_seqlens(torch.ones(M, dtype=torch.int32, device=device))

        def make_fn(score):
            return lambda: fast_topk_transform_fused(
                score=score,
                lengths=lengths,
                page_table_size_1=page_table_size_1,
                cu_seqlens_q=cu_seqlens_q,
                topk=topk,
                row_starts=ks,
            )
    else:
        seqlens_q = torch.full((bs,), isl, dtype=torch.int32, device=device)
        cu_seqlens_q_topk = compute_cu_seqlens(seqlens_q)
        cu_topk_indices_offset = torch.repeat_interleave(cu_seqlens_q_topk[:-1], seqlens_q)

        def make_fn(score):
            return lambda: fast_topk_transform_ragged_fused(
                score=score,
                lengths=lengths,
                topk_indices_offset=cu_topk_indices_offset,
                topk=topk,
                row_starts=ks,
            )

    results = []
    for mode in ("flat", "top_last"):
        score = _make_glm5_topk_scores(mode, M, seq, device, generator, topk)
        r = _bench_cuda_graph(make_fn(score), allow_graph_fail=True, device=device)
        results.append((mode, round(r["latency_ms"], 6)))
    return results, kernel_src


def _bench_glm5_dsa_attn(M, past_kv, isl, *, native_heads, d_qk, d_v, topk, device):  # noqa: N803
    """flash_mla_sparse_fwd — sparse FMLA, ragged batch of bs = M // isl reqs.
    q (M=bs*isl, heads->pad128, d_qk), kv = CONCATENATED bs segments of
    (past_kv + isl); indices (M, 1, K) are absolute into each token's own
    segment. K = min(topk, per-request context)."""
    from sgl_kernel.flash_mla import flash_mla_sparse_fwd

    bs = max(1, M // isl)
    seg = past_kv + isl
    full_s = max(1, bs * seg)  # concatenated bs segments of (past_kv + isl)
    k = min(topk, seg)  # topk capped by PER-REQUEST context
    pad_heads = 128 if (native_heads % 128) else native_heads
    q = torch.randn(M, pad_heads, d_qk, dtype=torch.bfloat16, device=device)
    kv = torch.randn(full_s, 1, d_qk, dtype=torch.bfloat16, device=device)
    # each token selects k positions inside its own segment [r*seg, r*seg+seg)
    seg_start = torch.repeat_interleave(torch.arange(bs, dtype=torch.int32, device=device) * seg, isl)
    indices = seg_start.view(M, 1, 1) + torch.arange(k, dtype=torch.int32, device=device).view(1, 1, k)
    if k % 64:
        pad = 64 - k % 64
        indices = torch.cat(
            [indices, torch.full((M, 1, pad), -1, dtype=torch.int32, device=device)], dim=-1
        ).contiguous()
    sm_scale = 1.0 / (d_qk**0.5)

    def kernel_fn():
        return flash_mla_sparse_fwd(q=q, kv=kv, indices=indices, sm_scale=sm_scale, d_v=d_v)

    return _bench_cuda_graph(kernel_fn, allow_graph_fail=True, device=device)


def _bench_glm5_sparse_kernel_shape(kernel, prefix, isl, bs, sc, device):
    M = max(bs * isl, 1)  # noqa: N806
    if kernel == "topk":
        # flat/top_last DELTA calibration -> two rows; reports the fused/unfused
        # kernel it actually ran.
        results, kernel_source = _bench_glm5_topk(M, prefix, isl, bs, topk=sc.index_topk, device=device)
        return kernel_source, results
    if kernel == "mqa":
        r = _bench_glm5_mqa(
            M, prefix, isl, index_n_heads=sc.index_n_heads, index_head_dim=sc.index_head_dim, device=device
        )
    elif kernel == "dsa_attn":
        r = _bench_glm5_dsa_attn(
            M,
            prefix,
            isl,
            native_heads=sc.num_attention_heads,
            d_qk=sc.d_qk,
            d_v=sc.kv_lora_rank,
            topk=sc.index_topk,
            device=device,
        )
    else:
        raise ValueError(f"unknown glm5 kernel={kernel}")
    return KERNEL_TO_KERNEL_SOURCE[kernel], [(None, r["latency_ms"])]


def _dsa_context_derived_shapes(model_path):
    """Context (prefix, isl, bs) shapes derived DIRECTLY from the DSA context
    INPUT sweep (batch x seq x prefix + the same validity filter dsa_context
    uses), NOT read back from dsa_context_module_perf.txt.

    Why not 1:1 read the module CSV: the CP model looks up mqa_full / topk_last
    at the FULL chunk isl, but dsa_context drops large isl (bs*seq beyond the
    FlashMLA sched-meta smem cap). fp8_mqa_logits / fast_topk have no such cap,
    so reusing only the rows dsa_context survived makes the cheap kernels
    inherit a drop they can avoid. Deriving from the input grid collects every
    valid (isl, prefix) the indexer would actually see.

    isl==1 (single-token decode) is rejected by the context validity filter, so
    this returns prefill/context shapes only; decode shapes still come from the
    generation CSV.
    """
    try:
        from collector.case_generator import get_mla_module_sweep_spec
        from collector.sglang.collect_mla_module import (
            _dsa_ceiling_max_positions,
            _dsa_context_prefix_shape_is_valid,
            _filter_cases_from_env,
            _model_max_position_embeddings,
            dsa_indexer_total_kv_tokens_supported,
        )
    except ModuleNotFoundError:
        from case_generator import get_mla_module_sweep_spec
        from collect_mla_module import (
            _dsa_ceiling_max_positions,
            _dsa_context_prefix_shape_is_valid,
            _filter_cases_from_env,
            _model_max_position_embeddings,
            dsa_indexer_total_kv_tokens_supported,
        )
    sweep = get_mla_module_sweep_spec("sglang")
    max_pos = _model_max_position_embeddings(model_path)

    def _valid(bs, isl, prefix):
        # Reuse the dsa_context MODULE's skip verbatim: max_token
        # (context_max_tokens) + large-seq cap + per-request max_pos/indexer-shape
        # + KV-pool total-token limit (dsa_indexer_total_kv_tokens_supported).
        # Only the FlashMLA smem cap is omitted -- the cheap fp8_mqa_logits /
        # fast_topk / sparse_fwd kernels don't hit it.
        if bs * isl > sweep.context_max_tokens:
            return False
        if isl >= sweep.context_large_sequence_min and bs > sweep.context_large_sequence_max_batch_size:
            return False
        return _dsa_context_prefix_shape_is_valid(
            bs, isl, prefix, max_position_embeddings=max_pos
        ) and dsa_indexer_total_kv_tokens_supported(bs, isl, prefix, is_prefill=True)

    # AIC_DSA_CONTEXT_* env pin: _filter_cases_from_env wants (bs, seq, ip, prefix).
    def _env(cases):
        tagged = [(bs, isl, True, prefix) for (bs, isl, prefix) in cases]
        kept = _filter_cases_from_env(tagged, is_prefill=True, attn_type="dsa")
        return [(bs, isl, prefix) for (bs, isl, _ip, prefix) in kept]

    shapes = _derive_context_shapes(
        sweep.context_batch_sizes,
        sweep.context_sequence_lengths,
        sweep.context_prefix_lengths,
        _valid,
        env_filter=_env,
    )
    # Ceiling: put a real (prefix, isl) point at prefix + isl == each covered
    # max_position (own + same-precision siblings via _dsa_ceiling_max_positions;
    # e.g. when the dedup representative is GLM-5.2-NVFP4, also cover
    # GLM-5-NVFP4's 202752). Mirrors the DSA context module's ceiling so CP
    # near-max queries interpolate within data instead of extrapolating across
    # the coarse prefix grid.
    have = set(shapes)
    for cover in _dsa_ceiling_max_positions(model_path):
        for isl in sweep.context_sequence_lengths:
            ceil_prefix = cover - isl
            for bs in sweep.context_batch_sizes:
                key = (ceil_prefix, isl, bs)
                if ceil_prefix > 0 and key not in have and _valid(bs, isl, ceil_prefix):
                    shapes.append(key)
                    have.add(key)
    return shapes


def _dsa_generation_derived_shapes(model_path):
    """Decode ``(kv_len, isl=1, bs)`` shapes derived from the dsa_generation
    INPUT sweep (generation_batch_sizes x generation_sequence_lengths +
    generation validity), NOT read back from dsa_generation_module_perf.txt.
    Decode sweeps batch x kv_cache_len with isl==1 (perf ``step`` = kv_len);
    reuses _derive_context_shapes with seq_list=[1] and the kv-length sweep as
    the prefix dimension.
    """
    try:
        from collector.case_generator import get_mla_module_sweep_spec
        from collector.sglang.collect_mla_module import _dsa_ceiling_max_positions, _model_max_position_embeddings
    except ModuleNotFoundError:
        from case_generator import get_mla_module_sweep_spec
        from collect_mla_module import _dsa_ceiling_max_positions, _model_max_position_embeddings
    sweep = get_mla_module_sweep_spec("sglang")
    max_pos = _model_max_position_embeddings(model_path)

    def _valid(bs, isl, kv):
        if bs <= 0 or kv <= 0:
            return False
        if bs * kv > sweep.generation_max_tokens:
            return False
        if kv >= sweep.generation_large_sequence_min and bs > sweep.generation_large_sequence_max_batch_size:
            return False
        return not (max_pos and kv >= max_pos)  # decode kv length must be < max_position

    shapes = _derive_context_shapes(sweep.generation_batch_sizes, [1], sweep.generation_sequence_lengths, _valid)
    # Ceiling: extend the decode kv-length grid to each covered max_position-1
    # (own + same-precision siblings) at bs=1, so GLM-5.2 (and GLM-5 via dedup)
    # decode near max doesn't extrapolate. High-context decode is single-stream,
    # so add at bs=1 and bypass the bs*kv token budget for these ceiling points
    # (kept kv < max_position). Mirrors the DSA generation module's step ceiling.
    have = set(shapes)
    for cover in _dsa_ceiling_max_positions(model_path):
        kv = cover - 1
        key = (kv, 1, 1)
        if kv > 0 and (max_pos is None or kv < max_pos) and key not in have:
            shapes.append(key)
            have.add(key)
    return shapes


def run_glm5_dsa_sparse_kernel_worker(
    model_path,
    kernel,
    bs_only,
    *,
    perf_filename,
    device="cuda:0",
    architecture: str = GLM5_ARCHITECTURE,
    op_name_map: dict | None = None,
    label: str = "glm5",
):
    # ``architecture`` / ``op_name_map`` / ``label`` default to GLM-5; DeepSeek-V3.2
    # reuses this same worker with its own values (kernels/shapes come from the
    # model config, so only the output tag + filenames + log label differ).
    if op_name_map is None:
        op_name_map = KERNEL_TO_OP_NAME
    if kernel not in op_name_map:
        raise ValueError(f"unknown kernel={kernel}; expected one of {list(op_name_map)}")
    sc = _glm5_sparse_config(model_path)
    output_dir = os.path.dirname(perf_filename) or os.getcwd()
    perf_path = _make_perf_filename(kernel, output_dir, op_name_map)
    # Both context and decode shapes are derived from the DSA module INPUT
    # sweeps (no perf.txt read): context from dsa_context, decode (isl==1)
    # from dsa_generation. The sparse kernels thus cover every shape the
    # modules intend, not just the rows the full module survived at runtime.
    ctx_shapes = _dsa_context_derived_shapes(model_path)
    dec_shapes = _dsa_generation_derived_shapes(model_path)
    _seen = set(ctx_shapes)
    shapes = ctx_shapes + [sh for sh in dec_shapes if sh not in _seen]
    # this task owns one bs (collect.py distributes bs across GPU workers)
    shapes = [(prefix, isl, bs) for (prefix, isl, bs) in shapes if bs == bs_only]
    if not shapes:
        print(f"[{label}-sparse {kernel} bs={bs_only}] no shapes; skipping.")
        return
    device_name = torch.cuda.get_device_name(device)
    print(f"[{label}-sparse {kernel} bs={bs_only}] {len(shapes)} shapes -> {perf_path}")
    n_ok = 0
    for prefix, isl, bs in shapes:
        out = _guarded_bench(
            lambda: _bench_glm5_sparse_kernel_shape(kernel, prefix, isl, bs, sc, device),
            f"bs={bs} isl={isl} past_kv={prefix}",
        )
        if out is None:
            continue
        kernel_source, results = out
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
                kernel_source=kernel_source,
                architecture=architecture,
                op_name_map=op_name_map,
            )
    print(f"  {kernel}: benched {n_ok}/{len(shapes)} unique shapes")


def _glm5_sparse_kernel_cases(kernel):
    # One task per (model, bs) so collect.py spreads bs across the GPU workers
    # (no single-worker cuda-graph private-pool buildup -> no 1-worker-sweep
    # deadlock). All sparse kernels run a single fixed head config: the FMLA
    # pads to its required head count OUTSIDE the kernel (model TP zero-pad),
    # so the kernel is TP-independent -> one config per bs, no tp sweep. Each
    # task sweeps (isl, prefix) for its bs.
    cases = []
    for m in _selected_glm5_models():
        ctx = _dsa_context_derived_shapes(m)
        dec = _dsa_generation_derived_shapes(m)
        bss = sorted({b for (_p, _i, b) in ctx} | {b for (_p, _i, b) in dec})
        cases.extend([m, kernel, b] for b in bss)
    return cases


def get_glm5_mqa_test_cases():
    return _glm5_sparse_kernel_cases("mqa")


def get_glm5_topk_test_cases():
    return _glm5_sparse_kernel_cases("topk")


def get_glm5_dsa_attn_test_cases():
    return _glm5_sparse_kernel_cases("dsa_attn")
