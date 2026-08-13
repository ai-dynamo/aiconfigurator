# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generic ViT encoder op builder for multimodal VL models.

Provides :func:`build_encoder_ops`, a module-level function that constructs the
full list of encoder ops for any ViT-based vision encoder. Model classes call
this function in their ``__init__`` instead of duplicating op construction logic.

Op structure
------------
For a ViT with depth D and projector_dims with P (in, out) pairs::

  _vit_transformer_ops  →  10 ops, each with count=depth:
    encoder_add_norm_1    ElementWise
    encoder_qkv_gemm      GEMM
    encoder_attention     EncoderAttention   (non-causal, MHA, no KV cache)
    encoder_proj_gemm     GEMM  (low_precision_input=True)
    encoder_ar_1          CustomAllReduce
    encoder_add_norm_2    ElementWise
    encoder_ffn1_gemm     GEMM
    encoder_act           ElementWise
    encoder_ffn2_gemm     GEMM  (low_precision_input=True)
    encoder_ar_2          CustomAllReduce
    encoder_rope_apply    ElementWise  (only if partial_rotary_factor > 0;
                                        replaces the attention-internal RoPE term)

  _projector_ops  →  2*P ops + 1 AR  (or 0 ops if projector_dims is empty):
    encoder_projector_fc{i}_gemm  GEMM
    encoder_projector_fc{i}_act   ElementWise  (omitted for final layer)
    encoder_projector_ar          CustomAllReduce

  Encoder DP (enable_encoder_dp, default; vLLM mm_encoder_tp_mode="data" /
  SGLang --mm-enable-dp-encoder) builds all of the above with tp=1 — full
  replica per rank, images ceil-sharded across the tp_size ranks at query
  time in BaseBackend._run_encoder_phase — and appends for tp_size > 1:
    encoder_dp_all_gather         NCCL all_gather of post-merge embeddings

TP parallelism for projector layers
------------------------------------
The ViT transformer ends with a CustomAllReduce so every projector layer
receives a full (un-sharded) first-layer input.  For a two-layer projector
(the common case for PatchMerger-style architectures):

  - Layer 0: row-parallel   (M = out // tp, K = in      — shards the output)
  - Layer 1: column-parallel (M = out,       K = in // tp — input is sharded
                               from the previous layer, output is reduced by AR)

For P = 1 the single layer is row-parallel (M = out // tp, K = in) followed by
the AllReduce.  For P > 2 intermediate layers also receive sharded inputs; callers
are responsible for choosing a projector_dims layout that is TP-correct.
"""

from __future__ import annotations

import aiconfigurator_core.sdk.operations as ops
from aiconfigurator_core.sdk import common


def _vit_transformer_ops(enc_cfg: common.VisionEncoderConfig, tp_size: int) -> list:
    """Build the 10 ViT transformer block ops (each repeated enc_cfg.depth times),
    plus the optional encoder_rope_apply elementwise op.

    Raises ValueError if num_heads or intermediate_size is not divisible by tp_size.
    """
    depth = enc_cfg.depth
    h_vit = enc_cfg.hidden_size
    n_vit = enc_cfg.num_heads
    inter_vit = enc_cfg.intermediate_size
    head_size_vit = h_vit // n_vit

    if tp_size > 1:
        if n_vit % tp_size != 0:
            raise ValueError(f"ViT num_heads ({n_vit}) must be divisible by tp_size ({tp_size})")
        if inter_vit % tp_size != 0:
            raise ValueError(f"ViT intermediate_size ({inter_vit}) must be divisible by tp_size ({tp_size})")

    # ViT always runs in bfloat16 regardless of LLM quantization settings
    vit_gemm_mode = common.GEMMQuantMode.bfloat16
    vit_fmha_mode = common.FMHAQuantMode.bfloat16

    result = [
        ops.ElementWise("encoder_add_norm_1", depth, 2 * h_vit, 2 * h_vit, 0.8),
        ops.GEMM(
            "encoder_qkv_gemm",
            depth,
            3 * n_vit * head_size_vit // tp_size,
            h_vit,
            vit_gemm_mode,
        ),
        ops.EncoderAttention(
            "encoder_attention",
            depth,
            n_vit // tp_size,
            head_size_vit,
            fmha_quant_mode=vit_fmha_mode,
            partial_rotary_factor=0.0,
        ),
        ops.GEMM(
            "encoder_proj_gemm",
            depth,
            h_vit,
            n_vit * head_size_vit // tp_size,
            vit_gemm_mode,
            low_precision_input=True,
        ),
        ops.CustomAllReduce("encoder_ar_1", depth, h_vit, tp_size),
        ops.ElementWise("encoder_add_norm_2", depth, 2 * h_vit, 2 * h_vit, 0.8),
        ops.GEMM(
            "encoder_ffn1_gemm",
            depth,
            inter_vit // tp_size,
            h_vit,
            vit_gemm_mode,
        ),
        ops.ElementWise(
            "encoder_act",
            depth,
            inter_vit // tp_size,
            inter_vit // tp_size,
            0.8,
        ),
        ops.GEMM(
            "encoder_ffn2_gemm",
            depth,
            h_vit,
            inter_vit // tp_size,
            vit_gemm_mode,
            low_precision_input=True,
        ),
        ops.CustomAllReduce("encoder_ar_2", depth, h_vit, tp_size),
    ]

    if enc_cfg.partial_rotary_factor > 0:
        # the attention op's internal RoPE term is disabled in exchange (partial_rotary_factor=0.0).
        # partial_rotary_factor gates the op but does not shrink it: the eager kernel
        # duplicates the half-dim cos/sin table to full head_dim and rotates all of Q/K.
        rope_dim = 6 * (n_vit // tp_size) * head_size_vit
        result.append(ops.ElementWise("encoder_rope_apply", depth, rope_dim, rope_dim, 0.8))

    return result


def _projector_ops(enc_cfg: common.VisionEncoderConfig, tp_size: int) -> list:
    """Build the projector MLP ops from enc_cfg.projector_dims.

    TP layout per layer:
      - Non-final layers: row-parallel (M = out // tp, K = in; output sharded) + activation
      - Final layer: column-parallel if P > 1 (M = out, K = in // tp; input sharded)
                     row-parallel if P == 1 (M = out // tp, K = in; full input)
      - Always ends with a CustomAllReduce over the final output dimension.

    Returns [] if projector_dims is empty.
    """
    dims = enc_cfg.projector_dims
    if not dims:
        return []

    n_inst = enc_cfg.projector_n_instances
    vit_gemm_mode = common.GEMMQuantMode.bfloat16
    n_layers = len(dims)

    result = []
    for i, (in_d, out_d) in enumerate(dims):
        is_last = i == n_layers - 1
        # Final layer in a multi-layer projector takes sharded input from the previous
        # row-parallel layer (column-parallel style). Single-layer and non-final layers
        # always receive a full (non-sharded) input (row-parallel style).
        col_parallel = is_last and n_layers > 1
        if col_parallel:
            m, k = out_d, in_d // tp_size
        else:
            m, k = out_d // tp_size, in_d
        result.append(ops.GEMM(f"encoder_projector_fc{i}_gemm", n_inst, m, k, vit_gemm_mode))
        if not is_last:
            result.append(
                ops.ElementWise(
                    f"encoder_projector_fc{i}_act",
                    n_inst,
                    out_d // tp_size,
                    out_d // tp_size,
                    0.8,
                )
            )

    result.append(ops.CustomAllReduce("encoder_projector_ar", n_inst, dims[-1][1], tp_size))
    return result


def build_encoder_ops(enc_cfg: common.VisionEncoderConfig, tp_size: int, enable_encoder_dp: bool = True) -> list:
    """Build the complete list of encoder ops for a ViT-based vision encoder.

    Combines ViT transformer ops (10 ops x depth repetitions) with projector ops
    (2 x n_layers + 1 ops with AllReduce, or 0 if no projector configured).

    Args:
        enc_cfg: VisionEncoderConfig populated with ViT and projector parameters.
        tp_size: Worker tensor-parallel degree — the DP degree under encoder DP,
                 else the ViT weight-sharding degree (must evenly divide
                 num_heads and intermediate_size when tp_size > 1).
        enable_encoder_dp: Encoder data parallelism over the TP group (default
                 True) — see module docstring.

    Returns:
        Flat list of operation objects ready to assign to model.encoder_ops.
    """
    if not enable_encoder_dp:
        return _vit_transformer_ops(enc_cfg, tp_size) + _projector_ops(enc_cfg, tp_size)

    # DP: full-replica ops (tp=1); the per-layer AllReduces degenerate to no-ops.
    result = _vit_transformer_ops(enc_cfg, 1) + _projector_ops(enc_cfg, 1)
    if tp_size > 1:
        result.append(
            ops.NCCL(
                "encoder_dp_all_gather",
                1,
                "all_gather",
                num_elements_per_token=enc_cfg.out_hidden_size * enc_cfg.projector_n_instances * tp_size,
                num_gpus=tp_size,
                comm_quant_mode=common.CommQuantMode.half,
            )
        )
    return result


def _gemma4_vision_transformer_ops(enc_cfg: common.Gemma4VisionEncoderConfig, tp_size: int) -> list:
    """Build the Gemma 4 vision tower without Qwen3-VL merger assumptions.

    The graph follows ``Gemma4VisionModel``: learned patch + 2-D position
    embeddings, 27 non-causal transformer blocks with Q/K/V normalization,
    full x/y RoPE and a gated MLP, followed by position-aware average pooling,
    standardization, RMS normalization, and one language-space projection.
    """
    depth = enc_cfg.depth
    h_vit = enc_cfg.hidden_size
    n_vit = enc_cfg.num_heads
    n_kv = enc_cfg.num_key_value_heads
    head_dim = enc_cfg.head_dim
    inter_vit = enc_cfg.intermediate_size
    pool = enc_cfg.pooling_kernel_size

    if min(depth, h_vit, n_vit, n_kv, head_dim, inter_vit, enc_cfg.patch_size, pool) <= 0:
        raise ValueError("Gemma 4 vision encoder dimensions must all be positive")
    if n_vit != n_kv:
        raise ValueError("Gemma 4 vision encoder requires full MHA (num_heads == num_key_value_heads)")
    if n_vit * head_dim != h_vit:
        raise ValueError("Gemma 4 vision num_heads * head_dim must equal hidden_size")
    for field, value in (
        ("num_heads", n_vit),
        ("num_key_value_heads", n_kv),
        ("intermediate_size", inter_vit),
    ):
        if value % tp_size != 0:
            raise ValueError(f"Gemma 4 vision {field} ({value}) must be divisible by tp_size ({tp_size})")

    vit_gemm_mode = common.GEMMQuantMode.bfloat16
    vit_fmha_mode = common.FMHAQuantMode.bfloat16
    qkv_width = (n_vit + 2 * n_kv) * head_dim // tp_size
    attn_width = n_vit * head_dim // tp_size
    inter_per_tp = inter_vit // tp_size

    result = [
        # Pixel patches are already flattened by the processor to 3 * patch².
        ops.GEMM(
            "encoder_patch_embed_gemm",
            1,
            h_vit,
            3 * enc_cfg.patch_size**2,
            vit_gemm_mode,
        ),
        # Two independent learned tables (x and y).  A scale factor of two
        # accounts for both lookups in latency and resident weights.
        ops.Embedding(
            "encoder_position_embedding",
            2,
            enc_cfg.position_embedding_size,
            h_vit,
            0.3,
        ),
        ops.ElementWise("encoder_patch_embed_add", 1, 3 * h_vit, h_vit, 0.8),
        ops.ElementWise("encoder_input_norm", depth, h_vit, h_vit, 0.8),
        ops.GEMM("encoder_qkv_gemm", depth, qkv_width, h_vit, vit_gemm_mode),
        # Per-head Q/K/V RMSNorm plus full two-dimensional RoPE on Q and K.
        # RoPE is explicit here rather than the Qwen-specific partial-RoPE
        # path inside EncoderAttention.
        ops.ElementWise("encoder_qkv_norm_rope_2d", depth, 3 * attn_width, 3 * attn_width, 0.8),
        ops.EncoderAttention(
            "encoder_attention",
            depth,
            n_vit // tp_size,
            head_dim,
            fmha_quant_mode=vit_fmha_mode,
            partial_rotary_factor=0.0,
        ),
        ops.GEMM(
            "encoder_proj_gemm",
            depth,
            h_vit,
            attn_width,
            vit_gemm_mode,
            low_precision_input=True,
        ),
        ops.CustomAllReduce("encoder_ar_1", depth, h_vit, tp_size),
        ops.ElementWise("encoder_post_attn_norm_residual", depth, 3 * h_vit, h_vit, 0.8),
        ops.ElementWise("encoder_pre_ffn_norm", depth, h_vit, h_vit, 0.8),
        # Gemma 4's vision MLP is gated: separate gate/up projections are
        # represented as one fused GEMM with a 2*intermediate output.
        ops.GEMM("encoder_ffn_gate_up_gemm", depth, 2 * inter_per_tp, h_vit, vit_gemm_mode),
        ops.ElementWise("encoder_ffn_act_mul", depth, 2 * inter_per_tp, inter_per_tp, 0.8),
        ops.GEMM(
            "encoder_ffn_down_gemm",
            depth,
            h_vit,
            inter_per_tp,
            vit_gemm_mode,
            low_precision_input=True,
        ),
        ops.CustomAllReduce("encoder_ar_2", depth, h_vit, tp_size),
        ops.ElementWise("encoder_post_ffn_norm_residual", depth, 3 * h_vit, h_vit, 0.8),
        # Query x is the pre-pooling patch count. scale_num_tokens converts it
        # to pooled tokens while dim_in accounts for all pool² source patches.
        ops.ElementWise(
            "encoder_gemma4_pool_avg",
            1,
            pool**2 * h_vit,
            h_vit,
            0.8,
            scale_num_tokens=pool**2,
        ),
        # sqrt(hidden) scaling and optional checkpoint standardization execute
        # on the pooled soft-token stream.
        ops.ElementWise(
            "encoder_gemma4_pool_postprocess",
            1,
            (3 if enc_cfg.standardize else 1) * h_vit,
            h_vit,
            0.8,
        ),
        ops.ElementWise("encoder_projector_pre_norm", 1, h_vit, h_vit, 0.8),
        # Gemma4MultimodalEmbedder uses a ReplicatedLinear in vLLM and is
        # absent from the Hugging Face vision TP plan.  Keep the complete
        # 1152 -> language-hidden projection on every rank; there is no Qwen
        # PatchMerger-style projector sharding or projector AllReduce.
        ops.GEMM(
            "encoder_projector_fc0_gemm",
            1,
            enc_cfg.out_hidden_size,
            h_vit,
            vit_gemm_mode,
        ),
    ]
    return result


def build_gemma4_vision_encoder_ops(
    enc_cfg: common.Gemma4VisionEncoderConfig,
    tp_size: int,
    enable_encoder_dp: bool = True,
) -> list:
    """Build Gemma 4 vision ops under encoder-DP or legacy encoder-TP.

    Encoder-DP keeps one complete vision tower per TP rank, shards whole images
    across those replicas, then all-gathers the projected soft-token embeddings.
    Encoder-TP shards attention and gated-MLP weights with per-block
    all-reduces.  Gemma's language projection remains replicated, matching the
    engine contract.
    """
    if not isinstance(enc_cfg, common.Gemma4VisionEncoderConfig):
        raise TypeError(
            f"build_gemma4_vision_encoder_ops requires Gemma4VisionEncoderConfig, got {type(enc_cfg).__name__}"
        )

    if not enable_encoder_dp:
        return _gemma4_vision_transformer_ops(enc_cfg, tp_size)

    result = _gemma4_vision_transformer_ops(enc_cfg, 1)
    if tp_size > 1:
        result.append(
            ops.NCCL(
                "encoder_dp_all_gather",
                1,
                "all_gather",
                num_elements_per_token=enc_cfg.out_hidden_size * tp_size,
                num_gpus=tp_size,
                comm_quant_mode=common.CommQuantMode.half,
            )
        )
    return result
