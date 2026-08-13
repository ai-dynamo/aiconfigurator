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


def _llama4_projector_ops(enc_cfg: common.VisionEncoderConfig, tp_size: int) -> list:
    """Build Llama 4's pixel-shuffle adaptor and multimodal connector.

    The checkpoint contains two distinct modules and therefore two distinct
    communication boundaries:

    * ``vision_adapter.mlp`` is column-parallel then row-parallel.  Its second
      GEMM reduces the 4096-wide output across TP ranks.
    * ``multi_modal_projector.linear_1`` is column-parallel with gathered
      output, projecting the 4096-wide vision embedding to the 5120-wide text
      embedding consumed by the hybrid-MoE backbone.
    """
    dims = enc_cfg.projector_dims
    if len(dims) != 3:
        raise ValueError(f"Llama 4 expects exactly three projector dimensions, got {dims!r}")
    (merge_in, adapter_hidden), (adapter_in, adapter_out), (connector_in, connector_out) = dims
    if adapter_hidden != adapter_in or adapter_out != connector_in:
        raise ValueError(f"Llama 4 projector dimensions are not composable: {dims!r}")
    for dim_name, dim in (
        ("adapter_hidden", adapter_hidden),
        ("adapter_in", adapter_in),
        ("connector_out", connector_out),
    ):
        if dim % tp_size != 0:
            raise ValueError(f"Llama 4 {dim_name} ({dim}) must be divisible by tp_size ({tp_size})")

    q = common.GEMMQuantMode.bfloat16
    result = [
        # pixel_shuffle turns four 1408-wide patch vectors into one 5632-wide
        # vector before the MLP; it is a real memory-movement stage even though
        # it has no trainable weights.
        ops.ElementWise("encoder_projector_pixel_shuffle", 1, merge_in, merge_in, 0.8),
        # vLLM ColumnParallelLinear: output features are sharded.
        ops.GEMM("encoder_projector_adapter_fc0_gemm", 1, adapter_hidden // tp_size, merge_in, q),
        ops.ElementWise(
            "encoder_projector_adapter_fc0_act",
            1,
            adapter_hidden // tp_size,
            adapter_hidden // tp_size,
            0.8,
        ),
        # vLLM RowParallelLinear: input features are sharded; output is reduced.
        ops.GEMM("encoder_projector_adapter_fc1_gemm", 1, adapter_out, adapter_in // tp_size, q),
        ops.CustomAllReduce("encoder_projector_adapter_ar", 1, adapter_out, tp_size),
        ops.ElementWise("encoder_projector_adapter_fc1_act", 1, adapter_out, adapter_out, 0.8),
        # The final connector is ColumnParallelLinear(gather_output=True).
        ops.GEMM("encoder_projector_mm_gemm", 1, connector_out // tp_size, connector_in, q),
    ]
    if tp_size > 1:
        result.append(
            ops.NCCL(
                "encoder_projector_mm_all_gather",
                1,
                "all_gather",
                num_elements_per_token=connector_out,
                num_gpus=tp_size,
                comm_quant_mode=common.CommQuantMode.half,
            )
        )
    return result


def build_llama4_encoder_ops(
    enc_cfg: common.VisionEncoderConfig,
    tp_size: int,
    enable_encoder_dp: bool = True,
) -> list:
    """Build the checkpoint-faithful Llama 4 image-tower operation graph.

    Unlike Qwen3-VL's shared ViT path, Llama 4 has a learned patch-embedding
    linear, a per-tile CLS token, a pixel-shuffle adaptor, and a separate
    vision-to-text connector.  The backend supplies their different sequence
    lengths: raw patches for patch embedding, raw patches + CLS for the ViT,
    and post-shuffle image tokens for adaptor/connector operations.
    """
    if enc_cfg.image_size <= 0:
        raise ValueError("Llama 4 vision_config.image_size must be positive")
    if not enc_cfg.has_cls_token:
        raise ValueError("Llama 4 vision encoder requires a CLS token")
    if enc_cfg.image_size % enc_cfg.patch_size != 0:
        raise ValueError(
            f"Llama 4 image_size ({enc_cfg.image_size}) must be divisible by patch_size ({enc_cfg.patch_size})"
        )
    merge_stride = enc_cfg.patch_size * enc_cfg.spatial_merge_size
    if enc_cfg.image_size % merge_stride != 0:
        raise ValueError(
            f"Llama 4 image_size ({enc_cfg.image_size}) must be divisible by the merged-patch stride ({merge_stride})"
        )

    encoder_tp = 1 if enable_encoder_dp else tp_size
    if enc_cfg.hidden_size % encoder_tp != 0:
        raise ValueError(
            f"Llama 4 vision hidden_size ({enc_cfg.hidden_size}) must be divisible by tp_size ({encoder_tp})"
        )
    patch_input = enc_cfg.num_channels * enc_cfg.patch_size**2
    q = common.GEMMQuantMode.bfloat16
    result = [
        ops.GEMM("encoder_patch_embedding_gemm", 1, enc_cfg.hidden_size // encoder_tp, patch_input, q),
    ]
    if encoder_tp > 1:
        # vLLM's patch-embedding ColumnParallelLinear uses gather_output=True,
        # so every TP rank sees the full hidden vector before the ViT blocks.
        result.append(
            ops.NCCL(
                "encoder_patch_embedding_all_gather",
                1,
                "all_gather",
                num_elements_per_token=enc_cfg.hidden_size,
                num_gpus=encoder_tp,
                comm_quant_mode=common.CommQuantMode.half,
            )
        )
    result.extend(
        [
            # Class/position additions plus the pre-transformer LayerNorm.
            ops.ElementWise(
                "encoder_class_position_norm",
                1,
                3 * enc_cfg.hidden_size,
                3 * enc_cfg.hidden_size,
                0.8,
            ),
            *_vit_transformer_ops(enc_cfg, encoder_tp),
            ops.ElementWise("encoder_post_norm", 1, 2 * enc_cfg.hidden_size, 2 * enc_cfg.hidden_size, 0.8),
            *_llama4_projector_ops(enc_cfg, encoder_tp),
        ]
    )

    if enable_encoder_dp and tp_size > 1:
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
