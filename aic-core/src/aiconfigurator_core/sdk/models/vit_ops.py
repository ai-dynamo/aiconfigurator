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

  _projector_ops  →  2*P ops + 1 AR  (or 0 ops if projector_dims is empty):
    encoder_projector_fc{i}_gemm  GEMM
    encoder_projector_fc{i}_act   ElementWise  (omitted for final layer)
    encoder_projector_ar          CustomAllReduce

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
    """Build the 10 ViT transformer block ops (each repeated enc_cfg.depth times).

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

    return [
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
            partial_rotary_factor=enc_cfg.partial_rotary_factor,
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


def build_encoder_ops(enc_cfg: common.VisionEncoderConfig, tp_size: int) -> list:
    """Build the complete list of encoder ops for a ViT-based vision encoder.

    Combines ViT transformer ops (10 ops x depth repetitions) with projector ops
    (2 x n_layers + 1 ops with AllReduce, or 0 if no projector configured).

    Args:
        enc_cfg: VisionEncoderConfig populated with ViT and projector parameters.
        tp_size: Tensor-parallel degree.  Must evenly divide num_heads and
                 intermediate_size when tp_size > 1.

    Returns:
        Flat list of operation objects ready to assign to model.encoder_ops.
    """
    return _vit_transformer_ops(enc_cfg, tp_size) + _projector_ops(enc_cfg, tp_size)
