# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging

import aiconfigurator.sdk.operations as ops
from aiconfigurator.sdk import common
from aiconfigurator.sdk.models.base import BaseModel, register_model
from aiconfigurator.sdk.models.helpers import calc_expectation

logger = logging.getLogger(__name__)


@register_model("DEEPSEEK")
class DeepSeekModel(BaseModel):
    """
    DeepSeek V3/R1 uses this model impl.
    """

    def __init__(self, topk: int, num_experts: int, moe_inter_size: int, *args) -> None:
        super().__init__(*args)

        # make sure the paralel width is same
        assert (
            self.config.tp_size * self.config.attention_dp_size == self.config.moe_tp_size * self.config.moe_ep_size
        ), (
            f"tp_size ({self.config.tp_size}) * attention_dp_size "
            f"({self.config.attention_dp_size}) should be equal to moe_tp_size "
            f"({self.config.moe_tp_size}) * moe_ep_size ({self.config.moe_ep_size})"
        )

        assert num_experts >= self.config.moe_ep_size, f"ep size cannot be larger than num_experts {num_experts}"
        assert self.config.tp_size * self.config.attention_dp_size <= 256, (
            f"moe ep size {self.config.moe_ep_size} * moe tp size {self.config.moe_tp_size} "
            f"should not be larger than 256"
        )

        self._topk = topk
        self._num_experts = num_experts
        self._moe_inter_size = moe_inter_size

        # used to scale the tpot to reflect mtp effect:
        # 1. mtp will reduce the overall time by expected_tokens_per_step
        # 2. mtp module introduces nextn new transformer layers+linear layers
        #    (we ignore the linear layers for now)
        # 3. special correction in agg step due to we leveraging ctx phase for gen tokens
        #    non-attn part
        # meanwhile, needs to scale the actual bs of generation by nextn,
        # this is covered in inferencesession
        self._mtp_scale_factor = (
            1.0
            / (1 + calc_expectation(self._nextn, self._nextn_accept_rates))
            * (self._nextn + self._num_layers)
            / self._num_layers
        )
        self._power_law_alpha = 1.01

        gemm_quant_mode = self.config.gemm_quant_mode
        moe_quant_mode = self.config.moe_quant_mode

        mla_bmm_quant_mode = (
            common.GEMMQuantMode.fp8
            if gemm_quant_mode != common.GEMMQuantMode.float16
            else common.GEMMQuantMode.float16
        )

        h = self._hidden_size  # 7168
        tp_size = self.config.tp_size
        moe_tp_size = self.config.moe_tp_size
        moe_ep_size = self.config.moe_ep_size
        attention_dp_size = self.config.attention_dp_size
        pp_size = self.config.pp_size

        kvcache_quant_mode = self.config.kvcache_quant_mode
        fmha_quant_mode = self.config.fmha_quant_mode
        workload_distribution = (
            self.config.workload_distribution + f"_{self._power_law_alpha}"
            if self.config.workload_distribution == "power_law"
            else self.config.workload_distribution
        )

        self.context_ops.extend(
            [
                ops.Embedding("context_embedding", 1, self._vocab_size, h, 0.3),
                ops.ElementWise("context_add_norm_1", self._num_layers, 2 * h, 2 * h, 0.8),
                ops.GEMM("context_downscale_gemm", self._num_layers, 2112, h, gemm_quant_mode),
                ops.GEMM(
                    "context_q_b_proj_gemm",
                    self._num_layers,
                    24576 // tp_size,
                    1536,
                    gemm_quant_mode,
                ),
                ops.GEMM(
                    "context_kv_b_proj_gemm",
                    self._num_layers,
                    32768 // tp_size,
                    512,
                    gemm_quant_mode,
                ),  # agg ctx attn part
                ops.ContextMLA(
                    "context_attention",
                    self._num_layers,
                    128 // tp_size,
                    kvcache_quant_mode,
                    fmha_quant_mode,
                ),  # agg ctx attn part
                ops.GEMM(
                    "context_proj_gemm", self._num_layers, h, 128 * 128 // tp_size, gemm_quant_mode
                ),  # agg ctx attn part
                ops.ElementWise("context_add_norm_2", self._num_layers, 2 * h, 2 * h, 0.8),
            ]
        )

        # Context shared moe: gate+up fused into one GEMM (matches TRT-LLM GatedMLP).
        # Context phase runs sequentially (no CUDA Graph), so no OverlapOp here
        # unlike the generation phase which overlaps shared/routed on parallel streams.
        self.context_ops.extend(
            [
                ops.GEMM(
                    "context_shared_gate_up_gemm",
                    self._num_layers,
                    2 * self._moe_inter_size // tp_size,
                    h,
                    gemm_quant_mode,
                ),
                ops.ElementWise(
                    "context_shared_act_gate",
                    self._num_layers,
                    2 * self._moe_inter_size // tp_size,
                    self._moe_inter_size // tp_size,
                    0.8,
                ),
                ops.GEMM(
                    "context_shared_ffn2_gemm",
                    self._num_layers,
                    h,
                    self._moe_inter_size // tp_size,
                    gemm_quant_mode,
                ),
            ]
        )

        # router gemm, num_experts is large enough, cannot be ignored anymore.
        self.context_ops.extend(
            [
                ops.GEMM(
                    "context_router_gemm",
                    self._num_layers,
                    self._num_experts,
                    h,
                    common.GEMMQuantMode.float16,
                )
            ]
        )

        # dispatch tokens to experts, pre-dispatch
        self.context_ops.extend(
            [
                ops.MoEDispatch(
                    "context_moe_pre_dispatch",
                    self._num_layers,
                    h,
                    self._topk,
                    self._num_experts,
                    moe_tp_size,
                    moe_ep_size,
                    attention_dp_size,
                    True,
                    quant_mode=moe_quant_mode,
                )
            ]
        )

        # moe part
        self.context_ops.extend(
            [
                ops.MoE(
                    "context_moe",
                    self._num_layers,
                    h,
                    self._moe_inter_size,
                    self._topk,
                    self._num_experts,
                    moe_tp_size,
                    moe_ep_size,
                    moe_quant_mode,
                    workload_distribution,
                    attention_dp_size,
                )
            ]
        )

        # dispatch tokens to experts, post-dispatch
        self.context_ops.extend(
            [
                ops.MoEDispatch(
                    "context_moe_post_dispatch",
                    self._num_layers,
                    h,
                    self._topk,
                    self._num_experts,
                    moe_tp_size,
                    moe_ep_size,
                    attention_dp_size,
                    False,
                    quant_mode=moe_quant_mode,
                )
            ]
        )

        self.context_ops.extend(
            [
                ops.GEMM(
                    "context_logits_gemm",
                    1,
                    self._vocab_size // tp_size,
                    h,
                    common.GEMMQuantMode.float16,
                )
            ]
        )
        #####generation part, only generation part is scaled by mtp_scale_factor
        self.generation_ops.extend(
            [
                ops.Embedding("generation_embedding", 1 * self._mtp_scale_factor, self._vocab_size, h, 0.3),
                ops.ElementWise(
                    "generation_add_norm_1",
                    self._num_layers * self._mtp_scale_factor,
                    2 * h,
                    2 * h,
                    0.8,
                ),
                ops.GEMM(
                    "generation_downscale_gemm",
                    self._num_layers * self._mtp_scale_factor,
                    2112,
                    h,
                    gemm_quant_mode,
                ),
                ops.GEMM(
                    "generation_q_b_proj_gemm",
                    self._num_layers * self._mtp_scale_factor,
                    24576 // tp_size,
                    1536,
                    gemm_quant_mode,
                ),
                ops.MLABmm(
                    "generation_bmm_pre",
                    self._num_layers * self._mtp_scale_factor,
                    self._num_heads // tp_size,
                    mla_bmm_quant_mode,
                    if_pre=True,
                ),  # agg gen attn part
                ops.GenerationMLA(
                    "generation_attention",
                    self._num_layers * self._mtp_scale_factor,
                    128 // tp_size,
                    kvcache_quant_mode,
                ),  # agg gen attn part
                ops.MLABmm(
                    "generation_bmm_post",
                    self._num_layers * self._mtp_scale_factor,
                    self._num_heads // tp_size,
                    mla_bmm_quant_mode,
                    if_pre=False,
                ),  # agg gen attn part
                ops.GEMM(
                    "generation_proj_gemm",
                    self._num_layers * self._mtp_scale_factor,
                    h,
                    h // tp_size,
                    gemm_quant_mode,
                ),
                ops.ElementWise(
                    "generation_add_norm_2",
                    self._num_layers * self._mtp_scale_factor,
                    2 * h,
                    2 * h,
                    0.8,
                ),
            ]
        )

        # Generation MoE: shared experts and routed experts run in parallel
        # on different CUDA streams (via maybe_execute_in_parallel) when CUDA
        # Graph is enabled. Model with OverlapOp: latency = max(shared, routed).

        # group_b: shared expert path (aux CUDA stream)
        gen_shared_ops = [
            ops.GEMM(
                "generation_shared_gate_up_gemm",
                self._num_layers * self._mtp_scale_factor,
                2 * self._moe_inter_size // tp_size,
                h,
                gemm_quant_mode,
            ),
            ops.ElementWise(
                "generation_shared_act_gate",
                self._num_layers * self._mtp_scale_factor,
                2 * self._moe_inter_size // tp_size,
                self._moe_inter_size // tp_size,
                0.8,
            ),
            ops.GEMM(
                "generation_shared_ffn2_gemm",
                self._num_layers * self._mtp_scale_factor,
                h,
                self._moe_inter_size // tp_size,
                gemm_quant_mode,
            ),
        ]

        # group_a: routed expert path (main CUDA stream)
        gen_routed_ops = [
            ops.GEMM(
                "generation_router_gemm",
                self._num_layers * self._mtp_scale_factor,
                self._num_experts,
                h,
                common.GEMMQuantMode.float16,
            ),
            ops.MoEDispatch(
                "generation_moe_pre_dispatch",
                self._num_layers * self._mtp_scale_factor,
                h,
                self._topk,
                self._num_experts,
                moe_tp_size,
                moe_ep_size,
                attention_dp_size,
                True,
                quant_mode=moe_quant_mode,
            ),
            ops.MoE(
                "generation_moe",
                self._num_layers * self._mtp_scale_factor,
                h,
                self._moe_inter_size,
                self._topk,
                self._num_experts,
                moe_tp_size,
                moe_ep_size,
                moe_quant_mode,
                workload_distribution,
                attention_dp_size,
            ),
            ops.MoEDispatch(
                "generation_moe_post_dispatch",
                self._num_layers * self._mtp_scale_factor,
                h,
                self._topk,
                self._num_experts,
                moe_tp_size,
                moe_ep_size,
                attention_dp_size,
                False,
                quant_mode=moe_quant_mode,
            ),
        ]

        self.generation_ops.append(
            ops.OverlapOp("generation_moe_overlap", group_a=gen_routed_ops, group_b=gen_shared_ops)
        )

        self.generation_ops.extend(
            [
                ops.GEMM(
                    "generation_logits_gemm",
                    1 * self._mtp_scale_factor,
                    self._vocab_size // tp_size,
                    h,
                    common.GEMMQuantMode.float16,
                )
            ]
        )

        # pp
        pp_scale_factor = pp_size - 1
        self.context_ops.append(ops.P2P("context_p2p", pp_scale_factor, h, pp_size))
        self.generation_ops.append(ops.P2P("generation_p2p", pp_scale_factor * self._mtp_scale_factor, h, pp_size))

        # TODO
        # a lot of quantization ops


class TrtllmWideEPDeepSeekModel(BaseModel):
    """
    DeepSeek V3/R1 with TensorRT-LLM WideEP support.

    This model enables WideEP (Wide Expert Parallelism) for TensorRT-LLM backend:
    - MoE computation uses WideEP path (query_wideep_moe_compute) with configurable EPLB modes
    - All2All communication uses WideEP path (query_wideep_alltoall with auto kernel selection)

    Token handling (handled in MoE/MoEDispatch query methods):
    - MoE compute: total tokens (x * attention_dp_size)
    - All2All communication: per-DP tokens (x)

    Kernel auto-selection:
    - MoE kernel: deepgemm (SM>=100 + fp8_block) or moe_torch_flow (default)
    - All2All kernel: NVLinkTwoSided (SM>=100), DeepEP/DeepEPLowLatency (SM>=90), NCCL (fallback)
    """

    def __init__(self, topk: int, num_experts: int, moe_inter_size: int, *args) -> None:
        super().__init__(*args)

        # make sure the parallel width is same
        assert (
            self.config.tp_size * self.config.attention_dp_size == self.config.moe_tp_size * self.config.moe_ep_size
        ), (
            f"tp_size ({self.config.tp_size}) * attention_dp_size "
            f"({self.config.attention_dp_size}) should be equal to moe_tp_size "
            f"({self.config.moe_tp_size}) * moe_ep_size ({self.config.moe_ep_size})"
        )

        assert num_experts >= self.config.moe_ep_size, f"ep size cannot be larger than num_experts {num_experts}"
        assert self.config.tp_size * self.config.attention_dp_size <= 256, (
            f"moe ep size {self.config.moe_ep_size} * moe tp size {self.config.moe_tp_size} "
            f"should not be larger than 256"
        )

        self._topk = topk
        self._num_experts = num_experts
        self._moe_inter_size = moe_inter_size

        # MTP scale factor for generation phase
        self._mtp_scale_factor = (
            1.0
            / (1 + calc_expectation(self._nextn, self._nextn_accept_rates))
            * (self._nextn + self._num_layers)
            / self._num_layers
        )
        self._pdl_factor = 0.9
        self._power_law_alpha = 1.01

        gemm_quant_mode = self.config.gemm_quant_mode
        moe_quant_mode = self.config.moe_quant_mode

        mla_bmm_quant_mode = (
            common.GEMMQuantMode.fp8
            if gemm_quant_mode != common.GEMMQuantMode.float16
            else common.GEMMQuantMode.float16
        )

        h = self._hidden_size  # 7168
        tp_size = self.config.tp_size
        moe_tp_size = self.config.moe_tp_size
        moe_ep_size = self.config.moe_ep_size
        attention_dp_size = self.config.attention_dp_size
        pp_size = self.config.pp_size

        kvcache_quant_mode = self.config.kvcache_quant_mode
        fmha_quant_mode = self.config.fmha_quant_mode

        # WideEP workload distribution
        # - EPLB off: "power_law_1.01" (no _eplb suffix)
        # - EPLB on/redundant: "power_law_1.01_eplb" (with _eplb suffix)
        eplb_enabled = self.config.enable_eplb
        if self.config.workload_distribution == "power_law":
            if eplb_enabled:
                workload_distribution = f"{self.config.workload_distribution}_{self._power_law_alpha}_eplb"
            else:
                workload_distribution = f"{self.config.workload_distribution}_{self._power_law_alpha}"
        else:
            workload_distribution = self.config.workload_distribution

        # ===================== WideEP Configuration Validation =====================
        # Based on TensorRT-LLM WideEPMoE constraints (fused_moe_wide_ep.py)

        # 1. Attention DP must be enabled for WideEP
        if attention_dp_size <= 1:
            raise ValueError(
                f"WideEP requires attention_dp_size > 1, got {attention_dp_size}. "
                "Attention DP should be used with WideEP."
            )

        # 2. EP size must be > 1 for WideEP (parallel_size > 1)
        if moe_ep_size <= 1:
            raise ValueError(
                f"WideEP requires moe_ep_size > 1, got {moe_ep_size}. "
                "WideEP should only be enabled with parallel_size > 1."
            )

        # 3. EP size must be > top_k for AlltoAll to be effective
        # FIXME: this warning should make the comm mode fallback to NCCL!!
        if moe_ep_size <= topk:
            logger.warning(
                f"moe_ep_size ({moe_ep_size}) <= top_k ({topk}), "
                "AlltoAll communication will be disabled. Consider increasing moe_ep_size."
            )

        # 4. num_slots validation
        wideep_num_slots = self.config.wideep_num_slots if self.config.wideep_num_slots else num_experts

        # num_slots must be >= num_experts
        if wideep_num_slots < num_experts:
            raise ValueError(
                f"wideep_num_slots ({wideep_num_slots}) must be >= num_experts ({num_experts}). "
                "There should be at least num_experts slots in the model engine."
            )

        # When EPLB is off, num_slots must equal num_experts
        if not eplb_enabled and wideep_num_slots != num_experts:
            raise ValueError(
                f"When enable_eplb=False, wideep_num_slots ({wideep_num_slots}) must equal "
                f"num_experts ({num_experts}). Redundant slots require EPLB to be enabled."
            )

        # ===================== Context Phase =====================
        # Note: Context phase does NOT use CUDA Graph, so maybe_execute_in_parallel
        # falls back to sequential execution. All ops are modeled sequentially here.
        self.context_ops.extend(
            [
                ops.Embedding("context_embedding", 1, self._vocab_size, h, 0.3),
                ops.ElementWise("context_add_norm_1", self._num_layers, 2 * h, 2 * h, 0.8),
                # kv_a_proj_with_mqa: projects hidden_size -> compressed_dim (1536+512+64=2112)
                ops.GEMM("context_downscale_gemm", self._num_layers, 2112, h, gemm_quant_mode),
                # q_a_layernorm: RMSNorm on q_compressed (dim=1536)
                ops.ElementWise("context_q_a_layernorm", self._num_layers, 1536, 1536, 0.8),
                ops.GEMM(
                    "context_q_b_proj_gemm",
                    self._num_layers,
                    24576 // tp_size,
                    1536,
                    gemm_quant_mode,
                ),
                ops.GEMM(
                    "context_kv_b_proj_gemm",
                    self._num_layers,
                    32768 // tp_size,
                    512,
                    gemm_quant_mode,
                ),
                ops.ContextMLA(
                    "context_attention",
                    self._num_layers,
                    128 // tp_size,
                    kvcache_quant_mode,
                    fmha_quant_mode,
                ),
                ops.GEMM("context_proj_gemm", self._num_layers, h, 128 * 128 // tp_size, gemm_quant_mode),
                ops.ElementWise("context_add_norm_2", self._num_layers, 2 * h, 2 * h, 0.8),
            ]
        )

        # shared moe (sequential in context phase - no CUDA Graph overlap)
        # In WideEP ADP mode, shared_tp_size=1: each rank computes full shared expert.
        # TRT-LLM uses fused gate_up_proj: one GEMM with output dim = 2 * inter_size.
        self.context_ops.extend(
            [
                ops.GEMM(
                    "context_shared_gate_up_gemm",
                    self._num_layers,
                    2 * self._moe_inter_size,
                    h,
                    gemm_quant_mode,
                ),
                ops.ElementWise(
                    "context_shared_act_gate",
                    self._num_layers,
                    2 * self._moe_inter_size,
                    self._moe_inter_size,
                    0.8,
                ),
                ops.GEMM(
                    "context_shared_ffn2_gemm",
                    self._num_layers,
                    h,
                    self._moe_inter_size,
                    gemm_quant_mode,
                ),
            ]
        )

        # router gemm
        self.context_ops.extend(
            [
                ops.GEMM(
                    "context_router_gemm",
                    self._num_layers,
                    self._num_experts,
                    h,
                    common.GEMMQuantMode.float16,
                )
            ]
        )

        # WideEP: dispatch tokens to experts, pre-dispatch (prepare + dispatch)
        self.context_ops.extend(
            [
                ops.TrtLLMWideEPMoEDispatch(
                    "context_moe_pre_dispatch",
                    self._num_layers,
                    h,
                    self._topk,
                    self._num_experts,
                    moe_tp_size,
                    moe_ep_size,
                    attention_dp_size,
                    True,  # pre_dispatch
                    quant_mode=moe_quant_mode,
                )
            ]
        )

        # WideEP: MoE computation with EPLB support
        self.context_ops.extend(
            [
                ops.TrtLLMWideEPMoE(
                    "context_moe",
                    self._num_layers,
                    h,
                    self._moe_inter_size,
                    self._topk,
                    self._num_experts,
                    moe_tp_size,
                    moe_ep_size,
                    moe_quant_mode,
                    workload_distribution,
                    attention_dp_size,
                    num_slots=wideep_num_slots,
                )
            ]
        )

        # WideEP: dispatch tokens to experts, post-dispatch (combine)
        self.context_ops.extend(
            [
                ops.TrtLLMWideEPMoEDispatch(
                    "context_moe_post_dispatch",
                    self._num_layers,
                    h,
                    self._topk,
                    self._num_experts,
                    moe_tp_size,
                    moe_ep_size,
                    attention_dp_size,
                    False,  # post_dispatch (combine)
                    quant_mode=moe_quant_mode,
                )
            ]
        )

        # moe_reduce_add_shared_output: sum routed output over top_k + add shared output
        self.context_ops.append(
            ops.ElementWise(
                "context_moe_reduce_add",
                self._num_layers,
                2 * h,
                h,
                0.8,
            )
        )

        self.context_ops.extend(
            [
                ops.GEMM(
                    "context_logits_gemm",
                    1,
                    self._vocab_size // tp_size,
                    h,
                    common.GEMMQuantMode.float16,
                )
            ]
        )

        # ===================== Generation Phase =====================
        # _gen_layer_scale = num_layers * mtp_scale * pdl_factor
        self.generation_ops.extend(
            [
                ops.Embedding("generation_embedding", 1 * self._mtp_scale_factor, self._vocab_size, h, 0.3),
                ops.ElementWise(
                    "generation_add_norm_1",
                    self._num_layers * self._mtp_scale_factor * self._pdl_factor,
                    2 * h,
                    2 * h,
                    0.8,
                ),
                # kv_a_proj_with_mqa: projects hidden_size -> compressed_dim (1536+512+64=2112)
                ops.GEMM(
                    "generation_downscale_gemm",
                    self._num_layers * self._mtp_scale_factor * self._pdl_factor,
                    2112,
                    h,
                    gemm_quant_mode,
                ),
                # q_a_layernorm: RMSNorm on q_compressed (dim=1536)
                # In TRT-LLM, kv_a_layernorm (dim=512) runs in parallel but is much smaller,
                # so we model only q_a_layernorm as the dominant one.
                ops.ElementWise(
                    "generation_q_a_layernorm",
                    self._num_layers * self._mtp_scale_factor * self._pdl_factor,
                    1536,
                    1536,
                    0.8,
                ),
                ops.GEMM(
                    "generation_q_b_proj_gemm",
                    self._num_layers * self._mtp_scale_factor * self._pdl_factor,
                    24576 // tp_size,
                    1536,
                    gemm_quant_mode,
                ),
                # BMM_pre (Absorption) || RoPE+KV cache prep (overlap on two streams)
                # Main stream: q_nope * W_absorption -> absorbed_q
                # Aux stream: RoPE(q_pe) + write compressed_kv to KV cache
                # Effective latency = max(bmm_pre, rope_kvcache)
                ops.OverlapOp(
                    "generation_bmm_rope_overlap",
                    group_a=[
                        ops.MLABmm(
                            "generation_bmm_pre",
                            self._num_layers * self._mtp_scale_factor * self._pdl_factor,
                            self._num_heads // tp_size,
                            mla_bmm_quant_mode,
                            if_pre=True,
                        ),
                    ],
                    group_b=[
                        # mla_rope_generation: RoPE on q_pe (64d) + KV cache write (512+64=576d)
                        ops.ElementWise(
                            "generation_rope_kvcache",
                            self._num_layers * self._mtp_scale_factor * self._pdl_factor,
                            576,  # kv_lora_rank(512) + qk_rope_head_dim(64)
                            576,
                            0.8,
                        ),
                    ],
                ),
                ops.GenerationMLA(
                    "generation_attention",
                    self._num_layers * self._mtp_scale_factor * self._pdl_factor,
                    128 // tp_size,
                    kvcache_quant_mode,
                ),
                ops.MLABmm(
                    "generation_bmm_post",
                    self._num_layers * self._mtp_scale_factor * self._pdl_factor,
                    self._num_heads // tp_size,
                    mla_bmm_quant_mode,
                    if_pre=False,
                ),
                ops.GEMM(
                    "generation_proj_gemm",
                    self._num_layers * self._mtp_scale_factor * self._pdl_factor,
                    h,
                    h // tp_size,
                    gemm_quant_mode,
                ),
                ops.ElementWise(
                    "generation_add_norm_2",
                    self._num_layers * self._mtp_scale_factor * self._pdl_factor,
                    2 * h,
                    2 * h,
                    0.8,
                ),
            ]
        )

        # ---- MoE: Shared Expert || Routed Expert (OverlapOp) ----
        # In TRT-LLM generation phase (CUDA Graph enabled), shared expert runs
        # on aux stream in parallel with routed expert on main stream.
        # Latency = max(routed_path, shared_path) instead of sum.

        # Group B (Aux Stream): Shared Expert
        # Note: In WideEP ADP mode, shared_tp_size=1 (no TP for shared expert),
        # so we use full moe_inter_size without dividing by tp_size.
        # TRT-LLM uses fused gate_up_proj: one GEMM with output dim = 2 * inter_size.
        _shared_expert_ops = [
            ops.GEMM(
                "generation_shared_gate_up_gemm",
                self._num_layers * self._mtp_scale_factor * self._pdl_factor,
                2 * self._moe_inter_size,
                h,
                gemm_quant_mode,
            ),
            ops.ElementWise(
                "generation_shared_act_gate",
                self._num_layers * self._mtp_scale_factor * self._pdl_factor,
                2 * self._moe_inter_size,
                self._moe_inter_size,
                0.8,
            ),
            ops.GEMM(
                "generation_shared_ffn2_gemm",
                self._num_layers * self._mtp_scale_factor * self._pdl_factor,
                h,
                self._moe_inter_size,
                gemm_quant_mode,
            ),
        ]

        # Group A (Main Stream): Router + AllToAll Dispatch + MoE Compute + AllToAll Combine
        _routed_expert_ops = [
            ops.GEMM(
                "generation_router_gemm",
                self._num_layers * self._mtp_scale_factor * self._pdl_factor,
                self._num_experts,
                h,
                common.GEMMQuantMode.float16,
            ),
            ops.TrtLLMWideEPMoEDispatch(
                "generation_moe_pre_dispatch",
                self._num_layers * self._mtp_scale_factor * self._pdl_factor,
                h,
                self._topk,
                self._num_experts,
                moe_tp_size,
                moe_ep_size,
                attention_dp_size,
                True,  # pre_dispatch
                quant_mode=moe_quant_mode,
            ),
            ops.TrtLLMWideEPMoE(
                "generation_moe",
                self._num_layers * self._mtp_scale_factor * self._pdl_factor,
                h,
                self._moe_inter_size,
                self._topk,
                self._num_experts,
                moe_tp_size,
                moe_ep_size,
                moe_quant_mode,
                workload_distribution,
                attention_dp_size,
                num_slots=wideep_num_slots,
            ),
            ops.TrtLLMWideEPMoEDispatch(
                "generation_moe_post_dispatch",
                self._num_layers * self._mtp_scale_factor * self._pdl_factor,
                h,
                self._topk,
                self._num_experts,
                moe_tp_size,
                moe_ep_size,
                attention_dp_size,
                False,  # post_dispatch (combine)
                quant_mode=moe_quant_mode,
                use_low_precision_combine=(moe_quant_mode == common.MoEQuantMode.nvfp4),
            ),
        ]

        self.generation_ops.append(
            ops.OverlapOp(
                "generation_moe_overlap",
                group_a=_routed_expert_ops,
                group_b=_shared_expert_ops,
            )
        )

        # moe_reduce_add_shared_output: sum routed output over top_k + add shared output
        # This runs after both streams synchronize.
        self.generation_ops.append(
            ops.ElementWise(
                "generation_moe_reduce_add",
                self._num_layers * self._mtp_scale_factor * self._pdl_factor,
                2 * h,
                h,
                0.8,
            )
        )

        self.generation_ops.extend(
            [
                ops.GEMM(
                    "generation_logits_gemm",
                    1 * self._mtp_scale_factor,
                    self._vocab_size // tp_size,
                    h,
                    common.GEMMQuantMode.float16,
                )
            ]
        )

        # pp
        pp_scale_factor = pp_size - 1
        self.context_ops.append(ops.P2P("context_p2p", pp_scale_factor, h, pp_size))
        self.generation_ops.append(ops.P2P("generation_p2p", pp_scale_factor * self._mtp_scale_factor, h, pp_size))


class WideEPDeepSeekModel(BaseModel):
    """
    DeepSeek V3/R1 disaggregated model for SGLang backend.
    """

    def __init__(self, topk: int, num_experts: int, moe_inter_size: int, *args) -> None:
        super().__init__(*args)

        assert num_experts >= self.config.moe_ep_size, f"ep size cannot be larger than num_experts {num_experts}"

        self._topk = topk
        self._num_experts = num_experts
        self._moe_inter_size = moe_inter_size
        self._mtp_scale_factor = (
            1.0
            / (1 + calc_expectation(self._nextn, self._nextn_accept_rates))
            * (self._nextn + self._num_layers)
            / self._num_layers
        )

        h = self._hidden_size
        tp_size = self.config.tp_size
        moe_tp_size = self.config.moe_tp_size
        moe_ep_size = self.config.moe_ep_size
        attention_dp_size = self.config.attention_dp_size

        kvcache_quant_mode = self.config.kvcache_quant_mode
        fmha_quant_mode = self.config.fmha_quant_mode
        moe_quant_mode = self.config.moe_quant_mode
        gemm_quant_mode = self.config.gemm_quant_mode
        moe_backend = self.config.moe_backend
        attn_backend = self.config.attention_backend

        self._power_law_alpha_prefill = 0.6 if self.config.enable_eplb else 1.01
        self._power_law_alpha_decode = 1.01

        context_workload_distribution = (
            self.config.workload_distribution + f"_{self._power_law_alpha_prefill}"
            if self.config.workload_distribution == "power_law"
            else self.config.workload_distribution
        )
        generation_workload_distribution = (
            self.config.workload_distribution + f"_{self._power_law_alpha_decode}"
            if self.config.workload_distribution == "power_law"
            else self.config.workload_distribution
        )

        sms = self.config.sms

        # qkv_a projection (fused q_a + kv_a + rope): hidden_size -> q_lora_rank + kv_lora_rank + qk_rope_head_dim
        # This is replicated on every GPU (not TP-sharded), matching narrow EP's context_downscale_gemm.
        # In sglang >=0.5.6, qkv_a_proj is computed outside the MLA attention forward via communicator,
        # so it must be modeled as a separate GEMM op rather than included in WideEPContextMLA.
        self.context_ops.extend(
            [
                ops.GEMM(
                    "context_qkv_a_proj_gemm",
                    self._num_layers,
                    1536 + 512 + 64,  # q_lora_rank + kv_lora_rank + qk_rope_head_dim = 2112
                    h,
                    gemm_quant_mode,
                    scale_num_tokens=tp_size,
                ),
            ]
        )

        # context mla attention
        self.context_ops.extend(
            [
                *(
                    [
                        ops.NCCL(
                            "context_tp_all_gather",
                            self._num_layers,
                            "all_gather",
                            h,
                            tp_size,
                            common.CommQuantMode.half,
                        )
                    ]
                    if tp_size > 1
                    else []
                ),
                ops.WideEPContextMLA(
                    "context_attention",
                    self._num_layers,
                    tp_size,
                    kvcache_quant_mode,
                    fmha_quant_mode,
                    attn_backend,
                ),
                *(
                    [
                        ops.NCCL(
                            "context_tp_reduce_scatter",
                            self._num_layers,
                            "reduce_scatter",
                            h,
                            tp_size,
                            common.CommQuantMode.half,
                        )
                    ]
                    if tp_size > 1
                    else []
                ),
            ]
        )

        # shared expert
        # TODO: support shared expert TP
        self.context_ops.extend(
            [
                ops.GEMM(
                    "context_gate_ffn1_gemm",
                    self._num_layers,
                    2 * self._moe_inter_size,
                    h,
                    gemm_quant_mode,
                    scale_num_tokens=tp_size,
                ),
                ops.ElementWise(
                    "context_act_gate",
                    self._num_layers,
                    2 * self._moe_inter_size,
                    self._moe_inter_size,
                    0.8,
                    scale_num_tokens=tp_size,
                ),
                ops.GEMM(
                    "context_ffn2_gemm",
                    self._num_layers,
                    h,
                    self._moe_inter_size,
                    gemm_quant_mode,
                    scale_num_tokens=tp_size,
                ),
            ]
        )

        # dispatch tokens to experts
        self.context_ops.extend(
            [
                ops.MoEDispatch(
                    "context_moe_pre_dispatch",
                    self._num_layers,
                    h,
                    self._topk,
                    self._num_experts,
                    moe_tp_size,
                    moe_ep_size,
                    attention_dp_size,
                    True,
                    quant_mode=moe_quant_mode,
                    sms=sms,
                    moe_backend=moe_backend,
                    is_context=True,
                    scale_num_tokens=tp_size,
                )
            ]
        )

        # moe computation
        self.context_ops.extend(
            [
                ops.MoE(
                    "context_moe",
                    self._num_layers,
                    h,
                    self._moe_inter_size,
                    self._topk,
                    self._num_experts,
                    moe_tp_size,
                    moe_ep_size,
                    moe_quant_mode,
                    context_workload_distribution,
                    attention_dp_size,
                    is_context=True,
                    moe_backend=moe_backend,
                    enable_eplb=self.config.enable_eplb,
                )
            ]
        )

        # qkv_a projection for generation (same as context but per-token, not per-seq)
        self.generation_ops.extend(
            [
                ops.GEMM(
                    "generation_qkv_a_proj_gemm",
                    self._num_layers * self._mtp_scale_factor,
                    1536 + 512 + 64,  # q_lora_rank + kv_lora_rank + qk_rope_head_dim = 2112
                    h,
                    gemm_quant_mode,
                ),
            ]
        )

        # generation mla attention
        self.generation_ops.extend(
            [
                ops.WideEPGenerationMLA(
                    "generation_attention",
                    self._num_layers * self._mtp_scale_factor,
                    tp_size,
                    kvcache_quant_mode,
                    fmha_quant_mode,
                    attn_backend,
                )
            ]
        )

        # shared expert
        self.generation_ops.extend(
            [
                ops.GEMM(
                    "generation_gate_ffn1_gemm",
                    self._num_layers * self._mtp_scale_factor,
                    2 * self._moe_inter_size,
                    h,
                    gemm_quant_mode,
                ),
                ops.ElementWise(
                    "generation_act_gate",
                    self._num_layers * self._mtp_scale_factor,
                    2 * self._moe_inter_size,
                    self._moe_inter_size,
                    0.8,
                ),
                ops.GEMM(
                    "generation_ffn2_gemm",
                    self._num_layers * self._mtp_scale_factor,
                    h,
                    self._moe_inter_size,
                    gemm_quant_mode,
                ),
            ]
        )

        # dispatch tokens to experts
        self.generation_ops.extend(
            [
                ops.MoEDispatch(
                    "generation_moe_pre_dispatch",
                    self._num_layers * self._mtp_scale_factor,
                    h,
                    self._topk,
                    self._num_experts,
                    moe_tp_size,
                    moe_ep_size,
                    attention_dp_size,
                    True,
                    quant_mode=moe_quant_mode,
                    sms=sms,
                    moe_backend=moe_backend,
                    is_context=False,
                )
            ]
        )

        # moe computation
        self.generation_ops.extend(
            [
                ops.MoE(
                    "generation_moe",
                    self._num_layers * self._mtp_scale_factor,
                    h,
                    self._moe_inter_size,
                    self._topk,
                    self._num_experts,
                    moe_tp_size,
                    moe_ep_size,
                    moe_quant_mode,
                    generation_workload_distribution,
                    attention_dp_size,
                    is_context=False,
                    moe_backend=moe_backend,
                    enable_eplb=False,
                )
            ]
        )
