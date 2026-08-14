# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging

import aiconfigurator_core.sdk.operations as ops
from aiconfigurator_core.sdk import common
from aiconfigurator_core.sdk.models.base import BaseModel, register_model
from aiconfigurator_core.sdk.models.blocks.moe import MoEBlockShape
from aiconfigurator_core.sdk.models.helpers import (
    attention_projection_exclusions,
    build_large_ep_moe_ops,
    large_ep_gpus_per_node,
    mtp_scale_factor,
    quant_exclude_patterns,
    validate_trtllm_large_ep,
)

logger = logging.getLogger(__name__)

# Historical DeepSeek-compatible 128-head kernel-table convention: sglang/
# trtllm kernel queries key at ``128 // tp_size`` even for Kimi K2.5
# (64-native). Collection side pinned in the Kimi case YAML; rationale and the
# true-geometry follow-up in docs/perf_database/head-axis-keying.md. Scoped to
# THIS class — Kimi-K3 (#1435) already queries true local geometry, and the
# WideEP MLA ops on the sglang large-EP path use 128 as DSV3's actual
# native, not this convention.
_MLA_KERNEL_TABLE_NATIVE_HEADS = 128


@register_model("DEEPSEEK", "KIMIK25")
class DeepSeekModel(BaseModel):
    """
    DeepSeek V3/R1 uses this model impl. Also serves as the entry point for
    Kimi K2.5 (registered under the ``KIMIK25`` family).
    """

    @classmethod
    def supports_cp(cls, backend_name: str) -> bool:
        # Dense MLA prefill CP: SGLang AllGather (uniform full/cp, like 1145).
        # Gates the whole DEEPSEEK family -- both the fused and the sglang
        # large-EP branches of __init__ are CP-wired; trtllm is rejected here
        # (its Ring CP has no perf data / _cp_attn_comm_ops).
        return backend_name == "sglang"

    @classmethod
    def create(cls, model_info: dict, model_config, backend_name: str) -> BaseModel:
        moe_args = (model_info["topk"], model_info["num_experts"], model_info["moe_inter_size"])
        base_args = (
            model_info["model_path"],
            model_info["model_family"],
            model_info["architecture"],
            model_info["layers"],
            model_info["n"],
            model_info["n_kv"],
            model_info["d"],
            model_info["hidden_size"],
            model_info["inter_size"],
            model_info["vocab"],
            model_info["context"],
            model_config,
        )
        extra_params = model_info["extra_params"]
        # One class for the whole family (DEEPSEEK + KIMIK25), both regimes:
        # ``__init__`` branches on ``model_config.moe_comm_backend`` for large
        # EP. backend_name is threaded so backend-specific modeling (vLLM TP
        # allreduce / head size, the sglang large-EP attention stack) fires.
        # Per-checkpoint, per-projection fact: Kimi-K2.5/R1 NVFP4 exclude the
        # whole self_attn block from quantization; DeepSeek-V3.1-NVFP4 excludes
        # only q/kv projections and keeps o_proj NVFP4; native FP8 checkpoints
        # exclude nothing. Drives per-GEMM dtypes and the MLA-module perf key.
        attn_exclusions = attention_projection_exclusions(model_info.get("raw_config") or {})
        shared_expert_excluded = any(
            "shared_expert" in str(pattern).lower()
            for pattern in quant_exclude_patterns(model_info.get("raw_config") or {})
        )
        shared_expert_quant_mode = model_config.gemm_quant_mode
        if shared_expert_excluded and not model_info["gemm_quant_mode_is_explicit"]:
            shared_expert_quant_mode = common.GEMMQuantMode.bfloat16
        return cls(
            *moe_args,
            *base_args,
            extra_params,
            backend_name=backend_name,
            attention_quant_exclusions=attn_exclusions,
            shared_expert_quant_mode=shared_expert_quant_mode,
        )

    #: TRT-LLM large-EP decode PDL overlap discount, transcribed from the
    #: deleted ``TrtllmWideEPDeepSeekModel._pdl_factor`` (deepseek.py:604 at
    #: commit 8372e60). Scales every decode-layer op, attention included.
    _PDL_FACTOR = 0.9

    def _large_ep_moe_ops(self, phase: str, shape: MoEBlockShape, scale_factor: float) -> list:
        """MoE block for a large-EP config (``cfg.moe_comm_backend`` set).

        Body shared with the DeepSeekV32 family in
        ``helpers.build_large_ep_moe_ops`` (distribution transcription notes
        live there); this family keeps the model-wide shared-expert quant mode.
        """
        return build_large_ep_moe_ops(
            phase,
            shape,
            self.config,
            scale_factor=scale_factor,
            backend_name=self._backend_name,
            model_family=self.model_family,
            power_law_alpha=self._power_law_alpha,
            gpus_per_node=self._gpus_per_node,
            shared_gemm_quant_mode=self._shared_expert_quant_mode,
        )

    def __init__(
        self,
        topk: int,
        num_experts: int,
        moe_inter_size: int,
        *args,
        backend_name: str = "",
        attention_quant_exclusions: frozenset = frozenset(),
        shared_expert_quant_mode: common.GEMMQuantMode | None = None,
    ) -> None:
        super().__init__(*args)
        # Resolve vLLM attention head size. MLA models (e.g., KIMI K2.5) store v_head_dim=128
        # in extra_params; generic hidden_size // n_heads would give the wrong value (e.g., 112).
        self._vllm_head_size = (
            self.extra_params.get("v_head_dim") or self._head_size
            if isinstance(self.extra_params, dict)
            else self._head_size
        )

        self._backend_name = backend_name
        # Large EP: the enumerator sets a per-phase MoE comm backend; the MoE
        # block builder then emits the MoEAllToAll/ModeledEPMoE graph and
        # sglang swaps in its deepep attention stack. No user flag selects it -- see
        # ``ModelConfig.moe_comm_backend``.
        self._is_large_ep = bool(self.config.moe_comm_backend)
        # Node width is a hardware fact with no default: an unset value would
        # silently mis-price cross-node all-to-all (see large_ep_gpus_per_node).
        self._gpus_per_node = large_ep_gpus_per_node(self.config) if self._is_large_ep else 0

        # make sure the paralel width is same (cp is an independent attention
        # dimension that also contributes to the width the MoE must match)
        assert (
            self.config.tp_size * self.config.attention_dp_size * self.config.cp_size
            == self.config.moe_tp_size * self.config.moe_ep_size
        ), (
            f"tp_size ({self.config.tp_size}) * attention_dp_size "
            f"({self.config.attention_dp_size}) * cp_size ({self.config.cp_size}) should be equal to "
            f"moe_tp_size ({self.config.moe_tp_size}) * moe_ep_size ({self.config.moe_ep_size})"
        )

        assert num_experts >= self.config.moe_ep_size, f"ep size cannot be larger than num_experts {num_experts}"

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
        self._mtp_scale_factor = mtp_scale_factor(self._nextn, self._num_layers)
        self._power_law_alpha = 1.01

        gemm_quant_mode = self.config.gemm_quant_mode
        shared_gemm_quant_mode = shared_expert_quant_mode or gemm_quant_mode
        self._shared_expert_quant_mode = shared_gemm_quant_mode
        moe_quant_mode = self.config.moe_quant_mode

        # Attention projections follow the checkpoint's PER-PROJECTION dtype,
        # not the global gemm mode: serving loads excluded projections in BF16
        # (vLLM ReplicatedLinear/ColumnParallelLinear pick up the unquantized
        # tensors) while non-excluded ones stay quantized — V3.1-NVFP4 keeps
        # o_proj NVFP4 with BF16 q/kv. Drives per-GEMM perf rows and
        # get_weights() byte widths.
        excl = attention_quant_exclusions

        def _attn_mode(group: str) -> common.GEMMQuantMode:
            return common.GEMMQuantMode.bfloat16 if group in excl else gemm_quant_mode

        attn_q_gemm_quant_mode = _attn_mode("q")
        attn_kv_gemm_quant_mode = _attn_mode("kv")
        attn_o_gemm_quant_mode = _attn_mode("o")
        # downscale GEMM fuses q_a + kv_a: BF16 only when both groups are excluded.
        attn_downscale_gemm_quant_mode = common.GEMMQuantMode.bfloat16 if {"q", "kv"} <= excl else gemm_quant_mode
        # Module perf rows are keyed by ONE gemm_type; when the checkpoint
        # mixes dtypes across projections no row matches exactly, so key on
        # o_proj's dtype — the largest projection by bytes and FLOPs (heads x
        # v_head_dim x hidden ~ 58% of block weights at 64 heads).
        attn_modes = {attn_q_gemm_quant_mode, attn_kv_gemm_quant_mode, attn_o_gemm_quant_mode}
        # A module perf row is keyed by ONE gemm_type, so an exact module
        # identity exists only when every projection shares a dtype. Mixed
        # checkpoints (V3.1/V3.2-NVFP4: BF16 q/kv + NVFP4 o_proj) must use the
        # granular per-projection path — an all-NVFP4 module profile measures
        # kernels the checkpoint never runs.
        attn_module_identity_exact = len(attn_modes) == 1
        attn_gemm_quant_mode = next(iter(attn_modes)) if attn_module_identity_exact else attn_o_gemm_quant_mode
        # Absorbed kv_b BMMs inherit the kv projection dtype.
        mla_bmm_quant_mode = (
            common.GEMMQuantMode.bfloat16
            if attn_kv_gemm_quant_mode == common.GEMMQuantMode.bfloat16
            else common.GEMMQuantMode.fp8
            if gemm_quant_mode != common.GEMMQuantMode.bfloat16
            else common.GEMMQuantMode.bfloat16
        )
        extra = self.extra_params if isinstance(self.extra_params, dict) else {}
        q_lora_rank = int(extra.get("q_lora_rank") or 1536)
        kv_lora_rank = int(extra.get("kv_lora_rank") or 512)
        qk_nope_head_dim = int(extra.get("qk_nope_head_dim") or 128)
        qk_rope_head_dim = int(extra.get("qk_rope_head_dim") or 64)
        v_head_dim = int(extra.get("v_head_dim") or 128)
        q_projection_width = self._num_heads * (qk_nope_head_dim + qk_rope_head_dim)
        kv_projection_width = self._num_heads * (qk_nope_head_dim + v_head_dim)
        o_projection_width = self._num_heads * v_head_dim
        downscale_width = q_lora_rank + kv_lora_rank + qk_rope_head_dim

        # Perf-row key for the profiled MLA attention module. When the
        # checkpoint keeps attention projections unquantized (every NVFP4
        # DeepSeek/Kimi release), serving executes BF16 projection GEMMs, so
        # querying the module table at the global gemm_quant_mode (nvfp4)
        # selects rows for kernels that never run. Weights accounting is NOT
        # switched here: attention byte-width, MoE layer count and encoder
        # residency are coupled and land together in a follow-up (#1396).

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
        # Context parallelism (sglang AllGather, prefill-only), 1145 uniform form:
        # the MLA module/attention count is scaled by 1/cp (attn_count_div);
        # token-major ops divide tokens by cp (seq_split); MoEDispatch uses
        # attn_cp_size for the AG_hidden+RS comm; one MLA-latent KV all-gather
        # (_cp_attn_comm_ops). Generation/decode is NOT CP-modeled.
        cp = self.config.cp_size
        cp_style = self.config.cp_style
        attn_count_div = cp if cp_style in ("allgather", "ring") else 1

        # MoE block shape (large-EP regime only; the fused spans below stay
        # hand-wired because their generation dialect -- is_context=False
        # dispatches inside an OverlapOp -- differs from the builder's).
        moe_shape = MoEBlockShape(
            hidden_size=h,
            moe_inter_size=self._moe_inter_size,
            topk=self._topk,
            num_experts=self._num_experts,
            # The legacy wideEP graphs model exactly one full-size shared
            # expert (WideEP ADP mode, shared_tp_size=1), which is what every
            # DeepSeek/Kimi checkpoint carries.
            num_shared_experts=1,
            # Descriptor-only: the builder scales by the model-owned
            # scale_factor (the legacy classes use the full layer count).
            num_moe_layers=self._num_layers,
        )
        if self._is_large_ep and backend_name == "trtllm":
            # ===== TRT-LLM large EP (wideEP) =====
            # Attention + non-MoE wiring transcribed verbatim from the deleted
            # TrtllmWideEPDeepSeekModel (deepseek.py:638-1049 at commit
            # 8372e60): the GRANULAR MLA stack (no MLAModule FallbackOp -- this
            # is what the ``context_mla_granular`` / ``generation_mla``
            # capability keys feed), an explicit q_a_layernorm, the BMM/RoPE
            # overlap, and the PDL-discounted decode scale.
            validate_trtllm_large_ep(
                attention_dp_size=attention_dp_size,
                moe_ep_size=moe_ep_size,
                topk=topk,
                num_experts=num_experts,
                wideep_num_slots=self.config.wideep_num_slots,
                enable_eplb=self.config.enable_eplb,
            )
            # Attention projections follow the checkpoint's per-projection
            # dtype (attn_*_gemm_quant_mode above) — #1423, carried from the
            # deleted TrtllmWideEPDeepSeekModel.
            # _gen_layer_scale = num_layers * mtp_scale * pdl_factor
            gen_scale = self._num_layers * self._mtp_scale_factor * self._PDL_FACTOR
            # Context phase does NOT use CUDA Graph, so maybe_execute_in_parallel
            # falls back to sequential execution. All ops are modeled sequentially.
            self.context_ops.extend(
                [
                    ops.Embedding("context_embedding", 1, self._vocab_size, h, 0.3),
                    ops.ElementWise("context_add_norm_1", self._num_layers, 2 * h, 2 * h, 0.8),
                    # kv_a_proj_with_mqa: hidden_size -> q_lora + kv_lora + qk_rope.
                    ops.GEMM(
                        "context_downscale_gemm",
                        self._num_layers,
                        downscale_width,
                        h,
                        attn_downscale_gemm_quant_mode,
                    ),
                    # q_a_layernorm: RMSNorm on q_compressed.
                    ops.ElementWise("context_q_a_layernorm", self._num_layers, q_lora_rank, q_lora_rank, 0.8),
                    ops.GEMM(
                        "context_q_b_proj_gemm",
                        self._num_layers,
                        q_projection_width // tp_size,
                        q_lora_rank,
                        attn_q_gemm_quant_mode,
                    ),
                    ops.GEMM(
                        "context_kv_b_proj_gemm",
                        self._num_layers,
                        kv_projection_width // tp_size,
                        kv_lora_rank,
                        attn_kv_gemm_quant_mode,
                    ),
                    ops.ContextMLA(
                        "context_attention",
                        self._num_layers,
                        128 // tp_size,
                        kvcache_quant_mode,
                        fmha_quant_mode,
                    ),
                    ops.GEMM(
                        "context_proj_gemm",
                        self._num_layers,
                        h,
                        o_projection_width // tp_size,
                        attn_o_gemm_quant_mode,
                    ),
                    ops.ElementWise("context_add_norm_2", self._num_layers, 2 * h, 2 * h, 0.8),
                ]
            )
            # Shared experts (full size -- WideEP ADP mode, shared_tp_size=1),
            # router, A2A dispatch/ModeledEPMoE/combine and moe_reduce_add.
            self.context_ops.extend(self._large_ep_moe_ops("context", moe_shape, self._num_layers))
            self.context_ops.append(
                ops.GEMM(
                    "context_logits_gemm",
                    1,
                    self._vocab_size // tp_size,
                    h,
                    common.GEMMQuantMode.bfloat16,
                )
            )

            self.generation_ops.extend(
                [
                    ops.Embedding("generation_embedding", 1 * self._mtp_scale_factor, self._vocab_size, h, 0.3),
                    ops.ElementWise("generation_add_norm_1", gen_scale, 2 * h, 2 * h, 0.8),
                    ops.GEMM(
                        "generation_downscale_gemm",
                        gen_scale,
                        downscale_width,
                        h,
                        attn_downscale_gemm_quant_mode,
                    ),
                    # q_a_layernorm: RMSNorm on q_compressed. In TRT-LLM
                    # kv_a_layernorm runs in parallel but is much smaller,
                    # so we model only q_a_layernorm as the dominant one.
                    ops.ElementWise("generation_q_a_layernorm", gen_scale, q_lora_rank, q_lora_rank, 0.8),
                    ops.GEMM(
                        "generation_q_b_proj_gemm",
                        gen_scale,
                        q_projection_width // tp_size,
                        q_lora_rank,
                        attn_q_gemm_quant_mode,
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
                                gen_scale,
                                self._num_heads // tp_size,
                                mla_bmm_quant_mode,
                                if_pre=True,
                            ),
                        ],
                        group_b=[
                            # mla_rope_generation: RoPE on q_pe (64d) + KV cache write (512+64=576d)
                            ops.ElementWise(
                                "generation_rope_kvcache",
                                gen_scale,
                                kv_lora_rank + qk_rope_head_dim,
                                kv_lora_rank + qk_rope_head_dim,
                                0.8,
                            ),
                        ],
                    ),
                    ops.GenerationMLA("generation_attention", gen_scale, 128 // tp_size, kvcache_quant_mode),
                    ops.MLABmm(
                        "generation_bmm_post",
                        gen_scale,
                        self._num_heads // tp_size,
                        mla_bmm_quant_mode,
                        if_pre=False,
                    ),
                    ops.GEMM(
                        "generation_proj_gemm",
                        gen_scale,
                        h,
                        o_projection_width // tp_size,
                        attn_o_gemm_quant_mode,
                    ),
                    ops.ElementWise("generation_add_norm_2", gen_scale, 2 * h, 2 * h, 0.8),
                ]
            )
            # OverlapOp(routed, shared) + moe_reduce_add, from the builder.
            self.generation_ops.extend(self._large_ep_moe_ops("generation", moe_shape, gen_scale))
            self.generation_ops.append(
                ops.GEMM(
                    "generation_logits_gemm",
                    1 * self._mtp_scale_factor,
                    self._vocab_size // tp_size,
                    h,
                    common.GEMMQuantMode.bfloat16,
                )
            )

            # pp
            pp_scale_factor = pp_size - 1
            self.context_ops.append(ops.P2P("context_p2p", pp_scale_factor, h, pp_size))
            self.generation_ops.append(ops.P2P("generation_p2p", pp_scale_factor * self._mtp_scale_factor, h, pp_size))
            return

        if self._is_large_ep and backend_name == "sglang":
            # ===== sglang large-EP (deepep) =====
            # Attention downscale (fused q_a+kv_a) follows the checkpoint's
            # per-projection dtype (attn_downscale_gemm_quant_mode above); the
            # q_b/kv_b/o projections live inside the WideEP MLA module rows,
            # whose query carries no gemm axis, so only the granular downscale
            # GEMMs are switched here (#1423, carried from the deleted
            # WideEPDeepSeekModel).
            # Attention + non-MoE wiring transcribed verbatim from the deleted
            # WideEPDeepSeekModel (deepseek.py:1105-1291 at commit 8372e60):
            # qkv_a_proj outside the MLA forward, TP all_gather/reduce_scatter
            # around attention, and NO add_norm_2 / logits_gemm / P2P (the
            # legacy graph never emitted them).
            attn_backend = self.config.attention_backend
            self.context_ops.extend(
                [
                    # qkv_a projection (fused q_a + kv_a + rope): hidden_size ->
                    # q_lora_rank + kv_lora_rank + qk_rope_head_dim. Replicated on
                    # every GPU (not TP-sharded); computed outside the MLA forward
                    # via the sglang >=0.5.6 communicator, hence its own GEMM op.
                    ops.GEMM(
                        "context_qkv_a_proj_gemm",
                        self._num_layers,
                        1536 + 512 + 64,  # = 2112
                        h,
                        attn_downscale_gemm_quant_mode,
                        scale_num_tokens=tp_size,
                        seq_split=cp,
                    ),
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
                    ops.Embedding("context_embedding", 1, self._vocab_size, h, 0.3, seq_split=cp),
                    ops.ElementWise("context_add_norm_1", self._num_layers, 2 * h, 2 * h, 0.8, seq_split=cp),
                    ops.GEMM(
                        "context_downscale_gemm",
                        self._num_layers,
                        2112,
                        h,
                        attn_downscale_gemm_quant_mode,
                        seq_split=cp,
                    ),  # on every gpu, fused_a
                    ops.WideEPContextMLA(
                        "context_attention",
                        self._num_layers,
                        tp_size,
                        kvcache_quant_mode,
                        fmha_quant_mode,
                        attn_backend,
                        cp_size=cp,
                    ),
                    *self._cp_attn_comm_ops(),
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
            # Shared experts + dispatch/compute/combine (the builder emits the
            # deepep-dialect shared triplet: full size, scale_num_tokens=tp).
            self.context_ops.extend(self._large_ep_moe_ops("context", moe_shape, self._num_layers))

            self.generation_ops.extend(
                [
                    ops.GEMM(
                        "generation_qkv_a_proj_gemm",
                        self._num_layers * self._mtp_scale_factor,
                        1536 + 512 + 64,  # = 2112
                        h,
                        attn_downscale_gemm_quant_mode,
                    ),
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
                        attn_downscale_gemm_quant_mode,
                    ),
                    ops.WideEPGenerationMLA(
                        "generation_attention",
                        self._num_layers * self._mtp_scale_factor,
                        tp_size,
                        kvcache_quant_mode,
                        fmha_quant_mode,
                        attn_backend,
                    ),
                ]
            )
            self.generation_ops.extend(
                self._large_ep_moe_ops("generation", moe_shape, self._num_layers * self._mtp_scale_factor)
            )
            return

        # Mixed checkpoints bypass the module row entirely (no exact
        # identity exists in a single-gemm_type table); uniform checkpoints
        # keep the profiled-module primary with the granular fallback.
        context_mla_granular = [
            ops.GEMM(
                "context_downscale_gemm",
                self._num_layers,
                2112,
                h,
                attn_downscale_gemm_quant_mode,
                seq_split=cp,
            ),
            ops.GEMM(
                "context_q_b_proj_gemm",
                self._num_layers,
                # heads x (qk_nope 128 + qk_rope 64); DSV3's 128 heads gave the old 24576 literal
                self._num_heads * 192 // tp_size,
                1536,
                attn_q_gemm_quant_mode,
                seq_split=cp,
            ),
            ops.GEMM(
                "context_kv_b_proj_gemm",
                self._num_layers,
                # heads x (qk_nope 128 + v_head_dim 128)
                self._num_heads * 256 // tp_size,
                512,
                attn_kv_gemm_quant_mode,
                seq_split=cp,
            ),
            ops.ContextAttention(
                "context_attention",
                self._num_layers / attn_count_div,
                self._num_heads // tp_size,
                self._num_kv_heads // tp_size,
                kvcache_quant_mode,
                fmha_quant_mode,
                head_size=self._vllm_head_size,
            )
            if self._backend_name == "vllm"
            else ops.ContextMLA(
                "context_attention",
                self._num_layers,
                _MLA_KERNEL_TABLE_NATIVE_HEADS // tp_size,
                kvcache_quant_mode,
                fmha_quant_mode,
                cp_size=cp,
            ),
            ops.GEMM(
                "context_proj_gemm",
                self._num_layers,
                h,
                # o_proj input: heads x v_head_dim 128
                self._num_heads * 128 // tp_size,
                attn_o_gemm_quant_mode,
                seq_split=cp,
            ),
        ]
        if attn_module_identity_exact:
            context_mla_block_ops = [
                ops.FallbackOp(
                    "context_mla_block",
                    primary=ops.MLAModule(
                        "context_mla_module",
                        self._num_layers / attn_count_div,
                        True,
                        # Model head count, not the DSV3 literal: Kimi K2.5 has 64
                        # heads, so tp1 must hit the heads=64 rows (present in the
                        # module tables as the DSV3 tp2 shard).
                        self._num_heads // tp_size,
                        kvcache_quant_mode,
                        fmha_quant_mode,
                        attn_gemm_quant_mode,
                        # [native][local] module-table identity (#1458);
                        # resolves nearest (-> DSV3 128) until per-model data lands.
                        native_num_heads=self._num_heads,
                    ),
                    fallback=context_mla_granular,
                )
            ]
        else:
            context_mla_block_ops = context_mla_granular

        self.context_ops.extend(
            [
                ops.Embedding("context_embedding", 1, self._vocab_size, h, 0.3, seq_split=cp),
                ops.ElementWise("context_add_norm_1", self._num_layers, 2 * h, 2 * h, 0.8, seq_split=cp),
                *context_mla_block_ops,
                *self._cp_attn_comm_ops(),
                ops.ElementWise("context_add_norm_2", self._num_layers, 2 * h, 2 * h, 0.8, seq_split=cp),
            ]
        )

        if self._is_large_ep:
            # Large EP on a framework without its own attention stack (see the
            # generation site below).
            self.context_ops.extend(self._large_ep_moe_ops("context", moe_shape, self._num_layers))
        else:
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
                        shared_gemm_quant_mode,
                        seq_split=cp,
                    ),
                    ops.ElementWise(
                        "context_shared_act_gate",
                        self._num_layers,
                        2 * self._moe_inter_size // tp_size,
                        self._moe_inter_size // tp_size,
                        0.8,
                        seq_split=cp,
                    ),
                    ops.GEMM(
                        "context_shared_ffn2_gemm",
                        self._num_layers,
                        h,
                        self._moe_inter_size // tp_size,
                        shared_gemm_quant_mode,
                        seq_split=cp,
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
                        common.GEMMQuantMode.bfloat16,
                        seq_split=cp,
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
                        attn_cp_size=cp,
                        backend=self._backend_name,
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
                        attn_cp_size=cp,
                        backend=self._backend_name,
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
                    common.GEMMQuantMode.bfloat16,
                    seq_split=cp,
                )
            ]
        )

        # vLLM TP allreduce, prefill/mixed-step side. Same per-layer pattern as
        # the generation_ops counterpart below; context_ops is not MTP-scaled.
        # Chunked prefill iterations pay this unfused cost since
        # AllReduceFusionPass only fires in pure decode CUDA-graph steps.
        if self._backend_name == "vllm":
            self.context_ops.append(
                ops.CustomAllReduce(
                    "context_tp_allreduce",
                    2 * self._num_layers,
                    h,
                    tp_size,
                )
            )
        #####generation part, only generation part is scaled by mtp_scale_factor
        # Same mixed-identity gate as the context block above.
        generation_mla_granular = [
            ops.GEMM(
                "generation_downscale_gemm",
                self._num_layers * self._mtp_scale_factor,
                2112,
                h,
                attn_downscale_gemm_quant_mode,
            ),
            ops.GEMM(
                "generation_q_b_proj_gemm",
                self._num_layers * self._mtp_scale_factor,
                self._num_heads * 192 // tp_size,
                1536,
                attn_q_gemm_quant_mode,
            ),
            *(
                # KIMI K2.5 on vLLM: same reasoning as ContextAttention above —
                # vLLM absorbs the KV projection and runs standard GenerationAttention
                # with v_head_dim=128. TRT-LLM and SGLang use the full MLA path
                # (MLABmm + GenerationMLA + MLABmm).
                [
                    ops.GenerationAttention(
                        "generation_attention",
                        self._num_layers * self._mtp_scale_factor,
                        self._num_heads // tp_size,
                        self._num_kv_heads // tp_size,
                        kvcache_quant_mode,
                        head_size=self._vllm_head_size,
                    )
                ]
                if self._backend_name == "vllm"
                else [
                    ops.MLABmm(
                        "generation_bmm_pre",
                        self._num_layers * self._mtp_scale_factor,
                        self._num_heads // tp_size,
                        mla_bmm_quant_mode,
                        if_pre=True,
                    ),
                    ops.GenerationMLA(
                        "generation_attention",
                        self._num_layers * self._mtp_scale_factor,
                        _MLA_KERNEL_TABLE_NATIVE_HEADS // tp_size,
                        kvcache_quant_mode,
                    ),
                    ops.MLABmm(
                        "generation_bmm_post",
                        self._num_layers * self._mtp_scale_factor,
                        self._num_heads // tp_size,
                        mla_bmm_quant_mode,
                        if_pre=False,
                    ),
                ]
            ),
            ops.GEMM(
                "generation_proj_gemm",
                self._num_layers * self._mtp_scale_factor,
                h,
                # o_proj input is heads x v_head_dim 128 (the old h//tp
                # literal was wrong even for DSV3: 7168 vs 16384)
                self._num_heads * 128 // tp_size,
                attn_o_gemm_quant_mode,
            ),
        ]
        if attn_module_identity_exact:
            generation_mla_block_ops = [
                ops.FallbackOp(
                    "generation_mla_block",
                    primary=ops.MLAModule(
                        "generation_mla_module",
                        self._num_layers * self._mtp_scale_factor,
                        False,
                        self._num_heads // tp_size,
                        kvcache_quant_mode,
                        fmha_quant_mode,
                        attn_gemm_quant_mode,
                        native_num_heads=self._num_heads,
                    ),
                    fallback=generation_mla_granular,
                )
            ]
        else:
            generation_mla_block_ops = generation_mla_granular

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
                *generation_mla_block_ops,
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

        if self._is_large_ep:
            # Large EP on a framework without its own attention stack (the
            # sglang/trtllm branches returned above): regular attention wiring,
            # builder MoE block.
            self.generation_ops.extend(
                self._large_ep_moe_ops("generation", moe_shape, self._num_layers * self._mtp_scale_factor)
            )
        else:
            # group_b: shared expert path (aux CUDA stream)
            gen_shared_ops = [
                ops.GEMM(
                    "generation_shared_gate_up_gemm",
                    self._num_layers * self._mtp_scale_factor,
                    2 * self._moe_inter_size // tp_size,
                    h,
                    shared_gemm_quant_mode,
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
                    shared_gemm_quant_mode,
                ),
            ]

            # group_a: routed expert path (main CUDA stream)
            gen_routed_ops = [
                ops.GEMM(
                    "generation_router_gemm",
                    self._num_layers * self._mtp_scale_factor,
                    self._num_experts,
                    h,
                    common.GEMMQuantMode.bfloat16,
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
                    attn_cp_size=cp,
                    is_context=False,
                    backend=self._backend_name,
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
                    attn_cp_size=cp,
                    is_context=False,
                    backend=self._backend_name,
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
                    common.GEMMQuantMode.bfloat16,
                )
            ]
        )

        # vLLM TP allreduce: one collective after attention proj, one after MoE,
        # per transformer layer. vLLM models with tp_size > 1 always pay this cost
        # (cross_device_reduce); the FlashInfer fused variant only kicks in during
        # pure decode steps with AllReduceFusionPass. Modeled here as the unfused
        # cost since collect_all_reduce.py benchmarks vLLM's native allreduce.
        # TRT-LLM (narrow EP) and SGLang paths model their allreduce/all-gather
        # cost elsewhere (WideEP variants below; SGLang via NCCL ops).
        if self._backend_name == "vllm":
            self.generation_ops.append(
                ops.CustomAllReduce(
                    "generation_tp_allreduce",
                    2 * self._num_layers * self._mtp_scale_factor,
                    h,
                    tp_size,
                )
            )

        # pp
        pp_scale_factor = pp_size - 1
        self.context_ops.append(ops.P2P("context_p2p", pp_scale_factor, h, pp_size))
        self.generation_ops.append(ops.P2P("generation_p2p", pp_scale_factor * self._mtp_scale_factor, h, pp_size))
