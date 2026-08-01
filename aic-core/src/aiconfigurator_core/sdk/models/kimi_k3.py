# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import aiconfigurator_core.sdk.operations as ops
from aiconfigurator_core.sdk import common
from aiconfigurator_core.sdk.models.base import BaseModel, register_model


@register_model("KIMIK3")
class KimiK3Model(BaseModel):
    """
    Kimi-K3 hybrid KDA + MLA LatentMoE model (2.8T, 93 layers = 69 KDA + 24 MLA).

    Layer types from KimiK3Config.layer_types:
      - "linear_attention": KDA (Kimi Delta Attention) layers — fused qkvg
        projection (full-rank gate), forget-gate GEMV chain, short conv, and
        the KDA delta-rule kernels (chunk_kda / fused_recurrent_kda_packed_decode).
      - "full_attention": DeepSeek-geometry MLA (96 heads, NoPE) plus an
        output-gate GEMM (mla_use_output_gate).

    FFN: layer 0 is a dense SwiGLU MLP (first_k_dense_replace=1); all other
    layers run LatentMoE — router in hidden space, a replicated hidden->latent
    down projection, 896 routed experts (top-16, SiTU) entirely in the 3584
    latent space, a replicated latent->hidden up projection, plus 2 shared
    experts in the full hidden space. AttnRes adds elementwise-only cost.

    Speculative decoding (DSPARK): ``nextn`` is the dspark block size (draft
    tokens proposed per step). The backend scales the generation batch by
    (nextn + 1) to model the verify width; this class additionally adds the
    5-layer dense draft-model ops and switches KDA generation kernels to the
    target-verify tables. The classic MTP scale factor does NOT apply (the
    draft is not made of target-shaped layers).
    """

    # DSPARK draft model (RadixArk/Kimi-K3-DSpark) fixed geometry.
    DRAFT_NUM_LAYERS = 5
    DRAFT_NUM_HEADS = 64
    DRAFT_NUM_KV_HEADS = 16
    DRAFT_HEAD_DIM = 64
    DRAFT_INTER_SIZE = 14336

    # KDA state slots per request (sglang mamba radix cache, extra_buffer
    # strategy). Governs the per-request constant state-pool cost.
    #
    # DELIBERATELY a flat proxy (owner decision 2026-07-31: annotate, don't
    # model the split). Serving actually charges this in two parts — 1-2
    # ACTIVE slots per running request (live state + snapshot buffer) plus a
    # GLOBAL radix checkpoint cache pool with its own budget/LRU (optionally
    # INT8-compressed, 4x denser), which does not scale with concurrency
    # (see the Day-0 cache write-ups referenced in
    # docs/perf_database/linear-attention-module-design.md). The flat 5x is
    # acceptable because the KDA term is secondary for K3's target
    # workloads: at TP8 one slot is ~54 MB/rank (5x = 270 MB/request) vs
    # ~1.8 GB/request of MLA KV at 128k context — it only dominates for
    # short-context high-concurrency mixes, where it over-charges.
    KDA_STATE_SLOTS_PER_REQUEST = 5

    @classmethod
    def create(cls, model_info: dict, model_config, backend_name: str) -> BaseModel:
        return cls(
            backend_name,
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
            model_info["extra_params"],
        )

    def __init__(self, backend_name: str, *args) -> None:
        super().__init__(*args)
        self._backend_name = backend_name
        cfg: common.KimiK3Config = self.extra_params
        assert isinstance(cfg, common.KimiK3Config), "KimiK3Model requires KimiK3Config extra_params"

        assert (
            self.config.tp_size * self.config.attention_dp_size * self.config.cp_size
            == self.config.moe_tp_size * self.config.moe_ep_size
        ), (
            f"tp_size ({self.config.tp_size}) * attention_dp_size "
            f"({self.config.attention_dp_size}) * cp_size ({self.config.cp_size}) should equal moe_tp_size "
            f"({self.config.moe_tp_size}) * moe_ep_size ({self.config.moe_ep_size})"
        )
        assert cfg.num_experts >= self.config.moe_ep_size

        # Context parallelism over KDA linear attention is a sequential-state
        # problem the serving frameworks themselves do not support for Kimi-K3
        # (and DCP is not modeled either) — fail loudly instead of silently
        # mispricing.
        if self.config.cp_size > 1:
            raise NotImplementedError(
                "Kimi-K3 modeling does not support context parallelism (cp_size > 1): "
                "KDA recurrent state is sequential and the serving frameworks do not "
                "shard it; DCP (decode context parallel) is likewise not modeled."
            )

        self._build_context_ops()
        self._build_generation_ops()

    # ------------------------------------------------------------------
    # Layer bookkeeping
    # ------------------------------------------------------------------

    def _count_layer_types(self) -> dict[str, int]:
        cfg: common.KimiK3Config = self.extra_params
        return {
            "linear": cfg.layer_types.count("linear_attention"),
            "full": cfg.layer_types.count("full_attention"),
        }

    def _bmm_heads_pow2(self, tp: int) -> int:
        heads = self._num_heads // tp
        p = 1
        while p < heads:
            p *= 2
        return p

    def _mla_dims(self, cfg: common.KimiK3Config) -> dict[str, int]:
        n = self._num_heads
        return {
            "fused_qkv_a_out": cfg.q_lora_rank + cfg.kv_lora_rank + cfg.qk_rope_head_dim,  # 2112
            "q_b_out": n * (cfg.qk_nope_head_dim + cfg.qk_rope_head_dim),  # 96*192
            "kv_b_out": n * (cfg.qk_nope_head_dim + cfg.v_head_dim),  # 96*256
            "o_in": n * cfg.v_head_dim,  # 96*128
        }

    # ------------------------------------------------------------------
    # Context phase
    # ------------------------------------------------------------------

    def _build_context_ops(self) -> None:
        cfg: common.KimiK3Config = self.extra_params
        h = self._hidden_size
        tp = self.config.tp_size
        pp = self.config.pp_size
        gemm_q = self.config.gemm_quant_mode
        kvcache_q = self.config.kvcache_quant_mode
        fmha_q = self.config.fmha_quant_mode
        counts = self._count_layer_types()
        mla = self._mla_dims(cfg)

        # KDA per-shard dims for kernel lookup (collector rows are per attention-TP shard)
        nk_local = cfg.kda_num_heads // tp
        hk = cfg.kda_head_dim
        p_local = nk_local * hk  # per-rank projection width

        self.context_ops = [
            ops.Embedding("context_embedding", 1, self._vocab_size // tp, h, 0.3),
            ops.CustomAllReduce("context_embedding_ar", 1, h, tp),
        ]

        # --- KDA linear-attention layers ---
        if counts["linear"] > 0:
            c = counts["linear"]
            self.context_ops.extend(
                [
                    ops.ElementWise("context_kda_norm", c, 2 * h, 2 * h, 0.8),
                    # fused q/k/v/gate projection: 4 * P // tp
                    ops.GEMM("context_kda_qkvg_gemm", c, 4 * p_local, h, gemm_q),
                    # merged [f_a | beta] skinny GEMM + f_b up-projection
                    ops.GEMM("context_kda_bfa_gemm", c, 232, h, gemm_q),
                    ops.GEMM("context_kda_fb_gemm", c, p_local, 128, gemm_q),
                    ops.KDAKernel(
                        "context_kda_conv1d",
                        c,
                        "causal_conv1d_fn_qkv3",
                        "context",
                        h,
                        nk_local,
                        hk,
                        nk_local,
                        hk,
                        cfg.kda_conv_kernel,
                    ),
                    ops.KDAKernel(
                        "context_kda_scan",
                        c,
                        # vLLM's SM90+ prefill default is the FlashKDA CUDA
                        # kernel; sglang runs the Triton chunk kernel.
                        "flashkda_fwd" if self._backend_name == "vllm" else "chunk_kda",
                        "context",
                        h,
                        nk_local,
                        hk,
                        nk_local,
                        hk,
                        cfg.kda_conv_kernel,
                    ),
                    # gated RMSNorm on the attention output
                    ops.ElementWise("context_kda_onorm", c, 2 * p_local, p_local, 0.8),
                    ops.GEMM("context_kda_o_gemm", c, h, p_local, gemm_q, low_precision_input=True),
                    ops.CustomAllReduce("context_kda_ar", c, h, tp),
                ]
            )

        # --- MLA full-attention layers ---
        if counts["full"] > 0:
            c = counts["full"]
            self.context_ops.extend(
                [
                    ops.ElementWise("context_mla_norm", c, 2 * h, 2 * h, 0.8),
                    ops.GEMM("context_mla_downscale_gemm", c, mla["fused_qkv_a_out"], h, gemm_q),
                    ops.GEMM("context_mla_q_b_gemm", c, mla["q_b_out"] // tp, cfg.q_lora_rank, gemm_q),
                    ops.GEMM("context_mla_kv_b_gemm", c, mla["kv_b_out"] // tp, cfg.kv_lora_rank, gemm_q),
                    # vLLM models MLA as standard attention with v_head_dim
                    # (K2.5/DeepSeek vLLM convention); sglang uses the granular
                    # MLA tables.
                    ops.ContextAttention(
                        "context_attention",
                        c,
                        self._num_heads // tp,
                        self._num_kv_heads // tp,
                        kvcache_q,
                        fmha_q,
                        head_size=cfg.v_head_dim,
                    )
                    if self._backend_name == "vllm"
                    else ops.ContextMLA(
                        "context_attention",
                        c,
                        self._num_heads // tp,
                        kvcache_q,
                        fmha_q,
                    ),
                    # output gate: extra hidden -> n*v_head_dim GEMM + sigmoid multiply
                    ops.GEMM("context_mla_gate_gemm", c, mla["o_in"] // tp, h, gemm_q),
                    ops.ElementWise("context_mla_gate_mul", c, 2 * mla["o_in"] // tp, mla["o_in"] // tp, 0.8),
                    ops.GEMM("context_mla_o_gemm", c, h, mla["o_in"] // tp, gemm_q, low_precision_input=True),
                    ops.CustomAllReduce("context_mla_ar", c, h, tp),
                ]
            )

        # --- AttnRes (elementwise only): two aggregation points per layer ---
        if cfg.attn_res_block_size > 0:
            self.context_ops.append(ops.ElementWise("context_attn_res", 2 * self._num_layers, 4 * h, 2 * h, 0.8))

        # --- FFN: dense layer(s) + LatentMoE layers ---
        self.context_ops.extend(self._ffn_ops("context", context=True))

        self.context_ops.extend(
            [
                ops.GEMM("context_logits_gemm", 1, self._vocab_size // tp, h, common.GEMMQuantMode.bfloat16),
                ops.P2P("context_p2p", pp - 1, h, pp),
            ]
        )

    # ------------------------------------------------------------------
    # Generation phase
    # ------------------------------------------------------------------

    def _build_generation_ops(self) -> None:
        cfg: common.KimiK3Config = self.extra_params
        h = self._hidden_size
        tp = self.config.tp_size
        pp = self.config.pp_size
        gemm_q = self.config.gemm_quant_mode
        kvcache_q = self.config.kvcache_quant_mode
        counts = self._count_layer_types()
        mla = self._mla_dims(cfg)

        mla_bmm_q = (
            common.GEMMQuantMode.fp8 if gemm_q != common.GEMMQuantMode.bfloat16 else common.GEMMQuantMode.bfloat16
        )

        nk_local = cfg.kda_num_heads // tp
        hk = cfg.kda_head_dim
        p_local = nk_local * hk

        spec = self._nextn > 0
        is_vllm = self._backend_name == "vllm"
        verify_kernel = "fused_recurrent_kda" if is_vllm else "fused_sigmoid_gating_delta_rule_update"
        draft_tokens = self._nextn + 1 if spec else 0
        # Backend queries generation ops at batch * (nextn + 1); ops that run
        # once per step per request (not per verify token) scale down by this.
        per_step = 1.0 / draft_tokens if spec else 1.0

        self.generation_ops = [
            ops.Embedding("generation_embedding", 1, self._vocab_size // tp, h, 0.3),
            ops.CustomAllReduce("generation_embedding_ar", 1, h, tp),
        ]

        # --- KDA linear-attention layers ---
        if counts["linear"] > 0:
            c = counts["linear"]
            kda_kernels = (
                [
                    ops.KDAKernel(
                        "generation_kda_conv1d",
                        c,
                        "causal_conv1d_update",
                        "verify",
                        h,
                        nk_local,
                        hk,
                        nk_local,
                        hk,
                        cfg.kda_conv_kernel,
                        draft_tokens=draft_tokens,
                    ),
                    ops.KDAKernel(
                        "generation_kda_verify",
                        c,
                        verify_kernel,
                        "verify",
                        h,
                        nk_local,
                        hk,
                        nk_local,
                        hk,
                        cfg.kda_conv_kernel,
                        draft_tokens=draft_tokens,
                    ),
                ]
                if spec
                else (
                    [
                        # vLLM NOSPEC decode: one fused CUDA kernel covering the
                        # conv update, recurrence and gated RMSNorm.
                        ops.KDAKernel(
                            "generation_kda_fused_decode",
                            c,
                            "fused_kda_decode",
                            "generation",
                            h,
                            nk_local,
                            hk,
                            nk_local,
                            hk,
                            cfg.kda_conv_kernel,
                        ),
                    ]
                    if is_vllm
                    else [
                        ops.KDAKernel(
                            "generation_kda_conv1d",
                            c,
                            "causal_conv1d_update",
                            "generation",
                            h,
                            nk_local,
                            hk,
                            nk_local,
                            hk,
                            cfg.kda_conv_kernel,
                        ),
                        ops.KDAKernel(
                            "generation_kda_recurrent",
                            c,
                            "fused_recurrent_kda_packed_decode",
                            "generation",
                            h,
                            nk_local,
                            hk,
                            nk_local,
                            hk,
                            cfg.kda_conv_kernel,
                        ),
                    ]
                )
            )
            self.generation_ops.extend(
                [
                    ops.ElementWise("generation_kda_norm", c, 2 * h, 2 * h, 0.8),
                    ops.GEMM("generation_kda_qkvg_gemm", c, 4 * p_local, h, gemm_q),
                    ops.GEMM("generation_kda_bfa_gemm", c, 232, h, gemm_q),
                    ops.GEMM("generation_kda_fb_gemm", c, p_local, 128, gemm_q),
                    *kda_kernels,
                    # on vLLM NOSPEC the gated RMSNorm is fused into
                    # fused_kda_decode; keep it for all other paths.
                    *(
                        []
                        if (is_vllm and not spec)
                        else [ops.ElementWise("generation_kda_onorm", c, 2 * p_local, p_local, 0.8)]
                    ),
                    ops.GEMM("generation_kda_o_gemm", c, h, p_local, gemm_q, low_precision_input=True),
                    ops.CustomAllReduce("generation_kda_ar", c, h, tp),
                ]
            )

        # --- MLA full-attention layers ---
        if counts["full"] > 0:
            c = counts["full"]
            self.generation_ops.extend(
                [
                    ops.ElementWise("generation_mla_norm", c, 2 * h, 2 * h, 0.8),
                    ops.GEMM("generation_mla_downscale_gemm", c, mla["fused_qkv_a_out"], h, gemm_q),
                    ops.GEMM("generation_mla_q_b_gemm", c, mla["q_b_out"] // tp, cfg.q_lora_rank, gemm_q),
                    # The mla_bmm table only carries power-of-two head slices
                    # (DeepSeek geometry); K3's 96 heads shard to 12/24/48/96
                    # local heads. The bmm is per-head batched, so query the
                    # next power-of-two slice and scale the count by the head
                    # ratio.
                    *(
                        [
                            ops.GenerationAttention(
                                "generation_attention",
                                c,
                                self._num_heads // tp,
                                self._num_kv_heads // tp,
                                kvcache_q,
                                head_size=cfg.v_head_dim,
                            )
                        ]
                        if self._backend_name == "vllm"
                        else [
                            ops.MLABmm(
                                "generation_bmm_pre",
                                c * (self._num_heads // tp) / self._bmm_heads_pow2(tp),
                                self._bmm_heads_pow2(tp),
                                mla_bmm_q,
                                if_pre=True,
                            ),
                            ops.GenerationMLA(
                                "generation_attention",
                                c,
                                self._num_heads // tp,
                                kvcache_q,
                            ),
                            ops.MLABmm(
                                "generation_bmm_post",
                                c * (self._num_heads // tp) / self._bmm_heads_pow2(tp),
                                self._bmm_heads_pow2(tp),
                                mla_bmm_q,
                                if_pre=False,
                            ),
                        ]
                    ),
                    ops.GEMM("generation_mla_gate_gemm", c, mla["o_in"] // tp, h, gemm_q),
                    ops.ElementWise("generation_mla_gate_mul", c, 2 * mla["o_in"] // tp, mla["o_in"] // tp, 0.8),
                    ops.GEMM("generation_mla_o_gemm", c, h, mla["o_in"] // tp, gemm_q, low_precision_input=True),
                    ops.CustomAllReduce("generation_mla_ar", c, h, tp),
                ]
            )

        if cfg.attn_res_block_size > 0:
            self.generation_ops.append(ops.ElementWise("generation_attn_res", 2 * self._num_layers, 4 * h, 2 * h, 0.8))

        self.generation_ops.extend(self._ffn_ops("generation", context=False))

        self.generation_ops.extend(
            [
                ops.GEMM(
                    "generation_logits_gemm",
                    1,
                    self._vocab_size // tp,
                    h,
                    common.GEMMQuantMode.bfloat16,
                ),
                ops.P2P("generation_p2p", pp - 1, h, pp),
            ]
        )

        # --- DSPARK draft model: 5 dense GQA layers at hidden 7168 ---
        if spec:
            n_q = self.DRAFT_NUM_HEADS
            n_kv = self.DRAFT_NUM_KV_HEADS
            hd = self.DRAFT_HEAD_DIM
            inter = self.DRAFT_INTER_SIZE
            dc = self.DRAFT_NUM_LAYERS
            # Draft forwards gamma = nextn tokens per step while the op batch is
            # scaled by (nextn + 1): scale counts by nextn / (nextn + 1).
            dsf = self._nextn / draft_tokens
            qkv_out = (n_q * hd + 2 * n_kv * hd) // tp
            self.generation_ops.extend(
                [
                    ops.Embedding("draft_embedding", 1 * dsf, self._vocab_size // tp, h, 0.3),
                    ops.ElementWise("draft_norm", 2 * dc * dsf, 2 * h, 2 * h, 0.8),
                    ops.GEMM("draft_qkv_gemm", dc * dsf, qkv_out, h, gemm_q),
                    ops.GenerationAttention(
                        "draft_attention",
                        dc * dsf,
                        n_q // tp,
                        max(1, n_kv // tp),
                        kvcache_q,
                        head_size=hd,
                    ),
                    ops.GEMM("draft_proj_gemm", dc * dsf, h, n_q * hd // tp, gemm_q, low_precision_input=True),
                    ops.GEMM("draft_gate_up_gemm", dc * dsf, 2 * inter // tp, h, gemm_q),
                    ops.ElementWise("draft_act_gate", dc * dsf, 2 * inter // tp, inter // tp, 0.8),
                    ops.GEMM("draft_down_gemm", dc * dsf, h, inter // tp, gemm_q, low_precision_input=True),
                    ops.CustomAllReduce("draft_ar", 2 * dc * dsf, h, tp),
                    # target-hidden -> draft context-KV projection (per committed token)
                    ops.GEMM("draft_kv_proj_gemm", dc * per_step, 2 * n_kv * hd // tp, h, gemm_q),
                    # draft base logits once per step
                    ops.GEMM(
                        "draft_logits_gemm",
                        1 * per_step,
                        self._vocab_size // tp,
                        h,
                        common.GEMMQuantMode.bfloat16,
                    ),
                ]
            )

    # ------------------------------------------------------------------
    # FFN (dense layer 0 + LatentMoE layers 1..L-1)
    # ------------------------------------------------------------------

    def _ffn_ops(self, prefix: str, context: bool) -> list:
        cfg: common.KimiK3Config = self.extra_params
        h = self._hidden_size
        tp = self.config.tp_size
        moe_tp = self.config.moe_tp_size
        moe_ep = self.config.moe_ep_size
        attn_dp = self.config.attention_dp_size
        gemm_q = self.config.gemm_quant_mode
        moe_q = self.config.moe_quant_mode
        workload_dist = (
            self.config.workload_distribution + "_1.01"
            if self.config.workload_distribution == "power_law"
            else self.config.workload_distribution
        )

        latent = cfg.routed_expert_hidden_size or h
        num_dense = cfg.first_k_dense_replace
        num_moe = self._num_layers - num_dense
        shared_inter = cfg.num_shared_experts * cfg.moe_inter_size * 2  # 2*3072=6144 total width

        ops_list = [ops.ElementWise(f"{prefix}_ffn_norm", self._num_layers, 2 * h, 2 * h, 0.8)]

        # Dense SwiGLU layer(s)
        if num_dense > 0:
            ops_list.extend(
                [
                    ops.GEMM(f"{prefix}_dense_gate_up_gemm", num_dense, 2 * cfg.dense_inter_size // tp, h, gemm_q),
                    ops.ElementWise(
                        f"{prefix}_dense_act_gate",
                        num_dense,
                        2 * cfg.dense_inter_size // tp,
                        cfg.dense_inter_size // tp,
                        0.8,
                    ),
                    ops.GEMM(
                        f"{prefix}_dense_down_gemm",
                        num_dense,
                        h,
                        cfg.dense_inter_size // tp,
                        gemm_q,
                        low_precision_input=True,
                    ),
                    ops.CustomAllReduce(f"{prefix}_dense_ffn_ar", num_dense, h, tp),
                ]
            )

        # LatentMoE layers
        if num_moe > 0 and cfg.num_experts > 0:
            ops_list.extend(
                [
                    # router in hidden space
                    ops.GEMM(f"{prefix}_router_gemm", num_moe, cfg.num_experts, h, common.GEMMQuantMode.bfloat16),
                    # replicated hidden -> latent down projection (bf16, unsharded)
                    ops.GEMM(f"{prefix}_latent_down_gemm", num_moe, latent, h, common.GEMMQuantMode.bfloat16),
                    ops.ElementWise(f"{prefix}_latent_norm", num_moe, 2 * latent, 2 * latent, 0.8),
                    ops.MoEDispatch(
                        f"{prefix}_moe_pre_dispatch",
                        num_moe,
                        latent,
                        cfg.topk,
                        cfg.num_experts,
                        moe_tp,
                        moe_ep,
                        attn_dp,
                        True,
                        quant_mode=moe_q,
                    ),
                    # routed experts entirely in latent space (3584 / 3072)
                    ops.MoE(
                        f"{prefix}_moe",
                        num_moe,
                        latent,
                        cfg.moe_inter_size,
                        cfg.topk,
                        cfg.num_experts,
                        moe_tp,
                        moe_ep,
                        moe_q,
                        workload_dist,
                        attn_dp,
                    ),
                    ops.MoEDispatch(
                        f"{prefix}_moe_post_dispatch",
                        num_moe,
                        latent,
                        cfg.topk,
                        cfg.num_experts,
                        moe_tp,
                        moe_ep,
                        attn_dp,
                        False,
                        quant_mode=moe_q,
                    ),
                    # replicated latent -> hidden up projection (bf16, unsharded)
                    ops.GEMM(f"{prefix}_latent_up_gemm", num_moe, h, latent, common.GEMMQuantMode.bfloat16),
                ]
            )
            # shared experts in full hidden space (bf16, TP sharded)
            if shared_inter > 0:
                ops_list.extend(
                    [
                        ops.GEMM(
                            f"{prefix}_shared_gate_up_gemm",
                            num_moe,
                            2 * shared_inter // tp,
                            h,
                            common.GEMMQuantMode.bfloat16,
                        ),
                        ops.ElementWise(
                            f"{prefix}_shared_act_gate",
                            num_moe,
                            2 * shared_inter // tp,
                            shared_inter // tp,
                            0.8,
                        ),
                        ops.GEMM(
                            f"{prefix}_shared_down_gemm",
                            num_moe,
                            h,
                            shared_inter // tp,
                            common.GEMMQuantMode.bfloat16,
                        ),
                    ]
                )
        return ops_list

    # ------------------------------------------------------------------
    # Memory: dual-pool (paged MLA KV + constant KDA state per request)
    # ------------------------------------------------------------------

    def get_kvcache_elements_per_token(self) -> int:
        """Per-token paged KV: only MLA layers hold token-linear latent KV
        (kv_lora_rank + qk_rope_head_dim, TP-replicated). With DSPARK the
        draft model's GQA KV (5 layers, TP-sharded) adds per-token elements."""
        cfg: common.KimiK3Config = self.extra_params
        n_mla = cfg.layer_types.count("full_attention")
        elements = n_mla * (cfg.kv_lora_rank + cfg.qk_rope_head_dim)
        if self._nextn > 0:
            tp = self.config.tp_size
            draft_kv_heads = max(1, self.DRAFT_NUM_KV_HEADS // tp)
            elements += self.DRAFT_NUM_LAYERS * 2 * draft_kv_heads * self.DRAFT_HEAD_DIM
        return elements

    def _kda_state_bytes_per_request(self) -> float:
        """Constant KDA state-pool bytes per admitted request on one GPU:
        (SSM fp32 state + conv bf16 window) per KDA layer, TP-sharded, times
        the radix-cache slot multiplier."""
        cfg: common.KimiK3Config = self.extra_params
        tp = self.config.tp_size
        n_kda = cfg.layer_types.count("linear_attention")
        heads_local = max(1, cfg.kda_num_heads // tp)
        ssm_bytes = heads_local * cfg.kda_head_dim * cfg.kda_head_dim * 4
        conv_bytes = (cfg.kda_conv_kernel - 1) * 3 * heads_local * cfg.kda_head_dim * 2
        return n_kda * (ssm_bytes + conv_bytes) * self.KDA_STATE_SLOTS_PER_REQUEST

    def get_kvcache_bytes_per_sequence(self, seq_len: int) -> float:
        """KDA state is charged INSIDE the kvcache budget: MLA KV and KDA
        state draw from one elastic byte pool here. That matches sglang's
        opt-in --enable-unified-memory mode; the DEFAULT is two separately
        sized pools (MLA KV vs KDA state) with a boot-time ratio, where a
        mismatched workload saturates one pool while the other idles — so
        this estimate is optimistic for default-mode deployments (owner
        decision 2026-07-31: annotate, don't model the split)."""
        seq_len = max(0, seq_len)
        token_bytes = seq_len * self.config.kvcache_quant_mode.value.memory * self.get_kvcache_elements_per_token()
        return token_bytes + self._kda_state_bytes_per_request()

    def get_kvcache_max_tokens(self, kv_budget_bytes: float) -> int:
        # Same single-elastic-budget assumption as
        # get_kvcache_bytes_per_sequence — see the note there.
        per_token = self.config.kvcache_quant_mode.value.memory * self.get_kvcache_elements_per_token()
        budget = kv_budget_bytes - self._kda_state_bytes_per_request()
        if budget <= 0 or per_token <= 0:
            return 0
        return int(budget // per_token)
