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

    # DSPARK draft model fixed geometry — sglang lane
    # (RadixArk/Kimi-K3-DSpark: 5-layer dense GQA, model_type qwen3).
    DRAFT_NUM_LAYERS = 5
    DRAFT_NUM_HEADS = 64
    DRAFT_NUM_KV_HEADS = 16
    DRAFT_HEAD_DIM = 64
    DRAFT_INTER_SIZE = 14336
    # DSPARK draft model fixed geometry — vllm lane
    # (Inferact/Kimi-K3-DSpark: 5-layer dense, MLA-style attention with
    # latent projections; output gate disabled; same layer count and FFN
    # width as the RadixArk draft, so DRAFT_NUM_LAYERS/DRAFT_INTER_SIZE
    # are shared).
    VLLM_DRAFT_NUM_HEADS = 64
    VLLM_DRAFT_Q_LORA_RANK = 1536
    VLLM_DRAFT_KV_LORA_RANK = 512
    VLLM_DRAFT_QK_NOPE_DIM = 128
    VLLM_DRAFT_QK_ROPE_DIM = 64
    VLLM_DRAFT_V_DIM = 128

    # KDA state slots per request (sglang mamba radix cache, extra_buffer
    # strategy) — a deliberate flat proxy for serving's 1-2 active slots plus
    # a global radix checkpoint pool; the owner decision (2026-07-31:
    # annotate, don't model the split) is documented on
    # get_kvcache_bytes_per_sequence. Magnitude: at TP8 one slot is ~54
    # MB/rank (5x = 270 MB/request) vs ~1.8 GB/request of MLA KV at 128k
    # context — over-charges only short-context high-concurrency mixes.
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

        # KDA heads shard evenly across attention TP; a non-divisor (or tp >
        # heads) would silently truncate nk_local to a shard geometry that has
        # no collector row (rows exist for 96/48/24/12 only) — fail loudly
        # instead of mispricing (review fix).
        if cfg.kda_num_heads % self.config.tp_size != 0:
            raise ValueError(
                f"Kimi-K3 KDA heads ({cfg.kda_num_heads}) are not divisible by "
                f"tp_size ({self.config.tp_size}); per-rank KDA shard would be "
                "truncated and has no collected kernel rows."
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
                    # beta (per v-head) + f_a low-rank down (head_dim); the full-rank
                    # onorm gate is fused into the qkvg GEMM above (fla kda.py
                    # projections; b_proj=num_heads, f_proj[0]=head_dim).
                    ops.GEMM("context_kda_bfa_gemm", c, cfg.kda_num_heads + cfg.kda_head_dim, h, gemm_q),
                    ops.GEMM("context_kda_fb_gemm", c, p_local, cfg.kda_head_dim, gemm_q),
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
            # Context MLA block (vLLM): q_b + kv_b projections, attention core
            # and o_proj. Since vLLM 0.27.0 upstream ships Kimi-K3's own
            # ``MultiHeadLatentAttention`` (vllm/models/kimi_k3/nvidia/mla.py)
            # whose attention core is a REAL MLA lane (``get_attn_backend(
            # use_mla=True)`` + MLAAttentionImpl on latent-dim KV cache,
            # num_kv_heads=1, selected by vllm's MLA back/prefill selectors),
            # price it against the module-level MLA tables (#1458 [native][local]
            # identity). The granular GQA stack below stays as FallbackOp
            # fallback for systems whose module tables still have no rows
            # (e.g. before per-head-count collection lands).
            context_mla_q_b = ops.GEMM("context_mla_q_b_gemm", c, mla["q_b_out"] // tp, cfg.q_lora_rank, gemm_q)
            context_mla_kv_b = ops.GEMM("context_mla_kv_b_gemm", c, mla["kv_b_out"] // tp, cfg.kv_lora_rank, gemm_q)
            context_mla_o = ops.GEMM("context_mla_o_gemm", c, h, mla["o_in"] // tp, gemm_q, low_precision_input=True)
            context_mla_granular = [
                context_mla_q_b,
                context_mla_kv_b,
                ops.ContextAttention(
                    "context_attention",
                    c,
                    self._num_heads // tp,
                    self._num_kv_heads // tp,
                    kvcache_q,
                    fmha_q,
                    head_size=cfg.v_head_dim,
                ),
                context_mla_o,
            ]
            if self._backend_name == "vllm":
                context_mla_block = [
                    ops.FallbackOp(
                        "context_mla_block",
                        primary=ops.MLAModule(
                            "context_mla_module",
                            c,
                            True,
                            self._num_heads // tp,
                            kvcache_q,
                            fmha_q,
                            gemm_q,
                            native_num_heads=self._num_heads,
                        ),
                        fallback=context_mla_granular,
                    )
                ]
            else:
                # sglang keeps the granular ContextMLA tables.
                context_mla_block = [
                    context_mla_q_b,
                    context_mla_kv_b,
                    ops.ContextMLA(
                        "context_attention",
                        c,
                        self._num_heads // tp,
                        kvcache_q,
                        fmha_q,
                    ),
                    context_mla_o,
                ]
            self.context_ops.extend(
                [
                    ops.ElementWise("context_mla_norm", c, 2 * h, 2 * h, 0.8),
                    ops.GEMM("context_mla_downscale_gemm", c, mla["fused_qkv_a_out"], h, gemm_q),
                    *context_mla_block,
                    # output gate: extra hidden -> n*v_head_dim GEMM + sigmoid multiply
                    ops.GEMM("context_mla_gate_gemm", c, mla["o_in"] // tp, h, gemm_q),
                    ops.ElementWise("context_mla_gate_mul", c, 2 * mla["o_in"] // tp, mla["o_in"] // tp, 0.8),
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
        fmha_q = self.config.fmha_quant_mode
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
                    # beta (per v-head) + f_a low-rank down (head_dim); the full-rank
                    # onorm gate is fused into the qkvg GEMM above (fla kda.py
                    # projections; b_proj=num_heads, f_proj[0]=head_dim).
                    ops.GEMM("generation_kda_bfa_gemm", c, cfg.kda_num_heads + cfg.kda_head_dim, h, gemm_q),
                    ops.GEMM("generation_kda_fb_gemm", c, p_local, cfg.kda_head_dim, gemm_q),
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
            # Generation MLA block (vLLM): q_b up-projection, attention core
            # and o_proj. Same 0.27.0 serving-truth change as the context
            # side: upstream MultiHeadLatentAttention runs the real MLA lane,
            # so price the block against the module-level MLA tables; the
            # granular GQA stack is the FallbackOp fallback for systems whose
            # module tables have no rows yet.
            generation_mla_q_b = ops.GEMM("generation_mla_q_b_gemm", c, mla["q_b_out"] // tp, cfg.q_lora_rank, gemm_q)
            generation_mla_o = ops.GEMM(
                "generation_mla_o_gemm", c, h, mla["o_in"] // tp, gemm_q, low_precision_input=True
            )
            generation_mla_granular = [
                generation_mla_q_b,
                ops.GenerationAttention(
                    "generation_attention",
                    c,
                    self._num_heads // tp,
                    self._num_kv_heads // tp,
                    kvcache_q,
                    head_size=cfg.v_head_dim,
                ),
                generation_mla_o,
            ]
            if self._backend_name == "vllm":
                generation_mla_block = [
                    ops.FallbackOp(
                        "generation_mla_block",
                        primary=ops.MLAModule(
                            "generation_mla_module",
                            c,
                            False,
                            self._num_heads // tp,
                            kvcache_q,
                            fmha_q,
                            gemm_q,
                            native_num_heads=self._num_heads,
                        ),
                        fallback=generation_mla_granular,
                    )
                ]
            else:
                # sglang: granular absorbed-BMM stack. The absorb BMMs are
                # priced per exact local head count (96-family for K3). The
                # mla_bmm query routes exact-first: systems without exact
                # rows (only b200 sglang carries them today) fall back to the
                # next-pow2 DeepSeek slice scaled by the head ratio — see
                # engine's mla_bmm table (operators/mla.rs).
                generation_mla_block = [
                    generation_mla_q_b,
                    ops.MLABmm(
                        "generation_bmm_pre",
                        c,
                        self._num_heads // tp,
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
                        c,
                        self._num_heads // tp,
                        mla_bmm_q,
                        if_pre=False,
                    ),
                    generation_mla_o,
                ]
            self.generation_ops.extend(
                [
                    ops.ElementWise("generation_mla_norm", c, 2 * h, 2 * h, 0.8),
                    ops.GEMM("generation_mla_downscale_gemm", c, mla["fused_qkv_a_out"], h, gemm_q),
                    *generation_mla_block,
                    ops.GEMM("generation_mla_gate_gemm", c, mla["o_in"] // tp, h, gemm_q),
                    ops.ElementWise("generation_mla_gate_mul", c, 2 * mla["o_in"] // tp, mla["o_in"] // tp, 0.8),
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

        # --- DSPARK draft model: 5 dense layers at hidden 7168 ---
        # Backend-conditional geometry (the two published draft checkpoints
        # differ): sglang serves RadixArk/Kimi-K3-DSpark (GQA 64q/16kv/hd64),
        # vllm serves Inferact/Kimi-K3-DSpark (MLA-style 64q/64kv,
        # qk_nope 128 + rope 64, v 128, q_lora 1536 / kv_lora 512). The vllm
        # draft is priced through the same MLA-as-attention convention as the
        # main vllm MLA layers above; its output gate is disabled in the
        # checkpoint, so no gate ops.
        if spec:
            inter = self.DRAFT_INTER_SIZE
            dc = self.DRAFT_NUM_LAYERS
            # Draft forwards gamma = nextn tokens per step while the op batch is
            # scaled by (nextn + 1): scale counts by nextn / (nextn + 1).
            dsf = self._nextn / draft_tokens
            if is_vllm:
                n_q = self.VLLM_DRAFT_NUM_HEADS
                qk_dim = self.VLLM_DRAFT_QK_NOPE_DIM + self.VLLM_DRAFT_QK_ROPE_DIM
                v_dim = self.VLLM_DRAFT_V_DIM
                latent = self.VLLM_DRAFT_KV_LORA_RANK + self.VLLM_DRAFT_QK_ROPE_DIM
                attn_and_proj = [
                    # fused latent downscale (q_lora + kv_lora + rope), replicated
                    ops.GEMM(
                        "draft_downscale_gemm",
                        dc * dsf,
                        self.VLLM_DRAFT_Q_LORA_RANK + latent,
                        h,
                        gemm_q,
                    ),
                    ops.GEMM("draft_q_b_gemm", dc * dsf, n_q * qk_dim // tp, self.VLLM_DRAFT_Q_LORA_RANK, gemm_q),
                    ops.GenerationAttention(
                        "draft_attention",
                        dc * dsf,
                        n_q // tp,
                        n_q // tp,
                        kvcache_q,
                        head_size=v_dim,
                    ),
                    ops.GEMM("draft_proj_gemm", dc * dsf, h, n_q * v_dim // tp, gemm_q, low_precision_input=True),
                ]
                # Latent draft KV (TP-replicated) is written by the downscale
                # GEMM above; the committed-token projection reuses it.
                kv_proj = ops.GEMM("draft_kv_proj_gemm", dc * per_step, latent, h, gemm_q)
            else:
                n_q = self.DRAFT_NUM_HEADS
                n_kv = self.DRAFT_NUM_KV_HEADS
                hd = self.DRAFT_HEAD_DIM
                qkv_out = (n_q * hd + 2 * n_kv * hd) // tp
                attn_and_proj = [
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
                ]
                # target-hidden -> draft context-KV projection (per committed token)
                kv_proj = ops.GEMM("draft_kv_proj_gemm", dc * per_step, 2 * n_kv * hd // tp, h, gemm_q)
            self.generation_ops.extend(
                [
                    ops.Embedding("draft_embedding", 1 * dsf, self._vocab_size // tp, h, 0.3),
                    ops.ElementWise("draft_norm", 2 * dc * dsf, 2 * h, 2 * h, 0.8),
                    *attn_and_proj,
                    ops.GEMM("draft_gate_up_gemm", dc * dsf, 2 * inter // tp, h, gemm_q),
                    ops.ElementWise("draft_act_gate", dc * dsf, 2 * inter // tp, inter // tp, 0.8),
                    ops.GEMM("draft_down_gemm", dc * dsf, h, inter // tp, gemm_q, low_precision_input=True),
                    ops.CustomAllReduce("draft_ar", 2 * dc * dsf, h, tp),
                    kv_proj,
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
        # Total shared-expert intermediate width: 2 experts x 3072 = 6144. The
        # gate/up factor is applied at the GEMM sites below (2 * shared_inter),
        # so it must NOT be baked in here (review fix: a stray *2 inflated the
        # shared FFN ~2x).
        shared_inter = cfg.num_shared_experts * cfg.moe_inter_size

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
                        backend=self._backend_name,
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
                        backend=self._backend_name,
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
            if self._backend_name == "vllm":
                # Inferact draft is MLA-style: latent KV, TP-replicated.
                elements += self.DRAFT_NUM_LAYERS * (self.VLLM_DRAFT_KV_LORA_RANK + self.VLLM_DRAFT_QK_ROPE_DIM)
            else:
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
