# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import aiconfigurator_core.sdk.operations as ops
from aiconfigurator_core.sdk import common
from aiconfigurator_core.sdk.models.base import BaseModel, register_model
from aiconfigurator_core.sdk.models.helpers import mtp_scale_factor


@register_model("QWEN35")
class Qwen35Model(BaseModel):
    """
    Qwen3.5 hybrid GDN + full-attention model (dense and MoE variants).

    Handles two layer types from Qwen35Config.layer_types:
      - "linear_attention": Gated DeltaNet (GDN) layers using chunk_gated_delta_rule
      - "full_attention":   Standard GQA transformer layers

    All layers share the same FFN:
      - Dense models (27B):          SwiGLU dense FFN
      - MoE models (35B-A3B, 397B): All-MoE FFN (num_experts > 0)
    """

    @classmethod
    def create(cls, model_info: dict, model_config, backend_name: str) -> BaseModel:
        return cls(
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
            backend_name=backend_name,
        )

    def __init__(self, *args, backend_name: str) -> None:
        super().__init__(*args)
        self._backend_name = backend_name
        cfg: common.Qwen35Config = self.extra_params
        assert isinstance(cfg, common.Qwen35Config), "Qwen35Model requires Qwen35Config extra_params"

        tp = self.config.tp_size
        if cfg.linear_num_key_heads % tp != 0 or cfg.linear_num_value_heads % tp != 0:
            raise ValueError(
                "Qwen3.5 GDN head counts must both be divisible by tensor parallel size: "
                f"num_k_heads={cfg.linear_num_key_heads}, "
                f"num_v_heads={cfg.linear_num_value_heads}, tp_size={tp}"
            )

        if self.config.cp_size > 1:
            raise ValueError("Qwen3.5 does not model context parallelism; cp_size must be 1")

        self._mtp_scale_factor = mtp_scale_factor(self._nextn, self._num_layers)

        if cfg.num_experts > 0:
            assert (
                self.config.tp_size * self.config.attention_dp_size * self.config.cp_size
                == self.config.moe_tp_size * self.config.moe_ep_size
            ), (
                f"tp_size ({self.config.tp_size}) * attention_dp_size "
                f"({self.config.attention_dp_size}) * cp_size ({self.config.cp_size}) should equal moe_tp_size "
                f"({self.config.moe_tp_size}) * moe_ep_size ({self.config.moe_ep_size})"
            )
            assert cfg.num_experts >= self.config.moe_ep_size

        self._build_context_ops()
        self._build_generation_ops()

    def _count_layer_types(self) -> dict[str, int]:
        cfg: common.Qwen35Config = self.extra_params
        return {
            "linear": cfg.layer_types.count("linear_attention"),
            "full": cfg.layer_types.count("full_attention"),
        }

    # ------------------------------------------------------------------
    # Memory: paged KV on full-attention layers + constant GDN state
    # ------------------------------------------------------------------

    def get_kvcache_elements_per_token(self) -> int:
        """Only full_attention layers hold token-linear KV; GDN layers keep a
        constant per-request state priced in _gdn_state_bytes_per_request."""
        cfg: common.Qwen35Config = self.extra_params
        tp = self.config.tp_size
        n_kv_per_tp = (self._num_kv_heads + tp - 1) // tp
        return cfg.layer_types.count("full_attention") * 2 * n_kv_per_tp * self._head_size

    def _gdn_state_bytes_per_request(self) -> float:
        """Constant GDN state per request on one GPU: fp32 SSM state (Qwen3.5
        pins mamba_ssm_dtype=float32) + model-dtype conv window, per GDN layer,
        TP-sharded."""
        cfg: common.Qwen35Config = self.extra_params
        tp = self.config.tp_size
        n_gdn = cfg.layer_types.count("linear_attention")
        nk, hk = cfg.linear_num_key_heads, cfg.linear_key_head_dim
        nv, hv = cfg.linear_num_value_heads, cfg.linear_value_head_dim
        ssm_bytes = (nv // tp) * hk * hv * 4
        conv_bytes = (2 * nk * hk + nv * hv) // tp * (cfg.linear_conv_kernel_dim - 1) * 2
        return n_gdn * (ssm_bytes + conv_bytes)

    def get_kvcache_bytes_per_sequence(self, seq_len: int) -> float:
        seq_len = max(0, seq_len)
        token_bytes = seq_len * self.config.kvcache_quant_mode.value.memory * self.get_kvcache_elements_per_token()
        return token_bytes + self._gdn_state_bytes_per_request()

    def get_kvcache_max_tokens(self, kv_budget_bytes: float) -> int:
        # Single elastic byte pool holding one request's GDN state — same
        # assumption as kimi_k3.get_kvcache_max_tokens.
        per_token = self.config.kvcache_quant_mode.value.memory * self.get_kvcache_elements_per_token()
        budget = kv_budget_bytes - self._gdn_state_bytes_per_request()
        if budget <= 0 or per_token <= 0:
            return 0
        return int(budget // per_token)

    def _build_context_ops(self) -> None:
        cfg: common.Qwen35Config = self.extra_params
        h = self._hidden_size
        tp = self.config.tp_size
        pp = self.config.pp_size
        moe_tp = self.config.moe_tp_size
        moe_ep = self.config.moe_ep_size
        attn_dp = self.config.attention_dp_size
        attn_ar_folded = self._sglang_folds_attn_ar()
        gemm_q = self.config.gemm_quant_mode
        kvcache_q = self.config.kvcache_quant_mode
        fmha_q = self.config.fmha_quant_mode
        moe_q = self.config.moe_quant_mode
        workload_dist = (
            self.config.workload_distribution + "_1.2"
            if self.config.workload_distribution == "power_law"
            else self.config.workload_distribution
        )
        counts = self._count_layer_types()

        # GDN kernel lookups use TP-local head counts; projection GEMM widths
        # derive from the global dims.
        nk = cfg.linear_num_key_heads
        hk = cfg.linear_key_head_dim
        nv = cfg.linear_num_value_heads
        hv = cfg.linear_value_head_dim
        d_conv = cfg.linear_conv_kernel_dim

        # Per-TP sizes
        n_q_per_tp = self._num_heads // tp
        n_kv_per_tp = (self._num_kv_heads + tp - 1) // tp
        gdn_nk_per_tp = nk // tp
        gdn_nv_per_tp = nv // tp
        # in_proj_qkvz (Q+K+V+gate Z) and in_proj_ba (b+a, 2 scalars per V head)
        # are separate runtime GEMM kernels.
        gdn_in_proj_out = (nk * hk + nk * hk + nv * hv + nv * hv) // tp
        gdn_ba_out = 2 * nv // tp
        gdn_out_proj_in = nv * hv // tp

        self.context_ops = [
            ops.Embedding("context_embedding", 1, self._vocab_size // tp, h, 0.3),
            ops.CustomAllReduce("context_embedding_ar", 1, h, tp),
        ]

        # --- linear_attention (GDN) layers ---
        if counts["linear"] > 0:
            c = counts["linear"]
            self.context_ops.extend(
                [
                    ops.ElementWise("context_gdn_norm", c, 2 * h, 2 * h, 0.8),
                    ops.GEMM("context_gdn_in_proj_gemm", c, gdn_in_proj_out, h, gemm_q),
                    # 2*nv/tp drops below the collected GEMM n-grid at high TP.
                    ops.GEMM("context_gdn_in_proj_ba_gemm", c, gdn_ba_out, h, gemm_q, below_grid_sol=True),
                    ops.GDNKernel(
                        "context_gdn_conv1d",
                        c,
                        "causal_conv1d_fn",
                        "context",
                        h,
                        gdn_nk_per_tp,
                        hk,
                        gdn_nv_per_tp,
                        hv,
                        d_conv,
                    ),
                    ops.GDNKernel(
                        "context_gdn_scan",
                        c,
                        "chunk_gated_delta_rule",
                        "context",
                        h,
                        gdn_nk_per_tp,
                        hk,
                        gdn_nv_per_tp,
                        hv,
                        d_conv,
                    ),
                    ops.GEMM("context_gdn_out_proj_gemm", c, h, gdn_out_proj_in, gemm_q, low_precision_input=True),
                    *([] if attn_ar_folded else [ops.CustomAllReduce("context_gdn_ar", c, h, tp)]),
                ]
            )
            self.context_ops.extend(
                self._ffn_context_ops(
                    "context_gdn", c, h, tp, moe_tp, moe_ep, attn_dp, gemm_q, moe_q, workload_dist, cfg
                )
            )

        # --- full_attention (GQA) layers ---
        if counts["full"] > 0:
            c = counts["full"]
            # attn_output_gate=True on all Qwen3.5 checkpoints doubles the q slice
            # (query + output gate).
            qkv_out = 2 * n_q_per_tp * self._head_size + n_kv_per_tp * self._head_size * 2
            self.context_ops.extend(
                [
                    ops.ElementWise("context_full_attn_norm", c, 2 * h, 2 * h, 0.8),
                    ops.GEMM("context_qkv_gemm", c, qkv_out, h, gemm_q),
                    ops.ContextAttention(
                        "context_attention",
                        c,
                        n_q_per_tp,
                        n_kv_per_tp,
                        kvcache_q,
                        fmha_q,
                        head_size=self._head_size,
                        use_qk_norm=True,
                    ),
                    ops.GEMM("context_proj_gemm", c, h, n_q_per_tp * self._head_size, gemm_q, low_precision_input=True),
                    *([] if attn_ar_folded else [ops.CustomAllReduce("context_full_ar", c, h, tp)]),
                ]
            )
            self.context_ops.extend(
                self._ffn_context_ops(
                    "context_full", c, h, tp, moe_tp, moe_ep, attn_dp, gemm_q, moe_q, workload_dist, cfg
                )
            )

        self.context_ops.extend(
            [
                ops.GEMM("context_logits_gemm", 1, self._vocab_size // tp, h, common.GEMMQuantMode.bfloat16),
                ops.P2P("context_p2p", pp - 1, h, pp),
            ]
        )

    def _sglang_deepep(self) -> bool:
        return self._backend_name == "sglang" and self.config.moe_backend == "deepep_moe"

    def _sglang_folds_attn_ar(self) -> bool:
        # sglang projections never all-reduce (reduce_results=False); with DP
        # attention the attn-TP reduction folds into the pre-MLP gather that
        # moe_pre_dispatch prices. DeepEP scatters and keeps the model-side AR.
        return (
            self._backend_name == "sglang"
            and not self._sglang_deepep()
            and self.extra_params.num_experts > 0
            and self.config.attention_dp_size > 1
        )

    def _sglang_fused_shared_gate(self) -> bool:
        # sglang (non-DeepEP) folds the scalar-gate GEMV + sigmoid + mul +
        # final add into one kernel after the streams join; the shared branch
        # itself is gate_up/act/down only. DeepEP and vLLM apply the gate
        # inside the branch.
        return self._backend_name == "sglang" and not self._sglang_deepep()

    def _shared_expert_ops(self, prefix, count, h, tp, gemm_q, cfg: common.Qwen35Config, *, scale_num_tokens=1):
        """Shared-expert branch: gated-SiLU MLP with a fused gate_up
        projection; the scalar expert gate is applied inside the branch
        except on the sglang fused-gate path (merged into the post-join
        kernel). DeepEP replicates the shared expert across ranks (tp_size=1)
        and runs it on the attn-TP scattered token slice in context."""
        if self._sglang_deepep():
            tp = 1
        fused_gate = self._sglang_fused_shared_gate()
        ops_list = []
        if not fused_gate:
            # Scalar expert gate (ReplicatedLinear hidden->1); n=1 sits below
            # the collected GEMM n-grid.
            ops_list.append(
                ops.GEMM(
                    f"{prefix}_shared_expert_gate_gemm",
                    count,
                    1,
                    h,
                    common.GEMMQuantMode.bfloat16,
                    scale_num_tokens=scale_num_tokens,
                    below_grid_sol=True,
                )
            )
        ops_list.extend(
            [
                ops.GEMM(
                    f"{prefix}_shared_gate_up_gemm",
                    count,
                    2 * cfg.shared_expert_inter_size // tp,
                    h,
                    gemm_q,
                    scale_num_tokens=scale_num_tokens,
                ),
                ops.ElementWise(
                    f"{prefix}_shared_act_gate",
                    count,
                    2 * cfg.shared_expert_inter_size // tp,
                    cfg.shared_expert_inter_size // tp,
                    0.8,
                    scale_num_tokens=scale_num_tokens,
                ),
                ops.GEMM(
                    f"{prefix}_shared_down_gemm",
                    count,
                    h,
                    cfg.shared_expert_inter_size // tp,
                    gemm_q,
                    low_precision_input=True,
                    scale_num_tokens=scale_num_tokens,
                ),
            ]
        )
        if not fused_gate:
            # sigmoid(expert gate) * shared output.
            ops_list.append(
                ops.ElementWise(
                    f"{prefix}_shared_expert_gate_mul",
                    count,
                    h,
                    h,
                    0.8,
                    scale_num_tokens=scale_num_tokens,
                )
            )
        return ops_list

    def _shared_merge_op(self, prefix, count, h, *, scale_num_tokens=1):
        if self._sglang_fused_shared_gate():
            # One fused kernel: scalar-gate GEMV + sigmoid + mul + final add
            # (reads hidden, shared and routed outputs; writes h).
            return ops.ElementWise(f"{prefix}_shared_merge", count, 3 * h, h, 0.8, scale_num_tokens=scale_num_tokens)
        # Final add of routed and gated shared outputs (after the streams join).
        return ops.ElementWise(f"{prefix}_shared_merge", count, 2 * h, h, 0.8, scale_num_tokens=scale_num_tokens)

    def _ffn_context_ops(
        self, prefix, count, h, tp, moe_tp, moe_ep, attn_dp, gemm_q, moe_q, workload_dist, cfg: common.Qwen35Config
    ):
        """Return FFN ops for context phase: dense SwiGLU or MoE."""
        ops_list = [ops.ElementWise(f"{prefix}_ffn_norm", count, 2 * h, 2 * h, 0.8)]
        if cfg.num_experts > 0:
            if cfg.num_experts >= 128:
                ops_list.append(
                    ops.GEMM(f"{prefix}_router_gemm", count, cfg.num_experts, h, common.GEMMQuantMode.bfloat16)
                )
            # StandardDispatcher (sglang default MoE path) has no pre-dispatch
            # collective when attention_dp == 1; with DP the LayerCommunicator
            # pre-MLP gather is priced by the dispatch op.
            if not (self._backend_name == "sglang" and self.config.moe_backend is None and attn_dp == 1):
                ops_list.append(
                    ops.MoEDispatch(
                        f"{prefix}_moe_pre_dispatch",
                        count,
                        h,
                        cfg.topk,
                        cfg.num_experts,
                        moe_tp,
                        moe_ep,
                        attn_dp,
                        True,
                        quant_mode=moe_q,
                        sms=self.config.sms,
                        moe_backend=self.config.moe_backend,
                        is_context=True,
                        scale_num_tokens=tp,
                        attn_ar_modeled=True,
                    )
                )
            ops_list.append(
                ops.MoE(
                    f"{prefix}_moe",
                    count,
                    h,
                    cfg.moe_inter_size,
                    cfg.topk,
                    cfg.num_experts,
                    moe_tp,
                    moe_ep,
                    moe_q,
                    workload_dist,
                    attn_dp,
                    is_context=True,
                    moe_backend=self.config.moe_backend,
                    # EPLB is not modeled for Qwen3.5 (the load curve stays 1.2).
                    enable_eplb=False,
                )
            )
            # DeepEP rows hold the full dispatch+combine round trip; the pre
            # op prices it once (SGLangEPMOEModel precedent), so no post op.
            if not self._sglang_deepep():
                ops_list.append(
                    ops.MoEDispatch(
                        f"{prefix}_moe_post_dispatch",
                        count,
                        h,
                        cfg.topk,
                        cfg.num_experts,
                        moe_tp,
                        moe_ep,
                        attn_dp,
                        False,
                        quant_mode=moe_q,
                        sms=self.config.sms,
                        moe_backend=self.config.moe_backend,
                        is_context=True,
                        attn_ar_modeled=True,
                    )
                )
            if cfg.shared_expert_inter_size > 0:
                # DeepEP context runs the shared expert (and merge) on the
                # attn-TP scattered token slice.
                shared_scale = tp if self._sglang_deepep() else 1
                ops_list.extend(
                    self._shared_expert_ops(prefix, count, h, tp, gemm_q, cfg, scale_num_tokens=shared_scale)
                )
                ops_list.append(self._shared_merge_op(prefix, count, h, scale_num_tokens=shared_scale))
        else:
            ops_list.extend(
                [
                    ops.GEMM(f"{prefix}_gate_ffn1_gemm", count, 2 * self._inter_size // tp, h, gemm_q),
                    ops.ElementWise(
                        f"{prefix}_act_gate", count, 2 * self._inter_size // tp, self._inter_size // tp, 0.8
                    ),
                    ops.GEMM(f"{prefix}_ffn2_gemm", count, h, self._inter_size // tp, gemm_q, low_precision_input=True),
                    ops.CustomAllReduce(f"{prefix}_ffn_ar", count, h, tp),
                ]
            )
        return ops_list

    def _build_generation_ops(self) -> None:
        cfg: common.Qwen35Config = self.extra_params
        h = self._hidden_size
        tp = self.config.tp_size
        pp = self.config.pp_size
        moe_tp = self.config.moe_tp_size
        moe_ep = self.config.moe_ep_size
        attn_dp = self.config.attention_dp_size
        attn_ar_folded = self._sglang_folds_attn_ar()
        gemm_q = self.config.gemm_quant_mode
        kvcache_q = self.config.kvcache_quant_mode
        moe_q = self.config.moe_quant_mode
        workload_dist = (
            self.config.workload_distribution + "_1.2"
            if self.config.workload_distribution == "power_law"
            else self.config.workload_distribution
        )
        counts = self._count_layer_types()

        nk = cfg.linear_num_key_heads
        hk = cfg.linear_key_head_dim
        nv = cfg.linear_num_value_heads
        hv = cfg.linear_value_head_dim
        d_conv = cfg.linear_conv_kernel_dim

        n_q_per_tp = self._num_heads // tp
        n_kv_per_tp = (self._num_kv_heads + tp - 1) // tp
        gdn_nk_per_tp = nk // tp
        gdn_nv_per_tp = nv // tp
        gdn_in_proj_out = (nk * hk + nk * hk + nv * hv + nv * hv) // tp
        gdn_ba_out = 2 * nv // tp
        gdn_out_proj_in = nv * hv // tp

        sf = self._mtp_scale_factor

        self.generation_ops = [
            ops.Embedding("generation_embedding", 1 * sf, self._vocab_size // tp, h, 0.3),
            ops.CustomAllReduce("generation_embedding_ar", 1 * sf, h, tp),
        ]

        # --- linear_attention (GDN) layers ---
        if counts["linear"] > 0:
            c = counts["linear"] * sf
            self.generation_ops.extend(
                [
                    ops.ElementWise("generation_gdn_norm", c, 2 * h, 2 * h, 0.8),
                    ops.GEMM("generation_gdn_in_proj_gemm", c, gdn_in_proj_out, h, gemm_q),
                    ops.GEMM("generation_gdn_in_proj_ba_gemm", c, gdn_ba_out, h, gemm_q, below_grid_sol=True),
                    ops.GDNKernel(
                        "generation_gdn_conv1d",
                        c,
                        "causal_conv1d_update",
                        "generation",
                        h,
                        gdn_nk_per_tp,
                        hk,
                        gdn_nv_per_tp,
                        hv,
                        d_conv,
                    ),
                    ops.GDNKernel(
                        "generation_gdn_recurrence",
                        c,
                        "fused_sigmoid_gating_delta_rule_update",
                        "generation",
                        h,
                        gdn_nk_per_tp,
                        hk,
                        gdn_nv_per_tp,
                        hv,
                        d_conv,
                    ),
                    ops.GEMM("generation_gdn_out_proj_gemm", c, h, gdn_out_proj_in, gemm_q, low_precision_input=True),
                    *([] if attn_ar_folded else [ops.CustomAllReduce("generation_gdn_ar", c, h, tp)]),
                ]
            )
            self.generation_ops.extend(
                self._ffn_generation_ops(
                    "generation_gdn", c, h, tp, moe_tp, moe_ep, attn_dp, gemm_q, moe_q, workload_dist, cfg
                )
            )

        # --- full_attention (GQA) layers ---
        if counts["full"] > 0:
            c = counts["full"] * sf
            # attn_output_gate=True on all Qwen3.5 checkpoints doubles the q slice
            # (query + output gate).
            qkv_out = 2 * n_q_per_tp * self._head_size + n_kv_per_tp * self._head_size * 2
            self.generation_ops.extend(
                [
                    ops.ElementWise("generation_full_attn_norm", c, 2 * h, 2 * h, 0.8),
                    ops.GEMM("generation_qkv_gemm", c, qkv_out, h, gemm_q),
                    ops.GenerationAttention(
                        "generation_attention",
                        c,
                        n_q_per_tp,
                        n_kv_per_tp,
                        kvcache_q,
                        head_size=self._head_size,
                        use_qk_norm=True,
                    ),
                    ops.GEMM(
                        "generation_proj_gemm", c, h, n_q_per_tp * self._head_size, gemm_q, low_precision_input=True
                    ),
                    *([] if attn_ar_folded else [ops.CustomAllReduce("generation_full_ar", c, h, tp)]),
                ]
            )
            self.generation_ops.extend(
                self._ffn_generation_ops(
                    "generation_full", c, h, tp, moe_tp, moe_ep, attn_dp, gemm_q, moe_q, workload_dist, cfg
                )
            )

        self.generation_ops.extend(
            [
                ops.GEMM("generation_logits_gemm", 1 * sf, self._vocab_size // tp, h, common.GEMMQuantMode.bfloat16),
                ops.P2P("generation_p2p", (pp - 1) * sf, h, pp),
            ]
        )

    def _ffn_generation_ops(
        self, prefix, count, h, tp, moe_tp, moe_ep, attn_dp, gemm_q, moe_q, workload_dist, cfg: common.Qwen35Config
    ):
        """Return FFN ops for generation phase: dense SwiGLU or MoE."""
        ops_list = [ops.ElementWise(f"{prefix}_ffn_norm", count, 2 * h, 2 * h, 0.8)]
        if cfg.num_experts > 0:
            routed_ops = []
            if cfg.num_experts >= 128:
                routed_ops.append(
                    ops.GEMM(f"{prefix}_router_gemm", count, cfg.num_experts, h, common.GEMMQuantMode.bfloat16)
                )
            # StandardDispatcher (sglang default MoE path) has no pre-dispatch
            # collective when attention_dp == 1; with DP the LayerCommunicator
            # pre-MLP gather is priced by the dispatch op.
            if not (self._backend_name == "sglang" and self.config.moe_backend is None and attn_dp == 1):
                pre_dispatch = ops.MoEDispatch(
                    f"{prefix}_moe_pre_dispatch",
                    count,
                    h,
                    cfg.topk,
                    cfg.num_experts,
                    moe_tp,
                    moe_ep,
                    attn_dp,
                    True,
                    quant_mode=moe_q,
                    sms=self.config.sms,
                    moe_backend=self.config.moe_backend,
                    is_context=False,
                    attn_ar_modeled=True,
                )
                # sglang's pre-MLP gather completes before the dual-stream
                # fork; vLLM's DP dispatch runs on the main stream inside it.
                if self._backend_name == "sglang":
                    ops_list.append(pre_dispatch)
                else:
                    routed_ops.append(pre_dispatch)
            routed_ops.append(
                ops.MoE(
                    f"{prefix}_moe",
                    count,
                    h,
                    cfg.moe_inter_size,
                    cfg.topk,
                    cfg.num_experts,
                    moe_tp,
                    moe_ep,
                    moe_q,
                    workload_dist,
                    attn_dp,
                    is_context=False,
                    moe_backend=self.config.moe_backend,
                    enable_eplb=False,
                )
            )
            shared_ops = (
                self._shared_expert_ops(prefix, count, h, tp, gemm_q, cfg) if cfg.shared_expert_inter_size > 0 else []
            )
            # vLLM and sglang decode run shared and routed experts on
            # parallel CUDA streams (vLLM only up to 256 tokens per rank —
            # modeled as always overlapped); sglang DeepEP runs them serially.
            if shared_ops and self._backend_name in ("vllm", "sglang") and not self._sglang_deepep():
                ops_list.append(ops.OverlapOp(f"{prefix}_moe_overlap", group_a=routed_ops, group_b=shared_ops))
            else:
                ops_list.extend(routed_ops)
                ops_list.extend(shared_ops)
            if cfg.shared_expert_inter_size > 0:
                ops_list.append(self._shared_merge_op(prefix, count, h))
            # DeepEP rows hold the full dispatch+combine round trip; the pre
            # op prices it once (SGLangEPMOEModel precedent), so no post op.
            # Both frameworks all-reduce the merged shared+routed sum, so the
            # post collective sits after the join, outside the overlap.
            if not self._sglang_deepep():
                ops_list.append(
                    ops.MoEDispatch(
                        f"{prefix}_moe_post_dispatch",
                        count,
                        h,
                        cfg.topk,
                        cfg.num_experts,
                        moe_tp,
                        moe_ep,
                        attn_dp,
                        False,
                        quant_mode=moe_q,
                        sms=self.config.sms,
                        moe_backend=self.config.moe_backend,
                        is_context=False,
                        attn_ar_modeled=True,
                    )
                )
        else:
            ops_list.extend(
                [
                    ops.GEMM(f"{prefix}_gate_ffn1_gemm", count, 2 * self._inter_size // tp, h, gemm_q),
                    ops.ElementWise(
                        f"{prefix}_act_gate", count, 2 * self._inter_size // tp, self._inter_size // tp, 0.8
                    ),
                    ops.GEMM(f"{prefix}_ffn2_gemm", count, h, self._inter_size // tp, gemm_q, low_precision_input=True),
                    ops.CustomAllReduce(f"{prefix}_ffn_ar", count, h, tp),
                ]
            )
        return ops_list
