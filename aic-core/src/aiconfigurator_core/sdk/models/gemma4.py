# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import aiconfigurator_core.sdk.operations as ops
from aiconfigurator_core.sdk import common
from aiconfigurator_core.sdk.models.base import BaseModel, register_model
from aiconfigurator_core.sdk.models.helpers import mtp_scale_factor


@register_model("GEMMA4MIX")
class Gemma4MixModel(BaseModel):
    """
    Google Gemma 4 (gemma4_text): hybrid SWA/global attention + shared dense MLP
    running in parallel with routed top-k MoE on every layer.

    Two layer-type recipes are emitted, driven by ``Gemma4MixConfig.layer_types``:

    - **sliding_attention (SWA)**: 16 Q heads x ``swa_head_dim``, ``swa_num_kv_heads``
      KV heads, separate K and V projections (standard GQA), token window =
      ``sliding_window_size``.
    - **full_attention (global)**: 16 Q heads x ``global_head_dim``, ``global_num_kv_heads``
      KV heads, no window. When ``attention_k_eq_v`` is set, V is reused from the K
      projection output -- there is no v_proj weight or v_proj GEMM on these layers,
      but K and V are still distinct tensors in the KV cache (post-norm/post-RoPE).

    Every layer (gated on global ``enable_moe_block`` at parse time) runs both:

    - a shared dense MLP at intermediate_size ``inter_size``, gated SwiGLU; and
    - a routed top-k MoE at expert intermediate ``moe_inter_size`` with ``num_experts``
      experts and a router GEMM (the router GEMM is emitted only when num_experts >= 128,
      mirroring the MoEModel/HybridMoEModel convention).

    Outputs of the two FFN branches are summed before the post-feedforward norm.
    """

    @classmethod
    def supports_cp(cls, backend_name: str) -> bool:
        # Dense SWA/global GQA prefill CP: SGLang AllGather (zigzag FMHA).
        return backend_name == "sglang"

    @classmethod
    def create(cls, model_info: dict, model_config, backend_name: str) -> BaseModel:
        model = cls(
            model_info["topk"],
            model_info["num_experts"],
            model_info["moe_inter_size"],
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
        model.set_gemma4_config(model_info["extra_params"])
        return model

    def __init__(self, topk: int, num_experts: int, moe_inter_size: int, *args) -> None:
        super().__init__(*args)
        # Gemma 4 family includes both MoE variants (e.g. gemma-4-26B-A4B-it,
        # topk=8/num_experts=128) and dense variants (e.g. gemma-4-31B-it,
        # gemma-4-E2B-it, gemma-4-E4B-it: topk=0/num_experts=None). For dense
        # variants every layer is just the shared dense MLP -- there is no
        # routed-MoE block, so MoE-related parallelism constraints don't apply.
        self._is_dense = not topk or not num_experts
        if not self._is_dense:
            assert (
                self.config.tp_size * self.config.attention_dp_size * self.config.cp_size
                == self.config.moe_tp_size * self.config.moe_ep_size
            ), (
                f"tp_size ({self.config.tp_size}) * attention_dp_size "
                f"({self.config.attention_dp_size}) * cp_size ({self.config.cp_size}) should be equal to "
                f"moe_tp_size ({self.config.moe_tp_size}) * moe_ep_size ({self.config.moe_ep_size})"
            )
            assert num_experts >= self.config.moe_ep_size, f"ep size cannot be larger than num_experts {num_experts}"
        else:
            # Dense Gemma 4: collapse the MoE search-space to ep=1 so pareto
            # iteration doesn't enumerate equivalent dense configurations.
            assert self.config.moe_ep_size == 1, (
                f"dense Gemma 4 variants require moe_ep_size=1, got {self.config.moe_ep_size}"
            )
        self._topk = topk
        self._num_experts = num_experts
        self._moe_inter_size = moe_inter_size
        self._mtp_scale_factor = mtp_scale_factor(self._nextn, self._num_layers)
        self._gemma4_config: common.Gemma4MixConfig | None = None
        self._power_law_alpha = 1.01

    def set_gemma4_config(self, cfg: common.Gemma4MixConfig) -> None:
        """Apply Gemma4MixConfig and rebuild context/generation ops.

        Validates that ``layer_types`` length matches ``num_layers`` and contains only
        recognized values before accepting the config.
        """
        if cfg is None or not isinstance(cfg, common.Gemma4MixConfig):
            raise ValueError(f"Gemma4MixModel requires a Gemma4MixConfig, got {type(cfg).__name__}")
        if len(cfg.layer_types) != self._num_layers:
            raise ValueError(
                f"Gemma4MixConfig.layer_types length ({len(cfg.layer_types)}) "
                f"does not match num_layers ({self._num_layers})"
            )
        for i, lt in enumerate(cfg.layer_types):
            if lt not in ("sliding_attention", "full_attention"):
                raise ValueError(f"Gemma4MixConfig layer {i} has invalid type {lt!r}")
        self._gemma4_config = cfg
        self._build_context_ops()
        self._build_generation_ops()
        if self.config.cp_size > 1:
            # decode never runs CP. Route the generation MoEDispatch ops to their
            # decode-CP comm path (pre=0 / post=all_reduce) rather than prefill's
            # all_gather/reduce_scatter -- attn_cp_size>1 + is_context=False. The
            # context loop in _build_context_ops handles the prefill side.
            cp = self.config.cp_size
            for op in self.generation_ops:
                if isinstance(op, ops.MoEDispatch):
                    op._attn_cp_size = cp
                    op._is_context = False

    def _count_layer_types(self) -> dict[str, int]:
        cfg = self._gemma4_config
        return {
            "swa": cfg.layer_types.count("sliding_attention"),
            "global": cfg.layer_types.count("full_attention"),
        }

    def _resolve_dims(self, tp_size: int) -> dict:
        """Resolve per-layer-type attention dims and dense-MLP intermediate per TP shard."""
        cfg = self._gemma4_config
        swa_n_kv = cfg.swa_num_kv_heads
        global_n_kv = cfg.global_num_kv_heads
        swa_hd = cfg.swa_head_dim
        global_hd = cfg.global_head_dim
        swa_n_kv_per_gpu = (swa_n_kv + tp_size - 1) // tp_size
        global_n_kv_per_gpu = (global_n_kv + tp_size - 1) // tp_size

        # QKV-projection output width. SWA always has separate K and V buffers.
        swa_qkv_out = self._num_heads * swa_hd // tp_size + swa_n_kv_per_gpu * swa_hd * 2
        # Global: when attention_k_eq_v, V reuses K's projection output -> only Q + K.
        if cfg.attention_k_eq_v:
            global_qkv_out = self._num_heads * global_hd // tp_size + global_n_kv_per_gpu * global_hd
        else:
            global_qkv_out = self._num_heads * global_hd // tp_size + global_n_kv_per_gpu * global_hd * 2

        return {
            "swa_n_kv_per_gpu": swa_n_kv_per_gpu,
            "global_n_kv_per_gpu": global_n_kv_per_gpu,
            "swa_qkv_out": swa_qkv_out,
            "global_qkv_out": global_qkv_out,
            "swa_proj_in": self._num_heads * swa_hd // tp_size,
            "global_proj_in": self._num_heads * global_hd // tp_size,
            "swa_hd": swa_hd,
            "global_hd": global_hd,
            "dense_inter_per_tp": self._inter_size // tp_size,
        }

    def _shared_mlp_ops(self, prefix: str, count: float, h: int, dense_inter_per_tp: int) -> list:
        """Shared dense MLP (Gemma4TextMLP): gated SwiGLU. Runs on every layer."""
        gemm_q = self.config.gemm_quant_mode
        return [
            ops.GEMM(f"{prefix}_shared_mlp_gate_up_gemm", count, 2 * dense_inter_per_tp, h, gemm_q),
            ops.ElementWise(f"{prefix}_shared_mlp_act", count, 2 * dense_inter_per_tp, dense_inter_per_tp, 0.8),
            ops.GEMM(f"{prefix}_shared_mlp_down_gemm", count, h, dense_inter_per_tp, gemm_q, low_precision_input=True),
        ]

    def _moe_ops(
        self,
        prefix: str,
        count: float,
        h: int,
        moe_tp: int,
        moe_ep: int,
        attn_dp: int,
        moe_q: common.MoEQuantMode,
        wl_dist: str,
    ) -> list:
        """Routed-MoE ops: router GEMM (when num_experts ≥ 128), pre-dispatch, MoE, post-dispatch.

        Returns an empty list for dense Gemma 4 variants (no routed-MoE block).
        """
        if self._is_dense:
            return []
        router_ops = (
            [ops.GEMM(f"{prefix}_router_gemm", count, self._num_experts, h, common.GEMMQuantMode.bfloat16)]
            if self._num_experts >= 128
            else []
        )
        return router_ops + [
            ops.MoEDispatch(
                f"{prefix}_moe_pre_dispatch",
                count,
                h,
                self._topk,
                self._num_experts,
                moe_tp,
                moe_ep,
                attn_dp,
                True,
                quant_mode=moe_q,
            ),
            ops.MoE(
                f"{prefix}_moe",
                count,
                h,
                self._moe_inter_size,
                self._topk,
                self._num_experts,
                moe_tp,
                moe_ep,
                moe_q,
                wl_dist,
                attn_dp,
            ),
            ops.MoEDispatch(
                f"{prefix}_moe_post_dispatch",
                count,
                h,
                self._topk,
                self._num_experts,
                moe_tp,
                moe_ep,
                attn_dp,
                False,
                quant_mode=moe_q,
            ),
        ]

    def _build_context_ops(self) -> None:
        if not self._gemma4_config:
            return

        cfg = self._gemma4_config
        counts = self._count_layer_types()
        h = self._hidden_size
        tp = self.config.tp_size
        moe_tp = self.config.moe_tp_size
        moe_ep = self.config.moe_ep_size
        attn_dp = self.config.attention_dp_size
        pp = self.config.pp_size
        gemm_q = self.config.gemm_quant_mode
        kvcache_q = self.config.kvcache_quant_mode
        fmha_q = self.config.fmha_quant_mode
        moe_q = self.config.moe_quant_mode
        wl_dist = (
            self.config.workload_distribution + f"_{self._power_law_alpha}"
            if self.config.workload_distribution == "power_law"
            else self.config.workload_distribution
        )
        d = self._resolve_dims(tp)

        self.context_ops = [ops.Embedding("context_embedding", 1, self._vocab_size, h, 0.3)]

        # --- sliding_attention layers ---
        if counts["swa"] > 0:
            c = counts["swa"]
            self.context_ops.extend(
                [
                    ops.ElementWise("context_swa_attn_norm", c, 2 * h, 2 * h, 0.8),
                    ops.GEMM("context_swa_qkv_gemm", c, d["swa_qkv_out"], h, gemm_q),
                    ops.ContextAttention(
                        "context_attention",
                        c,
                        self._num_heads // tp,
                        d["swa_n_kv_per_gpu"],
                        kvcache_q,
                        fmha_q,
                        window_size=cfg.sliding_window_size,
                        head_size=d["swa_hd"],
                    ),
                    ops.GEMM("context_swa_proj_gemm", c, h, d["swa_proj_in"], gemm_q, low_precision_input=True),
                    ops.ElementWise("context_swa_ffn_norm", c, 2 * h, 2 * h, 0.8),
                ]
                + self._shared_mlp_ops("context_swa", c, h, d["dense_inter_per_tp"])
                + self._moe_ops("context_swa", c, h, moe_tp, moe_ep, attn_dp, moe_q, wl_dist)
            )

        # --- full_attention (global) layers ---
        if counts["global"] > 0:
            c = counts["global"]
            self.context_ops.extend(
                [
                    ops.ElementWise("context_global_attn_norm", c, 2 * h, 2 * h, 0.8),
                    ops.GEMM("context_global_qkv_gemm", c, d["global_qkv_out"], h, gemm_q),
                    ops.ContextAttention(
                        "context_attention",
                        c,
                        self._num_heads // tp,
                        d["global_n_kv_per_gpu"],
                        kvcache_q,
                        fmha_q,
                        window_size=0,
                        head_size=d["global_hd"],
                    ),
                    ops.GEMM("context_global_proj_gemm", c, h, d["global_proj_in"], gemm_q, low_precision_input=True),
                    ops.ElementWise("context_global_ffn_norm", c, 2 * h, 2 * h, 0.8),
                ]
                + self._shared_mlp_ops("context_global", c, h, d["dense_inter_per_tp"])
                + self._moe_ops("context_global", c, h, moe_tp, moe_ep, attn_dp, moe_q, wl_dist)
            )

        self.context_ops.extend(
            [
                ops.GEMM("context_logits_gemm", 1, self._vocab_size // tp, h, common.GEMMQuantMode.bfloat16),
                ops.P2P("context_p2p", pp - 1, h, pp),
            ]
        )

        # cp (SGLang prefill AllGather CP). Gemma4 has heterogeneous KV per
        # layer-type (SWA vs global) -> emit one NCCL all_gather per type,
        # weighted by its layer count, so total comm matches the runtime.
        # Dense FMHA uses zigzag (``cp_size`` on the attention op, balanced
        # full/cp work); token-major ops shrink the M-axis via ``seq_split``
        # (DB lookup at per-rank M). This bypasses the BaseModel CP helper,
        # which assumes one uniform per-token KV size.
        # NOTE: the SWA all_gather is sized by the full new-token count (not the
        # window) on purpose -- this matches sglang v0.5.13
        # ``cp_allgather_and_save_kv_cache`` / ``cp_all_gather_rerange_kv_cache``,
        # which gather the FULL per-layer new-token KV across CP ranks; the
        # sliding window only caps the SWA write target / stored KV (handled in
        # get_kvcache_bytes_per_sequence), not this per-layer comm volume.
        if self.config.cp_size > 1:
            cp = self.config.cp_size
            kvcache_bytes = self.config.kvcache_quant_mode.value.memory
            comm_bytes = self.config.comm_quant_mode.value.memory
            # Post-construction CP wiring (not the __init__ _CP_AWARE gate): this
            # family has heterogeneous layer types (SWA vs global / dense vs MoE)
            # built across separate passes, so CP is applied here once every op
            # exists. The per-op _CP_AWARE opt-in is re-asserted in the loop so an
            # un-audited op still fails loud instead of silently skipping CP.
            for op in self.context_ops:
                if isinstance(op, ops.ContextAttention):
                    op._cp_size = cp
                elif isinstance(op, ops.MoEDispatch):
                    # MoEDispatch keys CP off attn_cp_size (AG pre / RS post),
                    # NOT seq_split; with moe_ep=cp its attention_tp_size>1 would
                    # otherwise wrongly take the TP all-reduce path.
                    op._attn_cp_size = cp
                elif op._CP_AWARE:
                    # Token-major op: shrink the M-axis. This post-construction
                    # mutation bypasses the constructor's _CP_AWARE gate, so
                    # re-assert the opt-in here -- an un-audited op in a
                    # CP-enabled pipeline must fail loud, not silently skip CP.
                    op._seq_split = cp
                else:
                    raise NotImplementedError(
                        f"{type(op).__name__} ('{op._name}') has not been audited for "
                        f"context parallelism but appears in a CP-enabled context pipeline."
                    )
            for ctx_key, n_kv_per_gpu, head_dim in (
                ("swa", d["swa_n_kv_per_gpu"], d["swa_hd"]),
                ("global", d["global_n_kv_per_gpu"], d["global_hd"]),
            ):
                if counts[ctx_key] <= 0:
                    continue
                kv_bytes_per_token = n_kv_per_gpu * head_dim * 2 * kvcache_bytes
                self.context_ops.append(
                    ops.NCCL(
                        f"context_cp_all_gather_{ctx_key}",
                        counts[ctx_key],
                        "all_gather",
                        num_elements_per_token=kv_bytes_per_token / comm_bytes,
                        num_gpus=cp,
                        comm_quant_mode=self.config.comm_quant_mode,
                    )
                )

    def _build_generation_ops(self) -> None:
        if not self._gemma4_config:
            return

        cfg = self._gemma4_config
        counts = self._count_layer_types()
        sf = self._mtp_scale_factor
        h = self._hidden_size
        tp = self.config.tp_size
        moe_tp = self.config.moe_tp_size
        moe_ep = self.config.moe_ep_size
        attn_dp = self.config.attention_dp_size
        pp = self.config.pp_size
        gemm_q = self.config.gemm_quant_mode
        kvcache_q = self.config.kvcache_quant_mode
        moe_q = self.config.moe_quant_mode
        wl_dist = (
            self.config.workload_distribution + f"_{self._power_law_alpha}"
            if self.config.workload_distribution == "power_law"
            else self.config.workload_distribution
        )
        d = self._resolve_dims(tp)

        self.generation_ops = [ops.Embedding("generation_embedding", 1 * sf, self._vocab_size, h, 0.3)]

        # --- sliding_attention layers ---
        if counts["swa"] > 0:
            c = counts["swa"] * sf
            self.generation_ops.extend(
                [
                    ops.ElementWise("generation_swa_attn_norm", c, 2 * h, 2 * h, 0.8),
                    ops.GEMM("generation_swa_qkv_gemm", c, d["swa_qkv_out"], h, gemm_q),
                    ops.GenerationAttention(
                        "generation_attention",
                        c,
                        self._num_heads // tp,
                        d["swa_n_kv_per_gpu"],
                        kvcache_q,
                        window_size=cfg.sliding_window_size,
                        head_size=d["swa_hd"],
                    ),
                    ops.GEMM("generation_swa_proj_gemm", c, h, d["swa_proj_in"], gemm_q, low_precision_input=True),
                    ops.ElementWise("generation_swa_ffn_norm", c, 2 * h, 2 * h, 0.8),
                ]
                + self._shared_mlp_ops("generation_swa", c, h, d["dense_inter_per_tp"])
                + self._moe_ops("generation_swa", c, h, moe_tp, moe_ep, attn_dp, moe_q, wl_dist)
            )

        # --- full_attention (global) layers ---
        if counts["global"] > 0:
            c = counts["global"] * sf
            self.generation_ops.extend(
                [
                    ops.ElementWise("generation_global_attn_norm", c, 2 * h, 2 * h, 0.8),
                    ops.GEMM("generation_global_qkv_gemm", c, d["global_qkv_out"], h, gemm_q),
                    ops.GenerationAttention(
                        "generation_attention",
                        c,
                        self._num_heads // tp,
                        d["global_n_kv_per_gpu"],
                        kvcache_q,
                        window_size=0,
                        head_size=d["global_hd"],
                    ),
                    ops.GEMM(
                        "generation_global_proj_gemm", c, h, d["global_proj_in"], gemm_q, low_precision_input=True
                    ),
                    ops.ElementWise("generation_global_ffn_norm", c, 2 * h, 2 * h, 0.8),
                ]
                + self._shared_mlp_ops("generation_global", c, h, d["dense_inter_per_tp"])
                + self._moe_ops("generation_global", c, h, moe_tp, moe_ep, attn_dp, moe_q, wl_dist)
            )

        self.generation_ops.extend(
            [
                ops.GEMM("generation_logits_gemm", 1 * sf, self._vocab_size // tp, h, common.GEMMQuantMode.bfloat16),
                ops.P2P("generation_p2p", (pp - 1) * sf, h, pp),
            ]
        )

    def get_kvcache_elements_per_token(self) -> int:
        """Per-token KV-cache element count (per GPU), summed over all layers.

        Both K and V tensors are stored even on global layers where K=V at the
        projection — by the time the cache is written, K has had RoPE applied and V
        has not, so they are distinct tensors.

        This count does NOT account for the SWA window cap; callers that need a
        sequence-length-aware byte count should use ``get_kvcache_bytes_per_sequence``.
        """
        if not self._gemma4_config:
            return super().get_kvcache_elements_per_token()
        cfg = self._gemma4_config
        tp = self.config.tp_size
        swa_kv_per_gpu = (cfg.swa_num_kv_heads + tp - 1) // tp
        global_kv_per_gpu = (cfg.global_num_kv_heads + tp - 1) // tp
        num_swa = cfg.layer_types.count("sliding_attention")
        num_global = cfg.layer_types.count("full_attention")
        return 2 * (num_swa * swa_kv_per_gpu * cfg.swa_head_dim + num_global * global_kv_per_gpu * cfg.global_head_dim)

    def get_kvcache_bytes_per_sequence(self, seq_len: int) -> float:
        """KV-cache bytes for one sequence on one GPU.

        SWA layers cap at ``sliding_window_size`` tokens; global layers grow with
        ``seq_len``. This per-layer-type math is what makes Gemma 4's 256K-context
        memory footprint tractable.
        """
        if not self._gemma4_config:
            return super().get_kvcache_bytes_per_sequence(seq_len)
        seq_len = max(0, seq_len)
        cfg = self._gemma4_config
        bytes_per_elem = self.config.kvcache_quant_mode.value.memory
        tp = self.config.tp_size
        swa_kv_per_gpu = (cfg.swa_num_kv_heads + tp - 1) // tp
        global_kv_per_gpu = (cfg.global_num_kv_heads + tp - 1) // tp
        num_swa = cfg.layer_types.count("sliding_attention")
        num_global = cfg.layer_types.count("full_attention")
        swa_seq = min(seq_len, cfg.sliding_window_size) if cfg.sliding_window_size > 0 else seq_len
        swa_bytes = num_swa * swa_kv_per_gpu * cfg.swa_head_dim * 2 * bytes_per_elem * swa_seq
        global_bytes = num_global * global_kv_per_gpu * cfg.global_head_dim * 2 * bytes_per_elem * seq_len
        return float(swa_bytes + global_bytes)

    def get_kvcache_max_tokens(self, kv_budget_bytes: float) -> int:
        """Capacity inverse over the window-capped KV curve (non-linear past the window)."""
        if not self._gemma4_config:
            return super().get_kvcache_max_tokens(kv_budget_bytes)
        return self._binary_search_kvcache_max_tokens(kv_budget_bytes)
