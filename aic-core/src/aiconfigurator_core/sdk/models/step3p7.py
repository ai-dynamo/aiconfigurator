# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
StepFun Step-3.7-Flash (custom): hybrid sliding-window/global attention +
dense-first-K then MoE FFN, with a single shared expert on the MoE layers, and
MTP (num_nextn_predict_layers).

This reuses the ``HybridMoEModel`` op pipeline (which already models a mixed
attention pattern -- SWA vs global -- and a mixed FFN pattern -- dense vs MoE --
driven by ``HybridMoEConfig``) and adds the two things that class does NOT model
but Step-3.7 needs:

1. **Hybrid sliding-window KV cache.** ``HybridMoEModel`` inherits ``BaseModel``'s
   *linear* KV growth (every layer grows with seq_len), which over-counts KV for a
   short-window (512-token) model. Step-3.7 caps the SWA layers at the window while
   the global layers keep growing -- the KV feature that makes this arch cheap.
   We override the three KV methods (mirroring ``Gemma4MixModel``) so the memory
   model, capacity/max-tokens, and batch sizing all see the window-capped curve.

2. **Shared expert.** Each MoE layer runs one always-on shared expert (dense
   SwiGLU at ``share_expert_dim``) in parallel with the routed top-k experts. We
   append its FFN GEMMs to the context/generation op lists.
"""

from __future__ import annotations

from aiconfigurator_core.sdk.models.base import register_model
from aiconfigurator_core.sdk.models.hybrid_moe import HybridMoEModel


@register_model("STEP3P7")
class Step3p7Model(HybridMoEModel):
    """Step-3.7-Flash: HybridMoE pipeline + SWA KV capping + shared expert."""

    @classmethod
    def create(cls, model_info: dict, model_config, backend_name: str):
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
        raw_config = model_info.get("raw_config", {}) or {}
        model._share_expert_dim = int(raw_config.get("share_expert_dim", 0) or 0)
        model.set_hybrid_config(model_info["extra_params"])
        return model

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Set by create() before set_hybrid_config(); default 0 = no shared expert.
        self._share_expert_dim: int = 0

    # ------------------------------------------------------------------ #
    # Shared expert: one always-on dense SwiGLU per MoE layer.
    # ------------------------------------------------------------------ #
    def set_hybrid_config(self, cfg) -> None:
        super().set_hybrid_config(cfg)
        self._append_shared_expert_ops()

    def _append_shared_expert_ops(self) -> None:
        """Append the shared-expert dense FFN to context + generation op lists.

        Runs on every MoE layer (``moe_layer_freq == 1``). Aggregate op time is a
        sum, so appending (rather than inserting mid-block) does not change the
        modeled latency/throughput.

        These ops are appended after ``_build_context_ops`` has already wired CP,
        so they must be audited explicitly. Without that they keep
        ``_seq_split=1`` while the inherited dense FFN ops carry ``cp``, charging
        the full shared-expert token count on every rank -- and they also miss the
        un-audited-op guard, so it shows up as a wrong number, not an error.
        """
        if self._share_expert_dim <= 0:
            return
        cfg = self._hybrid_config
        num_moe_layers = sum(cfg.moe_layer_freq)
        if num_moe_layers <= 0:
            return
        h = self._hidden_size
        tp = self.config.tp_size
        gemm_q = self.config.gemm_quant_mode
        share_inter_per_tp = self._share_expert_dim // tp

        context_shared_ops = self._dense_ffn_ops(
            "context_shared_expert", num_moe_layers, h, tp, share_inter_per_tp, gemm_q
        )
        if self.config.cp_size > 1:
            self.apply_cp_to_context_ops(context_shared_ops, self.config.cp_size)
        self.context_ops.extend(context_shared_ops)
        sf = self._mtp_scale_factor
        self.generation_ops.extend(
            self._dense_ffn_ops("generation_shared_expert", num_moe_layers * sf, h, tp, share_inter_per_tp, gemm_q)
        )

    # ------------------------------------------------------------------ #
    # Hybrid sliding-window KV cache (SWA layers cap at the window; global
    # layers keep growing). Step-3.7 uses the same KV head geometry
    # (num_kv_heads, head_dim) on both attention types.
    # ------------------------------------------------------------------ #
    def _swa_global_counts(self) -> tuple[int, int]:
        cfg = self._hybrid_config
        num_global = sum(cfg.attn_layer_pattern)  # 1 = global/full attention
        num_swa = cfg.attn_layer_pattern.count(0)  # 0 = sliding-window attention
        return num_swa, num_global

    def _kv_per_layer_per_token(self) -> float:
        """Per-GPU KV bytes for ONE layer at ONE token (K + V)."""
        tp = self.config.tp_size
        kv_heads_per_gpu = (self._num_kv_heads + tp - 1) // tp
        bytes_per_elem = self.config.kvcache_quant_mode.value.memory
        return 2 * kv_heads_per_gpu * self._head_size * bytes_per_elem

    def get_kvcache_elements_per_token(self) -> int:
        """Uncapped per-token KV element count (per GPU), summed over all layers."""
        if not self._hybrid_config:
            return super().get_kvcache_elements_per_token()
        tp = self.config.tp_size
        kv_heads_per_gpu = (self._num_kv_heads + tp - 1) // tp
        return 2 * kv_heads_per_gpu * self._head_size * self._num_layers

    def get_kvcache_bytes_per_sequence(self, seq_len: int) -> float:
        """KV bytes for one sequence on one GPU, window-capping the SWA layers."""
        if not self._hybrid_config:
            return super().get_kvcache_bytes_per_sequence(seq_len)
        seq_len = max(0, seq_len)
        cfg = self._hybrid_config
        window = cfg.sliding_window_size
        num_swa, num_global = self._swa_global_counts()
        per = self._kv_per_layer_per_token()
        swa_seq = min(seq_len, window) if window > 0 else seq_len
        return float(num_swa * per * swa_seq + num_global * per * seq_len)

    def get_kvcache_max_tokens(self, kv_budget_bytes: float) -> int:
        """Capacity inverse over the window-capped (piecewise) KV curve."""
        if not self._hybrid_config:
            return super().get_kvcache_max_tokens(kv_budget_bytes)
        return self._binary_search_kvcache_max_tokens(kv_budget_bytes)
