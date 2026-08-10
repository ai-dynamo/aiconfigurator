# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import aiconfigurator_core.sdk.operations as ops
from aiconfigurator_core.sdk import common
from aiconfigurator_core.sdk.models.base import BaseModel, register_model
from aiconfigurator_core.sdk.models.helpers import mtp_scale_factor


@register_model("MUSEGLIMMER")
class MuseGlimmerModel(BaseModel):
    """
    Meta Muse Glimmer (muse_glimmer_text): dense hybrid SWA/global attention.

    Two layer-type recipes are emitted, driven by ``MuseGlimmerConfig.layer_types``:

    - **sliding_attention (SWA)**: standard GQA at the model-wide head geometry,
      token window = ``sliding_window_size``.
    - **full_attention (global)**: same geometry, no window.

    Unlike Gemma 4 there is no per-layer-type head geometry, no
    ``attention_k_eq_v``, and no routed-MoE branch: every layer runs one gated
    SwiGLU dense MLP at ``inter_size``.

    Deliberately unmodeled (shape-neutral for op-level perf): NoPE on global
    layers (layer_rope_theta = 0), final logit softcapping, qk_scale_factor,
    output_multiplier. The vision tower (muse_glimmer_vision) is not priced —
    text-serving convention shared with Kimi-K2.5/K3, Llama-4, and Qwen3.5.
    """

    @classmethod
    def supports_cp(cls, backend_name: str) -> bool:
        # Dense SWA/global GQA prefill CP: SGLang AllGather (zigzag FMHA).
        return backend_name == "sglang"

    @classmethod
    def create(cls, model_info: dict, model_config, backend_name: str) -> BaseModel:
        model = cls(
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
        model.set_muse_glimmer_config(model_info["extra_params"])
        return model

    def __init__(self, *args) -> None:
        super().__init__(*args)
        self._mtp_scale_factor = mtp_scale_factor(self._nextn, self._num_layers)
        self._muse_glimmer_config: common.MuseGlimmerConfig | None = None

    def set_muse_glimmer_config(self, cfg: common.MuseGlimmerConfig) -> None:
        """Apply MuseGlimmerConfig and rebuild context/generation ops.

        Validates that ``layer_types`` length matches ``num_layers`` and contains
        only recognized values before accepting the config.
        """
        if cfg is None or not isinstance(cfg, common.MuseGlimmerConfig):
            raise ValueError(f"MuseGlimmerModel requires a MuseGlimmerConfig, got {type(cfg).__name__}")
        if len(cfg.layer_types) != self._num_layers:
            raise ValueError(
                f"MuseGlimmerConfig.layer_types length ({len(cfg.layer_types)}) "
                f"does not match num_layers ({self._num_layers})"
            )
        for i, lt in enumerate(cfg.layer_types):
            if lt not in ("sliding_attention", "full_attention"):
                raise ValueError(f"MuseGlimmerConfig layer {i} has invalid type {lt!r}")
        if cfg.sliding_window_size <= 0:
            raise ValueError(
                f"MuseGlimmerConfig requires a positive sliding_window_size, got {cfg.sliding_window_size}"
            )
        self._muse_glimmer_config = cfg
        self._build_context_ops()
        self._build_generation_ops()

    def _count_layer_types(self) -> dict[str, int]:
        cfg = self._muse_glimmer_config
        return {
            "swa": cfg.layer_types.count("sliding_attention"),
            "global": cfg.layer_types.count("full_attention"),
        }

    def _resolve_dims(self, tp_size: int) -> dict:
        """Per-TP-shard attention dims and dense-MLP intermediate (uniform geometry)."""
        n_kv_per_gpu = (self._num_kv_heads + tp_size - 1) // tp_size
        return {
            "n_kv_per_gpu": n_kv_per_gpu,
            "qkv_out": self._num_heads * self._head_size // tp_size + n_kv_per_gpu * self._head_size * 2,
            "proj_in": self._num_heads * self._head_size // tp_size,
            "inter_per_tp": self._inter_size // tp_size,
        }

    def _shared_mlp_ops(self, prefix: str, count: float, h: int, inter_per_tp: int) -> list:
        """Dense gated SwiGLU MLP; runs on every layer."""
        gemm_q = self.config.gemm_quant_mode
        return [
            ops.GEMM(f"{prefix}_mlp_gate_up_gemm", count, 2 * inter_per_tp, h, gemm_q),
            ops.ElementWise(f"{prefix}_mlp_act", count, 2 * inter_per_tp, inter_per_tp, 0.8),
            ops.GEMM(f"{prefix}_mlp_down_gemm", count, h, inter_per_tp, gemm_q, low_precision_input=True),
        ]

    def _attention_block(self, phase: str, kind: str, count: float, window_size: int, d: dict) -> list:
        """One layer-type recipe: norm -> qkv -> attention -> proj -> norm + MLP."""
        h = self._hidden_size
        tp = self.config.tp_size
        gemm_q = self.config.gemm_quant_mode
        kvcache_q = self.config.kvcache_quant_mode
        if phase == "context":
            attention = ops.ContextAttention(
                "context_attention",
                count,
                self._num_heads // tp,
                d["n_kv_per_gpu"],
                kvcache_q,
                self.config.fmha_quant_mode,
                window_size=window_size,
                head_size=self._head_size,
            )
        else:
            attention = ops.GenerationAttention(
                "generation_attention",
                count,
                self._num_heads // tp,
                d["n_kv_per_gpu"],
                kvcache_q,
                window_size=window_size,
                head_size=self._head_size,
            )
        return [
            ops.ElementWise(f"{phase}_{kind}_attn_norm", count, 2 * h, 2 * h, 0.8),
            ops.GEMM(f"{phase}_{kind}_qkv_gemm", count, d["qkv_out"], h, gemm_q),
            attention,
            ops.GEMM(f"{phase}_{kind}_proj_gemm", count, h, d["proj_in"], gemm_q, low_precision_input=True),
            ops.ElementWise(f"{phase}_{kind}_ffn_norm", count, 2 * h, 2 * h, 0.8),
        ] + self._shared_mlp_ops(f"{phase}_{kind}", count, h, d["inter_per_tp"])

    def _build_context_ops(self) -> None:
        if not self._muse_glimmer_config:
            return
        cfg = self._muse_glimmer_config
        counts = self._count_layer_types()
        h = self._hidden_size
        tp = self.config.tp_size
        pp = self.config.pp_size
        d = self._resolve_dims(tp)

        self.context_ops = [ops.Embedding("context_embedding", 1, self._vocab_size, h, 0.3)]
        if counts["swa"] > 0:
            self.context_ops.extend(self._attention_block("context", "swa", counts["swa"], cfg.sliding_window_size, d))
        if counts["global"] > 0:
            self.context_ops.extend(self._attention_block("context", "global", counts["global"], 0, d))
        self.context_ops.extend(
            [
                ops.GEMM("context_logits_gemm", 1, self._vocab_size // tp, h, common.GEMMQuantMode.bfloat16),
                ops.P2P("context_p2p", pp - 1, h, pp),
            ]
        )

        # cp (SGLang prefill AllGather CP). KV bytes per token are uniform across
        # layer types (same n_kv/head_dim), so a single all_gather weighted by the
        # total layer count matches the runtime comm volume; the sliding window only
        # caps the stored KV (get_kvcache_bytes_per_sequence), not per-layer comm.
        if self.config.cp_size > 1:
            cp = self.config.cp_size
            kvcache_bytes = self.config.kvcache_quant_mode.value.memory
            comm_bytes = self.config.comm_quant_mode.value.memory
            for op in self.context_ops:
                if isinstance(op, ops.ContextAttention):
                    op._cp_size = cp
                elif op._CP_AWARE:
                    op._seq_split = cp
                else:
                    raise NotImplementedError(
                        f"{type(op).__name__} ('{op._name}') has not been audited for "
                        f"context parallelism but appears in a CP-enabled context pipeline."
                    )
            kv_bytes_per_token = d["n_kv_per_gpu"] * self._head_size * 2 * kvcache_bytes
            self.context_ops.append(
                ops.NCCL(
                    "context_cp_all_gather",
                    self._num_layers,
                    "all_gather",
                    num_elements_per_token=kv_bytes_per_token / comm_bytes,
                    num_gpus=cp,
                    comm_quant_mode=self.config.comm_quant_mode,
                )
            )

    def _build_generation_ops(self) -> None:
        if not self._muse_glimmer_config:
            return
        cfg = self._muse_glimmer_config
        counts = self._count_layer_types()
        sf = self._mtp_scale_factor
        h = self._hidden_size
        tp = self.config.tp_size
        pp = self.config.pp_size
        d = self._resolve_dims(tp)

        self.generation_ops = [ops.Embedding("generation_embedding", 1 * sf, self._vocab_size, h, 0.3)]
        if counts["swa"] > 0:
            self.generation_ops.extend(
                self._attention_block("generation", "swa", counts["swa"] * sf, cfg.sliding_window_size, d)
            )
        if counts["global"] > 0:
            self.generation_ops.extend(self._attention_block("generation", "global", counts["global"] * sf, 0, d))
        self.generation_ops.extend(
            [
                ops.GEMM("generation_logits_gemm", 1 * sf, self._vocab_size // tp, h, common.GEMMQuantMode.bfloat16),
                ops.P2P("generation_p2p", (pp - 1) * sf, h, pp),
            ]
        )

    def get_kvcache_elements_per_token(self) -> int:
        """Per-token KV elements per GPU over all layers (no window cap — see
        ``get_kvcache_bytes_per_sequence`` for the sequence-aware count)."""
        if not self._muse_glimmer_config:
            return super().get_kvcache_elements_per_token()
        tp = self.config.tp_size
        n_kv_per_gpu = (self._num_kv_heads + tp - 1) // tp
        return 2 * self._num_layers * n_kv_per_gpu * self._head_size

    def get_kvcache_bytes_per_sequence(self, seq_len: int) -> float:
        """KV bytes for one sequence on one GPU: SWA layers cap at the window,
        global layers grow with ``seq_len``."""
        if not self._muse_glimmer_config:
            return super().get_kvcache_bytes_per_sequence(seq_len)
        seq_len = max(0, seq_len)
        cfg = self._muse_glimmer_config
        bytes_per_elem = self.config.kvcache_quant_mode.value.memory
        tp = self.config.tp_size
        n_kv_per_gpu = (self._num_kv_heads + tp - 1) // tp
        counts = self._count_layer_types()
        per_token_layer = n_kv_per_gpu * self._head_size * 2 * bytes_per_elem
        swa_seq = min(seq_len, cfg.sliding_window_size)
        return float(counts["swa"] * per_token_layer * swa_seq + counts["global"] * per_token_layer * seq_len)

    def get_kvcache_max_tokens(self, kv_budget_bytes: float) -> int:
        """Capacity inverse over the window-capped KV curve (non-linear past the window)."""
        if not self._muse_glimmer_config:
            return super().get_kvcache_max_tokens(kv_budget_bytes)
        return self._binary_search_kvcache_max_tokens(kv_budget_bytes)
