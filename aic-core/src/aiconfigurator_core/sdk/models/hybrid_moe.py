# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import aiconfigurator_core.sdk.operations as ops
from aiconfigurator_core.sdk import common
from aiconfigurator_core.sdk.models.base import BaseModel, register_model
from aiconfigurator_core.sdk.models.blocks.moe import MoEBlockShape, build_moe_block_ops
from aiconfigurator_core.sdk.models.helpers import mtp_scale_factor
from aiconfigurator_core.sdk.utils import _load_model_config_from_model_path


@register_model("HYBRIDMOE")
class HybridMoEModel(BaseModel):
    """
    Hybrid attention + mixed FFN model (MiMo-V2-Flash, Llama 4 Scout/Maverick, and similar).

    Handles four layer types derived from HybridMoEConfig.attn_layer_pattern and moe_layer_freq:
    - global_moe:  global (full) attention + MoE FFN
    - swa_moe:     SWA/local attention + MoE FFN
    - swa_dense:   SWA/local attention + dense SwiGLU FFN
    - global_dense: global attention + dense SwiGLU FFN (rare but supported)

    SWA/local attention dims fall back to model-level defaults when HybridMoEConfig fields are 0.
    This lets same-dim models (Llama 4) and different-dim models (MiMo-V2-Flash) share one class.
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
            backend_name=backend_name,
        )
        model.set_hybrid_config(model_info["extra_params"])
        return model

    def __init__(self, topk: int, num_experts: int, moe_inter_size: int, *args, backend_name: str = "") -> None:
        super().__init__(*args)
        self._backend_name = backend_name
        # Fused-only family: the MoE-block builder keys its large-EP emission
        # off cfg.moe_comm_backend, but this family resolves no node width
        # (large_ep_gpus_per_node) — a comm backend here would silently
        # mis-price cross-node all-to-all with the builder's default. Loud,
        # not silent (same principle as the num_gpus_per_node raise).
        if getattr(self.config, "moe_comm_backend", None):
            raise ValueError(
                "large-EP is not wired for the HYBRIDMOE family yet — moe_comm_backend "
                "must not be set; see models/README blocks/ section"
            )
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
        self._mtp_scale_factor = mtp_scale_factor(self._nextn, self._num_layers)
        self._validate_fp8_block_quantized_moe_config()
        self._hybrid_config: common.HybridMoEConfig | None = None
        self._power_law_alpha = 1.01

    def _validate_fp8_block_quantized_moe_config(self) -> None:
        """Validate fp8_block MoE alignment: (moe_inter_size / moe_tp_size) % block_size == 0."""
        if self.config.moe_quant_mode != common.MoEQuantMode.fp8_block:
            return
        raw_config = _load_model_config_from_model_path(self.model_path)
        default_size = [128, 128]
        weight_block_size = raw_config.get("quantization_config", {}).get("weight_block_size", default_size)[0]
        moe_size_per_gpu = self._moe_inter_size // self.config.moe_tp_size
        if (moe_size_per_gpu % weight_block_size) != 0:
            raise ValueError(
                f"Invalid quantized MoE configuration: "
                f"(moe_intermediate_size={self._moe_inter_size} / moe_tp_size={self.config.moe_tp_size}) "
                f"% weight_block_size={weight_block_size} != 0. "
            )

    def set_hybrid_config(self, cfg: common.HybridMoEConfig) -> None:
        """Apply HybridMoEConfig and rebuild context/generation ops.

        Validates that attn_layer_pattern and moe_layer_freq have the same length,
        match self._num_layers, and contain only 0/1 values before accepting the config.
        """
        n = len(cfg.attn_layer_pattern)
        if n != len(cfg.moe_layer_freq):
            raise ValueError(
                f"HybridMoEConfig pattern length mismatch: "
                f"attn_layer_pattern has {n} entries "
                f"but moe_layer_freq has {len(cfg.moe_layer_freq)}"
            )
        if n != self._num_layers:
            raise ValueError(f"HybridMoEConfig pattern length ({n}) does not match num_layers ({self._num_layers})")
        for i, (a, m) in enumerate(zip(cfg.attn_layer_pattern, cfg.moe_layer_freq, strict=True)):
            if a not in (0, 1) or m not in (0, 1):
                raise ValueError(f"HybridMoEConfig layer {i} has invalid values: attn={a}, moe={m} (expected 0 or 1)")
        self._hybrid_config = cfg
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
        """Count layers per type: global_moe, swa_moe, swa_dense, global_dense."""
        cfg = self._hybrid_config
        counts: dict[str, int] = {"global_moe": 0, "swa_moe": 0, "swa_dense": 0, "global_dense": 0}
        for attn, moe in zip(cfg.attn_layer_pattern, cfg.moe_layer_freq, strict=True):
            if attn == 1 and moe == 1:
                counts["global_moe"] += 1
            elif attn == 0 and moe == 1:
                counts["swa_moe"] += 1
            elif attn == 0 and moe == 0:
                counts["swa_dense"] += 1
            else:
                counts["global_dense"] += 1
        return counts

    def _resolve_dims(self, tp_size: int) -> dict:
        """Resolve SWA/local attention dims, falling back to model-level defaults when 0.

        Returns a dict with per-TP KV head counts, QKV GEMM output widths, proj GEMM input widths,
        Q/K head dims for attention kernels, and dense FFN intermediate size per TP.
        """
        cfg = self._hybrid_config
        swa_n_q = cfg.swa_num_heads if cfg.swa_num_heads > 0 else self._num_heads
        swa_n_kv = cfg.swa_num_kv_heads if cfg.swa_num_kv_heads > 0 else self._num_kv_heads
        swa_hd = cfg.swa_head_dim if cfg.swa_head_dim > 0 else self._head_size
        swa_v_hd = cfg.swa_v_head_dim if cfg.swa_v_head_dim > 0 else self._head_size
        global_v_hd = cfg.global_v_head_dim if cfg.global_v_head_dim > 0 else self._head_size
        swa_n_kv_per_gpu = (swa_n_kv + tp_size - 1) // tp_size
        global_n_kv_per_gpu = (self._num_kv_heads + tp_size - 1) // tp_size
        dense_inter = cfg.dense_inter_size if cfg.dense_inter_size > 0 else self._inter_size
        return {
            "swa_n_kv_per_gpu": swa_n_kv_per_gpu,
            "global_n_kv_per_gpu": global_n_kv_per_gpu,
            "swa_qkv_out": swa_n_q * swa_hd // tp_size + swa_n_kv_per_gpu * (swa_hd + swa_v_hd),
            "global_qkv_out": self._num_heads * self._head_size // tp_size
            + global_n_kv_per_gpu * (self._head_size + global_v_hd),
            "swa_proj_in": swa_n_q * swa_v_hd // tp_size,
            "global_proj_in": self._num_heads * global_v_hd // tp_size,
            "swa_n_q": swa_n_q,
            "swa_hd": swa_hd,
            "global_hd": self._head_size,
            "swa_v_hd": swa_v_hd,
            "global_v_hd": global_v_hd,
            "dense_inter_per_tp": dense_inter // tp_size,
        }

    def _moe_ops(
        self,
        prefix: str,
        count: float,
        h: int,
        moe_q: common.MoEQuantMode,
        wl_dist: str,
    ) -> list:
        """Return the MoE FFN block (router, pre-dispatch, compute, post-dispatch).

        Emitted through the generic MoE-block builder. The router GEMM is now
        always emitted (spec section 4.4.4); the legacy span gated it on
        ``num_experts >= 128``, so <128-expert models (Llama-4 Scout class)
        gain one router GEMM per MoE layer type — the ledgered A8 delta.

        The builder emits phase-canonical op names ("context_*"), so this
        family's layer-type prefix ("context_global", "generation_swa", ...)
        is restored on the returned ops. CP wiring stays post-hoc: the loops
        in ``set_hybrid_config`` / ``_build_context_ops`` mutate the returned
        ops exactly as they mutated the hand-wired ones.
        """
        phase = "context" if prefix.startswith("context") else "generation"
        shape = MoEBlockShape(
            hidden_size=h,
            moe_inter_size=self._moe_inter_size,
            topk=self._topk,
            num_experts=self._num_experts,
            num_shared_experts=0,  # hybrid-family checkpoints model none
            # Descriptor-only: the builder scales by the caller-supplied
            # per-layer-type ``count``.
            num_moe_layers=sum(self._hybrid_config.moe_layer_freq),
        )
        block_ops = build_moe_block_ops(
            phase,
            shape,
            self.config,
            moe_q,
            wl_dist,
            scale_factor=count,
            backend_name=self._backend_name,
            inference_phase=phase,
            model_family=self.model_family,
            # Legacy fidelity: this family's dispatch ops never carried a
            # quant_mode, while the builder's fused path forwards it by
            # default (a quant-aware trtllm SM100 dispatch-volume model).
            # Adopting quant-aware dispatch here is a deliberate modeling
            # change for a follow-up, not a rewire side effect.
            dispatch_quant_mode=None,
        )
        for op in block_ops:
            assert op._name.startswith(phase), (
                f"MoE-block builder returned op {op._name!r} that does not start with its phase "
                f"{phase!r}; the layer-type rename below would silently mangle it"
            )
            op._name = prefix + op._name[len(phase) :]
        return block_ops

    def _dense_ffn_ops(
        self, prefix: str, count: float, h: int, tp: int, dense_inter_per_tp: int, gemm_q: common.GEMMQuantMode
    ) -> list:
        """Return fused gate_up + activation + down ops for dense SwiGLU FFN."""
        return [
            ops.GEMM(f"{prefix}_dense_gate_up_gemm", count, 2 * dense_inter_per_tp, h, gemm_q),
            ops.ElementWise(f"{prefix}_dense_act", count, 2 * dense_inter_per_tp, dense_inter_per_tp, 0.8),
            ops.GEMM(f"{prefix}_dense_down_gemm", count, h, dense_inter_per_tp, gemm_q, low_precision_input=True),
        ]

    def apply_cp_to_context_ops(self, op_list, cp: int) -> None:
        """Wire context parallelism into ``op_list``.

        Subclasses that append context ops after ``_build_context_ops`` has run
        must call this on the ops they added; otherwise those ops silently keep
        ``_seq_split=1`` while the rest of the pipeline is split, AND they skip
        the un-audited-op guard below, so the miss surfaces as a wrong number
        rather than an error.
        """
        for op in op_list:
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

    def _attn_gate_ops(self, prefix: str, c: int, h: int, n_q_per_gpu: int, proj_in: int, gemm_q) -> list:
        """Head-wise attention gate, or ``[]`` when the model doesn't use one.

        Step-3.7 sets ``use_head_wise_attn_gate``: ``g_proj`` projects
        hidden_size -> num_attention_heads and its sigmoid scales each head's
        attention output before ``o_proj``. The gate width follows the layer's
        own head count, so sliding layers (96 heads) carry a wider gate than
        global ones (64).
        """
        if not (self._hybrid_config and self._hybrid_config.use_head_wise_attn_gate):
            return []
        return [
            ops.GEMM(f"{prefix}_attn_gate_gemm", c, n_q_per_gpu, h, gemm_q),
            # Reads the attention output plus the per-head gate, writes the
            # attention output back; one scalar per head, not per element.
            ops.ElementWise(f"{prefix}_attn_gate", c, proj_in + n_q_per_gpu, proj_in, 0.8),
        ]

    def _build_context_ops(self) -> None:
        """Build the context (prefill) operations for all four layer types."""
        if not self._hybrid_config:
            return

        cfg = self._hybrid_config
        counts = self._count_layer_types()
        h = self._hidden_size
        tp = self.config.tp_size
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

        # --- global attention + MoE FFN ---
        if counts["global_moe"] > 0:
            c = counts["global_moe"]
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
                        use_qk_norm=cfg.use_qk_norm,
                    ),
                    *self._attn_gate_ops("context_global", c, h, self._num_heads // tp, d["global_proj_in"], gemm_q),
                    ops.GEMM("context_global_proj_gemm", c, h, d["global_proj_in"], gemm_q, low_precision_input=True),
                    ops.ElementWise("context_global_moe_norm", c, 2 * h, 2 * h, 0.8),
                ]
                + self._moe_ops("context_global", c, h, moe_q, wl_dist)
            )

        # --- SWA/local attention + MoE FFN ---
        if counts["swa_moe"] > 0:
            c = counts["swa_moe"]
            self.context_ops.extend(
                [
                    ops.ElementWise("context_swa_attn_norm", c, 2 * h, 2 * h, 0.8),
                    ops.GEMM("context_swa_qkv_gemm", c, d["swa_qkv_out"], h, gemm_q),
                    ops.ContextAttention(
                        "context_attention",
                        c,
                        d["swa_n_q"] // tp,
                        d["swa_n_kv_per_gpu"],
                        kvcache_q,
                        fmha_q,
                        window_size=cfg.sliding_window_size,
                        head_size=d["swa_hd"],
                        use_qk_norm=cfg.use_qk_norm,
                    ),
                    *self._attn_gate_ops("context_swa", c, h, d["swa_n_q"] // tp, d["swa_proj_in"], gemm_q),
                    ops.GEMM("context_swa_proj_gemm", c, h, d["swa_proj_in"], gemm_q, low_precision_input=True),
                    ops.ElementWise("context_swa_moe_norm", c, 2 * h, 2 * h, 0.8),
                ]
                + self._moe_ops("context_swa", c, h, moe_q, wl_dist)
            )

        # --- SWA/local attention + dense FFN ---
        if counts["swa_dense"] > 0:
            c = counts["swa_dense"]
            self.context_ops.extend(
                [
                    ops.ElementWise("context_swa_dense_attn_norm", c, 2 * h, 2 * h, 0.8),
                    ops.GEMM("context_swa_dense_qkv_gemm", c, d["swa_qkv_out"], h, gemm_q),
                    ops.ContextAttention(
                        "context_attention",
                        c,
                        d["swa_n_q"] // tp,
                        d["swa_n_kv_per_gpu"],
                        kvcache_q,
                        fmha_q,
                        window_size=cfg.sliding_window_size,
                        head_size=d["swa_hd"],
                        use_qk_norm=cfg.use_qk_norm,
                    ),
                    *self._attn_gate_ops("context_swa_dense", c, h, d["swa_n_q"] // tp, d["swa_proj_in"], gemm_q),
                    ops.GEMM("context_swa_dense_proj_gemm", c, h, d["swa_proj_in"], gemm_q, low_precision_input=True),
                    ops.ElementWise("context_swa_dense_ffn_norm", c, 2 * h, 2 * h, 0.8),
                ]
                + self._dense_ffn_ops("context_swa", c, h, tp, d["dense_inter_per_tp"], gemm_q)
            )

        # --- global attention + dense FFN ---
        if counts["global_dense"] > 0:
            c = counts["global_dense"]
            self.context_ops.extend(
                [
                    ops.ElementWise("context_global_dense_attn_norm", c, 2 * h, 2 * h, 0.8),
                    ops.GEMM("context_global_dense_qkv_gemm", c, d["global_qkv_out"], h, gemm_q),
                    ops.ContextAttention(
                        "context_attention",
                        c,
                        self._num_heads // tp,
                        d["global_n_kv_per_gpu"],
                        kvcache_q,
                        fmha_q,
                        window_size=0,
                        head_size=d["global_hd"],
                        use_qk_norm=cfg.use_qk_norm,
                    ),
                    *self._attn_gate_ops(
                        "context_global_dense", c, h, self._num_heads // tp, d["global_proj_in"], gemm_q
                    ),
                    ops.GEMM(
                        "context_global_dense_proj_gemm", c, h, d["global_proj_in"], gemm_q, low_precision_input=True
                    ),
                    ops.ElementWise("context_global_dense_ffn_norm", c, 2 * h, 2 * h, 0.8),
                ]
                + self._dense_ffn_ops("context_global", c, h, tp, d["dense_inter_per_tp"], gemm_q)
            )

        self.context_ops.extend(
            [
                ops.GEMM("context_logits_gemm", 1, self._vocab_size // tp, h, common.GEMMQuantMode.bfloat16),
                ops.P2P("context_p2p", pp - 1, h, pp),
            ]
        )

        # cp (SGLang prefill AllGather CP). Heterogeneous KV per layer-type
        # (global vs SWA, and K/V head dims can differ for MiMo-V2-Flash) ->
        # emit one NCCL all_gather per type, weighted by its layer count.
        # Dense FMHA uses zigzag (``cp_size`` on the attention op, balanced
        # full/cp work); token-major ops shrink the M-axis via ``seq_split``.
        # Bypasses the BaseModel CP helper (one uniform per-token KV size).
        # NOTE: the SWA all_gather is sized by the full new-token count (not the
        # window) on purpose -- matches sglang v0.5.13
        # ``cp_allgather_and_save_kv_cache``, which gathers the FULL per-layer
        # new-token KV across CP ranks; the sliding window only caps stored KV,
        # not this per-layer comm volume.
        if self.config.cp_size > 1:
            cp = self.config.cp_size
            kvcache_bytes = self.config.kvcache_quant_mode.value.memory
            comm_bytes = self.config.comm_quant_mode.value.memory
            # Post-construction CP wiring (not the __init__ _CP_AWARE gate): this
            # family has heterogeneous layer types (SWA vs global / dense vs MoE)
            # built across separate passes, so CP is applied here once every op
            # exists. The per-op _CP_AWARE opt-in is re-asserted in the loop so an
            # un-audited op still fails loud instead of silently skipping CP.
            self.apply_cp_to_context_ops(self.context_ops, cp)
            global_layers = counts.get("global_moe", 0) + counts.get("global_dense", 0)
            swa_layers = counts.get("swa_moe", 0) + counts.get("swa_dense", 0)
            # Per-layer KV bytes = n_kv * (k_hd + v_hd) * bytes (K and V head
            # dims can differ for Hybrid configs), not the n_kv*2*head shortcut.
            if global_layers > 0:
                kv_bytes_per_token = d["global_n_kv_per_gpu"] * (d["global_hd"] + d["global_v_hd"]) * kvcache_bytes
                self.context_ops.append(
                    ops.NCCL(
                        "context_cp_all_gather_global",
                        global_layers,
                        "all_gather",
                        num_elements_per_token=kv_bytes_per_token / comm_bytes,
                        num_gpus=cp,
                        comm_quant_mode=self.config.comm_quant_mode,
                    )
                )
            if swa_layers > 0:
                kv_bytes_per_token = d["swa_n_kv_per_gpu"] * (d["swa_hd"] + d["swa_v_hd"]) * kvcache_bytes
                self.context_ops.append(
                    ops.NCCL(
                        "context_cp_all_gather_swa",
                        swa_layers,
                        "all_gather",
                        num_elements_per_token=kv_bytes_per_token / comm_bytes,
                        num_gpus=cp,
                        comm_quant_mode=self.config.comm_quant_mode,
                    )
                )

    def _build_generation_ops(self) -> None:
        """Build the generation (decoding) operations for all four layer types.

        All generation op counts are scaled by _mtp_scale_factor to account for
        multi-token prediction (nextn > 0), mirroring MOEModel's behavior.
        """
        if not self._hybrid_config:
            return

        cfg = self._hybrid_config
        counts = self._count_layer_types()
        sf = self._mtp_scale_factor
        h = self._hidden_size
        tp = self.config.tp_size
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

        # --- global attention + MoE FFN ---
        if counts["global_moe"] > 0:
            c = counts["global_moe"] * sf
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
                        use_qk_norm=cfg.use_qk_norm,
                    ),
                    *self._attn_gate_ops("generation_global", c, h, self._num_heads // tp, d["global_proj_in"], gemm_q),
                    ops.GEMM(
                        "generation_global_proj_gemm", c, h, d["global_proj_in"], gemm_q, low_precision_input=True
                    ),
                    ops.ElementWise("generation_global_moe_norm", c, 2 * h, 2 * h, 0.8),
                ]
                + self._moe_ops("generation_global", c, h, moe_q, wl_dist)
            )

        # --- SWA/local attention + MoE FFN ---
        if counts["swa_moe"] > 0:
            c = counts["swa_moe"] * sf
            self.generation_ops.extend(
                [
                    ops.ElementWise("generation_swa_attn_norm", c, 2 * h, 2 * h, 0.8),
                    ops.GEMM("generation_swa_qkv_gemm", c, d["swa_qkv_out"], h, gemm_q),
                    ops.GenerationAttention(
                        "generation_attention",
                        c,
                        d["swa_n_q"] // tp,
                        d["swa_n_kv_per_gpu"],
                        kvcache_q,
                        window_size=cfg.sliding_window_size,
                        head_size=d["swa_hd"],
                        use_qk_norm=cfg.use_qk_norm,
                    ),
                    *self._attn_gate_ops("generation_swa", c, h, d["swa_n_q"] // tp, d["swa_proj_in"], gemm_q),
                    ops.GEMM("generation_swa_proj_gemm", c, h, d["swa_proj_in"], gemm_q, low_precision_input=True),
                    ops.ElementWise("generation_swa_moe_norm", c, 2 * h, 2 * h, 0.8),
                ]
                + self._moe_ops("generation_swa", c, h, moe_q, wl_dist)
            )

        # --- SWA/local attention + dense FFN ---
        if counts["swa_dense"] > 0:
            c = counts["swa_dense"] * sf
            self.generation_ops.extend(
                [
                    ops.ElementWise("generation_swa_dense_attn_norm", c, 2 * h, 2 * h, 0.8),
                    ops.GEMM("generation_swa_dense_qkv_gemm", c, d["swa_qkv_out"], h, gemm_q),
                    ops.GenerationAttention(
                        "generation_attention",
                        c,
                        d["swa_n_q"] // tp,
                        d["swa_n_kv_per_gpu"],
                        kvcache_q,
                        window_size=cfg.sliding_window_size,
                        head_size=d["swa_hd"],
                        use_qk_norm=cfg.use_qk_norm,
                    ),
                    *self._attn_gate_ops("generation_swa_dense", c, h, d["swa_n_q"] // tp, d["swa_proj_in"], gemm_q),
                    ops.GEMM(
                        "generation_swa_dense_proj_gemm", c, h, d["swa_proj_in"], gemm_q, low_precision_input=True
                    ),
                    ops.ElementWise("generation_swa_dense_ffn_norm", c, 2 * h, 2 * h, 0.8),
                ]
                + self._dense_ffn_ops("generation_swa", c, h, tp, d["dense_inter_per_tp"], gemm_q)
            )

        # --- global attention + dense FFN ---
        if counts["global_dense"] > 0:
            c = counts["global_dense"] * sf
            self.generation_ops.extend(
                [
                    ops.ElementWise("generation_global_dense_attn_norm", c, 2 * h, 2 * h, 0.8),
                    ops.GEMM("generation_global_dense_qkv_gemm", c, d["global_qkv_out"], h, gemm_q),
                    ops.GenerationAttention(
                        "generation_attention",
                        c,
                        self._num_heads // tp,
                        d["global_n_kv_per_gpu"],
                        kvcache_q,
                        window_size=0,
                        head_size=d["global_hd"],
                        use_qk_norm=cfg.use_qk_norm,
                    ),
                    *self._attn_gate_ops(
                        "generation_global_dense", c, h, self._num_heads // tp, d["global_proj_in"], gemm_q
                    ),
                    ops.GEMM(
                        "generation_global_dense_proj_gemm",
                        c,
                        h,
                        d["global_proj_in"],
                        gemm_q,
                        low_precision_input=True,
                    ),
                    ops.ElementWise("generation_global_dense_ffn_norm", c, 2 * h, 2 * h, 0.8),
                ]
                + self._dense_ffn_ops("generation_global", c, h, tp, d["dense_inter_per_tp"], gemm_q)
            )

        self.generation_ops.extend(
            [
                ops.GEMM("generation_logits_gemm", 1 * sf, self._vocab_size // tp, h, common.GEMMQuantMode.bfloat16),
                ops.P2P("generation_p2p", (pp - 1) * sf, h, pp),
            ]
        )
