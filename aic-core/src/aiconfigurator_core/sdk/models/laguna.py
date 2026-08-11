# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import aiconfigurator_core.sdk.operations as ops
from aiconfigurator_core.sdk import common
from aiconfigurator_core.sdk.models.base import BaseModel, register_model
from aiconfigurator_core.sdk.models.helpers import mtp_scale_factor
from aiconfigurator_core.sdk.utils import _load_model_config_from_model_path


@register_model("LAGUNA")
class LagunaModel(BaseModel):
    """Laguna hybrid full/SWA attention with dense layer 0 and MoE elsewhere."""

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
            model_info["extra_params"],
        )
        model.set_laguna_config(model_info["extra_params"])
        return model

    def __init__(self, topk: int, num_experts: int, moe_inter_size: int, *args) -> None:
        super().__init__(*args)
        assert (
            self.config.tp_size * self.config.attention_dp_size == self.config.moe_tp_size * self.config.moe_ep_size
        ), (
            f"tp_size ({self.config.tp_size}) * attention_dp_size "
            f"({self.config.attention_dp_size}) should be equal to moe_tp_size "
            f"({self.config.moe_tp_size}) * moe_ep_size ({self.config.moe_ep_size})"
        )
        assert num_experts >= self.config.moe_ep_size, f"ep size cannot be larger than num_experts {num_experts}"
        self._topk = topk
        self._num_experts = num_experts
        self._moe_inter_size = moe_inter_size
        self._mtp_scale_factor = mtp_scale_factor(self._nextn, self._num_layers)
        self._laguna_config: common.LagunaConfig | None = None
        # Match existing hybrid MoE families until Laguna-specific routing imbalance is calibrated.
        self._power_law_alpha = 1.01
        self._validate_fp8_block_quantized_moe_config()

    def _validate_fp8_block_quantized_moe_config(self) -> None:
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

    def set_laguna_config(self, cfg: common.LagunaConfig) -> None:
        if cfg is None or not isinstance(cfg, common.LagunaConfig):
            raise ValueError(f"LagunaModel requires a LagunaConfig, got {type(cfg).__name__}")
        if len(cfg.layer_types) != self._num_layers:
            raise ValueError(
                f"LagunaConfig.layer_types length ({len(cfg.layer_types)}) does not match "
                f"num_layers ({self._num_layers})"
            )
        if len(cfg.mlp_layer_types) != self._num_layers:
            raise ValueError(
                f"LagunaConfig.mlp_layer_types length ({len(cfg.mlp_layer_types)}) does not match "
                f"num_layers ({self._num_layers})"
            )
        if len(cfg.num_attention_heads_per_layer) != self._num_layers:
            raise ValueError(
                "LagunaConfig.num_attention_heads_per_layer length "
                f"({len(cfg.num_attention_heads_per_layer)}) does not match num_layers ({self._num_layers})"
            )
        for i, (attn_type, mlp_type, n_heads) in enumerate(
            zip(cfg.layer_types, cfg.mlp_layer_types, cfg.num_attention_heads_per_layer, strict=True)
        ):
            if attn_type not in ("full_attention", "sliding_attention"):
                raise ValueError(f"LagunaConfig layer {i} has invalid attention type {attn_type!r}")
            if mlp_type not in ("dense", "sparse"):
                raise ValueError(f"LagunaConfig layer {i} has invalid MLP type {mlp_type!r}")
            if n_heads <= 0:
                raise ValueError(f"LagunaConfig layer {i} has invalid attention head count {n_heads}")
            if n_heads % self.config.tp_size != 0:
                raise ValueError(
                    f"LagunaConfig layer {i} attention head count {n_heads} must be divisible by "
                    f"tp_size {self.config.tp_size}"
                )
            if n_heads % self._num_kv_heads != 0:
                raise ValueError(
                    f"LagunaConfig layer {i} attention head count {n_heads} must be divisible by "
                    f"num_kv_heads {self._num_kv_heads}"
                )

        self._laguna_config = cfg
        self._build_context_ops()
        self._build_generation_ops()

    def _count_layer_types(self) -> dict[str, int]:
        cfg = self._laguna_config
        counts: dict[str, int] = {"global_dense": 0, "global_moe": 0, "swa_moe": 0, "swa_dense": 0}
        for attn_type, mlp_type in zip(cfg.layer_types, cfg.mlp_layer_types, strict=True):
            if attn_type == "full_attention" and mlp_type == "dense":
                counts["global_dense"] += 1
            elif attn_type == "full_attention" and mlp_type == "sparse":
                counts["global_moe"] += 1
            elif attn_type == "sliding_attention" and mlp_type == "sparse":
                counts["swa_moe"] += 1
            else:
                counts["swa_dense"] += 1
        return counts

    def _heads_for_layer_type(self, layer_type: str) -> int:
        cfg = self._laguna_config
        heads = {
            n_heads
            for attn_type, n_heads in zip(cfg.layer_types, cfg.num_attention_heads_per_layer, strict=True)
            if attn_type == layer_type
        }
        if len(heads) != 1:
            raise ValueError(f"Laguna {layer_type} layers must have one head count, got {sorted(heads)}")
        return next(iter(heads))

    def _resolve_bucket_dims(self, n_q: int, window_size: int) -> dict[str, int]:
        tp = self.config.tp_size
        n_q_per_gpu = n_q // tp
        n_kv_per_gpu = (self._num_kv_heads + tp - 1) // tp
        return {
            "n_q_per_gpu": n_q_per_gpu,
            "n_kv_per_gpu": n_kv_per_gpu,
            "qkv_out": n_q_per_gpu * self._head_size + n_kv_per_gpu * self._head_size * 2,
            "proj_in": n_q_per_gpu * self._head_size,
            "window_size": window_size,
            "head_size": self._head_size,
        }

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
        *,
        is_context: bool,
    ) -> list:
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
                is_context=is_context,
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
                is_context=is_context,
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
                is_context=is_context,
            ),
        ]

    def _dense_ffn_ops(
        self, prefix: str, count: float, h: int, dense_inter_per_tp: int, gemm_q: common.GEMMQuantMode
    ) -> list:
        return [
            ops.GEMM(f"{prefix}_dense_gate_up_gemm", count, 2 * dense_inter_per_tp, h, gemm_q),
            ops.ElementWise(f"{prefix}_dense_act", count, 2 * dense_inter_per_tp, dense_inter_per_tp, 0.8),
            ops.GEMM(f"{prefix}_dense_down_gemm", count, h, dense_inter_per_tp, gemm_q, low_precision_input=True),
        ]

    def _shared_expert_ops(
        self, prefix: str, count: float, h: int, shared_inter_per_tp: int, gemm_q: common.GEMMQuantMode
    ) -> list:
        if shared_inter_per_tp <= 0:
            return []
        return [
            ops.GEMM(f"{prefix}_shared_gate_up_gemm", count, 2 * shared_inter_per_tp, h, gemm_q),
            ops.ElementWise(f"{prefix}_shared_act", count, 2 * shared_inter_per_tp, shared_inter_per_tp, 0.8),
            ops.GEMM(f"{prefix}_shared_down_gemm", count, h, shared_inter_per_tp, gemm_q, low_precision_input=True),
        ]

    def _attn_ops(
        self,
        prefix: str,
        count: float,
        dims: dict[str, int],
        gemm_q: common.GEMMQuantMode,
        kvcache_q: common.KVCacheQuantMode,
        fmha_q: common.FMHAQuantMode | None,
        *,
        is_context: bool,
    ) -> list:
        cfg = self._laguna_config
        attention_op = (
            ops.ContextAttention(
                "context_attention",
                count,
                dims["n_q_per_gpu"],
                dims["n_kv_per_gpu"],
                kvcache_q,
                fmha_q,
                window_size=dims["window_size"],
                head_size=dims["head_size"],
                use_qk_norm=cfg.use_qk_norm,
            )
            if is_context
            else ops.GenerationAttention(
                "generation_attention",
                count,
                dims["n_q_per_gpu"],
                dims["n_kv_per_gpu"],
                kvcache_q,
                window_size=dims["window_size"],
                head_size=dims["head_size"],
                use_qk_norm=cfg.use_qk_norm,
            )
        )
        gate_ops = []
        if cfg.gating:
            gate_ops = [
                ops.GEMM(f"{prefix}_attention_gate_gemm", count, dims["n_q_per_gpu"], self._hidden_size, gemm_q),
                ops.ElementWise(
                    f"{prefix}_attention_gate_act", count, dims["n_q_per_gpu"], dims["n_q_per_gpu"], 0.8
                ),
            ]
        return [
            ops.ElementWise(f"{prefix}_attn_norm", count, 2 * self._hidden_size, 2 * self._hidden_size, 0.8),
            ops.GEMM(f"{prefix}_qkv_gemm", count, dims["qkv_out"], self._hidden_size, gemm_q),
            attention_op,
            *gate_ops,
            ops.GEMM(
                f"{prefix}_proj_gemm",
                count,
                self._hidden_size,
                dims["proj_in"],
                gemm_q,
                low_precision_input=True,
            ),
        ]

    def _workload_distribution(self) -> str:
        if self.config.workload_distribution == "power_law":
            return self.config.workload_distribution + f"_{self._power_law_alpha}"
        return self.config.workload_distribution

    def _build_context_ops(self) -> None:
        if not self._laguna_config:
            return

        cfg = self._laguna_config
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
        wl_dist = self._workload_distribution()
        dense_inter_per_tp = self._inter_size // tp
        shared_inter_per_tp = cfg.shared_expert_inter_size // tp
        global_dims = self._resolve_bucket_dims(self._heads_for_layer_type("full_attention"), 0)
        swa_dims = self._resolve_bucket_dims(self._heads_for_layer_type("sliding_attention"), cfg.sliding_window_size)

        self.context_ops = [ops.Embedding("context_embedding", 1, self._vocab_size, h, 0.3)]

        self._extend_context_bucket(
            "context_global_dense",
            counts["global_dense"],
            global_dims,
            dense_inter_per_tp,
            h,
            gemm_q,
            kvcache_q,
            fmha_q,
            is_moe=False,
            moe_args=(moe_tp, moe_ep, attn_dp, moe_q, wl_dist),
            shared_inter_per_tp=shared_inter_per_tp,
        )
        self._extend_context_bucket(
            "context_global",
            counts["global_moe"],
            global_dims,
            dense_inter_per_tp,
            h,
            gemm_q,
            kvcache_q,
            fmha_q,
            is_moe=True,
            moe_args=(moe_tp, moe_ep, attn_dp, moe_q, wl_dist),
            shared_inter_per_tp=shared_inter_per_tp,
        )
        self._extend_context_bucket(
            "context_swa",
            counts["swa_moe"],
            swa_dims,
            dense_inter_per_tp,
            h,
            gemm_q,
            kvcache_q,
            fmha_q,
            is_moe=True,
            moe_args=(moe_tp, moe_ep, attn_dp, moe_q, wl_dist),
            shared_inter_per_tp=shared_inter_per_tp,
        )
        self._extend_context_bucket(
            "context_swa_dense",
            counts["swa_dense"],
            swa_dims,
            dense_inter_per_tp,
            h,
            gemm_q,
            kvcache_q,
            fmha_q,
            is_moe=False,
            moe_args=(moe_tp, moe_ep, attn_dp, moe_q, wl_dist),
            shared_inter_per_tp=shared_inter_per_tp,
        )

        self.context_ops.extend(
            [
                ops.GEMM("context_logits_gemm", 1, self._vocab_size // tp, h, common.GEMMQuantMode.bfloat16),
                ops.P2P("context_p2p", pp - 1, h, pp),
            ]
        )

    def _extend_context_bucket(
        self,
        prefix: str,
        count: int,
        dims: dict[str, int],
        dense_inter_per_tp: int,
        h: int,
        gemm_q: common.GEMMQuantMode,
        kvcache_q: common.KVCacheQuantMode,
        fmha_q: common.FMHAQuantMode,
        *,
        is_moe: bool,
        moe_args: tuple[int, int, int, common.MoEQuantMode, str],
        shared_inter_per_tp: int,
    ) -> None:
        if count <= 0:
            return
        self.context_ops.extend(self._attn_ops(prefix, count, dims, gemm_q, kvcache_q, fmha_q, is_context=True))
        if is_moe:
            moe_tp, moe_ep, attn_dp, moe_q, wl_dist = moe_args
            self.context_ops.append(ops.ElementWise(f"{prefix}_moe_norm", count, 2 * h, 2 * h, 0.8))
            self.context_ops.extend(
                self._moe_ops(prefix, count, h, moe_tp, moe_ep, attn_dp, moe_q, wl_dist, is_context=True)
            )
            self.context_ops.extend(self._shared_expert_ops(prefix, count, h, shared_inter_per_tp, gemm_q))
        else:
            self.context_ops.append(ops.ElementWise(f"{prefix}_dense_ffn_norm", count, 2 * h, 2 * h, 0.8))
            self.context_ops.extend(self._dense_ffn_ops(prefix, count, h, dense_inter_per_tp, gemm_q))

    def _build_generation_ops(self) -> None:
        if not self._laguna_config:
            return

        cfg = self._laguna_config
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
        wl_dist = self._workload_distribution()
        dense_inter_per_tp = self._inter_size // tp
        shared_inter_per_tp = cfg.shared_expert_inter_size // tp
        global_dims = self._resolve_bucket_dims(self._heads_for_layer_type("full_attention"), 0)
        swa_dims = self._resolve_bucket_dims(self._heads_for_layer_type("sliding_attention"), cfg.sliding_window_size)

        self.generation_ops = [ops.Embedding("generation_embedding", 1 * sf, self._vocab_size, h, 0.3)]

        self._extend_generation_bucket(
            "generation_global_dense",
            counts["global_dense"] * sf,
            global_dims,
            dense_inter_per_tp,
            h,
            gemm_q,
            kvcache_q,
            is_moe=False,
            moe_args=(moe_tp, moe_ep, attn_dp, moe_q, wl_dist),
            shared_inter_per_tp=shared_inter_per_tp,
        )
        self._extend_generation_bucket(
            "generation_global",
            counts["global_moe"] * sf,
            global_dims,
            dense_inter_per_tp,
            h,
            gemm_q,
            kvcache_q,
            is_moe=True,
            moe_args=(moe_tp, moe_ep, attn_dp, moe_q, wl_dist),
            shared_inter_per_tp=shared_inter_per_tp,
        )
        self._extend_generation_bucket(
            "generation_swa",
            counts["swa_moe"] * sf,
            swa_dims,
            dense_inter_per_tp,
            h,
            gemm_q,
            kvcache_q,
            is_moe=True,
            moe_args=(moe_tp, moe_ep, attn_dp, moe_q, wl_dist),
            shared_inter_per_tp=shared_inter_per_tp,
        )
        self._extend_generation_bucket(
            "generation_swa_dense",
            counts["swa_dense"] * sf,
            swa_dims,
            dense_inter_per_tp,
            h,
            gemm_q,
            kvcache_q,
            is_moe=False,
            moe_args=(moe_tp, moe_ep, attn_dp, moe_q, wl_dist),
            shared_inter_per_tp=shared_inter_per_tp,
        )

        self.generation_ops.extend(
            [
                ops.GEMM("generation_logits_gemm", 1 * sf, self._vocab_size // tp, h, common.GEMMQuantMode.bfloat16),
                ops.P2P("generation_p2p", (pp - 1) * sf, h, pp),
            ]
        )

    def _extend_generation_bucket(
        self,
        prefix: str,
        count: float,
        dims: dict[str, int],
        dense_inter_per_tp: int,
        h: int,
        gemm_q: common.GEMMQuantMode,
        kvcache_q: common.KVCacheQuantMode,
        *,
        is_moe: bool,
        moe_args: tuple[int, int, int, common.MoEQuantMode, str],
        shared_inter_per_tp: int,
    ) -> None:
        if count <= 0:
            return
        self.generation_ops.extend(self._attn_ops(prefix, count, dims, gemm_q, kvcache_q, None, is_context=False))
        if is_moe:
            moe_tp, moe_ep, attn_dp, moe_q, wl_dist = moe_args
            self.generation_ops.append(ops.ElementWise(f"{prefix}_moe_norm", count, 2 * h, 2 * h, 0.8))
            self.generation_ops.extend(
                self._moe_ops(prefix, count, h, moe_tp, moe_ep, attn_dp, moe_q, wl_dist, is_context=False)
            )
            self.generation_ops.extend(self._shared_expert_ops(prefix, count, h, shared_inter_per_tp, gemm_q))
        else:
            self.generation_ops.append(ops.ElementWise(f"{prefix}_dense_ffn_norm", count, 2 * h, 2 * h, 0.8))
            self.generation_ops.extend(self._dense_ffn_ops(prefix, count, h, dense_inter_per_tp, gemm_q))

    def get_kvcache_elements_per_token(self) -> int:
        if not self._laguna_config:
            return super().get_kvcache_elements_per_token()
        tp = self.config.tp_size
        kv_per_gpu = (self._num_kv_heads + tp - 1) // tp
        return 2 * self._num_layers * kv_per_gpu * self._head_size

    def get_kvcache_bytes_per_sequence(self, seq_len: int) -> float:
        if not self._laguna_config:
            return super().get_kvcache_bytes_per_sequence(seq_len)
        seq_len = max(0, seq_len)
        cfg = self._laguna_config
        bytes_per_elem = self.config.kvcache_quant_mode.value.memory
        kv_per_gpu = (self._num_kv_heads + self.config.tp_size - 1) // self.config.tp_size
        num_swa = cfg.layer_types.count("sliding_attention")
        num_global = cfg.layer_types.count("full_attention")
        swa_seq = min(seq_len, cfg.sliding_window_size) if cfg.sliding_window_size > 0 else seq_len
        swa_bytes = num_swa * kv_per_gpu * self._head_size * 2 * bytes_per_elem * swa_seq
        global_bytes = num_global * kv_per_gpu * self._head_size * 2 * bytes_per_elem * seq_len
        return float(swa_bytes + global_bytes)

    def get_kvcache_max_tokens(self, kv_budget_bytes: float) -> int:
        """Capacity inverse over Laguna's window-capped KV curve."""
        if not self._laguna_config:
            return super().get_kvcache_max_tokens(kv_budget_bytes)
        return self._binary_search_kvcache_max_tokens(kv_budget_bytes)
