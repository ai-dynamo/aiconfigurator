# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging

import aiconfigurator.sdk.operations as ops
from aiconfigurator.sdk import common
from aiconfigurator.sdk.models.base import BaseModel, register_model
from aiconfigurator.sdk.models.helpers import calc_expectation

logger = logging.getLogger(__name__)


def _dsa_attention_modules_excluded_from_quant(raw_config: dict) -> bool:
    """Return whether a GLM/DSA checkpoint keeps DSA attention projections unquantized."""
    quant_config = raw_config.get("quantization_config")
    quant_config = quant_config if isinstance(quant_config, dict) else {}

    hf_quant_config = raw_config.get("hf_quant_config")
    hf_quant_config = hf_quant_config if isinstance(hf_quant_config, dict) else {}
    hf_quant = hf_quant_config.get("quantization")
    hf_quant = hf_quant if isinstance(hf_quant, dict) else {}

    patterns = [
        *list(quant_config.get("modules_to_not_convert") or []),
        *list(quant_config.get("exclude_modules") or []),
        *list(hf_quant.get("exclude_modules") or []),
    ]
    dsa_projection_markers = (
        "self_attn.q_a_proj",
        "self_attn.q_b_proj",
        "self_attn.kv_a_proj",
        "self_attn.kv_a_proj_with_mqa",
        "self_attn.kv_b_proj",
        "self_attn.o_proj",
    )
    return any(any(marker in str(pattern) for marker in dsa_projection_markers) for pattern in patterns)


def _dsa_gemm_quant_mode(extra_params: object, fallback: common.GEMMQuantMode) -> common.GEMMQuantMode:
    if isinstance(extra_params, dict):
        return extra_params.get("dsa_gemm_quant_mode", fallback)
    return fallback


@register_model("DEEPSEEKV32")
class DeepSeekV32Model(BaseModel):
    """
    DeepSeek-V3.2 / GLM-5 style DeepSeekV32-family model.

    Attention is modeled with the full DSA module-level perf tables so we can
    distinguish architectures such as ``DeepseekV32ForCausalLM`` and
    ``GlmMoeDsaForCausalLM`` without reusing the old DeepSeek-V3 MLA model.
    """

    @classmethod
    def supports_cp(cls, backend_name: str) -> bool:
        # SGLang AllGather only. See LLAMAModel.supports_cp for the
        # vLLM DCP exclusion rationale.
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
        extra_params = dict(model_info["extra_params"])
        if model_info["architecture"] == "GlmMoeDsaForCausalLM" and _dsa_attention_modules_excluded_from_quant(
            model_info.get("raw_config", {})
        ):
            extra_params.setdefault("dsa_gemm_quant_mode", common.GEMMQuantMode.bfloat16)

        if backend_name == "sglang" and model_config.enable_wideep:
            logger.debug(
                "WideEP is enabled for DeepSeekV32 model %s with backend %s",
                model_info["model_path"],
                backend_name,
            )
            return WideEPDeepSeekV32Model(*moe_args, *base_args, extra_params)
        if backend_name == "trtllm" and model_config.enable_wideep:
            logger.debug("TensorRT-LLM WideEP is enabled for DeepSeekV32 model %s", model_info["model_path"])
            return TrtllmWideEPDeepSeekV32Model(*moe_args, *base_args, extra_params)
        return cls(*moe_args, *base_args, extra_params)

    def _cp_attn_comm_ops(self) -> list:
        """DSV3.2 emits TWO per-layer AllGathers under sglang NSA prefill-CP:

        1. **Indexer key gather** -- so each rank's local Q rows can do
           topk selection against the full-sequence indexer keys.
        2. **MLA latent KV gather** ([k_nope, k_pe]) -- so each rank's
           local Q can do final attention against the full latent KV.

        BaseModel emits ONE combined gather sized for the sum -- the total
        bytes is correct but NCCL is alpha + beta*size, so the combined
        op undercounts launch overhead by one alpha per layer. Override
        here to emit two correctly-sized ops.
        """
        from aiconfigurator.sdk import operations as ops

        cp_size = self.config.cp_size
        if cp_size <= 1 or self.config.cp_style != "allgather":
            return super()._cp_attn_comm_ops()
        comm_bytes = self.config.comm_quant_mode.value.memory
        indexer_bytes, latent_bytes = self._dsa_indexer_and_latent_bytes(
            self.extra_params, self.config.kvcache_quant_mode
        )
        return [
            ops.NCCL(
                "context_cp_indexer_all_gather",
                self._num_layers,
                "all_gather",
                num_elements_per_token=indexer_bytes / comm_bytes,
                num_gpus=cp_size,
                comm_quant_mode=self.config.comm_quant_mode,
            ),
            ops.NCCL(
                "context_cp_latent_all_gather",
                self._num_layers,
                "all_gather",
                num_elements_per_token=latent_bytes / comm_bytes,
                num_gpus=cp_size,
                comm_quant_mode=self.config.comm_quant_mode,
            ),
        ]

    def __init__(self, topk: int, num_experts: int, moe_inter_size: int, *args) -> None:
        super().__init__(*args)

        # Attention/expert width matching: tp * cp * dp == moe_tp * moe_ep.
        assert self.config.attn_width == self.config.moe_tp_size * self.config.moe_ep_size, (
            f"attn_width tp*cp*dp ({self.config.attn_width}) must equal moe_tp_size * moe_ep_size "
            f"({self.config.moe_tp_size * self.config.moe_ep_size})"
        )
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
        self._power_law_alpha = 1.01

        h = self._hidden_size
        tp_size = self.config.tp_size
        moe_tp_size = self.config.moe_tp_size
        moe_ep_size = self.config.moe_ep_size
        attention_dp_size = self.config.attention_dp_size
        pp_size = self.config.pp_size

        gemm_quant_mode = self.config.gemm_quant_mode
        moe_quant_mode = self.config.moe_quant_mode
        kvcache_quant_mode = self.config.kvcache_quant_mode
        fmha_quant_mode = self.config.fmha_quant_mode
        dsa_gemm_quant_mode = _dsa_gemm_quant_mode(self.extra_params, gemm_quant_mode)
        workload_distribution = (
            self.config.workload_distribution + f"_{self._power_law_alpha}"
            if self.config.workload_distribution == "power_law"
            else self.config.workload_distribution
        )
        local_heads = self._num_heads // tp_size

        # CP (sequence parallelism). See Llama for the rationale. DSV3.2
        # only supports sglang AllGather here. ``attn_head_div`` is
        # computed for uniformity but collapses to 1 under "allgather".
        cp = self.config.cp_size
        cp_style = self.config.cp_style
        attn_count_div = cp if cp_style in ("allgather", "ring") else 1
        attn_head_div = cp if cp_style == "ulysses" else 1

        self.context_ops.extend(
            [
                ops.Embedding("context_embedding", 1, self._vocab_size, h, 0.3, seq_split=cp),
                ops.ElementWise("context_add_norm_1", self._num_layers, 2 * h, 2 * h, 0.8, seq_split=cp),
                ops.ContextDSAModule(
                    "context_attention",
                    self._num_layers / attn_count_div,
                    local_heads // attn_head_div,
                    kvcache_quant_mode,
                    fmha_quant_mode,
                    dsa_gemm_quant_mode,
                    architecture=self.architecture,
                ),
                *self._cp_attn_comm_ops(),
                ops.ElementWise("context_add_norm_2", self._num_layers, 2 * h, 2 * h, 0.8, seq_split=cp),
                ops.GEMM(
                    "context_shared_gate_up_gemm",
                    self._num_layers,
                    2 * self._moe_inter_size // tp_size,
                    h,
                    gemm_quant_mode,
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
                    gemm_quant_mode,
                    seq_split=cp,
                ),
                ops.GEMM(
                    "context_router_gemm",
                    self._num_layers,
                    self._num_experts,
                    h,
                    common.GEMMQuantMode.bfloat16,
                    seq_split=cp,
                ),
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
                    seq_split=cp,
                ),
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
                ),
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
                    seq_split=cp,
                ),
                ops.GEMM(
                    "context_logits_gemm",
                    1,
                    self._vocab_size // tp_size,
                    h,
                    common.GEMMQuantMode.bfloat16,
                    seq_split=cp,
                ),
            ]
        )

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
                ops.GenerationDSAModule(
                    "generation_attention",
                    self._num_layers * self._mtp_scale_factor,
                    local_heads,
                    kvcache_quant_mode,
                    dsa_gemm_quant_mode,
                    architecture=self.architecture,
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
        self.generation_ops.append(
            ops.GEMM(
                "generation_logits_gemm",
                1 * self._mtp_scale_factor,
                self._vocab_size // tp_size,
                h,
                common.GEMMQuantMode.bfloat16,
            )
        )

        pp_scale_factor = pp_size - 1
        self.context_ops.append(ops.P2P("context_p2p", pp_scale_factor, h, pp_size, seq_split=cp))
        self.generation_ops.append(ops.P2P("generation_p2p", pp_scale_factor * self._mtp_scale_factor, h, pp_size))

    # ----- KV bytes API: required override per BaseModel contract --------

    @staticmethod
    def _dsa_dims(extra_params) -> tuple[int, int, int]:
        """Pull DSA latent + indexer dims with DSV3.2 defaults."""
        extra = extra_params if isinstance(extra_params, dict) else {}
        return (
            extra.get("kv_lora_rank", 512),
            extra.get("qk_rope_head_dim", 64),
            extra.get("index_head_dim", 128),
        )

    @staticmethod
    def _dsa_bytes_per_layer_per_token(extra_params, kvcache_quant_mode) -> float:
        """DSA layout: kv_lora at kvcache dtype + qk_rope at BF16 + indexer cache."""
        kv_lora_rank, qk_rope_head_dim, index_head_dim = DeepSeekV32Model._dsa_dims(extra_params)
        return (
            kv_lora_rank * kvcache_quant_mode.value.memory
            + qk_rope_head_dim * common.GEMMQuantMode.bfloat16.value.memory
            + common.indexer_cache_entry_bytes(index_head_dim)
        )

    @staticmethod
    def _dsa_indexer_and_latent_bytes(extra_params, kvcache_quant_mode) -> tuple[float, float]:
        """Split per-layer per-token bytes into (indexer, latent) for CP gather sizing.

        SGLang's NSA prefill-CP path issues TWO separate AllGathers per layer
        (indexer key + MLA latent KV) -- see ``_cp_attn_comm_ops`` override.
        Sum equals ``_dsa_bytes_per_layer_per_token``.
        """
        kv_lora_rank, qk_rope_head_dim, index_head_dim = DeepSeekV32Model._dsa_dims(extra_params)
        indexer_bytes = float(common.indexer_cache_entry_bytes(index_head_dim))
        latent_bytes = (
            kv_lora_rank * kvcache_quant_mode.value.memory
            + qk_rope_head_dim * common.GEMMQuantMode.bfloat16.value.memory
        )
        return indexer_bytes, latent_bytes

    def get_kvcache_bytes_per_sequence(self, seq_len: int) -> float:
        seq_len = max(0, seq_len)
        per_layer_per_token = self._dsa_bytes_per_layer_per_token(self.extra_params, self.config.kvcache_quant_mode)
        return seq_len * self._num_layers * per_layer_per_token


class TrtllmWideEPDeepSeekV32Model(BaseModel):
    """TensorRT-LLM WideEP variant for DeepSeekV32-family models such as DeepSeek-V3.2 and GLM-5."""

    @classmethod
    def supports_cp(cls, backend_name: str) -> bool:
        # TRT-LLM Ring CP isn't wired up yet. See TrtllmWideEPDeepSeekModel.
        return False

    def get_kvcache_bytes_per_sequence(self, seq_len: int) -> float:
        """DSA layout (mirrors ``DeepSeekV32Model``)."""
        seq_len = max(0, seq_len)
        per_layer_per_token = DeepSeekV32Model._dsa_bytes_per_layer_per_token(
            self.extra_params, self.config.kvcache_quant_mode
        )
        return seq_len * self._num_layers * per_layer_per_token

    def __init__(self, topk: int, num_experts: int, moe_inter_size: int, *args) -> None:
        super().__init__(*args)

        # Attention/expert width matching: tp * cp * dp == moe_tp * moe_ep.
        assert self.config.attn_width == self.config.moe_tp_size * self.config.moe_ep_size, (
            f"attn_width tp*cp*dp ({self.config.attn_width}) must equal moe_tp_size * moe_ep_size "
            f"({self.config.moe_tp_size * self.config.moe_ep_size})"
        )
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
        self._pdl_factor = 0.9
        self._power_law_alpha = 1.01

        h = self._hidden_size
        tp_size = self.config.tp_size
        moe_tp_size = self.config.moe_tp_size
        moe_ep_size = self.config.moe_ep_size
        attention_dp_size = self.config.attention_dp_size
        pp_size = self.config.pp_size

        gemm_quant_mode = self.config.gemm_quant_mode
        moe_quant_mode = self.config.moe_quant_mode
        kvcache_quant_mode = self.config.kvcache_quant_mode
        fmha_quant_mode = self.config.fmha_quant_mode
        dsa_gemm_quant_mode = _dsa_gemm_quant_mode(self.extra_params, gemm_quant_mode)

        eplb_enabled = self.config.enable_eplb
        if self.config.workload_distribution == "power_law":
            if eplb_enabled:
                workload_distribution = f"{self.config.workload_distribution}_{self._power_law_alpha}_eplb"
            else:
                workload_distribution = f"{self.config.workload_distribution}_{self._power_law_alpha}"
        else:
            workload_distribution = self.config.workload_distribution

        if attention_dp_size <= 1:
            raise ValueError(
                f"WideEP requires attention_dp_size > 1, got {attention_dp_size}. "
                "Attention DP should be used with WideEP."
            )
        if moe_ep_size <= 1:
            raise ValueError(
                f"WideEP requires moe_ep_size > 1, got {moe_ep_size}. "
                "WideEP should only be enabled with parallel_size > 1."
            )
        if moe_ep_size <= topk:
            logger.warning(
                f"moe_ep_size ({moe_ep_size}) <= top_k ({topk}), "
                "AlltoAll communication will be disabled. Consider increasing moe_ep_size."
            )

        wideep_num_slots = self.config.wideep_num_slots if self.config.wideep_num_slots else num_experts
        if wideep_num_slots < num_experts:
            raise ValueError(
                f"wideep_num_slots ({wideep_num_slots}) must be >= num_experts ({num_experts}). "
                "There should be at least num_experts slots in the model engine."
            )
        if not eplb_enabled and wideep_num_slots != num_experts:
            raise ValueError(
                f"When enable_eplb=False, wideep_num_slots ({wideep_num_slots}) must equal "
                f"num_experts ({num_experts}). Redundant slots require EPLB to be enabled."
            )

        local_heads = self._num_heads // tp_size

        self.context_ops.extend(
            [
                ops.Embedding("context_embedding", 1, self._vocab_size, h, 0.3),
                ops.ElementWise("context_add_norm_1", self._num_layers, 2 * h, 2 * h, 0.8),
                ops.ContextDSAModule(
                    "context_attention",
                    self._num_layers,
                    local_heads,
                    kvcache_quant_mode,
                    fmha_quant_mode,
                    dsa_gemm_quant_mode,
                    architecture=self.architecture,
                ),
                ops.ElementWise("context_add_norm_2", self._num_layers, 2 * h, 2 * h, 0.8),
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
                ops.GEMM(
                    "context_router_gemm",
                    self._num_layers,
                    self._num_experts,
                    h,
                    common.GEMMQuantMode.bfloat16,
                ),
                ops.TrtLLMWideEPMoEDispatch(
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
                ),
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
                ),
                ops.TrtLLMWideEPMoEDispatch(
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
                ),
                ops.ElementWise("context_moe_reduce_add", self._num_layers, 2 * h, h, 0.8),
                ops.GEMM(
                    "context_logits_gemm",
                    1,
                    self._vocab_size // tp_size,
                    h,
                    common.GEMMQuantMode.bfloat16,
                ),
            ]
        )

        generation_scale = self._num_layers * self._mtp_scale_factor * self._pdl_factor
        self.generation_ops.extend(
            [
                ops.Embedding("generation_embedding", 1 * self._mtp_scale_factor, self._vocab_size, h, 0.3),
                ops.ElementWise("generation_add_norm_1", generation_scale, 2 * h, 2 * h, 0.8),
                ops.GenerationDSAModule(
                    "generation_attention",
                    generation_scale,
                    local_heads,
                    kvcache_quant_mode,
                    dsa_gemm_quant_mode,
                    architecture=self.architecture,
                ),
                ops.ElementWise("generation_add_norm_2", generation_scale, 2 * h, 2 * h, 0.8),
            ]
        )

        shared_ops = [
            ops.GEMM("generation_shared_gate_up_gemm", generation_scale, 2 * self._moe_inter_size, h, gemm_quant_mode),
            ops.ElementWise(
                "generation_shared_act_gate",
                generation_scale,
                2 * self._moe_inter_size,
                self._moe_inter_size,
                0.8,
            ),
            ops.GEMM("generation_shared_ffn2_gemm", generation_scale, h, self._moe_inter_size, gemm_quant_mode),
        ]
        routed_ops = [
            ops.GEMM("generation_router_gemm", generation_scale, self._num_experts, h, common.GEMMQuantMode.bfloat16),
            ops.TrtLLMWideEPMoEDispatch(
                "generation_moe_pre_dispatch",
                generation_scale,
                h,
                self._topk,
                self._num_experts,
                moe_tp_size,
                moe_ep_size,
                attention_dp_size,
                True,
                quant_mode=moe_quant_mode,
            ),
            ops.TrtLLMWideEPMoE(
                "generation_moe",
                generation_scale,
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
                generation_scale,
                h,
                self._topk,
                self._num_experts,
                moe_tp_size,
                moe_ep_size,
                attention_dp_size,
                False,
                quant_mode=moe_quant_mode,
                use_low_precision_combine=(moe_quant_mode == common.MoEQuantMode.nvfp4),
            ),
        ]
        self.generation_ops.append(ops.OverlapOp("generation_moe_overlap", group_a=routed_ops, group_b=shared_ops))
        self.generation_ops.append(ops.ElementWise("generation_moe_reduce_add", generation_scale, 2 * h, h, 0.8))
        self.generation_ops.append(
            ops.GEMM(
                "generation_logits_gemm",
                1 * self._mtp_scale_factor,
                self._vocab_size // tp_size,
                h,
                common.GEMMQuantMode.bfloat16,
            )
        )

        pp_scale_factor = pp_size - 1
        self.context_ops.append(ops.P2P("context_p2p", pp_scale_factor, h, pp_size))
        self.generation_ops.append(ops.P2P("generation_p2p", pp_scale_factor * self._mtp_scale_factor, h, pp_size))


class WideEPDeepSeekV32Model(BaseModel):
    """SGLang WideEP variant for DeepSeekV32-family models such as DeepSeek-V3.2 and GLM-5."""

    @classmethod
    def supports_cp(cls, backend_name: str) -> bool:
        return backend_name == "sglang"

    def _cp_attn_comm_ops(self) -> list:
        """Same indexer + latent split as ``DeepSeekV32Model._cp_attn_comm_ops``."""
        from aiconfigurator.sdk import operations as ops

        cp_size = self.config.cp_size
        if cp_size <= 1 or self.config.cp_style != "allgather":
            return super()._cp_attn_comm_ops()
        comm_bytes = self.config.comm_quant_mode.value.memory
        indexer_bytes, latent_bytes = DeepSeekV32Model._dsa_indexer_and_latent_bytes(
            self.extra_params, self.config.kvcache_quant_mode
        )
        return [
            ops.NCCL(
                "context_cp_indexer_all_gather",
                self._num_layers,
                "all_gather",
                num_elements_per_token=indexer_bytes / comm_bytes,
                num_gpus=cp_size,
                comm_quant_mode=self.config.comm_quant_mode,
            ),
            ops.NCCL(
                "context_cp_latent_all_gather",
                self._num_layers,
                "all_gather",
                num_elements_per_token=latent_bytes / comm_bytes,
                num_gpus=cp_size,
                comm_quant_mode=self.config.comm_quant_mode,
            ),
        ]

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

        gemm_quant_mode = self.config.gemm_quant_mode
        moe_quant_mode = self.config.moe_quant_mode
        kvcache_quant_mode = self.config.kvcache_quant_mode
        fmha_quant_mode = self.config.fmha_quant_mode
        dsa_gemm_quant_mode = _dsa_gemm_quant_mode(self.extra_params, gemm_quant_mode)
        moe_backend = self.config.moe_backend
        sms = self.config.sms

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
        local_heads = self._num_heads // tp_size

        # CP (sequence parallelism). DSV3.2 WideEP only supports SGLang
        # AllGather. ``attn_head_div`` unused here since ContextDSAModule
        # takes ``num_heads`` and DSA latent isn't head-shard-friendly.
        cp = self.config.cp_size
        cp_style = self.config.cp_style
        attn_count_div = cp if cp_style in ("allgather", "ring") else 1

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
                            seq_split=cp,
                        )
                    ]
                    if tp_size > 1
                    else []
                ),
                ops.ContextDSAModule(
                    "context_attention",
                    self._num_layers / attn_count_div,
                    local_heads,
                    kvcache_quant_mode,
                    fmha_quant_mode,
                    dsa_gemm_quant_mode,
                    architecture=self.architecture,
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
                            seq_split=cp,
                        )
                    ]
                    if tp_size > 1
                    else []
                ),
                ops.GEMM(
                    "context_gate_ffn1_gemm",
                    self._num_layers,
                    2 * self._moe_inter_size,
                    h,
                    gemm_quant_mode,
                    scale_num_tokens=tp_size,
                    seq_split=cp,
                ),
                ops.ElementWise(
                    "context_act_gate",
                    self._num_layers,
                    2 * self._moe_inter_size,
                    self._moe_inter_size,
                    0.8,
                    scale_num_tokens=tp_size,
                    seq_split=cp,
                ),
                ops.GEMM(
                    "context_ffn2_gemm",
                    self._num_layers,
                    h,
                    self._moe_inter_size,
                    gemm_quant_mode,
                    scale_num_tokens=tp_size,
                    seq_split=cp,
                ),
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
                    seq_split=cp,
                ),
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
                ),
            ]
        )

        generation_scale = self._num_layers * self._mtp_scale_factor
        self.generation_ops.extend(
            [
                ops.GenerationDSAModule(
                    "generation_attention",
                    generation_scale,
                    local_heads,
                    kvcache_quant_mode,
                    dsa_gemm_quant_mode,
                    architecture=self.architecture,
                ),
                ops.GEMM(
                    "generation_gate_ffn1_gemm",
                    generation_scale,
                    2 * self._moe_inter_size,
                    h,
                    gemm_quant_mode,
                ),
                ops.ElementWise(
                    "generation_act_gate",
                    generation_scale,
                    2 * self._moe_inter_size,
                    self._moe_inter_size,
                    0.8,
                ),
                ops.GEMM(
                    "generation_ffn2_gemm",
                    generation_scale,
                    h,
                    self._moe_inter_size,
                    gemm_quant_mode,
                ),
                ops.MoEDispatch(
                    "generation_moe_pre_dispatch",
                    generation_scale,
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
                ),
                ops.MoE(
                    "generation_moe",
                    generation_scale,
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
                ),
            ]
        )

    # ----- KV bytes API: required override per BaseModel contract --------

    def get_kvcache_bytes_per_sequence(self, seq_len: int) -> float:
        """DSA layout (mirrors ``DeepSeekV32Model``)."""
        seq_len = max(0, seq_len)
        per_layer_per_token = DeepSeekV32Model._dsa_bytes_per_layer_per_token(
            self.extra_params, self.config.kvcache_quant_mode
        )
        return seq_len * self._num_layers * per_layer_per_token
