# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging

import aiconfigurator_core.sdk.operations as ops
from aiconfigurator_core.sdk import common
from aiconfigurator_core.sdk.models.base import BaseModel, register_model
from aiconfigurator_core.sdk.models.helpers import mtp_scale_factor

logger = logging.getLogger(__name__)


def _dsa_full_layer_fraction(raw_config: dict, num_layers: int) -> float:
    """Fraction of DSA layers that COMPUTE the indexer (full) vs reuse a shared
    topk index (skip). Replicates sglang ``dsa_layer_skips_topk``: a layer skips
    when ``index_topk_pattern[lid]=='S'``, else (with explicit offset)
    ``max(lid - offset + 1, 0) % freq != 0`` or (no offset) ``max(lid-1,0)%freq``.
    GLM-5.2:
    freq=4, offset=3, 78 layers -> 21 full / 57 skip = 0.2692 (NOT 1/freq=0.25 —
    layers 0..2 are full and the periodic pattern starts at the offset). Returns
    1.0 when freq<=1 / no skipping (DeepSeek-V3.2 / GLM-5)."""
    freq = int(raw_config.get("index_topk_freq", 1) or 1)
    pattern = raw_config.get("index_topk_pattern")
    offset = raw_config.get("index_skip_topk_offset")
    if freq <= 1 and not pattern:
        return 1.0

    def _skips(lid: int) -> bool:
        if pattern is not None:
            return lid < len(pattern) and pattern[lid] == "S"
        # Match sglang dsa_layer_skips_topk EXACTLY: with an explicit offset use
        # max(lid-offset+1,0)%freq; with no offset the default is max(lid-1,0)%freq
        # (NOT offset=1 — that would be max(lid,0)). GLM-5.2 sets offset=3.
        if offset is not None:
            return max(lid - offset + 1, 0) % freq != 0
        return max(lid - 1, 0) % freq != 0

    n_full = sum(1 for lid in range(int(num_layers)) if not _skips(lid))
    return n_full / int(num_layers) if num_layers else 1.0


def _quant_exclude_patterns(raw_config: dict) -> list:
    """All module-exclusion globs a ModelOpt/HF quant config can carry."""
    quant_config = raw_config.get("quantization_config")
    quant_config = quant_config if isinstance(quant_config, dict) else {}

    hf_quant_config = raw_config.get("hf_quant_config")
    hf_quant_config = hf_quant_config if isinstance(hf_quant_config, dict) else {}
    hf_quant = hf_quant_config.get("quantization")
    hf_quant = hf_quant if isinstance(hf_quant, dict) else {}

    return [
        *list(quant_config.get("modules_to_not_convert") or []),
        *list(quant_config.get("exclude_modules") or []),
        *list(quant_config.get("ignore") or []),
        *list(hf_quant.get("exclude_modules") or []),
        *list(hf_quant.get("ignore") or []),
    ]


def _dsa_attention_modules_excluded_from_quant(raw_config: dict) -> bool:
    """Return whether a GLM/DSA checkpoint keeps DSA attention projections unquantized."""
    # Match either a full projection name (e.g. "self_attn.q_a_proj") or a
    # layer-prefixed glob the ModelOpt exporter emits (e.g.
    # "model.layers.10.self_attn*"). The latter is how nvidia/GLM-5-NVFP4
    # excludes DSA attention from NVFP4; the full-name-only check missed it.
    return any("self_attn" in str(pattern) for pattern in _quant_exclude_patterns(raw_config))


def _shared_experts_excluded_from_quant(raw_config: dict) -> bool:
    """Return whether a GLM/DSA checkpoint keeps the MoE shared experts unquantized.

    nvidia/GLM-5.2-NVFP4 excludes every ``model.layers.N.mlp.shared_experts*`` from
    NVFP4 (shared experts stay bf16; only the routed experts are quantized), so the
    shared-expert GEMMs must be modeled at bf16, not the global gemm_quant_mode."""
    return any("shared_expert" in str(pattern) for pattern in _quant_exclude_patterns(raw_config))


def _dsa_gemm_quant_mode(extra_params: object, fallback: common.GEMMQuantMode) -> common.GEMMQuantMode:
    if isinstance(extra_params, dict):
        return extra_params.get("dsa_gemm_quant_mode", fallback)
    return fallback


def _dsa_shared_expert_quant_mode(extra_params: object, fallback: common.GEMMQuantMode) -> common.GEMMQuantMode:
    if isinstance(extra_params, dict):
        return extra_params.get("dsa_shared_expert_quant_mode", fallback)
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
        # GLM-5 DSA prefill CP: SGLang AllGather only. CP is modeled INSIDE
        # ContextDSAModule (_query_cp) + DSA-specific MoE comm, NOT via the
        # dense _cp_attn_comm_ops / seq_split skeleton.
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
        if model_info["architecture"] == "GlmMoeDsaForCausalLM" and _shared_experts_excluded_from_quant(
            model_info.get("raw_config", {})
        ):
            extra_params.setdefault("dsa_shared_expert_quant_mode", common.GEMMQuantMode.bfloat16)
        # GLM-5.2 shares one DSA topk index across ``index_topk_freq`` layers
        # (GLM-5 / DeepSeek-V3.2 omit it => 1). The DSA modules amortize the
        # per-layer indexer cost over the group using the collected skip data.
        extra_params.setdefault("index_topk_freq", int(model_info.get("raw_config", {}).get("index_topk_freq", 1) or 1))
        # EXACT full-layer fraction (honors index_skip_topk_offset / pattern) so
        # the per-layer amortization weights real full vs skip counts, not the
        # 1/freq approximation (GLM-5.2: 21/78=0.2692, not 0.25 — under-counting
        # full made AIC predict too fast).
        # The skip-indexer perf rows are produced ONLY by the sglang collector.
        # On backends without a skip producer (e.g. trtllm) the per-layer
        # amortization must run all-full (fraction 1.0): otherwise the consumer
        # would weight in a skip table that was never collected for that backend.
        # The model still HAS skip layers (index_topk_freq reflects that); we just
        # cannot model their saving without data, so we count them as full.
        extra_params.setdefault(
            "dsa_full_layer_fraction",
            _dsa_full_layer_fraction(model_info.get("raw_config", {}), model_info["layers"])
            if backend_name == "sglang"
            else 1.0,
        )

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

    def __init__(self, topk: int, num_experts: int, moe_inter_size: int, *args) -> None:
        super().__init__(*args)

        assert (
            self.config.tp_size * self.config.attention_dp_size * self.config.cp_size
            == self.config.moe_tp_size * self.config.moe_ep_size
        ), (
            f"tp_size ({self.config.tp_size}) * attention_dp_size "
            f"({self.config.attention_dp_size}) * cp_size "
            f"({self.config.cp_size}) should be equal to moe_tp_size "
            f"({self.config.moe_tp_size}) * moe_ep_size ({self.config.moe_ep_size})"
        )
        assert num_experts >= self.config.moe_ep_size, f"ep size cannot be larger than num_experts {num_experts}"

        self._topk = topk
        self._num_experts = num_experts
        self._moe_inter_size = moe_inter_size
        self._mtp_scale_factor = mtp_scale_factor(self._nextn, self._num_layers)
        self._power_law_alpha = 1.01

        h = self._hidden_size
        tp_size = self.config.tp_size
        moe_tp_size = self.config.moe_tp_size
        moe_ep_size = self.config.moe_ep_size
        attention_dp_size = self.config.attention_dp_size
        cp_size = self.config.cp_size  # context parallelism (token split, orthogonal to tp)
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

        self.context_ops.extend(
            [
                ops.Embedding("context_embedding", 1, self._vocab_size, h, 0.3),
                ops.ElementWise("context_add_norm_1", self._num_layers, 2 * h, 2 * h, 0.8, scale_num_tokens=cp_size),
                ops.ContextDSAModule(
                    "context_attention",
                    self._num_layers,
                    local_heads,
                    kvcache_quant_mode,
                    fmha_quant_mode,
                    dsa_gemm_quant_mode,
                    architecture=self.architecture,
                    cp_size=self.config.cp_size,
                    index_topk_freq=self.extra_params.get("index_topk_freq", 1),
                    dsa_full_layer_fraction=self.extra_params.get("dsa_full_layer_fraction"),
                ),
                ops.ElementWise("context_add_norm_2", self._num_layers, 2 * h, 2 * h, 0.8, scale_num_tokens=cp_size),
                ops.GEMM(
                    "context_shared_gate_up_gemm",
                    self._num_layers,
                    2 * self._moe_inter_size // moe_tp_size,
                    h,
                    _dsa_shared_expert_quant_mode(self.extra_params, gemm_quant_mode),
                ),
                ops.ElementWise(
                    "context_shared_act_gate",
                    self._num_layers,
                    2 * self._moe_inter_size // moe_tp_size,
                    self._moe_inter_size // moe_tp_size,
                    0.8,
                ),
                ops.GEMM(
                    "context_shared_ffn2_gemm",
                    self._num_layers,
                    h,
                    self._moe_inter_size // moe_tp_size,
                    _dsa_shared_expert_quant_mode(self.extra_params, gemm_quant_mode),
                ),
                ops.GEMM(
                    "context_router_gemm",
                    self._num_layers,
                    self._num_experts,
                    h,
                    common.GEMMQuantMode.bfloat16,
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
                    attn_cp_size=self.config.cp_size,
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
                    attn_cp_size=self.config.cp_size,
                ),
                ops.GEMM(
                    "context_logits_gemm",
                    1,
                    self._vocab_size // tp_size,
                    h,
                    common.GEMMQuantMode.bfloat16,
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
                    index_topk_freq=self.extra_params.get("index_topk_freq", 1),
                    dsa_full_layer_fraction=self.extra_params.get("dsa_full_layer_fraction"),
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
                2 * self._moe_inter_size // moe_tp_size,
                h,
                _dsa_shared_expert_quant_mode(self.extra_params, gemm_quant_mode),
            ),
            ops.ElementWise(
                "generation_shared_act_gate",
                self._num_layers * self._mtp_scale_factor,
                2 * self._moe_inter_size // moe_tp_size,
                self._moe_inter_size // moe_tp_size,
                0.8,
            ),
            ops.GEMM(
                "generation_shared_ffn2_gemm",
                self._num_layers * self._mtp_scale_factor,
                h,
                self._moe_inter_size // moe_tp_size,
                _dsa_shared_expert_quant_mode(self.extra_params, gemm_quant_mode),
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
                attn_cp_size=self.config.cp_size,
                is_context=False,  # decode: MoEDispatch picks the decode-CP comm path
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
                attn_cp_size=self.config.cp_size,
                is_context=False,  # decode: MoEDispatch picks the decode-CP comm path
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
        self.context_ops.append(ops.P2P("context_p2p", pp_scale_factor, h, pp_size))
        self.generation_ops.append(ops.P2P("generation_p2p", pp_scale_factor * self._mtp_scale_factor, h, pp_size))

    def get_kvcache_bytes_per_sequence(self, seq_len: int) -> float:
        seq_len = max(0, seq_len)
        extra = self.extra_params if isinstance(self.extra_params, dict) else {}
        kv_lora_rank = extra.get("kv_lora_rank", 512)
        qk_rope_head_dim = extra.get("qk_rope_head_dim", 64)
        index_head_dim = extra.get("index_head_dim", 128)
        return (
            self._num_layers
            * seq_len
            * (
                kv_lora_rank * self.config.kvcache_quant_mode.value.memory
                + qk_rope_head_dim * common.GEMMQuantMode.bfloat16.value.memory
                + common.indexer_cache_entry_bytes(index_head_dim)
            )
        )


class TrtllmWideEPDeepSeekV32Model(BaseModel):
    """TensorRT-LLM WideEP variant for DeepSeekV32-family models such as DeepSeek-V3.2 and GLM-5."""

    def __init__(self, topk: int, num_experts: int, moe_inter_size: int, *args) -> None:
        super().__init__(*args)

        assert (
            self.config.tp_size * self.config.attention_dp_size * self.config.cp_size
            == self.config.moe_tp_size * self.config.moe_ep_size
        ), (
            f"tp_size ({self.config.tp_size}) * attention_dp_size "
            f"({self.config.attention_dp_size}) * cp_size "
            f"({self.config.cp_size}) should be equal to moe_tp_size "
            f"({self.config.moe_tp_size}) * moe_ep_size ({self.config.moe_ep_size})"
        )
        assert num_experts >= self.config.moe_ep_size, f"ep size cannot be larger than num_experts {num_experts}"

        self._topk = topk
        self._num_experts = num_experts
        self._moe_inter_size = moe_inter_size
        self._mtp_scale_factor = mtp_scale_factor(self._nextn, self._num_layers)
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
                    cp_size=self.config.cp_size,
                    index_topk_freq=self.extra_params.get("index_topk_freq", 1),
                    dsa_full_layer_fraction=self.extra_params.get("dsa_full_layer_fraction"),
                ),
                ops.ElementWise("context_add_norm_2", self._num_layers, 2 * h, 2 * h, 0.8),
                ops.GEMM(
                    "context_shared_gate_up_gemm",
                    self._num_layers,
                    2 * self._moe_inter_size,
                    h,
                    _dsa_shared_expert_quant_mode(self.extra_params, gemm_quant_mode),
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
                    _dsa_shared_expert_quant_mode(self.extra_params, gemm_quant_mode),
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
                    index_topk_freq=self.extra_params.get("index_topk_freq", 1),
                    dsa_full_layer_fraction=self.extra_params.get("dsa_full_layer_fraction"),
                ),
                ops.ElementWise("generation_add_norm_2", generation_scale, 2 * h, 2 * h, 0.8),
            ]
        )

        shared_ops = [
            ops.GEMM(
                "generation_shared_gate_up_gemm",
                generation_scale,
                2 * self._moe_inter_size,
                h,
                _dsa_shared_expert_quant_mode(self.extra_params, gemm_quant_mode),
            ),
            ops.ElementWise(
                "generation_shared_act_gate",
                generation_scale,
                2 * self._moe_inter_size,
                self._moe_inter_size,
                0.8,
            ),
            ops.GEMM(
                "generation_shared_ffn2_gemm",
                generation_scale,
                h,
                self._moe_inter_size,
                _dsa_shared_expert_quant_mode(self.extra_params, gemm_quant_mode),
            ),
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

    def __init__(self, topk: int, num_experts: int, moe_inter_size: int, *args) -> None:
        super().__init__(*args)

        assert num_experts >= self.config.moe_ep_size, f"ep size cannot be larger than num_experts {num_experts}"

        self._topk = topk
        self._num_experts = num_experts
        self._moe_inter_size = moe_inter_size
        self._mtp_scale_factor = mtp_scale_factor(self._nextn, self._num_layers)

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
                        )
                    ]
                    if tp_size > 1
                    else []
                ),
                ops.ContextDSAModule(
                    "context_attention",
                    self._num_layers,
                    local_heads,
                    kvcache_quant_mode,
                    fmha_quant_mode,
                    dsa_gemm_quant_mode,
                    architecture=self.architecture,
                    cp_size=self.config.cp_size,
                    index_topk_freq=self.extra_params.get("index_topk_freq", 1),
                    dsa_full_layer_fraction=self.extra_params.get("dsa_full_layer_fraction"),
                ),
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
                ops.GEMM(
                    "context_gate_ffn1_gemm",
                    self._num_layers,
                    2 * self._moe_inter_size,
                    h,
                    gemm_quant_mode,
                    scale_num_tokens=tp_size,
                ),
                ops.ElementWise(
                    "context_act_gate",
                    self._num_layers,
                    2 * self._moe_inter_size,
                    self._moe_inter_size,
                    0.8,
                    scale_num_tokens=tp_size,
                ),
                ops.GEMM(
                    "context_ffn2_gemm",
                    self._num_layers,
                    h,
                    self._moe_inter_size,
                    gemm_quant_mode,
                    scale_num_tokens=tp_size,
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
                    attn_cp_size=self.config.cp_size,
                    sms=sms,
                    moe_backend=moe_backend,
                    is_context=True,
                    scale_num_tokens=tp_size,
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
                    index_topk_freq=self.extra_params.get("index_topk_freq", 1),
                    dsa_full_layer_fraction=self.extra_params.get("dsa_full_layer_fraction"),
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
                    attn_cp_size=self.config.cp_size,
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
