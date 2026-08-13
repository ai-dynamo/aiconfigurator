# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging

import aiconfigurator_core.sdk.operations as ops
from aiconfigurator_core.sdk import common
from aiconfigurator_core.sdk.models.base import BaseModel, register_model
from aiconfigurator_core.sdk.models.blocks.moe import MoEBlockShape, build_moe_block_ops
from aiconfigurator_core.sdk.models.helpers import (
    attention_modules_excluded_from_quant,
    attention_projection_exclusions,
    large_ep_gpus_per_node,
    mtp_scale_factor,
    power_law_distribution,
    quant_exclude_patterns,
    validate_trtllm_large_ep,
)

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
    return quant_exclude_patterns(raw_config)


def _dsa_attention_modules_excluded_from_quant(raw_config: dict) -> bool:
    """Return whether a GLM/DSA checkpoint keeps DSA attention projections unquantized."""
    return attention_modules_excluded_from_quant(raw_config)


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


def _dsa_attention_quant_modes(
    extra_params: object, fallback: common.GEMMQuantMode
) -> tuple[dict, common.GEMMQuantMode]:
    """Per-projection quant modes and the single module perf key.

    An explicit ``dsa_gemm_quant_mode`` override applies to every projection
    (back-compat). Otherwise groups named in ``dsa_attn_quant_exclusions``
    run BF16 and the rest keep the global mode. Module perf rows carry ONE
    gemm_type; for mixed checkpoints no row matches exactly, so the key
    follows o_proj — the largest projection by bytes and FLOPs.
    """
    explicit = None
    exclusions: frozenset = frozenset()
    if isinstance(extra_params, dict):
        explicit = extra_params.get("dsa_gemm_quant_mode")
        exclusions = extra_params.get("dsa_attn_quant_exclusions") or frozenset()
    if explicit is not None:
        modes = dict.fromkeys(("q", "kv", "o", "indexer"), explicit)
        return modes, explicit
    modes = {g: common.GEMMQuantMode.bfloat16 if g in exclusions else fallback for g in ("q", "kv", "o", "indexer")}
    distinct = set(modes.values())
    return modes, (distinct.pop() if len(distinct) == 1 else modes["o"])


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
        # Checkpoint-driven, not architecture-gated: vLLM honors ModelOpt
        # exclude_modules wildcards for any architecture (hf_quant_config.json
        # is read in transformers_utils/config.py:726; excluded prefixes fall
        # back to the unquantized path via ModelOptNvFp4Config.is_layer_excluded
        # -> is_layer_skipped, modelopt.py:150-161 @0.24.0). Exclusions are
        # PER-PROJECTION: DeepSeek-V3.2-NVFP4 keeps q/kv/indexer in BF16 but
        # quantizes o_proj; GLM-5 NVFP4 excludes the whole self_attn block.
        extra_params.setdefault(
            "dsa_attn_quant_exclusions",
            attention_projection_exclusions(model_info.get("raw_config", {})),
        )
        if _shared_experts_excluded_from_quant(model_info.get("raw_config", {})):
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

        # One class for both regimes: ``__init__`` branches on
        # ``model_config.moe_comm_backend`` (set by the enumerator) for large EP.
        return cls(*moe_args, *base_args, extra_params, backend_name=backend_name)

    #: TRT-LLM large-EP decode PDL overlap discount, transcribed from the
    #: deleted ``TrtllmWideEPDeepSeekV32Model._pdl_factor`` (deepseek_v32.py:463
    #: at commit 8372e60). Scales every decode-layer op, attention included.
    _PDL_FACTOR = 0.9

    def _large_ep_moe_ops(self, phase: str, shape: MoEBlockShape, scale_factor: float) -> list:
        """MoE block for a large-EP config (``cfg.moe_comm_backend`` set).

        Workload-distribution strings transcribed from the deleted wideEP
        classes at commit 8372e60: sglang flattens the prefill alpha to 0.6
        under EPLB (deepseek_v32.py:740-751); trtllm keeps alpha 1.01 in both
        phases and marks EPLB with the ``_eplb`` suffix (deepseek_v32.py:479-486).

        The shared-expert dtype is asymmetric in the legacy classes and is
        reproduced as such: trtllm sized its shared GEMMs with
        ``_dsa_shared_expert_quant_mode`` (deepseek_v32.py:536-555, 631-653),
        i.e. bf16 for checkpoints like ``nvidia/GLM-5.2-NVFP4`` that exclude
        ``mlp.shared_experts*`` from quantization, while sglang used the plain
        ``gemm_quant_mode`` (deepseek_v32.py:796-819) -- so the override is
        passed on trtllm only.
        """
        base = self.config.workload_distribution
        shared_gemm_quant_mode = None
        if self._backend_name == "trtllm":
            distribution = power_law_distribution(base, self._power_law_alpha, eplb_suffix=self.config.enable_eplb)
            shared_gemm_quant_mode = _dsa_shared_expert_quant_mode(self.extra_params, self.config.gemm_quant_mode)
        else:
            alpha = 0.6 if (phase == "context" and self.config.enable_eplb) else self._power_law_alpha
            distribution = power_law_distribution(base, alpha)
        return build_moe_block_ops(
            phase,
            shape,
            self.config,
            self.config.moe_quant_mode,
            distribution,
            scale_factor=scale_factor,
            backend_name=self._backend_name,
            inference_phase=phase,
            model_family=self.model_family,
            attn_cp_size=self.config.cp_size,
            gpus_per_node=self._gpus_per_node,
            shared_gemm_quant_mode=shared_gemm_quant_mode,
        )

    def __init__(self, topk: int, num_experts: int, moe_inter_size: int, *args, backend_name: str = "") -> None:
        super().__init__(*args)

        self._backend_name = backend_name
        # Large EP: see ``ModelConfig.moe_comm_backend`` (enumerator-owned).
        self._is_large_ep = bool(self.config.moe_comm_backend)
        # Node width is a hardware fact with no default: an unset value would
        # silently mis-price cross-node all-to-all (see large_ep_gpus_per_node).
        self._gpus_per_node = large_ep_gpus_per_node(self.config) if self._is_large_ep else 0

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
        dsa_attn_quant_modes, dsa_gemm_quant_mode = _dsa_attention_quant_modes(self.extra_params, gemm_quant_mode)
        workload_distribution = (
            self.config.workload_distribution + f"_{self._power_law_alpha}"
            if self.config.workload_distribution == "power_law"
            else self.config.workload_distribution
        )
        local_heads = self._num_heads // tp_size

        # MoE block shape (large-EP regime only; the fused spans below stay
        # hand-wired -- their generation dialect differs from the builder's).
        moe_shape = MoEBlockShape(
            hidden_size=h,
            moe_inter_size=self._moe_inter_size,
            topk=self._topk,
            num_experts=self._num_experts,
            # The legacy wideEP graphs model exactly one full-size shared expert
            # (WideEP ADP mode, shared_tp_size=1).
            num_shared_experts=1,
            # Descriptor-only: the builder scales by the model-owned scale_factor.
            num_moe_layers=self._num_layers,
        )
        if self._is_large_ep and backend_name == "trtllm":
            # ===== TRT-LLM large EP (wideEP) =====
            # Attention + non-MoE wiring transcribed verbatim from the deleted
            # TrtllmWideEPDeepSeekV32Model (deepseek_v32.py:488-710 at commit
            # 8372e60): the add_norms carry NO ``scale_num_tokens=cp_size`` (the
            # fused path's CP form) and the whole decode stack carries the PDL
            # discount.
            validate_trtllm_large_ep(
                attention_dp_size=attention_dp_size,
                moe_ep_size=moe_ep_size,
                topk=topk,
                num_experts=num_experts,
                wideep_num_slots=self.config.wideep_num_slots,
                enable_eplb=self.config.enable_eplb,
            )
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
                        attn_projection_quant_modes=dsa_attn_quant_modes,
                    ),
                    ops.ElementWise("context_add_norm_2", self._num_layers, 2 * h, 2 * h, 0.8),
                ]
            )
            self.context_ops.extend(self._large_ep_moe_ops("context", moe_shape, self._num_layers))
            self.context_ops.append(
                ops.GEMM(
                    "context_logits_gemm",
                    1,
                    self._vocab_size // tp_size,
                    h,
                    common.GEMMQuantMode.bfloat16,
                )
            )

            generation_scale = self._num_layers * self._mtp_scale_factor * self._PDL_FACTOR
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
                        attn_projection_quant_modes=dsa_attn_quant_modes,
                    ),
                    ops.ElementWise("generation_add_norm_2", generation_scale, 2 * h, 2 * h, 0.8),
                ]
            )
            self.generation_ops.extend(self._large_ep_moe_ops("generation", moe_shape, generation_scale))
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
            return

        if self._is_large_ep and backend_name == "sglang":
            # ===== sglang large-EP (deepep) =====
            # DSA attention + non-MoE wiring transcribed verbatim from the
            # deleted WideEPDeepSeekV32Model (deepseek_v32.py:754-868 at commit
            # 8372e60): TP all_gather/reduce_scatter around the DSA module and
            # NO embedding / add_norm / logits_gemm / P2P.
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
                        attn_projection_quant_modes=dsa_attn_quant_modes,
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
                ]
            )
            self.context_ops.extend(self._large_ep_moe_ops("context", moe_shape, self._num_layers))

            generation_scale = self._num_layers * self._mtp_scale_factor
            self.generation_ops.append(
                ops.GenerationDSAModule(
                    "generation_attention",
                    generation_scale,
                    local_heads,
                    kvcache_quant_mode,
                    dsa_gemm_quant_mode,
                    architecture=self.architecture,
                    index_topk_freq=self.extra_params.get("index_topk_freq", 1),
                    dsa_full_layer_fraction=self.extra_params.get("dsa_full_layer_fraction"),
                    attn_projection_quant_modes=dsa_attn_quant_modes,
                )
            )
            self.generation_ops.extend(self._large_ep_moe_ops("generation", moe_shape, generation_scale))
            return

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
                    attn_projection_quant_modes=dsa_attn_quant_modes,
                ),
                ops.ElementWise("context_add_norm_2", self._num_layers, 2 * h, 2 * h, 0.8, scale_num_tokens=cp_size),
            ]
        )

        fused_context_moe_ops = [
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
        ]
        if self._is_large_ep:
            # Large EP on a framework without its own attention stack (see the
            # generation site below).
            self.context_ops.extend(self._large_ep_moe_ops("context", moe_shape, self._num_layers))
        else:
            self.context_ops.extend(fused_context_moe_ops)
        self.context_ops.append(
            ops.GEMM(
                "context_logits_gemm",
                1,
                self._vocab_size // tp_size,
                h,
                common.GEMMQuantMode.bfloat16,
            )
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
                    attn_projection_quant_modes=dsa_attn_quant_modes,
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

        if self._is_large_ep:
            # Large EP on a framework without its own attention stack (the
            # sglang/trtllm branches returned above).
            self.generation_ops.extend(
                self._large_ep_moe_ops("generation", moe_shape, self._num_layers * self._mtp_scale_factor)
            )
        else:
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
