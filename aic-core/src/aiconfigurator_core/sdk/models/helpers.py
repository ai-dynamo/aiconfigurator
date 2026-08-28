# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Helper functions for the models package.

Lookup helpers (architecture → family, family → MoE-ness, etc.) and the
quantization-mode resolution that runs once per ``get_model()`` invocation.
"""

from __future__ import annotations

import dataclasses
import logging
import warnings
from functools import cache
from typing import TYPE_CHECKING, Optional

from aiconfigurator_core.sdk import common, config
from aiconfigurator_core.sdk.utils import (
    get_model_config_from_model_path,
    parse_compressed_tensors_quant,
)

if TYPE_CHECKING:
    from aiconfigurator_core.sdk.models.blocks.moe import MoEBlockShape

logger = logging.getLogger(__name__)

_MOE_MODEL_FAMILIES = {
    "MOE",
    "DEEPSEEK",
    "DEEPSEEKV32",
    "DEEPSEEKV4",
    "KIMIK25",
    "KIMIK3",
    "HYBRIDMOE",
    "QWEN3VL_MOE",
    "GEMMA4MIX",
    "MINIMAXM3",
    "STEP3P7",
}


def quant_exclude_patterns(raw_config: dict) -> list:
    """All module-exclusion globs a ModelOpt/HF quant config can carry.

    Reads both declaration formats kept on ``raw_config`` by
    ``get_model_config_from_model_path``: the checkpoint's own
    ``quantization_config`` (compressed-tensors ``ignore`` /
    ``modules_to_not_convert`` / ``exclude_modules``) and the retained
    ``hf_quant_config`` (ModelOpt ``exclude_modules`` / ``ignore``).
    """
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


# Attention projection groups an exclusion pattern can cover. Exclusion
# granularity is a per-checkpoint fact: Kimi-K2.5-NVFP4 and R1-0528-FP4
# exclude the whole block (``self_attn*``), while DeepSeek-V3.1/V3.2-NVFP4
# exclude only q/kv projections (V3.2 also the indexer) and KEEP o_proj
# quantized; DeepSeek's native FP8 checkpoints exclude nothing.
ATTENTION_PROJECTION_GROUPS = ("q", "kv", "o", "indexer")

# substrings -> group; checked only when the whole-block glob did not match
_PROJECTION_GROUP_MARKERS = {
    "q": ("q_a_proj", "q_b_proj", "q_proj"),
    "kv": ("kv_a_proj", "kv_b_proj", "kv_proj", "k_proj", "v_proj"),
    "o": ("o_proj",),
    "indexer": ("indexer",),
}


def attention_projection_exclusions(raw_config: dict) -> frozenset:
    """Which attention projection groups the checkpoint keeps unquantized.

    Returns a subset of :data:`ATTENTION_PROJECTION_GROUPS`. A pattern naming
    the whole block (``self_attn*`` / ``re:.*self_attn.*``) covers every
    group; otherwise groups are matched per projection name.
    """
    excluded: set = set()
    for pattern in quant_exclude_patterns(raw_config):
        p = str(pattern)
        if "self_attn" in p and not any(m in p for markers in _PROJECTION_GROUP_MARKERS.values() for m in markers):
            # whole-block glob (e.g. "model.layers.N.self_attn*", "re:.*self_attn.*")
            return frozenset(ATTENTION_PROJECTION_GROUPS)
        for group, markers in _PROJECTION_GROUP_MARKERS.items():
            if any(m in p for m in markers):
                excluded.add(group)
    return frozenset(excluded)


def attention_modules_excluded_from_quant(raw_config: dict) -> bool:
    """Whether the checkpoint keeps ANY self-attention projection unquantized.

    Coarse boolean retained for callers that only need "is anything excluded";
    per-projection consumers (weights accounting, per-GEMM perf keys) must use
    :func:`attention_projection_exclusions` — V3.1/V3.2-NVFP4 exclude q/kv but
    keep o_proj quantized, so one boolean cannot describe the block.
    """
    return bool(attention_projection_exclusions(raw_config))


def attention_op_keys(
    model_family: str,
    backend_name: str,
    is_large_ep: bool = False,
    *,
    enable_wideep: bool | None = None,
) -> tuple[str, str]:
    """(context_op, generation_op) support-matrix keys for a model family /
    backend / large-EP combination.

    Single source of truth shared by ``task_v2.Task`` (resolve-time FMHA
    fallback + validate) and the estimate-path FMHA guard
    (:func:`resolve_context_fmha_by_data`). Every context op keys on fmha at
    the top level; every generation op keys on kv dtype.
    """
    if enable_wideep is not None:
        warnings.warn(
            "'enable_wideep' is deprecated; use 'is_large_ep' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        is_large_ep = enable_wideep

    if model_family == "DEEPSEEKV4":
        return "deepseek_v4_context_module", "deepseek_v4_generation_module"
    if model_family == "DEEPSEEKV32":
        return "dsa_context_module", "dsa_generation_module"
    if model_family == "KIMIK3" and backend_name != "vllm":
        # Kimi-K3 full-attention layers are DeepSeek-geometry MLA (NoPE + output
        # gate are shape-neutral); KDA linear-attention layers query the kda op
        # table directly (not an attention support-matrix key). The vLLM path
        # prices MLA through the plain attention tables (MLA-as-attention, see
        # models/kimi_k3.py) so it falls through to the GQA keys below.
        return "context_mla", "generation_mla"
    if model_family in ("DEEPSEEK", "KIMIK25") and backend_name != "vllm":
        if is_large_ep:
            if backend_name == "sglang":
                return "wideep_context_mla", "wideep_generation_mla"
            # trtllm wideep context queries the granular context_mla table
            # directly (plain ContextMLA op, no module primary), so its
            # capability key is the granular-only slice set: the merged
            # context_mla list may contain module-only slices (incl.
            # cross-framework shared-layer rows) this path can never hit.
            return "context_mla_granular", "generation_mla"
        return "context_mla", "generation_mla"
    return "context_attention", "generation_attention"


def power_law_distribution(base: str, alpha: float, *, eplb_suffix: bool = False) -> str:
    """Model-owned workload-distribution string for a MoE block.

    ``power_law`` is the only distribution with an alpha axis; any other value
    (``uniform``, ...) passes through untouched. ``eplb_suffix`` appends the
    TRT-LLM large-EP ``_eplb`` marker — those tables carry EPLB-balanced rows
    under their own distribution key instead of correcting the token count.
    """
    if base != "power_law":
        return base
    return f"{base}_{alpha}_eplb" if eplb_suffix else f"{base}_{alpha}"


def large_ep_gpus_per_node(model_config: config.ModelConfig) -> int:
    """Node width for a large-EP config; raises when the enumerator omitted it.

    The comm node span (``nodes_for(moe_ep * moe_tp, gpus_per_node)``) is a
    hardware fact that the model classes have no other channel to — the legacy
    ops read it from ``database.system_spec["node"]["num_gpus_per_node"]`` at
    query time, while the large-EP ops take ``node_num`` at construction (the
    PR 1 op contract). A default would therefore be silently wrong rather than
    merely approximate: on a GB200 NVL72 (4 GPUs/node) an EP32 decode would be
    priced as a 4-node cross-rack all-to-all instead of an intra-node one. Only
    large-EP construction calls this; fused configs never need the value.

    Raises:
        ValueError: If ``num_gpus_per_node`` is unset.
    """
    value = model_config.num_gpus_per_node
    if value is None:
        raise ValueError("moe_comm_backend is set but num_gpus_per_node is not — the enumerator must set both")
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        # The coordinate selects the all-to-all performance slice: zero would
        # ZeroDivisionError in nodes_for, a negative or fractional value
        # silently mis-prices a custom/direct builder call.
        raise ValueError(f"num_gpus_per_node must be a positive integer (hardware fact), got {value!r}")
    return value


def build_large_ep_moe_ops(
    phase: str,
    shape: MoEBlockShape,
    cfg: config.ModelConfig,
    *,
    scale_factor: float,
    backend_name: str,
    model_family: str,
    power_law_alpha: float,
    gpus_per_node: int,
    shared_gemm_quant_mode: common.GEMMQuantMode | None = None,
) -> list:
    """One MoE block for a large-EP config (``cfg.moe_comm_backend`` set).

    The shared body of ``DeepSeekModel._large_ep_moe_ops`` and
    ``DeepSeekV32Model._large_ep_moe_ops``: resolve the model-owned
    workload-distribution string, then delegate to ``build_moe_block_ops``.
    ``shared_gemm_quant_mode`` is the one caller-specific difference — the
    DeepSeekV32 family passes its ``dsa_shared_expert_quant_mode`` on trtllm.

    Workload-distribution strings are transcribed from the deleted wideEP
    classes at commit 8372e60: sglang flattens the prefill alpha to 0.6 under
    EPLB (deepseek.py:1089-1101 / deepseek_v32.py:740-751), while trtllm keeps
    the model alpha in both phases and marks EPLB with the ``_eplb`` suffix
    instead (deepseek.py:626-636 / deepseek_v32.py:479-486).
    """
    # Deferred import: blocks/moe.py imports this module (check_is_moe) at
    # import time, so a top-level import here would be circular.
    from aiconfigurator_core.sdk.models.blocks.moe import build_moe_block_ops

    base = cfg.workload_distribution
    if backend_name == "trtllm":
        distribution = power_law_distribution(base, power_law_alpha, eplb_suffix=cfg.enable_eplb)
    else:
        alpha = 0.6 if (phase == "context" and cfg.enable_eplb) else power_law_alpha
        distribution = power_law_distribution(base, alpha)
    return build_moe_block_ops(
        phase,
        shape,
        cfg,
        cfg.moe_quant_mode,
        distribution,
        scale_factor=scale_factor,
        backend_name=backend_name,
        inference_phase=phase,
        model_family=model_family,
        attn_cp_size=cfg.cp_size,
        gpus_per_node=gpus_per_node,
        shared_gemm_quant_mode=shared_gemm_quant_mode,
    )


def validate_trtllm_large_ep(
    *,
    attention_dp_size: int,
    moe_ep_size: int,
    topk: int,
    num_experts: int,
    wideep_num_slots: int | None,
    enable_eplb: bool,
) -> None:
    """Validate a TRT-LLM large-EP (wideEP) config.

    Transcribed verbatim from the deleted ``TrtllmWideEPDeepSeekModel`` /
    ``TrtllmWideEPDeepSeekV32Model`` constructors (deepseek.py:638-678 at
    commit 8372e60), which derive the rules from TensorRT-LLM's WideEPMoE
    constraints (``fused_moe_wide_ep.py``).

    Raises:
        ValueError: attention DP disabled, EP size not above 1, or a
            ``wideep_num_slots`` value that EPLB cannot serve.
    """
    # 1. Attention DP must be enabled for WideEP
    if attention_dp_size <= 1:
        raise ValueError(
            f"WideEP requires attention_dp_size > 1, got {attention_dp_size}. Attention DP should be used with WideEP."
        )

    # 2. EP size must be > 1 for WideEP (parallel_size > 1)
    if moe_ep_size <= 1:
        raise ValueError(
            f"WideEP requires moe_ep_size > 1, got {moe_ep_size}. WideEP should only be enabled with parallel_size > 1."
        )

    # 3. EP size must be > top_k for AlltoAll to be effective
    # FIXME: this warning should make the comm mode fallback to NCCL!!
    if moe_ep_size <= topk:
        logger.warning(
            f"moe_ep_size ({moe_ep_size}) <= top_k ({topk}), "
            "AlltoAll communication will be disabled. Consider increasing moe_ep_size."
        )

    # 4. num_slots validation
    num_slots = wideep_num_slots if wideep_num_slots else num_experts

    # num_slots must be >= num_experts
    if num_slots < num_experts:
        raise ValueError(
            f"wideep_num_slots ({num_slots}) must be >= num_experts ({num_experts}). "
            "There should be at least num_experts slots in the model engine."
        )

    # When EPLB is off, num_slots must equal num_experts
    if not enable_eplb and num_slots != num_experts:
        raise ValueError(
            f"When enable_eplb=False, wideep_num_slots ({num_slots}) must equal "
            f"num_experts ({num_experts}). Redundant slots require EPLB to be enabled."
        )


@cache
def _get_model_info(model_path: str) -> dict:
    """
    Get model configuration info from model path.

    Args:
        model_path: HuggingFace model path (e.g., 'meta-llama/Llama-2-7b-hf') or local path

    Returns:
        dict: Model configuration parameters and raw config under "raw_config",
        plus the derived MoE fields "num_shared_experts" and "num_moe_layers"
        (both 0 for dense models).
    """
    # Shallow-copy: get_model_config_from_model_path is @cache'd, so mutating
    # its return value would poison every other caller's dict.
    info = dict(get_model_config_from_model_path(model_path))
    info["num_shared_experts"] = _derive_num_shared_experts(info)
    info["num_moe_layers"] = _derive_num_moe_layers(info)
    return info


def _moe_source_config(info: dict) -> dict:
    """Raw HF config dict the MoE keys are read from.

    Mirrors ``_parse_hf_config_json``: for multimodal architectures the LLM
    parameters (including all MoE fields) live under a nested text config.
    """
    raw_config = info.get("raw_config")
    raw_config = raw_config if isinstance(raw_config, dict) else {}
    text_key = common.MULTIMODAL_TEXT_CONFIG_KEY.get(info.get("architecture"))
    nested = raw_config.get(text_key) if text_key else None
    return nested if isinstance(nested, dict) else raw_config


def _derive_num_shared_experts(info: dict) -> int:
    """Shared (always-active) expert count derived from the HF config.

    Sources, in order: an explicit count (``n_shared_experts`` /
    ``num_shared_experts``; DeepSeek/GLM/Kimi/MiniMax/NemotronH spelling), else
    a shared-expert intermediate size (``shared_expert_intermediate_size`` /
    ``moe_shared_expert_intermediate_size``; Qwen-MoE spelling) expressed as a
    multiple of the routed-expert intermediate size, else 0.
    """
    raw_config = _moe_source_config(info)
    for key in ("n_shared_experts", "num_shared_experts"):
        count = raw_config.get(key)
        # Explicit ``null`` (e.g. MiMo-V2-Flash) means "no shared experts".
        if count is not None:
            return int(count)
    shared_inter_size = (
        raw_config.get("shared_expert_intermediate_size") or raw_config.get("moe_shared_expert_intermediate_size") or 0
    )
    moe_inter_size = info.get("moe_inter_size") or 0
    if shared_inter_size and moe_inter_size:
        return max(1, int(shared_inter_size) // int(moe_inter_size))
    return 0


def _derive_num_moe_layers(info: dict) -> int:
    """Number of transformer layers that carry an MoE block; 0 for dense models.

    Honors the per-layer patterns the config parser already normalized into
    ``extra_params`` (NemotronH hybrid pattern; MiMo/Llama4 ``moe_layer_freq`` /
    ``interleave_moe_layer_step``), else derives from ``num_hidden_layers``
    minus ``first_k_dense_replace``, honoring a raw ``moe_layer_freq`` when
    present (HF DeepSeek semantics: layer ``i`` is MoE iff ``i >=
    first_k_dense_replace and i % moe_layer_freq == 0``).
    """
    if not info.get("num_experts") or not info.get("topk"):
        return 0
    extra_params = info.get("extra_params")
    if isinstance(extra_params, common.NemotronHConfig):
        return extra_params.hybrid_override_pattern.count("E")
    if isinstance(extra_params, common.HybridMoEConfig):
        return sum(1 for is_moe in extra_params.moe_layer_freq if is_moe)
    raw_config = _moe_source_config(info)
    layers = int(info["layers"])
    first_k_dense = int(raw_config.get("first_k_dense_replace") or 0)
    moe_layer_freq = raw_config.get("moe_layer_freq", 1)
    if isinstance(moe_layer_freq, (list, tuple)):
        return sum(1 for is_moe in moe_layer_freq if is_moe)
    moe_layer_freq = int(moe_layer_freq) if moe_layer_freq else 1
    if moe_layer_freq <= 1:
        return max(0, layers - first_k_dense)
    return sum(1 for layer_idx in range(first_k_dense, layers) if layer_idx % moe_layer_freq == 0)


def _architecture_to_model_family(architecture: str) -> str:
    """
    Convert architecture name to model family.
    Handles both HuggingFace architecture names (e.g., 'LlamaForCausalLM')
    and internal model family names (e.g., 'LLAMA').
    """
    if architecture in common.ARCHITECTURE_TO_MODEL_FAMILY:
        return common.ARCHITECTURE_TO_MODEL_FAMILY[architecture]
    if architecture in common.ModelFamily:
        return architecture
    raise ValueError(
        f"Unknown architecture or model family: {architecture}. "
        f"Supported architectures: {', '.join(common.ARCHITECTURE_TO_MODEL_FAMILY.keys())}. "
        f"Supported model families: {', '.join(common.ModelFamily)}."
    )


def _normalize_mixed_precision_layer_algo(value: object) -> str | None:
    """Normalize ModelOpt per-layer/group quantization algorithms."""
    if value is None:
        return None
    algo = str(value).strip().lower()
    if not algo:
        return None
    aliases = {
        "fp8": "fp8",
        "e4m3": "fp8",
        "e5m2": "fp8",
        # Kept distinct from "fp8": MXFP8 is 32-element block-scaled FP8, so
        # it must land on the fp8_block SDK modes, not the per-tensor
        # fp8_static/fp8 pair the "fp8" branch resolves to.
        "mxfp8": "mxfp8",
        "nvfp4": "nvfp4",
        "fp4": "nvfp4",
        "w4a16_nvfp4": "w4a16_nvfp4",
    }
    return aliases.get(algo, algo)


def _infer_mixed_precision_group_algo(group: dict) -> str | None:
    """Infer the layer algorithm from a ModelOpt config_groups entry."""
    weights = group.get("weights")
    if not isinstance(weights, dict):
        return None

    quant_algo = _normalize_mixed_precision_layer_algo(
        weights.get("quant_algo") or weights.get("quantization_algo") or weights.get("quant_method")
    )
    if quant_algo:
        return quant_algo

    num_bits = weights.get("num_bits")
    weight_type = str(weights.get("type", "")).lower()
    if not isinstance(num_bits, int):
        return None
    if num_bits == 8 and "float" in weight_type:
        # An 8-bit float group carrying 32-element block scales is MXFP8
        # (OCP MX block size), not per-tensor FP8 — without this check the
        # num_bits fallback would route such groups onto the fp8_static/fp8
        # per-tensor modes. Real ModelOpt MXFP8 exports seen so far
        # (nvidia/MiniMax-M3-NVFP4) declare the algo string per layer in
        # quantized_layers instead of config_groups, so this branch is a
        # guard for config_groups-style exports.
        if weights.get("group_size") == 32:
            return "mxfp8"
        return "fp8"
    if num_bits == 4 and "float" in weight_type:
        return "nvfp4"
    return None


def _is_routing_expert_target(target: object) -> bool:
    """Return whether a ModelOpt target path points at routed MoE experts."""
    target_name = str(target).lower()
    if "shared_expert" in target_name:
        return False
    return ".experts." in target_name or target_name.endswith(".experts") or "routing_expert" in target_name


def _collect_mixed_precision_layer_algos(raw_config: dict) -> tuple[set[str], set[str]]:
    """Collect ModelOpt mixed precision algorithms by SDK GEMM/MoE category."""
    gemm_algos: set[str] = set()
    moe_algos: set[str] = set()

    def add_target(target: object, algo_value: object) -> None:
        algo = _normalize_mixed_precision_layer_algo(algo_value)
        if algo is None:
            return
        if _is_routing_expert_target(target):
            moe_algos.add(algo)
        else:
            gemm_algos.add(algo)

    def add_quantized_layers(quantized_layers: object) -> None:
        if not isinstance(quantized_layers, dict):
            return
        for target, metadata in quantized_layers.items():
            if isinstance(metadata, dict):
                algo = metadata.get("quant_algo") or metadata.get("quantization_algo") or metadata.get("quant_method")
            else:
                algo = metadata
            add_target(target, algo)

    hf_quant_config = raw_config.get("hf_quant_config")
    if isinstance(hf_quant_config, dict):
        hf_quant_section = hf_quant_config.get("quantization")
        if isinstance(hf_quant_section, dict):
            add_quantized_layers(hf_quant_section.get("quantized_layers"))

    quant_cfg = raw_config.get("quantization_config")
    if isinstance(quant_cfg, dict):
        add_quantized_layers(quant_cfg.get("quantized_layers"))
        config_groups = quant_cfg.get("config_groups")
        if isinstance(config_groups, dict):
            for group in config_groups.values():
                if not isinstance(group, dict):
                    continue
                algo = _infer_mixed_precision_group_algo(group)
                if algo is None:
                    continue
                for target in group.get("targets") or []:
                    add_target(target, algo)

    return gemm_algos, moe_algos


def _infer_mixed_precision_quant_modes(raw_config: dict, quant_dynamic: bool | None) -> dict[str, object]:
    """Map ModelOpt MIXED_PRECISION metadata onto coarse SDK quant modes."""
    gemm_algos, moe_algos = _collect_mixed_precision_layer_algos(raw_config)
    overrides: dict[str, object] = {}

    # Some expert-only requants retain the base checkpoint's non-expert lane
    # in the mixed-precision header. Prefer that explicit base description to
    # broad config-group targets such as ``Linear``: the latter are filtered
    # by ``ignore`` at runtime and must not reclassify attention/shared GEMMs.
    quant_cfg = raw_config.get("quantization_config")
    base_quant_method = str(quant_cfg.get("quant_method", "")).lower() if isinstance(quant_cfg, dict) else ""
    weight_block_size = quant_cfg.get("weight_block_size") if isinstance(quant_cfg, dict) else None
    if base_quant_method == "fp8" and weight_block_size:
        overrides["gemm_quant_mode"] = common.GEMMQuantMode.fp8_block
    elif base_quant_method == "fp8":
        overrides["gemm_quant_mode"] = (
            common.GEMMQuantMode.fp8 if quant_dynamic is True else common.GEMMQuantMode.fp8_static
        )
    # MXFP8 -> fp8_block approximation (owner decision 2026-08-09): MXFP8 is
    # 32-element block-scaled FP8, i.e. the same 1 B/elem weights and the same
    # FP8 tensor-core throughput class as fp8_block; only the scale
    # granularity differs (32 vs 128), which is an accepted approximation.
    # Precedent: DeepSeek-V4's MXFP8-activation attention projections are
    # priced under gemm=fp8_block in the dsv4 module tables, and the shipped
    # MiniMax-M3 MSA tables carry the fp8_block gemm tier this lane resolves
    # to (a plain-fp8 mapping would miss that silicon). Checked before plain
    # fp8 so a checkpoint mixing per-tensor-FP8 and MXFP8 dense layers keeps
    # the conservative block-scaled lane. First seen on
    # nvidia/MiniMax-M3-NVFP4 (attention/dense-MLP/shared-expert projections
    # MXFP8, routed experts NVFP4).
    elif "mxfp8" in gemm_algos:
        overrides["gemm_quant_mode"] = common.GEMMQuantMode.fp8_block
    elif "fp8" in gemm_algos:
        if quant_dynamic is not True:
            overrides["gemm_quant_mode"] = common.GEMMQuantMode.fp8_static
        else:
            overrides["gemm_quant_mode"] = common.GEMMQuantMode.fp8
    elif "w4a16_nvfp4" in gemm_algos:
        overrides["gemm_quant_mode"] = common.GEMMQuantMode.w4a16_nvfp4
    elif "nvfp4" in gemm_algos:
        overrides["gemm_quant_mode"] = common.GEMMQuantMode.nvfp4

    # nvfp4-class stays first: routed experts are the artifact's headline
    # dtype when mixed with fp8-class expert layers. mxfp8 takes the same
    # fp8_block approximation as the GEMM side, ahead of per-tensor fp8.
    if "w4a16_nvfp4" in moe_algos:
        overrides["moe_quant_mode"] = common.MoEQuantMode.w4a16_nvfp4
    elif "nvfp4" in moe_algos:
        overrides["moe_quant_mode"] = common.MoEQuantMode.nvfp4
    elif "mxfp8" in moe_algos:
        overrides["moe_quant_mode"] = common.MoEQuantMode.fp8_block
    elif "fp8" in moe_algos:
        overrides["moe_quant_mode"] = common.MoEQuantMode.fp8

    if not overrides:
        logger.warning("Unable to infer SDK quant modes from mixed_precision metadata")

    return overrides


def _infer_quant_modes_from_raw_config(raw_config: dict, architecture: str | None = None) -> dict[str, object]:
    quant_algo = raw_config.get("quant_algo")
    quant_dynamic = raw_config.get("quant_dynamic")
    kv_cache_algo = raw_config.get("kv_cache_quant_algo")
    if architecture is None:
        architectures = raw_config.get("architectures") or []
        architecture = architectures[0] if architectures else raw_config.get("architecture")

    overrides: dict[str, object] = {}

    # GEMM quant mode, MoE quant mode
    if quant_algo == "fp8":
        # Non-block per-tensor FP8 is static unless the checkpoint explicitly
        # marks dynamic activations.  Dynamic FP8 always tags itself
        # (activation_scheme="dynamic" or config_groups[*].dynamic=true -> quant_dynamic=True),
        # and block-scaled FP8 is classified as fp8_block above, so treating the
        # unknown case (quant_dynamic is None) as static correctly covers
        # ModelOpt/NVIDIA per-tensor FP8 checkpoints that ship no activation scheme.
        if quant_dynamic is not True:
            overrides["gemm_quant_mode"] = common.GEMMQuantMode.fp8_static
        else:
            overrides["gemm_quant_mode"] = common.GEMMQuantMode.fp8
        overrides["moe_quant_mode"] = common.MoEQuantMode.fp8
    elif quant_algo == "fp8_block":
        overrides["gemm_quant_mode"] = common.GEMMQuantMode.fp8_block
        overrides["moe_quant_mode"] = common.MoEQuantMode.fp8_block
    elif quant_algo == "nvfp4":
        overrides["gemm_quant_mode"] = common.GEMMQuantMode.nvfp4
        overrides["moe_quant_mode"] = common.MoEQuantMode.nvfp4
    elif quant_algo == "mixed_precision":
        overrides.update(_infer_mixed_precision_quant_modes(raw_config, quant_dynamic))
    elif quant_algo == "mxfp4":
        overrides["gemm_quant_mode"] = common.GEMMQuantMode.bfloat16
        overrides["moe_quant_mode"] = common.MoEQuantMode.w4a16_mxfp4
    elif quant_algo == "compressed-tensors":
        # Parse the quantization_config to find which layer categories are quantized.
        # Only set overrides for quantized categories; unset modes fall through to the
        # global bfloat16 default in _apply_model_quant_defaults.
        quant_cfg = raw_config.get("quantization_config") or {}
        base_algo, ignored = parse_compressed_tensors_quant(quant_cfg)
        if base_algo:
            if "attention" not in ignored:
                overrides["gemm_quant_mode"] = getattr(common.GEMMQuantMode, base_algo)
            if "routing_experts" not in ignored:
                overrides["moe_quant_mode"] = getattr(common.MoEQuantMode, base_algo)
    elif quant_algo is not None:
        raise ValueError(f"Unsupported quant algorithm: {quant_algo}")

    # DeepSeek-V4 native checkpoints use MXFP4 routed-expert weights with MXFP8
    # activations, while non-expert weights remain FP8 block quantized.
    if (
        architecture == "DeepseekV4ForCausalLM"
        and str(raw_config.get("expert_dtype", "")).lower() == "fp4"
        and overrides.get("moe_quant_mode") != common.MoEQuantMode.nvfp4
    ):
        overrides["moe_quant_mode"] = common.MoEQuantMode.w4a8_mxfp4_mxfp8

    # KVCache quant mode
    # TODO: support fp4 kv cache
    if kv_cache_algo == "fp8":
        overrides["kvcache_quant_mode"] = common.KVCacheQuantMode.fp8
    elif kv_cache_algo == "bfloat16":
        overrides["kvcache_quant_mode"] = common.KVCacheQuantMode.bfloat16
    elif kv_cache_algo is not None:
        raise ValueError(f"Unsupported kv cache algorithm: {kv_cache_algo}")

    # DSV4 sparse attention requires FP8 KV cache across all backends.
    if architecture == "DeepseekV4ForCausalLM":
        overrides["kvcache_quant_mode"] = common.KVCacheQuantMode.fp8

    # FMHA quant mode
    if quant_algo is not None and (quant_algo in ("fp8", "fp8_block", "nvfp4") or kv_cache_algo in ("fp8",)):
        overrides["fmha_quant_mode"] = common.FMHAQuantMode.fp8
        if kv_cache_algo is None or kv_cache_algo != "fp8":
            overrides["kvcache_quant_mode"] = common.KVCacheQuantMode.fp8

    return overrides


# Architectures whose vLLM MoE dispatch is PROVEN w4a4 for W4A16_NVFP4-labeled
# checkpoints: the collected kernel_source is
# ``vllm_compressedtensorsw4a4nvfp4moe_*`` (activations quantized at run time,
# FP4 tensor cores), so the storage label misdescribes the execution lane.
# Scoped per-architecture deliberately: Qwen3.6-style checkpoints stay on the
# weight-only ``w4a16_nvfp4`` profile, which vLLM reaches through HYBRID's
# calibrated XPROFILE relation by design (see
# test_validate_w4a16_nvfp4_moe_xprofile_reachable_in_hybrid).
_VLLM_W4A4_MOE_ARCHITECTURES = frozenset({"NemotronHForCausalLM"})


def resolve_vllm_moe_execution_mode(
    moe_quant_mode: common.MoEQuantMode | None,
    backend_name: str | None,
    architecture: str | None,
) -> common.MoEQuantMode | None:
    """Map an HF/label-derived MoE mode to the lane vLLM actually executes.

    For NemotronH checkpoints, vLLM's compressed-tensors dispatch quantizes
    activations at run time and serves the FP4-tensor-core (w4a4) kernels
    regardless of the ``W4A16_NVFP4`` ModelOpt storage label (the collected
    kernel_source proves it), so the mode remaps to ``nvfp4`` — the lane the
    silicon rows live in. trtllm/sglang keep the label mode (their
    weight-only dequant-to-BF16 MoE path is real), and non-NemotronH
    architectures keep it on vLLM too (Qwen3.6's profile is served via the
    HYBRID XPROFILE relation by design). Callers apply this to HF-DERIVED
    modes only — a truly explicit user mode must bypass it
    (AIC-1748/AIC-1743).
    """
    if (
        backend_name == "vllm"
        and architecture in _VLLM_W4A4_MOE_ARCHITECTURES
        and moe_quant_mode == common.MoEQuantMode.w4a16_nvfp4
    ):
        return common.MoEQuantMode.nvfp4
    return moe_quant_mode


def _apply_model_quant_defaults(
    model_config: config.ModelConfig,
    raw_config: dict,
    architecture: str,
    backend_name: str,
    worker_name: Optional[str] = None,
) -> None:
    # Clone original model_config to track if any modifications were made
    original_config = dataclasses.replace(model_config)
    fmha_was_unset = model_config.fmha_quant_mode is None
    moe_was_unset = model_config.moe_quant_mode is None

    inferred = _infer_quant_modes_from_raw_config(raw_config, architecture)
    applied: list[str] = []
    for key, value in inferred.items():
        if getattr(model_config, key, None) is None:
            setattr(model_config, key, value)
            applied.append(f"{key}={value.name}")

    if model_config.gemm_quant_mode is None:
        model_config.gemm_quant_mode = common.GEMMQuantMode.bfloat16
    if model_config.moe_quant_mode is None:
        model_config.moe_quant_mode = common.MoEQuantMode.bfloat16
    if model_config.kvcache_quant_mode is None:
        model_config.kvcache_quant_mode = common.KVCacheQuantMode.bfloat16
    if model_config.fmha_quant_mode is None:
        model_config.fmha_quant_mode = common.FMHAQuantMode.bfloat16
    if model_config.comm_quant_mode is None:
        model_config.comm_quant_mode = common.CommQuantMode.half

    if applied:
        logger.debug("Using model-provided quantization defaults: %s", ", ".join(applied))

    # Legacy FMHA guards for direct get_model() callers that bypass Task /
    # estimate-path resolution.  They apply ONLY when fmha arrived unset (i.e.
    # this function inferred it from the checkpoint just above): a value that
    # arrived already resolved is owned by Task._resolve_quant_modes or
    # resolve_context_fmha_by_data, whose data-driven decision must not be
    # overridden here.
    if fmha_was_unset:
        # DSA module (DeepSeek-V3.2 / GLM-5): DSA perf tables only have bfloat16 FMHA currently.
        if (
            architecture in ("DeepseekV32ForCausalLM", "GlmMoeDsaForCausalLM")
            and backend_name in ("trtllm", "sglang")
            and model_config.fmha_quant_mode == common.FMHAQuantMode.fp8
        ):
            model_config.fmha_quant_mode = common.FMHAQuantMode.bfloat16

        # DeepSeek-V4 compressed attention collectors record the attention module as
        # BF16 even when projections/KV cache are quantized.
        if architecture == "DeepseekV4ForCausalLM" and model_config.fmha_quant_mode == common.FMHAQuantMode.fp8:
            model_config.fmha_quant_mode = common.FMHAQuantMode.bfloat16

        # FIXME: temporary workaround for Qwen3 32B FP8, only bfloat16+fp8kvcache is supported
        # VLLM perf tables only include bfloat16 FMHA; fall back to bfloat16 for estimation.
        if backend_name == "vllm" and model_config.fmha_quant_mode == common.FMHAQuantMode.fp8:
            model_config.fmha_quant_mode = common.FMHAQuantMode.bfloat16

    # Inferred-mode-only remap (see resolve_vllm_moe_execution_mode): an
    # explicit user mode wins, and validate fails fast on it — the
    # explicit-is-the-user's-contract doctrine.
    if moe_was_unset:
        model_config.moe_quant_mode = resolve_vllm_moe_execution_mode(
            model_config.moe_quant_mode, backend_name, architecture
        )

    # Only log if model_config was modified
    if original_config != model_config:
        logger.info(
            "Resolved quant modes for %s: gemm=%s moe=%s kvcache=%s fmha=%s comm=%s",
            worker_name or architecture,
            model_config.gemm_quant_mode,
            model_config.moe_quant_mode,
            model_config.kvcache_quant_mode,
            model_config.fmha_quant_mode,
            model_config.comm_quant_mode,
        )


def _is_dsv4_fp4_expert_model(model_path: str) -> bool:
    """True for DeepSeek-V4 checkpoints with native FP4 routed experts.

    Checks ``expert_dtype == "fp4"`` in the HF config rather than hardcoded
    paths, so third-party requant artifacts (e.g. RedHatAI NVFP4-FP8) are
    recognized. Checkpoints with explicit NVFP4 routed-expert metadata are not
    native MXFP4 checkpoints even when the base config retains
    ``expert_dtype == "fp4"``. FP8-only requants (sgl-project) have no
    ``expert_dtype`` and return False.
    """
    info = _get_model_info(model_path)
    if info.get("architecture") != "DeepseekV4ForCausalLM":
        return False
    raw_config = info.get("raw_config", {})
    if str(raw_config.get("expert_dtype") or "").lower() != "fp4":
        return False
    _gemm_algos, moe_algos = _collect_mixed_precision_layer_algos(raw_config)
    return "nvfp4" not in moe_algos


def resolve_dsv4_moe_arch_mode(
    model_path: str,
    system_name: str | None,
    backend_name: str | None,
    moe_backend: str | None = None,
) -> common.MoEQuantMode | None:
    """Arch-specific MoE quant mode for FP4-expert DeepSeek-V4 checkpoints on sglang.

    SGLang serves FP4-expert V4 checkpoints through arch-specific MoE kernels,
    and the perf DB files those rows under dedicated quant modes (loader
    routing in ``operations/moe.py``): Blackwell -> ``w4a8_mxfp4_mxfp8_trtllm``
    (kernel_source ``sglang_mxfp4_flashinfer_trtllm_moe``), Hopper ->
    ``w4a16_mxfp4_cutlass`` (``sglang_flashinfer_cutlass_moe``). Returns the
    remapped mode, or None when the rule does not apply (non-sglang backends,
    megamoe, FP8-only requant artifacts, other systems). An explicit user mode
    must win, so callers only apply this when moe_quant_mode was not explicitly
    set.
    """
    if backend_name != "sglang" or moe_backend == "megamoe":
        return None
    if not _is_dsv4_fp4_expert_model(model_path):
        return None
    from aiconfigurator_core.sdk.perf_database import is_blackwell_system, is_hopper_system

    if is_blackwell_system(system_name):
        return common.MoEQuantMode.w4a8_mxfp4_mxfp8_trtllm
    if is_hopper_system(system_name):
        return common.MoEQuantMode.w4a16_mxfp4_cutlass
    return None


def resolve_kimi_k3_moe_arch_mode(
    model_path: str,
    system_name: str | None,
    backend_name: str | None,
) -> common.MoEQuantMode | None:
    """Arch-specific MoE quant mode for Kimi-K3's MXFP4 routed experts on sglang.

    The kimi-k3 branch's ``Mxfp4MoEMethod`` default precision quantizes
    activations to mxfp8 on Blackwell (``per_token_group_quant`` /
    ``mxfp8_quantize`` -> ``trtllm_fp4_block_scale_moe``, mxfp4.py:1311-1330 @
    kimi-k3 branch), so the perf DB files those rows under
    ``w4a8_mxfp4_mxfp8``. Hopper serves the same checkpoint through the
    bf16-activation marlin W4A16 lane — the checkpoint's plain
    ``w4a16_mxfp4`` label is already correct there, so this returns None and
    the HF auto-inference stands.
    """
    if backend_name != "sglang":
        return None
    if model_path != "moonshotai/Kimi-K3":
        return None
    from aiconfigurator_core.sdk.perf_database import is_blackwell_system

    if is_blackwell_system(system_name):
        return common.MoEQuantMode.w4a8_mxfp4_mxfp8
    return None


def resolve_dsv4_moe_arch(
    model_config: config.ModelConfig,
    model_path: str,
    *,
    system_name: str | None,
    backend_name: str | None,
    moe_backend: str | None = None,
) -> None:
    """In-place ModelConfig variant for the estimate path.

    Mirrors ``resolve_context_fmha_by_data``'s contract: a non-None
    ``moe_quant_mode`` is treated as user-explicit and wins. Must be called
    BEFORE ``get_model`` so the arch mode lands before HF auto-inference
    resolves the plain label.
    """
    if model_config.moe_quant_mode is not None:
        return
    mode = resolve_dsv4_moe_arch_mode(model_path, system_name, backend_name, moe_backend)
    if mode is None:
        mode = resolve_kimi_k3_moe_arch_mode(model_path, system_name, backend_name)
    if mode is not None:
        model_config.moe_quant_mode = mode


def resolve_nvfp4_for_system(
    model_config: config.ModelConfig,
    system_name: str | None,
    model_path: str | None = None,
    *,
    backend_name: str | None = None,
) -> None:
    """Remap native nvfp4 to weight-only nvfp4_wo on non-Blackwell systems.

    Non-Blackwell hardware has no native FP4 tensor cores, so all runtimes
    dequantize weights to BF16 before the MMA. nvfp4_wo captures this:
    memory=9/16 (FP4 weight traffic) and compute=1 (BF16 speed). The
    transfer ladder then models the Marlin-class memory savings via the
    (0.5625, 1) util-level entry — no direct bfloat16 table aliasing needed.

    When the mode is inferred from a checkpoint label, resolve its backend
    execution lane before applying the hardware fallback. Deployability
    (whether a runtime can load the checkpoint at a given version) is a
    separate question handled by the support matrix.
    """
    from aiconfigurator_core.sdk.perf_database import is_blackwell_system

    if is_blackwell_system(system_name):
        return  # native FP4 TC — nvfp4 stays as-is

    gemm_q = model_config.gemm_quant_mode
    moe_q = model_config.moe_quant_mode
    if (gemm_q is None or moe_q is None) and model_path:
        info = _get_model_info(model_path)
        architecture = info.get("architecture", "")
        inferred = _infer_quant_modes_from_raw_config(
            info.get("raw_config", {}),
            architecture,
        )
        if gemm_q is None:
            gemm_q = inferred.get("gemm_quant_mode")
        if moe_q is None:
            moe_q = resolve_vllm_moe_execution_mode(inferred.get("moe_quant_mode"), backend_name, architecture)

    if gemm_q == common.GEMMQuantMode.nvfp4:
        model_config.gemm_quant_mode = common.GEMMQuantMode.nvfp4_wo
    if moe_q == common.MoEQuantMode.nvfp4:
        model_config.moe_quant_mode = common.MoEQuantMode.nvfp4_wo


def resolve_context_fmha_by_data(
    model_config: config.ModelConfig,
    model_path: str,
    database,
    backend_name: str,
    *,
    is_context_role: bool,
) -> None:
    """Data-driven context-FMHA guard for the estimate path (no Task involved).

    Must be called BEFORE ``get_model`` so the resolution lands before the
    model (and its attention ops) are built. Mirrors the resolve-time fallback
    in ``task_v2.Task._resolve_quant_modes``, driven by the perf DB's
    fmha-keyed context table instead of a hand-written architecture list:

    * Generation-only roles: no-op (no generation table keys on fmha).
    * fp8 slice present, or no DB information for the op: no-op.
    * fmha explicitly set to fp8 with no fp8 slice: raise a concise
      ``ValueError`` instead of a missing-perf-data traceback downstream.
    * fmha auto-inferred to fp8 with no fp8 slice but a bf16 one: downgrade
      to bfloat16 with a warning (predictions are conservative if the
      deployed engine runs fp8 FMHA).

    Args:
        model_config: ModelConfig whose ``fmha_quant_mode`` may be adjusted
            in place. A non-None value is treated as a user-explicit request.
        model_path: HF model path used to resolve family and quant config.
        database: the role's loaded perf database (its
            ``supported_quant_mode`` is consulted).
        backend_name: backend the estimate targets (selects the context op).
        is_context_role: True for context-attention roles (agg, prefill,
            static, static_ctx, AFD prefill); False for generation-only roles.
    """
    if not is_context_role:
        return

    from aiconfigurator_core.sdk.perf_database import context_fmha_supported_modes

    info = _get_model_info(model_path)
    family = _architecture_to_model_family(info["architecture"])
    inferred = _infer_quant_modes_from_raw_config(info.get("raw_config", {}), info.get("architecture"))
    # Joint (fmha, kv) capability: use the kv mode this estimate will actually
    # run with (explicit wins, else the checkpoint inference).
    kv_mode = model_config.kvcache_quant_mode or inferred.get("kvcache_quant_mode")
    ctx_op, _ = attention_op_keys(family, backend_name)
    supported = context_fmha_supported_modes(database, ctx_op, kv_mode)
    if not supported or common.FMHAQuantMode.fp8.name in supported:
        return

    if model_config.fmha_quant_mode is not None:
        if model_config.fmha_quant_mode == common.FMHAQuantMode.fp8:
            raise ValueError(
                f"fmha_quant_mode=fp8 has no {ctx_op!r} perf data for this system/backend/version. "
                "Use --fmha-quant-mode bfloat16, or omit --fmha-quant-mode to auto-select it."
            )
        return

    # Not user-explicit: mirror what get_model would infer, and downgrade only
    # when that inference would pick fp8 (bf16 checkpoints need no change).
    if inferred.get("fmha_quant_mode") == common.FMHAQuantMode.fp8 and common.FMHAQuantMode.bfloat16.name in supported:
        model_config.fmha_quant_mode = common.FMHAQuantMode.bfloat16
        logger.warning(
            "fmha_quant_mode=fp8 (inferred from the model checkpoint) has no %r perf data; "
            "falling back to bfloat16 FMHA data. Predictions are conservative if the deployed "
            "engine runs fp8 FMHA; set fmha_quant_mode explicitly to override.",
            ctx_op,
        )


def get_model_family(model_path: str) -> str:
    """
    Get model family.
    Converts architecture name to model family if needed.
    """
    architecture = _get_model_info(model_path)["architecture"]
    return _architecture_to_model_family(architecture)


def check_is_moe(model_path: str, model_info: dict | None = None) -> bool:
    """
    Check if the model is a MoE model.

    For NEMOTRONH models, checks if 'E' (MoE layer) is in hybrid_override_pattern..
    E.g., Nemotron_H is not an MoE model, but Nemotron_3 is an MoE model.
    """
    if model_info is None:
        model_info = _get_model_info(model_path)
    family = _architecture_to_model_family(model_info["architecture"])
    if family in _MOE_MODEL_FAMILIES:
        return True
    if family == "QWEN35":
        extra_params = model_info.get("extra_params")
        return isinstance(extra_params, common.Qwen35Config) and extra_params.num_experts > 0
    if family == "NEMOTRONH":
        extra_params = model_info.get("extra_params")
        if extra_params is None or not hasattr(extra_params, "hybrid_override_pattern"):
            logger.warning(f"NEMOTRONH model {model_path} missing hybrid_override_pattern, defaulting is_moe=False")
            return False
        # 'E' in pattern means MoE layers are present
        return "E" in extra_params.hybrid_override_pattern
    return False


def mtp_scale_factor(nextn: int, num_layers: int) -> float:
    """Per-iteration compute scale for MTP speculative decoding.

    A decode iteration evaluates ``num_layers + nextn`` layers' worth of work.
    Accepted-token progress is deliberately excluded: it is a workload-level
    assumption applied by the upper prediction layer.
    """
    if nextn <= 0:
        return 1.0
    return (nextn + num_layers) / num_layers
