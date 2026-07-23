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
from functools import cache
from typing import Optional

from aiconfigurator_core.sdk import common, config
from aiconfigurator_core.sdk.utils import (
    get_model_config_from_model_path,
    parse_compressed_tensors_quant,
)

logger = logging.getLogger(__name__)

_MOE_MODEL_FAMILIES = {
    "MOE",
    "DEEPSEEK",
    "DEEPSEEKV32",
    "DEEPSEEKV4",
    "KIMIK25",
    "HYBRIDMOE",
    "QWEN3VL_MOE",
    "GEMMA4MIX",
    "MINIMAXM3",
}


def attention_op_keys(model_family: str, backend_name: str, enable_wideep: bool = False) -> tuple[str, str]:
    """(context_op, generation_op) support-matrix keys for a model family /
    backend / wideep combination.

    Single source of truth shared by ``task_v2.Task`` (resolve-time FMHA
    fallback + validate) and the estimate-path FMHA guard
    (:func:`resolve_context_fmha_by_data`). Every context op keys on fmha at
    the top level; every generation op keys on kv dtype.
    """
    if model_family == "DEEPSEEKV4":
        return "deepseek_v4_context_module", "deepseek_v4_generation_module"
    if model_family == "DEEPSEEKV32":
        return "dsa_context_module", "dsa_generation_module"
    if model_family in ("DEEPSEEK", "KIMIK25") and backend_name != "vllm":
        if enable_wideep:
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


@cache
def _get_model_info(model_path: str) -> dict:
    """
    Get model configuration info from model path.

    Args:
        model_path: HuggingFace model path (e.g., 'meta-llama/Llama-2-7b-hf') or local path

    Returns:
        dict: Model configuration parameters and raw config under "raw_config".
    """
    return get_model_config_from_model_path(model_path)


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
        "nvfp4": "nvfp4",
        "fp4": "nvfp4",
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
        return "fp8"
    if num_bits == 4 and "float" in weight_type:
        return "nvfp4"
    return None


def _is_routing_expert_target(target: object) -> bool:
    """Return whether a ModelOpt target path points at routed MoE experts."""
    target_name = str(target).lower()
    if "shared_expert" in target_name:
        return False
    return ".experts." in target_name or "routing_expert" in target_name


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

    if "fp8" in gemm_algos:
        if quant_dynamic is not True:
            overrides["gemm_quant_mode"] = common.GEMMQuantMode.fp8_static
        else:
            overrides["gemm_quant_mode"] = common.GEMMQuantMode.fp8
    elif "nvfp4" in gemm_algos:
        overrides["gemm_quant_mode"] = common.GEMMQuantMode.nvfp4

    if "nvfp4" in moe_algos:
        overrides["moe_quant_mode"] = common.MoEQuantMode.nvfp4
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
    if architecture == "DeepseekV4ForCausalLM" and str(raw_config.get("expert_dtype", "")).lower() == "fp4":
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
    recognized. FP8-only requants (sgl-project) have no ``expert_dtype`` and
    return False.
    """
    info = _get_model_info(model_path)
    if info.get("architecture") != "DeepseekV4ForCausalLM":
        return False
    return str(info.get("raw_config", {}).get("expert_dtype") or "").lower() == "fp4"


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
    if mode is not None:
        model_config.moe_quant_mode = mode


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
