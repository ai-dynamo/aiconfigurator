# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve the AIC-backed model, dtype, and template capability profile."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from aiconfigurator.sdk import common
from aiconfigurator.sdk.models.helpers import _infer_quant_modes_from_raw_config
from aiconfigurator.sdk.perf_database import (
    context_fmha_supported_modes,
    get_database,
    get_latest_database_version,
)
from aiconfigurator.sdk.utils import _attach_inferred_quant_fields

from .model_capability import ResolvedModelConfig, load_model_config, resolve_attention_source

TEMPLATE_VERSION = 1


def _enum_name(value: object) -> str:
    return str(getattr(value, "name", value))


def _architecture(config: dict[str, object], declared: str | None) -> str | None:
    architectures = config.get("architectures") or []
    if isinstance(architectures, list) and architectures:
        return str(architectures[0])
    value = config.get("architecture") or declared
    return str(value) if value else None


def _is_moe(config: dict[str, object]) -> bool:
    return any(int(config.get(key) or 0) > 1 for key in ("n_routed_experts", "num_local_experts", "num_experts"))


def _attention_template(
    *,
    config: dict[str, object],
    architecture: str | None,
    model_family: str | None,
    is_moe: bool,
) -> tuple[str, str]:
    lowered = (architecture or "").lower()
    if model_family == "DEEPSEEKV4":
        return "dsv4_module", "moe_dsv4"
    if model_family == "MINIMAXM3":
        # MiniMax-M3's MSA op has no standalone silicon table. AIC transfers
        # utilization from DSA at the same (batch, suffix, prefix) coordinate.
        return "dsa_module", "moe_msa"
    if "dsa" in lowered or model_family == "DEEPSEEKV32":
        return "dsa_module", "moe_dsa"
    if any(config.get(key) is not None for key in ("kv_lora_rank", "q_lora_rank")) or model_family in {
        "DEEPSEEK",
        "DEEPSEEKV4",
        "KIMIK25",
    }:
        return "mla_module", "moe_mla" if is_moe else "dense_mla"
    query_heads = int(config.get("num_attention_heads") or 1)
    kv_heads = int(config.get("num_key_value_heads") or query_heads)
    suffix = "mha" if query_heads == kv_heads else "gqa"
    return "dense_attention", f"{'moe' if is_moe else 'dense'}_{suffix}"


def _attention_ops(source_name: str) -> tuple[str, str]:
    return {
        "dsa_module": ("dsa_context_module", "dsa_generation_module"),
        "dsv4_module": ("deepseek_v4_context_module", "deepseek_v4_generation_module"),
        "mla_module": ("context_mla", "generation_mla"),
        "dense_attention": ("context_attention", "generation_attention"),
    }[source_name]


@dataclass(frozen=True, slots=True)
class ResolvedDTypeProfile:
    gemm_quant_mode: str
    moe_quant_mode: str
    # Scalar fmha fields describe the FIRST resolved kv dtype; the maps carry
    # the per-kv-slice resolution that cells and rows must use.
    fmha_quant_mode: str
    comm_quant_mode: str
    native_kv_cache_dtype: str
    kv_cache_dtypes: tuple[str, ...]
    fmha_resolution: str
    fmha_by_kv_dtype: dict[str, str]
    fmha_resolution_by_kv_dtype: dict[str, str]

    def to_dict(self) -> dict[str, object]:
        return {
            "gemm_quant_mode": self.gemm_quant_mode,
            "moe_quant_mode": self.moe_quant_mode,
            "fmha_quant_mode": self.fmha_quant_mode,
            "comm_quant_mode": self.comm_quant_mode,
            "native_kv_cache_dtype": self.native_kv_cache_dtype,
            "kv_cache_dtypes": list(self.kv_cache_dtypes),
            "fmha_resolution": self.fmha_resolution,
            "fmha_by_kv_dtype": dict(self.fmha_by_kv_dtype),
            "fmha_resolution_by_kv_dtype": dict(self.fmha_resolution_by_kv_dtype),
        }


@dataclass(frozen=True, slots=True)
class ModelCapabilityProfile:
    architecture: str | None
    model_family: str | None
    is_moe: bool
    attention_source: str
    attention_kind: str
    support_level: str
    template_id: str
    template_version: int
    support_reason: str
    allow_pure_tp: bool
    aic_database_version: str
    model_config: ResolvedModelConfig
    dtype: ResolvedDTypeProfile

    def to_dict(self) -> dict[str, object]:
        return {
            "architecture": self.architecture,
            "model_family": self.model_family,
            "is_moe": self.is_moe,
            "attention_source": self.attention_source,
            "attention_kind": self.attention_kind,
            "support_level": self.support_level,
            "template_id": self.template_id,
            "template_version": self.template_version,
            "support_reason": self.support_reason,
            "allow_pure_tp": self.allow_pure_tp,
            "aic_database_version": self.aic_database_version,
            "model_config": self.model_config.to_dict(),
            "dtype": self.dtype.to_dict(),
        }


_KV_ALIASES = {
    "auto": "auto",
    "bf16": "bfloat16",
    "bfloat16": "bfloat16",
    "fp8": "fp8",
    "fp8_e4m3": "fp8",
    "int8": "int8",
}


def _normalize_kv_dtype(value: str) -> str:
    try:
        return _KV_ALIASES[value.lower()]
    except KeyError as error:
        raise ValueError(
            f"unsupported FPM KV-cache dtype {value!r}; AIC modes are "
            f"{sorted(mode.name for mode in common.KVCacheQuantMode)}"
        ) from error


def _supported_modes(supported: dict[str, Any], op: str) -> set[str]:
    return {str(value) for value in (supported.get(op) or ())}


def _require_mode(supported: dict[str, Any], op: str, mode: str) -> None:
    modes = _supported_modes(supported, op)
    if not modes:
        raise ValueError(f"AIC database does not declare dtype support for required op {op!r}")
    if mode not in modes:
        raise ValueError(f"AIC database does not support {op} dtype {mode!r}; supported modes: {sorted(modes)}")


def resolve_model_capability(
    *,
    backend: str,
    model_path: str,
    model_architecture: str | None,
    selected_ops: set[str],
    has_model_cases: bool,
    system: str,
    requested_weight_quantizations: tuple[str, ...],
    requested_kv_cache_dtypes: tuple[str, ...],
    model_config_path: str | None = None,
    database_version: str | None = None,
) -> ModelCapabilityProfile:
    """Resolve a conservative matrix profile from AIC code and perf data."""

    resolved_config = load_model_config(model_path, explicit_config_path=model_config_path)
    config = resolved_config.effective_payload
    _attach_inferred_quant_fields(config)
    architecture = _architecture(config, model_architecture)
    model_family = common.ARCHITECTURE_TO_MODEL_FAMILY.get(architecture) if architecture else None
    exact_source = resolve_attention_source(selected_ops, required=False)
    # DSA is currently an MoE-only exact source in AIC. MLA is not: dense MLA
    # models must remain on the dense topology path.
    is_moe = _is_moe(config) or exact_source == "dsa_module"
    template_source, template_kind = _attention_template(
        config=config,
        architecture=architecture,
        model_family=model_family,
        is_moe=is_moe,
    )
    # The dense-attention op pair is inherited by every ``include_base: true``
    # model file, so its presence in selected_ops is not exact evidence by
    # itself. When the family template deliberately routes attention away from
    # the dense path (e.g. MiniMax-M3 MSA transfers utilization from DSA), the
    # family routing wins; a dense exact source is authoritative only for
    # families that are themselves dense.
    if exact_source == "dense_attention" and template_source != "dense_attention":
        exact_source = None
    if has_model_cases and model_family and exact_source is not None:
        support_level = "exact"
        source_name = exact_source
        attention_kind = {
            "dsa_module": "moe_dsa",
            "dsv4_module": "moe_dsv4",
            "mla_module": "moe_mla" if is_moe else "dense_mla",
            "dense_attention": template_kind,
        }[source_name]
        template_id = f"aic_exact:{source_name}"
        support_reason = "collector model cases and AIC architecture registry both match"
    elif model_family:
        support_level = "family_template"
        source_name = template_source
        attention_kind = template_kind
        template_id = f"aic_family:{model_family.lower()}:{template_kind}"
        support_reason = "no exact collector model case; derived from the registered AIC model family"
    else:
        support_level = "bootstrap_template"
        source_name = template_source
        attention_kind = template_kind
        template_id = f"generic:{template_kind}"
        support_reason = "architecture is not registered by AIC; using a conservative config-derived template"

    inferred = _infer_quant_modes_from_raw_config(config, architecture)
    gemm = _enum_name(inferred.get("gemm_quant_mode", common.GEMMQuantMode.bfloat16))
    moe = _enum_name(inferred.get("moe_quant_mode", common.MoEQuantMode.bfloat16))
    native_kv = _enum_name(inferred.get("kvcache_quant_mode", common.KVCacheQuantMode.bfloat16))
    inferred_fmha = _enum_name(inferred.get("fmha_quant_mode", common.FMHAQuantMode.bfloat16))

    requested_weights = {value.lower() for value in requested_weight_quantizations}
    if requested_weights and requested_weights != {gemm}:
        raise ValueError(
            "one --model-path represents one checkpoint artifact; requested weight quantization must match "
            f"its AIC-inferred GEMM mode {gemm!r}, got {sorted(requested_weights)}"
        )

    version = database_version or get_latest_database_version(system=system, backend=backend)
    if not version:
        raise ValueError(f"no AIC database version is available for system={system!r}, backend={backend!r}")
    database = get_database(system, backend, version)
    if database is None:
        raise ValueError(f"failed to load AIC database for system={system!r}, backend={backend!r}, version={version!r}")
    supported = getattr(database, "supported_quant_mode", {}) or {}
    context_op, generation_op = _attention_ops(source_name)

    _require_mode(supported, "gemm", gemm)
    if is_moe:
        _require_mode(supported, "moe", moe)

    context_modes = _supported_modes(supported, context_op)
    if not context_modes:
        raise ValueError(f"AIC database has no context-attention dtype evidence for template op {context_op!r}")

    generation_modes = _supported_modes(supported, generation_op)
    if not generation_modes:
        raise ValueError(f"AIC database has no generation-attention dtype evidence for template op {generation_op!r}")
    requested_kv = tuple(dict.fromkeys(_normalize_kv_dtype(value) for value in requested_kv_cache_dtypes))
    resolved_kv = tuple(dict.fromkeys(native_kv if value == "auto" else value for value in requested_kv))
    unsupported_kv = [value for value in resolved_kv if value not in generation_modes]
    if unsupported_kv:
        raise ValueError(
            f"AIC database does not support KV-cache dtype(s) {unsupported_kv} for {generation_op}; "
            f"supported modes: {sorted(generation_modes)}"
        )

    # fmha_quant_mode names the AIC data slice a template transfers
    # utilization from, so it is resolved PER KV DTYPE against joint
    # (fmha, kv) evidence: the flat supported_quant_mode list unions fmha
    # keys across kv slices (see perf_database.context_fmha_supported_modes),
    # so a single flat-checked label could claim a slice that only exists
    # under a different kv dtype. Some module tables carry only a bfloat16
    # slice (e.g. DSV4), so a checkpoint-native mode without joint evidence
    # falls back to the bfloat16 slice — recorded per cell/row via
    # fmha_resolution so the physical checkpoint mode is never lost — and
    # a kv dtype with no usable slice at all fails loudly.
    fmha_by_kv: dict[str, str] = {}
    fmha_resolution_by_kv: dict[str, str] = {}
    for kv_name in resolved_kv:
        joint_modes = context_fmha_supported_modes(
            database, context_op, common.KVCacheQuantMode[kv_name]
        )
        if inferred_fmha in joint_modes:
            fmha_by_kv[kv_name] = inferred_fmha
            fmha_resolution_by_kv[kv_name] = "checkpoint_native"
        elif "bfloat16" in joint_modes:
            fmha_by_kv[kv_name] = "bfloat16"
            if inferred_fmha == "fp8" and kv_name != "fp8":
                # The fp8 FMHA path exists only with an fp8 KV cache — the SDK
                # checkpoint inference pairs them for the same reason — so
                # under a non-fp8 kv slice the engine itself dispatches the
                # bfloat16 attention path. This label follows that kv-coupled
                # dispatch; it is not a data-availability substitution.
                fmha_resolution_by_kv[kv_name] = f"kv_dtype_dispatch_from_{inferred_fmha}"
            else:
                # The checkpoint-native mode has no data under its own kv
                # slice: AIC transfers utilization from the bfloat16 slice
                # (e.g. DSV4 module tables carry only bfloat16 rows).
                fmha_resolution_by_kv[kv_name] = f"aic_data_fallback_from_{inferred_fmha}"
        else:
            raise ValueError(
                f"AIC database has no joint fmha evidence for ({inferred_fmha!r} fmha, "
                f"{kv_name!r} kv) on {context_op!r} and no bfloat16 slice to transfer from; "
                f"jointly supported fmha modes: {sorted(joint_modes)}"
            )
    primary_kv = resolved_kv[0]
    fmha = fmha_by_kv[primary_kv]
    fmha_resolution = fmha_resolution_by_kv[primary_kv]

    # vLLM folds MoE tensor parallelism into ordinary tensor parallelism, so
    # pure TP is a backend capability rather than an attention-family
    # capability. Enumerate it for every MoE checkpoint; the target runtime
    # remains authoritative for exact topology and quantization alignment.
    allow_pure_tp = is_moe and backend == "vllm"
    return ModelCapabilityProfile(
        architecture=architecture,
        model_family=model_family,
        is_moe=is_moe,
        attention_source=source_name,
        attention_kind=attention_kind,
        support_level=support_level,
        template_id=template_id,
        template_version=TEMPLATE_VERSION,
        support_reason=support_reason,
        allow_pure_tp=allow_pure_tp,
        aic_database_version=str(version),
        model_config=resolved_config,
        dtype=ResolvedDTypeProfile(
            gemm_quant_mode=gemm,
            moe_quant_mode=moe,
            fmha_quant_mode=fmha,
            comm_quant_mode=common.CommQuantMode.half.name,
            native_kv_cache_dtype=native_kv,
            kv_cache_dtypes=resolved_kv,
            fmha_resolution=fmha_resolution,
            fmha_by_kv_dtype=fmha_by_kv,
            fmha_resolution_by_kv_dtype=fmha_resolution_by_kv,
        ),
    )
