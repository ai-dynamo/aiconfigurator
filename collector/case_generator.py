# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""YAML-backed collector case generation.

This module is the bridge between collector v2 YAML files and runnable
per-framework test cases. Base op YAML owns shared sweeps, model YAML owns
model-specific dimensions, and these helpers mechanically expand them into the
legacy tuple/dataclass shapes consumed by collector modules.
"""

import copy
import dataclasses
import functools
import itertools
import os
from pathlib import Path
from typing import Optional

import yaml

COLLECTOR_ROOT = Path(__file__).resolve().parent
BASE_OP_CASES_DIR = COLLECTOR_ROOT / "cases" / "base_ops"
MODEL_CASES_DIR = COLLECTOR_ROOT / "cases" / "models"


def _load_yaml_mapping(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise TypeError(f"{path}: top-level YAML value must be a mapping")
    return data


def _merge_base_case_data(target: dict, source: dict) -> None:
    target.setdefault("model_ops", [])
    target["model_ops"].extend(source.get("model_ops") or [])
    target.setdefault("common_case_values", {}).update(source.get("common_case_values") or {})
    target.setdefault("all_frameworks_op_cases", {}).update(source.get("all_frameworks_op_cases") or {})

    target_framework_cases = target.setdefault("framework_specific_op_cases", {})
    for backend, backend_cases in (source.get("framework_specific_op_cases") or {}).items():
        target_framework_cases.setdefault(backend, {}).update(backend_cases or {})


@functools.lru_cache(maxsize=1)
def _load_base_cases_data() -> dict:
    merged: dict = {}
    if not BASE_OP_CASES_DIR.exists():
        return merged

    for path in sorted(BASE_OP_CASES_DIR.glob("*.yaml")):
        _merge_base_case_data(merged, _load_yaml_mapping(path))
    return merged


def get_base_op_case_specs(op_name: str) -> list[dict[str, object]]:
    """Return dict-style case specs for a framework-agnostic base op."""
    try:
        cases = _load_base_cases_data().get("all_frameworks_op_cases", {}).get(op_name, {}).get("cases")
    except FileNotFoundError:
        return []
    if not isinstance(cases, list):
        return []
    return [case for case in cases if isinstance(case, dict)]


def get_base_framework_op_case_specs(backend: str, op_name: str) -> list[dict[str, object]]:
    """Return dict-style case specs for a backend-specific base op."""
    try:
        framework_cases = _load_base_cases_data().get("framework_specific_op_cases", {})
    except FileNotFoundError:
        return []
    cases = framework_cases.get(backend, {}).get(op_name, {}).get("cases")
    if not isinstance(cases, list):
        return []
    return [case for case in cases if isinstance(case, dict)]


def get_merged_base_op_case_specs(backend: str, op_name: str) -> list[dict[str, object]]:
    """Return base op specs with backend-specific overrides applied by case id."""
    merged_cases = [copy.deepcopy(case) for case in get_base_op_case_specs(op_name)]
    index_by_id = {case.get("id"): index for index, case in enumerate(merged_cases) if case.get("id")}

    for override in get_base_framework_op_case_specs(backend, op_name):
        override = copy.deepcopy(override)
        case_id = override.get("id")
        if case_id in index_by_id:
            merged_cases[index_by_id[case_id]].update(override)
        else:
            merged_cases.append(override)

    return merged_cases


def get_attention_context_shape_sweeps(backend: str) -> list[dict[str, object]]:
    """Return YAML-backed context attention shape sweeps for one backend."""
    return get_merged_base_op_case_specs(backend, "attention_context")


def get_attention_generation_shape_sweeps(backend: str) -> list[dict[str, object]]:
    """Return YAML-backed generation attention shape sweeps for one backend."""
    return get_merged_base_op_case_specs(backend, "attention_generation")


def get_attention_encoder_shape_sweeps(backend: str) -> list[dict[str, object]]:
    """Return YAML-backed encoder (non-causal) attention shape sweeps for one backend."""
    return get_merged_base_op_case_specs(backend, "encoder_attention")


@dataclasses.dataclass(frozen=True)
class AttentionHeadConfig:
    """One structural attention lookup key before the batch/sequence sweep."""

    num_heads: int
    num_kv_heads: int
    head_dim: int
    window_size: int


@dataclasses.dataclass(frozen=True)
class EncoderAttentionHeadConfig:
    """One encoder-attention structural lookup key before the workload sweep."""

    num_heads: int
    head_dim: int


def _profile_int_values(
    profile: dict[str, object],
    plural: str,
    singular: str,
    *,
    fallback: object = None,
) -> list[int]:
    value = profile.get(plural)
    if value is None and profile.get(singular) is not None:
        value = [profile[singular]]
    if value is None:
        value = fallback
    if not isinstance(value, list):
        raise TypeError(f"attention profile {plural} must be a list")
    return [int(item) for item in value]


def _head_profiles(
    shape_sweep: dict[str, object],
    op_name: str,
) -> list[dict[str, object]]:
    raw_profiles = shape_sweep.get("head_profiles")
    if raw_profiles is None:
        base_profiles = [shape_sweep]
    else:
        if not isinstance(raw_profiles, list) or not all(isinstance(profile, dict) for profile in raw_profiles):
            raise TypeError(f"{op_name}.head_profiles must be a list of mappings")
        base_profiles = raw_profiles

    model_profiles = _model_case_values(op_name)
    if _get_model_path_filter() and model_profiles:
        return model_profiles
    return [*base_profiles, *model_profiles]


def get_attention_head_configs(
    shape_sweep: dict[str, object],
    *,
    phase: str,
) -> list[AttentionHeadConfig]:
    """Expand only valid ``(q, kv, head_dim, window)`` structural tuples.

    Profiles may either describe a default rectangular sub-grid or one native
    model topology plus its valid tensor-parallel sizes. The latter preserves
    correlations between query heads, KV heads, head dimension, and window.
    """

    if phase not in {"context", "generation"}:
        raise ValueError(f"Unknown attention phase: {phase}")

    configs: list[AttentionHeadConfig] = []
    seen: set[AttentionHeadConfig] = set()

    def append(num_heads: int, num_kv_heads: int, head_dim: int, window_size: int) -> None:
        if num_heads <= 0 or num_kv_heads <= 0 or head_dim <= 0 or window_size < 0:
            return
        if num_kv_heads > num_heads or num_heads % num_kv_heads != 0:
            return
        config = AttentionHeadConfig(num_heads, num_kv_heads, head_dim, window_size)
        if config not in seen:
            seen.add(config)
            configs.append(config)

    for profile in _head_profiles(shape_sweep, "attention"):
        head_dims = _profile_int_values(
            profile,
            "head_dims",
            "head_dim",
            fallback=shape_sweep.get("head_dims"),
        )
        window_sizes = _profile_int_values(
            profile,
            "window_sizes",
            "window_size",
            fallback=shape_sweep.get("window_sizes", [0]),
        )

        native_num_heads = profile.get("num_attention_heads")
        if native_num_heads is not None:
            native_num_kv_heads = profile.get("num_key_value_heads")
            if native_num_kv_heads is None:
                raise ValueError("native attention profiles require num_key_value_heads")
            tp_sizes = _profile_int_values(
                profile,
                "tensor_parallel_sizes",
                "tensor_parallel_size",
            )
            native_num_heads = int(native_num_heads)
            native_num_kv_heads = int(native_num_kv_heads)
            for tp_size in tp_sizes:
                if tp_size <= 0 or native_num_heads % tp_size != 0:
                    continue
                num_heads = native_num_heads // tp_size
                num_kv_heads = (native_num_kv_heads + tp_size - 1) // tp_size
                for head_dim in head_dims:
                    for window_size in window_sizes:
                        append(num_heads, num_kv_heads, head_dim, window_size)
            continue

        if phase == "context":
            query_head_counts = _profile_int_values(
                profile,
                "query_head_counts",
                "query_head_count",
                fallback=shape_sweep.get("query_head_counts"),
            )
            kv_head_options = profile.get("kv_head_options", shape_sweep.get("kv_head_options"))
            if not isinstance(kv_head_options, list):
                raise TypeError("attention profile kv_head_options must be a list")
            for head_dim in head_dims:
                for num_heads in sorted(query_head_counts, reverse=True):
                    for raw_num_kv_heads in kv_head_options:
                        num_kv_heads = (
                            num_heads if raw_num_kv_heads in {"self", 0, "0", None} else int(raw_num_kv_heads)
                        )
                        if num_kv_heads != num_heads and (num_kv_heads >= num_heads or num_heads % num_kv_heads != 0):
                            continue
                        for window_size in window_sizes:
                            append(num_heads, num_kv_heads, head_dim, window_size)
            continue

        mha_query_head_counts = _profile_int_values(
            profile,
            "mha_query_head_counts",
            "mha_query_head_count",
            fallback=shape_sweep.get("mha_query_head_counts", []),
        )
        xqa_query_head_counts = _profile_int_values(
            profile,
            "xqa_query_head_counts",
            "xqa_query_head_count",
            fallback=shape_sweep.get("xqa_query_head_counts", []),
        )
        kv_head_counts = _profile_int_values(
            profile,
            "kv_head_counts",
            "kv_head_count",
            fallback=shape_sweep.get("kv_head_counts", []),
        )
        allow_xqa_mha = bool(profile.get("allow_xqa_mha", False))
        require_divisible = bool(profile.get("require_divisible", False))
        for head_dim in head_dims:
            for num_heads in sorted(mha_query_head_counts, reverse=True):
                for window_size in window_sizes:
                    append(num_heads, num_heads, head_dim, window_size)
            for num_heads in sorted(xqa_query_head_counts, reverse=True):
                for num_kv_heads in kv_head_counts:
                    if num_kv_heads > num_heads or (num_kv_heads == num_heads and not allow_xqa_mha):
                        continue
                    if require_divisible and num_heads % num_kv_heads != 0:
                        continue
                    for window_size in window_sizes:
                        append(num_heads, num_kv_heads, head_dim, window_size)
    return configs


def get_attention_encoder_head_configs(shape_sweep: dict[str, object]) -> list[EncoderAttentionHeadConfig]:
    """Expand valid ``(num_heads, head_dim)`` encoder-attention structures.

    Native profiles describe a ViT's unsharded head count and valid tensor
    parallel sizes. Duplicate physical keys are removed while preserving their
    first-seen order.
    """

    configs: list[EncoderAttentionHeadConfig] = []
    seen: set[EncoderAttentionHeadConfig] = set()

    def append(num_heads: int, head_dim: int) -> None:
        if num_heads <= 0 or head_dim <= 0:
            return
        config = EncoderAttentionHeadConfig(num_heads, head_dim)
        if config not in seen:
            seen.add(config)
            configs.append(config)

    for profile in _head_profiles(shape_sweep, "encoder_attention"):
        head_dims = _profile_int_values(
            profile,
            "head_dims",
            "head_dim",
            fallback=shape_sweep.get("head_dims"),
        )
        native_num_heads = profile.get("num_attention_heads")
        if native_num_heads is not None:
            native_num_heads = int(native_num_heads)
            tp_sizes = _profile_int_values(
                profile,
                "tensor_parallel_sizes",
                "tensor_parallel_size",
            )
            for tp_size in tp_sizes:
                if tp_size <= 0 or native_num_heads % tp_size != 0:
                    continue
                for head_dim in head_dims:
                    append(native_num_heads // tp_size, head_dim)
            continue

        head_counts = _profile_int_values(
            profile,
            "head_counts",
            "head_count",
            fallback=shape_sweep.get("head_counts"),
        )
        for head_dim in head_dims:
            for num_heads in sorted(head_counts):
                append(num_heads, head_dim)

    return configs


def get_base_common_case_values(name: str) -> dict[str, object]:
    """Return shared scalar/list values from base op case YAML files."""
    try:
        values = _load_base_cases_data().get("common_case_values", {}).get(name, {})
    except FileNotFoundError:
        return {}
    if values is None:
        return {}
    if not isinstance(values, dict):
        raise TypeError(f"common_case_values.{name} must be a mapping")
    return values


def _required_base_common_case_values(name: str) -> dict[str, object]:
    values = get_base_common_case_values(name)
    if not values:
        raise RuntimeError(f"{BASE_OP_CASES_DIR} is missing common_case_values.{name}")
    return values


def _get_model_path_filter() -> str | None:
    """Return the model-path filter from the environment, or None for 'all'."""
    val = os.environ.get("COLLECTOR_MODEL_PATH", "").strip()
    return val if val else None


@functools.lru_cache(maxsize=1)
def _load_model_cases_data() -> tuple[dict, ...]:
    data = []
    for path in sorted(MODEL_CASES_DIR.glob("*_cases.yaml")):
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        if not isinstance(raw, dict):
            raise TypeError(f"{path}: top-level YAML value must be a mapping")
        data.append(raw)
    return tuple(data)


def _expand_model_case_entry(raw_value: object, *, field_name: str) -> list[dict]:
    if not isinstance(raw_value, dict):
        raise TypeError(f"{field_name} entries must be mappings")
    value = dict(raw_value)
    raw_model_paths = value.pop("model_paths", None)
    raw_model_aliases = value.get("model_aliases")
    if raw_model_aliases is not None:
        if raw_model_paths is not None:
            raise ValueError(f"{field_name} entries cannot set both model_paths and model_aliases")
        if value.get("model_path") is None:
            raise ValueError(f"{field_name}.model_aliases requires model_path")
        value["model_aliases"] = _as_str_list(
            raw_model_aliases,
            field_name=f"{field_name}.model_aliases",
        )
        # Aliases share one physical collector case.  Keep the representative
        # model path instead of multiplying the same kernel shape by artifact
        # names such as base/FP8/NVFP4.
        return [value]
    if raw_model_paths is None:
        return [value]
    if value.get("model_path") is not None:
        raise ValueError(f"{field_name} entries cannot set both model_path and model_paths")
    return [
        {**value, "model_path": model_path}
        for model_path in _as_str_list(raw_model_paths, field_name=f"{field_name}.model_paths")
    ]


def _model_case_matches_path(value: dict, model_path: str) -> bool:
    return value.get("model_path") == model_path or model_path in (value.get("model_aliases") or [])


def _model_case_values(op_name: str, *, apply_model_filter: bool = True) -> list[dict]:
    values = []
    for data in _load_model_cases_data():
        # Carry the model file's architecture onto every expanded case dict so
        # downstream model-family checks use config-derived metadata (the
        # architecture in the cases YAML mirrors config.json's
        # ``architectures[0]``) rather than fragile model_name string patterns.
        arch = data.get("architecture")
        op_values = data.get("model_case_values", {}).get(op_name, [])
        if op_values is None:
            continue
        if isinstance(op_values, dict):
            expanded = _expand_model_case_entry(op_values, field_name=f"model_case_values.{op_name}")
        elif isinstance(op_values, list):
            expanded = []
            for index, item in enumerate(op_values):
                expanded.extend(_expand_model_case_entry(item, field_name=f"model_case_values.{op_name}[{index}]"))
        else:
            raise TypeError(f"model_case_values.{op_name} must be a list or mapping")
        for value in expanded:
            if arch is not None:
                value.setdefault("architecture", arch)
            values.append(value)

    model_path = _get_model_path_filter() if apply_model_filter else None
    if model_path:
        values = [value for value in values if _model_case_matches_path(value, model_path)]
    return values


def _framework_specific_model_case_values(op_name: str, backend: str, *, apply_model_filter: bool = True) -> list[dict]:
    values = []
    for data in _load_model_cases_data():
        framework_values = data.get("framework_specific_model_case_values", {})
        if not isinstance(framework_values, dict):
            raise TypeError("framework_specific_model_case_values must be a mapping")
        backend_values = framework_values.get(backend, {})
        if not isinstance(backend_values, dict):
            raise TypeError(f"framework_specific_model_case_values.{backend} must be a mapping")
        op_values = backend_values.get(op_name, [])
        if op_values is None:
            continue
        if isinstance(op_values, dict):
            values.extend(
                _expand_model_case_entry(
                    op_values,
                    field_name=f"framework_specific_model_case_values.{backend}.{op_name}",
                )
            )
            continue
        if not isinstance(op_values, list):
            raise TypeError(f"framework_specific_model_case_values.{backend}.{op_name} must be a list or mapping")
        for index, item in enumerate(op_values):
            values.extend(
                _expand_model_case_entry(
                    item,
                    field_name=f"framework_specific_model_case_values.{backend}.{op_name}[{index}]",
                )
            )

    model_path = _get_model_path_filter() if apply_model_filter else None
    if model_path:
        values = [value for value in values if _model_case_matches_path(value, model_path)]
    return values


@dataclasses.dataclass(frozen=True)
class MLAModuleModelSpec:
    """Model metadata used by full MLA/DSA module collectors."""

    model_path: str
    attention_type: str
    architecture: str
    native_num_heads: int
    wideep_mla: bool


@dataclasses.dataclass(frozen=True)
class MLAModuleSweepSpec:
    """Shared micro-sweep values for full MLA/DSA module collectors."""

    batch_sizes: list[int]
    sequence_lengths: list[int]
    context_batch_sizes: list[int]
    context_sequence_lengths: list[int]
    generation_batch_sizes: list[int]
    generation_sequence_lengths: list[int]
    inner_sweep_head_counts: list[int]
    top_level_head_counts: list[int]
    module_tp_sizes: list[int]
    module_precision_combos: list[tuple[str, str, str]]
    context_max_tokens: int
    context_large_sequence_min: int
    context_large_sequence_max_batch_size: int
    generation_max_tokens: int
    generation_large_sequence_min: int
    generation_large_sequence_max_batch_size: int
    generation_large_cache_tokens: int
    context_prefix_lengths: list[int]


@dataclasses.dataclass(frozen=True)
class MLAModulePrecisionSpec:
    """Precision combo metadata for full MLA/DSA module collectors."""

    compute_dtype: str
    kv_cache_dtype: str
    gemm_type: str
    phases: tuple[str, ...]
    min_sm: int


def get_mla_module_model_specs(
    attention_type: str | None = None,
    *,
    backend: str | None = None,
    wideep_mla: bool | None = None,
    apply_model_filter: bool = True,
) -> list[MLAModuleModelSpec]:
    """Return YAML-backed model metadata for full MLA/DSA module collectors."""

    values = _model_case_values("mla_module", apply_model_filter=False)
    if backend is not None:
        values.extend(
            _framework_specific_model_case_values(
                "mla_module",
                backend,
                apply_model_filter=False,
            )
        )

    specs = []
    model_path_filter = _get_model_path_filter() if apply_model_filter else None
    for value in values:
        if model_path_filter and not _model_case_matches_path(value, model_path_filter):
            continue
        if attention_type is not None and value.get("attention_type") != attention_type:
            continue
        if wideep_mla is not None and bool(value.get("wideep_mla", False)) != wideep_mla:
            continue
        specs.append(
            MLAModuleModelSpec(
                model_path=str(value["model_path"]),
                attention_type=str(value["attention_type"]),
                architecture=str(value["architecture"]),
                native_num_heads=int(value["native_num_heads"]),
                wideep_mla=bool(value.get("wideep_mla", False)),
            )
        )

    if backend == "vllm" and apply_model_filter and model_path_filter is None:
        # vLLM 0.24 builds every module with the case's explicit precision and
        # head count, so checkpoint aliases no longer change the invocation.
        # MLA has one architecture-less consumer table; DSA is keyed by
        # architecture. Stable first-wins keeps DeepSeek-V3 and each DSA
        # architecture canonical while targeted artifact runs remain exact.
        canonical_specs = {}
        for spec in specs:
            key = (spec.attention_type, spec.architecture if spec.attention_type == "dsa" else None)
            canonical_specs.setdefault(key, spec)
        specs = list(canonical_specs.values())

    return specs


def _required_mapping(value: object, *, field_name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError(f"{field_name} must be a mapping")
    return value


def _optional_int(value: object, *, default: int = 0) -> int:
    return default if value is None else int(value)


def _optional_int_list(value: object, *, field_name: str, default: list[int]) -> list[int]:
    if value is None:
        return list(default)
    return _as_int_list(value, field_name=field_name)


def _merged_mla_module_values(backend: str | None = None) -> dict[str, object]:
    values = _required_base_common_case_values("mla_module")
    if backend:
        override = get_base_common_case_values(f"mla_module_{backend}")
        if override:
            merged = copy.deepcopy(values)
            merged.update(override)
            if "context_batch_sizes" in override or "generation_batch_sizes" in override:
                merged.pop("batch_sizes", None)
            if "context_sequence_lengths" in override or "generation_sequence_lengths" in override:
                merged.pop("sequence_lengths", None)
            if "head_counts" in override:
                merged.pop("inner_sweep_head_counts", None)
                merged.pop("top_level_head_counts", None)
            values = merged
    return values


def get_mla_module_precision_specs(
    backend: str | None = None,
    *,
    phase: str | None = None,
    sm_version: int | None = None,
) -> list[MLAModulePrecisionSpec]:
    """Return YAML-backed precision combos for module collectors."""

    values = _merged_mla_module_values(backend)
    raw_precision_combos = values.get("module_precision_combos")
    if not isinstance(raw_precision_combos, list):
        raise TypeError("mla_module.module_precision_combos must be a list")

    precision_specs = []
    for combo in raw_precision_combos:
        if not isinstance(combo, dict):
            raise TypeError("mla_module.module_precision_combos entries must be mappings")
        phases = combo.get("phases", ("context", "generation"))
        if isinstance(phases, str):
            phases = (phases,)
        elif isinstance(phases, list):
            phases = tuple(str(item) for item in phases)
        elif not isinstance(phases, tuple):
            raise TypeError("mla_module.module_precision_combos phases must be a string or list")

        min_sm = int(combo.get("min_sm", 0))
        if phase is not None and phase not in phases:
            continue
        if sm_version is not None and sm_version < min_sm:
            continue
        precision_specs.append(
            MLAModulePrecisionSpec(
                compute_dtype=str(combo["compute_dtype"]),
                kv_cache_dtype=str(combo["kv_cache_dtype"]),
                gemm_type=str(combo["gemm_type"]),
                phases=phases,
                min_sm=min_sm,
            )
        )
    return precision_specs


def get_mla_module_sweep_spec(backend: str | None = None) -> MLAModuleSweepSpec:
    """Return YAML-backed shared micro-sweep values for module collectors."""

    values = _merged_mla_module_values(backend)

    context = _required_mapping(values.get("context"), field_name="mla_module.context")
    generation = _required_mapping(values.get("generation"), field_name="mla_module.generation")
    precision_combos = [
        (spec.compute_dtype, spec.kv_cache_dtype, spec.gemm_type) for spec in get_mla_module_precision_specs(backend)
    ]

    batch_sizes = _optional_int_list(values.get("batch_sizes"), field_name="mla_module.batch_sizes", default=[])
    sequence_lengths = _optional_int_list(
        values.get("sequence_lengths"),
        field_name="mla_module.sequence_lengths",
        default=[],
    )
    context_batch_sizes = _optional_int_list(
        values.get("context_batch_sizes"),
        field_name="mla_module.context_batch_sizes",
        default=batch_sizes,
    )
    context_sequence_lengths = _optional_int_list(
        values.get("context_sequence_lengths"),
        field_name="mla_module.context_sequence_lengths",
        default=sequence_lengths,
    )
    generation_batch_sizes = _optional_int_list(
        values.get("generation_batch_sizes"),
        field_name="mla_module.generation_batch_sizes",
        default=batch_sizes,
    )
    generation_sequence_lengths = _optional_int_list(
        values.get("generation_sequence_lengths"),
        field_name="mla_module.generation_sequence_lengths",
        default=sequence_lengths,
    )
    inner_sweep_head_counts = _optional_int_list(
        values.get("inner_sweep_head_counts", values.get("head_counts")),
        field_name="mla_module.inner_sweep_head_counts",
        default=[],
    )
    top_level_head_counts = _optional_int_list(
        values.get("top_level_head_counts"),
        field_name="mla_module.top_level_head_counts",
        default=inner_sweep_head_counts,
    )
    module_tp_sizes = _optional_int_list(
        values.get("module_tp_sizes"),
        field_name="mla_module.module_tp_sizes",
        default=[1],
    )

    return MLAModuleSweepSpec(
        batch_sizes=batch_sizes or context_batch_sizes,
        sequence_lengths=sequence_lengths or context_sequence_lengths,
        context_batch_sizes=context_batch_sizes,
        context_sequence_lengths=context_sequence_lengths,
        generation_batch_sizes=generation_batch_sizes,
        generation_sequence_lengths=generation_sequence_lengths,
        inner_sweep_head_counts=inner_sweep_head_counts,
        top_level_head_counts=top_level_head_counts,
        module_tp_sizes=module_tp_sizes,
        module_precision_combos=precision_combos,
        context_max_tokens=int(context["max_tokens"]),
        context_large_sequence_min=_optional_int(context.get("large_sequence_min")),
        context_large_sequence_max_batch_size=_optional_int(context.get("large_sequence_max_batch_size")),
        generation_max_tokens=int(generation["max_tokens"]),
        generation_large_sequence_min=_optional_int(generation.get("large_sequence_min")),
        generation_large_sequence_max_batch_size=_optional_int(generation.get("large_sequence_max_batch_size")),
        generation_large_cache_tokens=_optional_int(generation.get("large_cache_tokens")),
        context_prefix_lengths=_optional_int_list(
            context.get("prefix_lengths"),
            field_name="mla_module.context.prefix_lengths",
            default=[0],
        ),
    )


def is_wideep_moe_model(model_name: str) -> bool:
    """Return True if *model_name* needs WideEP MoE collection."""
    return any(
        _model_case_matches_path(value, model_name) and value.get("wideep")
        for value in _model_case_values("moe", apply_model_filter=False)
    )


def get_all_model_names() -> list[str]:
    """Return all known model names across all op types.

    Reads directly from model case YAML — does not instantiate test
    case objects or call generator functions, so pruning logic in the generators
    cannot accidentally exclude models from the allowlist.
    """

    def append_case_value_model_names(raw_values: object, *, field_name: str) -> None:
        if isinstance(raw_values, dict):
            values = _expand_model_case_entry(raw_values, field_name=field_name)
        elif isinstance(raw_values, list):
            values = []
            for index, item in enumerate(raw_values):
                values.extend(_expand_model_case_entry(item, field_name=f"{field_name}[{index}]"))
        else:
            return
        for value in values:
            if value.get("model_path"):
                model_names.append(str(value["model_path"]))
            model_names.extend(str(alias) for alias in value.get("model_aliases", []))

    model_names = []
    for data in _load_model_cases_data():
        primary = data.get("model_path")
        if primary:
            model_names.append(str(primary))
        model_names.extend(str(path) for path in data.get("model_paths", []) or [])
        for op_name, values in (data.get("model_case_values") or {}).items():
            append_case_value_model_names(values, field_name=f"model_case_values.{op_name}")
        for backend, op_values in (data.get("framework_specific_model_case_values") or {}).items():
            if not isinstance(op_values, dict):
                continue
            for op_name, values in op_values.items():
                append_case_value_model_names(
                    values,
                    field_name=f"framework_specific_model_case_values.{backend}.{op_name}",
                )

    deduped = []
    seen = set()
    for model_name in model_names:
        if model_name in seen:
            continue
        seen.add(model_name)
        deduped.append(model_name)
    return deduped


@dataclasses.dataclass
class MoeCommonTestCase:
    num_tokens_list: list[int]
    hidden_size: int
    inter_size: int
    topk: int
    num_experts: int
    tp: int
    ep: int
    model_name: str
    token_expert_distribution: str
    power_law_alpha: Optional[float]
    architecture: str = ""  # config-derived (cases-YAML architecture); for model-family checks


@dataclasses.dataclass(frozen=True)
class MoeQuantizationSpec:
    """YAML-backed MoE quantization mode selection metadata."""

    name: str
    min_sm: Optional[int]
    min_sm_exclusive: Optional[int]
    requires_runtime_feature: Optional[str]
    requires_model_quantization_config: bool
    allowed_model_paths: tuple[str, ...]
    module_config: dict[str, object]


def _moe_token_expert_distributions(moe_sweep: dict[str, object]) -> list[tuple[str, Optional[float]]]:
    raw_distributions = moe_sweep.get("token_expert_distributions")
    if not isinstance(raw_distributions, list):
        raise TypeError("common_case_values.moe.token_expert_distributions must be a list")

    distributions = []
    for item in raw_distributions:
        if not isinstance(item, dict):
            raise TypeError("common_case_values.moe.token_expert_distributions entries must be mappings")
        name = item.get("name") or item.get("distribution")
        if not name:
            raise ValueError("MoE token expert distribution entries need a name")
        alpha = item.get("power_law_alpha")
        distributions.append((str(name), None if alpha is None else float(alpha)))
    return distributions


def _moe_backend_values(backend: str) -> dict[str, object]:
    return get_base_common_case_values(f"moe_{backend}")


def get_moe_quantization_specs(backend: str) -> list[MoeQuantizationSpec]:
    """Return YAML-backed MoE quantization mode metadata for one backend."""

    values = _moe_backend_values(backend)
    raw_modes = values.get("quantization_modes", [])
    if not isinstance(raw_modes, list):
        raise TypeError(f"common_case_values.moe_{backend}.quantization_modes must be a list")

    specs = []
    for raw_mode in raw_modes:
        if not isinstance(raw_mode, dict):
            raise TypeError(f"common_case_values.moe_{backend}.quantization_modes entries must be mappings")
        module_config = raw_mode.get("module_config", {})
        if not isinstance(module_config, dict):
            raise TypeError(f"common_case_values.moe_{backend}.quantization_modes module_config must be a mapping")
        specs.append(
            MoeQuantizationSpec(
                name=str(raw_mode["name"]),
                min_sm=None if raw_mode.get("min_sm") is None else int(raw_mode["min_sm"]),
                min_sm_exclusive=(
                    None if raw_mode.get("min_sm_exclusive") is None else int(raw_mode["min_sm_exclusive"])
                ),
                requires_runtime_feature=(
                    None
                    if raw_mode.get("requires_runtime_feature") is None
                    else str(raw_mode["requires_runtime_feature"])
                ),
                requires_model_quantization_config=bool(raw_mode.get("requires_model_quantization_config", False)),
                allowed_model_paths=tuple(
                    _as_str_list(
                        raw_mode.get("allowed_model_paths", []),
                        field_name=f"moe_{backend}.quantization_modes.allowed_model_paths",
                    )
                ),
                module_config=dict(module_config),
            )
        )
    return specs


def get_moe_quantization_modes(
    backend: str,
    *,
    sm_version: int,
    runtime_version: str = "",
    runtime_features: dict[str, bool] | None = None,
) -> list[str]:
    """Return enabled MoE quantization modes after YAML SM/runtime-feature filtering."""

    features = runtime_features or {}
    modes = []
    for spec in get_moe_quantization_specs(backend):
        if spec.min_sm is not None and sm_version < spec.min_sm:
            continue
        if spec.min_sm_exclusive is not None and sm_version <= spec.min_sm_exclusive:
            continue
        if spec.requires_runtime_feature and not features.get(spec.requires_runtime_feature, False):
            continue
        modes.append(spec.name)
    return modes


def _model_moe_backend_quantization(model_name: str, backend: str) -> dict[str, object]:
    for model_case in _model_case_values("moe", apply_model_filter=False):
        if not _model_case_matches_path(model_case, model_name):
            continue
        raw_frameworks = model_case.get("frameworks")
        if raw_frameworks is not None and backend not in _as_str_list(
            raw_frameworks,
            field_name="model_case_values.moe.frameworks",
        ):
            continue
        framework_quantization = model_case.get("framework_quantization", {})
        if not isinstance(framework_quantization, dict):
            raise TypeError("model_case_values.moe.framework_quantization must be a mapping")
        backend_quantization = framework_quantization.get(backend, {})
        if backend_quantization is None:
            return {}
        if not isinstance(backend_quantization, dict):
            raise TypeError(f"model_case_values.moe.framework_quantization.{backend} must be a mapping")
        return dict(backend_quantization)
    return {}


def _model_quantization_modes(
    model_quantization: dict[str, object],
    field_name: str,
) -> list[str] | None:
    modes = model_quantization.get(field_name)
    if modes is None:
        return None
    return _as_str_list(modes, field_name=f"model_case_values.moe.framework_quantization.{field_name}")


def get_moe_quantization_module_config(
    backend: str,
    moe_type: str,
    *,
    model_name: str | None = None,
) -> dict[str, object]:
    """Return optional framework module config for a MoE quantization mode."""

    if model_name is not None:
        model_quantization = _model_moe_backend_quantization(model_name, backend)
        module_config = model_quantization.get("module_config", {})
        if not isinstance(module_config, dict):
            raise TypeError("model_case_values.moe.framework_quantization.module_config must be a mapping")
        mode_config = module_config.get(moe_type, {})
        if mode_config is None:
            return {}
        if not isinstance(mode_config, dict):
            raise TypeError(f"model_case_values.moe.framework_quantization.module_config.{moe_type} must be a mapping")
        if mode_config:
            return dict(mode_config)

    for spec in get_moe_quantization_specs(backend):
        if spec.name == moe_type:
            return dict(spec.module_config)
    return {}


def moe_model_allows_quantization(backend: str, model_name: str, moe_type: str) -> bool:
    """Return whether backend YAML allows a MoE quantization mode for a model."""

    model_quantization = _model_moe_backend_quantization(model_name, backend)
    for spec in get_moe_quantization_specs(backend):
        if spec.name != moe_type:
            continue
        if spec.allowed_model_paths and model_name not in spec.allowed_model_paths:
            return False
        if spec.requires_model_quantization_config and not model_quantization:
            return False
        break

    allowed_modes = _model_quantization_modes(model_quantization, "allowed_modes")
    if allowed_modes is not None and moe_type not in allowed_modes:
        return False
    excluded_modes = _model_quantization_modes(model_quantization, "excluded_modes")
    if excluded_modes is not None and moe_type in excluded_modes:
        return False

    values = _moe_backend_values(backend)
    raw_policies = values.get("model_quantization_policies", [])
    if not isinstance(raw_policies, list):
        raise TypeError(f"common_case_values.moe_{backend}.model_quantization_policies must be a list")

    for raw_policy in raw_policies:
        if not isinstance(raw_policy, dict):
            raise TypeError(f"common_case_values.moe_{backend}.model_quantization_policies entries must be mappings")
        model_paths = _as_str_list(
            raw_policy.get("model_paths", []),
            field_name=f"moe_{backend}.model_quantization_policies.model_paths",
        )
        if model_name not in model_paths:
            continue
        allowed_modes = raw_policy.get("allowed_modes")
        if allowed_modes is not None and moe_type not in _as_str_list(
            allowed_modes,
            field_name=f"moe_{backend}.model_quantization_policies.allowed_modes",
        ):
            return False
        excluded_modes = raw_policy.get("excluded_modes")
        if excluded_modes is not None and moe_type in _as_str_list(
            excluded_modes,
            field_name=f"moe_{backend}.model_quantization_policies.excluded_modes",
        ):
            return False
    return True


def _moe_backend_model_cases(backend: str) -> list[dict[str, object]]:
    values = _moe_backend_values(backend)
    raw_cases = values.get("model_cases", [])
    if not isinstance(raw_cases, list):
        raise TypeError(f"common_case_values.moe_{backend}.model_cases must be a list")

    model_path_filter = _get_model_path_filter()
    cases = []
    for raw_case in raw_cases:
        if not isinstance(raw_case, dict):
            raise TypeError(f"common_case_values.moe_{backend}.model_cases entries must be mappings")
        case = dict(raw_case)
        if model_path_filter and not _model_case_matches_path(case, model_path_filter):
            continue
        cases.append(case)

    for model_case in _model_case_values("moe"):
        framework_cases = model_case.get("framework_cases", {})
        if not isinstance(framework_cases, dict):
            raise TypeError("model_case_values.moe.framework_cases must be a mapping")
        backend_case = framework_cases.get(backend)
        if backend_case is None:
            continue
        if not isinstance(backend_case, dict):
            raise TypeError(f"model_case_values.moe.framework_cases.{backend} must be a mapping")
        case = dict(model_case)
        case.update(backend_case)
        case.pop("framework_cases", None)
        case.pop("framework_quantization", None)
        cases.append(case)

    for model_case in _framework_specific_model_case_values("moe", backend):
        if "framework_cases" in model_case:
            raise TypeError(
                f"framework_specific_model_case_values.{backend}.moe entries cannot contain framework_cases"
            )
        cases.append(model_case)
    return cases


def get_moe_backend_model_activation(backend: str, model_name: str, *, default: str = "silu") -> str:
    """Return YAML-backed activation metadata for a backend-specific MoE model."""

    for model_case in _moe_backend_model_cases(backend):
        if _model_case_matches_path(model_case, model_name):
            return str(model_case.get("activation", default))
    return default


def _moe_backend_token_expert_distributions(backend_values: dict[str, object]) -> list[tuple[str, Optional[float]]]:
    raw_distributions = backend_values.get("token_expert_distributions")
    if raw_distributions is None:
        raw_distributions = _required_base_common_case_values("moe").get("token_expert_distributions")
    return _moe_token_expert_distributions({"token_expert_distributions": raw_distributions})


def get_moe_backend_test_cases(backend: str) -> list[MoeCommonTestCase]:
    """Return YAML-backed backend-specific MoE model/sweep cases."""

    values = _moe_backend_values(backend)
    token_counts = _as_int_list(values.get("token_counts"), field_name=f"moe_{backend}.token_counts")
    raw_sweeps = values.get("sweeps")
    if not isinstance(raw_sweeps, dict):
        raise TypeError(f"common_case_values.moe_{backend}.sweeps must be a mapping")
    token_distributions = _moe_backend_token_expert_distributions(values)

    test_cases: list[MoeCommonTestCase] = []
    for model_config in _moe_backend_model_cases(backend):
        sweep_name = str(model_config.get("sweep", "default"))
        sweep = raw_sweeps.get(sweep_name)
        if not isinstance(sweep, dict):
            raise TypeError(f"common_case_values.moe_{backend}.sweeps.{sweep_name} must be a mapping")

        tp_list = _as_int_list(
            sweep.get("tensor_parallel_sizes"),
            field_name=f"moe_{backend}.sweeps.{sweep_name}.tensor_parallel_sizes",
        )
        ep_list = _as_int_list(
            sweep.get("expert_parallel_sizes"),
            field_name=f"moe_{backend}.sweeps.{sweep_name}.expert_parallel_sizes",
        )
        num_gpu_list = _as_int_list(
            sweep.get("gpu_counts"),
            field_name=f"moe_{backend}.sweeps.{sweep_name}.gpu_counts",
        )

        hs = int(model_config["hidden_size"])
        inter_s = int(model_config["inter_size"])
        topk = int(model_config["topk"])
        num_experts = int(model_config["num_experts"])
        model_name = str(model_config["model_path"])
        max_tp_exclusive = model_config.get("max_tp_exclusive")

        for num_gpu, tp, ep, (token_distribution, power_law_alpha) in itertools.product(
            num_gpu_list,
            tp_list,
            ep_list,
            token_distributions,
        ):
            if max_tp_exclusive is not None and tp >= int(max_tp_exclusive):
                continue
            if tp * ep != num_gpu:
                continue
            if ep > num_experts:
                continue
            if num_experts % ep != 0:
                continue
            if inter_s % tp != 0:
                continue

            test_cases.append(
                MoeCommonTestCase(
                    num_tokens_list=token_counts,
                    hidden_size=hs,
                    inter_size=inter_s,
                    topk=topk,
                    num_experts=num_experts,
                    tp=tp,
                    ep=ep,
                    model_name=model_name,
                    token_expert_distribution=token_distribution,
                    power_law_alpha=power_law_alpha,
                    architecture=str(model_config.get("architecture") or ""),
                )
            )
    return test_cases


def get_common_moe_test_cases(*, backend: str | None = None):
    moe_sweep = _required_base_common_case_values("moe")
    num_tokens = _as_int_list(moe_sweep.get("token_counts"), field_name="moe.token_counts")
    tp_list = _as_int_list(moe_sweep.get("tensor_parallel_sizes"), field_name="moe.tensor_parallel_sizes")
    ep_list = _as_int_list(moe_sweep.get("expert_parallel_sizes"), field_name="moe.expert_parallel_sizes")
    num_gpu_list = _as_int_list(moe_sweep.get("gpu_counts"), field_name="moe.gpu_counts")
    token_distributions = _moe_token_expert_distributions(moe_sweep)

    allowed_parallel_topologies = None
    if backend is not None:
        raw_topologies = _moe_backend_values(backend).get("parallel_topologies")
        if raw_topologies is not None:
            if not isinstance(raw_topologies, list):
                raise TypeError(f"common_case_values.moe_{backend}.parallel_topologies must be a list")
            allowed_parallel_topologies = set()
            for index, topology in enumerate(raw_topologies):
                if not isinstance(topology, dict):
                    raise TypeError(f"common_case_values.moe_{backend}.parallel_topologies[{index}] must be a mapping")
                topology_tps = _as_int_list(
                    topology.get("tensor_parallel_sizes"),
                    field_name=f"moe_{backend}.parallel_topologies[{index}].tensor_parallel_sizes",
                )
                topology_eps = _as_int_list(
                    topology.get("expert_parallel_sizes"),
                    field_name=f"moe_{backend}.parallel_topologies[{index}].expert_parallel_sizes",
                )
                allowed_parallel_topologies.update(itertools.product(topology_tps, topology_eps))

    model_config_list = []
    for model_config in _model_case_values("moe"):
        raw_frameworks = model_config.get("frameworks")
        if raw_frameworks is not None:
            frameworks = _as_str_list(raw_frameworks, field_name="model_case_values.moe.frameworks")
            if backend is not None and backend not in frameworks:
                continue
        model_config_list.append(model_config)

    test_cases: list[MoeCommonTestCase] = []

    for (
        num_gpu,  # starting from fewer gpus. workaround for potential buffer bug in moe impl.
        model_config,
        tp,
        ep,
        (token_distribution, power_law_alpha),
    ) in itertools.product(
        num_gpu_list,
        model_config_list,
        tp_list,
        ep_list,
        token_distributions,
    ):
        hs = int(model_config["hidden_size"])
        inter_s = int(model_config["inter_size"])
        topk = int(model_config["topk"])
        num_experts = int(model_config["num_experts"])
        model_name = str(model_config["model_path"])

        max_tp_exclusive = model_config.get("max_tp_exclusive")
        if max_tp_exclusive is not None and tp >= int(max_tp_exclusive):
            continue

        if allowed_parallel_topologies is not None and (tp, ep) not in allowed_parallel_topologies:
            continue

        if tp * ep != num_gpu:
            continue
        if ep > num_experts:
            continue
        if num_experts % ep != 0:
            continue
        # we need to ensure inter_s can be divided by tp.
        if inter_s % tp != 0:
            continue

        test_cases.append(
            MoeCommonTestCase(
                num_tokens_list=num_tokens,
                hidden_size=hs,
                inter_size=inter_s,
                topk=topk,
                num_experts=num_experts,
                tp=tp,
                ep=ep,
                model_name=model_name,
                token_expert_distribution=token_distribution,
                power_law_alpha=power_law_alpha,
                architecture=str(model_config.get("architecture") or ""),
            )
        )

    return test_cases


@dataclasses.dataclass
class GemmCommonTestCase:
    x: int
    n: int
    k: int


def _as_int_list(value, *, field_name: str) -> list[int]:
    if not isinstance(value, list):
        raise TypeError(f"{field_name} must be a list")
    return [int(item) for item in value]


def _as_str_list(value, *, field_name: str) -> list[str]:
    if not isinstance(value, list):
        raise TypeError(f"{field_name} must be a list")
    return [str(item) for item in value]


def _get_base_gemm_shape_sweeps(backend: str | None = None) -> list[dict[str, object]]:
    shape_sweeps = get_merged_base_op_case_specs(backend, "gemm") if backend else get_base_op_case_specs("gemm")
    if not shape_sweeps:
        raise RuntimeError(f"{BASE_OP_CASES_DIR} is missing all_frameworks_op_cases.gemm.cases")
    return shape_sweeps


def get_gemm_case_specs(backend: str | None = None) -> list[GemmCommonTestCase]:
    test_cases = []
    for shape_sweep in _get_base_gemm_shape_sweeps(backend):
        token_counts = _as_int_list(shape_sweep.get("token_counts"), field_name="gemm.token_counts")
        feature_sizes = shape_sweep.get("feature_sizes")
        input_feature_sizes = _as_int_list(
            shape_sweep.get("input_feature_sizes", feature_sizes),
            field_name="gemm.input_feature_sizes",
        )
        output_feature_sizes = _as_int_list(
            shape_sweep.get("output_feature_sizes", feature_sizes),
            field_name="gemm.output_feature_sizes",
        )
        skip_shapes = {
            (int(skip["output_features"]), int(skip["input_features"])) for skip in shape_sweep.get("skip_shapes", [])
        }

        for token_count in sorted(token_counts, reverse=True):
            for output_features in sorted(output_feature_sizes, reverse=True):
                for input_features in sorted(input_feature_sizes, reverse=True):
                    if (output_features, input_features) in skip_shapes:
                        continue
                    if output_features * input_features == 65536 * 65536:
                        continue
                    test_cases.append(GemmCommonTestCase(x=token_count, n=output_features, k=input_features))

    return test_cases


def get_gemm_type_specs(backend: str) -> list[str]:
    """Return YAML-backed GEMM dtype/quantization labels for a backend."""

    gemm_types = []
    seen = set()
    for shape_sweep in _get_base_gemm_shape_sweeps(backend):
        for gemm_type in shape_sweep.get("gemm_types", []):
            gemm_type = str(gemm_type)
            if gemm_type in seen:
                continue
            seen.add(gemm_type)
            gemm_types.append(gemm_type)
    return gemm_types


@dataclasses.dataclass
class ComputeScaleCommonTestCase:
    m: int
    k: int


def get_compute_scale_case_specs() -> list[ComputeScaleCommonTestCase]:
    shape_sweeps = get_base_framework_op_case_specs("trtllm", "compute_scale")
    if not shape_sweeps:
        seen_mk = set()
        test_cases = []
        for gemm_common_testcase in get_gemm_case_specs():
            key = (gemm_common_testcase.x, gemm_common_testcase.k)
            if key in seen_mk:
                continue
            seen_mk.add(key)
            test_cases.append(ComputeScaleCommonTestCase(m=key[0], k=key[1]))
        return test_cases

    test_cases = []
    for shape_sweep in shape_sweeps:
        token_counts = _as_int_list(shape_sweep.get("token_counts"), field_name="compute_scale.token_counts")
        input_feature_sizes = _as_int_list(
            shape_sweep.get("input_feature_sizes"),
            field_name="compute_scale.input_feature_sizes",
        )
        max_input_features = max(input_feature_sizes) if input_feature_sizes else None
        # Keep the largest K case last so collection does not start with the heaviest allocation.
        ordered_input_feature_sizes = sorted(
            input_feature_sizes,
            key=lambda input_features: (input_features == max_input_features, -input_features),
        )
        for token_count in sorted(token_counts, reverse=True):
            for input_features in ordered_input_feature_sizes:
                test_cases.append(ComputeScaleCommonTestCase(m=token_count, k=input_features))

    return test_cases


@dataclasses.dataclass
class MLACommonTestCase:
    num_heads: int
    batch_size: int
    input_len: int
    is_context_phase: bool
    kv_cache_block_size: int
    q_lora_rank: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    model_name: str


def _get_mla_case_specs(is_context: bool):
    test_cases = []
    seen = set()

    model_config_list = _model_case_values("mla")
    mla_sweep = _required_base_common_case_values("mla")

    if is_context:
        b_list = _as_int_list(mla_sweep.get("context_batch_sizes"), field_name="mla.context_batch_sizes")
        s_list = _as_int_list(mla_sweep.get("context_sequence_lengths"), field_name="mla.context_sequence_lengths")
        max_tokens = int(mla_sweep["max_context_tokens"])
    else:
        b_list = _as_int_list(mla_sweep.get("generation_batch_sizes"), field_name="mla.generation_batch_sizes")
        s_list = _as_int_list(
            mla_sweep.get("generation_target_sequence_lengths"),
            field_name="mla.generation_target_sequence_lengths",
        )
        max_tokens = int(mla_sweep["max_generation_tokens"])
    kv_cache_block_size_list = _as_int_list(
        mla_sweep.get("kv_cache_block_sizes"),
        field_name="mla.kv_cache_block_sizes",
    )

    for (
        s,
        b,
        kv_cache_block_size,
        model_config,
    ) in itertools.product(
        s_list,
        b_list,
        kv_cache_block_size_list,
        model_config_list,
    ):
        if b * s > max_tokens:
            continue

        input_len = s if is_context else s - 1
        physical_key = (
            int(model_config["num_heads"]),
            b,
            input_len,
            is_context,
            kv_cache_block_size,
            int(model_config["q_lora_rank"]),
            int(model_config["kv_lora_rank"]),
            int(model_config["qk_nope_head_dim"]),
            int(model_config["qk_rope_head_dim"]),
            int(model_config["v_head_dim"]),
        )
        if physical_key in seen:
            continue
        seen.add(physical_key)

        test_cases.append(
            MLACommonTestCase(
                num_heads=physical_key[0],
                input_len=input_len,
                batch_size=b,
                is_context_phase=is_context,
                kv_cache_block_size=kv_cache_block_size,
                q_lora_rank=int(model_config["q_lora_rank"]),
                kv_lora_rank=int(model_config["kv_lora_rank"]),
                qk_nope_head_dim=int(model_config["qk_nope_head_dim"]),
                qk_rope_head_dim=int(model_config["qk_rope_head_dim"]),
                v_head_dim=int(model_config["v_head_dim"]),
                model_name=str(model_config["model_path"]),
            )
        )

    return test_cases


def get_context_mla_case_specs():
    return _get_mla_case_specs(is_context=True)


def get_generation_mla_case_specs():
    return _get_mla_case_specs(is_context=False)


@dataclasses.dataclass
class MLABMMCommonTestCase:
    num_tokens: int
    num_heads: int
    dtype: str
    num_warmups: int
    num_runs: int


def get_mla_bmm_case_specs(backend: str, op_name: str) -> list[MLABMMCommonTestCase]:
    """Return YAML-backed MLA generation BMM helper shapes."""
    shape_sweeps = get_merged_base_op_case_specs(backend, op_name)
    if not shape_sweeps:
        raise RuntimeError(f"{BASE_OP_CASES_DIR} is missing all_frameworks_op_cases.{op_name}.cases")

    test_cases = []
    for shape_sweep in shape_sweeps:
        token_counts = _as_int_list(shape_sweep.get("token_counts"), field_name=f"{op_name}.token_counts")
        head_counts = _as_int_list(shape_sweep.get("head_counts"), field_name=f"{op_name}.head_counts")
        dtypes = shape_sweep.get("dtypes")
        if not isinstance(dtypes, list):
            raise TypeError(f"{op_name}.dtypes must be a list")
        num_warmups = int(shape_sweep.get("num_warmups", 2))
        num_runs = int(shape_sweep.get("num_runs", 10))

        for num_tokens, num_heads, dtype in itertools.product(token_counts, head_counts, dtypes):
            test_cases.append(
                MLABMMCommonTestCase(
                    num_tokens=num_tokens,
                    num_heads=num_heads,
                    dtype=str(dtype),
                    num_warmups=num_warmups,
                    num_runs=num_runs,
                )
            )
    return test_cases


# =============================================================================
# Mamba2 SSM Test Cases
# =============================================================================


@dataclasses.dataclass
class Mamba2CommonTestCase:
    """Test case configuration for Mamba2 SSM benchmarking."""

    phase: str  # "context" or "generation"
    d_model: int  # hidden_size
    d_state: int  # SSM state dimension
    d_conv: int  # Conv1d kernel size
    nheads: int  # Number of Mamba heads
    head_dim: int  # Dimension per head
    n_groups: int  # Number of groups for B, C matrices
    chunk_size: int  # Chunk size for SSM scan
    num_tokens_list: Optional[list[int]]  # For context phase (continuous batching)
    batch_size_list: Optional[list[int]]  # For generation phase, or context static batching
    seq_len_list: Optional[list[int]]  # For context phase with static batching
    model_name: str


def get_common_mamba2_test_cases() -> list[Mamba2CommonTestCase]:
    """
    Generate common test cases for Mamba2 SSM benchmarking.

    Includes configurations for:
    - Nemotron-H 3-30B (primary target)
    - Other potential Mamba2-based models

    Returns:
        List of Mamba2CommonTestCase configurations
    """
    test_cases: list[Mamba2CommonTestCase] = []
    mamba2_sweep = _required_base_common_case_values("mamba2")
    context_seq_lens = _as_int_list(
        mamba2_sweep.get("context_sequence_lengths"),
        field_name="mamba2.context_sequence_lengths",
    )
    context_batch_sizes = _as_int_list(
        mamba2_sweep.get("context_batch_sizes"),
        field_name="mamba2.context_batch_sizes",
    )
    generation_batch_sizes = _as_int_list(
        mamba2_sweep.get("generation_batch_sizes"),
        field_name="mamba2.generation_batch_sizes",
    )

    raw_default_model_cases = mamba2_sweep.get("default_model_cases", [])
    if not isinstance(raw_default_model_cases, list):
        raise TypeError("common_case_values.mamba2.default_model_cases must be a list")
    default_model_cases = []
    for index, raw_case in enumerate(raw_default_model_cases):
        default_model_cases.extend(
            _expand_model_case_entry(
                raw_case,
                field_name=f"common_case_values.mamba2.default_model_cases[{index}]",
            )
        )
    model_path = _get_model_path_filter()
    if model_path:
        default_model_cases = [case for case in default_model_cases if _model_case_matches_path(case, model_path)]

    model_config_list = [*default_model_cases, *_model_case_values("mamba2")]

    for model_config in model_config_list:
        d_model = int(model_config["d_model"])
        d_state = int(model_config["d_state"])
        d_conv = int(model_config["d_conv"])
        nheads = int(model_config["nheads"])
        head_dim = int(model_config["head_dim"])
        n_groups = int(model_config["n_groups"])
        chunk_size = int(model_config["chunk_size"])
        model_name = str(model_config["model_path"])

        # Context (prefill) test case
        test_cases.append(
            Mamba2CommonTestCase(
                phase="context",
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                nheads=nheads,
                head_dim=head_dim,
                n_groups=n_groups,
                chunk_size=chunk_size,
                num_tokens_list=None,  # Not used for static batching
                batch_size_list=context_batch_sizes,
                seq_len_list=context_seq_lens,
                model_name=model_name,
            )
        )

        # Generation (decode) test case
        test_cases.append(
            Mamba2CommonTestCase(
                phase="generation",
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                nheads=nheads,
                head_dim=head_dim,
                n_groups=n_groups,
                chunk_size=chunk_size,
                num_tokens_list=None,
                batch_size_list=generation_batch_sizes,
                seq_len_list=None,  # Not used for generation
                model_name=model_name,
            )
        )

    return test_cases


# =============================================================================
# GDN (Gated DeltaNet) Test Cases  — Qwen3.5 linear_attention layers
# =============================================================================


@dataclasses.dataclass
class GdnCommonTestCase:
    """Test case configuration for GDN (Gated DeltaNet) kernel benchmarking."""

    phase: str  # "context" or "generation"
    d_model: int  # hidden_size
    d_conv: int  # Conv1d kernel size
    num_k_heads: int  # Number of GDN key heads
    head_k_dim: int  # Key head dimension
    num_v_heads: int  # Number of GDN value heads
    head_v_dim: int  # Value head dimension
    batch_size_list: Optional[list[int]]
    seq_len_list: Optional[list[int]]  # For context phase; None for generation
    model_name: str


# =============================================================================
# MHC (DeepSeek-V4 Hash-Compressed attention) Test Cases
# =============================================================================


@dataclasses.dataclass
class MhcCommonTestCase:
    """Test case configuration for DeepSeek-V4 mHC pre/post kernel benchmarking."""

    phase: str  # "pre" or "post"
    hidden_size: int
    hc_mult: int
    num_tokens_list: list[int]
    model_name: str


def get_common_mhc_test_cases() -> list[MhcCommonTestCase]:
    """Generate common test cases for mHC (pre/post) kernel benchmarking."""
    mhc_sweep = _required_base_common_case_values("mhc")
    num_tokens_list = _as_int_list(mhc_sweep.get("token_counts"), field_name="mhc.token_counts")

    model_config_list = _model_case_values("mhc")

    test_cases: list[MhcCommonTestCase] = []
    for model_config in model_config_list:
        hidden_size = int(model_config["hidden_size"])
        hc_mult = int(model_config["hc_mult"])
        model_name = str(model_config["model_path"])
        for phase in ("pre", "post"):
            test_cases.append(
                MhcCommonTestCase(
                    phase=phase,
                    hidden_size=hidden_size,
                    hc_mult=hc_mult,
                    num_tokens_list=num_tokens_list,
                    model_name=model_name,
                )
            )
    return test_cases


def get_common_gdn_test_cases() -> list[GdnCommonTestCase]:
    """
    Generate common test cases for GDN (Gated DeltaNet) kernel benchmarking.

    Covers all 8 unique dimension sets across the full Qwen3.5 collection
    for both context (prefill) and generation (decode) phases.
    """
    test_cases: list[GdnCommonTestCase] = []
    gdn_sweep = _required_base_common_case_values("gdn")
    context_seq_lens = _as_int_list(
        gdn_sweep.get("context_sequence_lengths"),
        field_name="gdn.context_sequence_lengths",
    )
    context_batch_sizes = _as_int_list(
        gdn_sweep.get("context_batch_sizes"),
        field_name="gdn.context_batch_sizes",
    )
    generation_batch_sizes = _as_int_list(
        gdn_sweep.get("generation_batch_sizes"),
        field_name="gdn.generation_batch_sizes",
    )

    model_config_list = _model_case_values("gdn")

    for model_config in model_config_list:
        d_model = int(model_config["d_model"])
        d_conv = int(model_config["d_conv"])
        num_k_heads = int(model_config["num_k_heads"])
        head_k_dim = int(model_config["head_k_dim"])
        num_v_heads = int(model_config["num_v_heads"])
        head_v_dim = int(model_config["head_v_dim"])
        model_name = str(model_config["model_path"])

        # Context (prefill) test case
        test_cases.append(
            GdnCommonTestCase(
                phase="context",
                d_model=d_model,
                d_conv=d_conv,
                num_k_heads=num_k_heads,
                head_k_dim=head_k_dim,
                num_v_heads=num_v_heads,
                head_v_dim=head_v_dim,
                batch_size_list=context_batch_sizes,
                seq_len_list=context_seq_lens,
                model_name=model_name,
            )
        )

        # Generation (decode) test case
        test_cases.append(
            GdnCommonTestCase(
                phase="generation",
                d_model=d_model,
                d_conv=d_conv,
                num_k_heads=num_k_heads,
                head_k_dim=head_k_dim,
                num_v_heads=num_v_heads,
                head_v_dim=head_v_dim,
                batch_size_list=generation_batch_sizes,
                seq_len_list=None,
                model_name=model_name,
            )
        )

    return test_cases


# ═══════════════════════════════════════════════════════════════════════
# DeepSeek-V4 attention test cases
# ═══════════════════════════════════════════════════════════════════════
# Used by ``collector.sglang.collect_dsv4_attn`` (full-module bench)
# and ``collector.sglang.deepseekv4_sparse_modules`` (sparse kernel bench).
# Both backends re-export the relevant ``get_*`` functions so collect.py
# can resolve them via getattr on each per-backend module.


def _dedupe_strs(values: list[object]) -> list[str]:
    deduped = []
    seen = set()
    for value in values:
        if value is None:
            continue
        value = str(value)
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _raw_model_case_config(op_name: str) -> tuple[dict, dict] | None:
    for data in _load_model_cases_data():
        raw = (data.get("model_case_values") or {}).get(op_name)
        if raw is None:
            continue
        if isinstance(raw, list):
            if not raw:
                continue
            if len(raw) != 1:
                raise TypeError(f"model_case_values.{op_name} must be a mapping or a single-item list")
            raw = raw[0]
        if not isinstance(raw, dict):
            raise TypeError(f"model_case_values.{op_name} must be a mapping")
        return data, dict(raw)
    return None


def _dsv4_config() -> dict:
    raw = _raw_model_case_config("dsv4") or _raw_model_case_config("dsv4_flash")
    if raw is None:
        raise RuntimeError("model_case_values.dsv4 is missing from model case YAML")
    model_data, config = raw

    default_model_paths = config.get("default_model_paths")
    if default_model_paths is None:
        default_model_paths = [config.get("model_path") or model_data.get("model_path")]
    default_model_paths = _dedupe_strs(default_model_paths)

    supported_model_paths = config.get("supported_model_paths")
    if supported_model_paths is None:
        supported_model_paths = [
            *default_model_paths,
            model_data.get("model_path"),
            *(model_data.get("model_paths") or []),
            config.get("model_path"),
        ]
    supported_model_paths = _dedupe_strs(supported_model_paths)
    if not default_model_paths:
        default_model_paths = supported_model_paths
    if not default_model_paths:
        raise RuntimeError("model_case_values.dsv4 needs at least one default model path")
    if len(default_model_paths) != 1:
        raise ValueError(
            "DeepSeek-V4 module keys cannot distinguish models; "
            "dsv4.default_model_paths must contain one canonical path"
        )

    config["default_model_paths"] = default_model_paths
    config["supported_model_paths"] = _dedupe_strs([*supported_model_paths, *default_model_paths])
    return config


def _dsv4_attention_kinds() -> tuple[str, ...]:
    return tuple(str(kind) for kind in _DSV4_CONFIG.get("attention_kinds", ["csa", "hca"]))


_DSV4_CONFIG = _dsv4_config()
_DSV4_DEFAULT_MODELS = tuple(_as_str_list(_DSV4_CONFIG["default_model_paths"], field_name="dsv4.default_model_paths"))
_DSV4_SUPPORTED_MODELS = tuple(
    _as_str_list(_DSV4_CONFIG["supported_model_paths"], field_name="dsv4.supported_model_paths")
)
DSV4_ATTN_KINDS = _dsv4_attention_kinds()
_DSV4_MODULE_BATCH_SIZES = _as_int_list(_DSV4_CONFIG["module_batch_sizes"], field_name="dsv4.module_batch_sizes")
_DSV4_MODULE_SEQ_LENGTHS = _as_int_list(
    _DSV4_CONFIG["module_sequence_lengths"],
    field_name="dsv4.module_sequence_lengths",
)
_DSV4_MODULE_PAST_KV_LIST = _as_int_list(
    _DSV4_CONFIG["module_past_kv_lengths"],
    field_name="dsv4.module_past_kv_lengths",
)
_DSV4_MODULE_TP_SIZES = _as_int_list(_DSV4_CONFIG["module_tp_sizes"], field_name="dsv4.module_tp_sizes")
_DSV4_SPARSE_BS_LIST = _as_int_list(_DSV4_CONFIG["sparse_batch_sizes"], field_name="dsv4.sparse_batch_sizes")
_DSV4_SPARSE_ISL_LIST = _as_int_list(
    _DSV4_CONFIG["sparse_input_lengths"],
    field_name="dsv4.sparse_input_lengths",
)
_DSV4_SPARSE_PAST_KV_LIST = _as_int_list(
    _DSV4_CONFIG["sparse_past_kv_lengths"],
    field_name="dsv4.sparse_past_kv_lengths",
)
_DSV4_SPARSE_CHUNK_PREFILL_SIZE = int(_DSV4_CONFIG["sparse_chunk_prefill_size"])
_DSV4_SPARSE_MAX_FULL_S = int(_DSV4_CONFIG["sparse_max_full_sequence_length"])
_DSV4_SPARSE_TP_SIZES = _required_mapping(_DSV4_CONFIG["sparse_tp_sizes"], field_name="dsv4.sparse_tp_sizes")
_DSV4_SPARSE_TP_LIST_ATTN = _as_int_list(
    _DSV4_SPARSE_TP_SIZES["hca_attn"],
    field_name="dsv4.sparse_tp_sizes.hca_attn",
)
_DSV4_SPARSE_TP_LIST_INDEXER = _as_int_list(
    _DSV4_SPARSE_TP_SIZES["paged_mqa_logits"],
    field_name="dsv4.sparse_tp_sizes.paged_mqa_logits",
)


def _selected_dsv4_models() -> tuple[str, ...]:
    """Apply collect.py's model-path filter to DSV4 case generation."""
    filt = _get_model_path_filter()
    if filt is None:
        return _DSV4_DEFAULT_MODELS
    if filt in _DSV4_SUPPORTED_MODELS or os.path.isdir(filt):
        return (filt,)
    return ()


def _has_native_fp4_experts() -> bool:
    """True when the device has native FP4 tensor cores (Blackwell SM100+)."""
    try:
        import torch as _t

        if not _t.cuda.is_available():
            return False
        return _t.cuda.get_device_capability(0)[0] >= 10
    except Exception:
        return False


def _dsv4_module_needs_native_fp4(model_path: str) -> bool:
    """Return True for native DeepSeek-V4 checkpoints with FP4 routed experts."""
    return model_path.startswith("deepseek-ai/")


def _dsv4_module_precision_combos(phase: str, model_path: str):
    """``(compute_dtype, kv_cache_dtype, gemm_type)`` triples.

    DeepseekV4ForCausalLM rejects bfloat16 KV cache (asserts at load time),
    so we only emit fp8 KV.  ``gemm_type`` switches projection dispatch:
      * ``bfloat16``  — projections through cuBLASLt nvjet kernels
      * ``fp8_block`` — fp8 block-quantised weights → DeepGEMM
                        ``sm90_fp8_gemm_1d2d_impl`` (matches production)

    ``fp8_block`` is omitted on pre-Blackwell parts only for native
    ``deepseek-ai/*`` checkpoints; ``sgl-project/*-FP8`` checkpoints are
    already converted for Hopper-side FP8 collection.
    """
    del phase
    combos = [("bfloat16", "fp8", "bfloat16")]
    if not _dsv4_module_needs_native_fp4(model_path) or _has_native_fp4_experts():
        combos.append(("bfloat16", "fp8", "fp8_block"))
    else:
        print(
            "[dsv4-test-cases] device lacks native FP4 experts (pre-Blackwell); omitting fp8_block from gemm_type sweep"
        )
    return combos


def _dsv4_module_is_valid_shape(mode: str, bs: int, sl: int, past_kv: int = 0) -> bool:
    """Return whether a DeepSeek-V4 full-module sample fits scheduler/model limits."""
    if bs <= 0 or sl <= 0 or past_kv < 0:
        return False
    if mode == "context":
        return True
    if mode == "generation":
        if bs * sl > 1024 * 1024:
            return False
        if sl >= 524288 and bs > 1:
            return False
        if sl >= 262144 and bs > 2:
            return False
        if sl >= 131072 and bs > 4:
            return False
        if sl >= 65536 and bs > 8:
            return False
        if sl >= 32768 and bs > 16:
            return False
        return not (sl >= 8192 and bs > 64)
    raise ValueError(f"unsupported DeepSeek-V4 mode: {mode}")


def _dsv4_module_filter_pairs(mode: str, batch_sizes, seq_lens):
    """Drop ``(bs, sl)`` pairs that exceed KV pool / kernel limits.

    Context (b * s):
        ≤ 8192 — matches sglang's default ``chunked_prefill_size``.
    Generation (b * s):
        ≤ 1M overall, with per-sl batch caps for long contexts (sl≥8192→bs≤64,
        sl≥32768→bs≤16, sl≥65536→bs≤8, sl≥131072→bs≤4, sl≥262144→bs≤2,
        sl≥524288→bs==1).  Ensures bs=1 is always allowed at every sl.
    """
    pairs = []
    for bs in batch_sizes:
        for sl in seq_lens:
            if not _dsv4_module_is_valid_shape(mode, bs, sl):
                continue
            pairs.append((bs, sl))
    return pairs


def _build_dsv4_module_test_cases(mode: str, attn_kinds=DSV4_ATTN_KINDS):
    """One case per ``(attn_kind, tp_size, gemm_type, batch_size)``.

    Test case shape (9 elements; ``perf_filename`` is bound by collect.py
    via OpEntry, NOT in the tuple)::

        [0, batch_size, tp_size, kv_cache_dtype, compute_dtype, gemm_type,
         model_path, attn_kind, attention_backend]

    Each spawned subprocess builds ONE ``ModelRunner`` for ``(bs, max_sl)``
    and sweeps every valid sl for that bs internally.
    """
    pairs = _dsv4_module_filter_pairs(
        mode,
        _DSV4_MODULE_BATCH_SIZES,
        _DSV4_MODULE_SEQ_LENGTHS,
    )
    bs_set = sorted({bs for bs, _ in pairs})

    cases: list[list] = []
    for model_path in _selected_dsv4_models():
        for attn_kind in attn_kinds:
            for compute_dtype, kv_dtype, gemm_type in _dsv4_module_precision_combos(mode, model_path):
                for tp_size in _DSV4_MODULE_TP_SIZES:
                    for bs in bs_set:
                        cases.append(
                            [
                                0,
                                bs,
                                tp_size,
                                kv_dtype,
                                compute_dtype,
                                gemm_type,
                                model_path,
                                attn_kind,
                                None,
                            ]
                        )
    return cases


def get_dsv4_csa_context_test_cases():
    return _build_dsv4_module_test_cases("context", ("csa",))


def get_dsv4_hca_context_test_cases():
    return _build_dsv4_module_test_cases("context", ("hca",))


def get_dsv4_csa_generation_test_cases():
    return _build_dsv4_module_test_cases("generation", ("csa",))


def get_dsv4_hca_generation_test_cases():
    return _build_dsv4_module_test_cases("generation", ("hca",))


def get_dsv4_topk_calib_test_cases():
    """One case per model (the topk DELTA calibration is just another member of
    the sparse-op family — same ``[model_path, kernel]`` shape as paged_mqa /
    csa_attn / hca_attn, dispatched through ``run_dsv4_sparse_kernel_worker``).

    The grid matches the CSA module data 1:1: the worker reads the
    already-collected ``dsv4_csa_*_module_perf.txt`` and benches exactly those
    ``(prefix, isl, batch_size)`` shapes — no separate grid is generated here.
    """
    return [[model_path, "topk"] for model_path in _selected_dsv4_models()]


DSV4_SPARSE_KERNELS = ("paged_mqa_logits", "hca_attn")


def _build_dsv4_sparse_test_cases(
    kernels=DSV4_SPARSE_KERNELS,
    bs_list=None,
    isl_list=None,
    past_kv_list=None,
    tp_list_attn=None,
    tp_list_indexer=None,
):
    """Generate ``(bs, isl, past_kv, tp_size, kernel, model)`` tuples.

    Filters mirror sglang prefill scheduler:
      * bs x isl ≤ chunked_prefill_size = 8192   — new-token budget per chunk
      * bs x (isl + past_kv) ≤ 1M                — model context cap
    """
    bs_list = list(bs_list) if bs_list is not None else list(_DSV4_SPARSE_BS_LIST)
    isl_list = list(isl_list) if isl_list is not None else list(_DSV4_SPARSE_ISL_LIST)
    past_kv_list = list(past_kv_list) if past_kv_list is not None else list(_DSV4_SPARSE_PAST_KV_LIST)
    tp_list_attn = list(tp_list_attn) if tp_list_attn is not None else list(_DSV4_SPARSE_TP_LIST_ATTN)
    tp_list_indexer = list(tp_list_indexer) if tp_list_indexer is not None else list(_DSV4_SPARSE_TP_LIST_INDEXER)

    cases = []
    for model_path in _selected_dsv4_models():
        for kernel in kernels:
            tp_list = tp_list_attn if kernel == "hca_attn" else tp_list_indexer
            for tp_size in tp_list:
                for bs in bs_list:
                    for isl in isl_list:
                        if bs * isl > _DSV4_SPARSE_CHUNK_PREFILL_SIZE:
                            continue
                        for past_kv in past_kv_list:
                            if bs * (isl + past_kv) > _DSV4_SPARSE_MAX_FULL_S:
                                continue
                            full_s = isl + past_kv
                            if kernel == "paged_mqa_logits" and full_s < 4:
                                continue
                            if kernel == "hca_attn" and full_s < 64:
                                continue
                            cases.append(
                                [
                                    bs,
                                    isl,
                                    past_kv,
                                    tp_size,
                                    kernel,
                                    model_path,
                                ]
                            )
    return cases


def get_dsv4_paged_mqa_logits_test_cases():
    """One case per model. The worker derives shapes from the CSA module CSV
    (paged_mqa_logits is the CSA indexer scoring sub-kernel), 1:1 with the
    csa-module rows — no separate sparse sweep grid."""
    return [[model_path, "paged_mqa_logits"] for model_path in _selected_dsv4_models()]


def get_dsv4_hca_attn_test_cases():
    """One case per model. The worker derives shapes from the HCA module CSV
    (hca_attn is the HCA c128 FMLA sub-kernel), 1:1 with the hca-module rows."""
    return [[model_path, "hca_attn"] for model_path in _selected_dsv4_models()]


def get_dsv4_csa_attn_test_cases():
    """One case per model. The worker derives shapes from the CSA module CSV
    (csa_attn is the CSA sparse FMLA over the topk-selected c4 positions),
    1:1 with the csa-module rows — same source as paged_mqa_logits/topk."""
    return [[model_path, "csa_attn"] for model_path in _selected_dsv4_models()]


# Backward-compatible names for older PR comments/tests while the registry and
# persisted perf files use the upstream DeepSeek-V4 op names.
_DSV4_FLASH_CONFIG = _DSV4_CONFIG
_DSV4_FLASH_MODEL_PATH = _DSV4_DEFAULT_MODELS[0]
DSV4_FLASH_ATTN_KINDS = DSV4_ATTN_KINDS
_DSV4_FLASH_MODULE_BATCH_SIZES = _DSV4_MODULE_BATCH_SIZES
_DSV4_FLASH_MODULE_SEQ_LENGTHS = _DSV4_MODULE_SEQ_LENGTHS
_DSV4_FLASH_MODULE_PAST_KV_LIST = _DSV4_MODULE_PAST_KV_LIST
_DSV4_FLASH_MODULE_TP_SIZES = _DSV4_MODULE_TP_SIZES
_DSV4_FLASH_SPARSE_BS_LIST = _DSV4_SPARSE_BS_LIST
_DSV4_FLASH_SPARSE_ISL_LIST = _DSV4_SPARSE_ISL_LIST
_DSV4_FLASH_SPARSE_PAST_KV_LIST = _DSV4_SPARSE_PAST_KV_LIST
_DSV4_FLASH_SPARSE_CHUNK_PREFILL_SIZE = _DSV4_SPARSE_CHUNK_PREFILL_SIZE
_DSV4_FLASH_SPARSE_MAX_FULL_S = _DSV4_SPARSE_MAX_FULL_S
_DSV4_FLASH_SPARSE_TP_LIST_ATTN = _DSV4_SPARSE_TP_LIST_ATTN
_DSV4_FLASH_SPARSE_TP_LIST_INDEXER = _DSV4_SPARSE_TP_LIST_INDEXER
DSV4_FLASH_SPARSE_KERNELS = DSV4_SPARSE_KERNELS
_dsv4_flash_config = _dsv4_config


def _dsv4_flash_active() -> bool:
    return bool(_selected_dsv4_models())


def _dsv4_flash_model_path() -> str:
    return (_selected_dsv4_models() or _DSV4_DEFAULT_MODELS)[0]


_dsv4_flash_module_precision_combos = _dsv4_module_precision_combos
_dsv4_flash_module_filter_pairs = _dsv4_module_filter_pairs
_build_dsv4_flash_module_test_cases = _build_dsv4_module_test_cases
_build_dsv4_flash_sparse_test_cases = _build_dsv4_sparse_test_cases
get_dsv4_flash_csa_context_test_cases = get_dsv4_csa_context_test_cases
get_dsv4_flash_hca_context_test_cases = get_dsv4_hca_context_test_cases
get_dsv4_flash_csa_generation_test_cases = get_dsv4_csa_generation_test_cases
get_dsv4_flash_hca_generation_test_cases = get_dsv4_hca_generation_test_cases
get_dsv4_flash_paged_mqa_logits_test_cases = get_dsv4_paged_mqa_logits_test_cases
get_dsv4_flash_hca_attn_test_cases = get_dsv4_hca_attn_test_cases
