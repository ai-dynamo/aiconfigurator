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
import itertools
import os
import sys
from pathlib import Path
from typing import Optional

import yaml

COLLECTOR_ROOT = Path(__file__).resolve().parent
BASE_OP_CASES_PATH = COLLECTOR_ROOT / "cases" / "base_op_cases.yaml"
BASE_OP_CASES_DIR = COLLECTOR_ROOT / "cases" / "base_ops"
MODEL_CASES_DIR = COLLECTOR_ROOT / "cases" / "models"


def _load_yaml_mapping(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise TypeError(f"{path}: top-level YAML value must be a mapping")
    return data


def _base_ops_dir(base_data: dict) -> Path:
    raw_dir = base_data.get("base_ops_dir", BASE_OP_CASES_DIR.name)
    path = Path(str(raw_dir))
    if path.is_absolute():
        return path
    return BASE_OP_CASES_PATH.parent / path


def _merge_base_case_data(target: dict, source: dict) -> None:
    target.setdefault("model_ops", [])
    target["model_ops"].extend(source.get("model_ops") or [])
    target.setdefault("common_case_values", {}).update(source.get("common_case_values") or {})
    target.setdefault("all_frameworks_op_cases", {}).update(source.get("all_frameworks_op_cases") or {})

    target_framework_cases = target.setdefault("framework_specific_op_cases", {})
    for backend, backend_cases in (source.get("framework_specific_op_cases") or {}).items():
        target_framework_cases.setdefault(backend, {}).update(backend_cases or {})


def _load_base_cases_data() -> dict:
    base_data = _load_yaml_mapping(BASE_OP_CASES_PATH)
    merged: dict = {}
    _merge_base_case_data(merged, base_data)

    base_ops_dir = _base_ops_dir(base_data)
    if not base_ops_dir.exists():
        return merged

    configured_files = base_data.get("base_ops")
    if configured_files is None:
        paths = sorted(base_ops_dir.glob("*.yaml"))
    elif isinstance(configured_files, list):
        paths = [base_ops_dir / str(filename) for filename in configured_files]
    else:
        raise TypeError("base_ops must be a list")

    for path in paths:
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
        raise RuntimeError(f"{BASE_OP_CASES_PATH} is missing common_case_values.{name}")
    return values


def _get_model_path_filter() -> str | None:
    """Return the model-path filter from the environment, or None for 'all'."""
    val = os.environ.get("COLLECTOR_MODEL_PATH", "").strip()
    return val if val else None


def _load_model_cases_data() -> list[dict]:
    data = []
    for path in sorted(MODEL_CASES_DIR.glob("*_cases.yaml")):
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        if not isinstance(raw, dict):
            raise TypeError(f"{path}: top-level YAML value must be a mapping")
        data.append(raw)
    return data


def _model_case_values(op_name: str, *, apply_model_filter: bool = True) -> list[dict]:
    values = []
    for data in _load_model_cases_data():
        op_values = data.get("model_case_values", {}).get(op_name, [])
        if op_values is None:
            continue
        if isinstance(op_values, dict):
            values.append(dict(op_values))
            continue
        if not isinstance(op_values, list):
            raise TypeError(f"model_case_values.{op_name} must be a list or mapping")
        values.extend(dict(item) for item in op_values)

    model_path = _get_model_path_filter() if apply_model_filter else None
    if model_path:
        values = [value for value in values if value.get("model_path") == model_path]
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
            values.append(dict(op_values))
            continue
        if not isinstance(op_values, list):
            raise TypeError(f"framework_specific_model_case_values.{backend}.{op_name} must be a list or mapping")
        values.extend(dict(item) for item in op_values)

    model_path = _get_model_path_filter() if apply_model_filter else None
    if model_path:
        values = [value for value in values if value.get("model_path") == model_path]
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
    module_precision_combos: list[tuple[str, str, str]]
    context_max_tokens: int
    context_large_sequence_min: int
    context_large_sequence_max_batch_size: int
    generation_max_tokens: int
    generation_large_sequence_min: int
    generation_large_sequence_max_batch_size: int
    generation_large_cache_tokens: int


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
    wideep_mla: bool | None = None,
    apply_model_filter: bool = True,
) -> list[MLAModuleModelSpec]:
    """Return YAML-backed model metadata for full MLA/DSA module collectors."""

    values = []
    model_path_filter = _get_model_path_filter() if apply_model_filter else None
    for data in _load_model_cases_data():
        raw_values = (data.get("model_case_values") or {}).get("mla_module", [])
        if raw_values is None:
            continue
        if isinstance(raw_values, dict):
            raw_values = [raw_values]
        if not isinstance(raw_values, list):
            raise TypeError("model_case_values.mla_module must be a list or mapping")

        architecture = data.get("architecture")
        for raw_value in raw_values:
            if not isinstance(raw_value, dict):
                raise TypeError("model_case_values.mla_module entries must be mappings")
            value = dict(raw_value)
            value.setdefault("architecture", architecture)
            if model_path_filter and value.get("model_path") != model_path_filter:
                continue
            if attention_type is not None and value.get("attention_type") != attention_type:
                continue
            if wideep_mla is not None and bool(value.get("wideep_mla", False)) != wideep_mla:
                continue
            values.append(
                MLAModuleModelSpec(
                    model_path=str(value["model_path"]),
                    attention_type=str(value["attention_type"]),
                    architecture=str(value["architecture"]),
                    native_num_heads=int(value["native_num_heads"]),
                    wideep_mla=bool(value.get("wideep_mla", False)),
                )
            )
    return values


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

    return MLAModuleSweepSpec(
        batch_sizes=batch_sizes or context_batch_sizes,
        sequence_lengths=sequence_lengths or context_sequence_lengths,
        context_batch_sizes=context_batch_sizes,
        context_sequence_lengths=context_sequence_lengths,
        generation_batch_sizes=generation_batch_sizes,
        generation_sequence_lengths=generation_sequence_lengths,
        inner_sweep_head_counts=inner_sweep_head_counts,
        top_level_head_counts=top_level_head_counts,
        module_precision_combos=precision_combos,
        context_max_tokens=int(context["max_tokens"]),
        context_large_sequence_min=_optional_int(context.get("large_sequence_min")),
        context_large_sequence_max_batch_size=_optional_int(context.get("large_sequence_max_batch_size")),
        generation_max_tokens=int(generation["max_tokens"]),
        generation_large_sequence_min=_optional_int(generation.get("large_sequence_min")),
        generation_large_sequence_max_batch_size=_optional_int(generation.get("large_sequence_max_batch_size")),
        generation_large_cache_tokens=_optional_int(generation.get("large_cache_tokens")),
    )


def is_wideep_moe_model(model_name: str) -> bool:
    """Return True if *model_name* needs WideEP MoE collection."""
    return any(
        value.get("model_path") == model_name and value.get("wideep")
        for value in _model_case_values("moe", apply_model_filter=False)
    )


def get_all_model_names() -> list[str]:
    """Return all known model names across all op types.

    Reads directly from model case YAML — does not instantiate test
    case objects or call generator functions, so pruning logic in the generators
    cannot accidentally exclude models from the allowlist.
    """
    model_names = []
    for data in _load_model_cases_data():
        primary = data.get("model_path")
        if primary:
            model_names.append(str(primary))
        model_names.extend(str(path) for path in data.get("model_paths", []) or [])
        for values in (data.get("model_case_values") or {}).values():
            if isinstance(values, dict):
                value_model_path = values.get("model_path")
                if value_model_path:
                    model_names.append(str(value_model_path))
                continue
            if isinstance(values, list):
                model_names.extend(
                    str(value["model_path"]) for value in values if isinstance(value, dict) and value.get("model_path")
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
        if model_case.get("model_path") != model_name:
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


def moe_shape_satisfies_constraints(
    backend: str,
    moe_type: str,
    *,
    hidden_size: int,
    inter_size: int,
    tensor_parallel_size: int,
    topk: int,
) -> bool:
    """Return whether a MoE shape satisfies backend YAML quantization limits."""

    values = _moe_backend_values(backend)
    raw_constraints = values.get("shape_constraints", [])
    if not isinstance(raw_constraints, list):
        raise TypeError(f"common_case_values.moe_{backend}.shape_constraints must be a list")

    local_inter_size = inter_size // tensor_parallel_size
    fields = {
        "hidden_size": hidden_size,
        "inter_size": inter_size,
        "local_inter_size": local_inter_size,
        "topk": topk,
    }
    for raw_constraint in raw_constraints:
        if not isinstance(raw_constraint, dict):
            raise TypeError(f"common_case_values.moe_{backend}.shape_constraints entries must be mappings")
        if str(raw_constraint.get("mode")) != moe_type:
            continue

        divisible_by = raw_constraint.get("divisible_by", {})
        if not isinstance(divisible_by, dict):
            raise TypeError(f"common_case_values.moe_{backend}.shape_constraints.divisible_by must be a mapping")
        for field_name, divisor in divisible_by.items():
            if field_name not in fields:
                raise ValueError(f"Unknown MoE shape constraint field: {field_name}")
            if fields[field_name] % int(divisor) != 0:
                return False

        max_topk = raw_constraint.get("max_topk")
        if max_topk is not None and topk > int(max_topk):
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
        if model_path_filter and case.get("model_path") != model_path_filter:
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
        if model_case.get("model_path") == model_name:
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
                )
            )
    return test_cases


def get_common_moe_test_cases():
    moe_sweep = _required_base_common_case_values("moe")
    num_tokens = _as_int_list(moe_sweep.get("token_counts"), field_name="moe.token_counts")
    tp_list = _as_int_list(moe_sweep.get("tensor_parallel_sizes"), field_name="moe.tensor_parallel_sizes")
    ep_list = _as_int_list(moe_sweep.get("expert_parallel_sizes"), field_name="moe.expert_parallel_sizes")
    num_gpu_list = _as_int_list(moe_sweep.get("gpu_counts"), field_name="moe.gpu_counts")
    token_distributions = _moe_token_expert_distributions(moe_sweep)

    model_config_list = _model_case_values("moe")

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
        raise RuntimeError(f"{BASE_OP_CASES_PATH} is missing all_frameworks_op_cases.gemm.cases")
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

        test_cases.append(
            MLACommonTestCase(
                num_heads=int(model_config["num_heads"]),
                input_len=s if is_context else s - 1,
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
        raise RuntimeError(f"{BASE_OP_CASES_PATH} is missing all_frameworks_op_cases.{op_name}.cases")

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

    model_config_list = _model_case_values("mamba2")

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
# DeepSeek-V4-Flash test cases
# ═══════════════════════════════════════════════════════════════════════
# Used by ``collector.sglang.collect_dsv4_flash_attn`` (full-module bench)
# and ``collector.sglang.deepseekv4_sparse_modules`` (sparse kernel bench).
# Both backends re-export the relevant ``get_*`` functions so collect.py
# can resolve them via getattr on each per-backend module.


def _dsv4_flash_config() -> dict:
    configs = _model_case_values("dsv4_flash", apply_model_filter=False)
    if not configs:
        raise RuntimeError("model_case_values.dsv4_flash is missing from model case YAML")
    return configs[0]


def _dsv4_flash_attention_kinds() -> tuple[str, ...]:
    return tuple(str(kind) for kind in _dsv4_flash_config().get("attention_kinds", ["csa", "hca"]))


_DSV4_FLASH_CONFIG = _dsv4_flash_config()
_DSV4_FLASH_MODEL_PATH = str(_DSV4_FLASH_CONFIG["model_path"])
DSV4_FLASH_ATTN_KINDS = _dsv4_flash_attention_kinds()
_DSV4_FLASH_MODULE_BATCH_SIZES = list(_DSV4_FLASH_CONFIG["module_batch_sizes"])
_DSV4_FLASH_MODULE_SEQ_LENGTHS = list(_DSV4_FLASH_CONFIG["module_sequence_lengths"])
_DSV4_FLASH_MODULE_TP_SIZES = list(_DSV4_FLASH_CONFIG["module_tp_sizes"])
_DSV4_FLASH_SPARSE_BS_LIST = list(_DSV4_FLASH_CONFIG["sparse_batch_sizes"])
_DSV4_FLASH_SPARSE_ISL_LIST = list(_DSV4_FLASH_CONFIG["sparse_input_lengths"])
_DSV4_FLASH_SPARSE_PAST_KV_LIST = list(_DSV4_FLASH_CONFIG["sparse_past_kv_lengths"])
_DSV4_FLASH_SPARSE_CHUNK_PREFILL_SIZE = int(_DSV4_FLASH_CONFIG["sparse_chunk_prefill_size"])
_DSV4_FLASH_SPARSE_MAX_FULL_S = int(_DSV4_FLASH_CONFIG["sparse_max_full_sequence_length"])
_DSV4_FLASH_SPARSE_TP_LIST_ATTN = list(_DSV4_FLASH_CONFIG["sparse_tp_sizes"]["hca_attn"])
_DSV4_FLASH_SPARSE_TP_LIST_INDEXER = list(_DSV4_FLASH_CONFIG["sparse_tp_sizes"]["paged_mqa_logits"])


def _dsv4_flash_active() -> bool:
    """Honour the ``--model-path`` (``COLLECTOR_MODEL_PATH``) filter.

    All V4-Flash test cases are V4-Flash-only by construction.  When the
    user filters to a different model, every V4-Flash op must emit zero
    cases so the collector skips it.
    """
    filt = _get_model_path_filter()
    return filt is None or filt == _dsv4_flash_config()["model_path"]


def _dsv4_flash_model_path() -> str:
    filt = _get_model_path_filter()
    return filt if filt is not None else str(_dsv4_flash_config()["model_path"])


def _has_native_fp4_experts() -> bool:
    """True when the device has native FP4 tensor cores (Blackwell sm_100+).

    DSv4-Flash's checkpoint ships routed experts in FP4. When the collector
    sets ``server_args.quantization="fp8"`` (our ``gemm_type="fp8_block"``
    sweep) sglang's FusedMoE quant_method tries to materialize these
    experts. The fork-specific check in ``sglang/srt/layers/quantization/
    fp8.py:process_weights_after_loading_block_quant`` raises
    ``NotImplementedError: DeepSeekV4 FP4 experts now require a native FP4
    MoE backend.`` before any ``ignored_layers`` / ``is_layer_skipped``
    logic gets a chance — both the upstream ``SGLANG_FP8_IGNORED_LAYERS``
    env var and ``quantization_config.ignored_layers`` are bypassed by
    that hard-coded check. The sglang error message suggests
    ``--moe-runner-backend marlin``, but Marlin software-emulates FP4 with
    INT4 unpack + bf16 compute, producing perf numbers that don't reflect
    how DSv4-Flash actually deploys on Hopper. So we skip the fp8_block
    sweep on pre-Blackwell entirely; the bfloat16 sweep still runs and
    measures the projection path Hopper would deploy.
    """
    try:
        import torch as _t

        if not _t.cuda.is_available():
            return False
        return _t.cuda.get_device_capability(0)[0] >= 10
    except Exception:
        return False


def _dsv4_flash_module_precision_combos(phase: str):
    """``(compute_dtype, kv_cache_dtype, gemm_type)`` triples.

    DeepseekV4ForCausalLM rejects bfloat16 KV cache (asserts at load time),
    so we only emit fp8 KV.  ``gemm_type`` switches projection dispatch:
      * ``bfloat16``  — projections through cuBLASLt nvjet kernels
      * ``fp8_block`` — fp8 block-quantised weights → DeepGEMM
                        ``sm90_fp8_gemm_1d2d_impl`` (matches production)

    ``fp8_block`` is omitted on pre-Blackwell parts; see
    ``_has_native_fp4_experts`` for the kernel-side rationale.
    """
    del phase
    combos = [("bfloat16", "fp8", "bfloat16")]
    if _has_native_fp4_experts():
        combos.append(("bfloat16", "fp8", "fp8_block"))
    else:
        print(
            "[dsv4-flash-test-cases] device lacks native FP4 experts (pre-Blackwell); "
            "omitting fp8_block from gemm_type sweep"
        )
    return combos


def _dsv4_flash_module_filter_pairs(mode: str, batch_sizes, seq_lens):
    """Drop ``(bs, sl)`` pairs that exceed KV pool / kernel limits.

    Context (b * s):
        ≤ 8192 — matches sglang's default ``chunked_prefill_size``.
    Generation (b * s):
        ≤ 1M overall, with per-sl batch caps for long contexts (sl≥8192→bs≤64,
        sl≥32768→bs≤16, sl≥65536→bs≤8, sl≥131072→bs≤4, sl≥262144→bs≤2,
        sl≥524288→bs==1).  Ensures bs=1 is always allowed at every sl.
    """
    is_context = mode == "context"
    pairs = []
    for bs in batch_sizes:
        for sl in seq_lens:
            if is_context:
                if bs * sl > 8192:
                    continue
            else:
                if bs * sl > 1024 * 1024:
                    continue
                if sl >= 524288 and bs > 1:
                    continue
                if sl >= 262144 and bs > 2:
                    continue
                if sl >= 131072 and bs > 4:
                    continue
                if sl >= 65536 and bs > 8:
                    continue
                if sl >= 32768 and bs > 16:
                    continue
                if sl >= 8192 and bs > 64:
                    continue
            pairs.append((bs, sl))
    return pairs


def _build_dsv4_flash_module_test_cases(mode: str, attn_kinds=None):
    """One case per ``(attn_kind, tp_size, gemm_type, batch_size)``.

    Test case shape (9 elements; ``perf_filename`` is bound by collect.py
    via OpEntry, NOT in the tuple)::

        [0, batch_size, tp_size, kv_cache_dtype, compute_dtype, gemm_type,
         model_path, attn_kind, attention_backend]

    Each spawned subprocess builds ONE ``ModelRunner`` for ``(bs, max_sl)``
    and sweeps every valid sl for that bs internally.
    """
    config = _dsv4_flash_config()
    attn_kinds = tuple(attn_kinds) if attn_kinds is not None else _dsv4_flash_attention_kinds()
    pairs = _dsv4_flash_module_filter_pairs(
        mode,
        config["module_batch_sizes"],
        config["module_sequence_lengths"],
    )
    bs_set = sorted({bs for bs, _ in pairs})
    model_path = _dsv4_flash_model_path()

    cases: list[list] = []
    for attn_kind in attn_kinds:
        for compute_dtype, kv_dtype, gemm_type in _dsv4_flash_module_precision_combos(mode):
            for tp_size in config["module_tp_sizes"]:
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


def get_dsv4_flash_csa_context_test_cases():
    if not _dsv4_flash_active():
        return []
    return _build_dsv4_flash_module_test_cases("context", ("csa",))


def get_dsv4_flash_hca_context_test_cases():
    if not _dsv4_flash_active():
        return []
    return _build_dsv4_flash_module_test_cases("context", ("hca",))


def get_dsv4_flash_csa_generation_test_cases():
    if not _dsv4_flash_active():
        return []
    return _build_dsv4_flash_module_test_cases("generation", ("csa",))


def get_dsv4_flash_hca_generation_test_cases():
    if not _dsv4_flash_active():
        return []
    return _build_dsv4_flash_module_test_cases("generation", ("hca",))


DSV4_FLASH_SPARSE_KERNELS = ("paged_mqa_logits", "hca_attn")


def _build_dsv4_flash_sparse_test_cases(
    kernels=DSV4_FLASH_SPARSE_KERNELS,
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
    config = _dsv4_flash_config()
    sparse_tp_sizes = config["sparse_tp_sizes"]
    bs_list = list(bs_list) if bs_list is not None else list(config["sparse_batch_sizes"])
    isl_list = list(isl_list) if isl_list is not None else list(config["sparse_input_lengths"])
    past_kv_list = list(past_kv_list) if past_kv_list is not None else list(config["sparse_past_kv_lengths"])
    tp_list_attn = list(tp_list_attn) if tp_list_attn is not None else list(sparse_tp_sizes["hca_attn"])
    tp_list_indexer = (
        list(tp_list_indexer) if tp_list_indexer is not None else list(sparse_tp_sizes["paged_mqa_logits"])
    )
    sparse_chunk_prefill_size = int(config["sparse_chunk_prefill_size"])
    sparse_max_full_s = int(config["sparse_max_full_sequence_length"])
    model_path = _dsv4_flash_model_path()

    cases = []
    for kernel in kernels:
        tp_list = tp_list_attn if kernel == "hca_attn" else tp_list_indexer
        for tp_size in tp_list:
            for bs in bs_list:
                for isl in isl_list:
                    if bs * isl > sparse_chunk_prefill_size:
                        continue
                    for past_kv in past_kv_list:
                        if bs * (isl + past_kv) > sparse_max_full_s:
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


def _dsv4_flash_sparse_smoke_or_full(kernel: str):
    if not _dsv4_flash_active():
        return []
    if "--smoke" in sys.argv:
        return [[1, 1024, 8192, 1, kernel, _dsv4_flash_model_path()]]
    return _build_dsv4_flash_sparse_test_cases(kernels=(kernel,))


def get_dsv4_flash_paged_mqa_logits_test_cases():
    """paged_mqa_logits sparse-kernel sweep (CSA indexer scoring)."""
    return _dsv4_flash_sparse_smoke_or_full("paged_mqa_logits")


def get_dsv4_flash_hca_attn_test_cases():
    """hca_attn sparse-kernel sweep (HCA c128 sparse FMLA)."""
    return _dsv4_flash_sparse_smoke_or_full("hca_attn")
