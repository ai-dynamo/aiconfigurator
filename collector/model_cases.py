# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model-centric collector case planning.

Collector v2 keeps model/SM intent in YAML and leaves kernel collectors focused
on generating runnable test cases. The planner merges:

1. shared base op cases,
2. one model's extra cases or all model case files for full mode,
3. optional SM-centric exceptions.

The resulting per-op plan is consumed by ``collector.collect`` to run only the
cases needed for a model/SM healing pass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

try:
    from helper import create_test_case_id
except ModuleNotFoundError:
    from collector.helper import create_test_case_id


COLLECTOR_ROOT = Path(__file__).resolve().parent
CASE_ROOT = COLLECTOR_ROOT / "cases"
BASE_OP_CASES_DIR = CASE_ROOT / "base_ops"
MODEL_CASES_DIR = CASE_ROOT / "models"
SM_EXCEPTIONS_DIR = CASE_ROOT / "sm_exceptions"
SYSTEMS_DIR = COLLECTOR_ROOT.parent / "src" / "aiconfigurator" / "systems"


@dataclass(slots=True)
class CaseSelector:
    """A selector for either included cases or excluded cases."""

    all_cases: bool = False
    case_specs: list[dict[str, Any]] = field(default_factory=list)
    rules: list[dict[str, Any]] = field(default_factory=list)
    case_ids: set[str] = field(default_factory=set)
    contains: set[str] = field(default_factory=set)
    indices: set[int] = field(default_factory=set)
    index_ranges: list[tuple[int, int]] = field(default_factory=list)
    limit: int | None = None

    def merge(self, other: CaseSelector) -> None:
        self.all_cases = self.all_cases or other.all_cases
        self.case_specs.extend(other.case_specs)
        self.rules.extend(other.rules)
        self.case_ids.update(other.case_ids)
        self.contains.update(other.contains)
        self.indices.update(other.indices)
        self.index_ranges.extend(other.index_ranges)
        if other.limit is not None:
            self.limit = other.limit if self.limit is None else min(self.limit, other.limit)

    def has_specific_selectors(self) -> bool:
        return bool(self.rules or self.case_ids or self.contains or self.indices or self.index_ranges)


@dataclass(slots=True)
class OpCasePlan:
    """Merged include/exclude rules for a single op."""

    include: CaseSelector = field(default_factory=lambda: CaseSelector(all_cases=True))
    exclude: CaseSelector = field(default_factory=CaseSelector)
    expected_failures: CaseSelector = field(default_factory=CaseSelector)
    drop: bool = False


@dataclass(slots=True)
class CollectionCasePlan:
    """Case plan for one backend collection run."""

    backend: str
    model_path: str | None
    model_architecture: str | None
    gpu_type: str | None
    sm_version: int | None
    op_cases: dict[str, OpCasePlan]
    base_cases_path: Path
    model_cases_paths: list[Path] = field(default_factory=list)
    sm_exceptions_path: Path | None = None
    requested_model_path: str | None = None

    @property
    def ops(self) -> list[str]:
        return sorted(self.op_cases)

    def to_log_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "model_path": self.model_path,
            "model_architecture": self.model_architecture,
            "requested_model_path": self.requested_model_path,
            "gpu_type": self.gpu_type,
            "sm_version": self.sm_version,
            "ops": self.ops,
            "base_cases_path": str(self.base_cases_path),
            "model_cases_paths": [str(path) for path in self.model_cases_paths],
            "sm_exceptions_path": str(self.sm_exceptions_path) if self.sm_exceptions_path else None,
        }


def sanitize_case_filename(value: str) -> str:
    """Return a stable filename stem for a model path or GPU id."""
    safe = []
    for char in value:
        if char.isalnum() or char in {".", "_", "-"}:
            safe.append(char)
        elif char == "/":
            safe.append("--")
        else:
            safe.append("_")
    return "".join(safe)


def default_model_cases_path(model_path: str) -> Path:
    return MODEL_CASES_DIR / f"{sanitize_case_filename(model_path)}_cases.yaml"


def default_architecture_cases_path(model_architecture: str) -> Path:
    return MODEL_CASES_DIR / f"{sanitize_case_filename(model_architecture)}_cases.yaml"


def default_sm_exceptions_path(sm_version: int) -> Path:
    return SM_EXCEPTIONS_DIR / f"sm{int(sm_version)}_exceptions.yaml"


def default_system_spec_path(gpu_type: str) -> Path:
    return SYSTEMS_DIR / f"{sanitize_case_filename(gpu_type)}.yaml"


def load_yaml_file(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise TypeError(f"{path}: top-level YAML value must be a mapping")
    return data


def _base_ops_dir(base_data: dict[str, Any], base_path: Path) -> Path:
    raw_dir = base_data.get("base_ops_dir", BASE_OP_CASES_DIR.name)
    path = Path(str(raw_dir))
    if path.is_absolute():
        return path
    return base_path.parent / path


def _load_base_case_files(base_path: Path) -> list[dict[str, Any]]:
    """Load per-op base case YAML files from a directory or legacy catalog."""
    if base_path.is_dir():
        return [load_yaml_file(path) for path in sorted(base_path.glob("*.yaml"))]

    base_data = load_yaml_file(base_path)
    if "base_ops" not in base_data and "base_ops_dir" not in base_data:
        return [base_data]

    data = [base_data]
    base_ops_dir = _base_ops_dir(base_data, base_path)
    if not base_ops_dir.exists():
        return data

    configured_files = base_data.get("base_ops")
    if configured_files is None:
        paths = sorted(base_ops_dir.glob("*.yaml"))
    else:
        paths = [base_ops_dir / str(filename) for filename in _as_list(configured_files, field_name="base_ops")]

    data.extend(load_yaml_file(path) for path in paths)
    return data


def resolve_sm_version(*, gpu_type: str | None = None, sm_version: int | str | None = None) -> int | None:
    """Return the explicit SM version, or infer it from a system YAML file."""
    if sm_version is not None:
        return int(sm_version)
    if not gpu_type:
        return None
    path = default_system_spec_path(gpu_type)
    if not path.exists():
        return None
    data = load_yaml_file(path)
    gpu = data.get("gpu", {})
    if not isinstance(gpu, dict) or gpu.get("sm_version") is None:
        return None
    return int(gpu["sm_version"])


def _section(data: dict[str, Any], canonical_name: str) -> dict[str, Any]:
    value = data.get(canonical_name)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError(f"{canonical_name} must be a mapping")
    return value


def _as_list(value: Any, *, field_name: str) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    raise ValueError(f"{field_name} must be a list")


def _parse_index_ranges(raw: Any) -> list[tuple[int, int]]:
    ranges = []
    for item in _as_list(raw, field_name="ranges"):
        if isinstance(item, str) and "-" in item:
            start, end = item.split("-", 1)
            ranges.append((int(start), int(end)))
        elif isinstance(item, list | tuple) and len(item) == 2:
            ranges.append((int(item[0]), int(item[1])))
        else:
            raise ValueError(f"Invalid index range {item!r}; expected 'start-end' or [start, end]")
    return ranges


def _parse_selector(raw: Any, *, default_all: bool) -> CaseSelector:
    if raw is None:
        return CaseSelector(all_cases=default_all)
    if raw == "all" or raw is True:
        return CaseSelector(all_cases=True)
    if raw is False:
        return CaseSelector(all_cases=False)
    if isinstance(raw, list):
        selector = CaseSelector()
        for item in raw:
            if isinstance(item, dict):
                selector.case_specs.append(item)
            else:
                selector.case_ids.add(str(item))
        selector.all_cases = bool(selector.case_specs and not selector.case_ids)
        return selector
    if not isinstance(raw, dict):
        raise TypeError(f"case selector must be 'all', a list, or a mapping; got {type(raw).__name__}")

    has_specific_selector = any(
        field_name in raw for field_name in ("case_ids", "contains", "indices", "ranges", "rules")
    )
    cases = raw.get("cases", "all" if default_all and not has_specific_selector else None)
    selector = CaseSelector(all_cases=cases == "all" or cases is True)
    if cases not in (None, "all", True):
        if not isinstance(cases, list):
            raise ValueError("'cases' must be 'all' or a list of case ids/case specs")
        for case in cases:
            if isinstance(case, dict):
                selector.case_specs.append(case)
            else:
                selector.case_ids.add(str(case))
        if selector.case_specs and not selector.case_ids:
            selector.all_cases = True

    selector.case_ids.update(str(item) for item in _as_list(raw.get("case_ids"), field_name="case_ids"))
    selector.contains.update(str(item) for item in _as_list(raw.get("contains"), field_name="contains"))
    selector.indices.update(int(item) for item in _as_list(raw.get("indices"), field_name="indices"))
    for rule in _as_list(raw.get("rules"), field_name="rules"):
        if not isinstance(rule, dict):
            raise TypeError("selector rules must be mappings")
        selector.rules.append(rule)
    selector.index_ranges.extend(_parse_index_ranges(raw.get("ranges")))
    if raw.get("limit") is not None:
        selector.limit = int(raw["limit"])
    return selector


def _ensure_op_plan(op_cases: dict[str, OpCasePlan], op: str) -> OpCasePlan:
    if op not in op_cases:
        # A newly discovered op has not selected any generated cases yet.  The
        # merge that activates it (``model_ops`` or an op case section) decides
        # whether it means "all" or a narrower selector.  Starting at ``all``
        # makes a YAML ``case_ids``/``rules`` selector impossible to narrow.
        op_cases[op] = OpCasePlan(include=CaseSelector())
    return op_cases[op]


def _merge_model_ops(op_cases: dict[str, OpCasePlan], data: dict[str, Any]) -> None:
    for op in _as_list(data.get("model_ops"), field_name="model_ops"):
        _ensure_op_plan(op_cases, str(op)).include.all_cases = True


def _merge_case_section(
    op_cases: dict[str, OpCasePlan],
    section: dict[str, Any],
    *,
    allowed_ops: set[str] | None = None,
    override_specific_selectors: bool = False,
) -> None:
    for op, raw_selector in section.items():
        op = str(op)
        if allowed_ops is not None and op not in allowed_ops:
            continue
        plan = _ensure_op_plan(op_cases, op)
        selector = _parse_selector(raw_selector, default_all=True)
        if override_specific_selectors and selector.has_specific_selectors() and not selector.all_cases:
            # Base files activate generator recipes and therefore select all
            # generated cases by default.  A model's concrete selector is an
            # intentional narrowing of that base workload, not an OR with the
            # inherited ``all``.  Keep recipe specs for plan inspection while
            # replacing the runnable selector state.
            selector.case_specs = [*plan.include.case_specs, *selector.case_specs]
            plan.include = selector
        else:
            plan.include.merge(selector)


def _merge_exception_section(op_cases: dict[str, OpCasePlan], section: dict[str, Any]) -> None:
    for op, raw_selector in section.items():
        # Exceptions constrain an existing workload; they must never activate a
        # model-unrelated op merely because an SM catalog happens to mention it.
        plan = op_cases.get(str(op))
        if plan is None:
            continue
        if raw_selector is True:
            plan.drop = True
            continue
        if isinstance(raw_selector, dict) and raw_selector.get("drop"):
            plan.drop = True
        plan.exclude.merge(_parse_selector(raw_selector, default_all=False))


_MATCH_OPERATOR_SUFFIXES = {
    "_lt": "lt",
    "_lte": "lte",
    "_gt": "gt",
    "_gte": "gte",
    "_ne": "ne",
}


def _normalize_match_spec(match_spec: dict[str, Any]) -> dict[str, Any]:
    """Normalize shorthand fields such as num_tokens_gte into rule operators."""
    normalized: dict[str, Any] = {}
    for key, value in match_spec.items():
        field_name = str(key)
        op_name = None
        for suffix, candidate_op in _MATCH_OPERATOR_SUFFIXES.items():
            if field_name.endswith(suffix):
                field_name = field_name[: -len(suffix)]
                op_name = candidate_op
                break
        if op_name is None:
            normalized[field_name] = value
            continue
        existing = normalized.setdefault(field_name, {})
        if isinstance(existing, dict):
            existing[op_name] = value
        else:
            normalized[field_name] = {"eq": existing, op_name: value}
    return normalized


def _merge_match_specs(*match_specs: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for match_spec in match_specs:
        for field_name, condition in _normalize_match_spec(match_spec).items():
            existing = merged.get(field_name)
            if isinstance(existing, dict) and isinstance(condition, dict):
                existing.update(condition)
            else:
                merged[field_name] = condition
    return merged


def _known_exception_rule(raw_exception: dict[str, Any], match_spec: dict[str, Any]) -> dict[str, Any]:
    rule = {
        "reason_type": raw_exception.get("reason_type", "known_exception"),
        "reason": raw_exception.get("reason"),
        "source": raw_exception.get("source"),
        "fields": raw_exception.get("fields") or raw_exception.get("case_fields") or [],
        "match": match_spec,
    }
    if raw_exception.get("version_prefixes") is not None:
        rule["version_prefixes"] = raw_exception["version_prefixes"]
    if raw_exception.get("conditions") is not None:
        rule["conditions"] = raw_exception["conditions"]
    return rule


def _known_exception_rules(raw_exception: dict[str, Any]) -> list[dict[str, Any]]:
    rules: list[dict[str, Any]] = []
    base_match = raw_exception.get("match") or raw_exception.get("where") or {}
    if not isinstance(base_match, dict):
        raise TypeError("known_exceptions match/where must be a mapping")
    if base_match or raw_exception.get("conditions"):
        rules.append(_known_exception_rule(raw_exception, _normalize_match_spec(base_match)))

    for group in _as_list(raw_exception.get("threshold_groups"), field_name="known_exceptions.threshold_groups"):
        if not isinstance(group, dict):
            raise TypeError("known_exceptions threshold_groups entries must be mappings")
        group_match = group.get("match") or {}
        if not isinstance(group_match, dict):
            raise TypeError("known_exceptions threshold_groups.match must be a mapping")
        for threshold in _as_list(group.get("thresholds"), field_name="known_exceptions.threshold_groups.thresholds"):
            if not isinstance(threshold, dict):
                raise TypeError("known_exceptions threshold entries must be mappings")
            rule = _known_exception_rule(raw_exception, _merge_match_specs(group_match, threshold))
            if group.get("label"):
                rule["label"] = str(group["label"])
            rules.append(rule)
    return rules


def _merge_known_exception_section(op_cases: dict[str, OpCasePlan], data: dict[str, Any], backend: str) -> None:
    for raw_exception in _as_list(data.get("known_exceptions"), field_name="known_exceptions"):
        if not isinstance(raw_exception, dict):
            raise TypeError("known_exceptions entries must be mappings")
        framework = raw_exception.get("framework")
        if framework is not None and str(framework) != backend:
            continue
        op = raw_exception.get("op")
        if not op:
            raise ValueError("known_exceptions entries must include op")
        selector = _parse_selector(raw_exception, default_all=False)
        selector.rules.extend(_known_exception_rules(raw_exception))
        plan = op_cases.get(str(op))
        if plan is not None:
            plan.expected_failures.merge(selector)


def _merge_case_file(
    op_cases: dict[str, OpCasePlan],
    data: dict[str, Any],
    backend: str,
    *,
    allowed_ops: set[str] | None = None,
    override_specific_selectors: bool = False,
) -> None:
    if allowed_ops is None:
        _merge_model_ops(op_cases, data)
    else:
        model_ops = {str(op) for op in _as_list(data.get("model_ops"), field_name="model_ops")}
        for op in sorted(model_ops & allowed_ops):
            _ensure_op_plan(op_cases, op).include.all_cases = True
    _merge_case_section(
        op_cases,
        _section(data, "all_frameworks_op_cases"),
        allowed_ops=allowed_ops,
        override_specific_selectors=override_specific_selectors,
    )
    framework_cases = _section(data, "framework_specific_op_cases")
    backend_cases = framework_cases.get(backend, {})
    if backend_cases is None:
        return
    if not isinstance(backend_cases, dict):
        raise TypeError(f"framework_specific_op_cases.{backend} must be a mapping")
    _merge_case_section(
        op_cases,
        backend_cases,
        allowed_ops=allowed_ops,
        override_specific_selectors=override_specific_selectors,
    )


def _merge_exception_file(op_cases: dict[str, OpCasePlan], data: dict[str, Any], backend: str) -> None:
    _merge_exception_section(op_cases, _section(data, "all_frameworks_op_exceptions"))
    framework_exceptions = _section(data, "framework_specific_op_exceptions")
    backend_exceptions = framework_exceptions.get(backend, {})
    if backend_exceptions is None:
        return
    if not isinstance(backend_exceptions, dict):
        raise TypeError(f"framework_specific_op_exceptions.{backend} must be a mapping")
    _merge_exception_section(op_cases, backend_exceptions)
    _merge_known_exception_section(op_cases, data, backend)


def _model_case_architecture(data: dict[str, Any]) -> str | None:
    value = data.get("architecture") or data.get("model_architecture")
    return str(value) if value else None


def _model_case_paths(data: dict[str, Any]) -> list[str]:
    values = []
    primary = data.get("model_path")
    if primary:
        values.append(str(primary))
    aliases = data.get("model_paths", [])
    if aliases is None:
        aliases = []
    if not isinstance(aliases, list):
        raise TypeError("model_paths must be a list")
    values.extend(str(alias) for alias in aliases)

    deduped = []
    seen = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _primary_model_path(data: dict[str, Any]) -> str | None:
    values = _model_case_paths(data)
    return values[0] if values else None


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    deduped = []
    seen = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(path)
    return deduped


def _matching_model_case_files(*, model_path: str | None, model_architecture: str | None) -> list[Path]:
    matches = []
    for path in sorted(MODEL_CASES_DIR.glob("*_cases.yaml")):
        data = load_yaml_file(path)
        if model_architecture and _model_case_architecture(data) == model_architecture:
            matches.append(path)
            continue
        if model_path and model_path in _model_case_paths(data):
            matches.append(path)
    return _dedupe_paths(matches)


def _load_model_case_files(
    model_path: str | None,
    model_architecture: str | None,
    model_cases_path: str | None,
    full: bool,
) -> list[Path]:
    if model_cases_path:
        return [Path(model_cases_path).expanduser().resolve()]
    if full:
        return sorted(MODEL_CASES_DIR.glob("*_cases.yaml"))
    matches = _matching_model_case_files(model_path=model_path, model_architecture=model_architecture)
    if matches:
        return matches
    if model_architecture:
        path = default_architecture_cases_path(model_architecture)
        if path.exists():
            return [path]
    if model_path:
        path = default_model_cases_path(model_path)
        return [path] if path.exists() else []
    return []


def _base_case_file_ops(data: dict[str, Any], backend: str) -> set[str]:
    """Return collectable op names exposed by one base case document."""

    ops = {str(op) for op in _as_list(data.get("model_ops"), field_name="model_ops")}
    ops.update(str(op) for op in _section(data, "all_frameworks_op_cases"))
    framework_cases = _section(data, "framework_specific_op_cases")
    backend_cases = framework_cases.get(backend, {})
    if backend_cases is not None:
        if not isinstance(backend_cases, dict):
            raise TypeError(f"framework_specific_op_cases.{backend} must be a mapping")
        ops.update(str(op) for op in backend_cases)
    return ops


def _selected_base_ops(
    base_data_files: list[dict[str, Any]],
    model_data: list[dict[str, Any]],
    backend: str,
    model_path: str | None,
) -> set[str]:
    """Resolve the shared recipe ops required by the selected model plans.

    ``base_ops`` is an explicit allowlist.  Legacy model files that only set
    ``include_base: true`` receive the small universal set declared through
    base-file ``model_ops``; they no longer activate every auxiliary recipe
    merely because another base YAML was added to the repository.
    """

    available_ops: set[str] = set()
    default_ops: set[str] = set()
    for data in base_data_files:
        available_ops.update(_base_case_file_ops(data, backend))
        default_ops.update(str(op) for op in _as_list(data.get("model_ops"), field_name="model_ops"))
    # Preserve compatibility with a legacy monolithic base catalog that did not
    # distinguish universal model ops from auxiliary recipes.
    if not default_ops:
        default_ops = set(available_ops)

    if not model_data:
        return default_ops

    selected: set[str] = set()
    for data in model_data:
        explicit = data.get("base_ops")
        if explicit is not None:
            selected.update(str(op) for op in _as_list(explicit, field_name="base_ops"))
        elif bool(data.get("include_base", True)):
            selected.update(default_ops)

        framework_base_ops = data.get("framework_specific_base_ops", {})
        if not isinstance(framework_base_ops, dict):
            raise TypeError("framework_specific_base_ops must be a mapping")
        backend_base_ops = framework_base_ops.get(backend, [])
        selected.update(
            str(op)
            for op in _as_list(
                backend_base_ops,
                field_name=f"framework_specific_base_ops.{backend}",
            )
        )

        model_specific_base_ops = data.get("model_specific_base_ops", {})
        if not isinstance(model_specific_base_ops, dict):
            raise TypeError("model_specific_base_ops must be a mapping")
        if model_path is None:
            # Full/raw plans collect the union needed by every listed artifact.
            selected_model_specific_ops = model_specific_base_ops.values()
        else:
            selected_model_specific_ops = [model_specific_base_ops.get(model_path, [])]
        for artifact_config in selected_model_specific_ops:
            if isinstance(artifact_config, dict):
                artifact_ops = artifact_config.get(backend, [])
                field_name = f"model_specific_base_ops.<model_path>.{backend}"
            else:
                # A flat list remains valid for artifacts whose recipe applies
                # to every backend exposing that base op.
                artifact_ops = artifact_config
                field_name = "model_specific_base_ops.<model_path>"
            selected.update(
                str(op)
                for op in _as_list(
                    artifact_ops,
                    field_name=field_name,
                )
            )

    unknown = selected - available_ops
    if unknown:
        raise ValueError(f"Unknown base_ops entries for backend {backend}: {sorted(unknown)}")
    return selected


def build_collection_case_plan(
    *,
    backend: str,
    model_path: str | None = None,
    model_architecture: str | None = None,
    gpu_type: str | None = None,
    sm_version: int | str | None = None,
    base_cases_path: str | None = None,
    model_cases_path: str | None = None,
    sm_exceptions_path: str | None = None,
    gpu_exceptions_path: str | None = None,
    full: bool = False,
) -> CollectionCasePlan:
    """Build a model/SM-aware op and case plan for one backend."""
    base_path = Path(base_cases_path).expanduser().resolve() if base_cases_path else BASE_OP_CASES_DIR
    base_data_files = _load_base_case_files(base_path)
    requested_model_path = model_path
    model_paths = _load_model_case_files(model_path, model_architecture, model_cases_path, full)
    model_data = [load_yaml_file(path) for path in model_paths]
    if model_path is None and len(model_data) == 1:
        model_path = _primary_model_path(model_data[0])
    if model_architecture is None and len(model_data) == 1:
        model_architecture = _model_case_architecture(model_data[0])

    op_cases: dict[str, OpCasePlan] = {}
    selected_base_ops = _selected_base_ops(base_data_files, model_data, backend, model_path)
    for base_data in base_data_files:
        _merge_case_file(op_cases, base_data, backend, allowed_ops=selected_base_ops)
    for data in model_data:
        # A targeted model may intentionally narrow a broad base recipe. Full
        # mode is additive: a later model selector must not replace cases
        # selected by an earlier model document.
        _merge_case_file(op_cases, data, backend, override_specific_selectors=not full)

    resolved_sm_version = resolve_sm_version(gpu_type=gpu_type, sm_version=sm_version)
    resolved_sm_exceptions_path = None
    explicit_exceptions_path = sm_exceptions_path or gpu_exceptions_path
    if explicit_exceptions_path:
        resolved_sm_exceptions_path = Path(explicit_exceptions_path).expanduser().resolve()
    elif resolved_sm_version is not None:
        default_path = default_sm_exceptions_path(resolved_sm_version)
        if default_path.exists():
            resolved_sm_exceptions_path = default_path

    if resolved_sm_exceptions_path:
        _merge_exception_file(op_cases, load_yaml_file(resolved_sm_exceptions_path), backend)

    op_cases = {op: plan for op, plan in op_cases.items() if not plan.drop}
    return CollectionCasePlan(
        backend=backend,
        model_path=model_path,
        model_architecture=model_architecture,
        gpu_type=gpu_type,
        sm_version=resolved_sm_version,
        op_cases=op_cases,
        base_cases_path=base_path,
        model_cases_paths=model_paths,
        sm_exceptions_path=resolved_sm_exceptions_path,
        requested_model_path=requested_model_path,
    )


def _index_matches(index: int, selector: CaseSelector) -> bool:
    return index in selector.indices or any(start <= index <= end for start, end in selector.index_ranges)


_MISSING = object()


def _field_names(rule: dict[str, Any]) -> list[str]:
    raw_fields = rule.get("fields", [])
    if isinstance(raw_fields, dict):
        return [str(field) for field, _index in sorted(raw_fields.items(), key=lambda item: int(item[1]))]
    if not isinstance(raw_fields, list):
        raise TypeError("rule fields must be a list or mapping")
    return [str(field) for field in raw_fields]


def _case_field_value(test_case: Any, field_name: str, fields: list[str]) -> Any:
    if isinstance(test_case, dict):
        return test_case.get(field_name, _MISSING)
    if hasattr(test_case, field_name):
        return getattr(test_case, field_name)
    if isinstance(test_case, list | tuple) and field_name in fields:
        index = fields.index(field_name)
        if index < len(test_case):
            return test_case[index]
    return _MISSING


def _comparison_matches(value: Any, op: str, expected: Any, fields: list[str], test_case: Any) -> bool:
    if value is _MISSING:
        return False
    if op.endswith("_field"):
        op = op.removesuffix("_field")
        expected = _case_field_value(test_case, str(expected), fields)
        if expected is _MISSING:
            return False
    if op.startswith("any_"):
        inner_op = op.removeprefix("any_")
        values = value if isinstance(value, list | tuple | set) else [value]
        return any(_comparison_matches(item, inner_op, expected, fields, test_case) for item in values)
    if op.startswith("all_"):
        inner_op = op.removeprefix("all_")
        values = value if isinstance(value, list | tuple | set) else [value]
        return all(_comparison_matches(item, inner_op, expected, fields, test_case) for item in values)

    if op == "eq":
        return value == expected
    if op == "ne":
        return value != expected
    if op == "in":
        return value in expected
    if op == "not_in":
        return value not in expected
    if op == "lt":
        return value < expected
    if op == "lte":
        return value <= expected
    if op == "gt":
        return value > expected
    if op == "gte":
        return value >= expected
    if op == "contains":
        return expected in value
    if op == "not_contains":
        return expected not in value
    if op == "not_in":
        return value not in _as_list(expected, field_name="not_in")
    if op == "prefix":
        return str(value).startswith(str(expected))
    if op == "suffix":
        return str(value).endswith(str(expected))
    if op in {"mod_eq", "mod_ne"}:
        if isinstance(expected, dict):
            divisor = int(expected["divisor"])
            remainder = int(expected.get("remainder", 0))
        else:
            divisor = int(expected[0])
            remainder = int(expected[1])
        matches = value % divisor == remainder
        return matches if op == "mod_eq" else not matches
    raise ValueError(f"Unsupported rule comparison operator {op!r}")


def _field_condition_matches(value: Any, condition: Any, fields: list[str], test_case: Any) -> bool:
    if isinstance(condition, dict):
        return all(
            _comparison_matches(value, str(op), expected, fields, test_case) for op, expected in condition.items()
        )
    if isinstance(condition, list):
        return value in condition
    return value == condition


def _numeric_condition_matches(value: Any, condition: dict[str, Any], fields: list[str], test_case: Any) -> bool:
    return all(
        _comparison_matches(value, str(op), expected, fields, test_case)
        for op, expected in condition.items()
        if op not in {"fields", "numerator", "denominator", "field"}
    )


def _computed_condition_matches(condition: dict[str, Any], fields: list[str], test_case: Any) -> bool:
    if "product" in condition:
        product_spec = condition["product"]
        value = 1
        for field_name in _as_list(product_spec.get("fields"), field_name="product.fields"):
            field_value = _case_field_value(test_case, str(field_name), fields)
            if field_value is _MISSING:
                return False
            value *= field_value
        return _numeric_condition_matches(value, product_spec, fields, test_case)

    if "ratio" in condition:
        ratio_spec = condition["ratio"]
        numerator = _case_field_value(test_case, str(ratio_spec["numerator"]), fields)
        denominator = _case_field_value(test_case, str(ratio_spec["denominator"]), fields)
        if numerator is _MISSING or denominator in (_MISSING, 0):
            return False
        return _numeric_condition_matches(numerator / denominator, ratio_spec, fields, test_case)

    if "floor_div" in condition:
        div_spec = condition["floor_div"]
        numerator = _case_field_value(test_case, str(div_spec["numerator"]), fields)
        denominator = _case_field_value(test_case, str(div_spec["denominator"]), fields)
        if numerator is _MISSING or denominator in (_MISSING, 0):
            return False
        return _numeric_condition_matches(numerator // denominator, div_spec, fields, test_case)

    if "field" in condition:
        field_name = str(condition["field"])
        value = _case_field_value(test_case, field_name, fields)
        return _numeric_condition_matches(value, condition, fields, test_case)

    raise ValueError(f"Unsupported computed rule condition {condition!r}")


def _version_matches(rule: dict[str, Any], runtime_version: str | None) -> bool:
    prefixes = rule.get("version_prefixes")
    if prefixes is None:
        return True
    if runtime_version is None:
        return False
    return any(runtime_version.startswith(str(prefix)) for prefix in _as_list(prefixes, field_name="version_prefixes"))


def _rule_matches(test_case: Any, rule: dict[str, Any], *, runtime_version: str | None) -> bool:
    if not _version_matches(rule, runtime_version):
        return False
    fields = _field_names(rule)
    match_spec = rule.get("match") or rule.get("where") or {}
    if not isinstance(match_spec, dict):
        raise TypeError("rule match/where must be a mapping")
    for field_name, condition in match_spec.items():
        value = _case_field_value(test_case, str(field_name), fields)
        if not _field_condition_matches(value, condition, fields, test_case):
            return False
    conditions = _as_list(rule.get("conditions"), field_name="conditions")
    for condition in conditions:
        if not isinstance(condition, dict):
            raise TypeError("rule conditions must be mappings")
        if not _computed_condition_matches(condition, fields, test_case):
            return False
    return bool(match_spec or conditions)


def _case_matches(
    test_case: Any,
    case_id: str,
    index: int,
    selector: CaseSelector,
    *,
    runtime_version: str | None,
) -> bool:
    return (
        _selector_match_details(
            test_case,
            case_id,
            index,
            selector,
            runtime_version=runtime_version,
        )
        is not None
    )


def _rule_match_details(rule: dict[str, Any]) -> dict[str, Any]:
    details = {
        "selector": "rule",
        "reason_type": str(rule.get("reason_type") or "expected_exception"),
    }
    if rule.get("source") is not None:
        details["reference_source"] = str(rule["source"])
    for field_name in ("reason", "label"):
        if rule.get(field_name) is not None:
            details[field_name] = str(rule[field_name])
    return details


def _selector_match_details(
    test_case: Any,
    case_id: str,
    index: int,
    selector: CaseSelector,
    *,
    runtime_version: str | None,
) -> dict[str, Any] | None:
    if selector.all_cases:
        return {"selector": "all", "reason_type": "expected_exception"}
    if case_id in selector.case_ids:
        return {"selector": "case_id", "reason_type": "expected_exception"}
    case_text = str(test_case)
    for fragment in selector.contains:
        if fragment in case_text or fragment in case_id:
            return {"selector": "contains", "reason_type": "expected_exception", "contains": fragment}
    if _index_matches(index, selector):
        return {"selector": "index", "reason_type": "expected_exception"}
    for rule in selector.rules:
        if _rule_matches(test_case, rule, runtime_version=runtime_version):
            return _rule_match_details(rule)
    return None


def expected_failure_for_test_case(
    test_case: Any,
    *,
    plan: OpCasePlan | None,
    full_module_name: str,
    run_func_name: str,
    runtime_version: str | None = None,
    index: int = 0,
) -> dict[str, Any] | None:
    """Return expected-failure metadata when a known exception covers a failed case."""
    if plan is None:
        return None
    case_id = create_test_case_id(test_case, run_func_name, full_module_name)
    details = _selector_match_details(
        test_case,
        case_id,
        index,
        plan.expected_failures,
        runtime_version=runtime_version,
    )
    if details is not None:
        return {"case_id": case_id, "source": "known_exception", **details}
    return None


def filter_test_cases(
    test_cases: list[Any],
    *,
    plan: OpCasePlan | None,
    full_module_name: str,
    run_func_name: str,
    runtime_version: str | None = None,
) -> list[Any]:
    """Apply an op case plan to generated collector cases."""
    filtered, _skipped = filter_test_cases_with_report(
        test_cases,
        plan=plan,
        full_module_name=full_module_name,
        run_func_name=run_func_name,
        runtime_version=runtime_version,
    )
    return filtered


def filter_test_cases_with_report(
    test_cases: list[Any],
    *,
    plan: OpCasePlan | None,
    full_module_name: str,
    run_func_name: str,
    runtime_version: str | None = None,
) -> tuple[list[Any], list[dict[str, Any]]]:
    """Apply an op case plan and return cases skipped by expected SM exceptions."""
    if plan is None:
        return test_cases, []
    filtered = []
    skipped = []
    include = plan.include
    exclude = plan.exclude

    for index, test_case in enumerate(test_cases):
        case_id = create_test_case_id(test_case, run_func_name, full_module_name)
        if (include.has_specific_selectors() or not include.all_cases) and not _case_matches(
            test_case, case_id, index, include, runtime_version=runtime_version
        ):
            continue
        exclude_details = _selector_match_details(
            test_case,
            case_id,
            index,
            exclude,
            runtime_version=runtime_version,
        )
        if exclude_details is not None:
            skipped.append({"case_id": case_id, "index": index, "source": "sm_exception", **exclude_details})
            continue
        filtered.append(test_case)

    if include.limit is not None:
        filtered = filtered[: include.limit]
    return filtered, skipped
