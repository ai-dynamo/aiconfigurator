# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model-centric collector case planning.

Collector v2 keeps model/GPU intent in YAML and leaves kernel collectors focused
on generating runnable test cases. The planner merges:

1. shared base op cases,
2. one model's extra cases or all model case files for full mode,
3. optional GPU-centric exceptions.

The resulting per-op plan is consumed by ``collector.collect`` to run only the
cases needed for a model/GPU healing pass.
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
BASE_OP_CASES_PATH = CASE_ROOT / "base_op_cases.yaml"
MODEL_CASES_DIR = CASE_ROOT / "models"
GPU_EXCEPTIONS_DIR = CASE_ROOT / "gpus"


SECTION_ALIASES = {
    "all_frameworks_op_cases": ("all_frameworks_op_cases", "all_frameworks_Op_cases", "All_frameworks_Op_cases"),
    "framework_specific_op_cases": (
        "framework_specific_op_cases",
        "framework_specific_Op_cases",
        "Framework_specific_Op_cases",
    ),
    "all_frameworks_op_exceptions": (
        "all_frameworks_op_exceptions",
        "all_frameworks_Op_exceptions",
        "All_frameworks_Op_exceptions",
    ),
    "framework_specific_op_exceptions": (
        "framework_specific_op_exceptions",
        "framework_specific_Op_exceptions",
        "Framework_specific_Op_exceptions",
    ),
}


@dataclass(slots=True)
class CaseSelector:
    """A selector for either included cases or excluded cases."""

    all_cases: bool = False
    case_specs: list[dict[str, Any]] = field(default_factory=list)
    case_ids: set[str] = field(default_factory=set)
    contains: set[str] = field(default_factory=set)
    indices: set[int] = field(default_factory=set)
    index_ranges: list[tuple[int, int]] = field(default_factory=list)
    limit: int | None = None

    def merge(self, other: CaseSelector) -> None:
        self.all_cases = self.all_cases or other.all_cases
        self.case_specs.extend(other.case_specs)
        self.case_ids.update(other.case_ids)
        self.contains.update(other.contains)
        self.indices.update(other.indices)
        self.index_ranges.extend(other.index_ranges)
        if other.limit is not None:
            self.limit = other.limit if self.limit is None else min(self.limit, other.limit)

    def has_specific_selectors(self) -> bool:
        return bool(self.case_ids or self.contains or self.indices or self.index_ranges)


@dataclass(slots=True)
class OpCasePlan:
    """Merged include/exclude rules for a single op."""

    include: CaseSelector = field(default_factory=lambda: CaseSelector(all_cases=True))
    exclude: CaseSelector = field(default_factory=CaseSelector)
    drop: bool = False


@dataclass(slots=True)
class CollectionCasePlan:
    """Case plan for one backend collection run."""

    backend: str
    model_path: str | None
    model_architecture: str | None
    gpu_type: str | None
    op_cases: dict[str, OpCasePlan]
    base_cases_path: Path
    model_cases_paths: list[Path] = field(default_factory=list)
    gpu_exceptions_path: Path | None = None
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
            "ops": self.ops,
            "base_cases_path": str(self.base_cases_path),
            "model_cases_paths": [str(path) for path in self.model_cases_paths],
            "gpu_exceptions_path": str(self.gpu_exceptions_path) if self.gpu_exceptions_path else None,
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


def default_gpu_exceptions_path(gpu_type: str) -> Path:
    return GPU_EXCEPTIONS_DIR / f"{sanitize_case_filename(gpu_type)}_exceptions.yaml"


def load_yaml_file(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise TypeError(f"{path}: top-level YAML value must be a mapping")
    return data


def _section(data: dict[str, Any], canonical_name: str) -> dict[str, Any]:
    for name in SECTION_ALIASES[canonical_name]:
        value = data.get(name)
        if value is None:
            continue
        if not isinstance(value, dict):
            raise TypeError(f"{canonical_name} must be a mapping")
        return value
    return {}


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

    cases = raw.get("cases", "all" if default_all else None)
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
    selector.index_ranges.extend(_parse_index_ranges(raw.get("ranges")))
    if raw.get("limit") is not None:
        selector.limit = int(raw["limit"])
    return selector


def _ensure_op_plan(op_cases: dict[str, OpCasePlan], op: str) -> OpCasePlan:
    if op not in op_cases:
        op_cases[op] = OpCasePlan()
    return op_cases[op]


def _merge_model_ops(op_cases: dict[str, OpCasePlan], data: dict[str, Any]) -> None:
    for op in _as_list(data.get("model_ops"), field_name="model_ops"):
        _ensure_op_plan(op_cases, str(op))


def _merge_case_section(op_cases: dict[str, OpCasePlan], section: dict[str, Any]) -> None:
    for op, raw_selector in section.items():
        _ensure_op_plan(op_cases, str(op)).include.merge(_parse_selector(raw_selector, default_all=True))


def _merge_exception_section(op_cases: dict[str, OpCasePlan], section: dict[str, Any]) -> None:
    for op, raw_selector in section.items():
        plan = _ensure_op_plan(op_cases, str(op))
        if raw_selector is True:
            plan.drop = True
            continue
        if isinstance(raw_selector, dict) and raw_selector.get("drop"):
            plan.drop = True
        plan.exclude.merge(_parse_selector(raw_selector, default_all=False))


def _merge_case_file(op_cases: dict[str, OpCasePlan], data: dict[str, Any], backend: str) -> None:
    _merge_model_ops(op_cases, data)
    _merge_case_section(op_cases, _section(data, "all_frameworks_op_cases"))
    framework_cases = _section(data, "framework_specific_op_cases")
    backend_cases = framework_cases.get(backend, {})
    if backend_cases is None:
        return
    if not isinstance(backend_cases, dict):
        raise TypeError(f"framework_specific_op_cases.{backend} must be a mapping")
    _merge_case_section(op_cases, backend_cases)


def _merge_exception_file(op_cases: dict[str, OpCasePlan], data: dict[str, Any], backend: str) -> None:
    _merge_exception_section(op_cases, _section(data, "all_frameworks_op_exceptions"))
    framework_exceptions = _section(data, "framework_specific_op_exceptions")
    backend_exceptions = framework_exceptions.get(backend, {})
    if backend_exceptions is None:
        return
    if not isinstance(backend_exceptions, dict):
        raise TypeError(f"framework_specific_op_exceptions.{backend} must be a mapping")
    _merge_exception_section(op_cases, backend_exceptions)


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


def build_collection_case_plan(
    *,
    backend: str,
    model_path: str | None = None,
    model_architecture: str | None = None,
    gpu_type: str | None = None,
    base_cases_path: str | None = None,
    model_cases_path: str | None = None,
    gpu_exceptions_path: str | None = None,
    full: bool = False,
) -> CollectionCasePlan:
    """Build a model/GPU-aware op and case plan for one backend."""
    base_path = Path(base_cases_path).expanduser().resolve() if base_cases_path else BASE_OP_CASES_PATH
    base_data = load_yaml_file(base_path)
    requested_model_path = model_path
    model_paths = _load_model_case_files(model_path, model_architecture, model_cases_path, full)
    model_data = [load_yaml_file(path) for path in model_paths]
    if model_path is None and len(model_data) == 1:
        model_path = _primary_model_path(model_data[0])
    if model_architecture is None and len(model_data) == 1:
        model_architecture = _model_case_architecture(model_data[0])

    include_base = True
    if len(model_data) == 1:
        include_base = bool(model_data[0].get("include_base", True))

    op_cases: dict[str, OpCasePlan] = {}
    if include_base:
        _merge_case_file(op_cases, base_data, backend)
    for data in model_data:
        _merge_case_file(op_cases, data, backend)

    resolved_gpu_exceptions_path = None
    if gpu_exceptions_path:
        resolved_gpu_exceptions_path = Path(gpu_exceptions_path).expanduser().resolve()
    elif gpu_type:
        default_path = default_gpu_exceptions_path(gpu_type)
        if default_path.exists():
            resolved_gpu_exceptions_path = default_path

    if resolved_gpu_exceptions_path:
        _merge_exception_file(op_cases, load_yaml_file(resolved_gpu_exceptions_path), backend)

    op_cases = {op: plan for op, plan in op_cases.items() if not plan.drop}
    return CollectionCasePlan(
        backend=backend,
        model_path=model_path,
        model_architecture=model_architecture,
        gpu_type=gpu_type,
        op_cases=op_cases,
        base_cases_path=base_path,
        model_cases_paths=model_paths,
        gpu_exceptions_path=resolved_gpu_exceptions_path,
        requested_model_path=requested_model_path,
    )


def _index_matches(index: int, selector: CaseSelector) -> bool:
    return index in selector.indices or any(start <= index <= end for start, end in selector.index_ranges)


def _case_matches(test_case: Any, case_id: str, index: int, selector: CaseSelector) -> bool:
    if selector.all_cases:
        return True
    case_text = str(test_case)
    return (
        case_id in selector.case_ids
        or any(fragment in case_text or fragment in case_id for fragment in selector.contains)
        or _index_matches(index, selector)
    )


def filter_test_cases(
    test_cases: list[Any],
    *,
    plan: OpCasePlan | None,
    full_module_name: str,
    run_func_name: str,
) -> list[Any]:
    """Apply an op case plan to generated collector cases."""
    if plan is None:
        return test_cases
    filtered = []
    include = plan.include
    exclude = plan.exclude

    for index, test_case in enumerate(test_cases):
        case_id = create_test_case_id(test_case, run_func_name, full_module_name)
        if (include.has_specific_selectors() or not include.all_cases) and not _case_matches(
            test_case, case_id, index, include
        ):
            continue
        if exclude.all_cases or _case_matches(test_case, case_id, index, exclude):
            continue
        filtered.append(test_case)

    if include.limit is not None:
        filtered = filtered[: include.limit]
    return filtered
