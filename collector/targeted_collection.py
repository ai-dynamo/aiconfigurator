# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Targeted collector selection for support-matrix healing.

The collector historically had two practical modes: sample a few generated
cases with ``--smoke`` or run the full generated case list.  Support-matrix
healing often needs a narrower third path: run the few generated cases that
represent missing rows for one GPU/model, while keeping GPU-specific skips
separate from the desired data-point set.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_MISSING = object()
_SUPPORTED_MODES = {"targeted", "full"}


@dataclass(frozen=True, slots=True)
class TargetContext:
    """Runtime scope used when applying a target collection spec."""

    backend: str
    op: str
    target_gpu: str | None = None
    target_models: tuple[str, ...] = ()
    mode: str = "targeted"

    def __post_init__(self):
        if self.mode not in _SUPPORTED_MODES:
            raise ValueError(f"Unsupported target mode {self.mode!r}; expected one of {sorted(_SUPPORTED_MODES)}")


@dataclass(frozen=True, slots=True)
class TargetSelector:
    """One data-point or GPU-exception selector from a target spec."""

    name: str = ""
    backend: str | None = None
    op: str | None = None
    models: tuple[str, ...] = ()
    gpus: tuple[str, ...] = ()
    match: Mapping[str, Any] = field(default_factory=dict)
    contains: tuple[Any, ...] = ()
    case_ids: tuple[str, ...] = ()
    id_contains: tuple[str, ...] = ()
    reason: str = ""

    def matches_scope(self, context: TargetContext, *, require_target_gpu: bool = False) -> bool:
        if self.backend and self.backend != context.backend:
            return False
        if self.op and self.op != context.op:
            return False
        if self.gpus:
            if context.target_gpu is None:
                return not require_target_gpu
            if context.target_gpu not in self.gpus:
                return False
        return not (context.target_models and self.models and not set(context.target_models).intersection(self.models))

    def matches_case(self, case: Any, task_id: str, context: TargetContext) -> bool:
        if self.case_ids and task_id not in self.case_ids:
            return False
        if self.id_contains and not all(needle in task_id for needle in self.id_contains):
            return False
        if not _matches_models(case, self.models or context.target_models):
            return False
        for key, expected in self.match.items():
            actual = _lookup(case, str(key))
            if actual is _MISSING or not _value_matches(actual, expected):
                return False
        flattened = tuple(_flatten(case))
        for needle in self.contains:
            if not any(_value_matches(value, needle) or str(needle) in str(value) for value in flattened):
                return False
        return True


@dataclass(frozen=True, slots=True)
class TargetFilterResult:
    cases: list[Any]
    original_count: int
    selected_count: int
    skipped_by_exception: int


@dataclass(frozen=True, slots=True)
class TargetCollectionSpec:
    """Parsed target collection spec.

    ``data_points`` is the allowlist for desired collection work. If it is
    empty, all generated cases remain eligible and only ``gpu_exceptions`` are
    applied.  This lets a GPU exception file be used without forcing a
    allowlist-style targeted run.
    """

    data_points: tuple[TargetSelector, ...] = ()
    gpu_exceptions: tuple[TargetSelector, ...] = ()

    def collection_selected(self, context: TargetContext) -> bool:
        if not self.data_points:
            return True
        return any(selector.matches_scope(context) for selector in self.data_points)

    def filter_cases(
        self,
        cases: Sequence[Any],
        context: TargetContext,
        task_id_factory: Callable[[Any], str],
    ) -> TargetFilterResult:
        original = list(cases)
        data_points = [selector for selector in self.data_points if selector.matches_scope(context)]
        exceptions = [
            selector
            for selector in self.gpu_exceptions
            if selector.matches_scope(context, require_target_gpu=bool(selector.gpus))
        ]

        if data_points:
            selected = [
                case
                for case in original
                if any(selector.matches_case(case, task_id_factory(case), context) for selector in data_points)
            ]
        elif self.data_points:
            selected = []
        else:
            selected = original

        filtered = []
        skipped_by_exception = 0
        for case in selected:
            task_id = task_id_factory(case)
            if any(selector.matches_case(case, task_id, context) for selector in exceptions):
                skipped_by_exception += 1
                continue
            filtered.append(case)

        return TargetFilterResult(
            cases=filtered,
            original_count=len(original),
            selected_count=len(selected),
            skipped_by_exception=skipped_by_exception,
        )


def load_target_collection_spec(path: str | Path) -> TargetCollectionSpec:
    """Load a target collection spec from JSON or YAML."""
    spec_path = Path(path).expanduser()
    if not spec_path.exists():
        raise ValueError(f"Target collection spec does not exist: {spec_path}")

    if spec_path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as e:
            raise ValueError("YAML target specs require PyYAML; use JSON or install pyyaml") from e
        with open(spec_path) as f:
            raw = yaml.safe_load(f)
    else:
        with open(spec_path) as f:
            raw = json.load(f)

    if not isinstance(raw, Mapping):
        raise TypeError("Target collection spec must be a mapping with data_points and/or gpu_exceptions")

    data_points = tuple(
        _selector_from_raw(item, section="data_points", index=i)
        for i, item in enumerate(_as_list(raw.get("data_points", [])))
    )
    gpu_exceptions = tuple(
        _selector_from_raw(item, section="gpu_exceptions", index=i)
        for i, item in enumerate(_as_list(raw.get("gpu_exceptions", raw.get("exceptions", []))))
    )
    return TargetCollectionSpec(data_points=data_points, gpu_exceptions=gpu_exceptions)


def _selector_from_raw(raw: Any, *, section: str, index: int) -> TargetSelector:
    if not isinstance(raw, Mapping):
        raise TypeError(f"{section}[{index}] must be a mapping")

    backend = _optional_str(raw.get("backend", raw.get("framework")))
    op = _optional_str(raw.get("op", raw.get("operation")))
    match = raw.get("match", raw.get("case_match", {}))
    if not isinstance(match, Mapping):
        raise TypeError(f"{section}[{index}].match must be a mapping")

    return TargetSelector(
        name=str(raw.get("name", "")),
        backend=backend,
        op=op,
        models=tuple(str(item) for item in _as_list(raw.get("models", raw.get("model", [])))),
        gpus=tuple(str(item) for item in _as_list(raw.get("gpus", raw.get("gpu", [])))),
        match=dict(match),
        contains=tuple(_as_list(raw.get("contains", raw.get("case_contains", [])))),
        case_ids=tuple(str(item) for item in _as_list(raw.get("case_ids", raw.get("task_ids", [])))),
        id_contains=tuple(str(item) for item in _as_list(raw.get("id_contains", []))),
        reason=str(raw.get("reason", "")),
    )


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _lookup(value: Any, path: str) -> Any:
    current = value
    for part in path.split("."):
        current = _lookup_one(current, part)
        if current is _MISSING:
            return _MISSING
    return current


def _lookup_one(value: Any, key: str) -> Any:
    if dataclasses.is_dataclass(value):
        return getattr(value, key, _MISSING)
    if isinstance(value, Mapping):
        if key in value:
            return value[key]
        if key.isdigit() and int(key) in value:
            return value[int(key)]
        return _MISSING
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if not key.isdigit():
            return _MISSING
        index = int(key)
        if index >= len(value):
            return _MISSING
        return value[index]
    return getattr(value, key, _MISSING)


def _flatten(value: Any):
    if dataclasses.is_dataclass(value):
        yield from _flatten(dataclasses.asdict(value))
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _flatten(item)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            yield from _flatten(item)
    else:
        yield value


def _value_matches(actual: Any, expected: Any) -> bool:
    if isinstance(expected, Sequence) and not isinstance(expected, (str, bytes, bytearray)):
        if isinstance(actual, Sequence) and not isinstance(actual, (str, bytes, bytearray)):
            return list(actual) == list(expected)
        return any(_value_matches(actual, item) for item in expected)
    return actual == expected or str(actual) == str(expected)


def _matches_models(case: Any, models: tuple[str, ...]) -> bool:
    if not models:
        return True

    values = {str(value) for value in _flatten(case)}
    if values.intersection(models):
        return True

    known_models = _known_model_names()
    if values.intersection(known_models):
        return False

    # Some generated cases, such as GEMM, represent model-derived dimensions
    # but do not carry the model name. Let explicit case predicates decide.
    return not (any("/" in value for value in values) and any("/" in model for model in models))


def _known_model_names() -> set[str]:
    try:
        from collector.common_test_cases import get_all_model_names
    except Exception:
        return set()
    return set(get_all_model_names())
