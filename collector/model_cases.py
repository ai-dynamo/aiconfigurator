# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model-centric collector case planning.

Collector v2 keeps model intent in YAML and leaves kernel collectors focused on
generating runnable test cases.  The plan answers exactly one question: which
ops does this model (or the full model set) collect on this backend?

    plan ops = base ops activated by the model, unioned with model op sections

Everything below the op level is owned elsewhere:

- case content comes from ``cases/base_ops/*.yaml`` sweep recipes crossed with
  ``model_case_values`` shapes (see ``case_generator``), deduplicated;
- hardware floors are applied by ``collector.capabilities`` (generation-time
  intersection, ``cases/capabilities.yaml``);
- everything else that cannot run is a runtime observation: it fails fast, is
  classified in the failure log, and the circuit breaker stops repetition.

There is intentionally no per-case selector or exception rule engine here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

COLLECTOR_ROOT = Path(__file__).resolve().parent
CASE_ROOT = COLLECTOR_ROOT / "cases"
BASE_OP_CASES_DIR = CASE_ROOT / "base_ops"
MODEL_CASES_DIR = CASE_ROOT / "models"
SYSTEMS_DIR = COLLECTOR_ROOT.parent / "src" / "aiconfigurator" / "systems"


@dataclass(slots=True)
class CollectionCasePlan:
    """Case plan for one backend collection run."""

    backend: str
    model_path: str | None
    model_architecture: str | None
    gpu_type: str | None
    sm_version: int | None
    selected_ops: set[str]
    base_cases_path: Path
    model_cases_paths: list[Path] = field(default_factory=list)
    requested_model_path: str | None = None

    @property
    def ops(self) -> list[str]:
        return sorted(self.selected_ops)

    def has_op(self, op: str) -> bool:
        return op in self.selected_ops

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


def _merge_case_file(
    selected: set[str],
    data: dict[str, Any],
    backend: str,
    *,
    allowed_ops: set[str] | None = None,
) -> None:
    """Add the ops a case file activates for this backend to ``selected``."""
    ops = {str(op) for op in _as_list(data.get("model_ops"), field_name="model_ops")}
    ops.update(str(op) for op in _section(data, "all_frameworks_op_cases"))
    framework_cases = _section(data, "framework_specific_op_cases")
    backend_cases = framework_cases.get(backend, {})
    if backend_cases is not None:
        if not isinstance(backend_cases, dict):
            raise TypeError(f"framework_specific_op_cases.{backend} must be a mapping")
        ops.update(str(op) for op in backend_cases)
    if allowed_ops is not None:
        ops &= allowed_ops
    selected.update(ops)


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

    ``base_ops`` is an explicit allowlist.  Model files that only set
    ``include_base: true`` receive the small universal set declared through
    base-file ``model_ops``; they no longer activate every auxiliary recipe
    merely because another base YAML was added to the repository.
    """

    available_ops: set[str] = set()
    default_ops: set[str] = set()
    for data in base_data_files:
        available_ops.update(_base_case_file_ops(data, backend))
        default_ops.update(str(op) for op in _as_list(data.get("model_ops"), field_name="model_ops"))

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
    full: bool = False,
) -> CollectionCasePlan:
    """Build a model-aware op plan for one backend."""
    base_path = Path(base_cases_path).expanduser().resolve() if base_cases_path else BASE_OP_CASES_DIR
    base_data_files = _load_base_case_files(base_path)
    requested_model_path = model_path
    model_paths = _load_model_case_files(model_path, model_architecture, model_cases_path, full)
    model_data = [load_yaml_file(path) for path in model_paths]
    if model_path is None and len(model_data) == 1:
        model_path = _primary_model_path(model_data[0])
    if model_architecture is None and len(model_data) == 1:
        model_architecture = _model_case_architecture(model_data[0])

    selected: set[str] = set()
    selected_base_ops = _selected_base_ops(base_data_files, model_data, backend, model_path)
    for base_data in base_data_files:
        _merge_case_file(selected, base_data, backend, allowed_ops=selected_base_ops)
    for data in model_data:
        _merge_case_file(selected, data, backend)

    resolved_sm_version = resolve_sm_version(gpu_type=gpu_type, sm_version=sm_version)
    return CollectionCasePlan(
        backend=backend,
        model_path=model_path,
        model_architecture=model_architecture,
        gpu_type=gpu_type,
        sm_version=resolved_sm_version,
        selected_ops=selected,
        base_cases_path=base_path,
        model_cases_paths=model_paths,
        requested_model_path=requested_model_path,
    )
