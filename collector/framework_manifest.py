# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Collector framework version and image manifest helpers."""

from __future__ import annotations

import importlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from packaging.version import InvalidVersion, Version

from collector.op_catalog import CATALOG_PATH, family_for_perf_file, load_family_map
from collector.registry_types import OpEntry

MANIFEST_PATH = Path(__file__).with_name("framework_manifest.yaml")

_DIGEST_RE = re.compile(r"@sha256:[0-9a-f]{64}$")


@dataclass(frozen=True)
class CollectorRuntime:
    framework: str  # manifest key: sglang | trtllm | vllm | wideep_sglang | wideep_trtllm
    version: str
    images: dict[str, str]
    source_repo: str | None = None
    collector_dir: str | None = None
    data_backend: str | None = None
    family: str | None = None  # set when resolved per-op with the catalog present
    workload: str = "default"

    def image(self, variant: str = "default") -> str:
        return self.images.get(variant) or self.images["default"]


def load_manifest(path: str | Path = MANIFEST_PATH) -> dict[str, Any]:
    manifest_path = Path(path)
    with manifest_path.open(encoding="utf-8") as manifest_file:
        manifest = yaml.safe_load(manifest_file) or {}
    if not isinstance(manifest, dict):
        raise TypeError("collector framework manifest must be a mapping")
    validate_manifest(manifest)
    return manifest


def get_collector_runtime(
    framework: str,
    *,
    workload: str = "default",
    path: str | Path = MANIFEST_PATH,
) -> CollectorRuntime:
    manifest = load_manifest(path)
    key = _framework_key(framework, workload)
    spec = manifest["frameworks"].get(key)
    if spec is None:
        raise KeyError(f"No collector runtime is configured for {framework!r} (workload={workload!r})")
    return _runtime_from_spec(key, spec, spec["default"], manifest, family=None)


def _framework_key(framework: str, workload: str) -> str:
    normalized = framework.lower()
    if workload == "wideep" and not normalized.startswith("wideep_"):
        return f"wideep_{normalized}"
    if workload not in ("default", "wideep"):
        raise KeyError(f"Unsupported collector workload {workload!r}")
    return normalized


def _runtime_from_spec(
    framework_key: str,
    spec: dict[str, Any],
    runtime_spec: dict[str, Any],
    manifest: dict[str, Any],
    *,
    family: str | None,
) -> CollectorRuntime:
    base = spec.get("base_framework")
    source_repo = spec.get("source_repo") or (manifest["frameworks"].get(base, {}).get("source_repo") if base else None)
    return CollectorRuntime(
        framework=framework_key,
        version=runtime_spec["version"],
        images=dict(runtime_spec["images"]),
        source_repo=source_repo,
        collector_dir=spec.get("collector_dir"),
        data_backend=spec.get("data_backend"),
        family=family,
        workload="wideep" if framework_key.startswith("wideep_") else "default",
    )


_REGISTRY_MODULES = {
    "sglang": "collector.sglang.registry",
    "trtllm": "collector.trtllm.registry",
    "vllm": "collector.vllm.registry",
    "wideep_sglang": "collector.wideep.sglang.registry",
    "wideep_trtllm": "collector.wideep.trtllm.registry",
}


def _registry_entries(framework_key: str) -> list[OpEntry]:
    module_path = _REGISTRY_MODULES.get(framework_key)
    if module_path is None:
        raise KeyError(f"No collector registry is known for framework {framework_key!r}")
    return list(importlib.import_module(module_path).REGISTRY)


def _resolve_from(
    manifest: dict[str, Any],
    family_map: dict[str, str] | None,
    framework_key: str,
    entry: OpEntry,
) -> CollectorRuntime:
    spec = manifest["frameworks"].get(framework_key)
    if spec is None:
        raise KeyError(f"No collector runtime is configured for {framework_key!r}")
    families = spec.get("families") or {}
    family: str | None = None
    if family_map is not None:
        family = family_for_perf_file(str(entry.perf_filename), family_map)
        if family is None:
            raise LookupError(
                f"{framework_key}:{entry.op} table {entry.perf_filename} has no family in the "
                "op catalog; add it before collecting (fail-closed identity gate, spec §2)"
            )
    elif families:
        raise LookupError(
            f"frameworks.{framework_key}.families overrides require the op catalog "
            f"({CATALOG_PATH.name}), but the op catalog is missing"
        )
    runtime_spec = families.get(family) or spec["default"]
    return _runtime_from_spec(framework_key, spec, runtime_spec, manifest, family=family)


def resolve_op_runtime(
    framework: str,
    op: str,
    *,
    manifest_path: str | Path = MANIFEST_PATH,
    catalog_path: str | Path = CATALOG_PATH,
) -> CollectorRuntime:
    """Resolve the exactly-one pinned runtime for (framework, op) — spec §4."""
    manifest = load_manifest(manifest_path)
    family_map = load_family_map(catalog_path)
    framework_key = framework.lower()
    for entry in _registry_entries(framework_key):
        if entry.op == op:
            return _resolve_from(manifest, family_map, framework_key, entry)
    raise KeyError(f"{framework_key} registry has no op {op!r}")


def _runtime_identity(runtime: CollectorRuntime) -> tuple[str, tuple[tuple[str, str], ...]]:
    """Executor identity: version plus the pinned images. Two runtimes with the
    same package version but different images are different containers. `family`
    stays out — it is routing metadata, not an executor property."""
    return runtime.version, tuple(sorted(runtime.images.items()))


def _describe_runtime(runtime: CollectorRuntime) -> str:
    """Version plus images, for errors where the version alone cannot distinguish."""
    images = ", ".join(f"{variant}={image}" for variant, image in sorted(runtime.images.items()))
    return f"{runtime.version} [{images}]"


def require_collector_runtime(
    framework: str,
    installed_version: str,
    *,
    requested_ops: set[str],
    wideep_ops: set[str] | None = None,
    path: str | Path = MANIFEST_PATH,
    catalog_path: str | Path = CATALOG_PATH,
) -> CollectorRuntime:
    """Resolve the single runtime the requested ops pin, and enforce it exactly.

    Collector V3 semantics: every op resolves independently (family override or
    framework default); one executor container serves exactly one runtime, so
    any spread across versions is an error telling the caller to split the run.
    """
    wideep_ops = wideep_ops or set()
    manifest = load_manifest(path)
    family_map = load_family_map(catalog_path)
    normalized = framework.lower()

    ops_by_key: dict[str, set[str]] = {}
    for op in requested_ops:
        key = f"wideep_{normalized}" if op in wideep_ops else normalized
        ops_by_key.setdefault(key, set()).add(op)
    if not requested_ops:
        ops_by_key[normalized] = set()  # empty selection = the whole stock registry

    resolved: dict[str, CollectorRuntime] = {}
    for key, ops in ops_by_key.items():
        entries = [e for e in _registry_entries(key) if not ops or e.op in ops]
        if ops:
            missing = ops - {e.op for e in entries}
            if missing:
                raise KeyError(f"{key} registry has no op(s): {sorted(missing)}")
        by_identity: dict[tuple[str, tuple[tuple[str, str], ...]], CollectorRuntime] = {}
        op_runtimes: dict[str, CollectorRuntime] = {}
        for entry in entries:
            runtime = _resolve_from(manifest, family_map, key, entry)
            by_identity.setdefault(_runtime_identity(runtime), runtime)
            op_runtimes[entry.op] = runtime
        if len(by_identity) > 1:
            if len({version for version, _ in by_identity}) > 1:
                split = ", ".join(f"{op}→{rt.version}" for op, rt in sorted(op_runtimes.items()))
                raise RuntimeError(
                    f"{framework} ops resolve to multiple runtime versions ({split}); "
                    "run each version group in its own container"
                )
            split = ", ".join(f"{op}→{_describe_runtime(rt)}" for op, rt in sorted(op_runtimes.items()))
            raise RuntimeError(
                f"{framework} ops resolve to the same runtime version but different images ({split}); "
                "run each image group in its own container"
            )
        if not by_identity:
            raise KeyError(f"{key} registry has no ops to resolve")
        resolved[key] = next(iter(by_identity.values()))

    wideep_key = f"wideep_{normalized}"
    if len({_runtime_identity(r) for r in resolved.values()}) > 1:
        stock, wideep = resolved[normalized], resolved[wideep_key]
        if stock.version != wideep.version:
            raise RuntimeError(
                f"Stock {framework} and WideEP ops require different runtime versions "
                f"({stock.version} != {wideep.version}); "
                "run them in separate containers"
            )
        raise RuntimeError(
            f"Stock {framework} and WideEP ops require different images for the same runtime version "
            f"({_describe_runtime(stock)} != {_describe_runtime(wideep)}); "
            "run them in separate containers"
        )
    runtime = resolved.get(wideep_key) or resolved[normalized]

    try:
        installed_public = Version(installed_version).public
    except InvalidVersion as error:
        raise RuntimeError(f"Invalid installed {framework} version {installed_version!r}") from error

    expected_public = Version(runtime.version).public
    if installed_public != expected_public:
        workload = "WideEP" if runtime.workload == "wideep" else "stock"
        raise RuntimeError(
            f"{framework} {workload} collector requires exactly {runtime.version}, found {installed_version}; "
            f"use {runtime.image()}"
        )
    return runtime


def validate_resolution(
    *,
    manifest_path: str | Path = MANIFEST_PATH,
    catalog_path: str | Path = CATALOG_PATH,
) -> list[str]:
    """Return one error per registry op that does not resolve to a pinned runtime."""
    try:
        manifest = load_manifest(manifest_path)
    except (TypeError, ValueError, yaml.YAMLError) as error:
        return [f"manifest: {error}"]
    try:
        family_map = load_family_map(catalog_path)
    except (ValueError, yaml.YAMLError) as error:
        return [f"op catalog: {error}"]

    errors: list[str] = []
    for framework_key in _REGISTRY_MODULES:
        if framework_key not in manifest["frameworks"]:
            errors.append(f"{framework_key}: registry exists but the manifest has no entry")
            continue
        try:
            entries = _registry_entries(framework_key)
        except ImportError as error:
            errors.append(f"{framework_key}: registry import failed: {error}")
            continue
        for entry in entries:
            try:
                _resolve_from(manifest, family_map, framework_key, entry)
            except (KeyError, LookupError, ValueError) as error:
                errors.append(f"{framework_key}:{entry.op}: {error}")
    return errors


def validate_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("schema_version") != 2:
        raise ValueError("collector framework manifest schema_version must be 2")

    frameworks = manifest.get("frameworks")
    if not isinstance(frameworks, dict) or not frameworks:
        raise ValueError("collector framework manifest must define frameworks")
    for framework, spec in frameworks.items():
        _validate_framework_spec(framework, spec, frameworks)


def _validate_framework_spec(name: str, spec: object, frameworks: dict[str, Any]) -> None:
    if not isinstance(spec, dict):
        raise TypeError(f"frameworks.{name} must be a mapping")
    if name.startswith("wideep_"):
        base = name.removeprefix("wideep_")
        if spec.get("base_framework") != base:
            raise ValueError(f"frameworks.{name}.base_framework must be {base!r}")
        if base not in frameworks:
            raise ValueError(f"frameworks.{name}.base_framework {base!r} has no manifest entry")
        if not spec.get("collector_dir"):
            raise ValueError(f"frameworks.{name}.collector_dir is required")
        if not spec.get("data_backend"):
            raise ValueError(f"frameworks.{name}.data_backend is required")
    _validate_runtime_spec(f"frameworks.{name}.default", spec.get("default"))
    families = spec.get("families") or {}
    if not isinstance(families, dict):
        raise TypeError(f"frameworks.{name}.families must be a mapping")
    for family, override in families.items():
        _validate_runtime_spec(f"frameworks.{name}.families.{family}", override)


def _validate_runtime_spec(name: str, spec: object) -> None:
    if not isinstance(spec, dict):
        raise TypeError(f"{name} must be a mapping")
    if not isinstance(spec.get("version"), str) or not spec["version"]:
        raise ValueError(f"{name}.version is required")
    images = spec.get("images")
    if not isinstance(images, dict) or not images.get("default"):
        raise ValueError(f"{name}.images.default is required")
    if not all(isinstance(key, str) and isinstance(value, str) and value for key, value in images.items()):
        raise ValueError(f"{name}.images must map image variants to non-empty strings")
    for variant, image in images.items():
        if "/" in image and not _DIGEST_RE.search(image):
            raise ValueError(
                f"{name}.images.{variant} must be digest-pinned (...@sha256:<64 hex>); "
                "bare internal image names without '/' are exempt"
            )
