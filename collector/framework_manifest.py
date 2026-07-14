# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Collector framework version and image manifest helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from packaging.version import InvalidVersion, Version

MANIFEST_PATH = Path(__file__).with_name("framework_manifest.yaml")


@dataclass(frozen=True)
class CollectorRuntime:
    framework: str
    version: str
    images: dict[str, str]
    source_repo: str | None = None
    collector_dir: str | None = None
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
    normalized = framework.lower()
    if workload == "wideep":
        section = manifest.get("wideep", {})
    elif workload == "default":
        section = manifest.get("frameworks", {})
    else:
        raise KeyError(f"Unsupported collector workload {workload!r}")

    spec = section.get(normalized)
    if spec is None:
        raise KeyError(f"No {workload} collector runtime is configured for {framework!r}")
    return CollectorRuntime(
        framework=normalized,
        version=spec["version"],
        images=dict(spec["images"]),
        source_repo=spec.get("source_repo") or manifest["frameworks"].get(normalized, {}).get("source_repo"),
        collector_dir=spec.get("collector_dir"),
        workload=workload,
    )


def require_collector_runtime(
    framework: str,
    installed_version: str,
    *,
    requested_ops: set[str],
    wideep_ops: set[str] | None = None,
    path: str | Path = MANIFEST_PATH,
) -> CollectorRuntime:
    """Select the requested workload runtime and enforce its exact public version."""
    wideep_ops = wideep_ops or set()
    requested_wideep = requested_ops & wideep_ops
    requested_stock = not requested_ops or bool(requested_ops - wideep_ops)
    runtime = get_collector_runtime(framework, path=path)

    if requested_wideep:
        wideep_runtime = get_collector_runtime(framework, workload="wideep", path=path)
        if requested_stock and runtime.version != wideep_runtime.version:
            raise RuntimeError(
                f"Stock {framework} and WideEP ops require different runtime versions "
                f"({runtime.version} != {wideep_runtime.version}); run them in separate containers"
            )
        runtime = wideep_runtime

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


def validate_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("schema_version") != 1:
        raise ValueError("collector framework manifest schema_version must be 1")

    frameworks = manifest.get("frameworks")
    if not isinstance(frameworks, dict) or not frameworks:
        raise ValueError("collector framework manifest must define frameworks")
    for framework, spec in frameworks.items():
        _validate_runtime_spec(f"frameworks.{framework}", spec)

    wideep = manifest.get("wideep", {})
    if not isinstance(wideep, dict):
        raise TypeError("collector framework manifest wideep section must be a mapping")
    for framework, spec in wideep.items():
        if framework not in frameworks:
            raise ValueError(f"wideep.{framework} does not have a matching framework entry")
        _validate_runtime_spec(f"wideep.{framework}", spec)
        if not spec.get("collector_dir"):
            raise ValueError(f"wideep.{framework}.collector_dir is required")


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
