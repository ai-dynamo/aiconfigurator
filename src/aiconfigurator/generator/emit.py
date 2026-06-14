# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Artifact emission + artifact_manifest.yaml (design v2 §4)."""
from __future__ import annotations

import pathlib
from typing import Any

import yaml

from .pipeline import PipelineResult

_SCHEMA_VERSION = "v2"


def build_artifact_manifest(result: PipelineResult, artifact_names: list[str]) -> str:
    ir = result.ir
    manifest: dict[str, Any] = {
        "schema_version": _SCHEMA_VERSION,
        "backend": {"name": ir.backend, "version": ir.backend_version},
        "topology": {"mode": ir.mode, "roles": [c.role for c in ir.components]},
        "artifacts": [{"path": name} for name in sorted(artifact_names)],
    }
    return yaml.safe_dump(manifest, sort_keys=False)


def emit(result: PipelineResult, output_dir: str) -> None:
    out = pathlib.Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, content in result.artifacts.items():
        target = out / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    manifest = build_artifact_manifest(result, list(result.artifacts))
    (out / "artifact_manifest.yaml").write_text(manifest, encoding="utf-8")
