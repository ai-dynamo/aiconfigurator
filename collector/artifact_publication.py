# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed publication of a collector-owned artifact set."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from collections.abc import Iterable
from pathlib import Path

PUBLICATION_MANIFEST = ".aic_moe_a2a_publication.json"


class ArtifactPublicationError(RuntimeError):
    """The visible artifact set is not one completely committed generation."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def validate_published_artifact_set(destination: str | Path) -> dict[str, str]:
    """Return committed checksums, or fail if a mixed/incomplete set is visible."""
    root = Path(destination)
    manifest_path = root / PUBLICATION_MANIFEST
    if not manifest_path.is_file():
        raise ArtifactPublicationError(f"{root}: missing publication manifest")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ArtifactPublicationError(f"{manifest_path}: invalid publication manifest") from error
    if payload.get("schema_version") != 1 or payload.get("status") != "complete":
        raise ArtifactPublicationError(f"{manifest_path}: artifact publication is not complete")
    artifacts = payload.get("artifacts")
    owned_patterns = payload.get("owned_patterns")
    if not isinstance(artifacts, dict) or not isinstance(owned_patterns, list):
        raise ArtifactPublicationError(f"{manifest_path}: malformed publication manifest")
    observed_owned = {
        path.name
        for pattern in owned_patterns
        for path in root.glob(pattern)
        if path.is_file() and path.name != PUBLICATION_MANIFEST
    }
    if observed_owned != set(artifacts):
        raise ArtifactPublicationError(
            f"{root}: committed artifact names differ; expected {sorted(artifacts)}, found {sorted(observed_owned)}"
        )
    for name, expected in artifacts.items():
        artifact = root / name
        if not artifact.is_file() or _sha256(artifact) != expected:
            raise ArtifactPublicationError(f"{artifact}: committed artifact checksum mismatch")
    return dict(artifacts)


def publish_artifact_set(
    *,
    staging: Path,
    destination: Path,
    artifact_names: Iterable[str],
    owned_patterns: Iterable[str],
    checksum_output: Path | None = None,
) -> dict[str, str]:
    """Publish staged artifacts and atomically install their commit record last."""
    names = tuple(artifact_names)
    patterns = tuple(owned_patterns)
    if len(names) != len(set(names)) or PUBLICATION_MANIFEST in names or any(Path(name).name != name for name in names):
        raise ArtifactPublicationError("publication artifact names must be unique and non-control")
    destination.mkdir(parents=True, exist_ok=True)
    generation = uuid.uuid4().hex
    sources = {name: staging / name for name in names}
    missing = [name for name, source in sources.items() if not source.is_file()]
    if missing:
        raise ArtifactPublicationError(f"staging is missing artifacts: {missing}")
    checksums = {name: _sha256(source) for name, source in sources.items()}
    manifest_path = destination / PUBLICATION_MANIFEST
    temporary_paths: list[Path] = []

    def prepare(target: Path, source: Path) -> Path:
        temporary = target.parent / f".{target.name}.tmp.{os.getpid()}.{generation}"
        shutil.copyfile(source, temporary)
        temporary_paths.append(temporary)
        return temporary

    try:
        publishing = staging / f"{PUBLICATION_MANIFEST}.publishing"
        _write_json(publishing, {"schema_version": 1, "status": "publishing", "generation": generation})
        os.replace(prepare(manifest_path, publishing), manifest_path)

        prepared = [(prepare(destination / name, source), destination / name) for name, source in sources.items()]
        prepared_checksum: tuple[Path, Path] | None = None
        if checksum_output is not None:
            checksum_output.parent.mkdir(parents=True, exist_ok=True)
            staged_checksum = staging / f"artifact_checksums.{generation}.json"
            _write_json(staged_checksum, checksums)
            prepared_checksum = (prepare(checksum_output, staged_checksum), checksum_output)

        for temporary, target in prepared:
            os.replace(temporary, target)
        for pattern in patterns:
            for stale in destination.glob(pattern):
                if stale.is_file() and stale.name not in checksums and stale.name != PUBLICATION_MANIFEST:
                    stale.unlink()
        if prepared_checksum is not None:
            os.replace(*prepared_checksum)

        complete = staging / f"{PUBLICATION_MANIFEST}.complete"
        _write_json(
            complete,
            {
                "schema_version": 1,
                "status": "complete",
                "generation": generation,
                "owned_patterns": list(patterns),
                "artifacts": checksums,
            },
        )
        os.replace(prepare(manifest_path, complete), manifest_path)
    finally:
        for temporary in temporary_paths:
            temporary.unlink(missing_ok=True)

    validate_published_artifact_set(destination)
    return checksums
