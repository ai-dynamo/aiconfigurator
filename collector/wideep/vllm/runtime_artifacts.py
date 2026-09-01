# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fail-closed validation and migration for staged vLLM images.

The schema-1 image metadata recorded an immutable platform-child digest but
predated the schema-2 distinction between the configured multi-arch index and
the observed platform child.  This module may upgrade that metadata only when
an independently checksummed schema-2 attestation binds the configured index
to the same child and both artifacts report exactly the same runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

IMAGE_INDEX_DIGEST = "sha256:251eba5cc7c12fed0b75da22a9240e582b1c9e39f6fbc064f86781b963bd814f"
CONFIGURED_IMAGE = f"vllm/vllm-openai:v0.24.0@{IMAGE_INDEX_DIGEST}"
DEEPEP_COMMIT = "73b6ea4a439ba03a695563f9fd242c8e4b02b37c"
DEEPEP_VERSION = f"1.2.1+{DEEPEP_COMMIT[:7]}"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
SYSTEM_ARCHITECTURES = {
    "gb200": "arm64",
    "gb300": "arm64",
    "b200_sxm": "amd64",
    "b300_sxm": "amd64",
    "h100_sxm": "amd64",
    "h200_sxm": "amd64",
}


class RuntimeArtifactError(ValueError):
    """A staged-image artifact cannot be attested."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_existing_path(raw: str | Path, *, label: str) -> Path:
    path = Path(raw).expanduser().resolve(strict=True)
    if path == Path("/mnt/cifs") or Path("/mnt/cifs") in path.parents:
        raise RuntimeArtifactError(f"{label} uses prohibited storage: {path}")
    if path == Path("/mnt/nvdl") or Path("/mnt/nvdl") in path.parents:
        raise RuntimeArtifactError(f"{label} uses prohibited storage: {path}")
    return path


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeArtifactError(f"cannot read {label} {path}: {error}") from error
    if not isinstance(payload, dict):
        raise RuntimeArtifactError(f"{label} must contain a JSON object")
    return payload


def _require_equal(payload: dict[str, Any], expected: dict[str, Any], *, label: str) -> None:
    mismatches = {
        key: {"expected": value, "observed": payload.get(key)}
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatches:
        raise RuntimeArtifactError(f"{label} mismatch: {mismatches}")


def _validate_runtime(runtime: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(runtime, dict):
        raise RuntimeArtifactError(f"{label} runtime is missing")
    _require_equal(
        runtime,
        {
            "vllm": "0.24.0",
            "deep_ep": DEEPEP_VERSION,
            "deep_ep_v2_available": False,
        },
        label=f"{label} runtime",
    )
    for field in ("torch", "cuda", "deep_ep_import"):
        if not isinstance(runtime.get(field), str) or not runtime[field]:
            raise RuntimeArtifactError(f"{label} runtime has invalid {field}")
    return runtime


def migrate_schema1_metadata(
    *,
    image_path: str | Path,
    destination_metadata_path: str | Path,
    source_metadata_path: str | Path,
    source_metadata_sha256: str,
    output_path: str | Path,
    target_system: str,
) -> dict[str, Any]:
    """Create schema-2 metadata for an existing schema-1 image.

    ``source_metadata_sha256`` is mandatory so copying the source attestation
    between clusters cannot silently substitute a different JSON document.
    The output path must not exist; publication remains a separate explicit
    operation by the campaign owner.
    """

    if not _SHA256.fullmatch(source_metadata_sha256):
        raise RuntimeArtifactError("source metadata SHA256 must be 64 lowercase hexadecimal characters")
    if target_system not in SYSTEM_ARCHITECTURES:
        raise RuntimeArtifactError(f"unsupported target system {target_system!r}")
    image = _safe_existing_path(image_path, label="destination image")
    destination_metadata = _safe_existing_path(destination_metadata_path, label="destination metadata")
    source_metadata = _safe_existing_path(source_metadata_path, label="source metadata")
    output = Path(output_path).expanduser().resolve(strict=False)
    if output.exists():
        raise RuntimeArtifactError(f"refusing to overwrite output metadata: {output}")
    if output.parent.resolve(strict=True) != image.parent:
        raise RuntimeArtifactError("output metadata must be written beside the destination image")
    if not image.is_file() or not destination_metadata.is_file() or not source_metadata.is_file():
        raise RuntimeArtifactError("image and metadata inputs must be regular files")

    observed_source_metadata_sha = _sha256(source_metadata)
    if observed_source_metadata_sha != source_metadata_sha256:
        raise RuntimeArtifactError(
            "source metadata checksum mismatch: "
            f"expected {source_metadata_sha256}, observed {observed_source_metadata_sha}"
        )
    destination = _load_json(destination_metadata, label="destination metadata")
    source = _load_json(source_metadata, label="source metadata")

    source_child = source.get("observed_image_digest")
    source_arch = source.get("architecture")
    _require_equal(
        source,
        {
            "schema_version": 2,
            "configured_image": CONFIGURED_IMAGE,
            "configured_image_digest": IMAGE_INDEX_DIGEST,
            "image_variant": f"linux/{source_arch}",
            "deep_ep_source_commit": DEEPEP_COMMIT,
            "image_reference_mode": "enroot-3.4-index-digest",
        },
        label="source schema-2 attestation",
    )
    if source_arch not in ("arm64", "amd64"):
        raise RuntimeArtifactError(f"source metadata has invalid architecture {source_arch!r}")
    if SYSTEM_ARCHITECTURES[target_system] != source_arch:
        raise RuntimeArtifactError(
            f"target system {target_system} requires {SYSTEM_ARCHITECTURES[target_system]}, found {source_arch}"
        )
    for label, payload in (("source", source), ("destination", destination)):
        declared_system = payload.get("system")
        if declared_system not in SYSTEM_ARCHITECTURES:
            raise RuntimeArtifactError(f"{label} metadata has invalid system {declared_system!r}")
        if SYSTEM_ARCHITECTURES[declared_system] != source_arch:
            raise RuntimeArtifactError(f"{label} metadata system/architecture mismatch")
    if not isinstance(source_child, str) or not _DIGEST.fullmatch(source_child):
        raise RuntimeArtifactError(f"source metadata has invalid observed child digest {source_child!r}")
    if not isinstance(source.get("sqsh_sha256"), str) or not _SHA256.fullmatch(source["sqsh_sha256"]):
        raise RuntimeArtifactError("source metadata has invalid squashfs checksum")

    _require_equal(
        destination,
        {
            "schema_version": 1,
            "architecture": source_arch,
            "source_image": f"vllm/vllm-openai:v0.24.0@{source_child}",
            "source_image_digest": source_child,
            "deep_ep_source_commit": DEEPEP_COMMIT,
        },
        label="destination schema-1 attestation",
    )
    destination_runtime = _validate_runtime(destination.get("runtime"), label="destination")
    source_runtime = _validate_runtime(source.get("runtime"), label="source")
    if destination_runtime != source_runtime:
        raise RuntimeArtifactError("source and destination runtime attestations differ")

    destination_sqsh_sha = _sha256(image)
    if destination.get("sqsh_sha256") != destination_sqsh_sha:
        raise RuntimeArtifactError(
            "destination squashfs checksum mismatch: "
            f"metadata={destination.get('sqsh_sha256')!r}, observed={destination_sqsh_sha!r}"
        )
    destination_metadata_sha = _sha256(destination_metadata)
    provenance = {
        "migration_type": "vllm-image-metadata-schema1-to-schema2",
        "source_metadata_sha256": observed_source_metadata_sha,
        "source_schema_version": 2,
        "source_system": source.get("system"),
        "source_sqsh_sha256": source["sqsh_sha256"],
        "source_configured_image_digest": IMAGE_INDEX_DIGEST,
        "source_observed_image_digest": source_child,
        "destination_metadata_sha256": destination_metadata_sha,
        "destination_schema_version": 1,
        "destination_declared_system": destination.get("system"),
        "destination_sqsh_sha256": destination_sqsh_sha,
        "destination_source_image_digest": destination["source_image_digest"],
    }
    migrated = {
        "schema_version": 2,
        "system": target_system,
        "architecture": source_arch,
        "image_variant": f"linux/{source_arch}",
        "configured_image": CONFIGURED_IMAGE,
        "configured_image_digest": IMAGE_INDEX_DIGEST,
        "observed_image_digest": source_child,
        "deep_ep_source_commit": DEEPEP_COMMIT,
        "sqsh_sha256": destination_sqsh_sha,
        "image": str(image),
        "image_reference_mode": "attested-schema1-migration",
        "runtime": destination_runtime,
        "staged_at": destination.get("staged_at"),
        "metadata_migration": provenance,
    }

    output.parent.mkdir(parents=False, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(migrated, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, output)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise
    return migrated


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--destination-metadata", required=True)
    parser.add_argument("--source-metadata", required=True)
    parser.add_argument("--source-metadata-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--target-system", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    migrate_schema1_metadata(
        image_path=args.image,
        destination_metadata_path=args.destination_metadata,
        source_metadata_path=args.source_metadata,
        source_metadata_sha256=args.source_metadata_sha256,
        output_path=args.output,
        target_system=args.target_system,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
