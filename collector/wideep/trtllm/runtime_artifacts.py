# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed validation for staged TRT-LLM campaign runtime artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

IMAGE_INDEX_DIGEST = "sha256:1532b38814b3faf2affdb5ef01ca91468685d314ffb7e8926a0567595355ed88"
IMAGE_CHILD_DIGESTS = {
    "arm64": "sha256:2202825c5950b4925e1add7d458228c9ad3368671789856f24d8947b4defd21c",
    "amd64": "sha256:9b3b4dfb811caa9420fa99a6f958155f6a1f727ffc2b5a5c2d9d2ce51fdc323d",
}
SYSTEM_RUNTIME = {
    "gb200": ("arm64", "100a-real"),
    "gb300": ("arm64", "103a-real"),
    "b200_sxm": ("amd64", "100a-real"),
    "b300_sxm": ("amd64", "103a-real"),
    "h100_sxm": ("amd64", "90-real"),
    "h200_sxm": ("amd64", "90-real"),
}
SOURCE_COMMIT = "14efb6ac673c0cbe828e1206cc5c7d5748d05ffa"
DEEPEP_COMMIT = "5be51b228a7c82dbdb213ea58e77bffd12b38af8"
NVSHMEM_VERSION = "3.2.5-1"
NVSHMEM_ARCHIVE_SHA256 = "eb2c8fb3b7084c2db86bd9fd905387909f1dfd483e7b45f7b3c3d5fcf5374b5a"
PYTHON_REQUIREMENTS = ["transformers==4.57.3"]
_SHA256 = re.compile(r"[0-9a-f]{64}")


class RuntimeArtifactError(ValueError):
    """A seed runtime artifact is incomplete or has the wrong identity."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeArtifactError(f"invalid runtime metadata {path}") from error
    if not isinstance(payload, dict):
        raise RuntimeArtifactError(f"runtime metadata is not an object: {path}")
    return payload


def _expect(meta: dict[str, Any], expected: dict[str, Any], *, kind: str) -> None:
    for key, value in expected.items():
        if meta.get(key) != value:
            raise RuntimeArtifactError(f"{kind} {key} mismatch")


def validate_image(image: Path, meta_path: Path, *, target_system: str) -> tuple[dict[str, Any], dict[str, Any]]:
    if target_system not in SYSTEM_RUNTIME:
        raise RuntimeArtifactError(f"unsupported target system {target_system}")
    arch, _ = SYSTEM_RUNTIME[target_system]
    image = image.resolve(strict=True)
    meta_path = meta_path.resolve(strict=True)
    meta = _load(meta_path)
    source_system = str(meta.get("system", ""))
    if source_system not in SYSTEM_RUNTIME or SYSTEM_RUNTIME[source_system][0] != arch:
        raise RuntimeArtifactError("seed image CPU architecture mismatch")
    _expect(
        meta,
        {
            "schema_version": 1,
            "architecture": arch,
            "image_variant": f"linux/{arch}",
            "configured_image": (f"nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc20@{IMAGE_INDEX_DIGEST}"),
            "configured_image_digest": IMAGE_INDEX_DIGEST,
            "observed_image_digest": IMAGE_CHILD_DIGESTS[arch],
        },
        kind="seed image",
    )
    recorded_sha = str(meta.get("sqsh_sha256", ""))
    actual_sha = _sha256(image)
    if not _SHA256.fullmatch(recorded_sha) or actual_sha != recorded_sha:
        raise RuntimeArtifactError("seed sqsh checksum mismatch")
    provenance = {
        "mode": "image",
        "source_system": source_system,
        "source_image_sha256": actual_sha,
        "source_image_meta_sha256": _sha256(meta_path),
        "source_image_digest": IMAGE_CHILD_DIGESTS[arch],
    }
    return meta, provenance


def validate_wheel(
    wheel_dir: Path,
    *,
    target_system: str,
    image_meta: dict[str, Any],
    provenance: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    arch, cuda_arches = SYSTEM_RUNTIME[target_system]
    wheel_dir = wheel_dir.resolve(strict=True)
    meta_path = wheel_dir / "build_meta.json"
    if not (wheel_dir / "SUCCESS").is_file():
        raise RuntimeArtifactError("seed wheel SUCCESS marker missing")
    meta = _load(meta_path)
    _expect(
        meta,
        {
            "schema_version": 1,
            "system": image_meta.get("system"),
            "architecture": arch,
            "configured_image_digest": IMAGE_INDEX_DIGEST,
            "observed_image_digest": IMAGE_CHILD_DIGESTS[arch],
            "sqsh_sha256": image_meta.get("sqsh_sha256"),
            "trtllm_version": "1.3.0rc11",
            "source_commit": SOURCE_COMMIT,
            "deep_ep": DEEPEP_COMMIT,
            "nvshmem": NVSHMEM_VERSION,
            "nvshmem_archive_sha256": NVSHMEM_ARCHIVE_SHA256,
            "cuda_arches": cuda_arches,
            "python_requirements": PYTHON_REQUIREMENTS,
        },
        kind="seed wheel",
    )
    wheel_name = str(meta.get("wheel", ""))
    wheel = (wheel_dir / wheel_name).resolve(strict=True)
    if wheel.parent != wheel_dir or not _SHA256.fullmatch(str(meta.get("wheel_sha256", ""))):
        raise RuntimeArtifactError("invalid seed wheel identity")
    if _sha256(wheel) != meta["wheel_sha256"]:
        raise RuntimeArtifactError("seed wheel checksum mismatch")
    dependency_dir = (wheel_dir / "dependencies").resolve(strict=True)
    expected_dependencies = meta.get("dependency_wheels")
    if not isinstance(expected_dependencies, dict) or not expected_dependencies:
        raise RuntimeArtifactError("seed dependency manifest missing")
    actual_names = {path.name for path in dependency_dir.glob("*.whl")}
    if actual_names != set(expected_dependencies):
        raise RuntimeArtifactError("seed dependency wheel set mismatch")
    for name, digest in expected_dependencies.items():
        if Path(name).name != name or not _SHA256.fullmatch(str(digest)):
            raise RuntimeArtifactError("invalid seed dependency identity")
        if _sha256(dependency_dir / name) != digest:
            raise RuntimeArtifactError(f"seed dependency checksum mismatch: {name}")
    provenance = dict(provenance) | {
        "mode": "runtime",
        "source_wheel_sha256": meta["wheel_sha256"],
        "source_wheel_meta_sha256": _sha256(meta_path),
        "cuda_arches": cuda_arches,
    }
    return meta, provenance


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--image-meta", type=Path, required=True)
    parser.add_argument("--target-system", required=True, choices=sorted(SYSTEM_RUNTIME))
    parser.add_argument("--wheel-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    image_meta, provenance = validate_image(args.image, args.image_meta, target_system=args.target_system)
    if args.wheel_dir is not None:
        _, provenance = validate_wheel(
            args.wheel_dir,
            target_system=args.target_system,
            image_meta=image_meta,
            provenance=provenance,
        )
    args.output.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
