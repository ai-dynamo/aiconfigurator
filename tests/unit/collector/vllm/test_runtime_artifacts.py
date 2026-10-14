# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from collector.wideep.vllm import runtime_artifacts

pytestmark = pytest.mark.unit


CHILD = "sha256:" + "3" * 64
RUNTIME = {
    "cuda": "13.0",
    "deep_ep": runtime_artifacts.DEEPEP_VERSION,
    "deep_ep_import": "/usr/local/lib/python3.12/dist-packages/deep_ep/__init__.py",
    "deep_ep_v2_available": False,
    "torch": "2.11.0+cu130",
    "vllm": "0.24.0",
}


def _write_inputs(tmp_path: Path) -> tuple[Path, Path, Path, str]:
    image = tmp_path / "runtime.sqsh"
    image.write_bytes(b"attested destination squashfs bytes")
    image_sha = hashlib.sha256(image.read_bytes()).hexdigest()
    destination = tmp_path / "schema1.json"
    destination.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "system": "gb200",
                "architecture": "arm64",
                "source_image": f"vllm/vllm-openai:v0.24.0@{CHILD}",
                "source_image_digest": CHILD,
                "deep_ep_source_commit": runtime_artifacts.DEEPEP_COMMIT,
                "sqsh_sha256": image_sha,
                "image": "/old/location/runtime.sqsh",
                "runtime": RUNTIME,
                "staged_at": "2026-08-25",
            }
        ),
        encoding="utf-8",
    )
    source = tmp_path / "schema2.json"
    source.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "system": "gb300",
                "architecture": "arm64",
                "image_variant": "linux/arm64",
                "configured_image": runtime_artifacts.CONFIGURED_IMAGE,
                "configured_image_digest": runtime_artifacts.IMAGE_INDEX_DIGEST,
                "observed_image_digest": CHILD,
                "deep_ep_source_commit": runtime_artifacts.DEEPEP_COMMIT,
                "sqsh_sha256": "6" * 64,
                "image": "/remote/location/runtime.sqsh",
                "image_reference_mode": "enroot-3.4-index-digest",
                "runtime": RUNTIME,
                "staged_at": "2026-08-26",
            }
        ),
        encoding="utf-8",
    )
    return image, destination, source, hashlib.sha256(source.read_bytes()).hexdigest()


def _migrate(tmp_path: Path, **overrides):
    image, destination, source, source_sha = _write_inputs(tmp_path)
    kwargs = {
        "image_path": image,
        "destination_metadata_path": destination,
        "source_metadata_path": source,
        "source_metadata_sha256": source_sha,
        "output_path": tmp_path / "runtime.sqsh.meta.json",
        "target_system": "gb300",
    }
    kwargs.update(overrides)
    return runtime_artifacts.migrate_schema1_metadata(**kwargs), kwargs


def test_schema1_migration_binds_index_child_runtime_and_both_artifacts(tmp_path):
    migrated, kwargs = _migrate(tmp_path)
    image = Path(kwargs["image_path"])
    destination = Path(kwargs["destination_metadata_path"])
    source = Path(kwargs["source_metadata_path"])

    assert migrated["configured_image_digest"] == runtime_artifacts.IMAGE_INDEX_DIGEST
    assert migrated["observed_image_digest"] == CHILD
    assert migrated["sqsh_sha256"] == hashlib.sha256(image.read_bytes()).hexdigest()
    assert migrated["runtime"] == RUNTIME
    assert migrated["image_reference_mode"] == "attested-schema1-migration"
    provenance = migrated["metadata_migration"]
    assert provenance["source_metadata_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert provenance["destination_metadata_sha256"] == hashlib.sha256(destination.read_bytes()).hexdigest()
    assert provenance["source_sqsh_sha256"] == "6" * 64
    assert provenance["destination_sqsh_sha256"] == migrated["sqsh_sha256"]
    assert json.loads(Path(kwargs["output_path"]).read_text()) == migrated


def test_schema1_migration_requires_out_of_band_source_metadata_checksum(tmp_path):
    with pytest.raises(runtime_artifacts.RuntimeArtifactError, match="source metadata checksum mismatch"):
        _migrate(tmp_path, source_metadata_sha256="0" * 64)


@pytest.mark.parametrize("changed", ["child", "runtime", "index", "sqsh"])
def test_schema1_migration_rejects_broken_attestation_chain(tmp_path, changed):
    image, destination_path, source_path, _ = _write_inputs(tmp_path)
    destination = json.loads(destination_path.read_text())
    source = json.loads(source_path.read_text())
    if changed == "child":
        destination["source_image_digest"] = "sha256:" + "4" * 64
    elif changed == "runtime":
        destination["runtime"]["torch"] = "different"
    elif changed == "index":
        source["configured_image_digest"] = "sha256:" + "5" * 64
    else:
        image.write_bytes(b"changed after schema-1 attestation")
    destination_path.write_text(json.dumps(destination), encoding="utf-8")
    source_path.write_text(json.dumps(source), encoding="utf-8")
    source_sha = hashlib.sha256(source_path.read_bytes()).hexdigest()

    with pytest.raises(runtime_artifacts.RuntimeArtifactError):
        runtime_artifacts.migrate_schema1_metadata(
            image_path=image,
            destination_metadata_path=destination_path,
            source_metadata_path=source_path,
            source_metadata_sha256=source_sha,
            output_path=tmp_path / "runtime.sqsh.meta.json",
            target_system="gb300",
        )


def test_schema1_migration_rejects_wrong_target_architecture_and_overwrite(tmp_path):
    image, destination, source, source_sha = _write_inputs(tmp_path)
    output = tmp_path / "runtime.sqsh.meta.json"
    output.write_text("existing", encoding="utf-8")
    with pytest.raises(runtime_artifacts.RuntimeArtifactError, match="refusing to overwrite"):
        runtime_artifacts.migrate_schema1_metadata(
            image_path=image,
            destination_metadata_path=destination,
            source_metadata_path=source,
            source_metadata_sha256=source_sha,
            output_path=output,
            target_system="gb300",
        )
    output.unlink()
    with pytest.raises(runtime_artifacts.RuntimeArtifactError, match="requires amd64"):
        runtime_artifacts.migrate_schema1_metadata(
            image_path=image,
            destination_metadata_path=destination,
            source_metadata_path=source,
            source_metadata_sha256=source_sha,
            output_path=output,
            target_system="h200_sxm",
        )
