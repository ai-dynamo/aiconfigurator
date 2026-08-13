# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json

import pytest

from tools.finalize_wheels_in_artifactory import FinalizeError, build_manifest, finalize


class FakeArtifacts:
    def __init__(self) -> None:
        self.files = {
            "aiconfigurator-2.0.0-py3-none-any.whl": (b"upper", "a" * 64),
            "aiconfigurator_core-2.0.0-cp311-abi3-manylinux_2_28_x86_64.whl": (b"core", "b" * 64),
        }
        self.uploads: list[tuple[str, bytes]] = []

    def list_files(self, subpath: str) -> list[dict]:
        return [
            {"filename": name, "sha256": checksum, "size": len(payload)}
            for name, (payload, checksum) in self.files.items()
        ]

    def upload(self, path: str, payload: bytes) -> None:
        self.uploads.append((path, payload))


def test_finalize_writes_manifest_before_completion_marker() -> None:
    artifacts = FakeArtifacts()

    manifest = finalize(
        artifacts,
        subpath="post-merge/sha/42/2",
        repository="ai-dynamo/aiconfigurator",
        commit_sha="sha",
        run_id=42,
        run_attempt=2,
    )

    assert [path for path, _ in artifacts.uploads] == [
        "post-merge/sha/42/2/_MANIFEST.json",
        "post-merge/sha/42/2/_COMPLETE.json",
    ]
    marker = json.loads(artifacts.uploads[1][1])
    assert marker["manifest_sha256"] == hashlib.sha256(artifacts.uploads[0][1]).hexdigest()
    assert marker["commit_sha"] == "sha"
    assert manifest["wheels"][0]["filename"] == "aiconfigurator-2.0.0-py3-none-any.whl"


def test_manifest_requires_upper_and_core_wheels() -> None:
    with pytest.raises(FinalizeError, match="upper wheel"):
        build_manifest(
            [{"filename": "aiconfigurator_core-2.0.0-linux.whl", "sha256": "a" * 64, "size": 1}],
            repository="ai-dynamo/aiconfigurator",
            commit_sha="sha",
            run_id=1,
            run_attempt=1,
        )

    with pytest.raises(FinalizeError, match="core wheel"):
        build_manifest(
            [{"filename": "aiconfigurator-2.0.0-py3-none-any.whl", "sha256": "a" * 64, "size": 1}],
            repository="ai-dynamo/aiconfigurator",
            commit_sha="sha",
            run_id=1,
            run_attempt=1,
        )
