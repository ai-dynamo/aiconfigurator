#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Publish a checksummed wheel manifest, then mark an Artifactory set complete."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

MANIFEST_SCHEMA = "aic-wheel-manifest/1.0.0"
UPPER_RE = re.compile(r"^aiconfigurator-[^-].*\.whl$")
CORE_RE = re.compile(r"^aiconfigurator_core-.*\.whl$")


class FinalizeError(RuntimeError):
    pass


class Artifacts(Protocol):
    def list_files(self, subpath: str) -> list[dict[str, Any]]: ...

    def upload(self, path: str, payload: bytes) -> None: ...


def build_manifest(
    files: Sequence[Mapping[str, Any]],
    *,
    repository: str,
    commit_sha: str,
    run_id: int,
    run_attempt: int,
) -> dict[str, Any]:
    wheels: list[dict[str, Any]] = []
    for item in files:
        filename = item.get("filename")
        checksum = item.get("sha256")
        size = item.get("size")
        if not isinstance(filename, str) or not filename.endswith(".whl"):
            continue
        if not isinstance(checksum, str) or not re.fullmatch(r"[0-9a-f]{64}", checksum):
            raise FinalizeError(f"wheel {filename} has no valid SHA-256")
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise FinalizeError(f"wheel {filename} has no positive size")
        wheels.append({"filename": filename, "sha256": checksum, "size": size})
    wheels.sort(key=lambda item: item["filename"])
    if len([item for item in wheels if UPPER_RE.fullmatch(item["filename"])]) != 1:
        raise FinalizeError("completed platform build must contain exactly one upper wheel")
    if not any(CORE_RE.fullmatch(item["filename"]) for item in wheels):
        raise FinalizeError("completed platform build must contain at least one core wheel")
    return {
        "schemaVersion": MANIFEST_SCHEMA,
        "repository": repository,
        "commitSha": commit_sha,
        "workflowRunId": run_id,
        "workflowRunAttempt": run_attempt,
        "wheels": wheels,
    }


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def finalize(
    artifacts: Artifacts,
    *,
    subpath: str,
    repository: str,
    commit_sha: str,
    run_id: int,
    run_attempt: int,
) -> dict[str, Any]:
    manifest = build_manifest(
        artifacts.list_files(subpath),
        repository=repository,
        commit_sha=commit_sha,
        run_id=run_id,
        run_attempt=run_attempt,
    )
    manifest_payload = _json_bytes(manifest)
    artifacts.upload(f"{subpath}/_MANIFEST.json", manifest_payload)
    marker = {
        "status": "complete",
        "repository": repository,
        "commit_sha": commit_sha,
        "run_id": str(run_id),
        "run_attempt": str(run_attempt),
        "manifest_sha256": hashlib.sha256(manifest_payload).hexdigest(),
    }
    artifacts.upload(f"{subpath}/_COMPLETE.json", _json_bytes(marker))
    return manifest


class ArtifactoryApi:
    def __init__(self, base_url: str, token: str, repository: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.repository = repository

    def _request(self, url: str, *, payload: bytes | None = None) -> bytes:
        headers = {"Authorization": f"Bearer {self.token}", "User-Agent": "aiconfigurator-wheel-finalizer"}
        if payload is not None:
            headers["Content-Type"] = "application/json"
        for attempt in range(3):
            request = urllib.request.Request(url, data=payload, headers=headers, method="PUT" if payload else "GET")
            try:
                with urllib.request.urlopen(request, timeout=300) as response:
                    return response.read()
            except urllib.error.HTTPError as exc:
                if exc.code < 500 and exc.code != 429:
                    raise FinalizeError(f"Artifactory request failed for {url}: {exc}") from exc
                last_error: Exception = exc
            except urllib.error.URLError as exc:
                last_error = exc
            if attempt < 2:
                time.sleep(2**attempt)
        raise FinalizeError(f"Artifactory request failed for {url}: {last_error}") from last_error

    def _storage_url(self, path: str) -> str:
        quoted = "/".join(urllib.parse.quote(part, safe="") for part in path.split("/"))
        return f"{self.base_url}/api/storage/{self.repository}/{quoted}"

    def _artifact_url(self, path: str) -> str:
        quoted = "/".join(urllib.parse.quote(part, safe="") for part in path.split("/"))
        return f"{self.base_url}/{self.repository}/{quoted}"

    def _json(self, url: str) -> Mapping[str, Any]:
        try:
            value = json.loads(self._request(url))
        except json.JSONDecodeError as exc:
            raise FinalizeError(f"Artifactory returned invalid JSON for {url}") from exc
        if not isinstance(value, Mapping):
            raise FinalizeError(f"Artifactory returned a non-object for {url}")
        return value

    def list_files(self, subpath: str) -> list[dict[str, Any]]:
        listing = self._json(f"{self._storage_url(subpath)}?list&deep=0&listFolders=0")
        files = listing.get("files")
        if not isinstance(files, list):
            raise FinalizeError("Artifactory wheel listing has no files list")
        result: list[dict[str, Any]] = []
        for item in files:
            uri = item.get("uri") if isinstance(item, Mapping) else None
            if not isinstance(uri, str) or not uri.startswith("/") or "/" in uri[1:]:
                continue
            filename = uri[1:]
            if not filename.endswith(".whl"):
                continue
            metadata = self._json(self._storage_url(f"{subpath}/{filename}"))
            checksums = metadata.get("checksums")
            sha256 = checksums.get("sha256") if isinstance(checksums, Mapping) else None
            try:
                size = int(metadata["size"])
            except (KeyError, TypeError, ValueError) as exc:
                raise FinalizeError(f"Artifactory metadata has no valid size for {filename}") from exc
            result.append({"filename": filename, "sha256": sha256, "size": size})
        return result

    def upload(self, path: str, payload: bytes) -> None:
        self._request(self._artifact_url(path), payload=payload)


def _required_environment(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise FinalizeError(f"{name} is not configured")
    return value


def main() -> int:
    try:
        base_url = _required_environment("ARTIFACTORY_URL")
        token = _required_environment("ARTIFACTORY_TOKEN")
        artifact_repository = _required_environment("ARTIFACTORY_PYPI_REPO_NAME")
        subpath = _required_environment("ARTIFACTORY_SUBPATH")
        repository = _required_environment("GITHUB_REPOSITORY")
        commit_sha = _required_environment("GITHUB_SHA")
        run_id = int(_required_environment("GITHUB_RUN_ID"))
        run_attempt = int(_required_environment("GITHUB_RUN_ATTEMPT"))
        if not base_url.startswith("https://"):
            raise FinalizeError("ARTIFACTORY_URL must use HTTPS")
        if subpath.startswith("/") or ".." in subpath.split("/"):
            raise FinalizeError("ARTIFACTORY_SUBPATH must be relative and contain no parent traversal")
        finalize(
            ArtifactoryApi(base_url, token, artifact_repository),
            subpath=subpath,
            repository=repository,
            commit_sha=commit_sha,
            run_id=run_id,
            run_attempt=run_attempt,
        )
    except (FinalizeError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from exc
    print(f"Completed {subpath}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
