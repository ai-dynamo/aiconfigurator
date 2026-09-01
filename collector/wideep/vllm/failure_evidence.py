# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Complete and checksum durable vLLM Slurm failure evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def complete_failure_evidence(
    root: Path,
    *,
    status: int,
    job_id: str,
    backend: str,
    run_kind: str,
    reason: str,
) -> dict[str, str]:
    root = root.resolve(strict=True)
    payload = {
        "benchmark_status": status,
        "job_id": job_id,
        "backend": backend,
        "run_kind": run_kind,
        "reason": reason,
    }
    (root / "job_failure.json").write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {}
    for artifact in sorted(root.rglob("*")):
        if artifact.is_file() and artifact.name != "artifact_checksums.json":
            manifest[str(artifact.relative_to(root))] = hashlib.sha256(artifact.read_bytes()).hexdigest()
    (root / "artifact_checksums.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("status", type=int)
    parser.add_argument("job_id")
    parser.add_argument("backend")
    parser.add_argument("run_kind")
    parser.add_argument("reason")
    args = parser.parse_args()
    complete_failure_evidence(
        args.root,
        status=args.status,
        job_id=args.job_id,
        backend=args.backend,
        run_kind=args.run_kind,
        reason=args.reason,
    )


if __name__ == "__main__":
    main()
