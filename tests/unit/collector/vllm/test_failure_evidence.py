# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
import subprocess
import sys

import pytest

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("partial", [True, False], ids=["partial", "fatal"])
def test_failure_evidence_cli_preserves_and_checksums_every_available_artifact(tmp_path, partial):
    evidence = tmp_path / "evidence"
    host = evidence / "node0"
    host.mkdir(parents=True)
    (host / "errors_moe_a2a_vllm.rank0.json").write_text('[{"error":"boom"}]\n', encoding="utf-8")
    if partial:
        (evidence / "moe_a2a_perf.parquet").write_bytes(b"partial parquet")
        (evidence / "collection_meta.yaml").write_text("schema_version: 1\n", encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            "-m",
            "collector.wideep.vllm.failure_evidence",
            str(evidence),
            "91" if partial else "134",
            "1234",
            "deepep_ll",
            "full",
            "case failures" if partial else "collector command failed",
        ],
        check=True,
    )

    failure = json.loads((evidence / "job_failure.json").read_text(encoding="utf-8"))
    assert failure["benchmark_status"] == (91 if partial else 134)
    manifest = json.loads((evidence / "artifact_checksums.json").read_text(encoding="utf-8"))
    expected_names = {"node0/errors_moe_a2a_vllm.rank0.json", "job_failure.json"}
    if partial:
        expected_names |= {"moe_a2a_perf.parquet", "collection_meta.yaml"}
    assert set(manifest) == expected_names
    for name, digest in manifest.items():
        assert hashlib.sha256((evidence / name).read_bytes()).hexdigest() == digest
