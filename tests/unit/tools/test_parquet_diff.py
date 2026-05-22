# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
PARQUET_DIFF = REPO_ROOT / "tools" / "perf_database" / "parquet_diff.py"
PERF_PATH = Path("src/aiconfigurator/systems/data/h100_sxm/trtllm/1.0.0/gemm_perf.txt")


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True)


def _init_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test User")
    return repo


def _write_perf_file(repo: Path, body: str = "op,latency_ms\ngemm,1.0\n") -> None:
    path = repo / PERF_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)


def _run_parquet_diff(repo: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            str(PARQUET_DIFF),
            "--base-ref",
            "HEAD~1",
            "--head-ref",
            "HEAD",
        ],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize("change_type", ["added", "modified"])
def test_parquet_diff_fails_when_legacy_perf_text_is_added_or_modified(tmp_path, change_type):
    repo = _init_repo(tmp_path)

    if change_type == "modified":
        _write_perf_file(repo)
        _git(repo, "add", str(PERF_PATH))

    _git(repo, "commit", "--allow-empty", "-m", "base")

    _write_perf_file(repo, "op,latency_ms\ngemm,2.0\n")
    _git(repo, "add", str(PERF_PATH))
    _git(repo, "commit", "-m", f"{change_type} legacy perf text")

    result = _run_parquet_diff(repo)

    assert result.returncode == 1
    assert "- Legacy `*_perf.txt` files added or modified: 1" in result.stdout
    assert "### Legacy Text Perf Files Still Changed" in result.stdout


def test_parquet_diff_allows_legacy_perf_text_deletions(tmp_path):
    repo = _init_repo(tmp_path)
    _write_perf_file(repo)
    _git(repo, "add", str(PERF_PATH))
    _git(repo, "commit", "-m", "base")

    _git(repo, "rm", str(PERF_PATH))
    _git(repo, "commit", "-m", "delete legacy perf text")

    result = _run_parquet_diff(repo)

    assert result.returncode == 0
    assert "- Legacy `*_perf.txt` files added or modified: 0" in result.stdout
