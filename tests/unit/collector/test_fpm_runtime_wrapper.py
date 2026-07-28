# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
RUNTIME_WRAPPER = REPO_ROOT / "collector/fpm_forward/runtime/run_with_etcd.sh"


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content)
    path.chmod(0o755)


def _run_follower_wrapper(tmp_path: Path, discovery_env: dict[str, str]) -> str:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    trace_file = tmp_path / "endpoint.txt"
    _write_executable(
        bin_dir / "python3",
        """#!/bin/bash
if [[ "${1:-}" == "-" ]]; then
  /bin/cat >/dev/null
fi
exit 0
""",
    )
    _write_executable(
        bin_dir / "bash",
        """#!/bin/bash
printf '%s\\n' "${ETCD_ENDPOINTS}" > "${TRACE_FILE}"
exit 0
""",
    )
    _write_executable(
        bin_dir / "tee",
        """#!/bin/bash
/bin/cat >/dev/null
""",
    )
    environment = {
        "PATH": str(bin_dir),
        "TRACE_FILE": str(trace_file),
        **discovery_env,
    }

    subprocess.run(
        ["/bin/bash", str(RUNTIME_WRAPPER)],
        cwd=REPO_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    )
    return trace_file.read_text().strip()


@pytest.mark.parametrize(
    ("discovery_env", "expected_endpoint"),
    [
        (
            {
                "FPM_NODE_RANK": "1",
                "FPM_MASTER_ADDR": "explicit-leader",
                "LWS_WORKER_INDEX": "7",
                "LWS_LEADER_ADDRESS": "ignored-lws-leader",
            },
            "http://explicit-leader:2379",
        ),
        (
            {
                "LWS_WORKER_INDEX": "1",
                "LWS_LEADER_ADDRESS": "lws-leader",
            },
            "http://lws-leader:2379",
        ),
        (
            {
                "GROVE_PCLQ_POD_INDEX": "1",
                "GROVE_PCLQ_NAME": "fpm-worker",
                "GROVE_HEADLESS_SERVICE": "fpm-headless",
            },
            "http://fpm-worker-0.fpm-headless:2379",
        ),
    ],
)
def test_runtime_wrapper_discovers_follower_rank_and_leader(
    tmp_path,
    discovery_env,
    expected_endpoint,
):
    assert _run_follower_wrapper(tmp_path, discovery_env) == expected_endpoint


def test_runtime_wrapper_rejects_partial_multinode_discovery(tmp_path):
    with pytest.raises(subprocess.CalledProcessError) as error:
        _run_follower_wrapper(tmp_path, {"GROVE_PCLQ_POD_INDEX": "1"})

    assert error.value.returncode == 2
    assert "requires complete rank and leader discovery" in error.value.stderr
