# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re
import subprocess
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
COLLECTOR_ROOT = REPO_ROOT / "collector"
NETWORK_ROOT = COLLECTOR_ROOT / "network"

SUBMIT_MOE_A2A = NETWORK_ROOT / "slurm" / "submit_moe_a2a.sh"
SUBMIT_TRTLLM_ALLTOALL = NETWORK_ROOT / "slurm" / "submit_trtllm_alltoall.sh"


def test_network_collectors_are_grouped_under_network_folder():
    expected_paths = [
        NETWORK_ROOT / "collect_comm.sh",
        NETWORK_ROOT / "collect_nccl.py",
        NETWORK_ROOT / "collect_oneccl_xpu.py",
        NETWORK_ROOT / "collect_all_reduce.py",
        NETWORK_ROOT / "slurm" / "collect_allreduce.py",
        NETWORK_ROOT / "slurm" / "collect_trtllm_alltoall.py",
        SUBMIT_TRTLLM_ALLTOALL,
        SUBMIT_MOE_A2A,
    ]

    for path in expected_paths:
        assert path.exists(), f"missing network collector: {path}"

    old_top_level_paths = [
        COLLECTOR_ROOT / "collect_comm.sh",
        COLLECTOR_ROOT / "collect_nccl.py",
        COLLECTOR_ROOT / "collect_oneccl_xpu.py",
        COLLECTOR_ROOT / "collect_all_reduce.py",
        COLLECTOR_ROOT / "slurm_comm_collector",
    ]

    for path in old_top_level_paths:
        assert not path.exists(), f"network collector should live under collector/network: {path}"


def test_slurm_network_docs_use_new_folder_name():
    docs = [
        NETWORK_ROOT / "README.md",
        NETWORK_ROOT / "slurm" / "README.md",
        SUBMIT_TRTLLM_ALLTOALL,
        SUBMIT_MOE_A2A,
    ]

    for path in docs:
        assert "slurm_comm_collector" not in path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Launcher scripts
# ---------------------------------------------------------------------------


def _default_container_image(script: Path) -> str:
    """The launcher's CONTAINER_IMAGE default (`${CONTAINER_IMAGE:-<value>}`)."""
    match = re.search(r'CONTAINER_IMAGE="\$\{CONTAINER_IMAGE:-([^}]+)\}"', script.read_text(encoding="utf-8"))
    assert match, f"{script} declares no CONTAINER_IMAGE default"
    return match.group(1)


def _manifest_frameworks() -> dict:
    with (COLLECTOR_ROOT / "framework_manifest.yaml").open(encoding="utf-8") as manifest_file:
        return yaml.safe_load(manifest_file)["frameworks"]


def test_submit_moe_a2a_default_image_matches_manifest():
    # The launcher default must be the manifest wideep_sglang pin
    # (grace_blackwell variant: the launcher's default partition is GB200) —
    # never a hardcoded stale ref, the precedent launcher's drift bug.
    manifest_image = _manifest_frameworks()["wideep_sglang"]["default"]["images"]["grace_blackwell"]
    assert _default_container_image(SUBMIT_MOE_A2A) == manifest_image


def test_submit_trtllm_alltoall_default_image_matches_manifest():
    # Same sync requirement against the trtllm pin. The collector gates the
    # installed tensorrt_llm version against this pin, so drift here would
    # fail every job loudly.
    manifest_image = _manifest_frameworks()["trtllm"]["default"]["images"]["default"]
    assert _default_container_image(SUBMIT_TRTLLM_ALLTOALL) == manifest_image


@pytest.mark.parametrize("script", [SUBMIT_MOE_A2A, SUBMIT_TRTLLM_ALLTOALL], ids=lambda path: path.name)
def test_launcher_scripts_pass_bash_syntax_check(script):
    result = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
    assert result.returncode == 0, f"bash -n {script.name} failed: {result.stderr}"


def test_submit_moe_a2a_invokes_the_collector_with_the_world_layout():
    text = SUBMIT_MOE_A2A.read_text(encoding="utf-8")
    # The collector module, launched with the divisor node_num derives from.
    assert "python -m collector.wideep.sglang.collect_moe_a2a" in text
    assert "--gpus-per-node" in text
    assert "--output-path" in text
    # Slurm rendezvous + full-rack segment pinning per the precedent.
    assert "scontrol show hostname" in text
    assert "MASTER_PORT" in text
    assert "--mpi=pmix" in text
    assert "--segment" in text


def test_submit_launchers_use_per_job_output_dirs():
    # Each job finalizes its own parquet and attests a world-specific
    # collection_meta.yaml, so jobs get per-world output dirs; within a job
    # all cases append one CSV through log_perf's lockfile.
    for script in (SUBMIT_MOE_A2A, SUBMIT_TRTLLM_ALLTOALL):
        text = script.read_text(encoding="utf-8")
        assert 'OUTPUT_DIR="${SCRIPT_DIR}/results/moe_a2a_' in text, script.name
        assert "${NUM_GPUS}gpu" in text, script.name
