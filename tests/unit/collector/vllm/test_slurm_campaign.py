# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from collector import provenance
from collector.wideep.vllm.finalize_campaign import SYSTEM_GPU_IDENTITIES, SYSTEM_LAYOUTS

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[4]
RUNNER = REPO_ROOT / "collector/wideep/vllm/slurm/run_vllm_moe_a2a_job.sh"
SUBMITTER = REPO_ROOT / "collector/wideep/vllm/slurm/submit_vllm_moe_a2a.sh"


def test_six_system_matrix_has_exact_formal_topologies():
    assert SYSTEM_LAYOUTS == {
        "gb200": (4, {2: 8, 4: 16}),
        "gb300": (4, {2: 8, 4: 16}),
        "b200_sxm": (8, {2: 16, 4: 32}),
        "b300_sxm": (8, {2: 16, 4: 32}),
        "h100_sxm": (8, {2: 16, 4: 32}),
        "h200_sxm": (8, {2: 16, 4: 32}),
    }
    assert SYSTEM_GPU_IDENTITIES == {
        "gb200": ("GB200", "10.0"),
        "gb300": ("GB300", "10.3"),
        "b200_sxm": ("B200", "10.0"),
        "b300_sxm": ("B300", "10.3"),
        "h100_sxm": ("H100", "9.0"),
        "h200_sxm": ("H200", "9.0"),
    }


@pytest.mark.parametrize("script", [RUNNER, SUBMITTER])
def test_slurm_campaign_scripts_are_valid_bash(script):
    subprocess.run(["bash", "-n", str(script)], check=True)


def test_runner_rejects_cifs_and_requires_authoritative_topology():
    source = RUNNER.read_text(encoding="utf-8")
    assert "/mnt/cifs|/mnt/cifs/*|/mnt/nvdl|/mnt/nvdl/*" in source
    assert "slurm_topology_verified" in source
    assert "AIC_APPROVED_NODELIST" in source
    assert "AIC_FABRIC_APPROVAL_ID" in source
    assert "topology_matches" in source
    assert 'staging_root="/tmp/aic-vllm-a2a-${SLURM_JOB_ID}"' in source
    assert "gpu_inventory" in source
    assert "--gpus-per-task" not in source
    assert "--gpu-bind" not in source


def test_submitter_requires_canaries_and_one_job_per_backend():
    source = SUBMITTER.read_text(encoding="utf-8")
    assert 'backends="deepep_ht,deepep_ll,deepep_v2"' in source
    assert "canary/2n/${backend}/job_*/SUCCESS" in source
    assert '--gpus-per-node="${gpus_per_node}"' in source
    assert "--exclusive" in source
    assert "--switches=1" in source
    assert "B200 formal/canary submission requires infra-approved nodelist" in source
    assert "B300 formal/canary submission requires infra-approved nodelist" in source


def test_vllm_collector_hash_closure_includes_campaign_pipeline():
    closures = provenance.load_closures(REPO_ROOT / "collector/hash_closures.yaml")
    closure = closures["collector.wideep.vllm.collect_moe_a2a"]
    assert "collector/wideep/vllm/finalize_campaign.py" in closure
    assert "collector/wideep/vllm/slurm/run_vllm_moe_a2a_job.sh" in closure
    assert "collector/wideep/vllm/slurm/submit_vllm_moe_a2a.sh" in closure
