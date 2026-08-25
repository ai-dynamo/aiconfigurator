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
OVERLAY_RUNNER = REPO_ROOT / "collector/wideep/vllm/slurm/run_deepep_sm103_overlay_job.sh"
OVERLAY_SUBMITTER = REPO_ROOT / "collector/wideep/vllm/slurm/submit_deepep_sm103_overlay.sh"
IMAGE_RUNNER = REPO_ROOT / "collector/wideep/vllm/slurm/run_vllm_image_stage_job.sh"
IMAGE_SUBMITTER = REPO_ROOT / "collector/wideep/vllm/slurm/submit_vllm_image_stage.sh"
LEGACY_NVL4_PATCH = REPO_ROOT / "collector/wideep/vllm/patches/deepep_73b_nvl4.patch"


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


@pytest.mark.parametrize(
    "script", [RUNNER, SUBMITTER, OVERLAY_RUNNER, OVERLAY_SUBMITTER, IMAGE_RUNNER, IMAGE_SUBMITTER]
)
def test_slurm_campaign_scripts_are_valid_bash(script):
    subprocess.run(["bash", "-n", str(script)], check=True)


def test_runner_rejects_cifs_and_requires_authoritative_topology():
    source = RUNNER.read_text(encoding="utf-8")
    assert "/mnt/cifs|/mnt/cifs/*|/mnt/nvdl|/mnt/nvdl/*" in source
    assert "slurm_topology_verified" in source
    assert "AIC_APPROVED_NODELIST" in source
    assert "AIC_FABRIC_APPROVAL_ID" in source
    assert "topology_matches" in source
    assert "prefix_matches" in source
    assert 'staging_root="/tmp/aic-vllm-a2a-${SLURM_JOB_ID}"' in source
    assert "gpu_inventory" in source
    assert "python3 -m collector.wideep.vllm.collect_moe_a2a" in source
    assert " python -m collector.wideep.vllm.collect_moe_a2a" not in source
    assert "failure_evidence" in source
    assert "benchmark_status=$?" in source
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
    assert "requires --legacy-overlay-dir" in source


def test_vllm_collector_hash_closure_includes_campaign_pipeline():
    closures = provenance.load_closures(REPO_ROOT / "collector/hash_closures.yaml")
    closure = closures["collector.wideep.vllm.collect_moe_a2a"]
    assert "collector/framework_manifest.yaml" in closure
    assert "collector/wideep/vllm/finalize_campaign.py" in closure
    assert "collector/wideep/vllm/slurm/run_vllm_moe_a2a_job.sh" in closure
    assert "collector/wideep/vllm/slurm/submit_vllm_moe_a2a.sh" in closure
    assert "collector/wideep/vllm/slurm/run_deepep_sm103_overlay_job.sh" in closure
    assert "collector/wideep/vllm/slurm/submit_deepep_sm103_overlay.sh" in closure
    assert "collector/wideep/vllm/patches/deepep_73b_nvl4.patch" in closure
    assert "collector/wideep/vllm/slurm/run_vllm_image_stage_job.sh" in closure
    assert "collector/wideep/vllm/slurm/submit_vllm_image_stage.sh" in closure


def test_backend_overlay_build_is_exact_and_separate_from_formal_data():
    runner = OVERLAY_RUNNER.read_text(encoding="utf-8")
    submitter = OVERLAY_SUBMITTER.read_text(encoding="utf-8")
    assert "73b6ea4a439ba03a695563f9fd242c8e4b02b37c" in runner
    assert "b306af06afd412c88e51e71802951606e40b7358" in runner
    assert "ee0da84ab9e04ac7610e28580af62c365e898389" in runner
    assert "NVSHMEM_VERSION=3.3.24" in runner
    assert "V2_NCCL_VERSION=2.30.4" in runner
    assert "legacy-nvl8|legacy-nvl4|v2" in submitter
    assert "deep_ep_wheel_sha256" in runner
    assert "deepep_73b_nvl4.patch" in runner
    assert 'container_image=$(safe_existing_path "container image"' in runner
    assert 'safe_existing_path "container image metadata"' in runner
    assert "local container squashfs checksum mismatch" in runner
    assert "nvidia-nccl-cu13==${AIC_NCCL_VERSION}" in runner
    assert "unset EP_NCCL_ROOT_DIR" in runner
    assert '"pyarrow==${AIC_PYARROW_VERSION}"' in runner
    assert '"pyarrow_wheel_sha256": pyarrow_sha' in runner
    assert 'path "*/nvidia/cu13/include"' in runner
    assert 'export CPATH="${bundled_cuda_include}:${CPATH:-}"' in runner
    assert 'export LIBRARY_PATH="${bundled_cuda_lib}:${LIBRARY_PATH:-}"' in runner
    assert "--container-image" in submitter
    assert "--exclusive" in submitter
    assert "--switches=1" in submitter


def test_image_stage_serializes_exact_digest_to_verified_sqsh():
    runner = IMAGE_RUNNER.read_text(encoding="utf-8")
    submitter = IMAGE_SUBMITTER.read_text(encoding="utf-8")
    assert "ENROOT_MAX_CONNECTIONS=1" in runner
    assert "ENROOT_TRANSFER_RETRIES=8" in runner
    assert '--container-save="${temporary_image}"' in runner
    assert "unsquashfs -s" in runner
    assert "source_image_digest" in runner
    assert 'runtime["vllm"] != "0.24.0"' in runner
    assert "73b6ea4a439ba03a695563f9fd242c8e4b02b37c" in runner
    assert '"deep_ep_v2_available": hasattr(deep_ep, "ElasticBuffer")' in runner
    assert "sqsh_sha256" in runner
    assert "registry-1.docker.io/v2/vllm/vllm-openai/manifests/v0.24.0" in runner
    assert "matches != [expected_digest]" in runner
    assert "image_reference_mode=enroot-3.4-digest" in runner
    assert 'enroot_library_dir="/tmp/aic-enroot-library-${SLURM_JOB_ID}"' in runner
    assert 'replacement = ".manifests[]?"' in runner
    assert '\npath.write_text(source.replace(needle, replacement))\nPY' in runner
    assert 'ENROOT_LIBRARY_PATH="${enroot_library_dir}" enroot import' in runner
    assert "amd64) enroot_arch=x86_64" in runner
    assert "arm64) enroot_arch=aarch64" in runner
    assert "image_reference_mode=verified-tag" in runner
    assert '"image_reference_mode": image_reference_mode' in runner
    assert "--exclusive" in submitter
    assert "--switches=1" in submitter


def test_legacy_nvl4_patch_updates_every_four_byte_token_mask_use():
    patch = LEGACY_NVL4_PATCH.read_text(encoding="utf-8")
    assert "uint32_t is_token_in_rank_uint32" in patch
    assert "while (is_token_in_rank_uint32 != 0" in patch
    assert "broadcast(is_token_in_rank_uint32, i)" in patch
    assert "recv_is_token_in_rank_uint32" in patch
    assert "if (is_token_in_rank_uint32 != 0)" in patch
