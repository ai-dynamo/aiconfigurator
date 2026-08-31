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
IBSTAT_WRAPPER = REPO_ROOT / "collector/wideep/vllm/slurm/host_tools/ibstat"
LEGACY_NVL4_PATCH = REPO_ROOT / "collector/wideep/vllm/patches/deepep_73b_nvl4.patch"


def test_six_system_matrix_has_exact_formal_topologies():
    assert SYSTEM_LAYOUTS == {
        "gb200": (4, {1: 4}),
        "gb300": (4, {1: 4}),
        "b200_sxm": (8, {1: 8}),
        "b300_sxm": (8, {1: 8}),
        "h100_sxm": (8, {1: 8}),
        "h200_sxm": (8, {1: 8}),
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
    "script",
    [RUNNER, SUBMITTER, OVERLAY_RUNNER, OVERLAY_SUBMITTER, IMAGE_RUNNER, IMAGE_SUBMITTER, IBSTAT_WRAPPER],
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
    assert '--container-workdir="${staging_root}"' in source
    assert '--container-workdir="${repo_dir}"' not in source
    assert "--gpus-per-task" not in source
    assert "--gpu-bind" not in source


def test_runner_never_publishes_a_failed_case_plan():
    source = RUNNER.read_text(encoding="utf-8")

    assert '[[ "${NODE_NUM}" == 1 ]]' in source
    assert "formal job produced unexpected case failures; evidence copied" in source
    assert "known_framework_limit" not in source
    assert 'cp -a -- "${output_dir}/." "${failure_dir}/"' in source
    assert 'python3 "${repo_dir}/collector/wideep/vllm/failure_evidence.py"' in source
    assert source.count("preserve_failure_evidence") == 5  # definition plus all four fail-closed exits
    assert 'preserve_failure_evidence "${benchmark_status}" "collector command failed"' in source
    assert 'preserve_failure_evidence 90 "collector omitted finalized artifacts"' in source
    assert 'preserve_failure_evidence 91 "collector returned success with case failures"' in source
    assert 'touch "${output_dir}/SUCCESS"' in source
    assert 'mv -- "${publish_stage}" "${campaign_job_dir}"' in source
    assert 'merge_lock="${output_dir}/moe_a2a_perf.parquet.mergelock"' in source
    assert '[[ ! -f "${merge_lock}" || -L "${merge_lock}" || -s "${merge_lock}" ]]' in source
    assert 'rm -f -- "${merge_lock}"' in source
    assert source.index('merge_lock="${output_dir}') < source.index('if [[ "${benchmark_status}" -ne 0 ]]')


def test_runner_validates_and_propagates_image_metadata_migration():
    source = RUNNER.read_text(encoding="utf-8")
    assert 'reference_mode != "attested-schema1-migration"' in source
    assert '"source_metadata_sha256", "source_sqsh_sha256", "destination_metadata_sha256"' in source
    assert 'payload["image_metadata_migration"] = json.dumps(' in source
    assert 'json.loads(sys.argv[14]), sort_keys=True, separators=(",", ":")' in source


def test_runner_discovers_and_records_a_routable_gloo_interface():
    source = RUNNER.read_text(encoding="utf-8")
    assert 'export AIC_GLOO_ROUTE_PROBE_NODES="${allocated_nodes[*]}"' in source
    assert 'getent ahostsv4 "${peer}"' in source
    assert 'ip -o route get "${peer_address}"' in source
    assert "ip -o route show default" in source
    assert '"${selected_interface}" != lo' in source
    assert '"allocated nodes use inconsistent Gloo interface names:' in source
    assert '"gloo_socket_ifname": sys.argv[10]' in source
    assert 'export GLOO_SOCKET_IFNAME="${gloo_socket_ifname}"' in source
    assert "export VLLM_DEEPEP_V2_ALLOW_HYBRID_MODE=0" in source
    assert 'collector_ref=$(git -C "${repo_dir}" rev-parse HEAD)' in source
    assert 'export AIC_COLLECTOR_REF="${collector_ref}"' in source
    assert 'container_command+=" --collector-ref ${collector_ref}"' in source
    assert 'export AIC_IBSTAT_TOOL_ROOT="${staging_root}/host-rdma-tools"' in source
    assert "container_command+=" in source and "ibstat_output=$(ibstat mlx5_0)" in source
    assert '"ibstat_bundle_sha256"' in source
    assert '"ibstat_mlx5_0_rate_gbps"' in source


def test_submitter_requires_canaries_and_one_job_per_backend():
    source = SUBMITTER.read_text(encoding="utf-8")
    assert 'h100_sxm) formal_backends="deepep_ht,deepep_ll,deepep_v2"' in source
    assert 'gb200|gb300|b200_sxm|b300_sxm|h200_sxm) formal_backends="deepep_ht,deepep_ll"' in source
    assert "backends=${backends:-${formal_backends}}" in source
    assert source.index('die "${backend} is not a formal backend for ${system}') < source.index("job_id=$(sbatch")
    assert source.index("declare -A overlay_dirs") < source.index('for node_num in "${node_values[@]}"')
    assert "canary/1n/${backend}/job_*/SUCCESS" in source
    assert '--gpus-per-node="${gpus_per_node}"' in source
    assert "--exclusive" in source
    assert "--switches=1" in source
    assert "${system} submission requires infra-approved nodelist" in source
    assert "--partition-override) partition_override=$2" in source
    assert "partition=${partition_override}" in source
    assert '[[ "${nodes}" == 1 ]]' in source
    assert "supports only --nodes 1" in source
    assert "node_values=(1)" in source
    assert "node_values=(1 2 4)" not in source
    assert "topology_mode=single_node" in RUNNER.read_text(encoding="utf-8")
    assert '--dependency="afterok:${afterok_job}"' in source
    assert "--afterok-job must be BACKEND=JOB_ID" in source
    assert '[[ "${afterok_spec%%=*}" != "${backend}" ]] || afterok_job=${afterok_spec#*=}' in source
    assert 'export AIC_CANARY_JOB_ID="${afterok_job}"' in source
    assert (
        '"${campaign_root}/${SYSTEM}/canary/${NODE_NUM}n/${BACKEND}/job_${canary_job_id}/SUCCESS"'
        in RUNNER.read_text(encoding="utf-8")
    )
    assert "requires --legacy-overlay-dir" in source


def test_submitter_rejects_unsupported_v2_before_path_or_submission_validation():
    result = _run_submitter_dependency_validation("--backends", "deepep_v2")

    assert result.returncode != 0
    assert "deepep_v2 is not a formal backend for h200_sxm" in result.stderr
    assert "/does/not/exist" not in result.stderr


def _run_submitter_dependency_validation(*extra_args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "bash",
            str(SUBMITTER),
            "--system",
            "h200_sxm",
            "--run-kind",
            "full",
            "--campaign-root",
            "/does/not/exist",
            "--repo-dir",
            "/does/not/exist",
            "--vllm-source-root",
            "/does/not/exist",
            *extra_args,
        ],
        capture_output=True,
        text=True,
    )


def test_submitter_rejects_unbound_scalar_canary_dependency():
    result = _run_submitter_dependency_validation("--backends", "deepep_ht", "--afterok-job", "123")

    assert result.returncode != 0
    assert "--afterok-job must be BACKEND=JOB_ID" in result.stderr


def test_submitter_rejects_canary_mapping_for_unselected_backend():
    result = _run_submitter_dependency_validation("--backends", "deepep_ht", "--afterok-job", "deepep_ll=123")

    assert result.returncode != 0
    assert "--afterok-job backend deepep_ll is not selected" in result.stderr


def test_submitter_rejects_missing_backend_canary_mapping():
    result = _run_submitter_dependency_validation("--backends", "deepep_ht,deepep_ll", "--afterok-job", "deepep_ht=123")

    assert result.returncode != 0
    assert "missing --afterok-job for selected backend deepep_ll" in result.stderr


def test_submitter_rejects_duplicate_or_shared_canary_mapping():
    duplicate = _run_submitter_dependency_validation(
        "--backends",
        "deepep_ht,deepep_ll",
        "--afterok-job",
        "deepep_ht=123",
        "--afterok-job",
        "deepep_ht=456",
    )
    shared = _run_submitter_dependency_validation(
        "--backends",
        "deepep_ht,deepep_ll",
        "--afterok-job",
        "deepep_ht=123",
        "--afterok-job",
        "deepep_ll=123",
    )

    assert duplicate.returncode != 0
    assert "duplicate --afterok-job for deepep_ht" in duplicate.stderr
    assert shared.returncode != 0
    assert "Slurm canary job 123 is bound to multiple backends" in shared.stderr


@pytest.mark.parametrize("nodes", ["2", "4", "all"])
def test_submitter_rejects_every_non_single_node_formal_request(nodes):
    result = subprocess.run(
        [
            "bash",
            str(SUBMITTER),
            "--system",
            "h200_sxm",
            "--run-kind",
            "full",
            "--nodes",
            nodes,
            "--campaign-root",
            "/does-not-need-to-exist",
            "--repo-dir",
            "/does-not-need-to-exist",
            "--vllm-source-root",
            "/does-not-need-to-exist",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "supports only --nodes 1" in result.stderr


def test_submitter_rejects_unsafe_partition_override():
    result = subprocess.run(
        [
            "bash",
            str(SUBMITTER),
            "--system",
            "gb300",
            "--run-kind",
            "canary",
            "--campaign-root",
            "/does-not-need-to-exist",
            "--repo-dir",
            "/does-not-need-to-exist",
            "--vllm-source-root",
            "/does-not-need-to-exist",
            "--partition-override",
            "unsafe;command",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "invalid Slurm partition override" in result.stderr


def test_vllm_collector_hash_closure_includes_campaign_pipeline():
    closures = provenance.load_closures(REPO_ROOT / "collector/hash_closures.yaml")
    closure = closures["collector.wideep.vllm.collect_moe_a2a"]
    assert "collector/framework_manifest.yaml" in closure
    assert "collector/wideep/vllm/finalize_campaign.py" in closure
    assert "collector/wideep/vllm/failure_evidence.py" in closure
    assert "collector/artifact_publication.py" in closure
    assert "collector/runtime_stage_publication.py" in closure
    assert "collector/wideep/vllm/slurm/run_vllm_moe_a2a_job.sh" in closure
    assert "collector/wideep/vllm/slurm/host_tools/ibstat" in closure
    assert "collector/wideep/vllm/slurm/submit_vllm_moe_a2a.sh" in closure
    assert "collector/wideep/vllm/slurm/run_deepep_sm103_overlay_job.sh" in closure
    assert "collector/wideep/vllm/slurm/submit_deepep_sm103_overlay.sh" in closure
    assert "collector/wideep/vllm/patches/deepep_73b_nvl4.patch" in closure
    assert "collector/wideep/vllm/slurm/run_vllm_image_stage_job.sh" in closure
    assert "collector/wideep/vllm/slurm/submit_vllm_image_stage.sh" in closure
    assert "collector/wideep/vllm/runtime_artifacts.py" in closure


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
    assert 'source.replace(v2_nvshmem, "")' in runner
    assert "library_dirs.extend([f'{nccl_root_dir}/lib'])" in runner
    assert 'grep -Eq "\\((RPATH|RUNPATH)\\)"' in runner
    assert '"schema_version": 3' in runner
    assert '"deep_ep_setup_sha256": setup_sha' in runner
    assert '"pyarrow==${AIC_PYARROW_VERSION}"' in runner
    assert '"pyarrow_wheel_sha256": pyarrow_sha' in runner
    assert 'path "*/nvidia/cu13/include"' in runner
    assert 'export CPATH="${bundled_cuda_include}:${CPATH:-}"' in runner
    assert 'export LIBRARY_PATH="${bundled_cuda_lib}:${LIBRARY_PATH:-}"' in runner
    assert "--container-image" in submitter
    assert "--gpus=1" in submitter
    assert "--exclusive" not in submitter
    assert "--switches=1" in submitter


def test_image_stage_serializes_exact_digest_to_verified_sqsh():
    runner = IMAGE_RUNNER.read_text(encoding="utf-8")
    submitter = IMAGE_SUBMITTER.read_text(encoding="utf-8")
    enroot_patch_program = runner.split(
        'python3 - "${enroot_library_dir}/docker.sh" "${enroot_library_dir}/common.sh" <<\'PY\'', 1
    )[1].split("\nPY\n", 1)[0]
    assert "ENROOT_MAX_CONNECTIONS=1" in runner
    assert "ENROOT_TRANSFER_RETRIES=8" in runner
    assert "unsquashfs -s" in runner
    assert "configured_image_digest" in runner
    assert "observed_image_digest" in runner
    assert '"image_variant": f"linux/{arch}"' in runner
    assert 'runtime["vllm"] != "0.24.0"' in runner
    assert "73b6ea4a439ba03a695563f9fd242c8e4b02b37c" in runner
    assert '"deep_ep_v2_available": hasattr(deep_ep, "ElasticBuffer")' in runner
    assert "sqsh_sha256" in runner
    assert "manifests/{expected_index_digest}" in runner
    assert "observed_index_digest != expected_index_digest" in runner
    assert "len(matches) != 1" in runner
    assert "image_reference_mode=enroot-3.4-index-digest" in runner
    assert "registry-1.docker.io#vllm/vllm-openai:${IMAGE_INDEX_DIGEST}" in runner
    assert 'CONTAINER_IMAGE="registry-1.docker.io#vllm/vllm-openai:${image_index_digest}"' in submitter
    assert 'enroot_library_dir="/tmp/aic-enroot-library-${SLURM_JOB_ID}"' in runner
    assert 'replacement = ".manifests[]?"' in runner
    assert "import re" in enroot_patch_program
    assert "re.subn" in enroot_patch_program
    assert 're.subn(r"\\.manifests\\[\\](?!\\?)", replacement, source)' in runner
    assert "replacement_count == 0 and replacement not in source" in runner
    assert runner.count("if replacement not in source:") == 1
    assert "docker_path.write_text(source)" in runner
    assert "common_path.write_text(source)" in runner
    assert "AIC_ENROOT_JSON_DEBUG_FILE" in runner
    assert 'runtime_success="${final_image}.SUCCESS"' in runner
    assert 'if [[ -f "${runtime_success}" ]]; then' in runner
    assert 'collector/runtime_stage_publication.py" "${runtime_success}"' in runner
    assert runner.index('mv "${temporary_meta}" "${image_meta}"') < runner.index('touch "${runtime_success}"')
    assert "${container_image}.SUCCESS" in (REPO_ROOT / "collector/wideep/vllm/slurm/submit_vllm_moe_a2a.sh").read_text(
        encoding="utf-8"
    )
    assert 'replacement = \'if ! tee "${AIC_ENROOT_JSON_DEBUG_FILE:-/dev/null}" | jq "$@"; then\'' in runner
    assert 'ENROOT_LIBRARY_PATH="${enroot_library_dir}" enroot import' in runner
    assert "amd64) enroot_arch=x86_64" in runner
    assert "arm64) enroot_arch=aarch64" in runner
    assert "verified-tag" not in runner
    assert '"image_reference_mode": image_reference_mode' in runner
    assert "b200_sxm|b300_sxm|h100_sxm|h200_sxm" in submitter
    assert "beta-users_fallback" in submitter
    assert "beta-users_b300" in submitter
    assert "image_stage_gpu_args=(--gpus-per-node=4)" in submitter
    assert "image_stage_gpu_args=(--gpus=1)" in submitter
    assert '"${image_stage_gpu_args[@]}"' in submitter
    assert "--exclusive" not in submitter
    assert "--switches=1" in submitter
    assert "Reused checksum-verified staged image" in runner
    assert 'collector/runtime_stage_publication.py" "${runtime_success}"' in runner
    assert 'export AIC_REPO_DIR="${repo_root}"' in submitter


def test_legacy_nvl4_patch_updates_every_four_byte_token_mask_use():
    patch = LEGACY_NVL4_PATCH.read_text(encoding="utf-8")
    assert "uint32_t is_token_in_rank_uint32" in patch
    assert "while (is_token_in_rank_uint32 != 0" in patch
    assert "broadcast(is_token_in_rank_uint32, i)" in patch
    assert "recv_is_token_in_rank_uint32" in patch
    assert "if (is_token_in_rank_uint32 != 0)" in patch
