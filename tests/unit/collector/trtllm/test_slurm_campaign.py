# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from collector import provenance
from collector.wideep.trtllm import runtime_artifacts
from collector.wideep.trtllm.finalize_campaign import SYSTEM_LAYOUTS

pytestmark = pytest.mark.unit
REPO_ROOT = Path(__file__).resolve().parents[4]
SLURM = REPO_ROOT / "collector/wideep/trtllm/slurm"
SCRIPTS = tuple(sorted(SLURM.glob("*.sh")))


def test_six_system_single_node_matrix():
    assert SYSTEM_LAYOUTS == {
        "gb200": (4, 4),
        "gb300": (4, 4),
        "b200_sxm": (8, 8),
        "b300_sxm": (8, 8),
        "h100_sxm": (8, 8),
        "h200_sxm": (8, 8),
    }


@pytest.mark.parametrize("script", SCRIPTS)
def test_trtllm_campaign_scripts_are_valid_bash(script: Path):
    subprocess.run(["bash", "-n", str(script)], check=True)


def test_image_stage_builds_and_attests_exact_source_runtime():
    source = (SLURM / "run_trtllm_image_stage_job.sh").read_text(encoding="utf-8")
    assert "14efb6ac673c0cbe828e1206cc5c7d5748d05ffa" in source
    assert "5be51b228a7c82dbdb213ea58e77bffd12b38af8" in source
    assert "3.2.5-1" in source
    assert "eb2c8fb3b7084c2db86bd9fd905387909f1dfd483e7b45f7b3c3d5fcf5374b5a" in source
    assert "manifests/{index}" in source
    assert "Docker-Content-Digest" in source
    assert "nvcr.io#nvidia/tensorrt-llm/release:${IMAGE_INDEX_DIGEST}" in source
    assert 're.subn(r"\\.manifests\\[\\](?!\\?)"' in source
    assert "python3 scripts/build_wheel.py" in source
    # The pinned rc11 setup.py requires the generated ``bindings/`` stub
    # package to exist before it will build the wheel.
    assert "--skip-stubs" not in source
    assert "srun --mpi=pmix" in source
    assert "--cuda_architectures" in source
    assert "transformers==4.57.3" in source
    assert "--only-binary=:all:" in source
    assert '"dependency_wheels": dependency_wheels' in source
    assert 'tensorrt_llm.__version__ == "1.3.0rc11"' in source
    assert '"wheel_sha256": wheel_sha' in source
    assert "/mnt/cifs|/mnt/cifs/*|/mnt/nvdl|/mnt/nvdl/*" in source
    assert "collector.wideep.trtllm.runtime_artifacts" in source
    assert 'seed_args+=(--wheel-dir "${seed_wheel_dir}")' in source


def test_submitters_keep_six_cluster_parameters_and_afterok_gate():
    stage = (SLURM / "submit_trtllm_image_stage.sh").read_text(encoding="utf-8")
    submit = (SLURM / "submit_trtllm_moe_a2a.sh").read_text(encoding="utf-8")
    for token in ("coreai_comparch_inferencex", "blackwell", "dl_frameworks", "beta-users_fallback", "beta-users_b300"):
        assert token in stage and token in submit
    assert "--switches=1" in stage and "--switches=1" in submit
    assert "--exclusive" not in stage and "--exclusive" in submit
    assert '"${campaign_root}/images/trtllm/${system}"' in stage
    assert '"${campaign_root}/runtime/trtllm/${system}"' in stage
    assert '--dependency="afterok:${afterok_job}"' in submit
    assert '--dependency="afterok:${afterok_stage_job}"' in submit
    assert "--afterok-stage-job requires a numeric canary dependency" in submit
    assert "--account) account_override" in stage and "--account) account_override" in submit
    assert "--partition) partition_override" in stage and "--partition) partition_override" in submit
    assert '--time="${time_limit}"' in stage and '--time="${time_limit}"' in submit
    assert "runtime paths do not match the attested stage outputs" in submit
    assert "future runtime parent escapes campaign root" in submit
    assert "2202825c5950b4925e1add7d458228c9ad3368671789856f24d8947b4defd21c" in submit
    assert "9b3b4dfb811caa9420fa99a6f958155f6a1f727ffc2b5a5c2d9d2ce51fdc323d" in submit
    assert "canary/1n/${backend}/job_*/SUCCESS" in submit
    assert "infra-approved nodelist and approval ID" in submit
    assert "trtllm_deepep_ht|trtllm_deepep_ll" in submit
    assert "--seed-image" in stage and "--seed-wheel-dir" in stage
    assert 'export AIC_REPO_DIR="${repo_root}"' in stage
    assert 'repo_root=$(realpath -e "${script_dir}/../../../..")' in stage


def test_runner_is_one_node_mpi_and_preserves_failed_rows():
    source = (SLURM / "run_trtllm_moe_a2a_job.sh").read_text(encoding="utf-8")
    assert "formal TRT-LLM campaign is single-node only" in source
    assert "--mpi=pmix" in source
    assert "python3 -m collector.wideep.trtllm.collect_moe_a2a" in source
    assert '--system "${AIC_SYSTEM}"' in source
    assert 'staging_root="/tmp/aic-trtllm-a2a-${SLURM_JOB_ID}"' in source
    assert "benchmark_status=$?" in source
    assert "all partial rows and rank evidence preserved" in source
    assert "failure_evidence/${SYSTEM}/trtllm" in source
    assert 'wm.get("python_requirements") != ["transformers==4.57.3"]' in source
    assert 'wm.get("cuda_arches") != cuda_arches' in source
    assert 'payload["seed_provenance"] = seed' in source
    assert "runtime dependency wheel set mismatch" in source
    assert '--no-index --find-links "${AIC_DEPENDENCY_DIR}"' in source
    assert 'touch "${destination}/SUCCESS"' in source
    assert "/mnt/cifs|/mnt/cifs/*|/mnt/nvdl|/mnt/nvdl/*" in source


def test_trtllm_hash_closure_includes_full_campaign_chain():
    closure = provenance.load_closures(REPO_ROOT / "collector/hash_closures.yaml")[
        "collector.wideep.trtllm.collect_moe_a2a"
    ]
    assert {
        "collector/wideep/trtllm/finalize_campaign.py",
        "collector/wideep/trtllm/slurm/run_trtllm_image_stage_job.sh",
        "collector/wideep/trtllm/slurm/submit_trtllm_image_stage.sh",
        "collector/wideep/trtllm/slurm/run_trtllm_moe_a2a_job.sh",
        "collector/wideep/trtllm/slurm/submit_trtllm_moe_a2a.sh",
    } <= set(closure)


def _seed_image(tmp_path: Path, *, system: str) -> tuple[Path, Path, dict[str, object]]:
    arch, cuda_arches = runtime_artifacts.SYSTEM_RUNTIME[system]
    image = tmp_path / "runtime.sqsh"
    image.write_bytes(b"seed-image")
    meta = {
        "schema_version": 1,
        "system": system,
        "architecture": arch,
        "image_variant": f"linux/{arch}",
        "configured_image": (f"nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc20@{runtime_artifacts.IMAGE_INDEX_DIGEST}"),
        "configured_image_digest": runtime_artifacts.IMAGE_INDEX_DIGEST,
        "observed_image_digest": runtime_artifacts.IMAGE_CHILD_DIGESTS[arch],
        "sqsh_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
        "cuda_arches": cuda_arches,
    }
    meta_path = image.with_suffix(image.suffix + ".meta.json")
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    return image, meta_path, meta


def _seed_wheel(tmp_path: Path, *, image_meta: dict[str, object]) -> Path:
    wheel_dir = tmp_path / "wheel"
    dependency_dir = wheel_dir / "dependencies"
    dependency_dir.mkdir(parents=True)
    wheel = wheel_dir / "tensorrt_llm-1.3.0rc11.whl"
    dependency = dependency_dir / "transformers-4.57.3.whl"
    wheel.write_bytes(b"wheel")
    dependency.write_bytes(b"dependency")
    meta = dict(image_meta) | {
        "trtllm_version": "1.3.0rc11",
        "source_commit": runtime_artifacts.SOURCE_COMMIT,
        "deep_ep": runtime_artifacts.DEEPEP_COMMIT,
        "nvshmem": runtime_artifacts.NVSHMEM_VERSION,
        "nvshmem_archive_sha256": runtime_artifacts.NVSHMEM_ARCHIVE_SHA256,
        "python_requirements": runtime_artifacts.PYTHON_REQUIREMENTS,
        "wheel": wheel.name,
        "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
        "dependency_wheels": {dependency.name: hashlib.sha256(dependency.read_bytes()).hexdigest()},
    }
    (wheel_dir / "build_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    (wheel_dir / "SUCCESS").touch()
    return wheel_dir


@pytest.mark.parametrize(("source", "target"), [("b200_sxm", "b300_sxm"), ("gb200", "gb300")])
def test_seed_image_allows_same_cpu_arch_with_target_specific_wheel(source: str, target: str, tmp_path: Path):
    image, meta_path, _ = _seed_image(tmp_path, system=source)
    _, seed = runtime_artifacts.validate_image(image, meta_path, target_system=target)
    assert seed["mode"] == "image"
    assert seed["source_system"] == source


def test_seed_runtime_allows_exact_hopper_arch(tmp_path: Path):
    image, meta_path, meta = _seed_image(tmp_path, system="h100_sxm")
    wheel_dir = _seed_wheel(tmp_path, image_meta=meta)
    image_meta, seed = runtime_artifacts.validate_image(image, meta_path, target_system="h200_sxm")
    _, seed = runtime_artifacts.validate_wheel(
        wheel_dir, target_system="h200_sxm", image_meta=image_meta, provenance=seed
    )
    assert seed["mode"] == "runtime"
    assert seed["cuda_arches"] == "90-real"


def test_seed_runtime_rejects_cross_cuda_arch_wheel(tmp_path: Path):
    image, meta_path, meta = _seed_image(tmp_path, system="b200_sxm")
    wheel_dir = _seed_wheel(tmp_path, image_meta=meta)
    image_meta, seed = runtime_artifacts.validate_image(image, meta_path, target_system="b300_sxm")
    with pytest.raises(runtime_artifacts.RuntimeArtifactError, match="cuda_arches mismatch"):
        runtime_artifacts.validate_wheel(wheel_dir, target_system="b300_sxm", image_meta=image_meta, provenance=seed)


def test_seed_image_rejects_checksum_mismatch(tmp_path: Path):
    image, meta_path, _ = _seed_image(tmp_path, system="h100_sxm")
    image.write_bytes(b"tampered")
    with pytest.raises(runtime_artifacts.RuntimeArtifactError, match="sqsh checksum mismatch"):
        runtime_artifacts.validate_image(image, meta_path, target_system="h200_sxm")
