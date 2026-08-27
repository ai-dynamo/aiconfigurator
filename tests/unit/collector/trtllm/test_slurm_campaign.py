# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from collector import provenance
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
    assert 'tensorrt_llm.__version__ == "1.3.0rc11"' in source
    assert '"wheel_sha256": wheel_sha' in source
    assert "/mnt/cifs|/mnt/cifs/*|/mnt/nvdl|/mnt/nvdl/*" in source


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


def test_runner_is_one_node_mpi_and_preserves_failed_rows():
    source = (SLURM / "run_trtllm_moe_a2a_job.sh").read_text(encoding="utf-8")
    assert "formal TRT-LLM campaign is single-node only" in source
    assert "--mpi=pmix" in source
    assert "python3 -m collector.wideep.trtllm.collect_moe_a2a" in source
    assert 'staging_root="/tmp/aic-trtllm-a2a-${SLURM_JOB_ID}"' in source
    assert "benchmark_status=$?" in source
    assert "all partial rows and rank evidence preserved" in source
    assert "failure_evidence/${SYSTEM}/trtllm" in source
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
