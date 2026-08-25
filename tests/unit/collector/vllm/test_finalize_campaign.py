# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest
import yaml

from collector import provenance
from collector.wideep.sglang.collect_moe_a2a import _build_moe_a2a_row
from collector.wideep.vllm import finalize_campaign as campaign
from collector.wideep.vllm.collect_moe_a2a import (
    TARGET_VLLM_SOURCE_COMMIT,
    case_plan_ids,
)

pytestmark = pytest.mark.unit


def _runtime(*, fabric_identity: str) -> dict:
    return {
        "framework": "vllm",
        "version": "0.24.0",
        "image": "vllm/vllm-openai:v0.24.0",
        "image_digest": "sha256:" + "1" * 64,
        "source_commit": TARGET_VLLM_SOURCE_COMMIT,
        "abi": {
            **campaign.REQUIRED_ABI,
            "slurm_topology_verified": "true",
            "fabric_identity": fabric_identity,
            "system": "h200_sxm",
            "gpu_name": "NVIDIA H200",
            "compute_capability": "9.0",
            "cross_node_nvlink_capable": "false",
            "rdma_device_count": "8",
        },
    }


def _write_job(root: Path, *, node_num: int, ep_size: int, backend: str) -> Path:
    job_dir = root / f"{node_num}n" / backend
    job_dir.mkdir(parents=True)
    cases = campaign._expected_cases(ep_size=ep_size, node_num=node_num, backend=backend)
    rows = []
    for case in cases:
        sms = case.sms if case.sms is not None else 24
        for phase in ("combine", "dispatch"):
            payload = _build_moe_a2a_row(
                comm_backend=backend,
                phase=phase,
                ep_size=ep_size,
                node_num=node_num,
                shape=case.shape,
                num_tokens=case.num_tokens,
                sms=sms,
                transmit_us=10.0,
                notify_us=2.0,
            )
            rows.append(
                {
                    "framework": "vLLM",
                    "version": "0.24.0",
                    "device": "NVIDIA H200",
                    "op_name": "moe_a2a",
                    "kernel_source": "deepep",
                    **payload,
                }
            )
    frame = pd.DataFrame(rows, columns=campaign.ROW_COLUMNS)
    frame.to_parquet(job_dir / "moe_a2a_perf.parquet", index=False)
    provenance.write_collection_meta(
        job_dir,
        _runtime(fabric_identity=f"leaf-{node_num}"),
        {
            "moe_a2a_perf": {
                "collector_ref": "deadbeef",
                "collector_hash": "sha256:" + "2" * 64,
                "case_plan_hash": provenance.case_plan_hash(
                    case_plan_ids(cases, world_size=ep_size, node_num=node_num)
                ),
                "collected_at": date.today().isoformat(),
                "rows": len(frame),
                "status": "complete",
            }
        },
    )
    return job_dir


def _six_h200_jobs(root: Path) -> list[Path]:
    return [
        _write_job(root, node_num=node_num, ep_size=ep_size, backend=backend)
        for node_num, ep_size in ((2, 16), (4, 32))
        for backend in campaign.BACKENDS
    ]


def test_merge_campaign_requires_and_validates_all_six_formal_jobs(tmp_path):
    jobs = _six_h200_jobs(tmp_path / "jobs")
    output = tmp_path / "published"
    output.mkdir()
    (output / "collection_meta.yaml").write_text(
        "schema_version: 1\nruntime:\n  framework: vllm\n  version: 0.24.0\n"
        "tables:\n  custom_allreduce_perf:\n    status: complete\n",
        encoding="utf-8",
    )
    checksums = tmp_path / "evidence" / "artifact_checksums.json"

    parquet_path, sidecar_path = campaign.merge_campaign(
        jobs,
        system="h200_sxm",
        output_dir=output,
        checksum_output=checksums,
    )

    merged = pd.read_parquet(parquet_path)
    assert len(merged) == sum(len(pd.read_parquet(job / "moe_a2a_perf.parquet")) for job in jobs)
    assert set(zip(merged["node_num"], merged["ep_size"], strict=True)) == {(2, 16), (4, 32)}
    assert set(merged["comm_backend"]) == set(campaign.BACKENDS)
    meta = yaml.safe_load(sidecar_path.read_text(encoding="utf-8"))
    assert meta["tables"]["custom_allreduce_perf"] == {"status": "complete"}
    assert meta["tables"]["moe_a2a_perf"]["rows"] == len(merged)
    assert meta["runtime"]["abi"]["campaign_system"] == "h200_sxm"
    assert checksums.is_file()


def test_merge_campaign_rejects_missing_backend_job(tmp_path):
    jobs = _six_h200_jobs(tmp_path / "jobs")

    with pytest.raises(campaign.CampaignValidationError, match="requires exactly"):
        campaign.merge_campaign(jobs[:-1], system="h200_sxm", output_dir=tmp_path / "published")


def test_validate_job_rejects_classified_failure(tmp_path):
    job = _write_job(tmp_path, node_num=2, ep_size=16, backend="deepep_ht")
    (job / "errors_moe_a2a_vllm.rank0.json").write_text('[{"error": "boom"}]', encoding="utf-8")

    with pytest.raises(campaign.CampaignValidationError, match="classified failures"):
        campaign.validate_job_dir(job, system="h200_sxm")


def test_validate_job_rejects_wrong_system_identity(tmp_path):
    job = _write_job(tmp_path, node_num=2, ep_size=16, backend="deepep_ht")

    with pytest.raises(campaign.CampaignValidationError, match="runtime system is not b200_sxm"):
        campaign.validate_job_dir(job, system="b200_sxm")
