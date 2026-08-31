# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
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


def _runtime(*, fabric_identity: str, backend: str, ep_size: int, system: str) -> dict:
    gpu_token, compute_capability = campaign.SYSTEM_GPU_IDENTITIES[system]
    scaleup_ranks = 4 if system in ("gb200", "gb300") else 8
    abi = campaign.VLLM_RUNTIME.abi_for_backend(backend) | {
        "slurm_topology_verified": "true",
        "fabric_identity": fabric_identity,
        "system": system,
        "gpu_name": f"NVIDIA {gpu_token}",
        "compute_capability": compute_capability,
        "cross_node_nvlink_capable": "runtime_probe_required",
        "rdma_device_count": "8",
    }
    if system in ("gb200", "gb300"):
        abi |= {
            "deep_ep_overlay_wheel_sha256": "4" * 64,
            "deep_ep_patch_sha256": campaign._sha256(campaign.LEGACY_NVL4_PATCH),
        }
    elif system == "b300_sxm":
        abi["deep_ep_overlay_wheel_sha256"] = "4" * 64
    if backend == "deepep_v2":
        abi |= {
            "deep_ep_topology_source": "nccl_lsa",
            "deep_ep_overlay_wheel_sha256": "3" * 64,
        }
        capability = {
            "backend": backend,
            "topology_source": "nccl_lsa",
            "num_scaleout_ranks": str(ep_size // scaleup_ranks),
            "num_scaleup_ranks": str(scaleup_ranks),
            "num_rdma_ranks": str(ep_size // scaleup_ranks),
            "num_nvlink_ranks": str(scaleup_ranks),
        }
        live_abi = {"deep_ep_api": "ElasticBuffer", "nccl": "2.30.4"}
    else:
        abi["deep_ep_scaleup_ranks"] = str(scaleup_ranks)
        capability = {
            "backend": backend,
            "topology_source": "legacy_compile_time",
            "num_scaleout_ranks": str(ep_size // scaleup_ranks),
            "num_scaleup_ranks": str(scaleup_ranks),
        }
        live_abi = {"deep_ep_api": "Buffer"}
    return {
        "framework": campaign.VLLM_RUNTIME.framework,
        "version": "0.24.0",
        "image": campaign.VLLM_RUNTIME.image(),
        "image_variant": "linux/arm64" if system in ("gb200", "gb300") else "linux/amd64",
        "image_digest": "sha256:" + "1" * 64,
        "source_commit": TARGET_VLLM_SOURCE_COMMIT,
        "abi": abi,
        "backend_capability": capability,
        "live_abi": live_abi,
    }


def _write_job(
    root: Path,
    *,
    node_num: int,
    ep_size: int,
    backend: str,
    system: str = "h200_sxm",
) -> Path:
    job_dir = root / f"{node_num}n" / backend
    job_dir.mkdir(parents=True)
    cases = campaign._expected_cases(ep_size=ep_size, node_num=node_num, backend=backend)
    rows = []
    for case in cases:
        sms = case.sms if case.sms is not None else 24
        for phase in ("combine", "dispatch"):
            payload = _build_moe_a2a_row(
                comm_backend=case.persisted_backend,
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
                    "device": f"NVIDIA {campaign.SYSTEM_GPU_IDENTITIES[system][0]}",
                    "op_name": "moe_a2a",
                    "kernel_source": "deepep",
                    **payload,
                }
            )
    frame = pd.DataFrame(rows, columns=campaign.ROW_COLUMNS)
    frame.to_parquet(job_dir / "moe_a2a_perf.parquet", index=False)
    provenance.write_collection_meta(
        job_dir,
        _runtime(
            fabric_identity=f"leaf-{node_num}",
            backend=backend,
            ep_size=ep_size,
            system=system,
        ),
        {
            "moe_a2a_perf": {
                "collector_ref": "deadbeef",
                "collector_hash": "sha256:" + "2" * 64,
                "case_plan_hash": provenance.case_plan_hash(
                    case_plan_ids(cases, world_size=ep_size, node_num=node_num)
                ),
                "collected_at": date.today().isoformat(),
                "rows": len(frame),
                "classified_failures": 0,
                "status": "complete",
            }
        },
    )
    artifacts = (job_dir / "moe_a2a_perf.parquet", job_dir / "collection_meta.yaml")
    (job_dir / "artifact_checksums.json").write_text(
        json.dumps({path.name: campaign._sha256(path) for path in artifacts}),
        encoding="utf-8",
    )
    (job_dir / "SUCCESS").touch()
    return job_dir


def _h100_jobs(root: Path) -> list[Path]:
    return [
        _write_job(root, node_num=1, ep_size=8, backend=backend, system="h100_sxm")
        for backend in campaign.FORMAL_BACKENDS_BY_SYSTEM["h100_sxm"]
    ]


def test_merge_campaign_requires_and_validates_all_three_v2_capable_jobs(tmp_path):
    jobs = _h100_jobs(tmp_path / "jobs")
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
        system="h100_sxm",
        output_dir=output,
        checksum_output=checksums,
    )

    merged = pd.read_parquet(parquet_path)
    assert len(merged) == sum(len(pd.read_parquet(job / "moe_a2a_perf.parquet")) for job in jobs)
    assert set(zip(merged["node_num"], merged["ep_size"], strict=True)) == {(1, 8)}
    assert set(merged["comm_backend"]) == {
        "deepep_ht",
        "deepep_ll",
        "deepep_v2_context",
        "deepep_v2_generation",
    }
    meta = yaml.safe_load(sidecar_path.read_text(encoding="utf-8"))
    assert meta["tables"]["custom_allreduce_perf"] == {"status": "complete"}
    assert meta["tables"]["moe_a2a_perf"]["rows"] == len(merged)
    assert meta["runtime"]["abi"]["campaign_system"] == "h100_sxm"
    assert meta["runtime"]["backend_abis"]["deepep_v2"]["deep_ep"] == "b306af06afd412c88e51e71802951606e40b7358"
    assert set(meta["runtime"]["backend_capabilities"]["deepep_v2"]) == {"1n_ep8"}
    assert checksums.is_file()


def test_clean_republish_removes_stale_error_files_and_checksum_entries(tmp_path):
    jobs = [
        _write_job(
            tmp_path / "jobs",
            node_num=1,
            ep_size=8,
            backend=backend,
            system="h100_sxm",
        )
        for backend in campaign.BACKENDS
    ]
    output = tmp_path / "published"
    output.mkdir()
    stale_error = output / "errors_moe_a2a_vllm.rank99.json"
    stale_error.write_text('[{"classification": "unexpected"}]\n', encoding="utf-8")
    checksums = tmp_path / "evidence" / "artifact_checksums.json"
    checksums.parent.mkdir()
    checksums.write_text(
        json.dumps({stale_error.name: campaign._sha256(stale_error)}),
        encoding="utf-8",
    )

    _, sidecar = campaign.merge_campaign(
        jobs,
        system="h100_sxm",
        output_dir=output,
        checksum_output=checksums,
    )

    assert not list(output.glob("errors_moe_a2a_vllm.rank*.json"))
    checksum_payload = json.loads(checksums.read_text(encoding="utf-8"))
    assert stale_error.name not in checksum_payload
    assert set(checksum_payload) == {"moe_a2a_perf.parquet", "collection_meta.yaml"}
    meta = yaml.safe_load(sidecar.read_text(encoding="utf-8"))
    assert meta["tables"]["moe_a2a_perf"]["classified_failures"] == 0


def test_merge_campaign_rejects_missing_backend_job(tmp_path):
    jobs = _h100_jobs(tmp_path / "jobs")

    with pytest.raises(campaign.CampaignValidationError, match="requires exactly"):
        campaign.merge_campaign(jobs[:-1], system="h100_sxm", output_dir=tmp_path / "published")


def test_capability_failed_system_requires_only_legacy_backends(tmp_path):
    jobs = [
        _write_job(tmp_path / "jobs", node_num=1, ep_size=4, backend=backend, system="gb200")
        for backend in campaign.FORMAL_BACKENDS_BY_SYSTEM["gb200"]
    ]

    parquet, sidecar = campaign.merge_campaign(
        jobs,
        system="gb200",
        output_dir=tmp_path / "published",
    )

    frame = pd.read_parquet(parquet)
    assert set(frame["comm_backend"]) == {"deepep_ht", "deepep_ll"}
    meta = yaml.safe_load(sidecar.read_text(encoding="utf-8"))
    assert set(meta["runtime"]["backend_abis"]) == {"deepep_ht", "deepep_ll"}


def test_capability_failed_system_rejects_extra_v2_job(tmp_path):
    jobs = [
        _write_job(tmp_path / "jobs", node_num=1, ep_size=8, backend=backend, system="h200_sxm")
        for backend in campaign.BACKENDS
    ]

    with pytest.raises(campaign.CampaignValidationError, match="requires exactly"):
        campaign.merge_campaign(jobs, system="h200_sxm", output_dir=tmp_path / "published")


def test_validate_job_requires_success_exact_checksums_and_recomputed_plan(tmp_path):
    job = _write_job(tmp_path, node_num=1, ep_size=8, backend="deepep_ht")
    (job / "SUCCESS").unlink()
    with pytest.raises(campaign.CampaignValidationError, match="missing SUCCESS"):
        campaign.validate_job_dir(job, system="h200_sxm")
    (job / "SUCCESS").touch()
    checksums = json.loads((job / "artifact_checksums.json").read_text())
    checksums["unexpected.log"] = "0" * 64
    (job / "artifact_checksums.json").write_text(json.dumps(checksums))
    with pytest.raises(campaign.CampaignValidationError, match="must contain exactly"):
        campaign.validate_job_dir(job, system="h200_sxm")
    checksums.pop("unexpected.log")
    (job / "artifact_checksums.json").write_text(json.dumps(checksums))
    sidecar = yaml.safe_load((job / "collection_meta.yaml").read_text())
    sidecar["tables"]["moe_a2a_perf"]["case_plan_hash"] = "sha256:" + "0" * 64
    (job / "collection_meta.yaml").write_text(yaml.safe_dump(sidecar, sort_keys=False))
    checksums["collection_meta.yaml"] = campaign._sha256(job / "collection_meta.yaml")
    (job / "artifact_checksums.json").write_text(json.dumps(checksums))
    with pytest.raises(campaign.CampaignValidationError, match="case_plan_hash mismatch"):
        campaign.validate_job_dir(job, system="h200_sxm")


def test_validate_job_rejects_classified_failure(tmp_path):
    job = _write_job(tmp_path, node_num=1, ep_size=8, backend="deepep_ht")
    (job / "errors_moe_a2a_vllm.rank0.json").write_text('[{"error": "boom"}]', encoding="utf-8")

    with pytest.raises(campaign.CampaignValidationError, match="unexpected failures"):
        campaign.validate_job_dir(job, system="h200_sxm")


def test_validate_job_rejects_kimi_k3_ep4_failure_even_when_recorded(tmp_path):
    job = _write_job(
        tmp_path,
        node_num=1,
        ep_size=4,
        backend="deepep_ht",
        system="gb200",
    )
    (job / "errors_moe_a2a_vllm.rank0.json").write_text(
        json.dumps(
            [
                {
                    "classification": "unexpected",
                    "error": "Kimi-K3 exceeds the pinned DeepEP local-expert limit",
                }
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(campaign.CampaignValidationError, match="unexpected failures"):
        campaign.validate_job_dir(job, system="gb200")


def test_validate_job_rejects_wrong_system_identity(tmp_path):
    job = _write_job(tmp_path, node_num=1, ep_size=8, backend="deepep_ht")

    with pytest.raises(campaign.CampaignValidationError, match="does not match b200_sxm"):
        campaign.validate_job_dir(job, system="b200_sxm")
