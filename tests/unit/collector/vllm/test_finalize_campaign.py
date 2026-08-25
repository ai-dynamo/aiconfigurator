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
    HT_BUFFER_SIZE_BYTES,
    HT_SMS,
    KNOWN_DEEPEP_HT_EP4_LIMIT_ERROR,
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
        "framework": "vllm",
        "version": "0.24.0",
        "image": "vllm/vllm-openai:v0.24.0",
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
    carry_known_kimi_limit: bool = False,
) -> Path:
    job_dir = root / f"{node_num}n" / backend
    job_dir.mkdir(parents=True)
    cases = campaign._expected_cases(ep_size=ep_size, node_num=node_num, backend=backend)
    known_failures = [
        case
        for case in cases
        if carry_known_kimi_limit
        and case.shape.hidden_size == 3584
        and case.shape.topk == 16
        and case.shape.num_experts == 896
    ]
    rows = []
    for case in cases:
        if case in known_failures:
            continue
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
                "classified_failures": len(known_failures),
                "status": "complete",
            }
        },
    )
    if known_failures:
        for rank in range(ep_size):
            records = [
                {
                    "module": "collector.wideep.vllm.collect_moe_a2a",
                    "op": "moe_a2a",
                    "classification": "known_framework_limit",
                    "error_type": "RuntimeError",
                    "error": f"DeepEP assertion: {KNOWN_DEEPEP_HT_EP4_LIMIT_ERROR}",
                    "rank": rank,
                    "case": {
                        "comm_backend": case.comm_backend,
                        "inference_phase": case.inference_phase,
                        "ep_size": ep_size,
                        "node_num": node_num,
                        "hidden_size": case.shape.hidden_size,
                        "topk": case.shape.topk,
                        "num_experts": case.shape.num_experts,
                        "num_tokens": case.num_tokens,
                        "sms": HT_SMS,
                        "capacity": HT_BUFFER_SIZE_BYTES,
                    },
                }
                for case in known_failures
            ]
            (job_dir / f"errors_moe_a2a_vllm.rank{rank}.json").write_text(
                json.dumps(records, indent=2),
                encoding="utf-8",
            )
        artifact_paths = [
            job_dir / "moe_a2a_perf.parquet",
            job_dir / "collection_meta.yaml",
            *sorted(job_dir.glob("errors_moe_a2a_vllm.rank*.json")),
        ]
        (job_dir / "artifact_checksums.json").write_text(
            json.dumps({path.name: campaign._sha256(path) for path in artifact_paths}, sort_keys=True),
            encoding="utf-8",
        )
    return job_dir


def _three_h200_jobs(root: Path) -> list[Path]:
    return [_write_job(root, node_num=1, ep_size=8, backend=backend) for backend in campaign.BACKENDS]


def test_merge_campaign_requires_and_validates_all_three_formal_jobs(tmp_path):
    jobs = _three_h200_jobs(tmp_path / "jobs")
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
    assert set(zip(merged["node_num"], merged["ep_size"], strict=True)) == {(1, 8)}
    assert set(merged["comm_backend"]) == set(campaign.BACKENDS)
    meta = yaml.safe_load(sidecar_path.read_text(encoding="utf-8"))
    assert meta["tables"]["custom_allreduce_perf"] == {"status": "complete"}
    assert meta["tables"]["moe_a2a_perf"]["rows"] == len(merged)
    assert meta["runtime"]["abi"]["campaign_system"] == "h200_sxm"
    assert meta["runtime"]["backend_abis"]["deepep_v2"]["deep_ep"] == "b306af06afd412c88e51e71802951606e40b7358"
    assert set(meta["runtime"]["backend_capabilities"]["deepep_v2"]) == {"1n_ep8"}
    assert checksums.is_file()


def test_merge_campaign_ships_and_checksums_original_known_failure_records(tmp_path):
    jobs = [
        _write_job(
            tmp_path / "jobs",
            node_num=1,
            ep_size=4,
            backend=backend,
            system="gb200",
            carry_known_kimi_limit=backend == "deepep_ht",
        )
        for backend in campaign.BACKENDS
    ]
    output = tmp_path / "published"
    checksums = tmp_path / "evidence" / "artifact_checksums.json"

    _, sidecar = campaign.merge_campaign(
        jobs,
        system="gb200",
        output_dir=output,
        checksum_output=checksums,
    )

    error_paths = sorted(output.glob("errors_moe_a2a_vllm.rank*.json"))
    assert len(error_paths) == 4
    checksum_payload = json.loads(checksums.read_text(encoding="utf-8"))
    assert all(checksum_payload[path.name] == campaign._sha256(path) for path in error_paths)
    meta = yaml.safe_load(sidecar.read_text(encoding="utf-8"))
    assert meta["tables"]["moe_a2a_perf"]["classified_failures"] == 13


def test_merge_campaign_rejects_missing_backend_job(tmp_path):
    jobs = _three_h200_jobs(tmp_path / "jobs")

    with pytest.raises(campaign.CampaignValidationError, match="requires exactly"):
        campaign.merge_campaign(jobs[:-1], system="h200_sxm", output_dir=tmp_path / "published")


def test_validate_job_rejects_classified_failure(tmp_path):
    job = _write_job(tmp_path, node_num=1, ep_size=8, backend="deepep_ht")
    (job / "errors_moe_a2a_vllm.rank0.json").write_text('[{"error": "boom"}]', encoding="utf-8")

    with pytest.raises(campaign.CampaignValidationError, match="classified failures"):
        campaign.validate_job_dir(job, system="h200_sxm")


def test_validate_job_accepts_exact_kimi_k3_deepep_ep4_limit(tmp_path):
    job = _write_job(
        tmp_path,
        node_num=1,
        ep_size=4,
        backend="deepep_ht",
        system="gb200",
        carry_known_kimi_limit=True,
    )

    validated = campaign.validate_job_dir(job, system="gb200")

    assert len(validated.frame) == 156
    assert validated.table["classified_failures"] == 13
    assert len(list(job.glob("errors_moe_a2a_vllm.rank*.json"))) == 4


def test_validate_job_rejects_a_variant_of_the_known_kimi_k3_limit(tmp_path):
    job = _write_job(
        tmp_path,
        node_num=1,
        ep_size=4,
        backend="deepep_ht",
        system="gb200",
        carry_known_kimi_limit=True,
    )
    error_path = job / "errors_moe_a2a_vllm.rank0.json"
    records = json.loads(error_path.read_text(encoding="utf-8"))
    records[0]["classification"] = "unexpected"
    error_path.write_text(json.dumps(records, indent=2), encoding="utf-8")

    with pytest.raises(campaign.CampaignValidationError, match="outside the accepted Kimi-K3"):
        campaign.validate_job_dir(job, system="gb200")


def test_validate_job_rejects_wrong_system_identity(tmp_path):
    job = _write_job(tmp_path, node_num=1, ep_size=8, backend="deepep_ht")

    with pytest.raises(campaign.CampaignValidationError, match="does not match b200_sxm"):
        campaign.validate_job_dir(job, system="b200_sxm")
