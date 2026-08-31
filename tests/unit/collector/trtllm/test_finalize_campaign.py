# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest
import yaml

from collector import provenance
from collector.wideep.sglang.collect_moe_a2a import _build_moe_a2a_row
from collector.wideep.trtllm import finalize_campaign as campaign
from collector.wideep.trtllm.collect_moe_a2a import case_plan_ids

pytestmark = pytest.mark.unit


def _write_job(
    root: Path,
    *,
    backend: str,
    system: str = "h200_sxm",
    partial: bool = False,
    seed_provenance: dict[str, str] | None = None,
) -> Path:
    ep_size = campaign.SYSTEM_LAYOUTS[system][1]
    path = root / backend
    path.mkdir(parents=True)
    cases = campaign._cases(system, backend)
    rows = []
    emitted_cases = cases[:-1] if partial else cases
    for case in emitted_cases:
        for phase in ("combine", "dispatch"):
            dtype = campaign.communication_dtype_for(
                system=system,
                backend=backend,
                model_quantization=case.quant.comm_dtype,
                communication_phase=phase,
            )
            payload = _build_moe_a2a_row(
                comm_backend=backend,
                phase=phase,
                comm_dtype=dtype,
                ep_size=ep_size,
                node_num=1,
                shape=case.shape,
                num_tokens=case.num_tokens,
                sms=0,
                transmit_us=10.0,
                notify_us=0.0,
            )
            rows.append(
                {
                    "framework": "TRTLLM",
                    "version": campaign.EXPECTED_VERSION,
                    "device": f"NVIDIA {campaign.SYSTEM_GPU_IDENTITIES[system][0]}",
                    "op_name": "moe_a2a",
                    "kernel_source": "deepep",
                    **payload,
                }
            )
    frame = pd.DataFrame(rows, columns=campaign.ROW_COLUMNS)
    frame.to_parquet(path / "moe_a2a_perf.parquet", index=False)
    configured_image, configured_digest = campaign.RUNTIME.image().split("@", 1)
    provenance.write_collection_meta(
        path,
        {
            "framework": campaign.RUNTIME.framework,
            "version": campaign.EXPECTED_VERSION,
            "image": configured_image,
            "image_digest": configured_digest,
            "source_commit": campaign.TARGET_SOURCE_COMMIT,
            "abi": campaign.RUNTIME.abi,
        },
        {
            "moe_a2a_perf": {
                "collector_ref": "a" * 40,
                "collector_hash": "sha256:" + "b" * 64,
                "case_plan_hash": provenance.case_plan_hash(case_plan_ids(cases)),
                "collected_at": date.today().isoformat(),
                "rows": len(frame),
                "status": provenance.STATUS_COMPLETE,
            }
        },
    )
    evidence = {
        "system": system,
        "node": "test-node",
        "configured_image_digest": configured_digest,
        "observed_image_digest": "sha256:" + "c" * 64,
        "image_variant": "linux/arm64" if system in ("gb200", "gb300") else "linux/amd64",
        "wheel_sha256": "d" * 64,
        "cuda_arches": campaign.SYSTEM_CUDA_ARCHES[system],
        "collector_ref": "a" * 40,
        "slurm_topology_verified": True,
    }
    if seed_provenance is not None:
        evidence["seed_provenance"] = seed_provenance
    (path / "runtime_evidence.json").write_text(json.dumps(evidence), encoding="utf-8")
    if partial:
        failed_case = cases[-1]
        failure = {
            "module": "collector.wideep.trtllm.collect_moe_a2a",
            "op": "moe_a2a",
            "classification": "unexpected",
            "error_type": "RuntimeError",
            "case": {
                "comm_backend": failed_case.comm_backend,
                "comm_dtype": failed_case.quant.comm_dtype,
                "inference_phase": failed_case.inference_phase,
                "ep_size": failed_case.ep_size,
                "node_num": failed_case.node_num,
                "hidden_size": failed_case.shape.hidden_size,
                "topk": failed_case.shape.topk,
                "num_experts": failed_case.shape.num_experts,
                "num_tokens": failed_case.num_tokens,
                "sms": failed_case.sms,
            },
            "error": "known observed kernel limit",
        }
        for rank in range(ep_size):
            rank_failure = failure | {"rank": rank}
            (path / f"errors_moe_a2a_trtllm.rank{rank}.json").write_text(json.dumps([rank_failure]), encoding="utf-8")
        (path / "job_failure.json").write_text(json.dumps({"benchmark_status": 1}), encoding="utf-8")
    artifact_names = ["moe_a2a_perf.parquet", "collection_meta.yaml", "runtime_evidence.json"]
    artifact_names.extend(error.name for error in sorted(path.glob("errors_moe_a2a_trtllm.rank*.json")))
    if partial:
        artifact_names.append("job_failure.json")
    checksums = {name: hashlib.sha256((path / name).read_bytes()).hexdigest() for name in artifact_names}
    (path / "artifact_checksums.json").write_text(json.dumps(checksums), encoding="utf-8")
    if not partial:
        (path / "SUCCESS").touch()
    return path


def test_merge_complete_ht_and_ll_campaign(tmp_path):
    jobs = [_write_job(tmp_path / "jobs", backend=backend) for backend in campaign.BACKENDS]
    output = tmp_path / "published"
    parquet, sidecar = campaign.merge_campaign(jobs, system="h200_sxm", output_dir=output)
    frame = pd.read_parquet(parquet)
    assert set(frame["comm_backend"]) == set(campaign.BACKENDS)
    assert len(frame) == sum(len(pd.read_parquet(job / "moe_a2a_perf.parquet")) for job in jobs)
    meta = yaml.safe_load(sidecar.read_text(encoding="utf-8"))
    assert meta["tables"]["moe_a2a_perf"]["status"] == provenance.STATUS_COMPLETE
    assert meta["tables"]["moe_a2a_perf"]["classified_failures"] == 0
    assert meta["runtime"]["image"] == campaign.RUNTIME.image()
    assert meta["runtime"]["image_digest"] == "sha256:" + "c" * 64
    assert meta["runtime"]["abi"]["source_wheel_sha256"] == "d" * 64


def test_partial_input_is_rejected_by_default_and_explicitly_partial_when_requested(tmp_path):
    jobs = [
        _write_job(tmp_path / "jobs", backend=backend, partial=backend == campaign.COMM_BACKEND_HT)
        for backend in campaign.BACKENDS
    ]
    with pytest.raises(campaign.CampaignValidationError, match="incomplete formal input"):
        campaign.merge_campaign(jobs, system="h200_sxm", output_dir=tmp_path / "formal")
    _, sidecar = campaign.merge_campaign(
        jobs,
        system="h200_sxm",
        output_dir=tmp_path / "evidence",
        allow_partial_evidence=True,
    )
    meta = yaml.safe_load(sidecar.read_text(encoding="utf-8"))
    assert meta["tables"]["moe_a2a_perf"]["status"] == provenance.STATUS_PARTIAL
    assert meta["tables"]["moe_a2a_perf"]["classified_failures"] == 1
    assert (tmp_path / "evidence" / f"errors_moe_a2a_trtllm.{campaign.COMM_BACKEND_HT}.json").is_file()


@pytest.mark.parametrize(
    ("system", "backend"),
    [(system, backend) for system in campaign.SYSTEM_LAYOUTS for backend in campaign.BACKENDS],
)
def test_collector_and_finalizer_use_the_same_system_plan(system, backend):
    ep_size = campaign.SYSTEM_LAYOUTS[system][1]
    collected = campaign.build_case_plan(
        shapes=campaign.get_moe_a2a_shapes(required_expert_parallel_size=ep_size),
        token_grid=campaign.get_moe_a2a_token_grid(),
        ep_size=ep_size,
        node_num=1,
        modes=(backend,),
        system=system,
    )
    assert campaign._cases(system, backend) == collected


@pytest.mark.parametrize(
    ("system", "ht_cases", "ll_cases", "ht_rows", "ll_rows"),
    [
        ("gb200", 156, 252, 312, 504),
        ("gb300", 156, 252, 312, 504),
        ("b200_sxm", 156, 252, 312, 504),
        ("b300_sxm", 156, 252, 312, 504),
        ("h100_sxm", 156, 168, 312, 336),
        ("h200_sxm", 156, 168, 312, 336),
    ],
)
def test_exact_post_fix_case_and_row_matrix(system, ht_cases, ll_cases, ht_rows, ll_rows):
    assert len(campaign._cases(system, campaign.COMM_BACKEND_HT)) == ht_cases
    assert len(campaign._cases(system, campaign.COMM_BACKEND_LL)) == ll_cases
    assert len(campaign._expected_keys(system, campaign.COMM_BACKEND_HT)) == ht_rows
    assert len(campaign._expected_keys(system, campaign.COMM_BACKEND_LL)) == ll_rows


def test_partial_evidence_requires_every_rank_to_agree(tmp_path):
    job = _write_job(tmp_path / "jobs", backend=campaign.COMM_BACKEND_HT, partial=True)
    (job / "errors_moe_a2a_trtllm.rank7.json").unlink()

    with pytest.raises(campaign.CampaignValidationError, match="agree across every rank"):
        campaign.validate_job_dir(job, system="h200_sxm", allow_partial_evidence=True)


def test_unrelated_failure_cannot_authorize_missing_rows(tmp_path):
    job = _write_job(tmp_path / "jobs", backend=campaign.COMM_BACKEND_HT, partial=True)
    unrelated = campaign._cases("h200_sxm", campaign.COMM_BACKEND_HT)[0]
    for error_path in job.glob("errors_moe_a2a_trtllm.rank*.json"):
        [failure] = json.loads(error_path.read_text(encoding="utf-8"))
        failure["case"] |= {
            "comm_dtype": unrelated.quant.comm_dtype,
            "hidden_size": unrelated.shape.hidden_size,
            "topk": unrelated.shape.topk,
            "num_experts": unrelated.shape.num_experts,
            "num_tokens": unrelated.num_tokens,
        }
        error_path.write_text(json.dumps([failure]), encoding="utf-8")

    with pytest.raises(campaign.CampaignValidationError, match="does not cover any missing physical row"):
        campaign.validate_job_dir(job, system="h200_sxm", allow_partial_evidence=True)


def test_fully_observed_redundant_failure_is_rejected(tmp_path):
    system = "h200_sxm"
    backend = campaign.COMM_BACKEND_HT
    job = _write_job(tmp_path / "jobs", backend=backend, partial=True)
    observed_case = campaign._cases(system, backend)[0]
    for error_path in job.glob("errors_moe_a2a_trtllm.rank*.json"):
        failures = json.loads(error_path.read_text(encoding="utf-8"))
        redundant = failures[0] | {
            "case": {
                "comm_backend": observed_case.comm_backend,
                "comm_dtype": observed_case.quant.comm_dtype,
                "inference_phase": observed_case.inference_phase,
                "ep_size": observed_case.ep_size,
                "node_num": observed_case.node_num,
                "hidden_size": observed_case.shape.hidden_size,
                "topk": observed_case.shape.topk,
                "num_experts": observed_case.shape.num_experts,
                "num_tokens": observed_case.num_tokens,
                "sms": observed_case.sms,
            }
        }
        error_path.write_text(json.dumps([*failures, redundant]), encoding="utf-8")

    with pytest.raises(campaign.CampaignValidationError, match="does not cover any missing physical row"):
        campaign.validate_job_dir(job, system=system, allow_partial_evidence=True)


def test_nvfp4_failure_maps_to_phase_specific_physical_dtypes():
    identity = (
        campaign.COMM_BACKEND_LL,
        "nvfp4",
        8,
        1,
        7168,
        8,
        256,
        2,
        0,
    )

    assert campaign._failure_physical_keys(identity, system="gb300") == {
        (campaign.COMM_BACKEND_LL, "combine", "fp4", 8, 1, 7168, 8, 256, 2, 0),
        (campaign.COMM_BACKEND_LL, "dispatch", "nvfp4", 8, 1, 7168, 8, 256, 2, 0),
    }


def test_failure_identity_accounts_only_for_phase_rows_still_missing(tmp_path):
    system = "h200_sxm"
    backend = campaign.COMM_BACKEND_HT
    job = _write_job(tmp_path / "jobs", backend=backend, partial=True)
    failed_case = campaign._cases(system, backend)[-1]
    frame = pd.read_parquet(job / "moe_a2a_perf.parquet")
    payload = _build_moe_a2a_row(
        comm_backend=backend,
        phase="combine",
        comm_dtype=campaign.communication_dtype_for(
            system=system,
            backend=backend,
            model_quantization=failed_case.quant.comm_dtype,
            communication_phase="combine",
        ),
        ep_size=failed_case.ep_size,
        node_num=1,
        shape=failed_case.shape,
        num_tokens=failed_case.num_tokens,
        sms=0,
        transmit_us=10.0,
        notify_us=0.0,
    )
    frame.loc[len(frame)] = {
        "framework": "TRTLLM",
        "version": campaign.EXPECTED_VERSION,
        "device": "NVIDIA H200",
        "op_name": "moe_a2a",
        "kernel_source": "deepep",
        **payload,
    }
    frame.to_parquet(job / "moe_a2a_perf.parquet", index=False)
    meta_path = job / "collection_meta.yaml"
    meta = yaml.safe_load(meta_path.read_text(encoding="utf-8"))
    meta["tables"]["moe_a2a_perf"]["rows"] = len(frame)
    meta_path.write_text(yaml.safe_dump(meta, sort_keys=False), encoding="utf-8")
    artifact_names = ["moe_a2a_perf.parquet", "collection_meta.yaml", "runtime_evidence.json", "job_failure.json"]
    artifact_names.extend(error.name for error in sorted(job.glob("errors_moe_a2a_trtllm.rank*.json")))
    checksums = {name: hashlib.sha256((job / name).read_bytes()).hexdigest() for name in artifact_names}
    (job / "artifact_checksums.json").write_text(json.dumps(checksums), encoding="utf-8")

    validated = campaign.validate_job_dir(job, system=system, allow_partial_evidence=True)
    assert validated.classified_failures == 1


def test_merge_requires_both_backends(tmp_path):
    job = _write_job(tmp_path / "jobs", backend=campaign.COMM_BACKEND_HT)
    with pytest.raises(campaign.CampaignValidationError, match="exactly one job"):
        campaign.merge_campaign([job], system="h200_sxm", output_dir=tmp_path / "published")


def test_validate_job_requires_success_exact_checksums_and_recomputed_plan(tmp_path):
    job = _write_job(tmp_path, backend=campaign.COMM_BACKEND_HT)
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


def test_merge_preserves_matching_seed_provenance(tmp_path):
    seed = {
        "mode": "runtime",
        "source_system": "h100_sxm",
        "source_image_sha256": "e" * 64,
        "source_image_meta_sha256": "f" * 64,
        "source_image_digest": "sha256:" + "c" * 64,
        "source_wheel_sha256": "d" * 64,
        "source_wheel_meta_sha256": "a" * 64,
        "cuda_arches": "90-real",
    }
    jobs = [_write_job(tmp_path / "jobs", backend=backend, seed_provenance=seed) for backend in campaign.BACKENDS]
    _, sidecar = campaign.merge_campaign(jobs, system="h200_sxm", output_dir=tmp_path / "published")
    meta = yaml.safe_load(sidecar.read_text(encoding="utf-8"))
    assert json.loads(meta["runtime"]["abi"]["runtime_seed_provenance"]) == seed


def test_merge_rejects_different_seed_provenance(tmp_path):
    common = {
        "mode": "image",
        "source_system": "h100_sxm",
        "source_image_sha256": "e" * 64,
        "source_image_meta_sha256": "f" * 64,
        "source_image_digest": "sha256:" + "c" * 64,
    }
    jobs = [
        _write_job(tmp_path / "jobs", backend=campaign.BACKENDS[0], seed_provenance=common),
        _write_job(
            tmp_path / "jobs",
            backend=campaign.BACKENDS[1],
            seed_provenance=common | {"source_image_sha256": "0" * 64},
        ),
    ]
    with pytest.raises(campaign.CampaignValidationError, match="seed provenance differs"):
        campaign.merge_campaign(jobs, system="h200_sxm", output_dir=tmp_path / "published")
