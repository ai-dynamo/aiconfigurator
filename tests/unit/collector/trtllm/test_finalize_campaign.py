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


def _write_job(root: Path, *, backend: str, system: str = "h200_sxm", partial: bool = False) -> Path:
    ep_size = campaign.SYSTEM_LAYOUTS[system][1]
    path = root / backend
    path.mkdir(parents=True)
    cases = campaign._cases(ep_size, backend)
    rows = []
    emitted_cases = cases[:-1] if partial else cases
    for case in emitted_cases:
        for phase in ("combine", "dispatch"):
            dtype = (
                "fp4"
                if backend == campaign.COMM_BACKEND_LL and case.quant.comm_dtype == "nvfp4" and phase == "combine"
                else case.quant.comm_dtype
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
        "collector_ref": "a" * 40,
        "slurm_topology_verified": True,
    }
    (path / "runtime_evidence.json").write_text(json.dumps(evidence), encoding="utf-8")
    if partial:
        failure = {
            "classification": "unexpected",
            "case": {"comm_backend": backend},
            "error": "known observed kernel limit",
        }
        (path / "errors_moe_a2a_trtllm.rank0.json").write_text(json.dumps([failure]), encoding="utf-8")
    checksums = {
        name: hashlib.sha256((path / name).read_bytes()).hexdigest()
        for name in ("moe_a2a_perf.parquet", "collection_meta.yaml", "runtime_evidence.json")
    }
    (path / "artifact_checksums.json").write_text(json.dumps(checksums), encoding="utf-8")
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


def test_merge_requires_both_backends(tmp_path):
    job = _write_job(tmp_path / "jobs", backend=campaign.COMM_BACKEND_HT)
    with pytest.raises(campaign.CampaignValidationError, match="exactly one job"):
        campaign.merge_campaign([job], system="h200_sxm", output_dir=tmp_path / "published")
