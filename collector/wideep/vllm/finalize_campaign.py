# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate and merge one six-job vLLM DeepEP campaign for one system.

Formal input is exactly one job for every supported ``(node_num, backend)``
pair.  A job with classified failures, incomplete provenance, an undeclared
row, or a duplicate physical key is rejected.  The merged parquet is built in
job-unique ``/tmp`` staging and copied atomically into the requested output
directory only after all validation succeeds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq
import yaml

from collector import provenance
from collector.registry_types import PerfFile
from collector.wideep.vllm.collect_moe_a2a import (
    BACKENDS,
    TARGET_VLLM_SOURCE_COMMIT,
    build_case_plan,
    case_plan_ids,
    get_moe_a2a_workload_grid,
    get_vllm_moe_a2a_shapes,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
EXPECTED_VERSION = "0.24.0"
SYSTEM_LAYOUTS: dict[str, tuple[int, dict[int, int]]] = {
    "gb200": (4, {2: 8, 4: 16}),
    "gb300": (4, {2: 8, 4: 16}),
    "b200_sxm": (8, {2: 16, 4: 32}),
    "b300_sxm": (8, {2: 16, 4: 32}),
    "h100_sxm": (8, {2: 16, 4: 32}),
    "h200_sxm": (8, {2: 16, 4: 32}),
}
ROW_COLUMNS = (
    "framework",
    "version",
    "device",
    "op_name",
    "kernel_source",
    "comm_backend",
    "phase",
    "comm_dtype",
    "ep_size",
    "node_num",
    "hidden_size",
    "topk",
    "num_experts",
    "num_tokens",
    "sms",
    "transmit_us",
    "notify_us",
    "latency",
)
PHYSICAL_KEY_COLUMNS = (
    "comm_backend",
    "phase",
    "comm_dtype",
    "ep_size",
    "node_num",
    "hidden_size",
    "topk",
    "num_experts",
    "num_tokens",
    "sms",
)
CASE_BASE_COLUMNS = (
    "comm_backend",
    "ep_size",
    "node_num",
    "hidden_size",
    "topk",
    "num_experts",
    "num_tokens",
)
REQUIRED_ABI = {
    "build_mode": "official-v0.24.0-image",
    "torch": "2.11.0",
    "cuda": "13.0.2",
    "deep_ep": "73b6ea4a439ba03a695563f9fd242c8e4b02b37c",
    "nvshmem": "3.3.24",
}


class CampaignValidationError(RuntimeError):
    """A formal campaign artifact is incomplete or incorrectly identified."""


@dataclass(frozen=True)
class ValidatedJob:
    path: Path
    frame: pd.DataFrame
    runtime: dict[str, Any]
    table: dict[str, Any]
    backend: str
    node_num: int
    ep_size: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _as_single(frame: pd.DataFrame, column: str) -> Any:
    values = frame[column].drop_duplicates().tolist()
    if len(values) != 1:
        raise CampaignValidationError(f"{column} must have exactly one value, found {values!r}")
    return values[0]


def _expected_cases(*, ep_size: int, node_num: int, backend: str):
    return build_case_plan(
        shapes=get_vllm_moe_a2a_shapes(required_expert_parallel_size=ep_size),
        grid=get_moe_a2a_workload_grid(),
        world_size=ep_size,
        node_num=node_num,
        backends=(backend,),
    )


def _load_sidecar(job_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    sidecar = job_dir / "collection_meta.yaml"
    if not sidecar.is_file():
        raise CampaignValidationError(f"missing sidecar {sidecar}")
    document = yaml.safe_load(sidecar.read_text(encoding="utf-8")) or {}
    if document.get("schema_version") != 1:
        raise CampaignValidationError(f"{sidecar}: expected schema_version 1")
    runtime = document.get("runtime")
    table = (document.get("tables") or {}).get(Path(PerfFile.MOE_A2A.value).stem)
    if not isinstance(runtime, dict) or not isinstance(table, dict):
        raise CampaignValidationError(f"{sidecar}: missing runtime or moe_a2a_perf table provenance")
    return runtime, table


def _validate_runtime(runtime: dict[str, Any], *, job_dir: Path) -> None:
    if str(runtime.get("framework", "")).lower() != "vllm":
        raise CampaignValidationError(f"{job_dir}: runtime framework is not vllm")
    if str(runtime.get("version")) != EXPECTED_VERSION:
        raise CampaignValidationError(f"{job_dir}: runtime version is not {EXPECTED_VERSION}")
    if runtime.get("source_commit") != TARGET_VLLM_SOURCE_COMMIT:
        raise CampaignValidationError(f"{job_dir}: wrong vLLM source commit")
    abi = runtime.get("abi")
    if not isinstance(abi, dict):
        raise CampaignValidationError(f"{job_dir}: runtime ABI is missing")
    mismatch = {key: (value, abi.get(key)) for key, value in REQUIRED_ABI.items() if abi.get(key) != value}
    if mismatch:
        raise CampaignValidationError(f"{job_dir}: runtime ABI mismatch {mismatch}")
    if abi.get("slurm_topology_verified") != "true":
        raise CampaignValidationError(f"{job_dir}: Slurm/fabric topology was not attested")
    image_digest = str(runtime.get("image_digest", ""))
    if not image_digest.startswith("sha256:") or len(image_digest) != 71:
        raise CampaignValidationError(f"{job_dir}: invalid image digest {image_digest!r}")


def _validate_failures(job_dir: Path) -> None:
    for error_path in sorted(job_dir.glob("errors_moe_a2a_vllm.rank*.json")):
        records = json.loads(error_path.read_text(encoding="utf-8"))
        if records:
            raise CampaignValidationError(f"{job_dir}: formal input has classified failures in {error_path.name}")


def validate_job_dir(job_dir: str | Path, *, system: str) -> ValidatedJob:
    """Validate one backend/node-count result directory."""
    if system not in SYSTEM_LAYOUTS:
        raise CampaignValidationError(f"unsupported system {system!r}")
    resolved = Path(job_dir).expanduser().resolve(strict=True)
    parquet_path = resolved / PerfFile.MOE_A2A.value.replace(".txt", ".parquet")
    if not parquet_path.is_file():
        raise CampaignValidationError(f"missing parquet {parquet_path}")
    runtime, table = _load_sidecar(resolved)
    _validate_runtime(runtime, job_dir=resolved)
    _validate_failures(resolved)

    frame = pd.read_parquet(parquet_path)
    if tuple(frame.columns) != ROW_COLUMNS:
        raise CampaignValidationError(
            f"{parquet_path}: schema drift; expected {ROW_COLUMNS!r}, found {tuple(frame.columns)!r}"
        )
    if frame.empty:
        raise CampaignValidationError(f"{parquet_path}: empty formal table")
    if frame[list(PHYSICAL_KEY_COLUMNS)].duplicated().any():
        raise CampaignValidationError(f"{parquet_path}: duplicate physical row key")
    if _as_single(frame, "framework").lower() != "vllm" or _as_single(frame, "version") != EXPECTED_VERSION:
        raise CampaignValidationError(f"{parquet_path}: row framework/version mismatch")
    if _as_single(frame, "op_name") != "moe_a2a" or _as_single(frame, "kernel_source") != "deepep":
        raise CampaignValidationError(f"{parquet_path}: row operation/kernel identity mismatch")
    if _as_single(frame, "comm_dtype") != "default":
        raise CampaignValidationError(f"{parquet_path}: unexpected communication dtype")

    backend = str(_as_single(frame, "comm_backend"))
    node_num = int(_as_single(frame, "node_num"))
    ep_size = int(_as_single(frame, "ep_size"))
    _, node_to_ep = SYSTEM_LAYOUTS[system]
    if node_to_ep.get(node_num) != ep_size or backend not in BACKENDS:
        raise CampaignValidationError(
            f"{parquet_path}: rejected formal identity system={system}, "
            f"nodes={node_num}, ep={ep_size}, backend={backend}"
        )

    cases = _expected_cases(ep_size=ep_size, node_num=node_num, backend=backend)
    if len(frame) != len(cases) * 2:
        raise CampaignValidationError(
            f"{parquet_path}: expected {len(cases) * 2} rows for the declared plan, found {len(frame)}"
        )
    expected_bases = {
        (backend, ep_size, node_num, case.shape.hidden_size, case.shape.topk, case.shape.num_experts, case.num_tokens)
        for case in cases
    }
    observed_bases = set(frame[list(CASE_BASE_COLUMNS)].itertuples(index=False, name=None))
    if observed_bases != expected_bases:
        missing = sorted(expected_bases - observed_bases)[:5]
        extra = sorted(observed_bases - expected_bases)[:5]
        raise CampaignValidationError(f"{parquet_path}: case population mismatch; missing={missing}, extra={extra}")
    phases = frame.groupby(list(CASE_BASE_COLUMNS), dropna=False)["phase"].agg(lambda values: tuple(sorted(values)))
    if not phases.map(lambda value: value == ("combine", "dispatch")).all():
        raise CampaignValidationError(f"{parquet_path}: every case must have exactly combine and dispatch rows")
    if not (frame["latency"] - frame["transmit_us"] - frame["notify_us"]).abs().lt(1e-6).all():
        raise CampaignValidationError(f"{parquet_path}: latency is not transmit_us + notify_us")

    if table.get("status") != provenance.STATUS_COMPLETE or int(table.get("rows", -1)) != len(frame):
        raise CampaignValidationError(f"{resolved}: incomplete or row-mismatched sidecar table")
    for field in ("collector_ref", "collector_hash", "case_plan_hash", "collected_at"):
        if not table.get(field):
            raise CampaignValidationError(f"{resolved}: sidecar table is missing {field}")

    checksum_path = resolved / "artifact_checksums.json"
    if checksum_path.is_file():
        checksums = json.loads(checksum_path.read_text(encoding="utf-8"))
        expected = checksums.get(parquet_path.name)
        if expected != _sha256(parquet_path):
            raise CampaignValidationError(f"{resolved}: parquet checksum mismatch")
        sidecar = resolved / "collection_meta.yaml"
        if checksums.get(sidecar.name) != _sha256(sidecar):
            raise CampaignValidationError(f"{resolved}: sidecar checksum mismatch")

    return ValidatedJob(resolved, frame, runtime, table, backend, node_num, ep_size)


def _merge_runtime(jobs: list[ValidatedJob], *, system: str) -> dict[str, Any]:
    immutable_fields = ("framework", "version", "image", "image_digest", "source_commit")
    for field in immutable_fields:
        values = {json.dumps(job.runtime.get(field), sort_keys=True) for job in jobs}
        if len(values) != 1:
            raise CampaignValidationError(f"campaign runtime field {field!r} differs across jobs: {values}")
    abi_values = [job.runtime["abi"] for job in jobs]
    immutable_abi = {
        key: value
        for key, value in abi_values[0].items()
        if all(candidate.get(key) == value for candidate in abi_values[1:])
    }
    immutable_abi.update(
        {
            "campaign_system": system,
            "campaign_node_counts": ",".join(str(value) for value in sorted({job.node_num for job in jobs})),
            "campaign_ep_sizes": ",".join(str(value) for value in sorted({job.ep_size for job in jobs})),
            "campaign_backends": ",".join(sorted({job.backend for job in jobs})),
            "slurm_topology_verified": "true",
            "fabric_identities": ",".join(sorted({str(job.runtime["abi"].get("fabric_identity")) for job in jobs})),
        }
    )
    return {field: jobs[0].runtime[field] for field in immutable_fields if field in jobs[0].runtime} | {
        "abi": immutable_abi
    }


def merge_campaign(
    input_dirs: list[str | Path],
    *,
    system: str,
    output_dir: str | Path,
    checksum_output: str | Path | None = None,
) -> tuple[Path, Path]:
    """Validate six formal jobs, merge them, and atomically publish artifacts."""
    if system not in SYSTEM_LAYOUTS:
        raise CampaignValidationError(f"unsupported system {system!r}")
    jobs = [validate_job_dir(path, system=system) for path in input_dirs]
    expected_combinations = {(node_num, backend) for node_num in SYSTEM_LAYOUTS[system][1] for backend in BACKENDS}
    observed_combinations = {(job.node_num, job.backend) for job in jobs}
    if len(jobs) != len(expected_combinations) or observed_combinations != expected_combinations:
        raise CampaignValidationError(
            f"formal campaign requires exactly {sorted(expected_combinations)}, found {sorted(observed_combinations)}"
        )

    collector_refs = {job.table["collector_ref"] for job in jobs}
    collector_hashes = {job.table["collector_hash"] for job in jobs}
    if len(collector_refs) != 1 or len(collector_hashes) != 1:
        raise CampaignValidationError("all campaign jobs must use one collector commit/hash")

    merged = pd.concat([job.frame for job in jobs], ignore_index=True)
    if merged[list(PHYSICAL_KEY_COLUMNS)].duplicated().any():
        raise CampaignValidationError("merged campaign contains duplicate physical keys")
    merged = merged.sort_values(list(PHYSICAL_KEY_COLUMNS), kind="stable").reset_index(drop=True)

    all_case_ids: list[str] = []
    for node_num, ep_size in SYSTEM_LAYOUTS[system][1].items():
        for backend in BACKENDS:
            cases = _expected_cases(ep_size=ep_size, node_num=node_num, backend=backend)
            all_case_ids.extend(case_plan_ids(cases, world_size=ep_size, node_num=node_num))

    destination = Path(output_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    parquet_name = PerfFile.MOE_A2A.value.replace(".txt", ".parquet")
    final_parquet = destination / parquet_name
    final_sidecar = destination / "collection_meta.yaml"
    with tempfile.TemporaryDirectory(prefix="aic-vllm-a2a-finalize-", dir="/tmp") as staging_name:
        staging = Path(staging_name)
        staged_parquet = staging / parquet_name
        merged.to_parquet(staged_parquet, index=False)
        if pq.read_metadata(staged_parquet).num_rows != len(merged):
            raise CampaignValidationError("staged parquet row-count verification failed")

        existing_tables: dict[str, dict[str, Any]] = {}
        if final_sidecar.is_file():
            existing_document = yaml.safe_load(final_sidecar.read_text(encoding="utf-8")) or {}
            existing_tables = dict(existing_document.get("tables") or {})
        existing_tables[Path(PerfFile.MOE_A2A.value).stem] = {
            "collector_ref": next(iter(collector_refs)),
            "collector_hash": next(iter(collector_hashes)),
            "case_plan_hash": provenance.case_plan_hash(all_case_ids),
            "collected_at": date.today().isoformat(),
            "rows": len(merged),
            "status": provenance.STATUS_COMPLETE,
        }
        staged_sidecar = provenance.write_collection_meta(
            staging,
            _merge_runtime(jobs, system=system),
            existing_tables,
        )

        staged_checksums = {
            staged_parquet.name: _sha256(staged_parquet),
            staged_sidecar.name: _sha256(staged_sidecar),
        }
        for staged_file, final_file in ((staged_parquet, final_parquet), (staged_sidecar, final_sidecar)):
            temporary_destination = destination / f".{final_file.name}.tmp.{os.getpid()}"
            shutil.copyfile(staged_file, temporary_destination)
            os.replace(temporary_destination, final_file)

    if checksum_output is not None:
        checksum_path = Path(checksum_output).expanduser()
        checksum_path.parent.mkdir(parents=True, exist_ok=True)
        checksum_path.write_text(json.dumps(staged_checksums, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    parquet_mismatch = _sha256(final_parquet) != staged_checksums[final_parquet.name]
    sidecar_mismatch = _sha256(final_sidecar) != staged_checksums[final_sidecar.name]
    if parquet_mismatch or sidecar_mismatch:
        raise CampaignValidationError("atomic publish checksum verification failed")
    return final_parquet, final_sidecar


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", choices=sorted(SYSTEM_LAYOUTS), required=True)
    parser.add_argument("--input", action="append", required=True, dest="inputs")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checksum-output")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    parquet_path, sidecar_path = merge_campaign(
        args.inputs,
        system=args.system,
        output_dir=args.output_dir,
        checksum_output=args.checksum_output,
    )
    print(json.dumps({"parquet": str(parquet_path), "sidecar": str(sidecar_path)}, indent=2))


if __name__ == "__main__":
    main()
