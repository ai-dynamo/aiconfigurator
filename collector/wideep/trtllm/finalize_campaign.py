# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate and merge one single-node TensorRT-LLM DeepEP campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
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
from collector.framework_manifest import get_collector_runtime
from collector.registry_types import PerfFile
from collector.wideep.trtllm.collect_moe_a2a import (
    COMM_BACKEND_HT,
    COMM_BACKEND_LL,
    SMS,
    TARGET_SOURCE_COMMIT,
    build_case_plan,
    case_plan_ids,
    communication_dtype_for,
    get_moe_a2a_shapes,
    get_moe_a2a_token_grid,
)

BACKENDS = (COMM_BACKEND_HT, COMM_BACKEND_LL)
EXPECTED_VERSION = "1.3.0rc11"
SYSTEM_LAYOUTS = {
    "gb200": (4, 4),
    "gb300": (4, 4),
    "b200_sxm": (8, 8),
    "b300_sxm": (8, 8),
    "h100_sxm": (8, 8),
    "h200_sxm": (8, 8),
}
SYSTEM_CUDA_ARCHES = {
    "gb200": "100a-real",
    "gb300": "103a-real",
    "b200_sxm": "100a-real",
    "b300_sxm": "103a-real",
    "h100_sxm": "90-real",
    "h200_sxm": "90-real",
}
SYSTEM_GPU_IDENTITIES = {
    "gb200": ("GB200", "10.0"),
    "gb300": ("GB300", "10.3"),
    "b200_sxm": ("B200", "10.0"),
    "b300_sxm": ("B300", "10.3"),
    "h100_sxm": ("H100", "9.0"),
    "h200_sxm": ("H200", "9.0"),
}
RUNTIME = get_collector_runtime("trtllm_a2a")
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


class CampaignValidationError(RuntimeError):
    """A campaign artifact is incomplete or incorrectly identified."""


@dataclass(frozen=True)
class ValidatedJob:
    path: Path
    frame: pd.DataFrame
    runtime: dict[str, Any]
    table: dict[str, Any]
    evidence: dict[str, Any]
    backend: str
    failures: tuple[dict[str, Any], ...]
    classified_failures: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _single(frame: pd.DataFrame, column: str) -> Any:
    values = frame[column].drop_duplicates().tolist()
    if len(values) != 1:
        raise CampaignValidationError(f"{column} must have exactly one value, found {values!r}")
    return values[0]


def _cases(system: str, backend: str):
    ep_size = SYSTEM_LAYOUTS[system][1]
    return build_case_plan(
        shapes=get_moe_a2a_shapes(required_expert_parallel_size=ep_size),
        token_grid=get_moe_a2a_token_grid(),
        ep_size=ep_size,
        node_num=1,
        modes=(backend,),
        system=system,
    )


def _expected_keys(system: str, backend: str) -> set[tuple[Any, ...]]:
    ep_size = SYSTEM_LAYOUTS[system][1]
    keys = set()
    for case in _cases(system, backend):
        for phase in ("combine", "dispatch"):
            dtype = communication_dtype_for(
                system=system,
                backend=backend,
                model_quantization=case.quant.comm_dtype,
                communication_phase=phase,
            )
            keys.add(
                (
                    backend,
                    phase,
                    dtype,
                    ep_size,
                    1,
                    case.shape.hidden_size,
                    case.shape.topk,
                    case.shape.num_experts,
                    case.num_tokens,
                    SMS,
                )
            )
    return keys


def _failure_identity(failure: dict[str, Any], *, backend: str) -> tuple[Any, ...]:
    case = failure.get("case")
    if not isinstance(case, dict):
        raise CampaignValidationError("failure evidence is missing its case identity")
    required = {
        "comm_backend",
        "comm_dtype",
        "inference_phase",
        "ep_size",
        "node_num",
        "hidden_size",
        "topk",
        "num_experts",
        "num_tokens",
        "sms",
    }
    if set(case) != required or case.get("comm_backend") != backend:
        raise CampaignValidationError("failure evidence has an invalid case identity")
    expected_phase = "context" if backend == COMM_BACKEND_HT else "generation"
    if case.get("inference_phase") != expected_phase or not isinstance(case.get("comm_dtype"), str):
        raise CampaignValidationError("failure evidence has an invalid backend/phase identity")
    numeric_fields = ("ep_size", "node_num", "hidden_size", "topk", "num_experts", "num_tokens", "sms")
    if any(not isinstance(case.get(field), int) or isinstance(case.get(field), bool) for field in numeric_fields):
        raise CampaignValidationError("failure evidence has a non-integer physical identity")
    return (
        case["comm_backend"],
        case["comm_dtype"],
        case["ep_size"],
        case["node_num"],
        case["hidden_size"],
        case["topk"],
        case["num_experts"],
        case["num_tokens"],
        case["sms"],
    )


def _failure_physical_keys(identity: tuple[Any, ...], *, system: str) -> set[tuple[Any, ...]]:
    backend, comm_dtype, ep_size, node_num, hidden_size, topk, num_experts, num_tokens, sms = identity
    return {
        (
            backend,
            phase,
            communication_dtype_for(
                system=system,
                backend=backend,
                model_quantization=comm_dtype,
                communication_phase=phase,
            ),
            ep_size,
            node_num,
            hidden_size,
            topk,
            num_experts,
            num_tokens,
            sms,
        )
        for phase in ("combine", "dispatch")
    }


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CampaignValidationError(f"invalid JSON artifact {path}") from error
    if not isinstance(payload, dict):
        raise CampaignValidationError(f"{path}: expected JSON object")
    return payload


def validate_job_dir(
    job_dir: str | Path,
    *,
    system: str,
    allow_partial_evidence: bool = False,
) -> ValidatedJob:
    if system not in SYSTEM_LAYOUTS:
        raise CampaignValidationError(f"unsupported system {system!r}")
    path = Path(job_dir).expanduser().resolve(strict=True)
    parquet_path = path / "moe_a2a_perf.parquet"
    sidecar_path = path / "collection_meta.yaml"
    evidence_path = path / "runtime_evidence.json"
    if not parquet_path.is_file() or not sidecar_path.is_file() or not evidence_path.is_file():
        raise CampaignValidationError(f"{path}: missing parquet, sidecar, or runtime evidence")
    frame = pd.read_parquet(parquet_path)
    if tuple(frame.columns) != ROW_COLUMNS or frame.empty:
        raise CampaignValidationError(f"{parquet_path}: empty table or schema drift")
    if frame[list(PHYSICAL_KEY_COLUMNS)].duplicated().any():
        raise CampaignValidationError(f"{parquet_path}: duplicate physical key")
    if str(_single(frame, "framework")).lower() != "trtllm" or _single(frame, "version") != EXPECTED_VERSION:
        raise CampaignValidationError(f"{parquet_path}: framework/version mismatch")
    if _single(frame, "op_name") != "moe_a2a" or _single(frame, "kernel_source") != "deepep":
        raise CampaignValidationError(f"{parquet_path}: operation/kernel mismatch")
    backend = str(_single(frame, "comm_backend"))
    expected_gpus, ep_size = SYSTEM_LAYOUTS[system]
    if backend not in BACKENDS or int(_single(frame, "node_num")) != 1 or int(_single(frame, "ep_size")) != ep_size:
        raise CampaignValidationError(f"{parquet_path}: rejected backend/node/EP identity")
    if SYSTEM_GPU_IDENTITIES[system][0] not in str(_single(frame, "device")).upper():
        raise CampaignValidationError(f"{parquet_path}: device does not match {system}")
    observed_keys = set(frame[list(PHYSICAL_KEY_COLUMNS)].itertuples(index=False, name=None))
    expected_keys = _expected_keys(system, backend)
    if not observed_keys <= expected_keys:
        raise CampaignValidationError(
            f"{parquet_path}: undeclared physical keys {sorted(observed_keys - expected_keys)[:3]}"
        )
    if not (frame["latency"] - frame["transmit_us"] - frame["notify_us"]).abs().lt(1e-6).all():
        raise CampaignValidationError(f"{parquet_path}: latency accounting mismatch")

    document = yaml.safe_load(sidecar_path.read_text(encoding="utf-8")) or {}
    runtime = document.get("runtime")
    table = (document.get("tables") or {}).get(Path(PerfFile.MOE_A2A.value).stem)
    if document.get("schema_version") != 1 or not isinstance(runtime, dict) or not isinstance(table, dict):
        raise CampaignValidationError(f"{sidecar_path}: invalid collector provenance")
    if runtime.get("framework") != RUNTIME.framework or str(runtime.get("version")) != EXPECTED_VERSION:
        raise CampaignValidationError(f"{sidecar_path}: runtime identity mismatch")
    if runtime.get("source_commit") != TARGET_SOURCE_COMMIT or runtime.get("abi") != RUNTIME.abi:
        raise CampaignValidationError(f"{sidecar_path}: source/ABI mismatch")
    configured_image, configured_digest = RUNTIME.image().split("@", 1)
    if runtime.get("image") != configured_image or runtime.get("image_digest") != configured_digest:
        raise CampaignValidationError(f"{sidecar_path}: configured image identity mismatch")
    evidence = _load_json(evidence_path)
    expected_variant = "linux/arm64" if system in ("gb200", "gb300") else "linux/amd64"
    if evidence.get("system") != system or evidence.get("image_variant") != expected_variant:
        raise CampaignValidationError(f"{evidence_path}: system/image variant mismatch")
    if evidence.get("cuda_arches") != SYSTEM_CUDA_ARCHES[system]:
        raise CampaignValidationError(f"{evidence_path}: CUDA architecture mismatch")
    if evidence.get("configured_image_digest") != configured_digest:
        raise CampaignValidationError(f"{evidence_path}: configured image digest mismatch")
    for key in ("observed_image_digest", "wheel_sha256", "collector_ref"):
        value = str(evidence.get(key, ""))
        expected_length = 71 if key == "observed_image_digest" else 64 if key == "wheel_sha256" else 40
        if len(value) != expected_length:
            raise CampaignValidationError(f"{evidence_path}: invalid {key}")
    if evidence.get("slurm_topology_verified") is not True:
        raise CampaignValidationError(f"{evidence_path}: topology was not verified")
    seed = evidence.get("seed_provenance")
    if seed is not None:
        required = {
            "mode",
            "source_system",
            "source_image_sha256",
            "source_image_meta_sha256",
            "source_image_digest",
        }
        if not isinstance(seed, dict) or not required <= set(seed) or seed.get("mode") not in {"image", "runtime"}:
            raise CampaignValidationError(f"{evidence_path}: invalid seed provenance")
        for field in ("source_image_sha256", "source_image_meta_sha256"):
            if not re.fullmatch(r"[0-9a-f]{64}", str(seed.get(field, ""))):
                raise CampaignValidationError(f"{evidence_path}: invalid seed provenance checksum")
        source_system = str(seed.get("source_system", ""))
        source_variant = "linux/arm64" if source_system in ("gb200", "gb300") else "linux/amd64"
        if source_system not in SYSTEM_LAYOUTS or source_variant != expected_variant:
            raise CampaignValidationError(f"{evidence_path}: seed image architecture mismatch")
        if seed.get("source_image_digest") != evidence["observed_image_digest"]:
            raise CampaignValidationError(f"{evidence_path}: seed image digest mismatch")
        if seed["mode"] == "runtime":
            if seed.get("cuda_arches") != SYSTEM_CUDA_ARCHES[system]:
                raise CampaignValidationError(f"{evidence_path}: seed runtime CUDA architecture mismatch")
            if seed.get("source_wheel_sha256") != evidence["wheel_sha256"]:
                raise CampaignValidationError(f"{evidence_path}: seed runtime wheel checksum mismatch")

    failures: list[dict[str, Any]] = []
    ranks_by_failure: dict[tuple[Any, ...], set[int]] = {}
    for error_path in sorted(path.glob("errors_moe_a2a_trtllm.rank*.json")):
        match = re.fullmatch(r"errors_moe_a2a_trtllm\.rank(\d+)\.json", error_path.name)
        if match is None:
            raise CampaignValidationError(f"{error_path}: malformed rank failure filename")
        file_rank = int(match.group(1))
        payload = json.loads(error_path.read_text(encoding="utf-8"))
        if not isinstance(payload, list) or any(row.get("classification") != "unexpected" for row in payload):
            raise CampaignValidationError(f"{error_path}: malformed failure evidence")
        for failure in payload:
            if failure.get("rank") != file_rank:
                raise CampaignValidationError(f"{error_path}: record rank does not match filename")
            identity = _failure_identity(failure, backend=backend)
            if file_rank in ranks_by_failure.setdefault(identity, set()):
                raise CampaignValidationError(f"{error_path}: duplicate failure identity for rank {file_rank}")
            ranks_by_failure[identity].add(file_rank)
            failures.append(failure)
    missing = expected_keys - observed_keys
    expected_ranks = set(range(ep_size))
    if any(ranks != expected_ranks for ranks in ranks_by_failure.values()):
        raise CampaignValidationError(f"{path}: failure evidence does not agree across every rank")
    missing_keys_by_failure = {
        identity: _failure_physical_keys(identity, system=system) - observed_keys for identity in ranks_by_failure
    }
    failed_missing_keys = set().union(*missing_keys_by_failure.values())
    if failures or missing:
        if not allow_partial_evidence:
            raise CampaignValidationError(
                f"{path}: incomplete formal input ({len(failures)} failures, {len(missing)} missing rows)"
            )
        if any(not identity_missing for identity_missing in missing_keys_by_failure.values()):
            raise CampaignValidationError(f"{path}: failure identity does not cover any missing physical row")
        if not failures or not missing or failed_missing_keys != missing:
            raise CampaignValidationError(f"{path}: partial rows and failure evidence are inconsistent")
    elif not (path / "SUCCESS").is_file():
        raise CampaignValidationError(f"{path}: missing SUCCESS marker")
    if int(table.get("rows", -1)) != len(frame):
        raise CampaignValidationError(f"{sidecar_path}: row count mismatch")
    for field in ("collector_ref", "collector_hash", "case_plan_hash", "collected_at"):
        if not table.get(field):
            raise CampaignValidationError(f"{sidecar_path}: missing {field}")
    expected_plan_hash = provenance.case_plan_hash(case_plan_ids(_cases(system, backend)))
    if table["case_plan_hash"] != expected_plan_hash:
        raise CampaignValidationError(
            f"{sidecar_path}: case_plan_hash mismatch; expected {expected_plan_hash}, found {table['case_plan_hash']}"
        )
    checksum_path = path / "artifact_checksums.json"
    if not checksum_path.is_file():
        raise CampaignValidationError(f"{path}: missing artifact_checksums.json")
    checksums = _load_json(checksum_path)
    checksum_artifacts = [
        parquet_path,
        sidecar_path,
        evidence_path,
        *sorted(path.glob("errors_moe_a2a_trtllm.rank*.json")),
    ]
    if failures or missing:
        failure_summary_path = path / "job_failure.json"
        if not failure_summary_path.is_file():
            raise CampaignValidationError(f"{path}: missing job_failure.json")
        failure_summary = _load_json(failure_summary_path)
        if int(failure_summary.get("benchmark_status", 0)) == 0:
            raise CampaignValidationError(f"{failure_summary_path}: invalid benchmark_status")
        checksum_artifacts.append(failure_summary_path)
    expected_names = {artifact.name for artifact in checksum_artifacts}
    if set(checksums) != expected_names:
        raise CampaignValidationError(
            f"{path}: checksum manifest must contain exactly {sorted(expected_names)}, found {sorted(checksums)}"
        )
    for artifact in checksum_artifacts:
        if checksums[artifact.name] != _sha256(artifact):
            raise CampaignValidationError(f"{artifact}: checksum mismatch")
    del expected_gpus
    return ValidatedJob(
        path,
        frame,
        runtime,
        table,
        evidence,
        backend,
        tuple(failures),
        len(ranks_by_failure),
    )


def merge_campaign(
    input_dirs: list[str | Path],
    *,
    system: str,
    output_dir: str | Path,
    checksum_output: str | Path | None = None,
    allow_partial_evidence: bool = False,
) -> tuple[Path, Path]:
    jobs = [validate_job_dir(path, system=system, allow_partial_evidence=allow_partial_evidence) for path in input_dirs]
    if len(jobs) != 2 or {job.backend for job in jobs} != set(BACKENDS):
        raise CampaignValidationError(f"campaign requires exactly one job for each of {BACKENDS}")
    if (
        len({job.table["collector_ref"] for job in jobs}) != 1
        or len({job.table["collector_hash"] for job in jobs}) != 1
    ):
        raise CampaignValidationError("campaign jobs use different collector refs/hashes")
    immutable_evidence = ("observed_image_digest", "image_variant", "wheel_sha256", "cuda_arches", "collector_ref")
    for field in immutable_evidence:
        if len({job.evidence[field] for job in jobs}) != 1:
            raise CampaignValidationError(f"campaign runtime evidence differs for {field}")
    seed_evidence = {json.dumps(job.evidence.get("seed_provenance"), sort_keys=True) for job in jobs}
    if len(seed_evidence) != 1:
        raise CampaignValidationError("campaign runtime seed provenance differs")
    merged = pd.concat([job.frame for job in jobs], ignore_index=True)
    if merged[list(PHYSICAL_KEY_COLUMNS)].duplicated().any():
        raise CampaignValidationError("merged campaign contains duplicate physical keys")
    merged = merged.sort_values(list(PHYSICAL_KEY_COLUMNS), kind="stable").reset_index(drop=True)
    total_failures = sum(job.classified_failures for job in jobs)
    status = provenance.STATUS_PARTIAL if total_failures else provenance.STATUS_COMPLETE
    all_case_ids = [case_id for backend in BACKENDS for case_id in case_plan_ids(_cases(system, backend))]

    destination = Path(output_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    final_parquet = destination / "moe_a2a_perf.parquet"
    final_sidecar = destination / "collection_meta.yaml"
    with tempfile.TemporaryDirectory(prefix="aic-trtllm-a2a-finalize-", dir="/tmp") as staging_name:
        staging = Path(staging_name)
        staged_parquet = staging / final_parquet.name
        merged.to_parquet(staged_parquet, index=False)
        if pq.read_metadata(staged_parquet).num_rows != len(merged):
            raise CampaignValidationError("staged parquet row count mismatch")
        existing_tables: dict[str, dict[str, Any]] = {}
        if final_sidecar.is_file():
            existing_tables = dict(
                (yaml.safe_load(final_sidecar.read_text(encoding="utf-8")) or {}).get("tables") or {}
            )
        existing_tables["moe_a2a_perf"] = {
            "collector_ref": jobs[0].table["collector_ref"],
            "collector_hash": jobs[0].table["collector_hash"],
            "case_plan_hash": provenance.case_plan_hash(all_case_ids),
            "collected_at": date.today().isoformat(),
            "rows": len(merged),
            "classified_failures": total_failures,
            "status": status,
        }
        runtime = dict(jobs[0].runtime)
        runtime["image"] = RUNTIME.image()
        runtime["image_variant"] = jobs[0].evidence["image_variant"]
        runtime["image_digest"] = jobs[0].evidence["observed_image_digest"]
        runtime["abi"] = dict(runtime["abi"]) | {
            "campaign_system": system,
            "campaign_ep_size": str(SYSTEM_LAYOUTS[system][1]),
            "campaign_backends": ",".join(BACKENDS),
            "source_wheel_sha256": jobs[0].evidence["wheel_sha256"],
            "configured_image_digest": RUNTIME.image().split("@", 1)[1],
            "slurm_topology_verified": "true",
        }
        if jobs[0].evidence.get("seed_provenance") is not None:
            runtime["abi"]["runtime_seed_provenance"] = json.dumps(
                jobs[0].evidence["seed_provenance"], sort_keys=True, separators=(",", ":")
            )
        staged_sidecar = provenance.write_collection_meta(staging, runtime, existing_tables)
        staged_errors: list[Path] = []
        if total_failures:
            for job in jobs:
                if job.failures:
                    target = staging / f"errors_moe_a2a_trtllm.{job.backend}.json"
                    target.write_text(json.dumps(job.failures, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                    staged_errors.append(target)
        staged_checksums = {p.name: _sha256(p) for p in (staged_parquet, staged_sidecar, *staged_errors)}
        publish = [(staged_parquet, final_parquet), (staged_sidecar, final_sidecar)] + [
            (path, destination / path.name) for path in staged_errors
        ]
        if checksum_output:
            checksum_path = Path(checksum_output).expanduser()
            checksum_path.parent.mkdir(parents=True, exist_ok=True)
            staged_checksum = staging / "artifact_checksums.json"
            staged_checksum.write_text(json.dumps(staged_checksums, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            publish.append((staged_checksum, checksum_path))
        for source, target in publish:
            temporary = target.parent / f".{target.name}.tmp.{os.getpid()}"
            shutil.copyfile(source, temporary)
            os.replace(temporary, target)
        for stale in destination.glob("errors_moe_a2a_trtllm.*.json"):
            if stale.name not in staged_checksums:
                stale.unlink()
    if (
        _sha256(final_parquet) != staged_checksums[final_parquet.name]
        or _sha256(final_sidecar) != staged_checksums[final_sidecar.name]
    ):
        raise CampaignValidationError("published artifact checksum mismatch")
    return final_parquet, final_sidecar


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", choices=sorted(SYSTEM_LAYOUTS), required=True)
    parser.add_argument("--input", action="append", required=True, dest="inputs")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checksum-output")
    parser.add_argument("--allow-partial-evidence", action="store_true")
    args = parser.parse_args(argv)
    parquet, sidecar = merge_campaign(
        args.inputs,
        system=args.system,
        output_dir=args.output_dir,
        checksum_output=args.checksum_output,
        allow_partial_evidence=args.allow_partial_evidence,
    )
    print(json.dumps({"parquet": str(parquet), "sidecar": str(sidecar)}, indent=2))


if __name__ == "__main__":
    main()
