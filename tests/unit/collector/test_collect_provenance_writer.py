# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for collect.py's `_write_collector_provenance` — the finalize-time
glue (Collector V3 design §5) that turns ResumeCheckpoint JSON files + registry
collections into a `collection_meta.yaml` sidecar beside the just-finalized parquet.

`collector/provenance.py` (the rendering/hashing primitives) is covered by
test_provenance.py; this file covers the production glue in collect.py that
calls it: checkpoint reading, table/status derivation, and the existing-sidecar
merge path (including the legacy-tier merge guard).
"""

import hashlib
import json
import logging
import re
import stat
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

# collect.py is a top-level script (`from helper import ...`) — put collector/
# on sys.path so it (and its flat-import siblings) resolve, same as the rest
# of the collect.py test suite (see test_parallel_run.py).
_COLLECTOR_DIR = str(Path(__file__).resolve().parents[3] / "collector")
if _COLLECTOR_DIR not in sys.path:
    sys.path.insert(0, _COLLECTOR_DIR)

# Mock torch BEFORE collect.py is imported, exactly like test_parallel_run.py.
# This module is imported first (alphabetical collection order), so whatever
# `torch` it leaves cached inside collect.py is what test_parallel_run's
# fork-worker tests see: with collect.torch = None, worker() dies in
# _require_torch() before consuming its queue sentinel and parallel_run
# deadlocks the whole xdist worker.
if "torch" not in sys.modules:
    from unittest.mock import MagicMock

    _torch = MagicMock()
    _torch.AcceleratorError = type("AcceleratorError", (Exception,), {})
    sys.modules["torch"] = _torch

import collect as collect_mod

from collector import provenance
from collector.framework_manifest import CollectorRuntime
from collector.registry_types import PerfFile

pytestmark = pytest.mark.unit

collect_mod.logger = logging.getLogger("test_collect_provenance_writer")

# A real, already-hash_closures.yaml-covered module — using it lets the writer's
# real load_closures()/collector_hash() calls run unmocked against the real repo
# tree, instead of needing to fabricate a fake repo layout.
REAL_MODULE = "collector.sglang.collect_gemm"
MLA_MODULE = "collector.sglang.collect_mla_bmm"
BACKEND = "sglang"
OP_TYPE = "gemm"
FULL_NAME = f"{BACKEND}.{OP_TYPE}"  # collection["name"] + "." + collection["type"]
RUN_FUNC = "run_gemm"


def _run_func_for(full_name: str) -> str:
    return {
        "gemm": RUN_FUNC,
        "moe": "run_moe_torch",
        "mla_bmm_gen_pre": "run_mla_gen_pre",
        "mla_bmm_gen_post": "run_mla_gen_post",
    }.get(full_name.split(".", 1)[-1], "run")


def _collections(table: str = "gemm_perf") -> list[dict]:
    return [
        {
            "name": BACKEND,
            "type": OP_TYPE,
            "module": REAL_MODULE,
            "run_func": RUN_FUNC,
            "perf_filename": f"{table}.txt",
        }
    ]


def _shared_collections() -> list[dict]:
    return [
        {
            "name": BACKEND,
            "type": "mla_bmm_gen_pre",
            "module": MLA_MODULE,
            "run_func": "run_mla_gen_pre",
            "perf_filename": "mla_bmm_perf.txt",
        },
        {
            "name": BACKEND,
            "type": "mla_bmm_gen_post",
            "module": MLA_MODULE,
            "run_func": "run_mla_gen_post",
            "perf_filename": "mla_bmm_perf.txt",
        },
    ]


def _provenance_ctx(collections: list[dict]) -> dict:
    runtime = CollectorRuntime(
        framework="sglang",
        version="0.5.14",
        images={"default": "lmsysorg/sglang:v0.5.14@sha256:" + "0" * 64},
    )
    return {
        "framework": runtime.framework,
        "installed_version": runtime.version,
        "runtime": runtime,
        "sm_version": 100,
        "collections": collections,
    }


def _write_checkpoint(
    checkpoint_dir: Path,
    *,
    done: list[str],
    failed: list[str],
    attempted: list[str] | None = None,
) -> Path:
    path = checkpoint_dir / BACKEND / f"{FULL_NAME}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema": collect_mod.RESUME_SCHEMA_VERSION,
                "backend": BACKEND,
                "module": FULL_NAME,
                "run_func": RUN_FUNC,
                "framework_version": "0.5.14",
                "sm_version": 100,
                "updated_at": "2026-07-20T00:00:00",
                "done": sorted(done),
                "failed": sorted(failed),
                "attempted": sorted(done + failed if attempted is None else attempted),
            }
        )
    )
    return path


def _write_checkpoint_for(
    checkpoint_dir: Path,
    *,
    backend: str,
    full_name: str,
    version: str,
    done: list[str],
    failed: list[str],
    attempted: list[str] | None = None,
) -> Path:
    path = checkpoint_dir / backend / f"{full_name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema": collect_mod.RESUME_SCHEMA_VERSION,
                "backend": backend,
                "module": full_name,
                "run_func": _run_func_for(full_name),
                "framework_version": version,
                "sm_version": 100,
                "updated_at": "2026-07-20T00:00:00",
                "done": sorted(done),
                "failed": sorted(failed),
                "attempted": sorted(done + failed if attempted is None else attempted),
            }
        )
    )
    return path


def _write_parquet(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table({"op": [row["op"] for row in rows], "latency": [row["latency"] for row in rows]})
    pq.write_table(table, path)


def _write_reduced_collection_meta(output_root: Path, runtime: dict, tables: dict) -> None:
    provenance.write_collection_meta(
        output_root,
        runtime,
        tables,
        provenance_tier="local",
    )


def _collection_event(*, rows: int = 1, status: str = "complete") -> dict:
    return {
        "collector_ref": "a" * 40,
        "collector_hash": "sha256:" + "b" * 64,
        "case_plan_hash": provenance.case_plan_hash(["case-prior"]),
        "collected_at": "2026-08-10",
        "rows": rows,
        "status": status,
    }


def _sidecar_runtime(version: str = "0.5.14") -> dict:
    return {
        "framework": "sglang",
        "version": version,
        "image": f"lmsysorg/sglang:v{version}",
        "image_digest": "sha256:" + "0" * 64,
    }


def _finalization_info_for(*parquet_paths: Path) -> dict[Path, collect_mod.PerfFinalizationInfo]:
    # main() retains each staging input until its sidecar transaction commits.
    for path in parquet_paths:
        staging_path = path.with_suffix(".txt")
        if not staging_path.exists():
            staging_path.write_text("retained staging\n", encoding="utf-8")
    finalization_info = {}
    for path in parquet_paths:
        staging_path = path.with_suffix(".txt")
        finalization_info[path.resolve()] = _finalization_fact(
            staging_path,
            new_rows=pq.read_metadata(path).num_rows,
            merged_existing=False,
        )
    return finalization_info


def _finalization_fact(
    staging_path: Path,
    *,
    new_rows: int,
    merged_existing: bool,
) -> collect_mod.PerfFinalizationInfo:
    staging_stat = staging_path.stat()
    return collect_mod.PerfFinalizationInfo(
        new_rows=new_rows,
        merged_existing=merged_existing,
        source_digest=_independent_digest(staging_path),
        source_device=staging_stat.st_dev,
        source_inode=staging_stat.st_ino,
    )


def _attempted(checkpoint_path: Path) -> set[str]:
    return set(json.loads(checkpoint_path.read_text(encoding="utf-8")).get("attempted", []))


def _independent_digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _staging_manifest_record(path: Path) -> dict:
    staging_stat = path.stat()
    return {
        "path": str(path.resolve()),
        "digest": _independent_digest(path),
        "device": staging_stat.st_dev,
        "inode": staging_stat.st_ino,
    }


def _replace_with_new_inode(path: Path, contents: bytes) -> int:
    replacement_path = path.with_name(f".{path.name}.replacement")
    replacement_path.write_bytes(contents)
    replacement_inode = replacement_path.stat().st_ino
    replacement_path.replace(path)
    return replacement_inode


def _file_attestation(path: Path) -> collect_mod._FileAttestation:
    file_stat = path.stat()
    return collect_mod._FileAttestation(
        path=path,
        digest=_independent_digest(path),
        device=file_stat.st_dev,
        inode=file_stat.st_ino,
    )


def _checkpoint_manifest_record(checkpoint_path: Path, attempted: set[str]) -> dict:
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    return {
        "path": str(checkpoint_path),
        "done": sorted(checkpoint["done"]),
        "failed": sorted(checkpoint["failed"]),
        "attempted": sorted(attempted),
        "identity": {field: checkpoint[field] for field in collect_mod._CHECKPOINT_IDENTITY_FIELDS},
    }


def _run_main_with_staged_output(
    monkeypatch,
    *,
    output_root: Path,
    checkpoint_dir: Path,
    provenance_ctx: dict,
    stage_output,
) -> None:
    def collect_backend(*_args, **_kwargs):
        stage_output()
        return [], provenance_ctx

    monkeypatch.chdir(output_root)
    monkeypatch.setattr(collect_mod, "collect_sglang", collect_backend)
    monkeypatch.setattr(collect_mod.resource, "setrlimit", lambda *_args: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "collect.py",
            "--backend",
            BACKEND,
            "--checkpoint-dir",
            str(checkpoint_dir),
            "--sm",
            "100",
            "--profile",
            "--limit",
            "1",
        ],
    )
    collect_mod.main()


def _write_sidecar_transaction(
    output_root: Path,
    participants: list[tuple[Path, set[str]]],
    staging_paths: list[Path],
    *,
    checkpoint_dir: Path,
    tagged: set[Path],
    pending_document: bool,
    participant_tables: dict[Path, str] | None = None,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    meta_path = output_root / "collection_meta.yaml"
    canonical_names = {str(perf_file) for perf_file in PerfFile}
    default_table = next(
        (path.stem for path in staging_paths if path.name in canonical_names),
        "gemm_perf",
    )
    participant_tables = participant_tables or {checkpoint_path: default_table for checkpoint_path, _ in participants}
    case_ids_by_table: dict[str, set[str]] = {}
    for checkpoint_path, attempted_case_ids in participants:
        case_ids_by_table.setdefault(participant_tables[checkpoint_path], set()).update(attempted_case_ids)
    meta_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "runtime": _sidecar_runtime(),
                "tables": {
                    table: {
                        **_collection_event(),
                        "case_plan_hash": provenance.case_plan_hash(sorted(case_ids_by_table[table])),
                    }
                    for table in case_ids_by_table
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    pending_path = output_root / collect_mod._SIDECAR_STAGING_FILENAME
    pending_path.write_bytes(meta_path.read_bytes())
    pending_attestation = _file_attestation(pending_path)
    if pending_document:
        meta_path.unlink()
    else:
        pending_path.unlink()
    transaction_id = "0" * 32
    for checkpoint_path, attempted_case_ids in participants:
        if checkpoint_path in tagged:
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            assert set(checkpoint["attempted"]) == attempted_case_ids
            checkpoint[collect_mod._SIDECAR_TRANSACTION_FIELD] = transaction_id
            checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")
    transaction = {
        "schema": collect_mod._SIDECAR_TRANSACTION_SCHEMA,
        "transaction_id": transaction_id,
        "output_root": str(output_root.resolve()),
        "backend": BACKEND,
        "checkpoint_root": str((checkpoint_dir.resolve() / BACKEND).resolve()),
        "sidecar_path": str(meta_path.resolve()),
        "sidecar_digest": pending_attestation.digest,
        "pending_sidecar": {
            "digest": pending_attestation.digest,
            "device": pending_attestation.device,
            "inode": pending_attestation.inode,
        },
        "previous_sidecar": None,
        "checkpoints": [
            _checkpoint_manifest_record(checkpoint_path, attempted_case_ids)
            for checkpoint_path, attempted_case_ids in participants
        ],
        "staging_paths": [_staging_manifest_record(path) for path in staging_paths if path.is_file()],
    }
    (output_root / collect_mod._SIDECAR_TRANSACTION_FILENAME).write_text(
        json.dumps(transaction),
        encoding="utf-8",
    )


def test_atomic_checkpoint_replace_failure_preserves_previous_document(tmp_path, monkeypatch):
    checkpoint_path = tmp_path / "checkpoint.json"
    checkpoint_path.write_text('{"attempted": ["old"]}', encoding="utf-8")

    def fail_replace(_source, _target):
        raise OSError("simulated replace failure")

    monkeypatch.setattr(collect_mod.os, "replace", fail_replace)

    with pytest.raises(OSError, match="simulated replace failure"):
        collect_mod._atomic_write_json(checkpoint_path, {"attempted": ["new"]})

    assert json.loads(checkpoint_path.read_text(encoding="utf-8")) == {"attempted": ["old"]}
    assert list(tmp_path.glob(".checkpoint.json.*.tmp")) == []


def test_atomic_exclusive_write_rejects_regular_temp_path_swap(tmp_path, monkeypatch):
    destination = tmp_path / "collection_meta.yaml"
    competitor = b"competitor"
    swapped_temp_path = None
    real_link = collect_mod.os.link

    def swap_temp_then_link(source, target, *args, **kwargs):
        nonlocal swapped_temp_path
        swapped_temp_path = Path(source)
        replacement = tmp_path / "replacement"
        replacement.write_bytes(competitor)
        replacement.replace(swapped_temp_path)
        return real_link(source, target, *args, **kwargs)

    monkeypatch.setattr(collect_mod.os, "link", swap_temp_then_link)

    with pytest.raises(RuntimeError, match=r"temporary|publication|changed"):
        collect_mod._atomic_write_bytes(destination, b"owned", replace_existing=False)

    assert swapped_temp_path is not None
    assert swapped_temp_path.read_bytes() == competitor
    assert not destination.exists()


def test_atomic_exclusive_write_rejects_symlink_temp_path_swap(tmp_path, monkeypatch):
    destination = tmp_path / "collection_meta.yaml"
    competitor = tmp_path / "competitor"
    competitor.write_bytes(b"competitor")
    swapped_temp_path = None
    real_link = collect_mod.os.link

    def swap_temp_then_link(source, target, *args, **kwargs):
        nonlocal swapped_temp_path
        swapped_temp_path = Path(source)
        swapped_temp_path.unlink()
        try:
            swapped_temp_path.symlink_to(competitor)
        except OSError as error:
            pytest.skip(f"symlinks unavailable: {error}")
        return real_link(source, target, *args, **kwargs)

    monkeypatch.setattr(collect_mod.os, "link", swap_temp_then_link)

    with pytest.raises(RuntimeError, match=r"temporary|publication|changed"):
        collect_mod._atomic_write_bytes(destination, b"owned", replace_existing=False)

    assert swapped_temp_path is not None
    assert swapped_temp_path.is_symlink()
    assert competitor.read_bytes() == b"competitor"
    assert not destination.exists()
    assert not destination.is_symlink()


def test_atomic_exclusive_write_publishes_owned_bytes_and_mode(tmp_path, monkeypatch):
    destination = tmp_path / "collection_meta.yaml"
    follow_symlinks = []
    real_link = collect_mod.os.link

    def record_link(source, target, *args, **kwargs):
        follow_symlinks.append(kwargs.get("follow_symlinks"))
        return real_link(source, target, *args, **kwargs)

    monkeypatch.setattr(collect_mod.os, "link", record_link)

    collect_mod._atomic_write_bytes(destination, b"owned", mode=0o640, replace_existing=False)

    assert destination.read_bytes() == b"owned"
    assert stat.S_IMODE(destination.stat().st_mode) == 0o640
    assert follow_symlinks == [False]
    assert list(tmp_path.glob(".collection_meta.yaml.*.tmp")) == []


def test_claim_rejects_broken_symlink_swap_after_batch_validation(tmp_path, monkeypatch):
    owned_path = tmp_path / "gemm_perf.txt"
    owned_path.write_bytes(b"owned staging")
    attestation = _file_attestation(owned_path)
    outside_owned = tmp_path / "outside-owned.txt"
    missing_target = tmp_path / "missing-target.txt"
    real_validate = collect_mod._validated_regular_file
    validations = 0

    def validate_then_swap(path, *args, **kwargs):
        nonlocal validations
        snapshot = real_validate(path, *args, **kwargs)
        if Path(path) == owned_path and kwargs.get("kind") == "staging file":
            validations += 1
            if validations == 1:
                owned_path.replace(outside_owned)
                try:
                    owned_path.symlink_to(missing_target)
                except OSError as error:
                    pytest.skip(f"symlinks unavailable: {error}")
        return snapshot

    monkeypatch.setattr(collect_mod, "_validated_regular_file", validate_then_swap)

    with pytest.raises(RuntimeError, match=r"staging|claim|changed"):
        collect_mod._claim_transaction_files(
            [attestation],
            context_path=tmp_path / "journal.json",
            require_present=True,
            transaction_id="1" * 32,
        )

    assert owned_path.is_symlink()
    assert outside_owned.read_bytes() == b"owned staging"


def test_claim_never_restores_replaced_quarantine_object_to_owned_path(tmp_path, monkeypatch):
    owned_path = tmp_path / "gemm_perf.txt"
    owned_path.write_bytes(b"owned staging")
    attestation = _file_attestation(owned_path)
    outside_owned = tmp_path / "outside-owned.txt"
    malicious = b"unowned quarantine replacement"
    claimed_path: Path | None = None
    real_validate = collect_mod._validated_regular_file

    def replace_claimed_then_validate(path, *args, **kwargs):
        nonlocal claimed_path
        path = Path(path)
        if kwargs.get("kind") == "claimed staging file" and claimed_path is None:
            claimed_path = path
            path.replace(outside_owned)
            path.write_bytes(malicious)
        return real_validate(path, *args, **kwargs)

    monkeypatch.setattr(collect_mod, "_validated_regular_file", replace_claimed_then_validate)

    with pytest.raises(RuntimeError, match=r"staging|claim|changed"):
        collect_mod._claim_transaction_files(
            [attestation],
            context_path=tmp_path / "journal.json",
            require_present=True,
            transaction_id="1" * 32,
        )

    assert claimed_path is not None
    assert owned_path.read_bytes() == b"owned staging"
    assert claimed_path.read_bytes() == malicious
    assert outside_owned.read_bytes() == b"owned staging"


def test_claim_never_overwrites_racing_deterministic_claim(tmp_path, monkeypatch):
    owned_path = tmp_path / "gemm_perf.txt"
    owned_path.write_bytes(b"owned staging")
    attestation = _file_attestation(owned_path)
    transaction_id = "1" * 32
    claim_path = collect_mod._transaction_claim_path(owned_path, transaction_id)
    competitor = b"competing claim"
    real_link = collect_mod.os.link

    def compete_before_link(source, target, *args, **kwargs):
        if Path(target) == claim_path:
            claim_path.write_bytes(competitor)
        return real_link(source, target, *args, **kwargs)

    monkeypatch.setattr(collect_mod.os, "link", compete_before_link)

    with pytest.raises(RuntimeError, match=r"claim|changed"):
        collect_mod._claim_transaction_files(
            [attestation],
            context_path=tmp_path / "journal.json",
            require_present=True,
            transaction_id=transaction_id,
        )

    assert owned_path.read_bytes() == b"owned staging"
    assert claim_path.read_bytes() == competitor


def test_main_rejects_staging_symlink_before_replacing_existing_parquet(tmp_path, monkeypatch):
    output_root = tmp_path / "out"
    output_root.mkdir()
    parquet_path = output_root / "gemm_perf.parquet"
    _write_parquet(parquet_path, [{"op": "old", "latency": 9.0}])
    parquet_before = parquet_path.read_bytes()

    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_path = _write_checkpoint(checkpoint_dir, done=["case-a"], failed=[])
    checkpoint_before = checkpoint_path.read_bytes()
    outside_staging = tmp_path / "outside.csv"
    outside_staging.write_text("op,latency\nmatmul,1.0\n", encoding="utf-8")
    staging_path = output_root / "gemm_perf.txt"

    def stage_output() -> None:
        try:
            staging_path.symlink_to(outside_staging)
        except OSError as error:
            pytest.skip(f"symlinks unavailable: {error}")

    with pytest.raises(RuntimeError, match="staging"):
        _run_main_with_staged_output(
            monkeypatch,
            output_root=output_root,
            checkpoint_dir=checkpoint_dir,
            provenance_ctx=_provenance_ctx(_collections()),
            stage_output=stage_output,
        )

    assert parquet_path.read_bytes() == parquet_before
    assert staging_path.is_symlink()
    assert outside_staging.read_text(encoding="utf-8") == "op,latency\nmatmul,1.0\n"
    assert checkpoint_path.read_bytes() == checkpoint_before
    assert not (output_root / "collection_meta.yaml").exists()
    assert not (output_root / collect_mod._SIDECAR_TRANSACTION_FILENAME).exists()


def test_main_rejects_checkpoint_symlink_before_replacing_existing_parquet(tmp_path, monkeypatch):
    output_root = tmp_path / "out"
    output_root.mkdir()
    parquet_path = output_root / "gemm_perf.parquet"
    _write_parquet(parquet_path, [{"op": "old", "latency": 9.0}])
    parquet_before = parquet_path.read_bytes()

    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_path = _write_checkpoint(checkpoint_dir, done=["case-a"], failed=[])
    outside_checkpoint = tmp_path / "outside-checkpoint.json"
    checkpoint_path.replace(outside_checkpoint)
    checkpoint_before = outside_checkpoint.read_bytes()
    try:
        checkpoint_path.symlink_to(outside_checkpoint)
    except OSError as error:
        pytest.skip(f"symlinks unavailable: {error}")
    staging_path = output_root / "gemm_perf.txt"

    with pytest.raises(RuntimeError, match="checkpoint"):
        _run_main_with_staged_output(
            monkeypatch,
            output_root=output_root,
            checkpoint_dir=checkpoint_dir,
            provenance_ctx=_provenance_ctx(_collections()),
            stage_output=lambda: staging_path.write_text("op,latency\nmatmul,1.0\n", encoding="utf-8"),
        )

    assert parquet_path.read_bytes() == parquet_before
    assert staging_path.read_text(encoding="utf-8") == "op,latency\nmatmul,1.0\n"
    assert checkpoint_path.is_symlink()
    assert outside_checkpoint.read_bytes() == checkpoint_before
    assert not (output_root / "collection_meta.yaml").exists()
    assert not (output_root / collect_mod._SIDECAR_TRANSACTION_FILENAME).exists()


def test_main_rejects_symlinked_checkpoint_backend_root_before_replacing_parquet(tmp_path, monkeypatch):
    output_root = tmp_path / "out"
    output_root.mkdir()
    parquet_path = output_root / "gemm_perf.parquet"
    _write_parquet(parquet_path, [{"op": "old", "latency": 9.0}])
    parquet_before = parquet_path.read_bytes()

    outside_checkpoint_dir = tmp_path / "outside-checkpoint"
    outside_checkpoint = _write_checkpoint(outside_checkpoint_dir, done=["case-a"], failed=[])
    checkpoint_before = outside_checkpoint.read_bytes()
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    try:
        (checkpoint_dir / BACKEND).symlink_to(outside_checkpoint_dir / BACKEND, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"symlinks unavailable: {error}")
    staging_path = output_root / "gemm_perf.txt"

    with pytest.raises(RuntimeError, match=r"checkpoint.*symlink|symlink.*checkpoint"):
        _run_main_with_staged_output(
            monkeypatch,
            output_root=output_root,
            checkpoint_dir=checkpoint_dir,
            provenance_ctx=_provenance_ctx(_collections()),
            stage_output=lambda: staging_path.write_text("op,latency\nmatmul,1.0\n", encoding="utf-8"),
        )

    assert parquet_path.read_bytes() == parquet_before
    assert not staging_path.exists()
    assert outside_checkpoint.read_bytes() == checkpoint_before
    assert (checkpoint_dir / BACKEND).is_symlink()
    assert not (output_root / "collection_meta.yaml").exists()
    assert not (output_root / collect_mod._SIDECAR_TRANSACTION_FILENAME).exists()


def test_main_rejects_producer_replacement_after_preflight_before_replacing_parquet(tmp_path, monkeypatch):
    output_root = tmp_path / "out"
    output_root.mkdir()
    parquet_path = output_root / "gemm_perf.parquet"
    _write_parquet(parquet_path, [{"op": "old", "latency": 9.0}])
    parquet_before = parquet_path.read_bytes()
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_path = _write_checkpoint(checkpoint_dir, done=["case-a"], failed=[])
    staging_path = output_root / "gemm_perf.txt"
    real_finalize = collect_mod.finalize_perf_files
    replacement_bytes: bytes | None = None

    def replace_checkpoint_then_finalize(*args, **kwargs):
        nonlocal replacement_bytes
        replacement_document = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        replacement_document.update(done=["case-b"], attempted=["case-b"])
        _replace_with_new_inode(checkpoint_path, json.dumps(replacement_document).encode())
        replacement_bytes = checkpoint_path.read_bytes()
        return real_finalize(*args, **kwargs)

    monkeypatch.setattr(collect_mod, "finalize_perf_files", replace_checkpoint_then_finalize)

    with pytest.raises(RuntimeError, match="changed after preflight"):
        _run_main_with_staged_output(
            monkeypatch,
            output_root=output_root,
            checkpoint_dir=checkpoint_dir,
            provenance_ctx=_provenance_ctx(_collections()),
            stage_output=lambda: staging_path.write_text("op,latency\nmatmul,1.0\n", encoding="utf-8"),
        )

    assert replacement_bytes is not None
    assert parquet_path.read_bytes() == parquet_before
    assert staging_path.read_bytes() == b"op,latency\nmatmul,1.0\n"
    assert checkpoint_path.read_bytes() == replacement_bytes
    assert not (output_root / "collection_meta.yaml").exists()
    assert not (output_root / collect_mod._SIDECAR_TRANSACTION_FILENAME).exists()


def test_main_rejects_staging_replacement_after_preflight_before_replacing_parquet(tmp_path, monkeypatch):
    output_root = tmp_path / "out"
    output_root.mkdir()
    parquet_path = output_root / "gemm_perf.parquet"
    _write_parquet(parquet_path, [{"op": "old", "latency": 9.0}])
    parquet_before = parquet_path.read_bytes()
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_path = _write_checkpoint(checkpoint_dir, done=["case-a"], failed=[])
    checkpoint_before = checkpoint_path.read_bytes()
    staging_path = output_root / "gemm_perf.txt"
    outside_original = tmp_path / "original-staging.txt"
    replacement = b"op,latency\nreplacement,8.0\n"
    real_finalize = collect_mod.finalize_perf_files

    def replace_staging_then_finalize(*args, **kwargs):
        staging_path.replace(outside_original)
        staging_path.write_bytes(replacement)
        return real_finalize(*args, **kwargs)

    monkeypatch.setattr(collect_mod, "finalize_perf_files", replace_staging_then_finalize)

    with pytest.raises(RuntimeError, match=r"staging|source|changed"):
        _run_main_with_staged_output(
            monkeypatch,
            output_root=output_root,
            checkpoint_dir=checkpoint_dir,
            provenance_ctx=_provenance_ctx(_collections()),
            stage_output=lambda: staging_path.write_text("op,latency\nmatmul,1.0\n", encoding="utf-8"),
        )

    assert parquet_path.read_bytes() == parquet_before
    assert staging_path.read_bytes() == replacement
    assert outside_original.read_bytes() == b"op,latency\nmatmul,1.0\n"
    assert checkpoint_path.read_bytes() == checkpoint_before
    assert not (output_root / "collection_meta.yaml").exists()
    assert not (output_root / collect_mod._SIDECAR_TRANSACTION_FILENAME).exists()


def test_main_rejects_sidecar_replacement_after_preflight_before_replacing_parquet(tmp_path, monkeypatch):
    output_root = tmp_path / "out"
    provenance.write_collection_meta(
        output_root,
        _sidecar_runtime(),
        {"gemm_perf": _collection_event()},
    )
    meta_path = output_root / "collection_meta.yaml"
    parquet_path = output_root / "gemm_perf.parquet"
    _write_parquet(parquet_path, [{"op": "old", "latency": 9.0}])
    parquet_before = parquet_path.read_bytes()
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_path = _write_checkpoint(checkpoint_dir, done=["case-a"], failed=[])
    checkpoint_before = checkpoint_path.read_bytes()
    staging_path = output_root / "gemm_perf.txt"
    replacement = b"schema_version: 99\nruntime: {}\ntables: {}\n"
    real_finalize = collect_mod.finalize_perf_files

    def replace_sidecar_then_finalize(*args, **kwargs):
        _replace_with_new_inode(meta_path, replacement)
        return real_finalize(*args, **kwargs)

    monkeypatch.setattr(collect_mod, "finalize_perf_files", replace_sidecar_then_finalize)

    with pytest.raises(RuntimeError, match=r"sidecar|document|changed"):
        _run_main_with_staged_output(
            monkeypatch,
            output_root=output_root,
            checkpoint_dir=checkpoint_dir,
            provenance_ctx=_provenance_ctx(_collections()),
            stage_output=lambda: staging_path.write_text("op,latency\nmatmul,1.0\n", encoding="utf-8"),
        )

    assert meta_path.read_bytes() == replacement
    assert parquet_path.read_bytes() == parquet_before
    assert checkpoint_path.read_bytes() == checkpoint_before
    assert staging_path.read_bytes() == b"op,latency\nmatmul,1.0\n"
    assert not (output_root / collect_mod._SIDECAR_STAGING_FILENAME).exists()
    assert not (output_root / collect_mod._SIDECAR_TRANSACTION_FILENAME).exists()


@pytest.mark.parametrize(
    "invalid_checkpoint",
    ["missing", "unreadable", "directory", "malformed ledger", "schema mismatch"],
)
def test_main_rejects_missing_or_mismatched_producer_before_replacing_parquet(
    tmp_path,
    monkeypatch,
    invalid_checkpoint,
):
    output_root = tmp_path / "out"
    output_root.mkdir()
    parquet_path = output_root / "gemm_perf.parquet"
    _write_parquet(parquet_path, [{"op": "old", "latency": 9.0}])
    parquet_before = parquet_path.read_bytes()

    checkpoint_dir = tmp_path / "checkpoint"
    if invalid_checkpoint == "missing":
        checkpoint_path = checkpoint_dir / BACKEND / f"{FULL_NAME}.json"
        checkpoint_before = None
    else:
        checkpoint_path = _write_checkpoint(checkpoint_dir, done=["case-a"], failed=[])
        if invalid_checkpoint == "unreadable":
            checkpoint_path.write_text("{", encoding="utf-8")
        elif invalid_checkpoint == "directory":
            checkpoint_path.unlink()
            checkpoint_path.mkdir()
        elif invalid_checkpoint == "malformed ledger":
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            checkpoint["attempted"] = "case-a"
            checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")
        else:
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            checkpoint["schema"] = "stale-schema"
            checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")
        checkpoint_before = None if invalid_checkpoint == "directory" else checkpoint_path.read_bytes()
    staging_path = output_root / "gemm_perf.txt"

    with pytest.raises(RuntimeError, match="checkpoint"):
        _run_main_with_staged_output(
            monkeypatch,
            output_root=output_root,
            checkpoint_dir=checkpoint_dir,
            provenance_ctx=_provenance_ctx(_collections()),
            stage_output=lambda: staging_path.write_text("op,latency\nmatmul,1.0\n", encoding="utf-8"),
        )

    assert parquet_path.read_bytes() == parquet_before
    assert staging_path.read_text(encoding="utf-8") == "op,latency\nmatmul,1.0\n"
    if invalid_checkpoint == "missing":
        assert not checkpoint_path.exists()
        assert not checkpoint_dir.exists()
    elif invalid_checkpoint == "directory":
        assert checkpoint_path.is_dir()
    else:
        assert checkpoint_path.read_bytes() == checkpoint_before
    assert not (output_root / "collection_meta.yaml").exists()
    assert not (output_root / collect_mod._SIDECAR_TRANSACTION_FILENAME).exists()


def test_main_prevalidates_mixed_tables_before_replacing_any_parquet(tmp_path, monkeypatch):
    output_root = tmp_path / "out"
    output_root.mkdir()
    gemm_parquet = output_root / "gemm_perf.parquet"
    moe_parquet = output_root / "moe_perf.parquet"
    _write_parquet(gemm_parquet, [{"op": "old-gemm", "latency": 9.0}])
    _write_parquet(moe_parquet, [{"op": "old-moe", "latency": 8.0}])
    parquet_before = {path: path.read_bytes() for path in (gemm_parquet, moe_parquet)}

    checkpoint_dir = tmp_path / "checkpoint"
    valid_checkpoint = _write_checkpoint(checkpoint_dir, done=["gemm-case"], failed=[])
    invalid_checkpoint = _write_checkpoint_for(
        checkpoint_dir,
        backend=BACKEND,
        full_name=f"{BACKEND}.moe",
        version="stale-version",
        done=["moe-case"],
        failed=[],
    )
    checkpoint_before = {path: path.read_bytes() for path in (valid_checkpoint, invalid_checkpoint)}
    collections = [
        *_collections(),
        {
            "name": BACKEND,
            "type": "moe",
            "module": REAL_MODULE,
            "run_func": "run_moe_torch",
            "perf_filename": "moe_perf.txt",
        },
    ]
    gemm_staging = output_root / "gemm_perf.txt"
    moe_staging = output_root / "moe_perf.txt"

    def stage_outputs() -> None:
        gemm_staging.write_text("op,latency\nmatmul,1.0\n", encoding="utf-8")
        moe_staging.write_text("op,latency\nmoe,2.0\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="framework_version"):
        _run_main_with_staged_output(
            monkeypatch,
            output_root=output_root,
            checkpoint_dir=checkpoint_dir,
            provenance_ctx=_provenance_ctx(collections),
            stage_output=stage_outputs,
        )

    assert {path: path.read_bytes() for path in (gemm_parquet, moe_parquet)} == parquet_before
    assert gemm_staging.exists()
    assert moe_staging.exists()
    assert {path: path.read_bytes() for path in (valid_checkpoint, invalid_checkpoint)} == checkpoint_before
    assert not (output_root / "collection_meta.yaml").exists()
    assert not (output_root / collect_mod._SIDECAR_TRANSACTION_FILENAME).exists()


@pytest.mark.parametrize("invalid_staging", ["active-lock", "unreadable"])
def test_main_prevalidates_every_staging_file_before_replacing_any_parquet(
    tmp_path,
    monkeypatch,
    invalid_staging,
):
    output_root = tmp_path / "out"
    output_root.mkdir()
    gemm_parquet = output_root / "gemm_perf.parquet"
    moe_parquet = output_root / "moe_perf.parquet"
    _write_parquet(gemm_parquet, [{"op": "old-gemm", "latency": 9.0}])
    _write_parquet(moe_parquet, [{"op": "old-moe", "latency": 8.0}])
    parquet_before = {path: path.read_bytes() for path in (gemm_parquet, moe_parquet)}

    checkpoint_dir = tmp_path / "checkpoint"
    _write_checkpoint(checkpoint_dir, done=["gemm-case"], failed=[])
    _write_checkpoint_for(
        checkpoint_dir,
        backend=BACKEND,
        full_name=f"{BACKEND}.moe",
        version="0.5.14",
        done=["moe-case"],
        failed=[],
    )
    collections = [
        *_collections(),
        {
            "name": BACKEND,
            "type": "moe",
            "module": REAL_MODULE,
            "run_func": "run_moe_torch",
            "perf_filename": "moe_perf.txt",
        },
    ]
    gemm_staging = output_root / "gemm_perf.txt"
    moe_staging = output_root / "moe_perf.txt"
    lock_path = Path(f"{moe_staging}.lock")

    def stage_outputs() -> None:
        gemm_staging.write_text("op,latency\nmatmul,1.0\n", encoding="utf-8")
        moe_staging.write_text("op,latency\nmoe,2.0\n", encoding="utf-8")
        if invalid_staging == "active-lock":
            lock_path.write_text("writer active", encoding="utf-8")
        else:
            moe_staging.chmod(0)

    try:
        with pytest.raises((OSError, RuntimeError), match=r"lock|read|Permission"):
            _run_main_with_staged_output(
                monkeypatch,
                output_root=output_root,
                checkpoint_dir=checkpoint_dir,
                provenance_ctx=_provenance_ctx(collections),
                stage_output=stage_outputs,
            )
    finally:
        if moe_staging.exists():
            moe_staging.chmod(0o600)

    assert {path: path.read_bytes() for path in (gemm_parquet, moe_parquet)} == parquet_before
    assert gemm_staging.exists()
    assert moe_staging.exists()


def test_main_prepares_every_staging_conversion_before_replacing_any_parquet(tmp_path, monkeypatch):
    output_root = tmp_path / "out"
    output_root.mkdir()
    gemm_parquet = output_root / "gemm_perf.parquet"
    moe_parquet = output_root / "moe_perf.parquet"
    _write_parquet(gemm_parquet, [{"op": "old-gemm", "latency": 9.0}])
    _write_parquet(moe_parquet, [{"op": "old-moe", "latency": 8.0}])
    parquet_before = {path: path.read_bytes() for path in (gemm_parquet, moe_parquet)}

    checkpoint_dir = tmp_path / "checkpoint"
    gemm_checkpoint = _write_checkpoint(checkpoint_dir, done=["gemm-case"], failed=[])
    moe_checkpoint = _write_checkpoint_for(
        checkpoint_dir,
        backend=BACKEND,
        full_name=f"{BACKEND}.moe",
        version="0.5.14",
        done=["moe-case"],
        failed=[],
    )
    checkpoint_before = {path: path.read_bytes() for path in (gemm_checkpoint, moe_checkpoint)}
    collections = [
        *_collections(),
        {
            "name": BACKEND,
            "type": "moe",
            "module": "collector.sglang.collect_moe",
            "run_func": "run_moe_torch",
            "perf_filename": "moe_perf.txt",
        },
    ]
    gemm_staging = output_root / "gemm_perf.txt"
    moe_staging = output_root / "moe_perf.txt"

    def stage_outputs() -> None:
        gemm_staging.write_text("op,latency\nmatmul,1.0\n", encoding="utf-8")
        moe_staging.write_text("op,latency\nmoe,2.0,unexpected\n", encoding="utf-8")

    with pytest.raises(ValueError):
        _run_main_with_staged_output(
            monkeypatch,
            output_root=output_root,
            checkpoint_dir=checkpoint_dir,
            provenance_ctx=_provenance_ctx(collections),
            stage_output=stage_outputs,
        )

    assert {path: path.read_bytes() for path in (gemm_parquet, moe_parquet)} == parquet_before
    assert gemm_staging.read_text(encoding="utf-8") == "op,latency\nmatmul,1.0\n"
    assert moe_staging.read_text(encoding="utf-8") == "op,latency\nmoe,2.0,unexpected\n"
    assert {path: path.read_bytes() for path in checkpoint_before} == checkpoint_before
    assert not (output_root / "collection_meta.yaml").exists()
    assert not (output_root / collect_mod._SIDECAR_TRANSACTION_FILENAME).exists()


@pytest.mark.parametrize("shared_table", [True, False], ids=["shared-table", "distinct-tables"])
def test_main_rejects_duplicate_attempt_ids_across_producers_before_replacing_parquet(
    tmp_path,
    monkeypatch,
    shared_table,
):
    output_root = tmp_path / "out"
    output_root.mkdir()
    checkpoint_dir = tmp_path / "checkpoint"
    if shared_table:
        collections = _shared_collections()
        table_names = ["mla_bmm_perf"]
    else:
        collections = [
            *_collections(),
            {
                "name": BACKEND,
                "type": "moe",
                "module": "collector.sglang.collect_moe",
                "run_func": "run_moe_torch",
                "perf_filename": "moe_perf.txt",
            },
        ]
        table_names = ["gemm_perf", "moe_perf"]

    checkpoints = [
        _write_checkpoint_for(
            checkpoint_dir,
            backend=BACKEND,
            full_name=f"{collection['name']}.{collection['type']}",
            version="0.5.14",
            done=["same-case"],
            failed=[],
        )
        for collection in collections
    ]
    checkpoint_before = {path: path.read_bytes() for path in checkpoints}
    parquet_paths = [output_root / f"{table}.parquet" for table in table_names]
    for parquet_path in parquet_paths:
        _write_parquet(parquet_path, [{"op": f"old-{parquet_path.stem}", "latency": 9.0}])
    parquet_before = {path: path.read_bytes() for path in parquet_paths}
    staging_paths = [output_root / f"{table}.txt" for table in table_names]

    def stage_outputs() -> None:
        for staging_path in staging_paths:
            staging_path.write_text(f"op,latency\nnew-{staging_path.stem},1.0\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match=r"attempt|case ID|duplicate"):
        _run_main_with_staged_output(
            monkeypatch,
            output_root=output_root,
            checkpoint_dir=checkpoint_dir,
            provenance_ctx=_provenance_ctx(collections),
            stage_output=stage_outputs,
        )

    assert {path: path.read_bytes() for path in parquet_paths} == parquet_before
    assert {path: path.read_bytes() for path in checkpoint_before} == checkpoint_before
    assert all(path.exists() for path in staging_paths)
    assert not (output_root / "collection_meta.yaml").exists()
    assert not (output_root / collect_mod._SIDECAR_TRANSACTION_FILENAME).exists()


@pytest.mark.parametrize(
    ("document_name", "invalid_kind"),
    [
        ("collection_meta.yaml", "symlink"),
        ("collection_meta.yaml", "directory"),
        ("collection_meta.yaml", "malformed"),
        ("collection_meta.yaml", "non-object"),
        (collect_mod._SIDECAR_STAGING_FILENAME, "symlink"),
        (collect_mod._SIDECAR_STAGING_FILENAME, "directory"),
        (collect_mod._SIDECAR_TRANSACTION_FILENAME, "symlink"),
        (collect_mod._SIDECAR_TRANSACTION_FILENAME, "directory"),
    ],
)
def test_main_rejects_invalid_sidecar_documents_before_replacing_parquet(
    tmp_path,
    monkeypatch,
    document_name,
    invalid_kind,
):
    output_root = tmp_path / "out"
    output_root.mkdir()
    parquet_path = output_root / "gemm_perf.parquet"
    _write_parquet(parquet_path, [{"op": "old", "latency": 9.0}])
    parquet_before = parquet_path.read_bytes()
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_path = _write_checkpoint(checkpoint_dir, done=["case-a"], failed=[])
    checkpoint_before = checkpoint_path.read_bytes()
    document_path = output_root / document_name
    outside_document = tmp_path / f"outside-{document_name}"
    if invalid_kind == "symlink":
        outside_document.write_text("outside document", encoding="utf-8")
        try:
            document_path.symlink_to(outside_document)
        except OSError as error:
            pytest.skip(f"symlinks unavailable: {error}")
    elif invalid_kind == "directory":
        document_path.mkdir()
    elif invalid_kind == "malformed":
        document_path.write_text("tables: [", encoding="utf-8")
    else:
        document_path.write_text("- not\n- an\n- object\n", encoding="utf-8")
    staging_path = output_root / "gemm_perf.txt"

    with pytest.raises(RuntimeError, match=r"sidecar|transaction|document|provenance"):
        _run_main_with_staged_output(
            monkeypatch,
            output_root=output_root,
            checkpoint_dir=checkpoint_dir,
            provenance_ctx=_provenance_ctx(_collections()),
            stage_output=lambda: staging_path.write_text("op,latency\nmatmul,1.0\n", encoding="utf-8"),
        )

    assert parquet_path.read_bytes() == parquet_before
    assert checkpoint_path.read_bytes() == checkpoint_before
    if invalid_kind == "symlink":
        assert document_path.is_symlink()
        assert outside_document.read_text(encoding="utf-8") == "outside document"
    elif invalid_kind == "directory":
        assert document_path.is_dir()
    else:
        assert document_path.exists()


@pytest.mark.parametrize(
    "document",
    [
        {
            "schema_version": 1,
            "provenance": "legacy",
            "runtime": _sidecar_runtime(),
            "tables": {"gemm_perf": {"status": "complete"}},
        },
        {
            "schema_version": 99,
            "runtime": _sidecar_runtime(),
            "tables": {"gemm_perf": _collection_event()},
        },
        {"schema_version": 1, "runtime": _sidecar_runtime(), "tables": []},
        {
            "schema_version": 1,
            "runtime": _sidecar_runtime(),
            "tables": {"gemm_perf": {"status": "complete"}},
        },
        {
            "schema_version": 2,
            "runtime": _sidecar_runtime(),
            "tables": {"gemm_perf": {"rows": 1, "status": "complete", "collections": []}},
        },
    ],
    ids=["legacy", "unknown-schema", "nonmapping-tables", "malformed-v1", "malformed-v2"],
)
def test_main_rejects_semantically_invalid_sidecar_before_replacing_parquet(tmp_path, monkeypatch, document):
    output_root = tmp_path / "out"
    output_root.mkdir()
    meta_path = output_root / "collection_meta.yaml"
    meta_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    meta_before = meta_path.read_bytes()
    parquet_path = output_root / "gemm_perf.parquet"
    _write_parquet(parquet_path, [{"op": "old", "latency": 9.0}])
    parquet_before = parquet_path.read_bytes()
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_path = _write_checkpoint(checkpoint_dir, done=["case-a"], failed=[])
    checkpoint_before = checkpoint_path.read_bytes()
    staging_path = output_root / "gemm_perf.txt"

    with pytest.raises(RuntimeError, match=r"sidecar document|schema|legacy|tables|gemm_perf"):
        _run_main_with_staged_output(
            monkeypatch,
            output_root=output_root,
            checkpoint_dir=checkpoint_dir,
            provenance_ctx=_provenance_ctx(_collections()),
            stage_output=lambda: staging_path.write_text("op,latency\nmatmul,1.0\n", encoding="utf-8"),
        )

    assert meta_path.read_bytes() == meta_before
    assert parquet_path.read_bytes() == parquet_before
    assert checkpoint_path.read_bytes() == checkpoint_before
    assert staging_path.read_bytes() == b"op,latency\nmatmul,1.0\n"
    assert not (output_root / collect_mod._SIDECAR_STAGING_FILENAME).exists()
    assert not (output_root / collect_mod._SIDECAR_TRANSACTION_FILENAME).exists()


def test_normal_commit_revalidates_sidecar_target_before_replacing_it(tmp_path, monkeypatch):
    output_root = tmp_path / "out"
    _write_reduced_collection_meta(
        output_root,
        _runtime_meta := {
            "framework": "sglang",
            "version": "0.5.14",
            "image": "lmsysorg/sglang:v0.5.14",
            "image_digest": "sha256:" + "0" * 64,
        },
        {"other_table": {"status": "complete"}},
    )
    meta_path = output_root / "collection_meta.yaml"
    original_meta = meta_path.read_bytes()
    outside_meta = tmp_path / "outside-meta.yaml"
    parquet_path = output_root / "gemm_perf.parquet"
    _write_parquet(parquet_path, [{"op": "matmul", "latency": 1.0}])
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_path = _write_checkpoint(checkpoint_dir, done=["case-a"], failed=[])
    checkpoint_before = checkpoint_path.read_bytes()
    staging_path = parquet_path.with_suffix(".txt")
    staging_path.write_text("op,latency\nmatmul,1.0\n", encoding="utf-8")
    staging_before = staging_path.read_bytes()
    finalization_info = {
        parquet_path.resolve(): _finalization_fact(
            staging_path,
            new_rows=1,
            merged_existing=False,
        )
    }
    real_write = provenance.write_collection_meta

    def render_then_replace_target(*args, **kwargs):
        rendered = real_write(*args, **kwargs)
        meta_path.replace(outside_meta)
        try:
            meta_path.symlink_to(outside_meta)
        except OSError as error:
            pytest.skip(f"symlinks unavailable: {error}")
        return rendered

    monkeypatch.setattr(provenance, "write_collection_meta", render_then_replace_target)

    with pytest.raises(RuntimeError, match=r"sidecar|document"):
        collect_mod._write_collector_provenance(
            output_root,
            [parquet_path],
            _provenance_ctx(_collections()),
            run_errors=[],
            backend=BACKEND,
            checkpoint_dir=str(checkpoint_dir),
            finalization_info=finalization_info,
        )

    assert _runtime_meta["version"] == "0.5.14"
    assert meta_path.is_symlink()
    assert outside_meta.read_bytes() == original_meta
    assert checkpoint_path.read_bytes() == checkpoint_before
    assert staging_path.read_bytes() == staging_before
    assert not (output_root / collect_mod._SIDECAR_STAGING_FILENAME).exists()
    assert not (output_root / collect_mod._SIDECAR_TRANSACTION_FILENAME).exists()


def test_normal_commit_rejects_pending_sidecar_swap_after_final_validation(tmp_path, monkeypatch):
    output_root = tmp_path / "out"
    output_root.mkdir()
    parquet_path = output_root / "gemm_perf.parquet"
    _write_parquet(parquet_path, [{"op": "matmul", "latency": 1.0}])
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_path = _write_checkpoint(checkpoint_dir, done=["case-a"], failed=[])
    staging_path = parquet_path.with_suffix(".txt")
    staging_path.write_bytes(b"owned staging")
    pending_path = output_root / collect_mod._SIDECAR_STAGING_FILENAME
    malicious = b"malicious: replacement\n"
    real_validate = collect_mod._validated_sidecar_document
    final_validations = 0

    def validate_then_swap(*args, **kwargs):
        nonlocal final_validations
        snapshot = real_validate(*args, **kwargs)
        if Path(args[0]) == pending_path and kwargs.get("expected_identity") is not None:
            final_validations += 1
            if final_validations == 2:
                _replace_with_new_inode(pending_path, malicious)
        return snapshot

    monkeypatch.setattr(collect_mod, "_validated_sidecar_document", validate_then_swap)

    with pytest.raises(RuntimeError, match=r"sidecar|staging|changed"):
        collect_mod._write_collector_provenance(
            output_root,
            [parquet_path],
            _provenance_ctx(_collections()),
            run_errors=[],
            backend=BACKEND,
            checkpoint_dir=str(checkpoint_dir),
            finalization_info={
                parquet_path.resolve(): _finalization_fact(
                    staging_path,
                    new_rows=1,
                    merged_existing=False,
                )
            },
        )

    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert final_validations == 2
    assert checkpoint["attempted"] == ["case-a"]
    assert checkpoint[collect_mod._SIDECAR_TRANSACTION_FIELD]
    assert pending_path.read_bytes() == malicious
    assert staging_path.read_bytes() == b"owned staging"
    assert not (output_root / "collection_meta.yaml").exists()
    assert (output_root / collect_mod._SIDECAR_TRANSACTION_FILENAME).exists()


@pytest.mark.parametrize("existing_target", [False, True], ids=["created", "swapped"])
def test_normal_commit_rejects_sidecar_target_replacement_after_final_validation(
    tmp_path,
    monkeypatch,
    existing_target,
):
    output_root = tmp_path / "out"
    if existing_target:
        provenance.write_collection_meta(
            output_root,
            _sidecar_runtime(),
            {"other_table": _collection_event()},
        )
    else:
        output_root.mkdir()
    meta_path = output_root / "collection_meta.yaml"
    parquet_path = output_root / "gemm_perf.parquet"
    _write_parquet(parquet_path, [{"op": "matmul", "latency": 1.0}])
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_path = _write_checkpoint(checkpoint_dir, done=["case-a"], failed=[])
    staging_path = parquet_path.with_suffix(".txt")
    staging_path.write_bytes(b"owned staging")
    pending_path = output_root / collect_mod._SIDECAR_STAGING_FILENAME
    malicious = b"malicious: target replacement\n"
    real_validate = collect_mod._validated_sidecar_document
    final_validations = 0

    def validate_then_replace_target(*args, **kwargs):
        nonlocal final_validations
        snapshot = real_validate(*args, **kwargs)
        if Path(args[0]) == pending_path and kwargs.get("expected_identity") is not None:
            final_validations += 1
            if final_validations == 2:
                if existing_target:
                    _replace_with_new_inode(meta_path, malicious)
                else:
                    meta_path.write_bytes(malicious)
        return snapshot

    monkeypatch.setattr(collect_mod, "_validated_sidecar_document", validate_then_replace_target)

    with pytest.raises(RuntimeError, match=r"sidecar|target|changed"):
        collect_mod._write_collector_provenance(
            output_root,
            [parquet_path],
            _provenance_ctx(_collections()),
            run_errors=[],
            backend=BACKEND,
            checkpoint_dir=str(checkpoint_dir),
            finalization_info={
                parquet_path.resolve(): _finalization_fact(
                    staging_path,
                    new_rows=1,
                    merged_existing=False,
                )
            },
        )

    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert final_validations == 2
    assert checkpoint["attempted"] == ["case-a"]
    assert checkpoint[collect_mod._SIDECAR_TRANSACTION_FIELD]
    assert meta_path.read_bytes() == malicious
    assert pending_path.exists()
    assert staging_path.read_bytes() == b"owned staging"
    assert (output_root / collect_mod._SIDECAR_TRANSACTION_FILENAME).exists()


@pytest.mark.parametrize("crash_point", ["after-first-claim", "before-link"])
def test_recovery_completes_deterministic_sidecar_claim_crash(tmp_path, monkeypatch, crash_point):
    class SimulatedCrash(BaseException):
        pass

    output_root = tmp_path / "out"
    if crash_point == "before-link":
        provenance.write_collection_meta(
            output_root,
            _sidecar_runtime(),
            {"other_table": _collection_event()},
        )
    else:
        output_root.mkdir()
    parquet_path = output_root / "gemm_perf.parquet"
    _write_parquet(parquet_path, [{"op": "matmul", "latency": 1.0}])
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_path = _write_checkpoint(checkpoint_dir, done=["case-a"], failed=[])
    staging_path = parquet_path.with_suffix(".txt")
    staging_path.write_bytes(b"owned staging")
    meta_path = output_root / "collection_meta.yaml"

    with monkeypatch.context() as fault:
        if crash_point == "after-first-claim":
            real_link = collect_mod.os.link

            def link_then_crash(source, target, *args, **kwargs):
                real_link(source, target, *args, **kwargs)
                if str(target).endswith(".transaction-claim"):
                    raise SimulatedCrash

            fault.setattr(collect_mod.os, "link", link_then_crash)
        else:
            real_atomic_write = collect_mod._atomic_write_bytes

            def crash_before_link(path, *args, **kwargs):
                if Path(path) == meta_path and kwargs.get("replace_existing") is False:
                    raise SimulatedCrash
                return real_atomic_write(path, *args, **kwargs)

            fault.setattr(collect_mod, "_atomic_write_bytes", crash_before_link)

        with pytest.raises(SimulatedCrash):
            collect_mod._write_collector_provenance(
                output_root,
                [parquet_path],
                _provenance_ctx(_collections()),
                run_errors=[],
                backend=BACKEND,
                checkpoint_dir=str(checkpoint_dir),
                finalization_info={
                    parquet_path.resolve(): _finalization_fact(
                        staging_path,
                        new_rows=1,
                        merged_existing=False,
                    )
                },
            )

    journal_path = output_root / collect_mod._SIDECAR_TRANSACTION_FILENAME
    transaction = json.loads(journal_path.read_text(encoding="utf-8"))
    transaction_id = transaction["transaction_id"]
    pending_path = output_root / collect_mod._SIDECAR_STAGING_FILENAME
    pending_claim = collect_mod._transaction_claim_path(pending_path, transaction_id)
    target_claim = collect_mod._transaction_claim_path(meta_path, transaction_id)
    assert pending_claim.exists()
    assert pending_path.exists() is (crash_point == "after-first-claim")
    if crash_point == "before-link":
        assert target_claim.exists()
        assert not meta_path.exists()

    assert (
        collect_mod._recover_collector_provenance_transaction(
            output_root,
            backend=BACKEND,
            checkpoint_dir=str(checkpoint_dir),
        )
        == meta_path
    )

    assert _attempted(checkpoint_path) == set()
    assert not staging_path.exists()
    assert not pending_path.exists()
    assert not pending_claim.exists()
    assert not target_claim.exists()
    assert not journal_path.exists()
    doc = yaml.safe_load(meta_path.read_text(encoding="utf-8"))
    assert "gemm_perf" in doc["tables"]


def test_recovery_cleans_deterministic_staging_claim_after_cleanup_crash(tmp_path, monkeypatch):
    class SimulatedCrash(BaseException):
        pass

    output_root = tmp_path / "out"
    output_root.mkdir()
    parquet_path = output_root / "gemm_perf.parquet"
    _write_parquet(parquet_path, [{"op": "matmul", "latency": 1.0}])
    staging_path = parquet_path.with_suffix(".txt")
    staging_path.write_bytes(b"owned staging")
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_path = _write_checkpoint(checkpoint_dir, done=["case-a"], failed=[])
    real_delete = collect_mod._delete_claimed_transaction_files

    def crash_before_staging_claim_delete(claimed_files, *args, **kwargs):
        if any(claimed.original.path == staging_path for claimed in claimed_files):
            raise SimulatedCrash
        return real_delete(claimed_files, *args, **kwargs)

    with monkeypatch.context() as fault:
        fault.setattr(collect_mod, "_delete_claimed_transaction_files", crash_before_staging_claim_delete)
        with pytest.raises(SimulatedCrash):
            collect_mod._write_collector_provenance(
                output_root,
                [parquet_path],
                _provenance_ctx(_collections()),
                run_errors=[],
                backend=BACKEND,
                checkpoint_dir=str(checkpoint_dir),
                finalization_info={
                    parquet_path.resolve(): _finalization_fact(
                        staging_path,
                        new_rows=1,
                        merged_existing=False,
                    )
                },
            )

    journal_path = output_root / collect_mod._SIDECAR_TRANSACTION_FILENAME
    transaction_id = json.loads(journal_path.read_text(encoding="utf-8"))["transaction_id"]
    staging_claim = collect_mod._transaction_claim_path(staging_path, transaction_id)
    assert not staging_path.exists()
    assert staging_claim.read_bytes() == b"owned staging"

    assert (
        collect_mod._recover_collector_provenance_transaction(
            output_root,
            backend=BACKEND,
            checkpoint_dir=str(checkpoint_dir),
        )
        is None
    )

    assert _attempted(checkpoint_path) == set()
    assert not staging_claim.exists()
    assert not journal_path.exists()


def test_recovery_validates_every_participant_before_restoring_sidecar_claim(tmp_path):
    output_root = tmp_path / "out"
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_path = _write_checkpoint(
        checkpoint_dir,
        done=["case-a"],
        failed=[],
        attempted=["case-a"],
    )
    staging_path = output_root / "gemm_perf.txt"
    staging_path.parent.mkdir(parents=True)
    staging_path.write_bytes(b"owned staging")
    _write_sidecar_transaction(
        output_root,
        [(checkpoint_path, {"case-a"})],
        [staging_path],
        checkpoint_dir=checkpoint_dir,
        tagged=set(),
        pending_document=True,
    )
    journal_path = output_root / collect_mod._SIDECAR_TRANSACTION_FILENAME
    transaction = json.loads(journal_path.read_text(encoding="utf-8"))
    transaction["checkpoints"][0]["unexpected"] = True
    journal_path.write_text(json.dumps(transaction), encoding="utf-8")
    pending_path = output_root / collect_mod._SIDECAR_STAGING_FILENAME
    pending_claim = collect_mod._transaction_claim_path(pending_path, transaction["transaction_id"])
    pending_path.replace(pending_claim)
    checkpoint_before = checkpoint_path.read_bytes()
    claim_before = pending_claim.read_bytes()

    with pytest.raises(RuntimeError, match="participant"):
        collect_mod._recover_collector_provenance_transaction(
            output_root,
            backend=BACKEND,
            checkpoint_dir=str(checkpoint_dir),
        )

    assert checkpoint_path.read_bytes() == checkpoint_before
    assert not pending_path.exists()
    assert pending_claim.read_bytes() == claim_before
    assert staging_path.read_bytes() == b"owned staging"
    assert journal_path.exists()


def test_recovery_rejects_same_byte_new_inode_pending_sidecar(tmp_path):
    output_root = tmp_path / "out"
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_path = _write_checkpoint(
        checkpoint_dir,
        done=["case-a"],
        failed=[],
        attempted=["case-a"],
    )
    staging_path = output_root / "gemm_perf.txt"
    staging_path.parent.mkdir(parents=True)
    staging_path.write_bytes(b"owned staging")
    _write_sidecar_transaction(
        output_root,
        [(checkpoint_path, {"case-a"})],
        [staging_path],
        checkpoint_dir=checkpoint_dir,
        tagged=set(),
        pending_document=True,
    )
    pending_path = output_root / collect_mod._SIDECAR_STAGING_FILENAME
    pending_bytes = pending_path.read_bytes()
    old_inode = pending_path.stat().st_ino
    new_inode = _replace_with_new_inode(pending_path, pending_bytes)
    assert new_inode != old_inode
    checkpoint_before = checkpoint_path.read_bytes()
    journal_path = output_root / collect_mod._SIDECAR_TRANSACTION_FILENAME
    journal_before = journal_path.read_bytes()

    with pytest.raises(RuntimeError, match=r"sidecar|document|changed"):
        collect_mod._recover_collector_provenance_transaction(
            output_root,
            backend=BACKEND,
            checkpoint_dir=str(checkpoint_dir),
        )

    assert checkpoint_path.read_bytes() == checkpoint_before
    assert pending_path.read_bytes() == pending_bytes
    assert pending_path.stat().st_ino == new_inode
    assert staging_path.read_bytes() == b"owned staging"
    assert journal_path.read_bytes() == journal_before
    assert not (output_root / "collection_meta.yaml").exists()


def test_normal_commit_rejects_journal_replacement_before_checkpoint_tag(tmp_path, monkeypatch):
    output_root = tmp_path / "out"
    output_root.mkdir()
    parquet_path = output_root / "gemm_perf.parquet"
    _write_parquet(parquet_path, [{"op": "matmul", "latency": 1.0}])
    staging_path = parquet_path.with_suffix(".txt")
    staging_path.write_bytes(b"owned staging")
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_path = _write_checkpoint(checkpoint_dir, done=["case-a"], failed=[])
    checkpoint_before = checkpoint_path.read_bytes()
    journal_path = output_root / collect_mod._SIDECAR_TRANSACTION_FILENAME
    malicious = b'{"malicious": true}'
    real_tag = collect_mod._tag_checkpoint_sidecar_transaction

    def replace_journal_then_tag(*args, **kwargs):
        _replace_with_new_inode(journal_path, malicious)
        return real_tag(*args, **kwargs)

    monkeypatch.setattr(collect_mod, "_tag_checkpoint_sidecar_transaction", replace_journal_then_tag)

    with pytest.raises(RuntimeError, match=r"journal|transaction|changed"):
        collect_mod._write_collector_provenance(
            output_root,
            [parquet_path],
            _provenance_ctx(_collections()),
            run_errors=[],
            backend=BACKEND,
            checkpoint_dir=str(checkpoint_dir),
            finalization_info={
                parquet_path.resolve(): _finalization_fact(
                    staging_path,
                    new_rows=1,
                    merged_existing=False,
                )
            },
        )

    assert checkpoint_path.read_bytes() == checkpoint_before
    assert staging_path.read_bytes() == b"owned staging"
    assert journal_path.read_bytes() == malicious
    assert not (output_root / "collection_meta.yaml").exists()


@pytest.mark.parametrize("replacement", ["rewrite", "symlink"])
def test_normal_commit_rejects_replaced_pending_sidecar_after_checkpoint_tag(tmp_path, monkeypatch, replacement):
    output_root = tmp_path / "out"
    output_root.mkdir()
    parquet_path = output_root / "gemm_perf.parquet"
    _write_parquet(parquet_path, [{"op": "matmul", "latency": 1.0}])
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_path = _write_checkpoint(checkpoint_dir, done=["case-a"], failed=[])
    finalization_info = _finalization_info_for(parquet_path)
    pending_path = output_root / collect_mod._SIDECAR_STAGING_FILENAME
    outside_document = tmp_path / "outside-pending.yaml"
    outside_document.write_text("outside document\n", encoding="utf-8")
    outside_before = outside_document.read_bytes()
    real_tag = collect_mod._tag_checkpoint_sidecar_transaction

    def tag_then_replace_pending(*args, **kwargs):
        real_tag(*args, **kwargs)
        if replacement == "rewrite":
            pending_path.write_text("tampered: true\n", encoding="utf-8")
        else:
            pending_path.unlink()
            try:
                pending_path.symlink_to(outside_document)
            except OSError as error:
                pytest.skip(f"symlinks unavailable: {error}")

    monkeypatch.setattr(collect_mod, "_tag_checkpoint_sidecar_transaction", tag_then_replace_pending)

    with pytest.raises(RuntimeError, match=r"sidecar|document|pending"):
        collect_mod._write_collector_provenance(
            output_root,
            [parquet_path],
            _provenance_ctx(_collections()),
            run_errors=[],
            backend=BACKEND,
            checkpoint_dir=str(checkpoint_dir),
            finalization_info=finalization_info,
        )

    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert checkpoint["attempted"] == ["case-a"]
    assert checkpoint[collect_mod._SIDECAR_TRANSACTION_FIELD]
    if replacement == "symlink":
        assert pending_path.is_symlink()
    else:
        assert pending_path.read_text(encoding="utf-8") == "tampered: true\n"
    assert outside_document.read_bytes() == outside_before
    assert not (output_root / "collection_meta.yaml").exists()
    assert (output_root / collect_mod._SIDECAR_TRANSACTION_FILENAME).exists()


def test_writes_sidecar_with_rows_case_plan_hash_status_and_collector_ref(tmp_path):
    output_root = tmp_path / "out"
    parquet_path = output_root / "gemm_perf.parquet"
    _write_parquet(parquet_path, [{"op": "matmul", "latency": 1.0}, {"op": "matmul", "latency": 2.0}])

    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_path = _write_checkpoint(checkpoint_dir, done=["case-a", "case-b"], failed=[])

    collect_mod._write_collector_provenance(
        output_root,
        [parquet_path],
        _provenance_ctx(_collections()),
        run_errors=[],
        backend=BACKEND,
        checkpoint_dir=str(checkpoint_dir),
        finalization_info=_finalization_info_for(parquet_path),
    )

    meta_path = output_root / "collection_meta.yaml"
    assert meta_path.exists()
    doc = yaml.safe_load(meta_path.read_text(encoding="utf-8"))
    table = doc["tables"]["gemm_perf"]

    assert table["rows"] == 2
    assert table["case_plan_hash"] == provenance.case_plan_hash(["case-a", "case-b"])
    assert table["status"] == provenance.STATUS_COMPLETE
    assert table["collector_ref"] == collect_mod._git_collector_ref(collect_mod._REPO_ROOT)
    assert table["collector_hash"].startswith("sha256:")
    assert _attempted(checkpoint_path) == set()


def test_status_complete_with_recorded_case_failures(tmp_path):
    """Recorded per-case failures are DATA and do not demote the table
    (owner decision tianhaox 2026-08-08, PR #1486) — the failed case still
    participates in case_plan_hash as an attempted case."""
    output_root = tmp_path / "out"
    parquet_path = output_root / "gemm_perf.parquet"
    _write_parquet(parquet_path, [{"op": "matmul", "latency": 1.0}])

    checkpoint_dir = tmp_path / "checkpoint"
    _write_checkpoint(checkpoint_dir, done=["case-a"], failed=["case-b"])

    collect_mod._write_collector_provenance(
        output_root,
        [parquet_path],
        _provenance_ctx(_collections()),
        run_errors=[],
        backend=BACKEND,
        checkpoint_dir=str(checkpoint_dir),
        finalization_info=_finalization_info_for(parquet_path),
    )

    doc = yaml.safe_load((output_root / "collection_meta.yaml").read_text(encoding="utf-8"))
    table = doc["tables"]["gemm_perf"]
    assert table["status"] == provenance.STATUS_COMPLETE
    assert table["case_plan_hash"] == provenance.case_plan_hash(["case-a", "case-b"])


def test_status_partial_when_module_collection_failure_recorded(tmp_path):
    """Even with zero unresolved checkpoint failures, a ModuleCollectionFailure
    for this table's producing module (design §5: "op failed before running a
    single case") forces status: partial."""
    output_root = tmp_path / "out"
    parquet_path = output_root / "gemm_perf.parquet"
    _write_parquet(parquet_path, [{"op": "matmul", "latency": 1.0}])

    checkpoint_dir = tmp_path / "checkpoint"
    _write_checkpoint(checkpoint_dir, done=["case-a"], failed=[])

    run_errors = [{"module": FULL_NAME, "error_type": "ModuleCollectionFailure"}]

    collect_mod._write_collector_provenance(
        output_root,
        [parquet_path],
        _provenance_ctx(_collections()),
        run_errors=run_errors,
        backend=BACKEND,
        checkpoint_dir=str(checkpoint_dir),
        finalization_info=_finalization_info_for(parquet_path),
    )

    doc = yaml.safe_load((output_root / "collection_meta.yaml").read_text(encoding="utf-8"))
    assert doc["tables"]["gemm_perf"]["status"] == provenance.STATUS_PARTIAL


def test_finalize_raises_when_no_op_has_checkpoint_evidence(tmp_path):
    """A parquet table whose ops ALL lack checkpoint files must fail loudly:
    writing status: complete with a case_plan_hash over an empty case set would
    be a fabricated attestation (collector doctrine: run it or raise)."""
    output_root = tmp_path / "out"
    parquet_path = output_root / "gemm_perf.parquet"
    _write_parquet(parquet_path, [{"op": "matmul", "latency": 1.0}])

    checkpoint_dir = tmp_path / "checkpoint"  # deliberately no checkpoint written

    with pytest.raises(RuntimeError) as excinfo:
        collect_mod._write_collector_provenance(
            output_root,
            [parquet_path],
            _provenance_ctx(_collections()),
            run_errors=[],
            backend=BACKEND,
            checkpoint_dir=str(checkpoint_dir),
            finalization_info=_finalization_info_for(parquet_path),
        )

    message = str(excinfo.value)
    assert "gemm_perf" in message
    assert FULL_NAME in message
    assert str(checkpoint_dir.resolve() / BACKEND) in message
    # No sidecar may be written for the unattestable table.
    assert not (output_root / "collection_meta.yaml").exists()


def test_finalize_raises_when_all_checkpoints_are_unreadable(tmp_path):
    """Corrupt checkpoint JSON for every op of a table is the same zero-evidence
    condition as missing checkpoints and must raise, not degrade to complete."""
    output_root = tmp_path / "out"
    parquet_path = output_root / "gemm_perf.parquet"
    _write_parquet(parquet_path, [{"op": "matmul", "latency": 1.0}])

    checkpoint_dir = tmp_path / "checkpoint"
    corrupt_path = checkpoint_dir / BACKEND / f"{FULL_NAME}.json"
    corrupt_path.parent.mkdir(parents=True, exist_ok=True)
    corrupt_path.write_text("{not valid json")

    with pytest.raises(RuntimeError, match="checkpoint"):
        collect_mod._write_collector_provenance(
            output_root,
            [parquet_path],
            _provenance_ctx(_collections()),
            run_errors=[],
            backend=BACKEND,
            checkpoint_dir=str(checkpoint_dir),
            finalization_info=_finalization_info_for(parquet_path),
        )

    assert not (output_root / "collection_meta.yaml").exists()


def test_existing_sidecar_merge_preserves_prior_tables(tmp_path):
    output_root = tmp_path / "out"
    output_root.mkdir(parents=True)
    existing_doc = {
        "schema_version": 1,
        "provenance": "local",
        "runtime": {
            "framework": "sglang",
            "version": "0.5.14",
            "image": "lmsysorg/sglang:v0.5.14",
            "image_digest": "sha256:" + "0" * 64,
        },
        "tables": {"other_table": {"status": "complete"}},
    }
    (output_root / "collection_meta.yaml").write_text(yaml.safe_dump(existing_doc, sort_keys=False))

    parquet_path = output_root / "gemm_perf.parquet"
    _write_parquet(parquet_path, [{"op": "matmul", "latency": 1.0}])
    checkpoint_dir = tmp_path / "checkpoint"
    _write_checkpoint(checkpoint_dir, done=["case-a"], failed=[])

    collect_mod._write_collector_provenance(
        output_root,
        [parquet_path],
        _provenance_ctx(_collections()),
        run_errors=[],
        backend=BACKEND,
        checkpoint_dir=str(checkpoint_dir),
        finalization_info=_finalization_info_for(parquet_path),
    )

    doc = yaml.safe_load((output_root / "collection_meta.yaml").read_text(encoding="utf-8"))
    assert set(doc["tables"]) == {"other_table", "gemm_perf"}
    assert doc["tables"]["other_table"] == {"status": "complete"}


def test_existing_v2_same_table_history_appends_fresh_event(tmp_path):
    output_root = tmp_path / "out"
    prior_event = {
        "collector_ref": "a" * 40,
        "collector_hash": "sha256:" + "b" * 64,
        "case_plan_hash": "sha256:" + "c" * 64,
        "collected_at": "2026-08-10",
        "rows": 1,
        "status": "complete",
        "runtime": {
            "framework": "sglang",
            "version": "0.5.14",
            "image": "lmsysorg/sglang:v0.5.14",
            "image_digest": "sha256:" + "f" * 64,
        },
    }
    provenance.write_collection_meta(
        output_root,
        {
            "framework": "sglang",
            "version": "0.5.14",
            "image": "lmsysorg/sglang:v0.5.14",
            "image_digest": "sha256:" + "0" * 64,
        },
        {"gemm_perf": {"rows": 1, "status": "complete", "collections": [prior_event]}},
        provenance_tier="local",
    )

    parquet_path = output_root / "gemm_perf.parquet"
    _write_parquet(parquet_path, [{"op": "matmul", "latency": 1.0}])
    perf_path = output_root / "gemm_perf.txt"
    perf_path.write_text("op,latency\nsoftmax,2.0\nsoftmax,3.0\n")
    finalization_info: dict[Path, collect_mod.PerfFinalizationInfo] = {}
    assert collect_mod.finalize_perf_files([perf_path], delete_source=False, finalization_info=finalization_info) == [
        parquet_path
    ]
    assert pq.read_metadata(parquet_path).num_rows == 2
    assert pq.read_table(parquet_path).to_pylist()[-1] == {"op": "softmax", "latency": 3.0}
    assert finalization_info[parquet_path.resolve()] == _finalization_fact(
        perf_path,
        new_rows=1,
        merged_existing=True,
    )
    checkpoint_dir = tmp_path / "checkpoint"
    _write_checkpoint(checkpoint_dir, done=["case-new"], failed=[])

    collect_mod._write_collector_provenance(
        output_root,
        [parquet_path],
        _provenance_ctx(_collections()),
        run_errors=[],
        backend=BACKEND,
        checkpoint_dir=str(checkpoint_dir),
        finalization_info=finalization_info,
    )

    doc = yaml.safe_load((output_root / "collection_meta.yaml").read_text(encoding="utf-8"))
    table = doc["tables"]["gemm_perf"]
    assert doc["schema_version"] == 2
    assert doc["provenance"] == "local"
    assert table["collections"][0] == prior_event
    assert table["collections"][1]["case_plan_hash"] == provenance.case_plan_hash(["case-new"])
    assert table["collections"][1]["rows"] == 1
    assert table["rows"] == 2


def test_retry_event_hash_attests_only_cases_attempted_this_invocation(tmp_path):
    output_root = tmp_path / "out"
    prior_event = {
        "collector_ref": "a" * 40,
        "collector_hash": "sha256:" + "b" * 64,
        "case_plan_hash": provenance.case_plan_hash(["case-old", "case-retried"]),
        "collected_at": "2026-08-10",
        "rows": 1,
        "status": "complete",
    }
    provenance.write_collection_meta(
        output_root,
        {
            "framework": "sglang",
            "version": "0.5.14",
            "image": "lmsysorg/sglang:v0.5.14",
            "image_digest": "sha256:" + "0" * 64,
        },
        {"gemm_perf": prior_event},
    )

    parquet_path = output_root / "gemm_perf.parquet"
    _write_parquet(parquet_path, [{"op": "old", "latency": 1.0}])
    perf_path = output_root / "gemm_perf.txt"
    perf_path.write_text("op,latency\nretried,2.0\n")
    finalization_info: dict[Path, collect_mod.PerfFinalizationInfo] = {}
    assert collect_mod.finalize_perf_files([perf_path], delete_source=False, finalization_info=finalization_info) == [
        parquet_path
    ]

    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_path = _write_checkpoint(
        checkpoint_dir,
        done=["case-old", "case-retried"],
        failed=[],
        attempted=["case-retried"],
    )
    collect_mod._write_collector_provenance(
        output_root,
        [parquet_path],
        _provenance_ctx(_collections()),
        run_errors=[],
        backend=BACKEND,
        checkpoint_dir=str(checkpoint_dir),
        finalization_info=finalization_info,
    )

    doc = yaml.safe_load((output_root / "collection_meta.yaml").read_text(encoding="utf-8"))
    retry_event = doc["tables"]["gemm_perf"]["collections"][1]
    assert retry_event["case_plan_hash"] == provenance.case_plan_hash(["case-retried"])
    assert _attempted(checkpoint_path) == set()


def test_deduped_zero_row_event_still_attests_its_attempted_cases(tmp_path):
    output_root = tmp_path / "out"
    prior_event = {
        "collector_ref": "a" * 40,
        "collector_hash": "sha256:" + "b" * 64,
        "case_plan_hash": provenance.case_plan_hash(["case-old"]),
        "collected_at": "2026-08-10",
        "rows": 1,
        "status": "complete",
    }
    provenance.write_collection_meta(
        output_root,
        {
            "framework": "sglang",
            "version": "0.5.14",
            "image": "lmsysorg/sglang:v0.5.14",
            "image_digest": "sha256:" + "0" * 64,
        },
        {"gemm_perf": prior_event},
    )
    parquet_path = output_root / "gemm_perf.parquet"
    _write_parquet(parquet_path, [{"op": "old", "latency": 1.0}])
    parquet_path.with_suffix(".txt").write_text("retained staging\n", encoding="utf-8")
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_path = _write_checkpoint(
        checkpoint_dir,
        done=["case-old"],
        failed=[],
        attempted=["case-deduped-a", "case-deduped-b"],
    )

    collect_mod._write_collector_provenance(
        output_root,
        [parquet_path],
        _provenance_ctx(_collections()),
        run_errors=[],
        backend=BACKEND,
        checkpoint_dir=str(checkpoint_dir),
        finalization_info={
            parquet_path.resolve(): _finalization_fact(
                parquet_path.with_suffix(".txt"),
                new_rows=0,
                merged_existing=True,
            )
        },
    )

    doc = yaml.safe_load((output_root / "collection_meta.yaml").read_text(encoding="utf-8"))
    event = doc["tables"]["gemm_perf"]["collections"][1]
    assert event["rows"] == 0
    assert event["case_plan_hash"] == provenance.case_plan_hash(["case-deduped-a", "case-deduped-b"])
    assert _attempted(checkpoint_path) == set()


def test_failed_sidecar_write_keeps_pending_attempts_for_retry(tmp_path, monkeypatch):
    output_root = tmp_path / "out"
    output_root.mkdir()
    staged = output_root / "gemm_perf.txt"
    staged.write_text("op,latency\nmatmul,1.0\n", encoding="utf-8")
    finalization_info: dict[Path, collect_mod.PerfFinalizationInfo] = {}
    parquet_path = collect_mod.finalize_perf_files(
        [staged],
        delete_source=False,
        finalization_info=finalization_info,
    )[0]
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_path = _write_checkpoint(
        checkpoint_dir,
        done=["case-a"],
        failed=["case-b"],
        attempted=["case-a", "case-b"],
    )

    monkeypatch.setattr(
        provenance,
        "write_collection_meta",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("simulated sidecar failure")),
    )

    with pytest.raises(RuntimeError, match="simulated sidecar failure"):
        collect_mod._write_collector_provenance(
            output_root,
            [parquet_path],
            _provenance_ctx(_collections()),
            run_errors=[],
            backend=BACKEND,
            checkpoint_dir=str(checkpoint_dir),
            finalization_info=finalization_info,
        )

    assert _attempted(checkpoint_path) == {"case-a", "case-b"}
    assert staged.exists()
    assert collect_mod._pending_resume_perf_outputs(
        output_root,
        _provenance_ctx(_collections()),
        backend=BACKEND,
        checkpoint_dir=str(checkpoint_dir),
        sm_version=100,
    ) == [staged]


def test_checkpoint_close_failure_retries_sidecar_transaction_without_duplicate_event(tmp_path, monkeypatch):
    output_root = tmp_path / "out"
    output_root.mkdir()
    staged = output_root / "gemm_perf.txt"
    staged.write_text("op,latency\nmatmul,1.0\n", encoding="utf-8")
    first_finalization: dict[Path, collect_mod.PerfFinalizationInfo] = {}
    parquet_path = collect_mod.finalize_perf_files(
        [staged],
        delete_source=False,
        finalization_info=first_finalization,
    )[0]
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_path = _write_checkpoint(
        checkpoint_dir,
        done=["case-a"],
        failed=[],
        attempted=["case-a"],
    )

    real_close = collect_mod._close_checkpoint_attempts
    monkeypatch.setattr(
        collect_mod,
        "_close_checkpoint_attempts",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("simulated checkpoint close failure")),
    )
    with pytest.raises(RuntimeError, match="simulated checkpoint close failure"):
        collect_mod._write_collector_provenance(
            output_root,
            [parquet_path],
            _provenance_ctx(_collections()),
            run_errors=[],
            backend=BACKEND,
            checkpoint_dir=str(checkpoint_dir),
            finalization_info=first_finalization,
        )

    first_doc = yaml.safe_load((output_root / "collection_meta.yaml").read_text(encoding="utf-8"))
    first_table = first_doc["tables"]["gemm_perf"]
    assert "collections" not in first_table
    assert _attempted(checkpoint_path) == {"case-a"}
    assert staged.exists()

    monkeypatch.setattr(collect_mod, "_close_checkpoint_attempts", real_close)
    retry_finalization: dict[Path, collect_mod.PerfFinalizationInfo] = {}
    assert collect_mod.finalize_perf_files(
        [staged],
        delete_source=False,
        finalization_info=retry_finalization,
    ) == [parquet_path]
    assert retry_finalization[parquet_path.resolve()].merged_existing is True

    collect_mod._write_collector_provenance(
        output_root,
        [parquet_path],
        _provenance_ctx(_collections()),
        run_errors=[],
        backend=BACKEND,
        checkpoint_dir=str(checkpoint_dir),
        finalization_info=retry_finalization,
    )

    retry_doc = yaml.safe_load((output_root / "collection_meta.yaml").read_text(encoding="utf-8"))
    retry_table = retry_doc["tables"]["gemm_perf"]
    assert retry_table == first_table
    assert _attempted(checkpoint_path) == set()


@pytest.mark.parametrize(
    ("ledger", "replacement", "error_match"),
    [
        ("attempted", ["case-a", "case-extra"], "live attempts"),
        ("done", ["case-a", "case-extra"], "done/failed"),
        ("failed", ["case-a"], "done/failed"),
    ],
)
def test_normal_commit_rejects_live_checkpoint_ledger_change_before_publish(
    tmp_path,
    monkeypatch,
    ledger,
    replacement,
    error_match,
):
    output_root = tmp_path / "out"
    output_root.mkdir()
    staging_path = output_root / "gemm_perf.txt"
    staging_path.write_text("op,latency\nmatmul,1.0\n", encoding="utf-8")
    finalization_info: dict[Path, collect_mod.PerfFinalizationInfo] = {}
    parquet_path = collect_mod.finalize_perf_files(
        [staging_path],
        delete_source=False,
        finalization_info=finalization_info,
    )[0]
    parquet_before = parquet_path.read_bytes()
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_path = _write_checkpoint(
        checkpoint_dir,
        done=["case-a"],
        failed=[],
        attempted=["case-a"],
    )
    real_write = provenance.write_collection_meta
    injected_checkpoint: bytes | None = None

    def render_then_change_checkpoint(*args, **kwargs):
        nonlocal injected_checkpoint
        rendered_path = real_write(*args, **kwargs)
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        checkpoint[ledger] = replacement
        checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")
        injected_checkpoint = checkpoint_path.read_bytes()
        return rendered_path

    monkeypatch.setattr(provenance, "write_collection_meta", render_then_change_checkpoint)

    with pytest.raises(RuntimeError, match=error_match):
        collect_mod._write_collector_provenance(
            output_root,
            [parquet_path],
            _provenance_ctx(_collections()),
            run_errors=[],
            backend=BACKEND,
            checkpoint_dir=str(checkpoint_dir),
            finalization_info=finalization_info,
        )

    assert injected_checkpoint is not None
    assert checkpoint_path.read_bytes() == injected_checkpoint
    assert parquet_path.read_bytes() == parquet_before
    assert staging_path.exists()
    assert not (output_root / "collection_meta.yaml").exists()
    assert not (output_root / collect_mod._SIDECAR_TRANSACTION_FILENAME).exists()


@pytest.mark.parametrize(
    ("pending_document", "tagged"),
    [(True, False), (False, True)],
    ids=["preparing", "tagged-committed"],
)
@pytest.mark.parametrize(
    ("ledger", "replacement", "error_match"),
    [
        ("attempted", ["case-a", "case-extra"], "live attempts"),
        ("done", ["case-a", "case-extra"], "done/failed"),
        ("failed", ["case-a"], "done/failed"),
    ],
)
def test_recovery_rejects_live_checkpoint_ledger_change_before_mutation(
    tmp_path,
    pending_document,
    tagged,
    ledger,
    replacement,
    error_match,
):
    output_root = tmp_path / "out"
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_path = _write_checkpoint(
        checkpoint_dir,
        done=["case-a"],
        failed=[],
        attempted=["case-a"],
    )
    staging_path = output_root / "gemm_perf.txt"
    staging_path.parent.mkdir(parents=True)
    staging_path.write_bytes(b"owned staging")
    _write_sidecar_transaction(
        output_root,
        [(checkpoint_path, {"case-a"})],
        [staging_path],
        checkpoint_dir=checkpoint_dir,
        tagged={checkpoint_path} if tagged else set(),
        pending_document=pending_document,
    )
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    checkpoint[ledger] = replacement
    checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")
    checkpoint_before = checkpoint_path.read_bytes()
    journal_path = output_root / collect_mod._SIDECAR_TRANSACTION_FILENAME

    with pytest.raises(RuntimeError, match=error_match):
        collect_mod._recover_collector_provenance_transaction(
            output_root,
            backend=BACKEND,
            checkpoint_dir=str(checkpoint_dir),
        )

    assert checkpoint_path.read_bytes() == checkpoint_before
    assert staging_path.read_bytes() == b"owned staging"
    assert journal_path.exists()
    assert (output_root / "collection_meta.yaml").exists() is not pending_document
    assert (output_root / collect_mod._SIDECAR_STAGING_FILENAME).exists() is pending_document


@pytest.mark.parametrize("live_attempts", [["case-a"], ["case-extra"]], ids=["transaction-id", "new-id"])
def test_committed_recovery_rejects_untagged_checkpoint_with_any_open_attempt(
    tmp_path,
    live_attempts,
):
    output_root = tmp_path / "out"
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_path = _write_checkpoint(
        checkpoint_dir,
        done=live_attempts,
        failed=[],
        attempted=live_attempts,
    )
    staging_path = output_root / "gemm_perf.txt"
    staging_path.parent.mkdir(parents=True)
    staging_path.write_bytes(b"owned staging")
    _write_sidecar_transaction(
        output_root,
        [(checkpoint_path, {"case-a"})],
        [staging_path],
        checkpoint_dir=checkpoint_dir,
        tagged=set(),
        pending_document=False,
    )
    checkpoint_before = checkpoint_path.read_bytes()
    journal_path = output_root / collect_mod._SIDECAR_TRANSACTION_FILENAME

    with pytest.raises(RuntimeError, match=r"untagged|live attempts"):
        collect_mod._recover_collector_provenance_transaction(
            output_root,
            backend=BACKEND,
            checkpoint_dir=str(checkpoint_dir),
        )

    assert checkpoint_path.read_bytes() == checkpoint_before
    assert staging_path.read_bytes() == b"owned staging"
    assert journal_path.exists()


@pytest.mark.parametrize("mutation", ["tag", "close"])
def test_transaction_checkpoint_mutation_rechecks_exact_live_attempts(tmp_path, mutation):
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_path = _write_checkpoint(
        checkpoint_dir,
        done=["case-a", "case-extra"],
        failed=[],
        attempted=["case-a", "case-extra"],
    )
    transaction_id = "1" * 32
    if mutation == "close":
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        checkpoint[collect_mod._SIDECAR_TRANSACTION_FIELD] = transaction_id
        checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")
    checkpoint_before = checkpoint_path.read_bytes()

    with pytest.raises(RuntimeError, match="live attempts"):
        if mutation == "tag":
            collect_mod._tag_checkpoint_sidecar_transaction(checkpoint_path, {"case-a"}, transaction_id)
        else:
            collect_mod._close_checkpoint_attempts(
                checkpoint_path,
                {"case-a"},
                transaction_id=transaction_id,
            )

    assert checkpoint_path.read_bytes() == checkpoint_before


def test_recovery_finishes_partial_prepare_when_pending_sidecar_matches_live_bytes(tmp_path):
    output_root = tmp_path / "out"
    checkpoint_dir = tmp_path / "checkpoint"
    first = _write_checkpoint_for(
        checkpoint_dir,
        backend=BACKEND,
        full_name="sglang.mla_bmm_gen_pre",
        version="0.5.14",
        done=["case-a"],
        failed=[],
        attempted=["case-a"],
    )
    second = _write_checkpoint_for(
        checkpoint_dir,
        backend=BACKEND,
        full_name="sglang.mla_bmm_gen_post",
        version="0.5.14",
        done=["case-b"],
        failed=[],
        attempted=["case-b"],
    )
    staged = output_root / "mla_bmm_perf.txt"
    staged.parent.mkdir(parents=True)
    staged.write_text("op,latency\nshared,1.0\n", encoding="utf-8")
    participants = [(first, {"case-a"}), (second, {"case-b"})]
    _write_sidecar_transaction(
        output_root,
        participants,
        [staged],
        checkpoint_dir=checkpoint_dir,
        tagged={first},
        pending_document=True,
    )

    assert collect_mod._recover_collector_provenance_transaction(
        output_root,
        backend=BACKEND,
        checkpoint_dir=str(checkpoint_dir),
    ) == (output_root / "collection_meta.yaml")

    assert _attempted(first) == set()
    assert _attempted(second) == set()
    assert not staged.exists()
    assert not (output_root / collect_mod._SIDECAR_STAGING_FILENAME).exists()
    assert not (output_root / collect_mod._SIDECAR_TRANSACTION_FILENAME).exists()


@pytest.mark.parametrize("replacement", ["rewrite", "symlink"])
def test_preparing_recovery_rejects_replaced_pending_sidecar_after_checkpoint_tag(
    tmp_path,
    monkeypatch,
    replacement,
):
    output_root = tmp_path / "out"
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_path = _write_checkpoint(
        checkpoint_dir,
        done=["case-a"],
        failed=[],
        attempted=["case-a"],
    )
    staging_path = output_root / "gemm_perf.txt"
    staging_path.parent.mkdir(parents=True)
    staging_path.write_bytes(b"owned staging")
    _write_sidecar_transaction(
        output_root,
        [(checkpoint_path, {"case-a"})],
        [staging_path],
        checkpoint_dir=checkpoint_dir,
        tagged=set(),
        pending_document=True,
    )
    meta_path = output_root / "collection_meta.yaml"
    assert not meta_path.exists()
    pending_path = output_root / collect_mod._SIDECAR_STAGING_FILENAME
    outside_document = tmp_path / "outside-pending.yaml"
    outside_document.write_text("outside document\n", encoding="utf-8")
    outside_before = outside_document.read_bytes()
    real_tag = collect_mod._tag_checkpoint_sidecar_transaction

    def tag_then_replace_pending(*args, **kwargs):
        real_tag(*args, **kwargs)
        if replacement == "rewrite":
            pending_path.write_text("tampered: true\n", encoding="utf-8")
        else:
            pending_path.unlink()
            try:
                pending_path.symlink_to(outside_document)
            except OSError as error:
                pytest.skip(f"symlinks unavailable: {error}")

    monkeypatch.setattr(collect_mod, "_tag_checkpoint_sidecar_transaction", tag_then_replace_pending)

    with pytest.raises(RuntimeError, match=r"sidecar|document|pending"):
        collect_mod._recover_collector_provenance_transaction(
            output_root,
            backend=BACKEND,
            checkpoint_dir=str(checkpoint_dir),
        )

    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert checkpoint["attempted"] == ["case-a"]
    assert checkpoint[collect_mod._SIDECAR_TRANSACTION_FIELD]
    assert not meta_path.exists()
    if replacement == "symlink":
        assert pending_path.is_symlink()
    else:
        assert pending_path.read_text(encoding="utf-8") == "tampered: true\n"
    assert outside_document.read_bytes() == outside_before
    assert staging_path.exists()
    assert (output_root / collect_mod._SIDECAR_TRANSACTION_FILENAME).exists()


@pytest.mark.parametrize("phase", ["preparing", "committed"])
def test_recovery_rejects_sidecar_swap_after_final_validation(tmp_path, monkeypatch, phase):
    output_root = tmp_path / "out"
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_path = _write_checkpoint(
        checkpoint_dir,
        done=["case-a"],
        failed=[],
        attempted=["case-a"],
    )
    staging_path = output_root / "gemm_perf.txt"
    staging_path.parent.mkdir(parents=True)
    staging_path.write_bytes(b"owned staging")
    preparing = phase == "preparing"
    _write_sidecar_transaction(
        output_root,
        [(checkpoint_path, {"case-a"})],
        [staging_path],
        checkpoint_dir=checkpoint_dir,
        tagged=set() if preparing else {checkpoint_path},
        pending_document=preparing,
    )
    source_path = output_root / (collect_mod._SIDECAR_STAGING_FILENAME if preparing else "collection_meta.yaml")
    malicious = f"malicious: {phase}\n".encode()
    real_validate = collect_mod._validated_sidecar_document
    expected_final_validations = 3 if preparing else 1
    final_validations = 0

    def validate_then_swap(*args, **kwargs):
        nonlocal final_validations
        snapshot = real_validate(*args, **kwargs)
        if Path(args[0]) == source_path and kwargs.get("expected_identity") is not None:
            final_validations += 1
            if final_validations == expected_final_validations:
                _replace_with_new_inode(source_path, malicious)
        return snapshot

    monkeypatch.setattr(collect_mod, "_validated_sidecar_document", validate_then_swap)

    with pytest.raises(RuntimeError, match=r"sidecar|staging|changed"):
        collect_mod._recover_collector_provenance_transaction(
            output_root,
            backend=BACKEND,
            checkpoint_dir=str(checkpoint_dir),
        )

    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert final_validations == expected_final_validations
    assert checkpoint["attempted"] == ["case-a"]
    assert checkpoint[collect_mod._SIDECAR_TRANSACTION_FIELD]
    assert source_path.read_bytes() == malicious
    assert staging_path.read_bytes() == b"owned staging"
    assert (output_root / collect_mod._SIDECAR_TRANSACTION_FILENAME).exists()


@pytest.mark.parametrize("phase", ["preparing", "committed"])
def test_recovery_rejects_checkpoint_inode_swap_after_participant_validation(tmp_path, monkeypatch, phase):
    output_root = tmp_path / "out"
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_path = _write_checkpoint(
        checkpoint_dir,
        done=["case-a"],
        failed=[],
        attempted=["case-a"],
    )
    staging_path = output_root / "gemm_perf.txt"
    staging_path.parent.mkdir(parents=True)
    staging_path.write_bytes(b"owned staging")
    preparing = phase == "preparing"
    _write_sidecar_transaction(
        output_root,
        [(checkpoint_path, {"case-a"})],
        [staging_path],
        checkpoint_dir=checkpoint_dir,
        tagged=set() if preparing else {checkpoint_path},
        pending_document=preparing,
    )
    replacement = checkpoint_path.read_bytes()
    replacement_inode: int | None = None
    real_participants = collect_mod._validated_checkpoint_participants_for_phase

    def validate_then_replace(*args, **kwargs):
        nonlocal replacement_inode
        participants = real_participants(*args, **kwargs)
        replacement_inode = _replace_with_new_inode(checkpoint_path, replacement)
        return participants

    monkeypatch.setattr(collect_mod, "_validated_checkpoint_participants_for_phase", validate_then_replace)
    sidecars_before = {
        path: path.read_bytes()
        for path in (
            output_root / "collection_meta.yaml",
            output_root / collect_mod._SIDECAR_STAGING_FILENAME,
            output_root / collect_mod._SIDECAR_TRANSACTION_FILENAME,
        )
        if path.exists()
    }

    with pytest.raises(RuntimeError, match=r"checkpoint|attest|changed"):
        collect_mod._recover_collector_provenance_transaction(
            output_root,
            backend=BACKEND,
            checkpoint_dir=str(checkpoint_dir),
        )

    assert replacement_inode is not None
    assert checkpoint_path.stat().st_ino == replacement_inode
    assert checkpoint_path.read_bytes() == replacement
    assert staging_path.read_bytes() == b"owned staging"
    assert {path: path.read_bytes() for path in sidecars_before} == sidecars_before


def test_preparing_recovery_rejects_missing_staging_before_mutation(tmp_path):
    output_root = tmp_path / "out"
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_path = _write_checkpoint(
        checkpoint_dir,
        done=["case-a"],
        failed=[],
        attempted=["case-a"],
    )
    staged = output_root / "gemm_perf.txt"
    staged.parent.mkdir(parents=True)
    staged.write_text("op,latency\nmatmul,1.0\n", encoding="utf-8")
    _write_sidecar_transaction(
        output_root,
        [(checkpoint_path, {"case-a"})],
        [staged],
        checkpoint_dir=checkpoint_dir,
        tagged=set(),
        pending_document=True,
    )
    staged.unlink()
    checkpoint_before = checkpoint_path.read_bytes()
    pending_path = output_root / collect_mod._SIDECAR_STAGING_FILENAME
    pending_before = pending_path.read_bytes()
    journal_path = output_root / collect_mod._SIDECAR_TRANSACTION_FILENAME
    journal_before = journal_path.read_bytes()

    with pytest.raises(RuntimeError, match="staging"):
        collect_mod._recover_collector_provenance_transaction(
            output_root,
            backend=BACKEND,
            checkpoint_dir=str(checkpoint_dir),
        )

    assert checkpoint_path.read_bytes() == checkpoint_before
    assert pending_path.read_bytes() == pending_before
    assert journal_path.read_bytes() == journal_before
    assert not (output_root / "collection_meta.yaml").exists()


def test_recovery_cleans_staging_before_removing_journal_after_all_checkpoints_closed(tmp_path):
    output_root = tmp_path / "out"
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_path = _write_checkpoint(
        checkpoint_dir,
        done=["case-a"],
        failed=[],
        attempted=[],
    )
    staged = output_root / "gemm_perf.txt"
    staged.parent.mkdir(parents=True)
    staged.write_text("op,latency\nmatmul,1.0\n", encoding="utf-8")
    _write_sidecar_transaction(
        output_root,
        [(checkpoint_path, {"case-a"})],
        [staged],
        checkpoint_dir=checkpoint_dir,
        tagged=set(),
        pending_document=False,
    )

    assert (
        collect_mod._recover_collector_provenance_transaction(
            output_root,
            backend=BACKEND,
            checkpoint_dir=str(checkpoint_dir),
        )
        is None
    )

    assert not staged.exists()
    assert not (output_root / collect_mod._SIDECAR_TRANSACTION_FILENAME).exists()


def test_recovery_rejects_outside_staging_path_without_deleting_it(tmp_path):
    output_root = tmp_path / "out"
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_path = _write_checkpoint(
        checkpoint_dir,
        done=["case-a"],
        failed=[],
        attempted=[],
    )
    outside_victim = tmp_path / "outside_perf.txt"
    outside_victim.write_bytes(b"outside staging must survive")
    _write_sidecar_transaction(
        output_root,
        [(checkpoint_path, {"case-a"})],
        [outside_victim],
        checkpoint_dir=checkpoint_dir,
        tagged=set(),
        pending_document=False,
    )
    journal_path = output_root / collect_mod._SIDECAR_TRANSACTION_FILENAME

    with pytest.raises(RuntimeError, match="staging"):
        collect_mod._recover_collector_provenance_transaction(
            output_root,
            backend=BACKEND,
            checkpoint_dir=str(checkpoint_dir),
        )

    assert outside_victim.read_bytes() == b"outside staging must survive"
    assert journal_path.exists()


def test_recovery_prevalidates_mixed_staging_paths_before_deleting_any(tmp_path):
    output_root = tmp_path / "out"
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_path = _write_checkpoint(
        checkpoint_dir,
        done=["case-a"],
        failed=[],
        attempted=[],
    )
    owned_staging = output_root / "gemm_perf.txt"
    owned_staging.parent.mkdir(parents=True)
    owned_staging.write_bytes(b"owned staging must survive")
    outside_victim = tmp_path / "outside_perf.txt"
    outside_victim.write_bytes(b"outside staging must survive")
    _write_sidecar_transaction(
        output_root,
        [(checkpoint_path, {"case-a"})],
        [owned_staging, outside_victim],
        checkpoint_dir=checkpoint_dir,
        tagged=set(),
        pending_document=False,
    )
    journal_path = output_root / collect_mod._SIDECAR_TRANSACTION_FILENAME

    with pytest.raises(RuntimeError, match="staging"):
        collect_mod._recover_collector_provenance_transaction(
            output_root,
            backend=BACKEND,
            checkpoint_dir=str(checkpoint_dir),
        )

    assert owned_staging.read_bytes() == b"owned staging must survive"
    assert outside_victim.read_bytes() == b"outside staging must survive"
    assert journal_path.exists()


def test_recovery_rejects_unrelated_canonical_staging_before_mutation(tmp_path):
    output_root = tmp_path / "out"
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_path = _write_checkpoint(
        checkpoint_dir,
        done=["case-a"],
        failed=[],
        attempted=["case-a"],
    )
    owned_staging = output_root / "gemm_perf.txt"
    unrelated_staging = output_root / "moe_perf.txt"
    owned_staging.parent.mkdir(parents=True)
    owned_staging.write_bytes(b"owned gemm staging")
    unrelated_staging.write_bytes(b"unrelated canonical staging")
    _write_sidecar_transaction(
        output_root,
        [(checkpoint_path, {"case-a"})],
        [owned_staging, unrelated_staging],
        checkpoint_dir=checkpoint_dir,
        tagged={checkpoint_path},
        pending_document=False,
    )
    checkpoint_before = checkpoint_path.read_bytes()
    journal_path = output_root / collect_mod._SIDECAR_TRANSACTION_FILENAME

    with pytest.raises(RuntimeError, match=r"staging|table"):
        collect_mod._recover_collector_provenance_transaction(
            output_root,
            backend=BACKEND,
            checkpoint_dir=str(checkpoint_dir),
        )

    assert owned_staging.read_bytes() == b"owned gemm staging"
    assert unrelated_staging.read_bytes() == b"unrelated canonical staging"
    assert checkpoint_path.read_bytes() == checkpoint_before
    assert journal_path.exists()


def test_recovery_rejects_unrelated_canonical_checkpoint_even_with_same_case_ids(tmp_path):
    output_root = tmp_path / "out"
    checkpoint_dir = tmp_path / "checkpoint"
    owned_checkpoint = _write_checkpoint(
        checkpoint_dir,
        done=["case-a"],
        failed=[],
        attempted=["case-a"],
    )
    unrelated_checkpoint = _write_checkpoint_for(
        checkpoint_dir,
        backend=BACKEND,
        full_name=f"{BACKEND}.moe",
        version="0.5.14",
        done=["case-a"],
        failed=[],
        attempted=["case-a"],
    )
    staging_path = output_root / "gemm_perf.txt"
    staging_path.parent.mkdir(parents=True)
    staging_path.write_bytes(b"owned gemm staging")
    _write_sidecar_transaction(
        output_root,
        [(owned_checkpoint, {"case-a"})],
        [staging_path],
        checkpoint_dir=checkpoint_dir,
        tagged=set(),
        pending_document=True,
    )
    journal_path = output_root / collect_mod._SIDECAR_TRANSACTION_FILENAME
    transaction = json.loads(journal_path.read_text(encoding="utf-8"))
    transaction["checkpoints"][0]["path"] = str(unrelated_checkpoint)
    transaction["checkpoints"][0]["identity"] = {
        field: json.loads(unrelated_checkpoint.read_text(encoding="utf-8"))[field]
        for field in collect_mod._CHECKPOINT_IDENTITY_FIELDS
    }
    unrelated_document = json.loads(unrelated_checkpoint.read_text(encoding="utf-8"))
    transaction["checkpoints"][0]["done"] = unrelated_document["done"]
    transaction["checkpoints"][0]["failed"] = unrelated_document["failed"]
    journal_path.write_text(json.dumps(transaction), encoding="utf-8")
    checkpoint_before = {path: path.read_bytes() for path in (owned_checkpoint, unrelated_checkpoint)}

    with pytest.raises(RuntimeError, match=r"[Oo]wned|producer"):
        collect_mod._recover_collector_provenance_transaction(
            output_root,
            backend=BACKEND,
            checkpoint_dir=str(checkpoint_dir),
        )

    assert {path: path.read_bytes() for path in checkpoint_before} == checkpoint_before
    assert staging_path.read_bytes() == b"owned gemm staging"
    assert journal_path.exists()


def test_recovery_commits_two_owned_tables_in_one_transaction(tmp_path):
    output_root = tmp_path / "out"
    checkpoint_dir = tmp_path / "checkpoint"
    gemm_checkpoint = _write_checkpoint(
        checkpoint_dir,
        done=["gemm-case"],
        failed=[],
        attempted=["gemm-case"],
    )
    moe_checkpoint = _write_checkpoint_for(
        checkpoint_dir,
        backend=BACKEND,
        full_name=f"{BACKEND}.moe",
        version="0.5.14",
        done=["moe-case"],
        failed=[],
        attempted=["moe-case"],
    )
    gemm_staging = output_root / "gemm_perf.txt"
    moe_staging = output_root / "moe_perf.txt"
    gemm_staging.parent.mkdir(parents=True)
    gemm_staging.write_bytes(b"gemm staging")
    moe_staging.write_bytes(b"moe staging")
    participant_tables = {gemm_checkpoint: "gemm_perf", moe_checkpoint: "moe_perf"}
    _write_sidecar_transaction(
        output_root,
        [(gemm_checkpoint, {"gemm-case"}), (moe_checkpoint, {"moe-case"})],
        [gemm_staging, moe_staging],
        checkpoint_dir=checkpoint_dir,
        tagged={gemm_checkpoint, moe_checkpoint},
        pending_document=False,
        participant_tables=participant_tables,
    )

    assert collect_mod._recover_collector_provenance_transaction(
        output_root,
        backend=BACKEND,
        checkpoint_dir=str(checkpoint_dir),
    ) == (output_root / "collection_meta.yaml")

    assert _attempted(gemm_checkpoint) == set()
    assert _attempted(moe_checkpoint) == set()
    assert not gemm_staging.exists()
    assert not moe_staging.exists()
    assert not (output_root / collect_mod._SIDECAR_TRANSACTION_FILENAME).exists()


@pytest.mark.parametrize("pending_document", [True, False], ids=["preparing", "committed"])
def test_recovery_rejects_same_path_staging_replacement_before_any_mutation(tmp_path, pending_document):
    output_root = tmp_path / "out"
    checkpoint_dir = tmp_path / "checkpoint"
    gemm_checkpoint = _write_checkpoint(
        checkpoint_dir,
        done=["gemm-case"],
        failed=[],
        attempted=["gemm-case"],
    )
    moe_checkpoint = _write_checkpoint_for(
        checkpoint_dir,
        backend=BACKEND,
        full_name=f"{BACKEND}.moe",
        version="0.5.14",
        done=["moe-case"],
        failed=[],
        attempted=["moe-case"],
    )
    gemm_staging = output_root / "gemm_perf.txt"
    moe_staging = output_root / "moe_perf.txt"
    gemm_staging.parent.mkdir(parents=True)
    gemm_staging.write_bytes(b"owned gemm staging")
    moe_staging.write_bytes(b"owned moe staging")
    _write_sidecar_transaction(
        output_root,
        [(gemm_checkpoint, {"gemm-case"}), (moe_checkpoint, {"moe-case"})],
        [gemm_staging, moe_staging],
        checkpoint_dir=checkpoint_dir,
        tagged={gemm_checkpoint, moe_checkpoint} if not pending_document else set(),
        pending_document=pending_document,
        participant_tables={gemm_checkpoint: "gemm_perf", moe_checkpoint: "moe_perf"},
    )
    replacement = b"unrelated same-path replacement"
    moe_staging.write_bytes(replacement)
    gemm_before = gemm_staging.read_bytes()
    checkpoints_before = {path: path.read_bytes() for path in (gemm_checkpoint, moe_checkpoint)}
    meta_path = output_root / "collection_meta.yaml"
    pending_path = output_root / collect_mod._SIDECAR_STAGING_FILENAME
    journal_path = output_root / collect_mod._SIDECAR_TRANSACTION_FILENAME
    sidecars_before = {path: path.read_bytes() for path in (meta_path, pending_path, journal_path) if path.exists()}

    with pytest.raises(RuntimeError, match=r"staging|digest"):
        collect_mod._recover_collector_provenance_transaction(
            output_root,
            backend=BACKEND,
            checkpoint_dir=str(checkpoint_dir),
        )

    assert gemm_staging.read_bytes() == gemm_before
    assert moe_staging.read_bytes() == replacement
    assert {path: path.read_bytes() for path in checkpoints_before} == checkpoints_before
    assert {path: path.read_bytes() for path in sidecars_before} == sidecars_before


def test_recovery_rechecks_staging_identity_immediately_before_cleanup(tmp_path, monkeypatch):
    output_root = tmp_path / "out"
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_path = _write_checkpoint(
        checkpoint_dir,
        done=["case-a"],
        failed=[],
        attempted=["case-a"],
    )
    staging_path = output_root / "gemm_perf.txt"
    staging_path.parent.mkdir(parents=True)
    staging_path.write_bytes(b"owned staging")
    _write_sidecar_transaction(
        output_root,
        [(checkpoint_path, {"case-a"})],
        [staging_path],
        checkpoint_dir=checkpoint_dir,
        tagged={checkpoint_path},
        pending_document=False,
    )
    replacement = staging_path.read_bytes()
    replacement_inode: int | None = None
    real_close = collect_mod._close_checkpoint_attempts

    def close_then_replace(*args, **kwargs):
        nonlocal replacement_inode
        real_close(*args, **kwargs)
        replacement_inode = _replace_with_new_inode(staging_path, replacement)

    monkeypatch.setattr(collect_mod, "_close_checkpoint_attempts", close_then_replace)
    journal_path = output_root / collect_mod._SIDECAR_TRANSACTION_FILENAME

    with pytest.raises(RuntimeError, match=r"staging|digest"):
        collect_mod._recover_collector_provenance_transaction(
            output_root,
            backend=BACKEND,
            checkpoint_dir=str(checkpoint_dir),
        )

    assert staging_path.read_bytes() == replacement
    assert replacement_inode is not None
    assert staging_path.stat().st_ino == replacement_inode
    assert _attempted(checkpoint_path) == set()
    assert journal_path.exists()


def test_recovery_uses_latest_shared_table_history_event_for_ownership(tmp_path):
    output_root = tmp_path / "out"
    checkpoint_dir = tmp_path / "checkpoint"
    first = _write_checkpoint_for(
        checkpoint_dir,
        backend=BACKEND,
        full_name=f"{BACKEND}.mla_bmm_gen_pre",
        version="0.5.14",
        done=["case-a"],
        failed=[],
        attempted=["case-a"],
    )
    second = _write_checkpoint_for(
        checkpoint_dir,
        backend=BACKEND,
        full_name=f"{BACKEND}.mla_bmm_gen_post",
        version="0.5.14",
        done=["case-b"],
        failed=[],
        attempted=["case-b"],
    )
    staging_path = output_root / "mla_bmm_perf.txt"
    staging_path.parent.mkdir(parents=True)
    staging_path.write_bytes(b"shared staging")
    participant_tables = {first: "mla_bmm_perf", second: "mla_bmm_perf"}
    _write_sidecar_transaction(
        output_root,
        [(first, {"case-a"}), (second, {"case-b"})],
        [staging_path],
        checkpoint_dir=checkpoint_dir,
        tagged={first, second},
        pending_document=False,
        participant_tables=participant_tables,
    )
    meta_path = output_root / "collection_meta.yaml"
    sidecar = yaml.safe_load(meta_path.read_text(encoding="utf-8"))
    sidecar["schema_version"] = 2
    sidecar["tables"]["mla_bmm_perf"] = {
        "rows": 2,
        "status": provenance.STATUS_COMPLETE,
        "collections": [
            {
                **_collection_event(),
                "case_plan_hash": provenance.case_plan_hash(["old-case"]),
            },
            {
                **_collection_event(),
                "case_plan_hash": provenance.case_plan_hash(["case-a", "case-b"]),
            },
        ],
    }
    meta_path.write_text(yaml.safe_dump(sidecar, sort_keys=False), encoding="utf-8")
    journal_path = output_root / collect_mod._SIDECAR_TRANSACTION_FILENAME
    transaction = json.loads(journal_path.read_text(encoding="utf-8"))
    transaction["sidecar_digest"] = _independent_digest(meta_path)
    journal_path.write_text(json.dumps(transaction), encoding="utf-8")

    assert (
        collect_mod._recover_collector_provenance_transaction(
            output_root,
            backend=BACKEND,
            checkpoint_dir=str(checkpoint_dir),
        )
        == meta_path
    )

    assert _attempted(first) == set()
    assert _attempted(second) == set()
    assert not staging_path.exists()
    assert not journal_path.exists()


def test_recovery_rejects_swapped_multi_table_participant_mapping_before_mutation(tmp_path):
    output_root = tmp_path / "out"
    checkpoint_dir = tmp_path / "checkpoint"
    gemm_checkpoint = _write_checkpoint(
        checkpoint_dir,
        done=["gemm-case"],
        failed=[],
        attempted=["gemm-case"],
    )
    moe_checkpoint = _write_checkpoint_for(
        checkpoint_dir,
        backend=BACKEND,
        full_name=f"{BACKEND}.moe",
        version="0.5.14",
        done=["moe-case"],
        failed=[],
        attempted=["moe-case"],
    )
    gemm_staging = output_root / "gemm_perf.txt"
    moe_staging = output_root / "moe_perf.txt"
    gemm_staging.parent.mkdir(parents=True)
    gemm_staging.write_bytes(b"gemm staging")
    moe_staging.write_bytes(b"moe staging")
    _write_sidecar_transaction(
        output_root,
        [(gemm_checkpoint, {"gemm-case"}), (moe_checkpoint, {"moe-case"})],
        [gemm_staging, moe_staging],
        checkpoint_dir=checkpoint_dir,
        tagged={gemm_checkpoint, moe_checkpoint},
        pending_document=False,
        participant_tables={gemm_checkpoint: "gemm_perf", moe_checkpoint: "moe_perf"},
    )
    journal_path = output_root / collect_mod._SIDECAR_TRANSACTION_FILENAME
    transaction = json.loads(journal_path.read_text(encoding="utf-8"))
    meta_path = output_root / "collection_meta.yaml"
    sidecar = yaml.safe_load(meta_path.read_text(encoding="utf-8"))
    sidecar["tables"]["gemm_perf"]["case_plan_hash"] = provenance.case_plan_hash(["moe-case"])
    sidecar["tables"]["moe_perf"]["case_plan_hash"] = provenance.case_plan_hash(["gemm-case"])
    meta_path.write_text(yaml.safe_dump(sidecar, sort_keys=False), encoding="utf-8")
    transaction["sidecar_digest"] = _independent_digest(meta_path)
    journal_path.write_text(json.dumps(transaction), encoding="utf-8")
    checkpoint_before = {path: path.read_bytes() for path in (gemm_checkpoint, moe_checkpoint)}

    with pytest.raises(RuntimeError, match=r"own table|case plan"):
        collect_mod._recover_collector_provenance_transaction(
            output_root,
            backend=BACKEND,
            checkpoint_dir=str(checkpoint_dir),
        )

    assert {path: path.read_bytes() for path in checkpoint_before} == checkpoint_before
    assert gemm_staging.read_bytes() == b"gemm staging"
    assert moe_staging.read_bytes() == b"moe staging"
    assert journal_path.exists()


@pytest.mark.parametrize("path_fragment", ["./gemm_perf.txt", "nested/../gemm_perf.txt"])
def test_recovery_rejects_staging_path_alias_before_mutation(tmp_path, path_fragment):
    output_root = tmp_path / "out"
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_path = _write_checkpoint(
        checkpoint_dir,
        done=["case-a"],
        failed=[],
        attempted=[],
    )
    staged = output_root / "gemm_perf.txt"
    staged.parent.mkdir(parents=True)
    staged.write_bytes(b"owned staging")
    _write_sidecar_transaction(
        output_root,
        [(checkpoint_path, {"case-a"})],
        [staged],
        checkpoint_dir=checkpoint_dir,
        tagged=set(),
        pending_document=False,
    )
    journal_path = output_root / collect_mod._SIDECAR_TRANSACTION_FILENAME
    transaction = json.loads(journal_path.read_text(encoding="utf-8"))
    transaction["staging_paths"] = [f"{output_root}/{path_fragment}"]
    journal_path.write_text(json.dumps(transaction), encoding="utf-8")
    checkpoint_before = checkpoint_path.read_bytes()

    with pytest.raises(RuntimeError, match="staging"):
        collect_mod._recover_collector_provenance_transaction(
            output_root,
            backend=BACKEND,
            checkpoint_dir=str(checkpoint_dir),
        )

    assert checkpoint_path.read_bytes() == checkpoint_before
    assert staged.read_bytes() == b"owned staging"
    assert journal_path.exists()


def test_recovery_rejects_staging_symlink_without_touching_target(tmp_path):
    output_root = tmp_path / "out"
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_path = _write_checkpoint(
        checkpoint_dir,
        done=["case-a"],
        failed=[],
        attempted=[],
    )
    outside_victim = tmp_path / "outside.csv"
    outside_victim.write_bytes(b"symlink target must survive")
    staged = output_root / "gemm_perf.txt"
    staged.parent.mkdir(parents=True)
    try:
        staged.symlink_to(outside_victim)
    except OSError as error:
        pytest.skip(f"symlinks unavailable: {error}")
    _write_sidecar_transaction(
        output_root,
        [(checkpoint_path, {"case-a"})],
        [staged],
        checkpoint_dir=checkpoint_dir,
        tagged=set(),
        pending_document=False,
    )
    journal_path = output_root / collect_mod._SIDECAR_TRANSACTION_FILENAME

    with pytest.raises(RuntimeError, match="staging"):
        collect_mod._recover_collector_provenance_transaction(
            output_root,
            backend=BACKEND,
            checkpoint_dir=str(checkpoint_dir),
        )

    assert staged.is_symlink()
    assert outside_victim.read_bytes() == b"symlink target must survive"
    assert journal_path.exists()


def test_recovery_prevalidates_mixed_checkpoint_participants_before_rewriting_any(tmp_path):
    output_root = tmp_path / "out"
    checkpoint_dir = tmp_path / "checkpoint"
    valid_checkpoint = _write_checkpoint(
        checkpoint_dir,
        done=["case-valid"],
        failed=[],
        attempted=["case-valid"],
    )
    outside_checkpoint = _write_checkpoint_for(
        tmp_path,
        backend=BACKEND,
        full_name="sglang.mla_bmm_gen_pre",
        version="0.5.14",
        done=["case-outside"],
        failed=[],
        attempted=["case-outside"],
    )
    staged = output_root / "gemm_perf.txt"
    staged.parent.mkdir(parents=True)
    staged.write_text("op,latency\nmatmul,1.0\n", encoding="utf-8")
    _write_sidecar_transaction(
        output_root,
        [(valid_checkpoint, {"case-valid"}), (outside_checkpoint, {"case-outside"})],
        [staged],
        checkpoint_dir=checkpoint_dir,
        tagged=set(),
        pending_document=True,
    )
    valid_before = valid_checkpoint.read_bytes()
    checkpoint_before = outside_checkpoint.read_bytes()
    journal_path = output_root / collect_mod._SIDECAR_TRANSACTION_FILENAME

    with pytest.raises(RuntimeError, match="checkpoint"):
        collect_mod._recover_collector_provenance_transaction(
            output_root,
            backend=BACKEND,
            checkpoint_dir=str(checkpoint_dir),
        )

    assert valid_checkpoint.read_bytes() == valid_before
    assert outside_checkpoint.read_bytes() == checkpoint_before
    assert staged.exists()
    assert journal_path.exists()
    assert (output_root / collect_mod._SIDECAR_STAGING_FILENAME).exists()


def test_recovery_rejects_checkpoint_symlink_without_touching_target(tmp_path):
    output_root = tmp_path / "out"
    checkpoint_dir = tmp_path / "checkpoint"
    outside_checkpoint = _write_checkpoint_for(
        tmp_path,
        backend=BACKEND,
        full_name="sglang.mla_bmm_gen_pre",
        version="0.5.14",
        done=["case-a"],
        failed=[],
        attempted=["case-a"],
    )
    checkpoint_path = checkpoint_dir / BACKEND / f"{FULL_NAME}.json"
    checkpoint_path.parent.mkdir(parents=True)
    try:
        checkpoint_path.symlink_to(outside_checkpoint)
    except OSError as error:
        pytest.skip(f"symlinks unavailable: {error}")
    staged = output_root / "gemm_perf.txt"
    staged.parent.mkdir(parents=True)
    staged.write_bytes(b"owned staging")
    _write_sidecar_transaction(
        output_root,
        [(checkpoint_path, {"case-a"})],
        [staged],
        checkpoint_dir=checkpoint_dir,
        tagged=set(),
        pending_document=True,
    )
    outside_before = outside_checkpoint.read_bytes()
    journal_path = output_root / collect_mod._SIDECAR_TRANSACTION_FILENAME

    with pytest.raises(RuntimeError, match="checkpoint"):
        collect_mod._recover_collector_provenance_transaction(
            output_root,
            backend=BACKEND,
            checkpoint_dir=str(checkpoint_dir),
        )

    assert checkpoint_path.is_symlink()
    assert outside_checkpoint.read_bytes() == outside_before
    assert staged.read_bytes() == b"owned staging"
    assert journal_path.exists()


@pytest.mark.parametrize("document_name", ["collection_meta.yaml", collect_mod._SIDECAR_STAGING_FILENAME])
def test_recovery_rejects_sidecar_document_symlink_before_mutation(tmp_path, document_name):
    output_root = tmp_path / "out"
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_path = _write_checkpoint(
        checkpoint_dir,
        done=["case-a"],
        failed=[],
        attempted=["case-a"],
    )
    staged = output_root / "gemm_perf.txt"
    staged.parent.mkdir(parents=True)
    staged.write_bytes(b"owned staging")
    _write_sidecar_transaction(
        output_root,
        [(checkpoint_path, {"case-a"})],
        [staged],
        checkpoint_dir=checkpoint_dir,
        tagged={checkpoint_path},
        pending_document=True,
    )
    document_path = output_root / document_name
    outside_document = tmp_path / f"outside-{document_name}"
    source_document = document_path if document_path.exists() else output_root / collect_mod._SIDECAR_STAGING_FILENAME
    outside_document.write_bytes(source_document.read_bytes())
    if document_path.exists():
        document_path.unlink()
    try:
        document_path.symlink_to(outside_document)
    except OSError as error:
        pytest.skip(f"symlinks unavailable: {error}")
    checkpoint_before = checkpoint_path.read_bytes()
    journal_path = output_root / collect_mod._SIDECAR_TRANSACTION_FILENAME

    with pytest.raises(RuntimeError, match="document"):
        collect_mod._recover_collector_provenance_transaction(
            output_root,
            backend=BACKEND,
            checkpoint_dir=str(checkpoint_dir),
        )

    assert checkpoint_path.read_bytes() == checkpoint_before
    assert outside_document.exists()
    assert staged.read_bytes() == b"owned staging"
    assert journal_path.exists()


@pytest.mark.parametrize(
    ("field", "mismatched_value"),
    [
        ("output_root", "/tmp/unrelated-output"),
        ("backend", "trtllm"),
        ("checkpoint_root", "/tmp/unrelated-checkpoints/sglang"),
    ],
)
def test_recovery_rejects_mismatched_transaction_context_before_mutation(
    tmp_path,
    field,
    mismatched_value,
):
    output_root = tmp_path / "out"
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_path = _write_checkpoint(
        checkpoint_dir,
        done=["case-a"],
        failed=[],
        attempted=["case-a"],
    )
    staged = output_root / "gemm_perf.txt"
    staged.parent.mkdir(parents=True)
    staged.write_bytes(b"owned staging")
    _write_sidecar_transaction(
        output_root,
        [(checkpoint_path, {"case-a"})],
        [staged],
        checkpoint_dir=checkpoint_dir,
        tagged={checkpoint_path},
        pending_document=False,
    )
    journal_path = output_root / collect_mod._SIDECAR_TRANSACTION_FILENAME
    transaction = json.loads(journal_path.read_text(encoding="utf-8"))
    transaction[field] = mismatched_value
    journal_path.write_text(json.dumps(transaction), encoding="utf-8")
    checkpoint_before = checkpoint_path.read_bytes()

    with pytest.raises(RuntimeError, match="transaction"):
        collect_mod._recover_collector_provenance_transaction(
            output_root,
            backend=BACKEND,
            checkpoint_dir=str(checkpoint_dir),
        )

    assert checkpoint_path.read_bytes() == checkpoint_before
    assert staged.read_bytes() == b"owned staging"
    assert journal_path.exists()


@pytest.mark.parametrize(
    ("field", "mismatched_value"),
    [
        ("schema", "stale-schema"),
        ("backend", "trtllm"),
        ("module", "sglang.other"),
        ("run_func", "other_run"),
        ("framework_version", "0.5.13"),
        ("sm_version", 90),
    ],
)
def test_recovery_rejects_mismatched_recorded_checkpoint_identity_before_mutation(
    tmp_path,
    field,
    mismatched_value,
):
    output_root = tmp_path / "out"
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_path = _write_checkpoint(
        checkpoint_dir,
        done=["case-a"],
        failed=[],
        attempted=["case-a"],
    )
    staged = output_root / "gemm_perf.txt"
    staged.parent.mkdir(parents=True)
    staged.write_bytes(b"owned staging")
    _write_sidecar_transaction(
        output_root,
        [(checkpoint_path, {"case-a"})],
        [staged],
        checkpoint_dir=checkpoint_dir,
        tagged={checkpoint_path},
        pending_document=False,
    )
    journal_path = output_root / collect_mod._SIDECAR_TRANSACTION_FILENAME
    transaction = json.loads(journal_path.read_text(encoding="utf-8"))
    transaction["checkpoints"][0]["identity"][field] = mismatched_value
    journal_path.write_text(json.dumps(transaction), encoding="utf-8")
    checkpoint_before = checkpoint_path.read_bytes()

    with pytest.raises(RuntimeError, match="checkpoint"):
        collect_mod._recover_collector_provenance_transaction(
            output_root,
            backend=BACKEND,
            checkpoint_dir=str(checkpoint_dir),
        )

    assert checkpoint_path.read_bytes() == checkpoint_before
    assert staged.read_bytes() == b"owned staging"
    assert journal_path.exists()


@pytest.mark.parametrize(
    "mutation",
    [
        "duplicate-participant",
        "duplicate-attempt",
        "malformed-attempt",
        "path-alias",
        "extra-participant-field",
    ],
)
def test_recovery_rejects_malformed_or_duplicate_participants_before_mutation(tmp_path, mutation):
    output_root = tmp_path / "out"
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_path = _write_checkpoint(
        checkpoint_dir,
        done=["case-a"],
        failed=[],
        attempted=["case-a"],
    )
    staged = output_root / "gemm_perf.txt"
    staged.parent.mkdir(parents=True)
    staged.write_bytes(b"owned staging")
    _write_sidecar_transaction(
        output_root,
        [(checkpoint_path, {"case-a"})],
        [staged],
        checkpoint_dir=checkpoint_dir,
        tagged={checkpoint_path},
        pending_document=False,
    )
    journal_path = output_root / collect_mod._SIDECAR_TRANSACTION_FILENAME
    transaction = json.loads(journal_path.read_text(encoding="utf-8"))
    participant = transaction["checkpoints"][0]
    if mutation == "duplicate-participant":
        transaction["checkpoints"].append(dict(participant))
    elif mutation == "duplicate-attempt":
        participant["attempted"].append("case-a")
    elif mutation == "malformed-attempt":
        participant["attempted"] = "case-a"
    elif mutation == "path-alias":
        participant["path"] = f"{checkpoint_path.parent}/./{checkpoint_path.name}"
    else:
        participant["unexpected"] = True
    journal_path.write_text(json.dumps(transaction), encoding="utf-8")
    checkpoint_before = checkpoint_path.read_bytes()

    with pytest.raises(RuntimeError, match="checkpoint"):
        collect_mod._recover_collector_provenance_transaction(
            output_root,
            backend=BACKEND,
            checkpoint_dir=str(checkpoint_dir),
        )

    assert checkpoint_path.read_bytes() == checkpoint_before
    assert staged.read_bytes() == b"owned staging"
    assert journal_path.exists()


def test_recovery_rejects_duplicate_attempt_ids_across_participants_before_mutation(tmp_path):
    output_root = tmp_path / "out"
    checkpoint_dir = tmp_path / "checkpoint"
    first = _write_checkpoint_for(
        checkpoint_dir,
        backend=BACKEND,
        full_name="sglang.mla_bmm_gen_pre",
        version="0.5.14",
        done=["case-shared"],
        failed=[],
        attempted=["case-shared"],
    )
    second = _write_checkpoint_for(
        checkpoint_dir,
        backend=BACKEND,
        full_name="sglang.mla_bmm_gen_post",
        version="0.5.14",
        done=["case-shared"],
        failed=[],
        attempted=["case-shared"],
    )
    staged = output_root / "mla_bmm_perf.txt"
    staged.parent.mkdir(parents=True)
    staged.write_bytes(b"owned staging")
    _write_sidecar_transaction(
        output_root,
        [(first, {"case-shared"}), (second, {"case-shared"})],
        [staged],
        checkpoint_dir=checkpoint_dir,
        tagged=set(),
        pending_document=True,
    )
    first_before = first.read_bytes()
    second_before = second.read_bytes()
    journal_path = output_root / collect_mod._SIDECAR_TRANSACTION_FILENAME

    with pytest.raises(RuntimeError, match="unique across"):
        collect_mod._recover_collector_provenance_transaction(
            output_root,
            backend=BACKEND,
            checkpoint_dir=str(checkpoint_dir),
        )

    assert first.read_bytes() == first_before
    assert second.read_bytes() == second_before
    assert staged.read_bytes() == b"owned staging"
    assert journal_path.exists()


def test_recovery_rejects_unexpected_journal_field_before_mutation(tmp_path):
    output_root = tmp_path / "out"
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_path = _write_checkpoint(
        checkpoint_dir,
        done=["case-a"],
        failed=[],
        attempted=["case-a"],
    )
    staged = output_root / "gemm_perf.txt"
    staged.parent.mkdir(parents=True)
    staged.write_bytes(b"owned staging")
    _write_sidecar_transaction(
        output_root,
        [(checkpoint_path, {"case-a"})],
        [staged],
        checkpoint_dir=checkpoint_dir,
        tagged={checkpoint_path},
        pending_document=False,
    )
    journal_path = output_root / collect_mod._SIDECAR_TRANSACTION_FILENAME
    transaction = json.loads(journal_path.read_text(encoding="utf-8"))
    transaction["unexpected"] = True
    journal_path.write_text(json.dumps(transaction), encoding="utf-8")
    checkpoint_before = checkpoint_path.read_bytes()

    with pytest.raises(RuntimeError, match="fields"):
        collect_mod._recover_collector_provenance_transaction(
            output_root,
            backend=BACKEND,
            checkpoint_dir=str(checkpoint_dir),
        )

    assert checkpoint_path.read_bytes() == checkpoint_before
    assert staged.read_bytes() == b"owned staging"
    assert journal_path.exists()


def test_closed_sidecar_transaction_allows_later_identical_case_plan_event(tmp_path):
    output_root = tmp_path / "out"
    output_root.mkdir()
    staged = output_root / "gemm_perf.txt"
    staged.write_text("op,latency\nmatmul,1.0\n", encoding="utf-8")
    first_finalization: dict[Path, collect_mod.PerfFinalizationInfo] = {}
    parquet_path = collect_mod.finalize_perf_files(
        [staged],
        delete_source=False,
        finalization_info=first_finalization,
    )[0]
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_path = _write_checkpoint(
        checkpoint_dir,
        done=["case-a"],
        failed=[],
        attempted=["case-a"],
    )
    ctx = _provenance_ctx(_collections())

    collect_mod._write_collector_provenance(
        output_root,
        [parquet_path],
        ctx,
        run_errors=[],
        backend=BACKEND,
        checkpoint_dir=str(checkpoint_dir),
        finalization_info=first_finalization,
    )
    assert _attempted(checkpoint_path) == set()

    staged.write_text("op,latency\nmatmul,1.0\n", encoding="utf-8")
    tracker = collect_mod._resume_tracker_for_collection(
        _collections()[0],
        ctx,
        backend=BACKEND,
        checkpoint_dir=str(checkpoint_dir),
        sm_version=100,
    )
    tracker.load_existing()
    tracker.mark_attempted("case-a")
    tracker.mark_passed("case-a")
    tracker.flush(force=True)
    second_finalization: dict[Path, collect_mod.PerfFinalizationInfo] = {}
    assert collect_mod.finalize_perf_files(
        [staged],
        delete_source=False,
        finalization_info=second_finalization,
    ) == [parquet_path]

    collect_mod._write_collector_provenance(
        output_root,
        [parquet_path],
        ctx,
        run_errors=[],
        backend=BACKEND,
        checkpoint_dir=str(checkpoint_dir),
        finalization_info=second_finalization,
    )

    table = yaml.safe_load((output_root / "collection_meta.yaml").read_text(encoding="utf-8"))["tables"]["gemm_perf"]
    assert len(table["collections"]) == 2
    assert [event["case_plan_hash"] for event in table["collections"]] == [
        provenance.case_plan_hash(["case-a"]),
        provenance.case_plan_hash(["case-a"]),
    ]
    assert _attempted(checkpoint_path) == set()


def test_shared_table_hashes_and_closes_every_producing_op_only(tmp_path):
    output_root = tmp_path / "out"
    parquet_path = output_root / "mla_bmm_perf.parquet"
    _write_parquet(parquet_path, [{"op": "shared", "latency": 1.0}])
    checkpoint_dir = tmp_path / "checkpoint"
    collections = _shared_collections()
    first = _write_checkpoint_for(
        checkpoint_dir,
        backend=BACKEND,
        full_name="sglang.mla_bmm_gen_pre",
        version="0.5.14",
        done=["case-a"],
        failed=[],
        attempted=["case-a"],
    )
    second = _write_checkpoint_for(
        checkpoint_dir,
        backend=BACKEND,
        full_name="sglang.mla_bmm_gen_post",
        version="0.5.14",
        done=["case-b"],
        failed=["case-c"],
        attempted=["case-b", "case-c"],
    )
    unrelated = _write_checkpoint_for(
        checkpoint_dir,
        backend=BACKEND,
        full_name="sglang.unrelated",
        version="0.5.14",
        done=["case-z"],
        failed=[],
        attempted=["case-z"],
    )

    collect_mod._write_collector_provenance(
        output_root,
        [parquet_path],
        _provenance_ctx(collections),
        run_errors=[],
        backend=BACKEND,
        checkpoint_dir=str(checkpoint_dir),
        finalization_info=_finalization_info_for(parquet_path),
    )

    table = yaml.safe_load((output_root / "collection_meta.yaml").read_text(encoding="utf-8"))["tables"]["mla_bmm_perf"]
    assert table["case_plan_hash"] == provenance.case_plan_hash(["case-a", "case-b", "case-c"])
    assert _attempted(first) == set()
    assert _attempted(second) == set()
    assert _attempted(unrelated) == {"case-z"}


def test_resume_can_finalize_untouched_staging_file_with_pending_evidence(tmp_path):
    output_root = tmp_path / "out"
    output_root.mkdir()
    staged = output_root / "gemm_perf.txt"
    staged.write_text("op,latency\nmatmul,1.0\n", encoding="utf-8")
    checkpoint_dir = tmp_path / "checkpoint"
    _write_checkpoint(checkpoint_dir, done=["case-from-kept-csv"], failed=[])

    pending = collect_mod._pending_resume_perf_outputs(
        output_root,
        _provenance_ctx(_collections()),
        backend=BACKEND,
        checkpoint_dir=str(checkpoint_dir),
        sm_version=100,
    )

    assert pending == [staged]


@pytest.mark.parametrize(
    ("field", "mismatched_value"),
    [
        ("schema", "stale-schema"),
        ("backend", "trtllm"),
        ("module", "sglang.other"),
        ("run_func", "other_run"),
        ("framework_version", "0.5.13"),
        ("sm_version", 90),
        (None, None),
    ],
    ids=["schema", "backend", "module", "run-func", "framework-version", "sm", "unreadable"],
)
def test_pending_resume_rejects_shared_table_when_one_present_producer_is_invalid(
    tmp_path,
    field,
    mismatched_value,
):
    output_root = tmp_path / "out"
    output_root.mkdir()
    staged = output_root / "mla_bmm_perf.txt"
    staged.write_text("op,latency\nmla_bmm,1.0\n", encoding="utf-8")
    checkpoint_dir = tmp_path / "checkpoint"
    first = _write_checkpoint_for(
        checkpoint_dir,
        backend=BACKEND,
        full_name="sglang.mla_bmm_gen_pre",
        version="0.5.14",
        done=["case-a"],
        failed=[],
        attempted=["case-a"],
    )
    second = _write_checkpoint_for(
        checkpoint_dir,
        backend=BACKEND,
        full_name="sglang.mla_bmm_gen_post",
        version="0.5.14",
        done=["case-b"],
        failed=[],
        attempted=["case-b"],
    )
    if field is None:
        second.write_text("{not valid json", encoding="utf-8")
    else:
        checkpoint = json.loads(second.read_text(encoding="utf-8"))
        checkpoint[field] = mismatched_value
        second.write_text(json.dumps(checkpoint), encoding="utf-8")
    second_before = second.read_bytes()

    with pytest.raises(RuntimeError, match="checkpoint"):
        collect_mod._pending_resume_perf_outputs(
            output_root,
            _provenance_ctx(_shared_collections()),
            backend=BACKEND,
            checkpoint_dir=str(checkpoint_dir),
            sm_version=100,
        )

    assert staged.exists()
    assert _attempted(first) == {"case-a"}
    assert second.read_bytes() == second_before


@pytest.mark.parametrize(
    ("field", "mismatched_value"),
    [
        ("schema", "stale-schema"),
        ("backend", "trtllm"),
        ("module", "sglang.other"),
        ("run_func", "other_run"),
        ("framework_version", "0.5.13"),
        ("sm_version", 90),
        (None, None),
    ],
    ids=["schema", "backend", "module", "run-func", "framework-version", "sm", "unreadable"],
)
def test_writer_rejects_shared_table_when_one_present_producer_is_invalid(
    tmp_path,
    field,
    mismatched_value,
):
    output_root = tmp_path / "out"
    parquet_path = output_root / "mla_bmm_perf.parquet"
    _write_parquet(parquet_path, [{"op": "mla_bmm", "latency": 1.0}])
    staged = output_root / "mla_bmm_perf.txt"
    staged.write_text("op,latency\nmla_bmm,1.0\n", encoding="utf-8")
    checkpoint_dir = tmp_path / "checkpoint"
    first = _write_checkpoint_for(
        checkpoint_dir,
        backend=BACKEND,
        full_name="sglang.mla_bmm_gen_pre",
        version="0.5.14",
        done=["case-a"],
        failed=[],
        attempted=["case-a"],
    )
    second = _write_checkpoint_for(
        checkpoint_dir,
        backend=BACKEND,
        full_name="sglang.mla_bmm_gen_post",
        version="0.5.14",
        done=["case-b"],
        failed=[],
        attempted=["case-b"],
    )
    if field is None:
        second.write_text("{not valid json", encoding="utf-8")
    else:
        checkpoint = json.loads(second.read_text(encoding="utf-8"))
        checkpoint[field] = mismatched_value
        second.write_text(json.dumps(checkpoint), encoding="utf-8")
    second_before = second.read_bytes()

    with pytest.raises(RuntimeError, match="checkpoint"):
        collect_mod._write_collector_provenance(
            output_root,
            [parquet_path],
            _provenance_ctx(_shared_collections()),
            run_errors=[],
            backend=BACKEND,
            checkpoint_dir=str(checkpoint_dir),
            finalization_info=_finalization_info_for(parquet_path),
        )

    assert not (output_root / "collection_meta.yaml").exists()
    assert staged.exists()
    assert _attempted(first) == {"case-a"}
    assert second.read_bytes() == second_before


def test_pending_resume_rejects_shared_table_with_missing_selected_producer_checkpoint(tmp_path):
    output_root = tmp_path / "out"
    output_root.mkdir()
    staged = output_root / "mla_bmm_perf.txt"
    staged.write_text("op,latency\nmla_bmm,1.0\n", encoding="utf-8")
    checkpoint_dir = tmp_path / "checkpoint"
    first = _write_checkpoint_for(
        checkpoint_dir,
        backend=BACKEND,
        full_name="sglang.mla_bmm_gen_pre",
        version="0.5.14",
        done=["case-a"],
        failed=[],
        attempted=["case-a"],
    )

    with pytest.raises(RuntimeError, match="no checkpoint"):
        collect_mod._pending_resume_perf_outputs(
            output_root,
            _provenance_ctx(_shared_collections()),
            backend=BACKEND,
            checkpoint_dir=str(checkpoint_dir),
            sm_version=100,
        )

    assert staged.exists()
    assert _attempted(first) == {"case-a"}


def test_pending_resume_rejects_open_checkpoint_with_missing_staging(tmp_path):
    output_root = tmp_path / "out"
    output_root.mkdir()
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_path = _write_checkpoint(
        checkpoint_dir,
        done=["case-a"],
        failed=[],
        attempted=["case-a"],
    )

    with pytest.raises(RuntimeError, match=r"staged table|staging"):
        collect_mod._pending_resume_perf_outputs(
            output_root,
            _provenance_ctx(_collections()),
            backend=BACKEND,
            checkpoint_dir=str(checkpoint_dir),
            sm_version=100,
        )

    assert _attempted(checkpoint_path) == {"case-a"}
    assert not (output_root / "gemm_perf.txt").exists()


def test_writer_rejects_shared_table_with_missing_selected_producer_checkpoint(tmp_path):
    output_root = tmp_path / "out"
    parquet_path = output_root / "mla_bmm_perf.parquet"
    _write_parquet(parquet_path, [{"op": "mla_bmm", "latency": 1.0}])
    staged = output_root / "mla_bmm_perf.txt"
    staged.write_text("op,latency\nmla_bmm,1.0\n", encoding="utf-8")
    checkpoint_dir = tmp_path / "checkpoint"
    first = _write_checkpoint_for(
        checkpoint_dir,
        backend=BACKEND,
        full_name="sglang.mla_bmm_gen_pre",
        version="0.5.14",
        done=["case-a"],
        failed=[],
        attempted=["case-a"],
    )

    with pytest.raises(RuntimeError, match="no checkpoint"):
        collect_mod._write_collector_provenance(
            output_root,
            [parquet_path],
            _provenance_ctx(_shared_collections()),
            run_errors=[],
            backend=BACKEND,
            checkpoint_dir=str(checkpoint_dir),
            finalization_info=_finalization_info_for(parquet_path),
        )

    assert not (output_root / "collection_meta.yaml").exists()
    assert staged.exists()
    assert _attempted(first) == {"case-a"}


def test_filtered_shared_table_collection_uses_only_selected_producer(tmp_path):
    output_root = tmp_path / "out"
    output_root.mkdir()
    staged = output_root / "mla_bmm_perf.txt"
    staged.write_text("op,latency\nmla_bmm,1.0\n", encoding="utf-8")
    parquet_path = output_root / "mla_bmm_perf.parquet"
    _write_parquet(parquet_path, [{"op": "mla_bmm", "latency": 1.0}])
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_path = _write_checkpoint_for(
        checkpoint_dir,
        backend=BACKEND,
        full_name="sglang.mla_bmm_gen_pre",
        version="0.5.14",
        done=["case-a"],
        failed=[],
        attempted=["case-a"],
    )
    selected_collections = [_shared_collections()[0]]
    ctx = _provenance_ctx(selected_collections)

    assert collect_mod._pending_resume_perf_outputs(
        output_root,
        ctx,
        backend=BACKEND,
        checkpoint_dir=str(checkpoint_dir),
        sm_version=100,
    ) == [staged]

    collect_mod._write_collector_provenance(
        output_root,
        [parquet_path],
        ctx,
        run_errors=[],
        backend=BACKEND,
        checkpoint_dir=str(checkpoint_dir),
        finalization_info=_finalization_info_for(parquet_path),
    )

    assert (output_root / "collection_meta.yaml").exists()
    assert _attempted(checkpoint_path) == set()


@pytest.mark.parametrize(
    ("field", "mismatched_value"),
    [
        ("schema", 999),
        ("backend", "trtllm"),
        ("module", "sglang.other"),
        ("run_func", "other_run"),
        ("framework_version", "0.5.13"),
        ("sm_version", 90),
    ],
)
def test_pending_resume_rejects_checkpoint_with_mismatched_runtime_identity(
    tmp_path,
    field,
    mismatched_value,
):
    output_root = tmp_path / "out"
    output_root.mkdir()
    staged = output_root / "gemm_perf.txt"
    staged.write_text("op,latency\nmatmul,1.0\n", encoding="utf-8")
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_path = _write_checkpoint(checkpoint_dir, done=["stale-case"], failed=[])
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    checkpoint[field] = mismatched_value
    checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")

    with pytest.raises(RuntimeError, match="checkpoint mismatch"):
        collect_mod._pending_resume_perf_outputs(
            output_root,
            _provenance_ctx(_collections()),
            backend=BACKEND,
            checkpoint_dir=str(checkpoint_dir),
            sm_version=100,
        )
    assert staged.exists()
    assert _attempted(checkpoint_path) == {"stale-case"}


def test_provenance_writer_rejects_mismatched_checkpoint_identity(tmp_path):
    output_root = tmp_path / "out"
    parquet_path = output_root / "gemm_perf.parquet"
    _write_parquet(parquet_path, [{"op": "matmul", "latency": 1.0}])
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_path = _write_checkpoint(checkpoint_dir, done=["stale-case"], failed=[])
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    checkpoint["framework_version"] = "0.5.13"
    checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")

    with pytest.raises(RuntimeError, match="checkpoint mismatch"):
        collect_mod._write_collector_provenance(
            output_root,
            [parquet_path],
            _provenance_ctx(_collections()),
            run_errors=[],
            backend=BACKEND,
            checkpoint_dir=str(checkpoint_dir),
            finalization_info=_finalization_info_for(parquet_path),
        )

    assert not (output_root / "collection_meta.yaml").exists()


def test_cumulative_checkpoint_without_current_attempts_cannot_attest_event(tmp_path):
    output_root = tmp_path / "out"
    parquet_path = output_root / "gemm_perf.parquet"
    _write_parquet(parquet_path, [{"op": "old", "latency": 1.0}])
    checkpoint_dir = tmp_path / "checkpoint"
    _write_checkpoint(checkpoint_dir, done=["case-from-prior-invocation"], failed=[], attempted=[])

    with pytest.raises(RuntimeError, match="Zero attempted cases"):
        collect_mod._write_collector_provenance(
            output_root,
            [parquet_path],
            _provenance_ctx(_collections()),
            run_errors=[],
            backend=BACKEND,
            checkpoint_dir=str(checkpoint_dir),
            finalization_info=_finalization_info_for(parquet_path),
        )

    assert not (output_root / "collection_meta.yaml").exists()


def test_schema_mismatch_overwrite_replaces_stale_table_history(tmp_path):
    output_root = tmp_path / "out"
    provenance.write_collection_meta(
        output_root,
        {
            "framework": "sglang",
            "version": "0.5.14",
            "image": "lmsysorg/sglang:v0.5.14",
            "image_digest": "sha256:" + "0" * 64,
        },
        {
            "gemm_perf": {
                "collector_ref": "a" * 40,
                "collector_hash": "sha256:" + "b" * 64,
                "case_plan_hash": "sha256:" + "c" * 64,
                "collected_at": "2026-08-10",
                "rows": 1,
                "status": "complete",
            }
        },
        provenance_tier="local",
    )

    parquet_path = output_root / "gemm_perf.parquet"
    _write_parquet(parquet_path, [{"op": "matmul", "latency": 1.0}])
    perf_path = output_root / "gemm_perf.txt"
    perf_path.write_text("shape,latency\nnew-shape,2.0\n")
    finalization_info: dict[Path, collect_mod.PerfFinalizationInfo] = {}
    assert collect_mod.finalize_perf_files([perf_path], delete_source=False, finalization_info=finalization_info) == [
        parquet_path
    ]
    assert pq.read_table(parquet_path).to_pylist() == [{"shape": "new-shape", "latency": 2.0}]
    assert finalization_info[parquet_path.resolve()] == _finalization_fact(
        perf_path,
        new_rows=1,
        merged_existing=False,
    )

    checkpoint_dir = tmp_path / "checkpoint"
    _write_checkpoint(checkpoint_dir, done=["case-new"], failed=[])
    collect_mod._write_collector_provenance(
        output_root,
        [parquet_path],
        _provenance_ctx(_collections()),
        run_errors=[],
        backend=BACKEND,
        checkpoint_dir=str(checkpoint_dir),
        finalization_info=finalization_info,
    )

    doc = yaml.safe_load((output_root / "collection_meta.yaml").read_text(encoding="utf-8"))
    table = doc["tables"]["gemm_perf"]
    assert doc["schema_version"] == 1
    assert "provenance" not in doc
    assert "collections" not in table
    assert table["case_plan_hash"] == provenance.case_plan_hash(["case-new"])
    assert table["rows"] == 1


def test_runtime_change_rejects_schema_replacement_when_an_old_table_survives(tmp_path):
    output_root = tmp_path / "out"
    existing_doc = {
        "schema_version": 1,
        "runtime": {
            "framework": "sglang",
            "version": "0.5.13",
            "image": "lmsysorg/sglang:v0.5.13",
            "image_digest": "sha256:" + "1" * 64,
        },
        "tables": {"gemm_perf": _collection_event(), "other_perf": _collection_event()},
    }
    output_root.mkdir(parents=True)
    meta_path = output_root / "collection_meta.yaml"
    meta_path.write_text(yaml.safe_dump(existing_doc, sort_keys=False))

    parquet_path = output_root / "gemm_perf.parquet"
    _write_parquet(parquet_path, [{"op": "matmul", "latency": 1.0}])
    perf_path = output_root / "gemm_perf.txt"
    perf_path.write_text("shape,latency\nnew-shape,2.0\n")
    finalization_info: dict[Path, collect_mod.PerfFinalizationInfo] = {}
    assert collect_mod.finalize_perf_files([perf_path], finalization_info=finalization_info) == [parquet_path]
    assert finalization_info[parquet_path.resolve()].merged_existing is False

    checkpoint_dir = tmp_path / "checkpoint"
    _write_checkpoint(checkpoint_dir, done=["case-new"], failed=[])
    with pytest.raises(RuntimeError, match="different runtime identity"):
        collect_mod._write_collector_provenance(
            output_root,
            [parquet_path],
            _provenance_ctx(_collections()),
            run_errors=[],
            backend=BACKEND,
            checkpoint_dir=str(checkpoint_dir),
            finalization_info=finalization_info,
        )

    assert yaml.safe_load(meta_path.read_text(encoding="utf-8")) == existing_doc


def test_runtime_change_preflight_preserves_sidecar_parquet_and_staging_csv(tmp_path):
    output_root = tmp_path / "out"
    existing_runtime = {
        "framework": "sglang",
        "version": "0.5.13",
        "image": "lmsysorg/sglang:v0.5.13",
        "image_digest": "sha256:" + "1" * 64,
    }
    _write_reduced_collection_meta(
        output_root,
        existing_runtime,
        {"gemm_perf": {"status": "complete"}},
    )
    meta_path = output_root / "collection_meta.yaml"
    parquet_path = output_root / "gemm_perf.parquet"
    _write_parquet(parquet_path, [{"op": "matmul", "latency": 1.0}])
    perf_path = output_root / "gemm_perf.txt"
    perf_path.write_text("shape,latency\nnew-shape,2.0\n")

    original_meta = meta_path.read_bytes()
    original_parquet = parquet_path.read_bytes()
    original_staging = perf_path.read_bytes()

    with pytest.raises(RuntimeError, match="different runtime identity"):
        collect_mod._preflight_collector_provenance(output_root, _provenance_ctx(_collections()))

    assert meta_path.read_bytes() == original_meta
    assert parquet_path.read_bytes() == original_parquet
    assert perf_path.read_bytes() == original_staging


def test_runtime_change_rejects_even_when_every_old_table_was_replaced(tmp_path):
    output_root = tmp_path / "out"
    _write_reduced_collection_meta(
        output_root,
        {
            "framework": "sglang",
            "version": "0.5.13",
            "image": "lmsysorg/sglang:v0.5.13",
            "image_digest": "sha256:" + "1" * 64,
        },
        {"gemm_perf": _collection_event(), "moe_perf": _collection_event()},
    )

    perf_paths = [output_root / "gemm_perf.txt", output_root / "moe_perf.txt"]
    parquet_paths = [path.with_suffix(".parquet") for path in perf_paths]
    for parquet_path in parquet_paths:
        _write_parquet(parquet_path, [{"op": "matmul", "latency": 1.0}])
    for perf_path in perf_paths:
        perf_path.write_text("shape,latency\nnew-shape,2.0\n")
    finalization_info: dict[Path, collect_mod.PerfFinalizationInfo] = {}
    assert collect_mod.finalize_perf_files(perf_paths, finalization_info=finalization_info) == parquet_paths
    assert all(not info.merged_existing for info in finalization_info.values())

    checkpoint_dir = tmp_path / "checkpoint"
    _write_checkpoint(checkpoint_dir, done=["case-new"], failed=[])
    _write_checkpoint_for(
        checkpoint_dir,
        backend=BACKEND,
        full_name="sglang.moe",
        version="0.5.14",
        done=["moe-case-new"],
        failed=[],
    )
    collections = [
        *_collections(),
        {
            "name": BACKEND,
            "type": "moe",
            "module": "collector.sglang.collect_moe",
            "run_func": "run_moe_torch",
            "perf_filename": "moe_perf.txt",
        },
    ]
    with pytest.raises(RuntimeError, match="different runtime identity"):
        collect_mod._write_collector_provenance(
            output_root,
            parquet_paths,
            _provenance_ctx(collections),
            run_errors=[],
            backend=BACKEND,
            checkpoint_dir=str(checkpoint_dir),
            finalization_info=finalization_info,
        )

    doc = yaml.safe_load((output_root / "collection_meta.yaml").read_text(encoding="utf-8"))
    assert doc["runtime"]["version"] == "0.5.13"


def test_finalize_raises_when_existing_sidecar_is_legacy_tier(tmp_path):
    """A legacy sidecar (provenance: legacy, synthesized by migrate_markers.py for
    pre-V3 data) must never be silently merged-and-rebuilt by a fresh collection —
    that would drop the legacy tier tag. This must fail loudly instead."""
    output_root = tmp_path / "out"
    output_root.mkdir(parents=True)
    legacy_doc = {
        "schema_version": 1,
        "provenance": "legacy",
        "runtime": {"framework": "sglang", "version": "0.5.10"},
        "tables": {"gemm_perf": {"status": "complete"}},
    }
    (output_root / "collection_meta.yaml").write_text(yaml.safe_dump(legacy_doc, sort_keys=False))

    parquet_path = output_root / "gemm_perf.parquet"
    _write_parquet(parquet_path, [{"op": "matmul", "latency": 1.0}])
    checkpoint_dir = tmp_path / "checkpoint"
    _write_checkpoint(checkpoint_dir, done=["case-a"], failed=[])

    with pytest.raises(RuntimeError, match=re.escape(str(output_root))):
        collect_mod._write_collector_provenance(
            output_root,
            [parquet_path],
            _provenance_ctx(_collections()),
            run_errors=[],
            backend=BACKEND,
            checkpoint_dir=str(checkpoint_dir),
            finalization_info=_finalization_info_for(parquet_path),
        )

    # The legacy sidecar itself must be left untouched, not partially rebuilt.
    doc = yaml.safe_load((output_root / "collection_meta.yaml").read_text(encoding="utf-8"))
    assert doc == legacy_doc
