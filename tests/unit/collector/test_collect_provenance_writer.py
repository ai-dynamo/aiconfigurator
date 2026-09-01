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
import multiprocessing as mp
import os
import re
import stat
import sys
from contextlib import contextmanager
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

import helper as helper_mod
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


def _write_perf_event_through_logger(path: Path) -> None:
    import helper as helper_mod

    assert helper_mod.log_perf(
        [{"op": "matmul", "shape": "new", "latency": 1.0}],
        "sglang",
        "0.5.14",
        "test-device",
        "matmul",
        "test",
        str(path),
    )


def _assert_retained_slot(path: Path) -> None:
    path_stat = path.lstat()
    assert stat.S_ISREG(path_stat.st_mode)
    assert path_stat.st_nlink == 1


def _private_atomic_artifact_names(destination: Path) -> set[str]:
    prefix = f".{destination.name}."
    return {path.name for path in destination.parent.iterdir() if path.name.startswith(prefix)}


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


def _single_table_finalization(tmp_path):
    output_root = tmp_path / "out"
    output_root.mkdir()
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_path = _write_checkpoint(
        checkpoint_dir,
        done=["case-a"],
        failed=[],
        attempted=["case-a"],
    )
    staging_path = output_root / "gemm_perf.txt"
    staging_path.write_text("op,shape,latency\nmatmul,new,1.0\n", encoding="utf-8")
    parquet_path = staging_path.with_suffix(".parquet")
    pq.write_table(pa.table({"op": ["matmul"], "shape": ["old"], "latency": [9.0]}), parquet_path)
    return output_root, checkpoint_dir, checkpoint_path, staging_path, parquet_path


def _run_hard_exit(target, args, expected_exitcode, description):
    process = mp.get_context("spawn").Process(target=target, args=args)
    process.start()
    process.join(timeout=20)
    if process.is_alive():
        process.terminate()
        process.join()
        pytest.fail(f"{description} subprocess did not exit")
    assert process.exitcode == expected_exitcode


def _mutate_checkpoint_and_exit(checkpoint_path_text: str, operation: str, transaction_id: str) -> None:
    checkpoint_path = Path(checkpoint_path_text)
    try:
        if operation == "tag":
            collect_mod._tag_checkpoint_sidecar_transaction(
                checkpoint_path,
                {"case-a"},
                transaction_id,
            )
        else:
            collect_mod._close_checkpoint_attempts(
                checkpoint_path,
                {"case-a"},
                transaction_id=transaction_id,
            )
    except BaseException:
        os._exit(17)
    os._exit(0)


def _swap_fifo_after_checkpoint_publication_and_exit(
    checkpoint_path_text: str,
    operation: str,
    transaction_id: str,
    reservation_evidence_text: str,
) -> None:
    checkpoint_path = Path(checkpoint_path_text)
    reservation_evidence = Path(reservation_evidence_text)
    expected = collect_mod._checkpoint_snapshot(
        checkpoint_path,
        None,
        context_path=checkpoint_path,
    ).attest(checkpoint_path)
    reservation_name = collect_mod.atomic_write_reservation_path(checkpoint_path).name
    real_rename_noreplace = collect_mod._rename_noreplace_at
    swapped = False

    def swap_after_publication(source, target, directory_fd):
        nonlocal swapped
        result = real_rename_noreplace(source, target, directory_fd)
        if not swapped and source == reservation_name and target == checkpoint_path.name:
            swapped = True
            os.rename(
                target,
                reservation_evidence.name,
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
            )
            os.mkfifo(target, 0o600, dir_fd=directory_fd)
            os.fsync(directory_fd)
        return result

    collect_mod._rename_noreplace_at = swap_after_publication
    try:
        if operation == "tag":
            collect_mod._tag_checkpoint_sidecar_transaction(
                checkpoint_path,
                {"case-a"},
                transaction_id,
                expected_attestation=expected,
            )
        else:
            collect_mod._close_checkpoint_attempts(
                checkpoint_path,
                {"case-a"},
                transaction_id=transaction_id,
                expected_attestation=expected,
            )
    except BaseException:
        os._exit(17)
    os._exit(0)


def _assert_subprocess_fails_without_blocking(target, args, description: str) -> None:
    process = mp.get_context("spawn").Process(target=target, args=args)
    process.start()
    process.join(timeout=3)
    if process.is_alive():
        process.terminate()
        process.join()
        pytest.fail(f"{description} blocked on a nonregular file")
    assert process.exitcode not in (None, 0)


def _assert_checkpoint_mutation_fails_without_blocking(checkpoint_path: Path, operation: str) -> None:
    _assert_subprocess_fails_without_blocking(
        _mutate_checkpoint_and_exit,
        (str(checkpoint_path), operation, "a" * 32),
        f"checkpoint {operation}",
    )


def _recover_finalization(output_root: Path, checkpoint_dir: Path):
    return collect_mod._recover_collector_provenance_transaction(
        output_root,
        backend=BACKEND,
        checkpoint_dir=str(checkpoint_dir),
    )


def _finalize_single_table(output_root: Path, checkpoint_dir: Path, staging_path: Path) -> None:
    collect_mod._finalize_collector_outputs_transaction(
        output_root,
        [staging_path],
        _provenance_ctx(_collections()),
        [],
        backend=BACKEND,
        checkpoint_dir=str(checkpoint_dir),
        sm_version=100,
    )


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


def _crash_after_first_parquet_publish(
    output_root_text: str,
    checkpoint_dir_text: str,
    collections: list[dict],
) -> None:
    import helper as helper_mod

    output_root = Path(output_root_text)
    real_rename_noreplace = helper_mod._rename_noreplace_at
    published = False

    def publish_then_exit(source, target, directory_fd):
        nonlocal published
        real_rename_noreplace(source, target, directory_fd)
        if source.endswith(".tmp") and Path(target).suffix == ".parquet" and not published:
            published = True
            os._exit(86)

    helper_mod._rename_noreplace_at = publish_then_exit
    collect_mod._finalize_collector_outputs_transaction(
        output_root,
        sorted(output_root.glob("*_perf.txt")),
        _provenance_ctx(collections),
        [],
        backend=BACKEND,
        checkpoint_dir=checkpoint_dir_text,
        sm_version=100,
    )


def _pause_displaced_collector_finalization(
    output_root_text: str,
    checkpoint_dir_text: str,
    ready,
    release,
) -> None:
    output_root = Path(output_root_text)
    real_lifecycle = collect_mod.perf_finalization_lifecycle

    @contextmanager
    def pause_after_lock(path, *args, **kwargs):
        with real_lifecycle(path, *args, **kwargs) as locked_root:
            ready.set()
            if not release.wait(timeout=20):
                raise RuntimeError("timed out waiting to resume displaced finalizer")
            yield locked_root

    collect_mod.perf_finalization_lifecycle = pause_after_lock
    collect_mod._finalize_collector_outputs_transaction(
        output_root,
        [output_root / "gemm_perf.txt"],
        _provenance_ctx(_collections()),
        [],
        backend=BACKEND,
        checkpoint_dir=checkpoint_dir_text,
        sm_version=100,
    )


def _pause_displaced_collector_recovery(
    output_root_text: str,
    checkpoint_dir_text: str,
    ready,
    release,
) -> None:
    output_root = Path(output_root_text)
    real_lifecycle = collect_mod.perf_finalization_lifecycle

    @contextmanager
    def pause_after_lock(path, *args, **kwargs):
        with real_lifecycle(path, *args, **kwargs) as locked_root:
            ready.set()
            if not release.wait(timeout=20):
                raise RuntimeError("timed out waiting to resume displaced recovery")
            yield locked_root

    collect_mod.perf_finalization_lifecycle = pause_after_lock
    collect_mod._recover_collector_provenance_transaction(
        output_root,
        backend=BACKEND,
        checkpoint_dir=checkpoint_dir_text,
    )


def _crash_before_perf_to_sidecar_handoff(output_root_text: str, checkpoint_dir_text: str) -> None:
    output_root = Path(output_root_text)

    def exit_before_handoff(*_args, **_kwargs):
        os._exit(87)

    collect_mod._CollectorPerfPublicationTransaction.handoff_to_sidecar = exit_before_handoff
    collect_mod._finalize_collector_outputs_transaction(
        output_root,
        [output_root / "gemm_perf.txt"],
        _provenance_ctx(_collections()),
        [],
        backend=BACKEND,
        checkpoint_dir=checkpoint_dir_text,
        sm_version=100,
    )


def _crash_after_unjournaled_sidecar_staging(output_root_text: str, checkpoint_dir_text: str) -> None:
    output_root = Path(output_root_text)
    real_atomic_write = collect_mod._atomic_write_bytes

    def write_then_exit(path, *args, **kwargs):
        result = real_atomic_write(path, *args, **kwargs)
        if Path(path).name == collect_mod._SIDECAR_STAGING_FILENAME:
            os._exit(88)
        return result

    collect_mod._atomic_write_bytes = write_then_exit
    collect_mod._finalize_collector_outputs_transaction(
        output_root,
        [output_root / "gemm_perf.txt"],
        _provenance_ctx(_collections()),
        [],
        backend=BACKEND,
        checkpoint_dir=checkpoint_dir_text,
        sm_version=100,
    )


def _crash_after_first_parquet_rollback(output_root_text: str, checkpoint_dir_text: str) -> None:
    import helper as helper_mod

    output_root = Path(output_root_text)
    real_rename_noreplace = helper_mod._rename_noreplace_at
    restored = False

    def restore_then_exit(source, target, directory_fd):
        nonlocal restored
        real_rename_noreplace(source, target, directory_fd)
        if Path(source).suffix == ".parquet" and target.endswith(".tmp") and not restored:
            restored = True
            os._exit(89)

    helper_mod._rename_noreplace_at = restore_then_exit
    collect_mod._recover_collector_provenance_transaction(
        output_root,
        backend=BACKEND,
        checkpoint_dir=checkpoint_dir_text,
    )


def _crash_after_perf_journal_cleanup(output_root_text: str, checkpoint_dir_text: str) -> None:
    output_root = Path(output_root_text)
    real_cleanup = collect_mod._cleanup_transaction_files

    def cleanup_then_exit(attestations, *args, **kwargs):
        attestations = list(attestations)
        result = real_cleanup(attestations, *args, **kwargs)
        if any(attestation.path.name == collect_mod._PERF_TRANSACTION_FILENAME for attestation in attestations):
            os._exit(90)
        return result

    collect_mod._cleanup_transaction_files = cleanup_then_exit
    collect_mod._finalize_collector_outputs_transaction(
        output_root,
        [output_root / "gemm_perf.txt"],
        _provenance_ctx(_collections()),
        [],
        backend=BACKEND,
        checkpoint_dir=checkpoint_dir_text,
        sm_version=100,
    )


def _crash_before_perf_journal_publish(output_root_text: str, checkpoint_dir_text: str) -> None:
    output_root = Path(output_root_text)

    def exit_before_journal(_transaction, _publications):
        os._exit(91)

    collect_mod._CollectorPerfPublicationTransaction.prepare = exit_before_journal
    collect_mod._finalize_collector_outputs_transaction(
        output_root,
        [output_root / "gemm_perf.txt"],
        _provenance_ctx(_collections()),
        [],
        backend=BACKEND,
        checkpoint_dir=checkpoint_dir_text,
        sm_version=100,
    )


def _crash_after_first_parquet_render(
    output_root_text: str,
    checkpoint_dir_text: str,
    collections: list[dict],
) -> None:
    import helper as helper_mod

    output_root = Path(output_root_text)
    real_prepare = helper_mod._prepare_perf_file
    rendered = False

    def prepare_then_exit(*args, **kwargs):
        nonlocal rendered
        prepared = real_prepare(*args, **kwargs)
        if not rendered:
            rendered = True
            os._exit(93)
        return prepared

    helper_mod._prepare_perf_file = prepare_then_exit
    collect_mod._finalize_collector_outputs_transaction(
        output_root,
        sorted(output_root.glob("*_perf.txt")),
        _provenance_ctx(collections),
        [],
        backend=BACKEND,
        checkpoint_dir=checkpoint_dir_text,
        sm_version=100,
    )


def _crash_during_first_parquet_render(output_root_text: str, checkpoint_dir_text: str) -> None:
    import pyarrow.parquet as child_pq

    output_root = Path(output_root_text)

    def write_partial_then_exit(_table, destination, **_kwargs):
        destination.write(b"partial parquet owned by interrupted render")
        destination.flush()
        os.fsync(destination.fileno())
        os._exit(94)

    child_pq.write_table = write_partial_then_exit
    collect_mod._finalize_collector_outputs_transaction(
        output_root,
        [output_root / "gemm_perf.txt"],
        _provenance_ctx(_collections()),
        [],
        backend=BACKEND,
        checkpoint_dir=checkpoint_dir_text,
        sm_version=100,
    )


def _crash_after_perf_journal_prepare(output_root_text: str, checkpoint_dir_text: str) -> None:
    output_root = Path(output_root_text)
    real_prepare = collect_mod._CollectorPerfPublicationTransaction.prepare

    def prepare_then_exit(transaction, publications):
        real_prepare(transaction, publications)
        os._exit(95)

    collect_mod._CollectorPerfPublicationTransaction.prepare = prepare_then_exit
    collect_mod._finalize_collector_outputs_transaction(
        output_root,
        [output_root / "gemm_perf.txt"],
        _provenance_ctx(_collections()),
        [],
        backend=BACKEND,
        checkpoint_dir=checkpoint_dir_text,
        sm_version=100,
    )


def _crash_around_atomic_journal_publication(
    output_root_text: str,
    checkpoint_dir_text: str,
    journal_name: str,
    after_publication: bool,
    exit_code: int,
) -> None:
    import helper as helper_mod

    output_root = Path(output_root_text)
    real_rename_noreplace = helper_mod._rename_noreplace_at

    def crash_at_selected_publication(source, target, directory_fd):
        if target != journal_name:
            return real_rename_noreplace(source, target, directory_fd)
        if after_publication:
            real_rename_noreplace(source, target, directory_fd)
        os._exit(exit_code)

    helper_mod._rename_noreplace_at = crash_at_selected_publication
    collect_mod._finalize_collector_outputs_transaction(
        output_root,
        [output_root / "gemm_perf.txt"],
        _provenance_ctx(_collections()),
        [],
        backend=BACKEND,
        checkpoint_dir=checkpoint_dir_text,
        sm_version=100,
    )


def _crash_checkpoint_before_publication(checkpoint_path_text: str, exit_code: int) -> None:
    checkpoint_path = Path(checkpoint_path_text)
    real_rename_noreplace_at = collect_mod._rename_noreplace_at

    def crash_at_checkpoint_publication(source, target, directory_fd):
        if source == checkpoint_path.name:
            os._exit(exit_code)
        return real_rename_noreplace_at(source, target, directory_fd)

    collect_mod._rename_noreplace_at = crash_at_checkpoint_publication
    collect_mod._atomic_write_json(checkpoint_path, {"attempted": ["new"]})


def _crash_checkpoint_after_canonical_rotation(checkpoint_path_text: str, exit_code: int) -> None:
    checkpoint_path = Path(checkpoint_path_text)
    replacement = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    replacement["done"] = ["replacement"]
    replacement["failed"] = []
    replacement["attempted"] = ["replacement"]
    real_rename_noreplace_at = collect_mod._rename_noreplace_at

    def crash_after_checkpoint_rotation(source, target, directory_fd):
        result = real_rename_noreplace_at(source, target, directory_fd)
        if source == checkpoint_path.name:
            os._exit(exit_code)
        return result

    collect_mod._rename_noreplace_at = crash_after_checkpoint_rotation
    collect_mod._atomic_write_json(checkpoint_path, replacement)


def _crash_during_transaction_checkpoint_rotation(
    output_root_text: str,
    checkpoint_dir_text: str,
    operation: str,
    exit_code: int,
) -> None:
    output_root = Path(output_root_text)
    checkpoint_path = Path(checkpoint_dir_text) / BACKEND / f"{FULL_NAME}.json"
    previous_name = f".{checkpoint_path.name}.tmp.previous"
    real_rename_noreplace_at = collect_mod._rename_noreplace_at
    rotations = 0

    def crash_after_selected_rotation(source, target, directory_fd):
        nonlocal rotations
        result = real_rename_noreplace_at(source, target, directory_fd)
        if source == checkpoint_path.name and target == previous_name:
            rotations += 1
            if (operation == "tag" and rotations == 1) or (operation == "close" and rotations == 2):
                os._exit(exit_code)
        return result

    collect_mod._rename_noreplace_at = crash_after_selected_rotation
    collect_mod._finalize_collector_outputs_transaction(
        output_root,
        [output_root / "gemm_perf.txt"],
        _provenance_ctx(_collections()),
        [],
        backend=BACKEND,
        checkpoint_dir=checkpoint_dir_text,
        sm_version=100,
    )


def _pause_checkpoint_writer(checkpoint_path_text: str, entered, release) -> None:
    checkpoint_path = Path(checkpoint_path_text)
    real_rename_noreplace_at = collect_mod._rename_noreplace_at

    def pause_before_checkpoint_rotation(source, target, directory_fd):
        if source == checkpoint_path.name:
            entered.set()
            if not release.wait(timeout=20):
                raise RuntimeError("timed out waiting to release checkpoint writer")
        return real_rename_noreplace_at(source, target, directory_fd)

    collect_mod._rename_noreplace_at = pause_before_checkpoint_rotation
    collect_mod._atomic_write_json(checkpoint_path, {"attempted": ["first"]})


def _pause_valid_checkpoint_writer_after_rotation(checkpoint_path_text: str, entered, release) -> None:
    checkpoint_path = Path(checkpoint_path_text)
    replacement = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    replacement["done"] = ["replacement"]
    replacement["failed"] = []
    replacement["attempted"] = ["replacement"]
    real_rename_noreplace_at = collect_mod._rename_noreplace_at

    def pause_after_checkpoint_rotation(source, target, directory_fd):
        result = real_rename_noreplace_at(source, target, directory_fd)
        if source == checkpoint_path.name:
            entered.set()
            if not release.wait(timeout=20):
                raise RuntimeError("timed out waiting to release checkpoint writer")
        return result

    collect_mod._rename_noreplace_at = pause_after_checkpoint_rotation
    collect_mod._atomic_write_json(checkpoint_path, replacement)


def _write_checkpoint_and_signal(checkpoint_path_text: str, started, finished) -> None:
    started.set()
    collect_mod._atomic_write_json(Path(checkpoint_path_text), {"attempted": ["second"]})
    finished.set()


def _load_checkpoint_and_signal(checkpoint_dir_text: str, started, finished, result_queue) -> None:
    started.set()
    checkpoint = collect_mod.ResumeCheckpoint(
        backend=BACKEND,
        module_name=FULL_NAME,
        run_func_name=RUN_FUNC,
        checkpoint_dir=checkpoint_dir_text,
        framework_version="0.5.14",
        sm_version=100,
    )
    checkpoint.load_existing()
    result_queue.put((checkpoint._done, checkpoint._failed, checkpoint._attempted))
    finished.set()


def _pause_checkpoint_after_parent_lock(
    checkpoint_dir_text: str,
    checkpoint_path_text: str,
    document: dict | None,
    entered,
    release,
) -> None:
    if document is not None:
        real_rename_noreplace_at = collect_mod._rename_noreplace_at
        paused = False

        def pause_before_first_rename(source, target, directory_fd):
            nonlocal paused
            if not paused:
                paused = True
                entered.set()
                if not release.wait(timeout=20):
                    raise RuntimeError("timed out waiting to release checkpoint operation")
            return real_rename_noreplace_at(source, target, directory_fd)

        collect_mod._rename_noreplace_at = pause_before_first_rename
        collect_mod._atomic_write_json(Path(checkpoint_path_text), document)
        return

    real_lifecycle = collect_mod.perf_finalization_lifecycle
    paused = False

    @contextmanager
    def pause_after_lock(directory, **kwargs):
        nonlocal paused
        with real_lifecycle(directory, **kwargs) as directory_fd:
            if not paused:
                paused = True
                entered.set()
                if not release.wait(timeout=20):
                    raise RuntimeError("timed out waiting to release checkpoint operation")
            yield directory_fd

    collect_mod.perf_finalization_lifecycle = pause_after_lock
    checkpoint = collect_mod.ResumeCheckpoint(
        backend=BACKEND,
        module_name=FULL_NAME,
        run_func_name=RUN_FUNC,
        checkpoint_dir=checkpoint_dir_text,
        framework_version="0.5.14",
        sm_version=100,
    )
    checkpoint.load_existing()


def _write_checkpoint_document_and_signal(checkpoint_path_text: str, document: dict, finished) -> None:
    collect_mod._atomic_write_json(Path(checkpoint_path_text), document)
    finished.set()


def _pause_live_finalization_before_sidecar(
    output_root_text: str,
    checkpoint_dir_text: str,
    entered,
    release,
) -> None:
    output_root = Path(output_root_text)
    real_write = collect_mod._write_collector_provenance

    def pause_then_write(*args, **kwargs):
        entered.set()
        if not release.wait(timeout=20):
            raise RuntimeError("timed out waiting to release live finalization")
        return real_write(*args, **kwargs)

    collect_mod._write_collector_provenance = pause_then_write
    collect_mod._finalize_collector_outputs_transaction(
        output_root,
        [output_root / "gemm_perf.txt"],
        _provenance_ctx(_collections()),
        [],
        backend=BACKEND,
        checkpoint_dir=checkpoint_dir_text,
        sm_version=100,
    )


def _recover_and_signal(
    output_root_text: str,
    checkpoint_dir_text: str,
    started,
    finished,
) -> None:
    started.set()
    try:
        collect_mod._recover_collector_provenance_transaction(
            Path(output_root_text),
            backend=BACKEND,
            checkpoint_dir=checkpoint_dir_text,
        )
    finally:
        finished.set()


def _crash_after_perf_journal_source_chmod(output_root_text: str, checkpoint_dir_text: str) -> None:
    output_root = Path(output_root_text)
    real_prepare = collect_mod._CollectorPerfPublicationTransaction.prepare

    def prepare_chmod_then_exit(transaction, publications):
        real_prepare(transaction, publications)
        publications[0].source.path.chmod(0o400)
        os._exit(92)

    collect_mod._CollectorPerfPublicationTransaction.prepare = prepare_chmod_then_exit
    collect_mod._finalize_collector_outputs_transaction(
        output_root,
        [output_root / "gemm_perf.txt"],
        _provenance_ctx(_collections()),
        [],
        backend=BACKEND,
        checkpoint_dir=checkpoint_dir_text,
        sm_version=100,
    )


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
        "perf_publications": None,
    }
    (output_root / collect_mod._SIDECAR_TRANSACTION_FILENAME).write_text(
        json.dumps(transaction),
        encoding="utf-8",
    )


def test_atomic_checkpoint_replace_failure_preserves_previous_document(tmp_path, monkeypatch):
    checkpoint_path = tmp_path / "checkpoint.json"
    checkpoint_path.write_text('{"attempted": ["old"]}', encoding="utf-8")

    real_rename_noreplace_at = collect_mod._rename_noreplace_at

    def fail_checkpoint_rotation(source, target, directory_fd):
        if source == checkpoint_path.name:
            raise OSError("simulated replace failure")
        return real_rename_noreplace_at(source, target, directory_fd)

    monkeypatch.setattr(collect_mod, "_rename_noreplace_at", fail_checkpoint_rotation)

    with pytest.raises(OSError, match="simulated replace failure"):
        collect_mod._atomic_write_json(checkpoint_path, {"attempted": ["new"]})

    assert json.loads(checkpoint_path.read_text(encoding="utf-8")) == {"attempted": ["old"]}
    assert (tmp_path / ".checkpoint.json.tmp").is_file()
    assert list(tmp_path.glob(".checkpoint.json.*.tmp")) == []


@pytest.mark.parametrize("operation", ["tag", "close"])
def test_checkpoint_mutation_rejects_late_canonical_swap_before_rotation(tmp_path, monkeypatch, operation):
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_path = _write_checkpoint(
        checkpoint_dir,
        done=["case-a"],
        failed=[],
        attempted=["case-a"],
    )
    transaction_id = "a" * 32
    if operation == "close":
        document = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        document[collect_mod._SIDECAR_TRANSACTION_FIELD] = transaction_id
        checkpoint_path.write_text(json.dumps(document), encoding="utf-8")
    original = checkpoint_path.read_bytes()
    expected = collect_mod._checkpoint_snapshot(
        checkpoint_path,
        None,
        context_path=checkpoint_path,
    ).attest(checkpoint_path)
    stale_path = checkpoint_path.with_name("stale-checkpoint.json")
    foreign = b'{"foreign": true}'
    real_atomic_replace = collect_mod._atomic_replace_bytes_at
    swapped = False

    def swap_canonical_before_rotation(directory_fd, path, contents, mode, *args, **kwargs):
        nonlocal swapped
        if not swapped:
            swapped = True
            os.rename(
                checkpoint_path.name,
                stale_path.name,
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
            )
            foreign_fd = os.open(
                checkpoint_path.name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=directory_fd,
            )
            with os.fdopen(foreign_fd, "wb") as foreign_file:
                foreign_file.write(foreign)
                foreign_file.flush()
                os.fsync(foreign_file.fileno())
            os.fsync(directory_fd)
        return real_atomic_replace(directory_fd, path, contents, mode, *args, **kwargs)

    monkeypatch.setattr(collect_mod, "_atomic_replace_bytes_at", swap_canonical_before_rotation)

    with pytest.raises(RuntimeError, match=r"checkpoint|artifact|changed"):
        if operation == "tag":
            collect_mod._tag_checkpoint_sidecar_transaction(
                checkpoint_path,
                {"case-a"},
                transaction_id,
                expected_attestation=expected,
            )
        else:
            collect_mod._close_checkpoint_attempts(
                checkpoint_path,
                {"case-a"},
                transaction_id=transaction_id,
                expected_attestation=expected,
            )

    assert swapped
    assert checkpoint_path.read_bytes() == foreign
    assert stale_path.read_bytes() == original


@pytest.mark.parametrize("operation", ["tag", "close"])
@pytest.mark.parametrize("replacement_kind", ["symlink", "directory", "fifo"])
def test_checkpoint_mutation_restores_unknown_object_swapped_during_rotation(
    tmp_path,
    monkeypatch,
    operation,
    replacement_kind,
):
    checkpoint_path = _write_checkpoint(
        tmp_path / "checkpoint",
        done=["case-a"],
        failed=[],
        attempted=["case-a"],
    )
    transaction_id = "a" * 32
    if operation == "close":
        document = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        document[collect_mod._SIDECAR_TRANSACTION_FIELD] = transaction_id
        checkpoint_path.write_text(json.dumps(document), encoding="utf-8")
    original = checkpoint_path.read_bytes()
    expected = collect_mod._checkpoint_snapshot(
        checkpoint_path,
        None,
        context_path=checkpoint_path,
    ).attest(checkpoint_path)
    victim = checkpoint_path.with_name("victim.json")
    victim.write_bytes(b"victim")
    displaced = checkpoint_path.with_name("displaced-checkpoint.json")
    previous = collect_mod._atomic_write_previous_path(checkpoint_path)
    reservation = collect_mod.atomic_write_reservation_path(checkpoint_path)
    real_rename_noreplace = collect_mod._rename_noreplace_at
    real_open = os.open
    swapped = False
    fifo_open_attempts = 0

    def reject_blocking_fifo_open(path, flags, *args, **kwargs):
        nonlocal fifo_open_attempts
        if path == previous.name:
            if not flags & os.O_NONBLOCK:
                raise AssertionError("checkpoint recovery attempted a blocking FIFO open")
            fifo_open_attempts += 1
        return real_open(path, flags, *args, **kwargs)

    if replacement_kind == "fifo":
        monkeypatch.setattr(collect_mod.os, "open", reject_blocking_fifo_open)

    def swap_inside_first_rotation(source, target, directory_fd):
        nonlocal swapped
        if not swapped and source == checkpoint_path.name and target == previous.name:
            swapped = True
            os.rename(
                source,
                displaced.name,
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
            )
            if replacement_kind == "symlink":
                os.symlink(victim.name, source, dir_fd=directory_fd)
            elif replacement_kind == "directory":
                os.mkdir(source, dir_fd=directory_fd)
                marker_fd = real_open(
                    f"{source}/marker",
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    0o600,
                    dir_fd=directory_fd,
                )
                with os.fdopen(marker_fd, "wb") as marker:
                    marker.write(b"directory victim")
                    marker.flush()
                    os.fsync(marker.fileno())
            else:
                os.mkfifo(source, 0o600, dir_fd=directory_fd)
            os.fsync(directory_fd)
        return real_rename_noreplace(source, target, directory_fd)

    monkeypatch.setattr(collect_mod, "_rename_noreplace_at", swap_inside_first_rotation)

    with pytest.raises(RuntimeError, match=r"checkpoint|artifact|changed"):
        if operation == "tag":
            collect_mod._tag_checkpoint_sidecar_transaction(
                checkpoint_path,
                {"case-a"},
                transaction_id,
                expected_attestation=expected,
            )
        else:
            collect_mod._close_checkpoint_attempts(
                checkpoint_path,
                {"case-a"},
                transaction_id=transaction_id,
                expected_attestation=expected,
            )

    assert swapped
    if replacement_kind == "symlink":
        assert checkpoint_path.is_symlink()
        assert os.readlink(checkpoint_path) == victim.name
        assert victim.read_bytes() == b"victim"
    elif replacement_kind == "directory":
        assert checkpoint_path.is_dir()
        assert (checkpoint_path / "marker").read_bytes() == b"directory victim"
    else:
        assert stat.S_ISFIFO(checkpoint_path.lstat().st_mode)
        assert fifo_open_attempts == 1
    assert displaced.read_bytes() == original
    assert reservation.is_file()
    assert not previous.exists() and not previous.is_symlink()


def test_checkpoint_mutation_preserves_unknown_previous_when_canonical_is_occupied(tmp_path, monkeypatch):
    checkpoint_path = _write_checkpoint(
        tmp_path / "checkpoint",
        done=["case-a"],
        failed=[],
        attempted=["case-a"],
    )
    original = checkpoint_path.read_bytes()
    expected = collect_mod._checkpoint_snapshot(
        checkpoint_path,
        None,
        context_path=checkpoint_path,
    ).attest(checkpoint_path)
    victim = checkpoint_path.with_name("victim.json")
    victim.write_bytes(b"victim")
    displaced = checkpoint_path.with_name("displaced-checkpoint.json")
    previous = collect_mod._atomic_write_previous_path(checkpoint_path)
    reservation = collect_mod.atomic_write_reservation_path(checkpoint_path)
    competitor = b"competitor"
    real_rename_noreplace = collect_mod._rename_noreplace_at
    swapped = False

    def occupy_canonical_after_first_rotation(source, target, directory_fd):
        nonlocal swapped
        if swapped or source != checkpoint_path.name or target != previous.name:
            return real_rename_noreplace(source, target, directory_fd)
        swapped = True
        os.rename(
            source,
            displaced.name,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
        )
        os.symlink(victim.name, source, dir_fd=directory_fd)
        real_rename_noreplace(source, target, directory_fd)
        competitor_fd = os.open(
            source,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
            dir_fd=directory_fd,
        )
        with os.fdopen(competitor_fd, "wb") as competitor_file:
            competitor_file.write(competitor)
            competitor_file.flush()
            os.fsync(competitor_file.fileno())
        os.fsync(directory_fd)

    monkeypatch.setattr(collect_mod, "_rename_noreplace_at", occupy_canonical_after_first_rotation)

    with pytest.raises(RuntimeError, match=r"checkpoint|artifact|changed"):
        collect_mod._tag_checkpoint_sidecar_transaction(
            checkpoint_path,
            {"case-a"},
            "a" * 32,
            expected_attestation=expected,
        )

    assert swapped
    assert checkpoint_path.read_bytes() == competitor
    assert previous.is_symlink()
    assert os.readlink(previous) == victim.name
    assert victim.read_bytes() == b"victim"
    assert displaced.read_bytes() == original
    assert reservation.is_file()


@pytest.mark.parametrize("operation", ["tag", "close"])
def test_checkpoint_mutation_retry_rejects_restored_fifo_without_blocking(tmp_path, operation):
    checkpoint_path = _write_checkpoint(
        tmp_path / "checkpoint",
        done=["case-a"],
        failed=[],
        attempted=["case-a"],
    )
    original = checkpoint_path.read_bytes()
    displaced = checkpoint_path.with_name("displaced-checkpoint.json")
    displaced.write_bytes(original)
    reservation = collect_mod.atomic_write_reservation_path(checkpoint_path)
    reservation_bytes = b"future checkpoint evidence"
    reservation.write_bytes(reservation_bytes)
    victim = checkpoint_path.with_name("victim.json")
    victim.write_bytes(b"victim")
    checkpoint_path.unlink()
    os.mkfifo(checkpoint_path, 0o600)

    _assert_checkpoint_mutation_fails_without_blocking(checkpoint_path, operation)

    assert stat.S_ISFIFO(checkpoint_path.lstat().st_mode)
    assert victim.read_bytes() == b"victim"
    assert displaced.read_bytes() == original
    assert reservation.read_bytes() == reservation_bytes


@pytest.mark.parametrize("operation", ["tag", "close"])
def test_checkpoint_mutation_rejects_fifo_swapped_after_publication_without_blocking(tmp_path, operation):
    checkpoint_path = _write_checkpoint(
        tmp_path / "checkpoint",
        done=["case-a"],
        failed=[],
        attempted=["case-a"],
    )
    transaction_id = "a" * 32
    if operation == "close":
        document = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        document[collect_mod._SIDECAR_TRANSACTION_FIELD] = transaction_id
        checkpoint_path.write_text(json.dumps(document), encoding="utf-8")
    original = checkpoint_path.read_bytes()
    previous = collect_mod._atomic_write_previous_path(checkpoint_path)
    reservation_evidence = checkpoint_path.with_name("reservation-evidence.json")
    victim = checkpoint_path.with_name("victim.json")
    victim.write_bytes(b"victim")

    _assert_subprocess_fails_without_blocking(
        _swap_fifo_after_checkpoint_publication_and_exit,
        (str(checkpoint_path), operation, transaction_id, str(reservation_evidence)),
        f"checkpoint {operation} publication attestation",
    )

    assert stat.S_ISFIFO(checkpoint_path.lstat().st_mode)
    assert victim.read_bytes() == b"victim"
    assert previous.read_bytes() == original
    published_document = json.loads(reservation_evidence.read_text(encoding="utf-8"))
    if operation == "tag":
        assert published_document[collect_mod._SIDECAR_TRANSACTION_FIELD] == transaction_id
        assert published_document["attempted"] == ["case-a"]
    else:
        assert collect_mod._SIDECAR_TRANSACTION_FIELD not in published_document
        assert published_document["attempted"] == []


@pytest.mark.parametrize("operation", ["tag", "close"])
def test_checkpoint_mutation_revalidates_journal_through_locked_output_root(tmp_path, monkeypatch, operation):
    output_root = tmp_path / "out"
    output_root.mkdir()
    journal_path = output_root / collect_mod._SIDECAR_TRANSACTION_FILENAME
    journal_path.write_bytes(b"journal")
    checkpoint_path = _write_checkpoint(
        tmp_path / "checkpoint",
        done=["case-a"],
        failed=[],
        attempted=["case-a"],
    )
    transaction_id = "a" * 32
    if operation == "close":
        document = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        document[collect_mod._SIDECAR_TRANSACTION_FIELD] = transaction_id
        checkpoint_path.write_text(json.dumps(document), encoding="utf-8")

    with helper_mod.perf_finalization_lifecycle(output_root) as locked_root:
        journal_attestation = collect_mod._validated_regular_file(
            journal_path,
            None,
            context_path=journal_path,
            kind="sidecar transaction journal",
            locked_output_root=locked_root,
        ).attest(journal_path)
        real_revalidate = collect_mod._revalidate_journal_attestation
        revalidations = 0

        def require_locked_root(attestation, candidate_root=None):
            nonlocal revalidations
            assert candidate_root is locked_root
            revalidations += 1
            return real_revalidate(attestation, candidate_root)

        monkeypatch.setattr(collect_mod, "_revalidate_journal_attestation", require_locked_root)
        if operation == "tag":
            collect_mod._tag_checkpoint_sidecar_transaction(
                checkpoint_path,
                {"case-a"},
                transaction_id,
                journal_attestation=journal_attestation,
                locked_output_root=locked_root,
            )
        else:
            collect_mod._close_checkpoint_attempts(
                checkpoint_path,
                {"case-a"},
                transaction_id=transaction_id,
                journal_attestation=journal_attestation,
                locked_output_root=locked_root,
            )

    assert revalidations == 2


def test_repeated_checkpoint_crashes_reuse_one_deterministic_reservation(tmp_path):
    checkpoint_path = tmp_path / "checkpoint.json"
    checkpoint_path.write_text('{"attempted": ["old"]}', encoding="utf-8")
    reservation = tmp_path / ".checkpoint.json.tmp"
    reservation_identity = None

    for _attempt in range(3):
        _run_hard_exit(
            _crash_checkpoint_before_publication,
            (str(checkpoint_path), 100),
            100,
            "collector checkpoint pre-publication crash",
        )
        assert json.loads(checkpoint_path.read_text(encoding="utf-8")) == {"attempted": ["old"]}
        assert reservation.is_file()
        assert reservation.stat().st_nlink == 1
        current_identity = (reservation.stat().st_dev, reservation.stat().st_ino)
        if reservation_identity is None:
            reservation_identity = current_identity
        assert current_identity == reservation_identity
        assert {path.name for path in tmp_path.glob(".checkpoint.json*.tmp")} == {reservation.name}

    collect_mod._atomic_write_json(checkpoint_path, {"attempted": ["new"]})

    assert json.loads(checkpoint_path.read_text(encoding="utf-8")) == {"attempted": ["new"]}
    assert (checkpoint_path.stat().st_dev, checkpoint_path.stat().st_ino) == reservation_identity
    assert reservation.is_file()
    assert reservation.stat().st_nlink == 1


def test_resume_recovers_old_ledgers_after_checkpoint_canonical_rotation_crash(tmp_path):
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_path = _write_checkpoint(
        checkpoint_dir,
        done=["case-a"],
        failed=["case-b"],
        attempted=["case-pending"],
    )

    _run_hard_exit(
        _crash_checkpoint_after_canonical_rotation,
        (str(checkpoint_path), 101),
        101,
        "collector checkpoint canonical-rotation crash",
    )

    assert not checkpoint_path.exists()
    assert (checkpoint_path.parent / f".{checkpoint_path.name}.tmp").is_file()
    assert (checkpoint_path.parent / f".{checkpoint_path.name}.tmp.previous").is_file()

    checkpoint = collect_mod.ResumeCheckpoint(
        backend=BACKEND,
        module_name=FULL_NAME,
        run_func_name=RUN_FUNC,
        checkpoint_dir=str(checkpoint_dir),
        framework_version="0.5.14",
        sm_version=100,
    )
    checkpoint.load_existing()

    assert checkpoint._done == {"case-a"}
    assert checkpoint._failed == {"case-b"}
    assert checkpoint._attempted == {"case-pending"}
    assert checkpoint_path.is_file()


def test_selected_producer_recovers_checkpoint_after_canonical_rotation_crash(tmp_path):
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_path = _write_checkpoint(
        checkpoint_dir,
        done=["case-a"],
        failed=["case-b"],
        attempted=["case-pending"],
    )
    _run_hard_exit(
        _crash_checkpoint_after_canonical_rotation,
        (str(checkpoint_path), 102),
        102,
        "selected producer checkpoint canonical-rotation crash",
    )

    tracker = collect_mod._load_selected_producer_checkpoint(
        _collections()[0],
        _provenance_ctx(_collections()),
        backend=BACKEND,
        checkpoint_dir=str(checkpoint_dir),
        sm_version=100,
        staging_path=tmp_path / "gemm_perf.txt",
        required=True,
        context="test finalization",
    )

    assert tracker is not None
    assert tracker._done == {"case-a"}
    assert tracker._failed == {"case-b"}
    assert tracker._attempted == {"case-pending"}


@pytest.mark.parametrize(("operation", "exit_code"), [("tag", 103), ("close", 104)])
def test_transaction_recovery_normalizes_checkpoint_rotation_crash(operation, exit_code, tmp_path):
    output_root = tmp_path / "out"
    output_root.mkdir()
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_path = _write_checkpoint(
        checkpoint_dir,
        done=["case-a"],
        failed=["case-b"],
        attempted=["case-a", "case-b"],
    )
    staging_path = output_root / "gemm_perf.txt"
    staging_path.write_text("op,shape,latency\nmatmul,new,1.0\n", encoding="utf-8")

    _run_hard_exit(
        _crash_during_transaction_checkpoint_rotation,
        (str(output_root), str(checkpoint_dir), operation, exit_code),
        exit_code,
        f"collector checkpoint {operation} canonical-rotation crash",
    )

    reservation = checkpoint_path.with_name(f".{checkpoint_path.name}.tmp")
    previous = checkpoint_path.with_name(f".{checkpoint_path.name}.tmp.previous")
    assert not checkpoint_path.exists()
    assert reservation.is_file()
    assert previous.is_file()
    assert (output_root / collect_mod._SIDECAR_TRANSACTION_FILENAME).is_file()

    assert (
        collect_mod._recover_collector_provenance_transaction(
            output_root,
            backend=BACKEND,
            checkpoint_dir=str(checkpoint_dir),
        )
        == output_root / "collection_meta.yaml"
    )

    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert checkpoint["done"] == ["case-a"]
    assert checkpoint["failed"] == ["case-b"]
    assert checkpoint["attempted"] == []
    assert collect_mod._SIDECAR_TRANSACTION_FIELD not in checkpoint
    assert not (output_root / collect_mod._SIDECAR_TRANSACTION_FILENAME).exists()


def test_live_checkpoint_writer_serializes_reservation_reuse(tmp_path):
    checkpoint_path = tmp_path / "checkpoint.json"
    checkpoint_path.write_text('{"attempted": ["old"]}', encoding="utf-8")
    context = mp.get_context("spawn")
    entered = context.Event()
    release = context.Event()
    waiter_started = context.Event()
    waiter_finished = context.Event()
    first = context.Process(target=_pause_checkpoint_writer, args=(str(checkpoint_path), entered, release))
    second = context.Process(
        target=_write_checkpoint_and_signal,
        args=(str(checkpoint_path), waiter_started, waiter_finished),
    )

    first.start()
    assert entered.wait(timeout=20)
    second.start()
    assert waiter_started.wait(timeout=20)
    assert not waiter_finished.wait(timeout=0.3)
    release.set()
    first.join(timeout=20)
    second.join(timeout=20)

    assert first.exitcode == 0
    assert second.exitcode == 0
    assert waiter_finished.is_set()
    assert json.loads(checkpoint_path.read_text(encoding="utf-8")) == {"attempted": ["second"]}
    assert (tmp_path / ".checkpoint.json.tmp").is_file()


def test_checkpoint_reader_waits_for_canonical_rotation_to_finish(tmp_path):
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_path = _write_checkpoint(
        checkpoint_dir,
        done=["case-a"],
        failed=["case-b"],
        attempted=["case-pending"],
    )
    context = mp.get_context("spawn")
    entered = context.Event()
    release = context.Event()
    reader_started = context.Event()
    reader_finished = context.Event()
    result_queue = context.Queue()
    writer = context.Process(
        target=_pause_valid_checkpoint_writer_after_rotation,
        args=(str(checkpoint_path), entered, release),
    )
    reader = context.Process(
        target=_load_checkpoint_and_signal,
        args=(str(checkpoint_dir), reader_started, reader_finished, result_queue),
    )

    writer.start()
    assert entered.wait(timeout=20)
    assert not checkpoint_path.exists()
    reader.start()
    assert reader_started.wait(timeout=20)
    assert not reader_finished.wait(timeout=0.3)
    release.set()
    writer.join(timeout=20)
    reader.join(timeout=20)

    assert writer.exitcode == 0
    assert reader.exitcode == 0
    assert reader_finished.is_set()
    assert result_queue.get(timeout=2) == ({"replacement"}, set(), {"replacement"})


@pytest.mark.parametrize("operation", ["write", "read"])
def test_checkpoint_operation_fails_closed_if_locked_parent_is_replaced(operation, tmp_path):
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_path = _write_checkpoint(
        checkpoint_dir,
        done=["initial"],
        failed=[],
        attempted=["initial"],
    )
    initial = checkpoint_path.read_bytes()
    replacement_document = json.loads(initial)
    replacement_document["done"] = ["replacement"]
    replacement_document["attempted"] = ["replacement"]
    displaced_document = json.loads(initial)
    displaced_document["done"] = ["displaced-writer"]
    displaced_document["attempted"] = ["displaced-writer"]

    context = mp.get_context("spawn")
    entered = context.Event()
    release = context.Event()
    replacement_finished = context.Event()
    displaced = checkpoint_path.parent.with_name(f"{checkpoint_path.parent.name}.displaced")
    first = context.Process(
        target=_pause_checkpoint_after_parent_lock,
        args=(
            str(checkpoint_dir),
            str(checkpoint_path),
            displaced_document if operation == "write" else None,
            entered,
            release,
        ),
    )
    second = context.Process(
        target=_write_checkpoint_document_and_signal,
        args=(str(checkpoint_path), replacement_document, replacement_finished),
    )

    first.start()
    assert entered.wait(timeout=20)
    checkpoint_path.parent.rename(displaced)
    checkpoint_path.parent.mkdir()
    second.start()
    assert replacement_finished.wait(timeout=20)
    second.join(timeout=20)
    assert second.exitcode == 0

    release.set()
    first.join(timeout=20)

    assert first.exitcode not in (None, 0)
    assert json.loads(checkpoint_path.read_text(encoding="utf-8")) == replacement_document
    expected_displaced = displaced_document if operation == "write" else json.loads(initial)
    assert json.loads((displaced / checkpoint_path.name).read_text(encoding="utf-8")) == expected_displaced


@pytest.mark.parametrize("operation", ["tag", "close"])
def test_checkpoint_mutation_revalidates_after_parent_lock(operation, tmp_path, monkeypatch):
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_path = _write_checkpoint(
        checkpoint_dir,
        done=["case-a"],
        failed=[],
        attempted=["case-a"],
    )
    expected = _file_attestation(checkpoint_path)
    replacement_document = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    replacement_document["done"] = ["foreign"]
    replacement = json.dumps(replacement_document).encode()
    real_atomic_write_json = collect_mod._atomic_write_json

    def replace_after_read_then_write(path, data, **kwargs):
        _replace_with_new_inode(Path(path), replacement)
        return real_atomic_write_json(path, data, **kwargs)

    monkeypatch.setattr(collect_mod, "_atomic_write_json", replace_after_read_then_write)

    mutation = (
        collect_mod._tag_checkpoint_sidecar_transaction
        if operation == "tag"
        else collect_mod._close_checkpoint_attempts
    )
    kwargs = {"transaction_id": "b" * 32} if operation == "tag" else {}
    with pytest.raises(RuntimeError, match=r"checkpoint|changed"):
        mutation(
            checkpoint_path,
            {"case-a"},
            expected_attestation=expected,
            **kwargs,
        )

    assert checkpoint_path.read_bytes() == replacement


def test_finalization_rejects_equal_output_and_checkpoint_roots(tmp_path):
    output_root = tmp_path / BACKEND
    output_root.mkdir()
    staging_path = output_root / "gemm_perf.txt"
    staging_path.write_text("op,latency\nmatmul,1.0\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match=r"output and checkpoint roots must be different"):
        collect_mod._finalize_collector_outputs_transaction(
            output_root,
            [staging_path],
            _provenance_ctx(_collections()),
            [],
            backend=BACKEND,
            checkpoint_dir=str(tmp_path),
            sm_version=100,
        )


def test_displaced_collector_finalizer_cannot_modify_recreated_output_root(tmp_path):
    output_root, checkpoint_dir, _checkpoint, _staging, _parquet = _single_table_finalization(tmp_path)
    displaced_root = tmp_path / "displaced-out"
    process_context = mp.get_context("spawn")
    ready = process_context.Event()
    release = process_context.Event()
    process = process_context.Process(
        target=_pause_displaced_collector_finalization,
        args=(str(output_root), str(checkpoint_dir), ready, release),
    )
    process.start()
    assert ready.wait(timeout=20), "displaced finalizer did not acquire its output-root lock"

    output_root.rename(displaced_root)

    def directory_state(directory):
        return {
            path.name: (
                path.read_bytes(),
                stat.S_IMODE(path.stat().st_mode),
                path.stat().st_nlink,
            )
            for path in directory.iterdir()
            if path.is_file() and not path.is_symlink()
        }

    displaced_before = directory_state(displaced_root)
    output_root.mkdir()
    replacement_checkpoint_dir = tmp_path / "replacement-checkpoint"
    _write_checkpoint(
        replacement_checkpoint_dir,
        done=["case-a"],
        failed=[],
        attempted=["case-a"],
    )
    replacement_staging = output_root / "gemm_perf.txt"
    replacement_staging.write_text("op,shape,latency\nmatmul,replacement,2.0\n", encoding="utf-8")
    _finalize_single_table(output_root, replacement_checkpoint_dir, replacement_staging)
    replacement_after_commit = directory_state(output_root)

    release.set()
    process.join(timeout=20)
    if process.is_alive():
        process.terminate()
        process.join()
        pytest.fail("displaced collector finalizer did not exit")

    assert process.exitcode != 0
    assert directory_state(displaced_root) == displaced_before
    assert directory_state(output_root) == replacement_after_commit


def test_displaced_collector_recovery_cannot_modify_recreated_output_root(tmp_path):
    output_root = tmp_path / "out"
    output_root.mkdir()
    reservation = collect_mod.atomic_write_reservation_path(output_root / collect_mod._PERF_TRANSACTION_FILENAME)
    reservation.write_bytes(b"interrupted old-root journal")
    checkpoint_dir = tmp_path / "checkpoint"
    displaced_root = tmp_path / "displaced-out"
    process_context = mp.get_context("spawn")
    ready = process_context.Event()
    release = process_context.Event()
    process = process_context.Process(
        target=_pause_displaced_collector_recovery,
        args=(str(output_root), str(checkpoint_dir), ready, release),
    )
    process.start()
    assert ready.wait(timeout=20), "displaced recovery did not acquire its output-root lock"

    output_root.rename(displaced_root)
    displaced_before = {path.name: path.read_bytes() for path in displaced_root.iterdir() if path.is_file()}
    output_root.mkdir()
    replacement_checkpoint_dir = tmp_path / "replacement-checkpoint"
    _write_checkpoint(
        replacement_checkpoint_dir,
        done=["case-a"],
        failed=[],
        attempted=["case-a"],
    )
    replacement_staging = output_root / "gemm_perf.txt"
    replacement_staging.write_text("op,shape,latency\nmatmul,replacement,2.0\n", encoding="utf-8")
    _finalize_single_table(output_root, replacement_checkpoint_dir, replacement_staging)
    replacement_after_commit = {path.name: path.read_bytes() for path in output_root.iterdir() if path.is_file()}

    release.set()
    process.join(timeout=20)
    if process.is_alive():
        process.terminate()
        process.join()
        pytest.fail("displaced collector recovery did not exit")

    assert process.exitcode != 0
    assert {path.name: path.read_bytes() for path in displaced_root.iterdir() if path.is_file()} == displaced_before
    assert {
        path.name: path.read_bytes() for path in output_root.iterdir() if path.is_file()
    } == replacement_after_commit


def test_atomic_exclusive_write_rejects_regular_temp_path_swap(tmp_path, monkeypatch):
    destination = tmp_path / "collection_meta.yaml"
    competitor = b"competitor"
    swapped_temp_path = None
    real_rename_noreplace = helper_mod._rename_noreplace_at

    def swap_temp_then_publish(source, target, directory_fd):
        nonlocal swapped_temp_path
        swapped_temp_path = tmp_path / source
        replacement = tmp_path / "replacement"
        replacement.write_bytes(competitor)
        replacement.replace(swapped_temp_path)
        return real_rename_noreplace(source, target, directory_fd)

    monkeypatch.setattr(helper_mod, "_rename_noreplace_at", swap_temp_then_publish)

    with pytest.raises(RuntimeError, match=r"temporary|publication|changed"):
        collect_mod._atomic_write_bytes(destination, b"owned")

    assert swapped_temp_path is not None
    assert not swapped_temp_path.exists()
    assert destination.read_bytes() == competitor


def test_atomic_exclusive_write_preserves_destination_replacement_after_mismatch(tmp_path, monkeypatch):
    destination = tmp_path / "collection_meta.yaml"
    observed_link = tmp_path / "observed-link"
    second_competitor = tmp_path / "second-competitor"
    second_competitor.write_bytes(b"second competitor")
    swapped_temp_path = None
    replacement_injected = False
    real_rename_noreplace = helper_mod._rename_noreplace_at
    real_digest_open_fd = collect_mod._digest_open_fd
    digest_calls = 0

    def swap_temp_then_publish(source, target, directory_fd):
        nonlocal swapped_temp_path
        swapped_temp_path = tmp_path / source
        replacement = tmp_path / "first-competitor"
        replacement.write_bytes(b"first competitor")
        replacement.replace(swapped_temp_path)
        return real_rename_noreplace(source, target, directory_fd)

    def replace_before_published_path_digest(file_descriptor):
        nonlocal digest_calls, replacement_injected
        result = real_digest_open_fd(file_descriptor)
        digest_calls += 1
        if digest_calls == 2 and not replacement_injected:
            replacement_injected = True
            destination.replace(observed_link)
            second_competitor.replace(destination)
        return result

    monkeypatch.setattr(helper_mod, "_rename_noreplace_at", swap_temp_then_publish)
    monkeypatch.setattr(collect_mod, "_digest_open_fd", replace_before_published_path_digest)

    with pytest.raises(RuntimeError, match=r"publication|changed"):
        collect_mod._atomic_write_bytes(destination, b"owned")

    assert replacement_injected
    assert destination.read_bytes() == b"second competitor"
    assert observed_link.read_bytes() == b"first competitor"
    assert swapped_temp_path is not None
    assert not swapped_temp_path.exists()


def test_validated_regular_file_rejects_chmod_during_digest(tmp_path, monkeypatch):
    artifact = tmp_path / "artifact"
    artifact.write_bytes(b"owned")
    artifact.chmod(0o600)
    real_sha256 = hashlib.sha256
    changed = False

    class ChmodDigest:
        def __init__(self):
            self._digest = real_sha256()

        def update(self, chunk):
            nonlocal changed
            self._digest.update(chunk)
            if not changed:
                changed = True
                artifact.chmod(0o400)

        def hexdigest(self):
            return self._digest.hexdigest()

    monkeypatch.setattr(collect_mod.hashlib, "sha256", ChmodDigest)

    with pytest.raises(RuntimeError, match="changed"):
        collect_mod._validated_regular_file(
            artifact,
            None,
            context_path=artifact,
            kind="test artifact",
        )

    assert changed


def test_atomic_exclusive_write_rejects_symlink_temp_path_swap(tmp_path, monkeypatch):
    destination = tmp_path / "collection_meta.yaml"
    competitor = tmp_path / "competitor"
    competitor.write_bytes(b"competitor")
    swapped_temp_path = None
    real_rename_noreplace = helper_mod._rename_noreplace_at

    def swap_temp_then_publish(source, target, directory_fd):
        nonlocal swapped_temp_path
        swapped_temp_path = tmp_path / source
        swapped_temp_path.unlink()
        try:
            swapped_temp_path.symlink_to(competitor)
        except OSError as error:
            pytest.skip(f"symlinks unavailable: {error}")
        return real_rename_noreplace(source, target, directory_fd)

    monkeypatch.setattr(helper_mod, "_rename_noreplace_at", swap_temp_then_publish)

    with pytest.raises(RuntimeError, match=r"temporary|publication|changed"):
        collect_mod._atomic_write_bytes(destination, b"owned")

    assert swapped_temp_path is not None
    assert not swapped_temp_path.exists()
    assert competitor.read_bytes() == b"competitor"
    assert destination.is_symlink()
    assert destination.samefile(competitor)


def test_atomic_exclusive_write_publishes_owned_bytes_and_mode(tmp_path, monkeypatch):
    destination = tmp_path / "collection_meta.yaml"
    reservation = tmp_path / ".collection_meta.yaml.tmp"
    published_sources = []
    events = []
    real_rename_noreplace = helper_mod._rename_noreplace_at
    real_fsync = collect_mod.os.fsync

    def record_publication(source, target, directory_fd):
        published_sources.append(tmp_path / source)
        events.append(("rename", reservation.exists(), destination.exists()))
        return real_rename_noreplace(source, target, directory_fd)

    def record_fsync(file_descriptor):
        kind = "file-fsync" if stat.S_ISREG(os.fstat(file_descriptor).st_mode) else "directory-fsync"
        events.append((kind, reservation.exists(), destination.exists()))
        return real_fsync(file_descriptor)

    monkeypatch.setattr(helper_mod, "_rename_noreplace_at", record_publication)
    monkeypatch.setattr(collect_mod.os, "fsync", record_fsync)

    collect_mod._atomic_write_bytes(destination, b"owned", mode=0o640)

    assert destination.read_bytes() == b"owned"
    assert stat.S_IMODE(destination.stat().st_mode) == 0o640
    assert published_sources == [reservation]
    assert events == [
        ("directory-fsync", True, False),
        ("file-fsync", True, False),
        ("rename", True, False),
        ("directory-fsync", False, True),
    ]
    assert not reservation.exists()


def test_atomic_exclusive_write_publishes_without_a_temporary_alias(tmp_path, monkeypatch):
    destination = tmp_path / "collection_meta.yaml"

    def reject_hardlink(*_args, **_kwargs):
        raise AssertionError("no-replace publication must not create a hardlink alias")

    monkeypatch.setattr(collect_mod.os, "link", reject_hardlink)

    collect_mod._atomic_write_bytes(destination, b"owned", mode=0o640)

    assert destination.read_bytes() == b"owned"
    assert destination.stat().st_nlink == 1
    assert not (tmp_path / ".collection_meta.yaml.tmp").exists()


def test_atomic_exclusive_write_reuses_one_retained_reservation(tmp_path):
    destination = tmp_path / "collection_meta.yaml"
    reservation = tmp_path / ".collection_meta.yaml.tmp"
    reservation.write_bytes(b"interrupted")
    retained_identity = (reservation.stat().st_dev, reservation.stat().st_ino)

    collect_mod._atomic_write_bytes(destination, b"published", mode=0o640)

    assert destination.read_bytes() == b"published"
    assert (destination.stat().st_dev, destination.stat().st_ino) == retained_identity
    assert stat.S_IMODE(destination.stat().st_mode) == 0o640
    assert destination.stat().st_nlink == 1
    assert not reservation.exists()


def test_atomic_exclusive_write_preserves_foreign_hardlink_before_cleanup(tmp_path, monkeypatch):
    destination = tmp_path / "collection_meta.yaml"
    reservation = tmp_path / ".collection_meta.yaml.tmp"
    foreign_link = tmp_path / "foreign-link"
    real_rename_noreplace = helper_mod._rename_noreplace_at

    def add_foreign_link_then_publish(source, target, directory_fd):
        os.link(tmp_path / source, foreign_link, follow_symlinks=False)
        return real_rename_noreplace(source, target, directory_fd)

    monkeypatch.setattr(helper_mod, "_rename_noreplace_at", add_foreign_link_then_publish)

    with pytest.raises(RuntimeError, match=r"publication|temporary|changed"):
        collect_mod._atomic_write_bytes(destination, b"owned", mode=0o640)

    assert not reservation.exists()
    assert destination.samefile(foreign_link)
    assert destination.stat().st_nlink == 2


def test_atomic_exclusive_write_rejects_same_inode_bytes_and_mode_race(tmp_path, monkeypatch):
    destination = tmp_path / "collection_meta.yaml"
    owned = b"owned"
    competitor = b"raced"
    published_temp = None
    real_rename_noreplace = helper_mod._rename_noreplace_at

    def publish_then_mutate(source, target, directory_fd):
        nonlocal published_temp
        published_temp = tmp_path / source
        result = real_rename_noreplace(source, target, directory_fd)
        with (tmp_path / target).open("r+b") as raced_file:
            raced_file.write(competitor)
            raced_file.flush()
            os.fchmod(raced_file.fileno(), 0o777)
            os.fsync(raced_file.fileno())
        return result

    monkeypatch.setattr(helper_mod, "_rename_noreplace_at", publish_then_mutate)

    with pytest.raises(RuntimeError, match=r"publication|changed"):
        collect_mod._atomic_write_bytes(destination, owned, mode=0o640)

    assert published_temp is not None
    assert not published_temp.exists()
    assert destination.read_bytes() == competitor
    assert stat.S_IMODE(destination.stat().st_mode) == 0o777


def test_recovery_rejects_retained_atomic_publication_mismatch_without_mutation(tmp_path):
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
        pending_document=True,
    )
    meta_path = output_root / "collection_meta.yaml"
    meta_path.write_bytes(b"retained mismatched publication")
    tracked_paths = (
        meta_path,
        output_root / collect_mod._SIDECAR_STAGING_FILENAME,
        output_root / collect_mod._SIDECAR_TRANSACTION_FILENAME,
        staging_path,
        checkpoint_path,
    )
    before = {path: (path.lstat().st_dev, path.lstat().st_ino, path.read_bytes()) for path in tracked_paths}
    entries_before = {path.relative_to(tmp_path) for path in tmp_path.rglob("*")}

    with pytest.raises(RuntimeError, match=r"sidecar target changed"):
        collect_mod._recover_collector_provenance_transaction(
            output_root,
            backend=BACKEND,
            checkpoint_dir=str(checkpoint_dir),
        )

    assert {path: (path.lstat().st_dev, path.lstat().st_ino, path.read_bytes()) for path in tracked_paths} == before
    assert {path.relative_to(tmp_path) for path in tmp_path.rglob("*")} == entries_before


def test_cleanup_fails_closed_on_legacy_source_claim_hardlink(tmp_path):
    staging_path = tmp_path / "gemm_perf.txt"
    staging_path.write_bytes(b"owned staging")
    attestation = _file_attestation(staging_path)
    transaction_id = "a" * 32
    claim_path = collect_mod._transaction_claim_path(staging_path, transaction_id)
    os.link(staging_path, claim_path)

    with pytest.raises(RuntimeError, match=r"legacy|hardlink|claim"):
        collect_mod._cleanup_transaction_files(
            [attestation],
            context_path=tmp_path / collect_mod._SIDECAR_TRANSACTION_FILENAME,
            transaction_id=transaction_id,
        )

    assert staging_path.samefile(claim_path)
    assert staging_path.stat().st_nlink == 2
    assert claim_path.stat().st_nlink == 2
    assert not collect_mod.collector_retained_path(staging_path).exists()


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


def test_claim_never_restores_foreign_quarantine_replacement_to_owned_path(tmp_path, monkeypatch):
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
    assert not owned_path.exists()
    assert claimed_path.read_bytes() == malicious
    assert outside_owned.read_bytes() == b"owned staging"


def test_claim_never_overwrites_racing_deterministic_claim(tmp_path, monkeypatch):
    owned_path = tmp_path / "gemm_perf.txt"
    owned_path.write_bytes(b"owned staging")
    attestation = _file_attestation(owned_path)
    transaction_id = "1" * 32
    claim_path = collect_mod._transaction_claim_path(owned_path, transaction_id)
    competitor = b"competing claim"
    real_rename_noreplace = collect_mod._rename_noreplace

    def compete_before_rename(source, target):
        if Path(target) == claim_path:
            claim_path.write_bytes(competitor)
        return real_rename_noreplace(source, target)

    monkeypatch.setattr(collect_mod, "_rename_noreplace", compete_before_rename)

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
    real_render = provenance.render_collection_meta

    def render_then_replace_target(*args, **kwargs):
        rendered = real_render(*args, **kwargs)
        meta_path.replace(outside_meta)
        try:
            meta_path.symlink_to(outside_meta)
        except OSError as error:
            pytest.skip(f"symlinks unavailable: {error}")
        return rendered

    monkeypatch.setattr(provenance, "render_collection_meta", render_then_replace_target)

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
            real_rename_noreplace = helper_mod._rename_noreplace_at

            def rename_then_crash(source, target, directory_fd):
                real_rename_noreplace(source, target, directory_fd)
                if str(target).endswith(".transaction-claim"):
                    raise SimulatedCrash

            fault.setattr(helper_mod, "_rename_noreplace_at", rename_then_crash)
        else:
            real_atomic_write = collect_mod._atomic_write_bytes

            def crash_before_link(path, *args, **kwargs):
                if Path(path) == meta_path:
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
    assert not pending_path.exists()
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


def test_recovery_retains_deterministic_staging_claim_after_cleanup_crash(tmp_path, monkeypatch):
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
    real_retain = collect_mod._retain_claimed_transaction_files

    def crash_before_staging_claim_retention(claimed_files, *args, **kwargs):
        if any(claimed.original.path == staging_path for claimed in claimed_files):
            raise SimulatedCrash
        return real_retain(claimed_files, *args, **kwargs)

    with monkeypatch.context() as fault:
        fault.setattr(collect_mod, "_retain_claimed_transaction_files", crash_before_staging_claim_retention)
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


def test_repeated_committed_claim_recovery_uses_fixed_retained_slots(tmp_path, monkeypatch):
    class SimulatedCrash(BaseException):
        pass

    output_root = tmp_path / "out"
    output_root.mkdir()
    parquet_path = output_root / "gemm_perf.parquet"
    _write_parquet(parquet_path, [{"op": "matmul", "latency": 1.0}])
    staging_path = parquet_path.with_suffix(".txt")
    checkpoint_dir = tmp_path / "checkpoint"
    pending_path = output_root / collect_mod._SIDECAR_STAGING_FILENAME
    meta_path = output_root / "collection_meta.yaml"
    transaction_ids = set()

    for _attempt in range(3):
        checkpoint_path = _write_checkpoint(checkpoint_dir, done=["case-a"], failed=[])
        _write_perf_event_through_logger(staging_path)
        real_retain = collect_mod._retain_claimed_transaction_files

        def crash_after_sidecar_publication(claimed_files, *args, **kwargs):
            if any(claimed.original.path in (pending_path, meta_path) for claimed in claimed_files):
                raise SimulatedCrash
            return real_retain(claimed_files, *args, **kwargs)

        with monkeypatch.context() as fault:
            fault.setattr(collect_mod, "_retain_claimed_transaction_files", crash_after_sidecar_publication)
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
        transaction_ids.add(transaction_id)

        assert (
            collect_mod._recover_collector_provenance_transaction(
                output_root,
                backend=BACKEND,
                checkpoint_dir=str(checkpoint_dir),
            )
            == meta_path
        )
        assert _attempted(checkpoint_path) == set()
        assert not any(".transaction-claim" in path.name for path in output_root.iterdir())
        assert collect_mod.collector_retained_path(pending_path).is_file()
        assert collect_mod.collector_retained_path(staging_path).is_file()

    assert len(transaction_ids) == 3
    assert collect_mod.collector_retained_path(meta_path).is_file()


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
        "render_collection_meta",
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


def test_hard_exit_after_first_parquet_publish_recovers_all_old_then_retries(tmp_path):
    output_root = tmp_path / "out"
    output_root.mkdir()
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
    staging_paths = [output_root / "gemm_perf.txt", output_root / "moe_perf.txt"]
    for path, shape, latency in zip(staging_paths, ("new-gemm", "new-moe"), (1.0, 2.0), strict=True):
        path.write_text(f"op,shape,latency\nmatmul,{shape},{latency}\n", encoding="utf-8")
    parquet_paths = [path.with_suffix(".parquet") for path in staging_paths]
    for path, shape, latency in zip(parquet_paths, ("old-gemm", "old-moe"), (9.0, 8.0), strict=True):
        pq.write_table(pa.table({"op": ["matmul"], "shape": [shape], "latency": [latency]}), path)
    provenance.write_collection_meta(
        output_root,
        _sidecar_runtime(),
        {"gemm_perf": _collection_event(), "moe_perf": _collection_event()},
    )
    parquet_before = {path: path.read_bytes() for path in parquet_paths}
    staging_before = {path: path.read_bytes() for path in staging_paths}
    sidecar_path = output_root / "collection_meta.yaml"
    sidecar_before = sidecar_path.read_bytes()

    process = mp.get_context("spawn").Process(
        target=_crash_after_first_parquet_publish,
        args=(str(output_root), str(checkpoint_dir), collections),
    )
    process.start()
    process.join(timeout=20)
    if process.is_alive():
        process.terminate()
        process.join()
        pytest.fail("collector finalization crash subprocess did not exit")

    assert process.exitcode == 86
    assert (output_root / collect_mod._PERF_TRANSACTION_FILENAME).is_file()
    assert {path: path.read_bytes() for path in parquet_paths} != parquet_before

    rollback_process = mp.get_context("spawn").Process(
        target=_crash_after_first_parquet_rollback,
        args=(str(output_root), str(checkpoint_dir)),
    )
    rollback_process.start()
    rollback_process.join(timeout=20)
    if rollback_process.is_alive():
        rollback_process.terminate()
        rollback_process.join()
        pytest.fail("collector rollback crash subprocess did not exit")

    assert rollback_process.exitcode == 89
    assert (output_root / collect_mod._PERF_TRANSACTION_FILENAME).is_file()
    assert not parquet_paths[0].exists()
    assert parquet_paths[1].read_bytes() == parquet_before[parquet_paths[1]]

    assert (
        collect_mod._recover_collector_provenance_transaction(
            output_root,
            backend=BACKEND,
            checkpoint_dir=str(checkpoint_dir),
        )
        is None
    )

    assert {path: path.read_bytes() for path in parquet_paths} == parquet_before
    assert {path: path.read_bytes() for path in staging_paths} == staging_before
    assert sidecar_path.read_bytes() == sidecar_before
    assert _attempted(gemm_checkpoint) == {"gemm-case"}
    assert _attempted(moe_checkpoint) == {"moe-case"}
    assert not (output_root / collect_mod._PERF_TRANSACTION_FILENAME).exists()

    collect_mod._finalize_collector_outputs_transaction(
        output_root,
        staging_paths,
        _provenance_ctx(collections),
        [],
        backend=BACKEND,
        checkpoint_dir=str(checkpoint_dir),
        sm_version=100,
    )

    assert all(pq.read_metadata(path).num_rows == 2 for path in parquet_paths)
    assert not any(path.exists() for path in staging_paths)
    assert _attempted(gemm_checkpoint) == set()
    assert _attempted(moe_checkpoint) == set()
    assert not (output_root / collect_mod._PERF_TRANSACTION_FILENAME).exists()


def test_incompatible_schema_publish_crash_retry_replaces_sidecar_event_once(tmp_path):
    output_root = tmp_path / "out"
    output_root.mkdir()
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_path = _write_checkpoint(
        checkpoint_dir,
        done=["replacement-case"],
        failed=[],
        attempted=["replacement-case"],
    )
    staging_path = output_root / "gemm_perf.txt"
    staging_path.write_text("op,new_shape,latency\nmatmul,new,1.0\n", encoding="utf-8")
    parquet_path = staging_path.with_suffix(".parquet")
    pq.write_table(pa.table({"op": ["matmul"], "old_shape": ["old"], "latency": [9.0]}), parquet_path)
    provenance.write_collection_meta(
        output_root,
        _sidecar_runtime(),
        {"gemm_perf": _collection_event(rows=1)},
    )
    parquet_before = parquet_path.read_bytes()
    sidecar_path = output_root / "collection_meta.yaml"
    sidecar_before = sidecar_path.read_bytes()

    process = mp.get_context("spawn").Process(
        target=_crash_after_first_parquet_publish,
        args=(str(output_root), str(checkpoint_dir), _collections()),
    )
    process.start()
    process.join(timeout=20)
    if process.is_alive():
        process.terminate()
        process.join()
        pytest.fail("collector replacement crash subprocess did not exit")

    assert process.exitcode == 86
    assert pq.read_table(parquet_path).to_pylist() == [{"op": "matmul", "new_shape": "new", "latency": 1.0}]
    transaction = json.loads((output_root / collect_mod._PERF_TRANSACTION_FILENAME).read_text(encoding="utf-8"))
    [publication] = transaction["publications"]
    assert publication["merged_existing"] is False
    assert publication["new_rows"] == 1
    assert publication["source"]["digest"] == _independent_digest(staging_path)
    assert publication["previous_target"]["digest"] == "sha256:" + hashlib.sha256(parquet_before).hexdigest()
    assert publication["prepared"]["digest"] == _independent_digest(parquet_path)
    assert (publication["prepared"]["device"], publication["prepared"]["inode"]) == (
        parquet_path.stat().st_dev,
        parquet_path.stat().st_ino,
    )
    assert transaction["previous_sidecar"]["digest"] == "sha256:" + hashlib.sha256(sidecar_before).hexdigest()
    [checkpoint_record] = transaction["checkpoints"]
    assert checkpoint_record["digest"] == _independent_digest(checkpoint_path)
    assert (checkpoint_record["device"], checkpoint_record["inode"]) == (
        checkpoint_path.stat().st_dev,
        checkpoint_path.stat().st_ino,
    )

    assert (
        collect_mod._recover_collector_provenance_transaction(
            output_root,
            backend=BACKEND,
            checkpoint_dir=str(checkpoint_dir),
        )
        is None
    )
    assert parquet_path.read_bytes() == parquet_before
    assert sidecar_path.read_bytes() == sidecar_before
    assert _attempted(checkpoint_path) == {"replacement-case"}

    collect_mod._finalize_collector_outputs_transaction(
        output_root,
        [staging_path],
        _provenance_ctx(_collections()),
        [],
        backend=BACKEND,
        checkpoint_dir=str(checkpoint_dir),
        sm_version=100,
    )

    assert pq.read_table(parquet_path).to_pylist() == [{"op": "matmul", "new_shape": "new", "latency": 1.0}]
    table = yaml.safe_load(sidecar_path.read_text(encoding="utf-8"))["tables"]["gemm_perf"]
    assert table["rows"] == 1
    assert table["case_plan_hash"] == provenance.case_plan_hash(["replacement-case"])
    assert "collections" not in table
    assert _attempted(checkpoint_path) == set()


def test_recovery_finishes_all_new_after_sidecar_journal_handoff(tmp_path):
    output_root = tmp_path / "out"
    output_root.mkdir()
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_path = _write_checkpoint(
        checkpoint_dir,
        done=["case-a"],
        failed=[],
        attempted=["case-a"],
    )
    staging_path = output_root / "gemm_perf.txt"
    staging_path.write_text("op,shape,latency\nmatmul,new,1.0\n", encoding="utf-8")
    parquet_path = staging_path.with_suffix(".parquet")
    pq.write_table(pa.table({"op": ["matmul"], "shape": ["old"], "latency": [9.0]}), parquet_path)
    provenance.write_collection_meta(
        output_root,
        _sidecar_runtime(),
        {"gemm_perf": _collection_event()},
    )

    process = mp.get_context("spawn").Process(
        target=_crash_before_perf_to_sidecar_handoff,
        args=(str(output_root), str(checkpoint_dir)),
    )
    process.start()
    process.join(timeout=20)
    if process.is_alive():
        process.terminate()
        process.join()
        pytest.fail("collector handoff crash subprocess did not exit")

    assert process.exitcode == 87
    assert (output_root / collect_mod._PERF_TRANSACTION_FILENAME).is_file()
    assert (output_root / collect_mod._SIDECAR_TRANSACTION_FILENAME).is_file()

    assert (
        collect_mod._recover_collector_provenance_transaction(
            output_root,
            backend=BACKEND,
            checkpoint_dir=str(checkpoint_dir),
        )
        == output_root / "collection_meta.yaml"
    )

    assert {row["shape"] for row in pq.read_table(parquet_path).to_pylist()} == {"old", "new"}
    assert not staging_path.exists()
    assert _attempted(checkpoint_path) == set()
    assert not (output_root / collect_mod._PERF_TRANSACTION_FILENAME).exists()
    assert not (output_root / collect_mod._SIDECAR_TRANSACTION_FILENAME).exists()
    assert list(output_root.glob(".*.rollback")) == []


def test_recovery_rolls_back_unjournaled_sidecar_staging_to_all_old(tmp_path):
    output_root = tmp_path / "out"
    output_root.mkdir()
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_path = _write_checkpoint(
        checkpoint_dir,
        done=["case-a"],
        failed=[],
        attempted=["case-a"],
    )
    staging_path = output_root / "gemm_perf.txt"
    staging_path.write_text("op,shape,latency\nmatmul,new,1.0\n", encoding="utf-8")
    parquet_path = staging_path.with_suffix(".parquet")
    pq.write_table(pa.table({"op": ["matmul"], "shape": ["old"], "latency": [9.0]}), parquet_path)
    provenance.write_collection_meta(
        output_root,
        _sidecar_runtime(),
        {"gemm_perf": _collection_event()},
    )
    parquet_before = parquet_path.read_bytes()
    sidecar_path = output_root / "collection_meta.yaml"
    sidecar_before = sidecar_path.read_bytes()

    process = mp.get_context("spawn").Process(
        target=_crash_after_unjournaled_sidecar_staging,
        args=(str(output_root), str(checkpoint_dir)),
    )
    process.start()
    process.join(timeout=20)
    if process.is_alive():
        process.terminate()
        process.join()
        pytest.fail("collector sidecar-staging crash subprocess did not exit")

    assert process.exitcode == 88
    assert (output_root / collect_mod._PERF_TRANSACTION_FILENAME).is_file()
    assert (output_root / collect_mod._SIDECAR_STAGING_FILENAME).is_file()
    assert not (output_root / collect_mod._SIDECAR_TRANSACTION_FILENAME).exists()

    assert (
        collect_mod._recover_collector_provenance_transaction(
            output_root,
            backend=BACKEND,
            checkpoint_dir=str(checkpoint_dir),
        )
        is None
    )

    assert parquet_path.read_bytes() == parquet_before
    assert sidecar_path.read_bytes() == sidecar_before
    assert staging_path.exists()
    assert _attempted(checkpoint_path) == {"case-a"}
    assert not (output_root / collect_mod._PERF_TRANSACTION_FILENAME).exists()
    assert not (output_root / collect_mod._SIDECAR_STAGING_FILENAME).exists()


def test_repeated_unjournaled_sidecar_staging_crashes_do_not_leak_render_directories(tmp_path):
    output_root = tmp_path / "out"
    output_root.mkdir()
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_path = _write_checkpoint(
        checkpoint_dir,
        done=["case-a"],
        failed=[],
        attempted=["case-a"],
    )
    staging_path = output_root / "gemm_perf.txt"
    staging_path.write_text("op,shape,latency\nmatmul,new,1.0\n", encoding="utf-8")
    parquet_path = staging_path.with_suffix(".parquet")
    pq.write_table(pa.table({"op": ["matmul"], "shape": ["old"], "latency": [9.0]}), parquet_path)
    provenance.write_collection_meta(
        output_root,
        _sidecar_runtime(),
        {"gemm_perf": _collection_event()},
    )
    render_counts = []

    for _attempt in range(3):
        _run_hard_exit(
            _crash_after_unjournaled_sidecar_staging,
            (str(output_root), str(checkpoint_dir)),
            88,
            "collector unjournaled sidecar-staging crash",
        )
        assert _recover_finalization(output_root, checkpoint_dir) is None
        assert staging_path.is_file()
        assert _attempted(checkpoint_path) == {"case-a"}
        render_counts.append(sum(path.name.startswith(".collection-meta-render") for path in output_root.iterdir()))

    assert render_counts == [0, 0, 0]


def test_recovery_uses_sidecar_perf_attestations_after_perf_journal_cleanup(tmp_path, monkeypatch):
    import helper as helper_mod

    output_root = tmp_path / "out"
    output_root.mkdir()
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_path = _write_checkpoint(
        checkpoint_dir,
        done=["case-a"],
        failed=[],
        attempted=["case-a"],
    )
    staging_path = output_root / "gemm_perf.txt"
    staging_path.write_text("op,shape,latency\nmatmul,new,1.0\n", encoding="utf-8")
    parquet_path = staging_path.with_suffix(".parquet")
    pq.write_table(pa.table({"op": ["matmul"], "shape": ["old"], "latency": [9.0]}), parquet_path)
    provenance.write_collection_meta(
        output_root,
        _sidecar_runtime(),
        {"gemm_perf": _collection_event()},
    )

    process = mp.get_context("spawn").Process(
        target=_crash_after_perf_journal_cleanup,
        args=(str(output_root), str(checkpoint_dir)),
    )
    process.start()
    process.join(timeout=20)
    if process.is_alive():
        process.terminate()
        process.join()
        pytest.fail("collector final-journal cleanup crash subprocess did not exit")

    assert process.exitcode == 90
    assert not (output_root / collect_mod._PERF_TRANSACTION_FILENAME).exists()
    sidecar_journal = output_root / collect_mod._SIDECAR_TRANSACTION_FILENAME
    assert sidecar_journal.is_file()
    transaction = json.loads(sidecar_journal.read_text(encoding="utf-8"))
    [publication] = transaction["perf_publications"]
    assert publication["prepared"]["digest"] == _independent_digest(parquet_path)
    assert (publication["prepared"]["device"], publication["prepared"]["inode"]) == (
        parquet_path.stat().st_dev,
        parquet_path.stat().st_ino,
    )
    assert not staging_path.exists()
    assert _attempted(checkpoint_path) == set()
    rows_before_recovery = pq.read_table(parquet_path).to_pylist()
    sidecar_before_recovery = (output_root / "collection_meta.yaml").read_bytes()
    fsync_states = []
    real_fsync = os.fsync
    output_identity = (output_root.stat().st_dev, output_root.stat().st_ino)

    def record_transaction_journal_state(file_descriptor):
        opened = os.fstat(file_descriptor)
        if stat.S_ISDIR(opened.st_mode) and (opened.st_dev, opened.st_ino) == output_identity:
            fsync_states.append(
                (
                    (output_root / collect_mod._PERF_TRANSACTION_FILENAME).exists(),
                    sidecar_journal.exists(),
                )
            )
        return real_fsync(file_descriptor)

    monkeypatch.setattr(helper_mod.os, "fsync", record_transaction_journal_state)

    assert (
        collect_mod._recover_collector_provenance_transaction(
            output_root,
            backend=BACKEND,
            checkpoint_dir=str(checkpoint_dir),
        )
        is None
    )

    assert pq.read_table(parquet_path).to_pylist() == rows_before_recovery
    assert (output_root / "collection_meta.yaml").read_bytes() == sidecar_before_recovery
    assert not sidecar_journal.exists()
    assert fsync_states.index((False, True)) < fsync_states.index((False, False))


def test_recovery_rejects_published_parquet_mode_change_before_irreversible_commit(tmp_path):
    output_root = tmp_path / "out"
    output_root.mkdir()
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_path = _write_checkpoint(
        checkpoint_dir,
        done=["case-a"],
        failed=[],
        attempted=["case-a"],
    )
    staging_path = output_root / "gemm_perf.txt"
    staging_path.write_text("op,shape,latency\nmatmul,new,1.0\n", encoding="utf-8")
    parquet_path = staging_path.with_suffix(".parquet")
    pq.write_table(pa.table({"op": ["matmul"], "shape": ["old"], "latency": [9.0]}), parquet_path)
    provenance.write_collection_meta(
        output_root,
        _sidecar_runtime(),
        {"gemm_perf": _collection_event()},
    )

    process = mp.get_context("spawn").Process(
        target=_crash_before_perf_to_sidecar_handoff,
        args=(str(output_root), str(checkpoint_dir)),
    )
    process.start()
    process.join(timeout=20)
    if process.is_alive():
        process.terminate()
        process.join()
        pytest.fail("collector handoff crash subprocess did not exit")

    assert process.exitcode == 87
    changed_mode = 0o600 if stat.S_IMODE(parquet_path.stat().st_mode) != 0o600 else 0o644
    parquet_path.chmod(changed_mode)

    with pytest.raises(RuntimeError, match="mode changed"):
        collect_mod._recover_collector_provenance_transaction(
            output_root,
            backend=BACKEND,
            checkpoint_dir=str(checkpoint_dir),
        )

    assert staging_path.exists()
    assert _attempted(checkpoint_path) == {"case-a"}
    assert (output_root / collect_mod._PERF_TRANSACTION_FILENAME).is_file()
    assert (output_root / collect_mod._SIDECAR_TRANSACTION_FILENAME).is_file()


def test_final_commit_revalidates_published_parquet_after_artifact_cleanup(tmp_path, monkeypatch):
    output_root = tmp_path / "out"
    output_root.mkdir()
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_path = _write_checkpoint(
        checkpoint_dir,
        done=["case-a"],
        failed=[],
        attempted=["case-a"],
    )
    staging_path = output_root / "gemm_perf.txt"
    staging_path.write_text("op,shape,latency\nmatmul,new,1.0\n", encoding="utf-8")
    parquet_path = staging_path.with_suffix(".parquet")
    pq.write_table(pa.table({"op": ["matmul"], "shape": ["old"], "latency": [9.0]}), parquet_path)
    provenance.write_collection_meta(
        output_root,
        _sidecar_runtime(),
        {"gemm_perf": _collection_event()},
    )
    real_cleanup = collect_mod.cleanup_perf_publication_artifacts
    tampered = False

    def tamper_during_cleanup(publications, *args, **kwargs):
        nonlocal tampered
        publications = tuple(publications)
        if not tampered:
            tampered = True
            attacker = output_root / ".attacker.parquet"
            pq.write_table(
                pa.table({"op": ["matmul"], "shape": ["attacker"], "latency": [666.0]}),
                attacker,
            )
            os.replace(attacker, publications[0].target)
        real_cleanup(publications, *args, **kwargs)

    monkeypatch.setattr(collect_mod, "cleanup_perf_publication_artifacts", tamper_during_cleanup)

    with pytest.raises(RuntimeError, match=r"published perf target|finalization artifact"):
        collect_mod._finalize_collector_outputs_transaction(
            output_root,
            [staging_path],
            _provenance_ctx(_collections()),
            [],
            backend=BACKEND,
            checkpoint_dir=str(checkpoint_dir),
            sm_version=100,
        )

    assert tampered
    assert pq.read_table(parquet_path).to_pylist() == [{"op": "matmul", "shape": "attacker", "latency": 666.0}]
    assert (output_root / collect_mod._PERF_TRANSACTION_FILENAME).is_file()
    assert (output_root / collect_mod._SIDECAR_TRANSACTION_FILENAME).is_file()
    assert _attempted(checkpoint_path) == set()


def test_final_commit_rejects_duplicate_prepared_path_without_unlinking_it(tmp_path, monkeypatch):
    output_root = tmp_path / "out"
    output_root.mkdir()
    checkpoint_dir = tmp_path / "checkpoint"
    _write_checkpoint(
        checkpoint_dir,
        done=["case-a"],
        failed=[],
        attempted=["case-a"],
    )
    staging_path = output_root / "gemm_perf.txt"
    staging_path.write_text("op,shape,latency\nmatmul,new,1.0\n", encoding="utf-8")
    parquet_path = staging_path.with_suffix(".parquet")
    pq.write_table(pa.table({"op": ["matmul"], "shape": ["old"], "latency": [9.0]}), parquet_path)
    real_cleanup = collect_mod.cleanup_perf_publication_artifacts
    duplicated_path = None

    def duplicate_prepared_path(publications, *args, **kwargs):
        nonlocal duplicated_path
        publications = tuple(publications)
        if duplicated_path is None:
            duplicated_path = publications[0].prepared.path
            os.link(publications[0].target, duplicated_path)
        return real_cleanup(publications, *args, **kwargs)

    monkeypatch.setattr(collect_mod, "cleanup_perf_publication_artifacts", duplicate_prepared_path)

    with pytest.raises(RuntimeError, match="prepared claim unexpectedly exists"):
        collect_mod._finalize_collector_outputs_transaction(
            output_root,
            [staging_path],
            _provenance_ctx(_collections()),
            [],
            backend=BACKEND,
            checkpoint_dir=str(checkpoint_dir),
            sm_version=100,
        )

    assert duplicated_path is not None
    assert duplicated_path.is_file()
    assert duplicated_path.samefile(parquet_path)
    assert (output_root / collect_mod._PERF_TRANSACTION_FILENAME).is_file()
    assert (output_root / collect_mod._SIDECAR_TRANSACTION_FILENAME).is_file()


def test_final_commit_rejects_published_parquet_mode_change(tmp_path, monkeypatch):
    output_root = tmp_path / "out"
    output_root.mkdir()
    checkpoint_dir = tmp_path / "checkpoint"
    _write_checkpoint(
        checkpoint_dir,
        done=["case-a"],
        failed=[],
        attempted=["case-a"],
    )
    staging_path = output_root / "gemm_perf.txt"
    staging_path.write_text("op,shape,latency\nmatmul,new,1.0\n", encoding="utf-8")
    parquet_path = staging_path.with_suffix(".parquet")
    pq.write_table(pa.table({"op": ["matmul"], "shape": ["old"], "latency": [9.0]}), parquet_path)
    real_cleanup = collect_mod.cleanup_perf_publication_artifacts
    changed_mode = None

    def chmod_during_cleanup(publications, *args, **kwargs):
        nonlocal changed_mode
        publications = tuple(publications)
        real_cleanup(publications, *args, **kwargs)
        changed_mode = publications[0].prepared.mode ^ stat.S_IXUSR
        publications[0].target.chmod(changed_mode)

    monkeypatch.setattr(collect_mod, "cleanup_perf_publication_artifacts", chmod_during_cleanup)

    with pytest.raises(RuntimeError, match="published perf target"):
        collect_mod._finalize_collector_outputs_transaction(
            output_root,
            [staging_path],
            _provenance_ctx(_collections()),
            [],
            backend=BACKEND,
            checkpoint_dir=str(checkpoint_dir),
            sm_version=100,
        )

    assert changed_mode is not None
    assert stat.S_IMODE(parquet_path.stat().st_mode) == changed_mode
    assert (output_root / collect_mod._PERF_TRANSACTION_FILENAME).is_file()
    assert (output_root / collect_mod._SIDECAR_TRANSACTION_FILENAME).is_file()


def test_perf_journal_retirement_is_fsynced_before_sidecar_journal_retirement(tmp_path, monkeypatch):
    import helper as helper_mod

    output_root = tmp_path / "out"
    output_root.mkdir()
    checkpoint_dir = tmp_path / "checkpoint"
    _write_checkpoint(
        checkpoint_dir,
        done=["case-a"],
        failed=[],
        attempted=["case-a"],
    )
    staging_path = output_root / "gemm_perf.txt"
    staging_path.write_text("op,shape,latency\nmatmul,new,1.0\n", encoding="utf-8")
    fsync_states = []
    real_fsync = os.fsync
    output_identity = (output_root.stat().st_dev, output_root.stat().st_ino)

    def record_transaction_journal_state(file_descriptor):
        opened = os.fstat(file_descriptor)
        if stat.S_ISDIR(opened.st_mode) and (opened.st_dev, opened.st_ino) == output_identity:
            fsync_states.append(
                (
                    (output_root / collect_mod._PERF_TRANSACTION_FILENAME).exists(),
                    (output_root / collect_mod._SIDECAR_TRANSACTION_FILENAME).exists(),
                )
            )
        return real_fsync(file_descriptor)

    monkeypatch.setattr(helper_mod.os, "fsync", record_transaction_journal_state)

    collect_mod._finalize_collector_outputs_transaction(
        output_root,
        [staging_path],
        _provenance_ctx(_collections()),
        [],
        backend=BACKEND,
        checkpoint_dir=str(checkpoint_dir),
        sm_version=100,
    )

    perf_unlink_fsync = fsync_states.index((False, True))
    sidecar_unlink_fsync = fsync_states.index((False, False), perf_unlink_fsync + 1)
    assert perf_unlink_fsync < sidecar_unlink_fsync


def test_cleanup_fsyncs_parent_when_attested_journal_is_already_absent(tmp_path, monkeypatch):
    journal_path = tmp_path / collect_mod._PERF_TRANSACTION_FILENAME
    journal_path.write_bytes(b"journal")
    attestation = _file_attestation(journal_path)
    journal_path.unlink()
    fsynced = []

    monkeypatch.setattr(collect_mod, "_fsync_directory", lambda directory: fsynced.append(Path(directory)))

    collect_mod._cleanup_transaction_files([attestation], context_path=journal_path)

    assert fsynced == [tmp_path]


def test_publish_rejects_target_replaced_after_durable_journal(tmp_path, monkeypatch):
    output_root = tmp_path / "out"
    output_root.mkdir()
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_path = _write_checkpoint(
        checkpoint_dir,
        done=["case-a"],
        failed=[],
        attempted=["case-a"],
    )
    staging_path = output_root / "gemm_perf.txt"
    staging_path.write_text("op,shape,latency\nmatmul,new,1.0\n", encoding="utf-8")
    parquet_path = staging_path.with_suffix(".parquet")
    pq.write_table(pa.table({"op": ["matmul"], "shape": ["old"], "latency": [9.0]}), parquet_path)
    real_prepare = collect_mod._CollectorPerfPublicationTransaction.prepare

    def prepare_then_replace_target(transaction, publications):
        real_prepare(transaction, publications)
        attacker = output_root / ".publish-attacker.parquet"
        pq.write_table(
            pa.table({"op": ["matmul"], "shape": ["attacker"], "latency": [666.0]}),
            attacker,
        )
        os.replace(attacker, publications[0].target)

    monkeypatch.setattr(
        collect_mod._CollectorPerfPublicationTransaction,
        "prepare",
        prepare_then_replace_target,
    )

    with pytest.raises(RuntimeError, match="changed"):
        collect_mod._finalize_collector_outputs_transaction(
            output_root,
            [staging_path],
            _provenance_ctx(_collections()),
            [],
            backend=BACKEND,
            checkpoint_dir=str(checkpoint_dir),
            sm_version=100,
        )

    assert pq.read_table(parquet_path).to_pylist() == [{"op": "matmul", "shape": "attacker", "latency": 666.0}]
    assert staging_path.exists()
    assert _attempted(checkpoint_path) == {"case-a"}
    assert (output_root / collect_mod._PERF_TRANSACTION_FILENAME).is_file()


def test_publish_atomically_preserves_target_replaced_at_claim(tmp_path, monkeypatch):
    import helper as helper_mod

    output_root = tmp_path / "out"
    output_root.mkdir()
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_path = _write_checkpoint(
        checkpoint_dir,
        done=["case-a"],
        failed=[],
        attempted=["case-a"],
    )
    staging_path = output_root / "gemm_perf.txt"
    staging_path.write_text("op,shape,latency\nmatmul,new,1.0\n", encoding="utf-8")
    parquet_path = staging_path.with_suffix(".parquet")
    pq.write_table(pa.table({"op": ["matmul"], "shape": ["old"], "latency": [9.0]}), parquet_path)
    real_rename_noreplace = helper_mod._rename_noreplace_at
    raced = False

    def replace_target_before_claim(source, target, directory_fd):
        nonlocal raced
        source_path = output_root / source
        target_path = output_root / target
        if source_path == parquet_path and target_path.name.endswith(".claim") and not raced:
            raced = True
            attacker = output_root / ".publish-claim-attacker.parquet"
            pq.write_table(
                pa.table({"op": ["matmul"], "shape": ["attacker"], "latency": [666.0]}),
                attacker,
            )
            os.replace(attacker, source_path)
        return real_rename_noreplace(source, target, directory_fd)

    monkeypatch.setattr(helper_mod, "_rename_noreplace_at", replace_target_before_claim)

    with pytest.raises(RuntimeError, match=r"strict recovery state|target claim changed"):
        collect_mod._finalize_collector_outputs_transaction(
            output_root,
            [staging_path],
            _provenance_ctx(_collections()),
            [],
            backend=BACKEND,
            checkpoint_dir=str(checkpoint_dir),
            sm_version=100,
        )

    assert raced
    assert pq.read_table(parquet_path).to_pylist() == [{"op": "matmul", "shape": "attacker", "latency": 666.0}]
    assert staging_path.exists()
    assert _attempted(checkpoint_path) == {"case-a"}
    assert (output_root / collect_mod._PERF_TRANSACTION_FILENAME).is_file()


def test_hard_exit_before_perf_journal_does_not_leave_rollback_snapshot(tmp_path):
    output_root, checkpoint_dir, checkpoint_path, staging_path, parquet_path = _single_table_finalization(tmp_path)
    parquet_before = parquet_path.read_bytes()

    for _attempt in range(3):
        _run_hard_exit(
            _crash_before_perf_journal_publish,
            (str(output_root), str(checkpoint_dir)),
            91,
            "collector pre-journal crash",
        )
        assert parquet_path.read_bytes() == parquet_before
        assert list(output_root.glob(".*.rollback")) == []
        assert _recover_finalization(output_root, checkpoint_dir) is None

    assert parquet_path.read_bytes() == parquet_before
    assert staging_path.exists()
    retained = output_root / ".gemm_perf.parquet.tmp"
    _assert_retained_slot(retained)
    assert list(output_root.glob(".*.tmp")) == [retained]
    assert not (output_root / collect_mod._PERF_TRANSACTION_FILENAME).exists()

    _finalize_single_table(output_root, checkpoint_dir, staging_path)

    assert {row["shape"] for row in pq.read_table(parquet_path).to_pylist()} == {"old", "new"}
    assert not staging_path.exists()
    assert _attempted(checkpoint_path) == set()


def test_hard_exit_during_parquet_render_recovers_exact_reserved_path_and_retries(tmp_path):
    output_root, checkpoint_dir, checkpoint_path, staging_path, parquet_path = _single_table_finalization(tmp_path)
    parquet_before = parquet_path.read_bytes()

    _run_hard_exit(
        _crash_during_first_parquet_render,
        (str(output_root), str(checkpoint_dir)),
        94,
        "collector mid-render crash",
    )

    assert parquet_path.read_bytes() == parquet_before
    assert staging_path.exists()
    assert _attempted(checkpoint_path) == {"case-a"}
    assert _recover_finalization(output_root, checkpoint_dir) is None
    retained = output_root / ".gemm_perf.parquet.tmp"
    _assert_retained_slot(retained)
    assert list(output_root.glob(".*.tmp")) == [retained]

    _finalize_single_table(output_root, checkpoint_dir, staging_path)

    assert {row["shape"] for row in pq.read_table(parquet_path).to_pylist()} == {"old", "new"}
    assert not staging_path.exists()
    assert _attempted(checkpoint_path) == set()


def test_hard_exit_after_first_multi_table_render_recovers_batch_and_retries(tmp_path):
    output_root = tmp_path / "out"
    output_root.mkdir()
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
    staging_paths = [output_root / "gemm_perf.txt", output_root / "moe_perf.txt"]
    parquet_paths = [path.with_suffix(".parquet") for path in staging_paths]
    for path, shape in zip(staging_paths, ("new-gemm", "new-moe"), strict=True):
        path.write_text(f"op,shape,latency\nmatmul,{shape},1.0\n", encoding="utf-8")
    for path, shape in zip(parquet_paths, ("old-gemm", "old-moe"), strict=True):
        pq.write_table(pa.table({"op": ["matmul"], "shape": [shape], "latency": [9.0]}), path)
    parquet_before = {path: path.read_bytes() for path in parquet_paths}

    _run_hard_exit(
        _crash_after_first_parquet_render,
        (str(output_root), str(checkpoint_dir), collections),
        93,
        "collector post-render crash",
    )

    assert {path: path.read_bytes() for path in parquet_paths} == parquet_before
    assert _recover_finalization(output_root, checkpoint_dir) is None
    retained = output_root / ".gemm_perf.parquet.tmp"
    _assert_retained_slot(retained)
    assert list(output_root.glob(".*.tmp")) == [retained]
    assert all(path.exists() for path in staging_paths)
    assert _attempted(gemm_checkpoint) == {"gemm-case"}
    assert _attempted(moe_checkpoint) == {"moe-case"}

    collect_mod._finalize_collector_outputs_transaction(
        output_root,
        staging_paths,
        _provenance_ctx(collections),
        [],
        backend=BACKEND,
        checkpoint_dir=str(checkpoint_dir),
        sm_version=100,
    )

    assert all(pq.read_metadata(path).num_rows == 2 for path in parquet_paths)
    assert not any(path.exists() for path in staging_paths)
    assert _attempted(gemm_checkpoint) == set()
    assert _attempted(moe_checkpoint) == set()


def test_recovery_retains_unjournaled_regular_reserved_parquet(tmp_path):
    output_root = tmp_path / "out"
    output_root.mkdir()
    reserved = output_root / ".gemm_perf.parquet.tmp"
    reserved.write_bytes(b"partial parquet from an interrupted render")
    unrelated = output_root / ".gemm_perf.parquet.unrelated.tmp"
    unrelated.write_bytes(b"not a canonical collector reservation")

    assert (
        collect_mod._recover_collector_provenance_transaction(
            output_root,
            backend=BACKEND,
            checkpoint_dir=str(tmp_path / "checkpoint"),
        )
        is None
    )

    _assert_retained_slot(reserved)
    assert unrelated.read_bytes() == b"not a canonical collector reservation"


@pytest.mark.parametrize("replacement", ["symlink", "directory", "hardlink"])
def test_recovery_preserves_unowned_reserved_parquet_objects(tmp_path, replacement):
    output_root = tmp_path / "out"
    output_root.mkdir()
    reserved = output_root / ".gemm_perf.parquet.tmp"
    outside = tmp_path / "outside"
    outside.write_bytes(b"must survive")
    if replacement == "symlink":
        try:
            reserved.symlink_to(outside)
        except OSError as error:
            pytest.skip(f"symlinks unavailable: {error}")
    elif replacement == "directory":
        reserved.mkdir()
    else:
        os.link(outside, reserved)

    with pytest.raises(RuntimeError, match=r"reserved|temporary|artifact"):
        collect_mod._recover_collector_provenance_transaction(
            output_root,
            backend=BACKEND,
            checkpoint_dir=str(tmp_path / "checkpoint"),
        )

    assert reserved.exists() or reserved.is_symlink()
    assert outside.read_bytes() == b"must survive"


def test_hard_exit_after_perf_journal_prepare_recovers_without_publication(tmp_path):
    output_root, checkpoint_dir, checkpoint_path, staging_path, parquet_path = _single_table_finalization(tmp_path)
    parquet_before = parquet_path.read_bytes()

    _run_hard_exit(
        _crash_after_perf_journal_prepare,
        (str(output_root), str(checkpoint_dir)),
        95,
        "collector post-prepare crash",
    )

    assert (output_root / collect_mod._PERF_TRANSACTION_FILENAME).is_file()
    assert parquet_path.read_bytes() == parquet_before
    assert _recover_finalization(output_root, checkpoint_dir) is None
    assert parquet_path.read_bytes() == parquet_before
    assert staging_path.exists()
    assert _attempted(checkpoint_path) == {"case-a"}
    assert not (output_root / collect_mod._PERF_TRANSACTION_FILENAME).exists()


def test_repeated_hard_exit_before_perf_journal_publication_is_bounded(tmp_path):
    output_root, checkpoint_dir, checkpoint_path, staging_path, parquet_path = _single_table_finalization(tmp_path)
    parquet_before = parquet_path.read_bytes()
    journal_path = output_root / collect_mod._PERF_TRANSACTION_FILENAME
    reservation = collect_mod.atomic_write_reservation_path(journal_path)
    cleanup_claim = output_root / f".{collect_mod._PERF_TRANSACTION_FILENAME}.tmp.cleanup"

    for _attempt in range(3):
        _run_hard_exit(
            _crash_around_atomic_journal_publication,
            (
                str(output_root),
                str(checkpoint_dir),
                collect_mod._PERF_TRANSACTION_FILENAME,
                False,
                96,
            ),
            96,
            "collector pre-publication perf-journal crash",
        )
        assert _recover_finalization(output_root, checkpoint_dir) is None
        assert reservation.is_file()
        assert reservation.stat().st_nlink == 1
        assert not cleanup_claim.exists()
        assert _private_atomic_artifact_names(journal_path) == {reservation.name}

    assert parquet_path.read_bytes() == parquet_before
    assert staging_path.exists()
    assert _attempted(checkpoint_path) == {"case-a"}


def test_repeated_hard_exit_after_perf_journal_publication_is_bounded(tmp_path):
    output_root, checkpoint_dir, checkpoint_path, staging_path, parquet_path = _single_table_finalization(tmp_path)
    parquet_before = parquet_path.read_bytes()
    journal_path = output_root / collect_mod._PERF_TRANSACTION_FILENAME
    reservation = collect_mod.atomic_write_reservation_path(journal_path)
    cleanup_claim = output_root / f".{collect_mod._PERF_TRANSACTION_FILENAME}.tmp.cleanup"
    retained = collect_mod.collector_retained_path(journal_path)

    for _attempt in range(3):
        _run_hard_exit(
            _crash_around_atomic_journal_publication,
            (
                str(output_root),
                str(checkpoint_dir),
                collect_mod._PERF_TRANSACTION_FILENAME,
                True,
                97,
            ),
            97,
            "collector post-publication perf-journal crash",
        )
        assert _recover_finalization(output_root, checkpoint_dir) is None
        assert not reservation.exists()
        assert not cleanup_claim.exists()
        assert _private_atomic_artifact_names(journal_path) == {retained.name}

    assert not journal_path.exists()
    assert parquet_path.read_bytes() == parquet_before
    assert staging_path.exists()
    assert _attempted(checkpoint_path) == {"case-a"}


def test_repeated_hard_exit_before_sidecar_journal_publication_is_bounded(tmp_path):
    output_root, checkpoint_dir, checkpoint_path, staging_path, parquet_path = _single_table_finalization(tmp_path)
    parquet_before = parquet_path.read_bytes()
    journal_path = output_root / collect_mod._SIDECAR_TRANSACTION_FILENAME
    reservation = collect_mod.atomic_write_reservation_path(journal_path)
    cleanup_claim = output_root / f".{collect_mod._SIDECAR_TRANSACTION_FILENAME}.tmp.cleanup"

    for _attempt in range(3):
        _run_hard_exit(
            _crash_around_atomic_journal_publication,
            (
                str(output_root),
                str(checkpoint_dir),
                collect_mod._SIDECAR_TRANSACTION_FILENAME,
                False,
                98,
            ),
            98,
            "collector pre-publication sidecar-journal crash",
        )
        assert _recover_finalization(output_root, checkpoint_dir) is None
        assert reservation.is_file()
        assert reservation.stat().st_nlink == 1
        assert not cleanup_claim.exists()
        assert _private_atomic_artifact_names(journal_path) == {reservation.name}

    assert not (output_root / collect_mod._PERF_TRANSACTION_FILENAME).exists()
    assert not (output_root / collect_mod._SIDECAR_TRANSACTION_FILENAME).exists()
    assert parquet_path.read_bytes() == parquet_before
    assert staging_path.exists()
    assert _attempted(checkpoint_path) == {"case-a"}


def test_repeated_hard_exit_after_sidecar_journal_publication_is_bounded(tmp_path):
    output_root, checkpoint_dir, checkpoint_path, staging_path, _parquet_path = _single_table_finalization(tmp_path)
    journal_path = output_root / collect_mod._SIDECAR_TRANSACTION_FILENAME
    reservation = collect_mod.atomic_write_reservation_path(journal_path)
    cleanup_claim = output_root / f".{collect_mod._SIDECAR_TRANSACTION_FILENAME}.tmp.cleanup"
    retained = collect_mod.collector_retained_path(journal_path)
    meta_path = output_root / "collection_meta.yaml"

    for attempt in range(3):
        if attempt:
            checkpoint_path = _write_checkpoint(
                checkpoint_dir,
                done=["case-a"],
                failed=[],
                attempted=["case-a"],
            )
            _write_perf_event_through_logger(staging_path)
        _run_hard_exit(
            _crash_around_atomic_journal_publication,
            (
                str(output_root),
                str(checkpoint_dir),
                collect_mod._SIDECAR_TRANSACTION_FILENAME,
                True,
                99,
            ),
            99,
            "collector post-publication sidecar-journal crash",
        )
        assert _recover_finalization(output_root, checkpoint_dir) == meta_path
        assert not reservation.exists()
        assert not cleanup_claim.exists()
        assert not staging_path.exists()
        assert _attempted(checkpoint_path) == set()
        assert _private_atomic_artifact_names(journal_path) == {retained.name}

    assert not (output_root / collect_mod._PERF_TRANSACTION_FILENAME).exists()
    assert not (output_root / collect_mod._SIDECAR_TRANSACTION_FILENAME).exists()
    assert meta_path.is_file()


@pytest.mark.parametrize("replacement", ["symlink", "directory", "hardlink"])
def test_recovery_preserves_unowned_atomic_journal_reservation(tmp_path, replacement):
    output_root = tmp_path / "out"
    output_root.mkdir()
    journal_path = output_root / collect_mod._PERF_TRANSACTION_FILENAME
    reservation = output_root / f".{collect_mod._PERF_TRANSACTION_FILENAME}.tmp"
    outside = tmp_path / "outside"
    outside.write_bytes(b"must survive")
    if replacement == "symlink":
        try:
            reservation.symlink_to(outside)
        except OSError as error:
            pytest.skip(f"symlinks unavailable: {error}")
    elif replacement == "directory":
        reservation.mkdir()
    else:
        os.link(outside, reservation)

    with pytest.raises(RuntimeError, match=r"atomic write|artifact|Unowned"):
        _recover_finalization(output_root, tmp_path / "checkpoint")

    assert not journal_path.exists()
    assert reservation.exists() or reservation.is_symlink()
    assert outside.read_bytes() == b"must survive"


def test_recovery_retains_only_exact_atomic_journal_reservation(tmp_path):
    output_root = tmp_path / "out"
    output_root.mkdir()
    reservation = output_root / f".{collect_mod._PERF_TRANSACTION_FILENAME}.tmp"
    reservation.write_bytes(b"interrupted exact reservation")
    historical_random = output_root / f".{collect_mod._PERF_TRANSACTION_FILENAME}.historical.tmp"
    historical_random.write_bytes(b"unrelated historical artifact")

    assert _recover_finalization(output_root, tmp_path / "checkpoint") is None

    assert reservation.read_bytes() == b"interrupted exact reservation"
    assert historical_random.read_bytes() == b"unrelated historical artifact"


def test_recovery_fsyncs_atomic_journal_reservation_claim_and_cleanup(tmp_path, monkeypatch):
    import helper as helper_mod

    output_root = tmp_path / "out"
    output_root.mkdir()
    reservation = output_root / f".{collect_mod._PERF_TRANSACTION_FILENAME}.tmp"
    cleanup_claim = output_root / f".{collect_mod._PERF_TRANSACTION_FILENAME}.tmp.cleanup"
    reservation.write_bytes(b"interrupted exact reservation")
    fsync_states = []
    real_fsync = os.fsync
    output_identity = (output_root.stat().st_dev, output_root.stat().st_ino)

    def record_reservation_state(file_descriptor):
        opened = os.fstat(file_descriptor)
        state = (reservation.exists(), cleanup_claim.exists())
        if (
            stat.S_ISDIR(opened.st_mode)
            and (opened.st_dev, opened.st_ino) == output_identity
            and state in ((False, True), (True, False))
            and (not fsync_states or fsync_states[-1] != state)
        ):
            fsync_states.append(state)
        return real_fsync(file_descriptor)

    monkeypatch.setattr(helper_mod.os, "fsync", record_reservation_state)

    assert _recover_finalization(output_root, tmp_path / "checkpoint") is None

    assert fsync_states == [(False, True), (True, False)]


@pytest.mark.parametrize("target_present", [False, True])
def test_recovery_resumes_after_hard_exit_from_atomic_reservation_claim(tmp_path, monkeypatch, target_present):
    import helper as helper_mod

    class SimulatedHardExit(BaseException):
        pass

    output_root = tmp_path / "out"
    output_root.mkdir()
    journal_path = output_root / collect_mod._PERF_TRANSACTION_FILENAME
    reservation = output_root / f".{collect_mod._PERF_TRANSACTION_FILENAME}.tmp"
    cleanup_claim = output_root / f".{collect_mod._PERF_TRANSACTION_FILENAME}.tmp.cleanup"
    reservation.write_bytes(b"interrupted reservation")
    if target_present:
        os.link(reservation, journal_path)
    real_rename_noreplace = helper_mod._rename_noreplace

    def crash_after_claim(source, target):
        real_rename_noreplace(source, target)
        if Path(target) == cleanup_claim:
            raise SimulatedHardExit

    with monkeypatch.context() as fault:
        fault.setattr(helper_mod, "_rename_noreplace", crash_after_claim)
        with pytest.raises(SimulatedHardExit):
            helper_mod.cleanup_atomic_write_reservations([journal_path])

    assert not reservation.exists()
    assert cleanup_claim.is_file()
    assert cleanup_claim.stat().st_nlink == (2 if target_present else 1)

    if target_present:
        with pytest.raises(RuntimeError, match=r"atomic write|legacy|artifact"):
            helper_mod.cleanup_atomic_write_reservations([journal_path])
        assert journal_path.samefile(reservation)
        assert journal_path.stat().st_nlink == 2
    else:
        helper_mod.cleanup_atomic_write_reservations([journal_path])
        assert reservation.is_file()
        assert reservation.stat().st_nlink == 1
    assert not cleanup_claim.exists()


def test_recovery_preserves_foreign_replacement_after_atomic_reservation_claim(tmp_path, monkeypatch):
    import helper as helper_mod

    output_root = tmp_path / "out"
    output_root.mkdir()
    reservation = output_root / f".{collect_mod._PERF_TRANSACTION_FILENAME}.tmp"
    cleanup_claim = output_root / f".{collect_mod._PERF_TRANSACTION_FILENAME}.tmp.cleanup"
    displaced_owned = output_root / ".displaced-owned"
    foreign = output_root / ".foreign"
    reservation.write_bytes(b"owned reservation")
    foreign.write_bytes(b"MUST SURVIVE")
    real_rename_noreplace = helper_mod._rename_noreplace_at

    def replace_after_claim(source, target, directory_fd):
        real_rename_noreplace(source, target, directory_fd)
        if output_root / target == cleanup_claim:
            cleanup_claim.rename(displaced_owned)
            foreign.rename(cleanup_claim)

    monkeypatch.setattr(helper_mod, "_rename_noreplace_at", replace_after_claim)

    with pytest.raises(RuntimeError, match=r"atomic write|artifact|changed"):
        _recover_finalization(output_root, tmp_path / "checkpoint")

    assert displaced_owned.read_bytes() == b"owned reservation"
    assert any(
        path.is_file() and path.read_bytes() == b"MUST SURVIVE" for path in (foreign, reservation, cleanup_claim)
    )


@pytest.mark.parametrize("mutation", ["equal-size-content", "size", "mode"])
def test_recovery_preserves_same_inode_atomic_reservation_mutation_after_claim(tmp_path, monkeypatch, mutation):
    import helper as helper_mod

    output_root = tmp_path / "out"
    output_root.mkdir()
    journal_path = output_root / collect_mod._PERF_TRANSACTION_FILENAME
    reservation = output_root / f".{collect_mod._PERF_TRANSACTION_FILENAME}.tmp"
    cleanup_claim = output_root / f".{collect_mod._PERF_TRANSACTION_FILENAME}.tmp.cleanup"
    reservation.write_bytes(b"owned reservation")
    reservation.chmod(0o600)
    original_identity = (reservation.stat().st_dev, reservation.stat().st_ino)
    real_rename_noreplace = helper_mod._rename_noreplace

    def mutate_after_claim(source, target):
        real_rename_noreplace(source, target)
        if Path(target) != cleanup_claim:
            return
        if mutation == "equal-size-content":
            cleanup_claim.write_bytes(b"raced reservation")
        elif mutation == "size":
            with cleanup_claim.open("ab") as claimed_file:
                claimed_file.write(b"!")
        else:
            cleanup_claim.chmod(0o700)

    monkeypatch.setattr(helper_mod, "_rename_noreplace", mutate_after_claim)

    with pytest.raises(RuntimeError, match=r"atomic write|artifact|changed"):
        helper_mod.cleanup_atomic_write_reservations([journal_path])

    assert (reservation.stat().st_dev, reservation.stat().st_ino) == original_identity
    assert not cleanup_claim.exists()
    if mutation == "equal-size-content":
        assert reservation.read_bytes() == b"raced reservation"
    elif mutation == "size":
        assert reservation.read_bytes() == b"owned reservation!"
    else:
        assert stat.S_IMODE(reservation.stat().st_mode) == 0o700


def test_recovery_fails_closed_on_legacy_same_inode_atomic_journal_alias(tmp_path):
    import helper as helper_mod

    output_root = tmp_path / "out"
    output_root.mkdir()
    journal_path = output_root / collect_mod._PERF_TRANSACTION_FILENAME
    reservation = output_root / f".{collect_mod._PERF_TRANSACTION_FILENAME}.tmp"
    journal_path.write_bytes(b"legacy hardlink publication")
    os.link(journal_path, reservation)

    with pytest.raises(RuntimeError, match=r"atomic write|legacy|artifact"):
        helper_mod.cleanup_atomic_write_reservations([journal_path])

    assert journal_path.samefile(reservation)
    assert journal_path.stat().st_nlink == 2
    assert reservation.stat().st_nlink == 2
    assert not helper_mod.collector_retained_path(journal_path).exists()


@pytest.mark.parametrize("mismatch", ["content", "size", "mode"])
def test_recovery_preserves_mismatched_atomic_journal_publication(tmp_path, mismatch):
    output_root = tmp_path / "out"
    output_root.mkdir()
    journal_path = output_root / collect_mod._PERF_TRANSACTION_FILENAME
    reservation = output_root / f".{collect_mod._PERF_TRANSACTION_FILENAME}.tmp"
    journal_path.write_bytes(b"target")
    reservation.write_bytes({"content": b"raced!", "size": b"different size", "mode": b"target"}[mismatch])
    journal_path.chmod(0o600)
    reservation.chmod(0o640 if mismatch == "mode" else 0o600)
    target_alias = output_root / ".target-alias"
    reservation_alias = output_root / ".reservation-alias"
    os.link(journal_path, target_alias)
    os.link(reservation, reservation_alias)

    with pytest.raises(RuntimeError, match=r"atomic write|publication|artifact"):
        _recover_finalization(output_root, tmp_path / "checkpoint")

    assert journal_path.exists()
    assert reservation.exists()
    assert target_alias.exists()
    assert reservation_alias.exists()


def test_transactional_retry_reuses_unjournaled_deterministic_reservation(tmp_path):
    output_root, checkpoint_dir, checkpoint_path, staging_path, parquet_path = _single_table_finalization(tmp_path)
    reserved = output_root / ".gemm_perf.parquet.tmp"
    reserved.write_bytes(b"partial parquet from interrupted transaction")
    reserved_identity = (reserved.stat().st_dev, reserved.stat().st_ino)

    _finalize_single_table(output_root, checkpoint_dir, staging_path)

    assert {row["shape"] for row in pq.read_table(parquet_path).to_pylist()} == {"old", "new"}
    assert (parquet_path.stat().st_dev, parquet_path.stat().st_ino) == reserved_identity
    _assert_retained_slot(reserved)
    assert not staging_path.exists()
    assert _attempted(checkpoint_path) == set()


def test_standalone_finalize_preserves_reservation_owned_by_perf_journal(tmp_path):
    output_root, checkpoint_dir, _checkpoint_path, staging_path, _parquet_path = _single_table_finalization(tmp_path)
    _run_hard_exit(
        _crash_after_perf_journal_prepare,
        (str(output_root), str(checkpoint_dir)),
        95,
        "collector post-prepare crash",
    )
    journal_path = output_root / collect_mod._PERF_TRANSACTION_FILENAME
    transaction = json.loads(journal_path.read_text(encoding="utf-8"))
    reserved = Path(transaction["publications"][0]["prepared"]["path"])
    reserved_before = reserved.read_bytes()

    with pytest.raises(RuntimeError, match="transaction"):
        collect_mod.finalize_perf_files([staging_path], delete_source=False)

    assert reserved.read_bytes() == reserved_before
    assert journal_path.exists()


def test_valid_perf_journal_prevents_unjournaled_reserved_cleanup(tmp_path):
    output_root, checkpoint_dir, checkpoint_path, staging_path, parquet_path = _single_table_finalization(tmp_path)
    parquet_before = parquet_path.read_bytes()

    _run_hard_exit(
        _crash_after_perf_journal_prepare,
        (str(output_root), str(checkpoint_dir)),
        95,
        "collector post-prepare crash",
    )

    journal_path = output_root / collect_mod._PERF_TRANSACTION_FILENAME
    transaction = json.loads(journal_path.read_text(encoding="utf-8"))
    prepared_path = Path(transaction["publications"][0]["prepared"]["path"])
    prepared_path.unlink()
    outside = tmp_path / "outside-prepared"
    outside.write_bytes(b"must survive journal recovery")
    os.link(outside, prepared_path)

    with pytest.raises(RuntimeError, match=r"changed|recovery state"):
        collect_mod._recover_collector_provenance_transaction(
            output_root,
            backend=BACKEND,
            checkpoint_dir=str(checkpoint_dir),
        )

    assert prepared_path.samefile(outside)
    assert outside.read_bytes() == b"must survive journal recovery"
    assert parquet_path.read_bytes() == parquet_before
    assert staging_path.exists()
    assert _attempted(checkpoint_path) == {"case-a"}
    assert journal_path.exists()


def test_recovery_waits_for_live_finalization_through_sidecar_commit(tmp_path):
    output_root, checkpoint_dir, checkpoint_path, staging_path, parquet_path = _single_table_finalization(tmp_path)

    context = mp.get_context("spawn")
    live_entered = context.Event()
    release_live = context.Event()
    recovery_started = context.Event()
    recovery_finished = context.Event()
    live_process = context.Process(
        target=_pause_live_finalization_before_sidecar,
        args=(str(output_root), str(checkpoint_dir), live_entered, release_live),
    )
    recovery_process = context.Process(
        target=_recover_and_signal,
        args=(
            str(output_root),
            str(checkpoint_dir),
            recovery_started,
            recovery_finished,
        ),
    )
    live_process.start()
    try:
        assert live_entered.wait(timeout=20)
        assert (output_root / collect_mod._PERF_TRANSACTION_FILENAME).is_file()
        recovery_process.start()
        assert recovery_started.wait(timeout=20)
        assert not recovery_finished.wait(timeout=1)
    finally:
        release_live.set()
        live_process.join(timeout=20)
        recovery_process.join(timeout=20)
        for process in (live_process, recovery_process):
            if process.is_alive():
                process.terminate()
                process.join()

    assert live_process.exitcode == 0
    assert recovery_process.exitcode == 0
    assert {row["shape"] for row in pq.read_table(parquet_path).to_pylist()} == {"old", "new"}
    assert not staging_path.exists()
    assert _attempted(checkpoint_path) == set()
    assert not (output_root / collect_mod._PERF_TRANSACTION_FILENAME).exists()
    assert not (output_root / collect_mod._SIDECAR_TRANSACTION_FILENAME).exists()


def test_recovery_rejects_source_mode_change_after_perf_journal(tmp_path):
    output_root = tmp_path / "out"
    output_root.mkdir()
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_path = _write_checkpoint(
        checkpoint_dir,
        done=["case-a"],
        failed=[],
        attempted=["case-a"],
    )
    staging_path = output_root / "gemm_perf.txt"
    staging_path.write_text("op,shape,latency\nmatmul,new,1.0\n", encoding="utf-8")
    staging_path.chmod(0o600)
    parquet_path = staging_path.with_suffix(".parquet")
    pq.write_table(pa.table({"op": ["matmul"], "shape": ["old"], "latency": [9.0]}), parquet_path)
    parquet_before = parquet_path.read_bytes()

    process = mp.get_context("spawn").Process(
        target=_crash_after_perf_journal_source_chmod,
        args=(str(output_root), str(checkpoint_dir)),
    )
    process.start()
    process.join(timeout=20)
    if process.is_alive():
        process.terminate()
        process.join()
        pytest.fail("collector post-journal source chmod subprocess did not exit")

    assert process.exitcode == 92
    assert stat.S_IMODE(staging_path.stat().st_mode) == 0o400
    assert (output_root / collect_mod._PERF_TRANSACTION_FILENAME).is_file()

    with pytest.raises(RuntimeError, match=r"mode changed|staging file changed"):
        collect_mod._recover_collector_provenance_transaction(
            output_root,
            backend=BACKEND,
            checkpoint_dir=str(checkpoint_dir),
        )

    assert parquet_path.read_bytes() == parquet_before
    assert staging_path.exists()
    assert _attempted(checkpoint_path) == {"case-a"}
    assert (output_root / collect_mod._PERF_TRANSACTION_FILENAME).is_file()


def test_publish_failure_retains_journal_when_restored_target_is_tampered(tmp_path, monkeypatch):
    import helper as helper_mod

    output_root = tmp_path / "out"
    output_root.mkdir()
    checkpoint_dir = tmp_path / "checkpoint"
    _write_checkpoint(
        checkpoint_dir,
        done=["gemm-case"],
        failed=[],
        attempted=["gemm-case"],
    )
    _write_checkpoint_for(
        checkpoint_dir,
        backend=BACKEND,
        full_name=f"{BACKEND}.moe",
        version="0.5.14",
        done=["moe-case"],
        failed=[],
        attempted=["moe-case"],
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
    staging_paths = [output_root / "gemm_perf.txt", output_root / "moe_perf.txt"]
    for path, shape, latency in zip(staging_paths, ("new-gemm", "new-moe"), (1.0, 2.0), strict=True):
        path.write_text(f"op,shape,latency\nmatmul,{shape},{latency}\n", encoding="utf-8")
    parquet_paths = [path.with_suffix(".parquet") for path in staging_paths]
    for path, shape, latency in zip(parquet_paths, ("old-gemm", "old-moe"), (9.0, 8.0), strict=True):
        pq.write_table(pa.table({"op": ["matmul"], "shape": [shape], "latency": [latency]}), path)

    real_rename_noreplace = helper_mod._rename_noreplace_at
    failed = False
    rollback_raced = False

    def fail_second_publish_and_race_rollback(source, target, directory_fd):
        nonlocal failed, rollback_raced
        source_path = output_root / source
        target_path = output_root / target
        if target_path == parquet_paths[1] and source_path.name.endswith(".tmp") and not failed:
            failed = True
            raise OSError("simulated second parquet publish failure")
        if source_path == parquet_paths[0] and target_path.name.endswith(".tmp") and failed and not rollback_raced:
            rollback_raced = True
            attacker = output_root / ".rollback-attacker.parquet"
            pq.write_table(
                pa.table({"op": ["matmul"], "shape": ["attacker"], "latency": [777.0]}),
                attacker,
            )
            os.replace(attacker, source_path)
        return real_rename_noreplace(source, target, directory_fd)

    monkeypatch.setattr(helper_mod, "_rename_noreplace_at", fail_second_publish_and_race_rollback)

    with pytest.raises((OSError, RuntimeError)):
        collect_mod._finalize_collector_outputs_transaction(
            output_root,
            staging_paths,
            _provenance_ctx(collections),
            [],
            backend=BACKEND,
            checkpoint_dir=str(checkpoint_dir),
            sm_version=100,
        )

    assert failed
    assert rollback_raced
    assert pq.read_table(parquet_paths[0]).to_pylist() == [{"op": "matmul", "shape": "attacker", "latency": 777.0}]
    assert (output_root / collect_mod._PERF_TRANSACTION_FILENAME).is_file()


def test_rollback_rejects_duplicate_old_claim_without_unlinking_it(tmp_path, monkeypatch):
    import helper as helper_mod

    output_root = tmp_path / "out"
    output_root.mkdir()
    checkpoint_dir = tmp_path / "checkpoint"
    _write_checkpoint(
        checkpoint_dir,
        done=["gemm-case"],
        failed=[],
        attempted=["gemm-case"],
    )
    _write_checkpoint_for(
        checkpoint_dir,
        backend=BACKEND,
        full_name=f"{BACKEND}.moe",
        version="0.5.14",
        done=["moe-case"],
        failed=[],
        attempted=["moe-case"],
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
    staging_paths = [output_root / "gemm_perf.txt", output_root / "moe_perf.txt"]
    for path, shape in zip(staging_paths, ("new-gemm", "new-moe"), strict=True):
        path.write_text(f"op,shape,latency\nmatmul,{shape},1.0\n", encoding="utf-8")
    parquet_paths = [path.with_suffix(".parquet") for path in staging_paths]
    for path, shape in zip(parquet_paths, ("old-gemm", "old-moe"), strict=True):
        pq.write_table(pa.table({"op": ["matmul"], "shape": [shape], "latency": [9.0]}), path)

    real_rename_noreplace = helper_mod._rename_noreplace_at
    failed = False

    def fail_second_publish(source, target, directory_fd):
        nonlocal failed
        source_path = output_root / source
        target_path = output_root / target
        if target_path == parquet_paths[1] and source_path.name.endswith(".tmp") and not failed:
            failed = True
            raise OSError("simulated second parquet publish failure")
        return real_rename_noreplace(source, target, directory_fd)

    real_rollback_complete = collect_mod._CollectorPerfPublicationTransaction.rollback_complete
    duplicate_claim = None

    def duplicate_claim_before_validation(transaction, publications):
        nonlocal duplicate_claim
        duplicate_claim = publications[0].target_claim
        os.link(publications[0].target, duplicate_claim)
        return real_rollback_complete(transaction, publications)

    monkeypatch.setattr(helper_mod, "_rename_noreplace_at", fail_second_publish)
    monkeypatch.setattr(
        collect_mod._CollectorPerfPublicationTransaction,
        "rollback_complete",
        duplicate_claim_before_validation,
    )

    with pytest.raises(RuntimeError, match="target claim was not cleared"):
        collect_mod._finalize_collector_outputs_transaction(
            output_root,
            staging_paths,
            _provenance_ctx(collections),
            [],
            backend=BACKEND,
            checkpoint_dir=str(checkpoint_dir),
            sm_version=100,
        )

    assert failed
    assert duplicate_claim is not None
    assert duplicate_claim.is_file()
    assert duplicate_claim.samefile(parquet_paths[0])
    assert (output_root / collect_mod._PERF_TRANSACTION_FILENAME).is_file()


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
    assert collect_mod._recover_collector_provenance_transaction(
        output_root,
        backend=BACKEND,
        checkpoint_dir=str(checkpoint_dir),
    ) == (output_root / "collection_meta.yaml")

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
    real_render = provenance.render_collection_meta
    injected_checkpoint: bytes | None = None

    def render_then_change_checkpoint(*args, **kwargs):
        nonlocal injected_checkpoint
        rendered = real_render(*args, **kwargs)
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        checkpoint[ledger] = replacement
        checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")
        injected_checkpoint = checkpoint_path.read_bytes()
        return rendered

    monkeypatch.setattr(provenance, "render_collection_meta", render_then_change_checkpoint)

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


@pytest.mark.parametrize("later_failure", ["malformed-json", "ledger-mismatch"])
def test_recovery_preflights_all_checkpoint_participants_before_normalizing_any(tmp_path, later_failure):
    output_root = tmp_path / "out"
    checkpoint_dir = tmp_path / "checkpoint"
    first = _write_checkpoint_for(
        checkpoint_dir,
        backend=BACKEND,
        full_name="sglang.mla_bmm_gen_pre",
        version="0.5.14",
        done=["case-pre"],
        failed=[],
        attempted=["case-pre"],
    )
    second = _write_checkpoint_for(
        checkpoint_dir,
        backend=BACKEND,
        full_name="sglang.mla_bmm_gen_post",
        version="0.5.14",
        done=["case-post"],
        failed=[],
        attempted=["case-post"],
    )
    staged = output_root / "mla_bmm_perf.txt"
    staged.parent.mkdir(parents=True)
    staged.write_bytes(b"owned staging")
    _write_sidecar_transaction(
        output_root,
        [(first, {"case-pre"}), (second, {"case-post"})],
        [staged],
        checkpoint_dir=checkpoint_dir,
        tagged=set(),
        pending_document=True,
    )

    first_previous = first.with_name(f".{first.name}.tmp.previous")
    first_reservation = first.with_name(f".{first.name}.tmp")
    first.replace(first_previous)
    first_reservation.write_bytes(b"future checkpoint bytes")
    tracked_first_paths = (first, first_previous, first_reservation)

    def namespace_state(paths):
        return {path.name: (path.read_bytes(), stat.S_IMODE(path.stat().st_mode)) for path in paths if path.exists()}

    first_before = namespace_state(tracked_first_paths)
    if later_failure == "malformed-json":
        second.write_bytes(b"{")
    else:
        second_document = json.loads(second.read_text(encoding="utf-8"))
        second_document["done"] = ["different-case"]
        second.write_text(json.dumps(second_document), encoding="utf-8")

    with pytest.raises(RuntimeError, match=r"checkpoint participant|checkpoint"):
        collect_mod._recover_collector_provenance_transaction(
            output_root,
            backend=BACKEND,
            checkpoint_dir=str(checkpoint_dir),
        )

    assert namespace_state(tracked_first_paths) == first_before


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
    _write_perf_event_through_logger(staged)
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

    _write_perf_event_through_logger(staged)
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
