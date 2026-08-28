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

import json
import logging
import re
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

pytestmark = pytest.mark.unit

collect_mod.logger = logging.getLogger("test_collect_provenance_writer")

# A real, already-hash_closures.yaml-covered module — using it lets the writer's
# real load_closures()/collector_hash() calls run unmocked against the real repo
# tree, instead of needing to fabricate a fake repo layout.
REAL_MODULE = "collector.sglang.collect_gemm"
BACKEND = "sglang"
OP_TYPE = "gemm"
FULL_NAME = f"{BACKEND}.{OP_TYPE}"  # collection["name"] + "." + collection["type"]


def _collections(table: str = "gemm_perf") -> list[dict]:
    return [
        {
            "name": BACKEND,
            "type": OP_TYPE,
            "module": REAL_MODULE,
            "run_func": "run",
            "perf_filename": f"{table}.txt",
        }
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


def _xpu_provenance_ctx(collections: list[dict]) -> dict:
    runtime = CollectorRuntime(
        framework="vllm_xpu",
        version="0.26.0",
        images={"default": "vllm/vllm-openai-xpu:v0.26.0@sha256:" + "5" * 64},
        data_backend="vllm",
    )
    return {
        "framework": runtime.framework,
        "installed_version": "0.26.0+xpu",
        "runtime": runtime,
        "sm_version": None,
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
                "run_func": "run",
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
                "run_func": "run",
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


def _finalization_info_for(*parquet_paths: Path) -> dict[Path, collect_mod.PerfFinalizationInfo]:
    return {
        path.resolve(): collect_mod.PerfFinalizationInfo(
            new_rows=pq.read_metadata(path).num_rows,
            merged_existing=False,
        )
        for path in parquet_paths
    }


def _attempted(checkpoint_path: Path) -> set[str]:
    return set(json.loads(checkpoint_path.read_text(encoding="utf-8")).get("attempted", []))


def _write_sidecar_transaction(
    output_root: Path,
    participants: list[tuple[Path, set[str]]],
    staging_paths: list[Path],
    *,
    tagged: set[Path],
    pending_document: bool,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    meta_path = output_root / "collection_meta.yaml"
    meta_path.write_text("same sidecar bytes\n", encoding="utf-8")
    if pending_document:
        (output_root / collect_mod._SIDECAR_STAGING_FILENAME).write_bytes(meta_path.read_bytes())
    transaction_id = "test-transaction"
    for checkpoint_path, attempted_case_ids in participants:
        if checkpoint_path in tagged:
            collect_mod._tag_checkpoint_sidecar_transaction(
                checkpoint_path,
                attempted_case_ids,
                transaction_id,
            )
    transaction = {
        "schema": collect_mod._SIDECAR_TRANSACTION_SCHEMA,
        "transaction_id": transaction_id,
        "sidecar_path": str(meta_path.resolve()),
        "sidecar_digest": collect_mod._sidecar_digest(meta_path),
        "checkpoints": [
            {"path": str(checkpoint_path), "attempted": sorted(attempted_case_ids)}
            for checkpoint_path, attempted_case_ids in participants
        ],
        "staging_paths": [str(path.resolve()) for path in staging_paths],
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

    with pytest.raises(RuntimeError, match="gemm_perf"):
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
    assert collect_mod.finalize_perf_files([perf_path], finalization_info=finalization_info) == [parquet_path]
    assert pq.read_metadata(parquet_path).num_rows == 2
    assert pq.read_table(parquet_path).to_pylist()[-1] == {"op": "softmax", "latency": 3.0}
    assert finalization_info[parquet_path.resolve()] == collect_mod.PerfFinalizationInfo(
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
    assert collect_mod.finalize_perf_files([perf_path], finalization_info=finalization_info) == [parquet_path]

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
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_path = _write_checkpoint(
        checkpoint_dir,
        done=["case-old"],
        failed=[],
        attempted=["case-deduped-a", "case-deduped-a", "case-deduped-b"],
    )

    collect_mod._write_collector_provenance(
        output_root,
        [parquet_path],
        _provenance_ctx(_collections()),
        run_errors=[],
        backend=BACKEND,
        checkpoint_dir=str(checkpoint_dir),
        finalization_info={parquet_path.resolve(): collect_mod.PerfFinalizationInfo(new_rows=0, merged_existing=True)},
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


def test_recovery_finishes_partial_prepare_when_pending_sidecar_matches_live_bytes(tmp_path):
    output_root = tmp_path / "out"
    checkpoint_dir = tmp_path / "checkpoint"
    first = _write_checkpoint_for(
        checkpoint_dir,
        backend=BACKEND,
        full_name="sglang.gemm",
        version="0.5.14",
        done=["case-a"],
        failed=[],
        attempted=["case-a"],
    )
    second = _write_checkpoint_for(
        checkpoint_dir,
        backend=BACKEND,
        full_name="sglang.gemm_extra",
        version="0.5.14",
        done=["case-b"],
        failed=[],
        attempted=["case-b"],
    )
    staged = output_root / "shared_perf.txt"
    staged.parent.mkdir(parents=True)
    staged.write_text("op,latency\nshared,1.0\n", encoding="utf-8")
    participants = [(first, {"case-a"}), (second, {"case-b"})]
    _write_sidecar_transaction(
        output_root,
        participants,
        [staged],
        tagged={first},
        pending_document=True,
    )

    assert collect_mod._recover_collector_provenance_transaction(output_root) == (output_root / "collection_meta.yaml")

    assert _attempted(first) == set()
    assert _attempted(second) == set()
    assert not staged.exists()
    assert not (output_root / collect_mod._SIDECAR_STAGING_FILENAME).exists()
    assert not (output_root / collect_mod._SIDECAR_TRANSACTION_FILENAME).exists()


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
        tagged=set(),
        pending_document=False,
    )

    assert collect_mod._recover_collector_provenance_transaction(output_root) is None

    assert not staged.exists()
    assert not (output_root / collect_mod._SIDECAR_TRANSACTION_FILENAME).exists()


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
    parquet_path = output_root / "shared_perf.parquet"
    _write_parquet(parquet_path, [{"op": "shared", "latency": 1.0}])
    checkpoint_dir = tmp_path / "checkpoint"
    collections = [
        {
            "name": BACKEND,
            "type": "gemm",
            "module": REAL_MODULE,
            "run_func": "run",
            "perf_filename": "shared_perf.txt",
        },
        {
            "name": BACKEND,
            "type": "gemm_extra",
            "module": REAL_MODULE,
            "run_func": "run",
            "perf_filename": "shared_perf.txt",
        },
    ]
    first = _write_checkpoint_for(
        checkpoint_dir,
        backend=BACKEND,
        full_name="sglang.gemm",
        version="0.5.14",
        done=["case-a"],
        failed=[],
        attempted=["case-a"],
    )
    second = _write_checkpoint_for(
        checkpoint_dir,
        backend=BACKEND,
        full_name="sglang.gemm_extra",
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

    table = yaml.safe_load((output_root / "collection_meta.yaml").read_text(encoding="utf-8"))["tables"]["shared_perf"]
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
        ("schema", 999),
        ("backend", "trtllm"),
        ("module", "sglang.other"),
        ("run_func", "other_run"),
        ("framework_version", "0.5.13"),
        ("sm_version", 90),
    ],
)
def test_pending_resume_ignores_checkpoint_with_mismatched_runtime_identity(
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

    assert (
        collect_mod._pending_resume_perf_outputs(
            output_root,
            _provenance_ctx(_collections()),
            backend=BACKEND,
            checkpoint_dir=str(checkpoint_dir),
            sm_version=100,
        )
        == []
    )


def test_provenance_writer_rejects_mismatched_checkpoint_identity(tmp_path):
    output_root = tmp_path / "out"
    parquet_path = output_root / "gemm_perf.parquet"
    _write_parquet(parquet_path, [{"op": "matmul", "latency": 1.0}])
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_path = _write_checkpoint(checkpoint_dir, done=["stale-case"], failed=[])
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    checkpoint["framework_version"] = "0.5.13"
    checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")

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
    assert collect_mod.finalize_perf_files([perf_path], finalization_info=finalization_info) == [parquet_path]
    assert pq.read_table(parquet_path).to_pylist() == [{"shape": "new-shape", "latency": 2.0}]
    assert finalization_info[parquet_path.resolve()] == collect_mod.PerfFinalizationInfo(
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
        "tables": {
            "gemm_perf": {"status": "complete"},
            "other_perf": {"status": "complete"},
        },
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
    provenance.write_collection_meta(
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
    provenance.write_collection_meta(
        output_root,
        {
            "framework": "sglang",
            "version": "0.5.13",
            "image": "lmsysorg/sglang:v0.5.13",
            "image_digest": "sha256:" + "1" * 64,
        },
        {
            "gemm_perf": {"status": "complete"},
            "other_perf": {"status": "complete"},
        },
    )

    perf_paths = [output_root / "gemm_perf.txt", output_root / "other_perf.txt"]
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
    with pytest.raises(RuntimeError, match="different runtime identity"):
        collect_mod._write_collector_provenance(
            output_root,
            parquet_paths,
            _provenance_ctx([*_collections("gemm_perf"), *_collections("other_perf")]),
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
