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
    return [{"name": BACKEND, "type": OP_TYPE, "module": REAL_MODULE, "perf_filename": f"{table}.txt"}]


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
        "collections": collections,
    }


def _write_checkpoint(checkpoint_dir: Path, *, done: list[str], failed: list[str]) -> Path:
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
                "sm_version": None,
                "updated_at": "2026-07-20T00:00:00",
                "done": sorted(done),
                "failed": sorted(failed),
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


def test_writes_sidecar_with_rows_case_plan_hash_status_and_collector_ref(tmp_path):
    output_root = tmp_path / "out"
    parquet_path = output_root / "gemm_perf.parquet"
    _write_parquet(parquet_path, [{"op": "matmul", "latency": 1.0}, {"op": "matmul", "latency": 2.0}])

    checkpoint_dir = tmp_path / "checkpoint"
    _write_checkpoint(checkpoint_dir, done=["case-a", "case-b"], failed=[])

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
    with pytest.raises(RuntimeError, match="other_perf"):
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


def test_runtime_change_allows_every_old_table_to_be_replaced(tmp_path):
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
    assert doc["runtime"] == {
        "framework": "sglang",
        "version": "0.5.14",
        "image": "lmsysorg/sglang:v0.5.14",
        "image_digest": "sha256:" + "0" * 64,
    }
    assert set(doc["tables"]) == {"gemm_perf", "other_perf"}
    assert all("collections" not in table for table in doc["tables"].values())


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
