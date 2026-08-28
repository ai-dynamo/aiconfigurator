# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for full-node collector orchestration."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from collector import fullnode, provenance
from collector.framework_manifest import CollectorRuntime

_COLLECTOR_DIR = str(Path(__file__).resolve().parents[3] / "collector")
if _COLLECTOR_DIR not in sys.path:
    sys.path.insert(0, _COLLECTOR_DIR)

import collect as collect_mod

pytestmark = pytest.mark.unit


class _Checkpoint:
    instances: ClassVar[list] = []

    def __init__(self, **_kwargs):
        self.passed = []
        self.failed = []
        self.attempted = []
        self.flushed = False
        self.__class__.instances.append(self)

    def mark_attempted(self, task_id):
        self.attempted.append(task_id)

    def mark_attempted_many(self, task_ids):
        self.attempted.extend(task_ids)

    def mark_passed(self, task_id):
        self.passed.append(task_id)

    def mark_failed(self, task_id):
        self.failed.append(task_id)

    def flush(self, force=False):
        self.flushed = force


def _logger():
    return SimpleNamespace(info=lambda *_args: None, warning=lambda *_args: None, exception=lambda *_args: None)


def test_shape_index_selects_exact_case_before_limit(monkeypatch):
    cases = [{"shape": 0}, {"shape": 1}, {"shape": 2}]
    monkeypatch.setenv("DEEPEP_LL_SHAPE_INDEX", "2")

    assert fullnode.select_cases("deepep_ll", cases, limit=1) == [{"shape": 2}]


def test_shape_index_rejects_out_of_range(monkeypatch):
    monkeypatch.setenv("DEEPEP_NORMAL_SHAPE_INDEX", "3")

    with pytest.raises(RuntimeError, match=r"DEEPEP_NORMAL_SHAPE_INDEX=3.*0\.\.1"):
        fullnode.select_cases("deepep_normal", [{"shape": 0}, {"shape": 1}], limit=None)


def test_runner_maps_reported_failure_to_exact_checkpoint_task(monkeypatch):
    cases = [
        {"hidden_size": 2048, "num_experts": 128, "topk": 8},
        {"hidden_size": 8192, "num_experts": 512, "topk": 22},
    ]

    def get_cases():
        return list(cases)

    def run_cases(*, perf_filename, limit, cases):
        assert perf_filename == "perf.txt"
        assert limit is None
        return {"succeeded": 1, "failed": [cases[1]]}

    module_name = "fake_fullnode_collector"
    monkeypatch.setitem(
        sys.modules,
        module_name,
        SimpleNamespace(__compat__=">=0.5.0", get_cases=get_cases, run_cases=run_cases),
    )
    monkeypatch.setattr(fullnode, "filter_cases", lambda values, **_kwargs: (values, []))
    _Checkpoint.instances.clear()

    errors = fullnode.collect_sglang_fullnode_op(
        {
            "name": "fake",
            "type": "deepep_ll",
            "module": module_name,
            "get_func": "get_cases",
            "run_func": "run_cases",
            "perf_filename": "perf.txt",
        },
        runtime_version="0.5.12",
        limit=None,
        shuffle=False,
        shuffle_seed=42,
        backend="sglang",
        resume_options=None,
        model_path=None,
        case_plan=None,
        sm_version=100,
        case_filters=None,
        get_test_cases_for_model=lambda get_func, _model_path: get_func(),
        resume_checkpoint_cls=_Checkpoint,
        logger=_logger(),
    )

    checkpoint = _Checkpoint.instances[-1]
    assert len(checkpoint.attempted) == 2
    assert len(checkpoint.passed) == 1
    assert len(checkpoint.failed) == 1
    assert checkpoint.flushed is True
    assert len(errors) == 1
    assert errors[0]["error_type"] == "FullNodeCaseFailure"
    assert errors[0]["task_params"] == str(cases[1])


def test_runner_fails_closed_when_failure_list_is_missing(monkeypatch):
    case = {"hidden_size": 2048, "num_experts": 128, "topk": 8}
    module_name = "fake_fullnode_collector_missing_failures"
    monkeypatch.setitem(
        sys.modules,
        module_name,
        SimpleNamespace(
            __compat__=">=0.5.0",
            get_cases=lambda: [case],
            run_cases=lambda **_kwargs: None,
        ),
    )
    monkeypatch.setattr(fullnode, "filter_cases", lambda values, **_kwargs: (values, []))
    _Checkpoint.instances.clear()

    errors = fullnode.collect_sglang_fullnode_op(
        {
            "name": "fake",
            "type": "deepep_ll",
            "module": module_name,
            "get_func": "get_cases",
            "run_func": "run_cases",
            "perf_filename": "perf.txt",
        },
        runtime_version="0.5.12",
        limit=None,
        shuffle=False,
        shuffle_seed=42,
        backend="sglang",
        resume_options=None,
        model_path=None,
        case_plan=None,
        sm_version=100,
        case_filters=None,
        get_test_cases_for_model=lambda get_func, _model_path: get_func(),
        resume_checkpoint_cls=_Checkpoint,
        logger=_logger(),
    )

    assert len(_Checkpoint.instances[-1].attempted) == 1
    assert len(_Checkpoint.instances[-1].failed) == 1
    assert errors[0]["error_type"] == "FullNodeCollectionFailure"
    assert "must return" in errors[0]["error_message"]


REAL_FULLNODE_MODULE = "collector.wideep.sglang.collect_deepep_ll"


def _real_collection() -> dict:
    return {
        "name": "sglang",
        "type": "deepep_ll",
        "module": REAL_FULLNODE_MODULE,
        "get_func": "get_cases",
        "run_func": "run_cases",
        "perf_filename": "wideep_deepep_ll_perf.txt",
    }


def _run_with_real_checkpoint(
    monkeypatch,
    tmp_path,
    *,
    cases,
    failed_cases=(),
    resume=False,
    retry_failed=False,
):
    calls = []

    def run_cases(*, perf_filename, limit, cases):
        calls.append(list(cases))
        return {"succeeded": len(cases) - len(failed_cases), "failed": list(failed_cases)}

    monkeypatch.setitem(
        sys.modules,
        REAL_FULLNODE_MODULE,
        SimpleNamespace(__compat__=">=0.5.0", get_cases=lambda: list(cases), run_cases=run_cases),
    )
    monkeypatch.setattr(fullnode, "filter_cases", lambda values, **_kwargs: (values, []))
    checkpoint_dir = tmp_path / "checkpoint"
    errors = fullnode.collect_sglang_fullnode_op(
        _real_collection(),
        runtime_version="0.5.14",
        limit=None,
        shuffle=False,
        shuffle_seed=42,
        backend="sglang",
        resume_options={
            "checkpoint_dir": str(checkpoint_dir),
            "resume": resume,
            "retry_failed": retry_failed,
        },
        model_path=None,
        case_plan=None,
        sm_version=100,
        case_filters=None,
        get_test_cases_for_model=lambda get_func, _model_path: get_func(),
        resume_checkpoint_cls=collect_mod.ResumeCheckpoint,
        logger=_logger(),
    )
    checkpoint_path = checkpoint_dir / "sglang" / "sglang.deepep_ll.json"
    return errors, calls, checkpoint_path


def _checkpoint_sets(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    return {key: set(data.get(key, [])) for key in ("done", "failed", "attempted")}


def _finalize_pending_fullnode_event(tmp_path, checkpoint_path):
    pending = _checkpoint_sets(checkpoint_path)["attempted"]
    output_root = tmp_path / "out"
    output_root.mkdir()
    parquet_path = output_root / "wideep_deepep_ll_perf.parquet"
    pq.write_table(pa.table({"shape": [1, 2], "latency": [1.0, 2.0]}), parquet_path)
    runtime = CollectorRuntime(
        framework="sglang",
        version="0.5.14",
        images={"default": "lmsysorg/sglang:v0.5.14@sha256:" + "0" * 64},
    )
    collect_mod._write_collector_provenance(
        output_root,
        [parquet_path],
        {
            "installed_version": runtime.version,
            "runtime": runtime,
            "sm_version": 100,
            "collections": [_real_collection()],
        },
        run_errors=[],
        backend="sglang",
        checkpoint_dir=str(tmp_path / "checkpoint"),
        finalization_info={parquet_path.resolve(): collect_mod.PerfFinalizationInfo(new_rows=2, merged_existing=False)},
    )
    table = yaml.safe_load((output_root / "collection_meta.yaml").read_text(encoding="utf-8"))["tables"][
        "wideep_deepep_ll_perf"
    ]
    return pending, table


def test_real_checkpoint_fresh_fullnode_event_finalizes_exact_selected_set(monkeypatch, tmp_path):
    cases = [{"shape": 1}, {"shape": 2}]
    errors, calls, checkpoint_path = _run_with_real_checkpoint(monkeypatch, tmp_path, cases=cases)

    assert errors == []
    assert calls == [cases]
    before = _checkpoint_sets(checkpoint_path)
    assert before["done"] == before["attempted"]
    assert len(before["attempted"]) == 2

    pending, table = _finalize_pending_fullnode_event(tmp_path, checkpoint_path)
    assert pending == before["attempted"]
    assert table["case_plan_hash"] == provenance.case_plan_hash(sorted(before["attempted"]))
    assert _checkpoint_sets(checkpoint_path)["attempted"] == set()


def test_real_checkpoint_interrupted_resume_unions_old_and_new_fullnode_cases(monkeypatch, tmp_path):
    cases = [{"shape": 1}, {"shape": 2}]
    _, _, checkpoint_path = _run_with_real_checkpoint(monkeypatch, tmp_path, cases=[cases[0]])

    errors, calls, _ = _run_with_real_checkpoint(monkeypatch, tmp_path, cases=cases, resume=True)

    assert errors == []
    assert calls == [[cases[1]]]
    state = _checkpoint_sets(checkpoint_path)
    assert state["done"] == state["attempted"]
    assert len(state["attempted"]) == 2


def test_real_checkpoint_retry_after_closed_event_attests_only_retry(monkeypatch, tmp_path):
    cases = [{"shape": 1}, {"shape": 2}]
    _, _, checkpoint_path = _run_with_real_checkpoint(
        monkeypatch,
        tmp_path,
        cases=cases,
        failed_cases=[cases[1]],
    )
    _pending, _table = _finalize_pending_fullnode_event(tmp_path, checkpoint_path)
    assert _checkpoint_sets(checkpoint_path)["attempted"] == set()

    _, calls, _ = _run_with_real_checkpoint(
        monkeypatch,
        tmp_path,
        cases=cases,
        resume=True,
        retry_failed=True,
    )

    assert calls == [[cases[1]]]
    state = _checkpoint_sets(checkpoint_path)
    assert len(state["attempted"]) == 1
    assert state["attempted"] <= state["done"]
    assert state["failed"] == set()


def test_real_checkpoint_zero_work_preserves_pending_and_failure_marks_all_selected(monkeypatch, tmp_path):
    cases = [{"shape": 1}, {"shape": 2}]
    _, _, checkpoint_path = _run_with_real_checkpoint(
        monkeypatch,
        tmp_path,
        cases=cases,
        failed_cases=cases,
    )
    pending = _checkpoint_sets(checkpoint_path)["attempted"]

    errors, calls, _ = _run_with_real_checkpoint(monkeypatch, tmp_path, cases=cases, resume=True)

    assert calls == []
    assert _checkpoint_sets(checkpoint_path)["attempted"] == pending
    assert len(_checkpoint_sets(checkpoint_path)["failed"]) == 2
    assert any(error["error_type"] == "UnresolvedFailures" for error in errors)
