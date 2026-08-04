# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for full-node collector orchestration."""

import sys
from types import SimpleNamespace
from typing import ClassVar

import pytest

from collector import fullnode

pytestmark = pytest.mark.unit


class _Checkpoint:
    instances: ClassVar[list] = []

    def __init__(self, **_kwargs):
        self.passed = []
        self.failed = []
        self.flushed = False
        self.__class__.instances.append(self)

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

    assert len(_Checkpoint.instances[-1].failed) == 1
    assert errors[0]["error_type"] == "FullNodeCollectionFailure"
    assert "must return" in errors[0]["error_message"]
