# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

from collector.case_generator import get_common_mhc_test_cases
from collector.model_cases import build_collection_case_plan
from collector.registry_types import PerfFile
from collector.trtllm.registry import REGISTRY

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[4]


def _load_module_with_torch_stub(monkeypatch):
    """Import the collector with a stub torch so case population is testable
    without a GPU env (torch is only used inside run-path functions)."""
    monkeypatch.setitem(sys.modules, "torch", ModuleType("torch"))
    module_path = REPO_ROOT / "collector" / "trtllm" / "collect_mhc_module.py"
    spec = importlib.util.spec_from_file_location("trtllm_mhc_target", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_registry_entry_wires_mhc_module():
    entries = {entry.op: entry for entry in REGISTRY}
    entry = entries["mhc_module"]
    assert entry.module == "collector.trtllm.collect_mhc_module"
    assert entry.get_func == "get_mhc_module_test_cases"
    assert entry.run_func == "run_mhc_module_worker"
    assert entry.perf_filename == PerfFile.MHC_MODULE
    # New collector stays default-open: SM markers only with probe evidence
    # (failure_handling.md decision tree).
    assert entry.unverified is False
    assert entry.unverified_sms == ()


def test_trtllm_dsv4_plan_schedules_mhc_module():
    plan = build_collection_case_plan(backend="trtllm", model_path="sgl-project/DeepSeek-V4-Pro-FP8")
    assert "mhc_module" in plan.selected_ops
    assert "mhc_module" in plan.ops


def test_mhc_case_population_dedups_model_aliases(monkeypatch):
    module = _load_module_with_torch_stub(monkeypatch)

    cases = module.get_mhc_module_test_cases()
    ids = [case["id"] for case in cases]
    assert len(ids) == len(set(ids)), "duplicate case ids"

    # Physical dedup: 4 model paths collapse to 2 geometries x 2 phases; the
    # full sweep is (phase, hidden, hc_mult) x num_tokens.
    keys = {(case["params"][0], case["params"][2], case["params"][3]) for case in cases}
    assert keys == {
        ("pre", 4096, 4),
        ("pre", 7168, 4),
        ("post", 4096, 4),
        ("post", 7168, 4),
    }

    expected_tokens = {
        (case.phase, case.hidden_size, case.hc_mult): case.num_tokens_list for case in get_common_mhc_test_cases()
    }
    assert len(cases) == sum(len(tokens) for tokens in expected_tokens.values())
    for case in cases:
        phase, num_tokens, hidden_size, hc_mult = case["params"]
        assert num_tokens in expected_tokens[(phase, hidden_size, hc_mult)]


def test_mhc_worker_params_match_run_signature(monkeypatch):
    module = _load_module_with_torch_stub(monkeypatch)

    captured = {}

    def fake_run(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(module, "run_mhc_module", fake_run)
    case = module.get_mhc_module_test_cases()[0]
    module.run_mhc_module_worker(*case["params"], perf_filename="/tmp/out/mhc_module_perf.txt")

    assert captured["ops"] == [case["params"][0]]
    assert captured["num_tokens_cases"] == [case["params"][1]]
    assert captured["hidden_size"] == case["params"][2]
    assert captured["hc_mult"] == case["params"][3]
    assert captured["perf_filename"] == "mhc_module_perf.txt"
    assert captured["output_path"] == "/tmp/out"
