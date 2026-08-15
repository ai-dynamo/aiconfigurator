# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
import os
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_functions(source_path: Path, *names: str, namespace: dict | None = None) -> dict:
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    loaded = dict(namespace or {})
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(source_path), "exec"), loaded)
    return loaded


def test_moe_resolves_the_worker_current_cuda_device():
    source_path = REPO_ROOT / "collector" / "sglang" / "collect_moe.py"
    calls = []
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(current_device=lambda: 3),
        device=lambda value: calls.append(value) or value,
    )
    resolver = _load_functions(source_path, "_current_cuda_device", namespace={"torch": fake_torch})[
        "_current_cuda_device"
    ]

    assert resolver() == "cuda:3"
    assert calls == ["cuda:3"]


def test_moe_benchmark_calls_the_explicit_device_resolver():
    source_path = REPO_ROOT / "collector" / "sglang" / "collect_moe.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    benchmark = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "benchmark_config"
    )
    assignments = [node for node in benchmark.body if isinstance(node, ast.Assign)]

    assert any(
        any(isinstance(target, ast.Name) and target.id == "device" for target in assignment.targets)
        and isinstance(assignment.value, ast.Call)
        and isinstance(assignment.value.func, ast.Name)
        and assignment.value.func.id == "_current_cuda_device"
        for assignment in assignments
    )


def test_allreduce_stamps_true_and_overwrites_stale_true(monkeypatch):
    source_path = REPO_ROOT / "collector" / "network" / "collect_all_reduce.py"
    stamp = _load_functions(source_path, "_stamp_measure_power_env", namespace={"os": os})["_stamp_measure_power_env"]

    monkeypatch.setenv("COLLECTOR_MEASURE_POWER", "false")
    stamp(True)
    assert os.environ["COLLECTOR_MEASURE_POWER"] == "true"
    stamp(False)
    assert os.environ["COLLECTOR_MEASURE_POWER"] == "false"

    source = source_path.read_text(encoding="utf-8")
    assert "_stamp_measure_power_env(args.measure_power)" in source


def test_dsa_context_power_sampling_replays_the_same_target(monkeypatch):
    source_path = REPO_ROOT / "collector" / "sglang" / "collect_mla_module.py"
    calls = {"target": 0, "sync": 0, "device": []}

    class Monitor:
        def stop_sampling(self):
            return {"power": 420.0, "power_limit": 1000.0}

    @contextmanager
    def power_monitoring_only(device):
        calls["device"].append(device)
        yield Monitor()

    fake_torch = SimpleNamespace(
        device=lambda value: f"resolved:{value}",
        no_grad=nullcontext,
        cuda=SimpleNamespace(synchronize=lambda: calls.__setitem__("sync", calls["sync"] + 1)),
    )
    measure = _load_functions(
        source_path,
        "_measure_back_to_back_power",
        namespace={"os": os, "torch": fake_torch, "power_monitoring_only": power_monitoring_only},
    )["_measure_back_to_back_power"]
    monkeypatch.setenv("COLLECTOR_POWER_MIN_DURATION", "0.01")

    def target():
        calls["target"] += 1

    result = measure("cuda:2", target, avg_time_ms=2.0, minimum_runs=3)

    assert result == {"power": 420.0, "power_limit": 1000.0}
    assert calls == {"target": 6, "sync": 1, "device": ["resolved:cuda:2"]}

    source = source_path.read_text(encoding="utf-8")
    assert "power_stats = _measure_back_to_back_power(device, power_target, avg_time_ms, num_iterations)" in source
    assert "power_stats=power_stats" in source


def test_sparse_collectors_forward_power_to_their_writers():
    dsv4_source = REPO_ROOT / "collector" / "sglang" / "deepseekv4_sparse_modules.py"
    glm5_source = REPO_ROOT / "collector" / "sglang" / "glm5_dsa_sparse_modules.py"
    dsv4_text = dsv4_source.read_text(encoding="utf-8")
    glm5_text = glm5_source.read_text(encoding="utf-8")

    assert "for score_mode, latency_ms, power_stats in results:" in dsv4_text
    assert "for score_mode, latency_ms, power_stats in results:" in glm5_text
    assert "power_stats=power_stats" in dsv4_text
    assert "power_stats=power_stats" in glm5_text


def test_glm5_chunked_topk_uses_latency_weighted_power_aggregation():
    source = (REPO_ROOT / "collector" / "sglang" / "glm5_dsa_sparse_modules.py").read_text(encoding="utf-8")

    assert 'mode_power_measurements = {"flat": [], "top_last": []}' in source
    assert 'mode_power_measurements[mode].append((measured["latency_ms"], measured.get("power_stats")))' in source
    assert "aggregate_latency_weighted_power(mode_power_measurements[mode])" in source


def test_structural_zero_work_uses_the_explicit_zero_power_contract():
    dsv4_source = (REPO_ROOT / "collector" / "sglang" / "deepseekv4_sparse_modules.py").read_text(encoding="utf-8")
    glm5_source = (REPO_ROOT / "collector" / "sglang" / "glm5_dsa_sparse_modules.py").read_text(encoding="utf-8")

    assert "stats = zero_work_power_stats(torch.device(device))" in dsv4_source
    assert "stats = zero_work_power_stats(torch.device(device))" in glm5_source
