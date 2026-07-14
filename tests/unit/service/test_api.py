# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import ClassVar

import pytest

fastapi = pytest.importorskip("fastapi")
pandas = pytest.importorskip("pandas")

from fastapi.testclient import TestClient

from aiconfigurator.cli.api import CLIResult
from aiconfigurator.service.app import create_app


def _client():
    return TestClient(create_app())


@pytest.fixture(autouse=True)
def _run_store(monkeypatch, tmp_path):
    monkeypatch.setenv("AIC_SERVICE_RUN_STORE", str(tmp_path / "runs.json"))


def test_options_returns_quant_modes(monkeypatch):
    class FakeDatabase:
        supported_quant_mode: ClassVar = {"gemm": ["bfloat16", "fp8"], "moe": ["bfloat16"]}

    monkeypatch.setattr(
        "aiconfigurator.service.app.get_default_models",
        lambda: {"Qwen/Qwen3-8B", "Qwen/Qwen3-32B"},
    )
    monkeypatch.setattr(
        "aiconfigurator.service.app.get_supported_databases",
        lambda: {"h200_sxm": {"vllm": ["0.19.0"]}},
    )
    monkeypatch.setattr("aiconfigurator.service.app.get_database", lambda system, backend, version: FakeDatabase())

    response = _client().get("/api/v1/options?system=h200_sxm&backend=vllm&version=0.19.0")

    assert response.status_code == 200
    payload = response.json()
    assert payload["systems"] == ["h200_sxm"]
    assert payload["quant_modes"]["gemm"] == ["bfloat16", "fp8"]


def test_estimate_returns_serialized_cli_result(monkeypatch):
    monkeypatch.setattr("aiconfigurator.service.app.get_database", lambda **kwargs: object())
    best_df = pandas.DataFrame([{"parallel": "tp4pp1", "tokens/s/gpu": 123.4, "ttft": 250.0, "tpot": 8.0}])
    disagg_df = pandas.DataFrame([{"parallel": "tp2pp1", "tokens/s/gpu": 111.1, "ttft": 280.0, "tpot": 10.0}])
    pareto_df = pandas.DataFrame([{"parallel": "tp4pp1", "tokens/s/gpu": 123.4}])
    result = CLIResult(
        chosen_exp="agg",
        best_configs={"agg": best_df, "disagg": disagg_df},
        pareto_fronts={"agg": pareto_df, "disagg": disagg_df},
        best_throughputs={"agg": 123.4, "disagg": 111.1},
        best_latencies={"agg": {"ttft": 250.0, "tpot": 8.0}, "disagg": {"ttft": 280.0, "tpot": 10.0}},
        task_configs={},
    )
    monkeypatch.setattr("aiconfigurator.service.app.cli_default", lambda **kwargs: result)

    response = _client().post(
        "/api/v1/estimate",
        json={
            "model_path": "Qwen/Qwen3-8B",
            "total_gpus": 8,
            "system": "h200_sxm",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["chosen_exp"] == "agg"
    assert payload["recommended_mode"] == "agg"
    assert payload["mode_comparison"]["agg"]["top1"]["parallel"] == "tp4pp1"
    assert payload["mode_comparison"]["disagg"]["top1"]["parallel"] == "tp2pp1"
    assert payload["best_configs"]["agg"][0]["parallel"] == "tp4pp1"
    assert payload["normalized_input"]["model_path"] == "Qwen/Qwen3-8B"


def test_generate_k8s_deployment_direct_params(monkeypatch):
    monkeypatch.setattr(
        "aiconfigurator.service.app.prepare_generator_params",
        lambda config_path, overrides, backend: {
            "ServiceConfig": {"model_path": "Qwen/Qwen3-8B"},
            "K8sConfig": {"k8s_namespace": "ai-platform"},
        },
    )
    monkeypatch.setattr(
        "aiconfigurator.service.app.generate_backend_artifacts",
        lambda **kwargs: {
            "k8s_deploy.yaml": "kind: Deployment\nmetadata:\n  name: demo\n",
            "run.sh": "#!/bin/bash\necho hello\n",
        },
    )

    response = _client().post(
        "/api/v1/generate-k8s-deployment",
        json={
            "backend": "vllm",
            "direct_generator_params": {
                "ServiceConfig": {"model_path": "Qwen/Qwen3-8B"},
                "K8sConfig": {"k8s_namespace": "ai-platform"},
            },
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["artifact_name"] == "k8s_deploy.yaml"
    assert "kind: Deployment" in payload["content"]
    assert "run.sh" in payload["extras"]
    assert payload["jid"].startswith("aic-")
    assert payload["dgd_id"] == f"{payload['jid']}:direct:0"

    run_response = _client().get(f"/api/v1/runs/{payload['jid']}")
    assert run_response.status_code == 200
    assert run_response.json()["payload"]["content"] == payload["content"]


def test_delete_k8s_deployment_removes_deployed_record(monkeypatch):
    monkeypatch.setattr(
        "aiconfigurator.service.app.prepare_generator_params",
        lambda config_path, overrides, backend: {
            "ServiceConfig": {"model_path": "Qwen/Qwen3-8B"},
            "K8sConfig": {"k8s_namespace": "ai-platform", "name": "demo"},
        },
    )
    monkeypatch.setattr(
        "aiconfigurator.service.app.generate_backend_artifacts",
        lambda **kwargs: {
            "k8s_deploy.yaml": "kind: Deployment\nmetadata:\n  name: demo\n  namespace: ai-platform\n",
        },
    )
    monkeypatch.setattr("aiconfigurator.service.app.shutil.which", lambda command: "/usr/bin/kubectl")

    calls = []

    class Result:
        returncode = 0
        stdout = "ok"
        stderr = ""

    def fake_run_kubectl(args, stdin=None, timeout=180):
        calls.append((args, stdin, timeout))
        return Result()

    monkeypatch.setattr("aiconfigurator.service.app._run_kubectl", fake_run_kubectl)

    client = _client()
    generated = client.post(
        "/api/v1/generate-k8s-deployment",
        json={
            "backend": "vllm",
            "direct_generator_params": {
                "ServiceConfig": {"model_path": "Qwen/Qwen3-8B"},
                "K8sConfig": {"k8s_namespace": "ai-platform", "name": "demo"},
            },
        },
    ).json()

    apply_response = client.post(
        "/api/v1/apply-k8s-deployment",
        json={
            "content": generated["content"],
            "namespace": "ai-platform",
            "jid": generated["jid"],
            "dgd_id": generated["dgd_id"],
            "mode": "direct",
            "row_index": 0,
        },
    )
    assert apply_response.status_code == 200
    assert client.get("/api/v1/deployment-records").json()["records"][0]["jid"] == generated["jid"]

    delete_response = client.post(
        "/api/v1/delete-k8s-deployment",
        json={
            "content": generated["content"],
            "namespace": "ai-platform",
            "jid": generated["jid"],
            "dgd_id": generated["dgd_id"],
            "mode": "direct",
            "row_index": 0,
        },
    )

    assert delete_response.status_code == 200
    assert delete_response.json()["deleted"] is True
    assert calls[-1][0] == ["kubectl", "delete", "-f", "-", "--ignore-not-found=true"]
    assert client.get("/api/v1/deployment-records").json()["records"] == []

    run_payload = client.get(f"/api/v1/runs/{generated['jid']}").json()["payload"]
    assert "deployed_dgd" not in run_payload
    assert run_payload["deleted_dgd"]["dgd_id"] == generated["dgd_id"]


def test_generate_k8s_deployment_accepts_structured_k8s_overrides(monkeypatch):
    captured = {}

    def _prepare(config_path, overrides, backend):
        captured["overrides"] = overrides
        return {"K8sConfig": overrides.get("K8sConfig", {})}

    monkeypatch.setattr("aiconfigurator.service.app.prepare_generator_params", _prepare)
    monkeypatch.setattr(
        "aiconfigurator.service.app.generate_backend_artifacts",
        lambda **kwargs: {"k8s_deploy.yaml": "kind: Deployment\n"},
    )

    response = _client().post(
        "/api/v1/generate-k8s-deployment",
        json={
            "backend": "vllm",
            "direct_generator_params": {"ServiceConfig": {"model_path": "Qwen/Qwen3-8B"}},
            "generator_overrides": {
                "k8s_config": {
                    "name": "dynamo-agg-test",
                    "frontend_node_selector": {"kubernetes.io/hostname": "kubeflow-master"},
                    "worker_extra_envs": [{"name": "HOME", "value": "/tmp"}],
                }
            },
        },
    )

    assert response.status_code == 200
    assert captured["overrides"]["K8sConfig"]["name"] == "dynamo-agg-test"
    assert captured["overrides"]["K8sConfig"]["frontend_node_selector"]["kubernetes.io/hostname"] == "kubeflow-master"
    assert captured["overrides"]["K8sConfig"]["worker_extra_envs"][0]["name"] == "HOME"


def test_generate_k8s_benchmark_from_estimate(monkeypatch):
    monkeypatch.setattr("aiconfigurator.service.app.get_database", lambda **kwargs: object())
    best_df = pandas.DataFrame([{"parallel": "tp8pp1", "tokens/s/gpu": 321.0, "ttft": 210.0, "tpot": 9.0}])
    result = CLIResult(
        chosen_exp="agg",
        best_configs={"agg": best_df},
        pareto_fronts={"agg": best_df},
        best_throughputs={"agg": 321.0},
        best_latencies={"agg": {"ttft": 210.0, "tpot": 9.0}},
        task_configs={"agg": object()},
    )
    monkeypatch.setattr("aiconfigurator.service.app.cli_default", lambda **kwargs: result)
    monkeypatch.setattr(
        "aiconfigurator.service.app.task_config_to_generator_config",
        lambda task_config, row, generator_overrides=None: {"BenchConfig": {"model": "Qwen/Qwen3-8B"}},
    )
    monkeypatch.setattr(
        "aiconfigurator.service.app.generate_backend_artifacts",
        lambda **kwargs: {
            "k8s_bench.yaml": "kind: Job\nmetadata:\n  name: bench\n",
            "bench_run.sh": "#!/bin/bash\necho bench\n",
        },
    )

    response = _client().post(
        "/api/v1/generate-k8s-benchmark",
        json={
            "backend": "trtllm",
            "estimate_request": {
                "model_path": "Qwen/Qwen3-8B",
                "total_gpus": 8,
                "system": "h200_sxm",
            },
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["artifact_name"] == "k8s_bench.yaml"
    assert "kind: Job" in payload["content"]
    assert "bench_run.sh" in payload["extras"]


def test_generate_k8s_deployment_from_estimate_returns_top1_for_agg_and_disagg(monkeypatch):
    monkeypatch.setattr("aiconfigurator.service.app.get_database", lambda **kwargs: object())
    agg_df = pandas.DataFrame(
        [
            {"parallel": "tp8pp1", "tokens/s/gpu": 321.0, "ttft": 210.0, "tpot": 9.0},
            {"parallel": "tp4pp1", "tokens/s/gpu": 300.0, "ttft": 215.0, "tpot": 10.0},
        ]
    )
    disagg_df = pandas.DataFrame([{"parallel": "tp4pp1", "tokens/s/gpu": 280.0, "ttft": 190.0, "tpot": 12.0}])
    result = CLIResult(
        chosen_exp="agg",
        best_configs={"agg": agg_df, "disagg": disagg_df},
        pareto_fronts={"agg": agg_df, "disagg": disagg_df},
        best_throughputs={"agg": 321.0, "disagg": 280.0},
        best_latencies={"agg": {"ttft": 210.0, "tpot": 9.0}, "disagg": {"ttft": 190.0, "tpot": 12.0}},
        task_configs={"agg": object(), "disagg": object()},
    )
    monkeypatch.setattr("aiconfigurator.service.app.cli_default", lambda **kwargs: result)
    monkeypatch.setattr(
        "aiconfigurator.service.app.task_config_to_generator_config",
        lambda task_config, row, generator_overrides=None: {
            "DynConfig": {"mode": "agg" if task_config is result.task_configs["agg"] else "disagg"},
            "row_parallel": row["parallel"],
        },
    )
    monkeypatch.setattr(
        "aiconfigurator.service.app.generate_backend_artifacts",
        lambda **kwargs: {
            "k8s_deploy.yaml": (
                f"kind: Deployment\n"
                f"mode: {kwargs['params']['DynConfig']['mode']}\n"
                f"row: {kwargs['params']['row_parallel']}\n"
            ),
            "run.sh": "#!/bin/bash\n",
        },
    )

    response = _client().post(
        "/api/v1/generate-k8s-deployment",
        json={
            "backend": "vllm",
            "estimate_request": {
                "model_path": "Qwen/Qwen3-8B",
                "total_gpus": 8,
                "system": "h200_sxm",
            },
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["recommended_mode"] == "agg"
    assert payload["jid"].startswith("aic-")
    assert payload["candidates_by_mode"]["agg"][1]["row_index"] == 1
    assert payload["candidates_by_mode"]["agg"][1]["dgd_id"] == f"{payload['jid']}:agg:1"
    assert "agg" in payload["artifacts_by_mode"]
    assert "disagg" in payload["artifacts_by_mode"]
    assert "mode: agg" in payload["artifacts_by_mode"]["agg"]["content"]
    assert "mode: disagg" in payload["artifacts_by_mode"]["disagg"]["content"]

    preview_response = _client().post(
        f"/api/v1/runs/{payload['jid']}/dgd",
        json={"mode": "agg", "row_index": 1},
    )
    assert preview_response.status_code == 200
    preview_payload = preview_response.json()
    assert preview_payload["dgd_id"] == f"{payload['jid']}:agg:1"
    assert "row: tp4pp1" in preview_payload["content"]

    run_response = _client().get(f"/api/v1/runs/{payload['jid']}")
    assert run_response.status_code == 200
    run_payload = run_response.json()
    assert run_payload["payload"]["dgds"][preview_payload["dgd_id"]]["content"] == preview_payload["content"]
