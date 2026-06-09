# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for naive generator version override routing."""

import pytest

from aiconfigurator.generator import api

pytestmark = pytest.mark.unit


def _minimal_generator_params(backend: str = "vllm") -> dict:
    return {
        "ServiceConfig": {"model_path": "Qwen/Qwen3-32B"},
        "K8sConfig": {"k8s_image": "nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.2.0"},
        "WorkerConfig": {"agg_workers": 8, "agg_gpus_per_worker": 1},
        "params": {
            "agg": {
                "tensor_parallel_size": 1,
                "pipeline_parallel_size": 1,
                "gpus_per_worker": 1,
                "max_batch_size": 128,
            }
        },
        "backend": backend,
    }


def _patch_naive_dependencies(monkeypatch):
    calls = {"build": None, "render": None}

    def fake_build_naive_generator_params(**kwargs):
        calls["build"] = kwargs
        return _minimal_generator_params(kwargs["backend_name"])

    def fake_render_backend_templates(param_values, backend, templates_dir, backend_version, deployment_target):
        calls["render"] = {
            "param_values": param_values,
            "backend": backend,
            "templates_dir": templates_dir,
            "backend_version": backend_version,
            "deployment_target": deployment_target,
        }
        return {"cli_args_agg": "--model Qwen/Qwen3-32B"}

    monkeypatch.setattr(api, "build_naive_generator_params", fake_build_naive_generator_params)
    monkeypatch.setattr(api, "render_backend_templates", fake_render_backend_templates)
    return calls


def test_generate_naive_config_resolves_backend_version_from_dynamo(monkeypatch):
    calls = _patch_naive_dependencies(monkeypatch)

    result = api.generate_naive_config(
        model_path="Qwen/Qwen3-32B",
        total_gpus=8,
        system="h200_sxm",
        backend="vllm",
        generator_dynamo_version="1.2.0",
    )

    assert result["backend_version"] == "0.20.1"
    assert calls["build"]["generator_dynamo_version"] == "1.2.0"
    assert calls["render"]["backend_version"] == "0.20.1"


def test_generate_naive_config_generated_config_version_wins(monkeypatch):
    calls = _patch_naive_dependencies(monkeypatch)

    result = api.generate_naive_config(
        model_path="Qwen/Qwen3-32B",
        total_gpus=8,
        system="h200_sxm",
        backend="vllm",
        generated_config_version="9.9.9-does-not-exist",
        generator_dynamo_version="1.2.0",
    )

    assert result["backend_version"] == "9.9.9-does-not-exist"
    assert calls["build"]["generator_dynamo_version"] == "1.2.0"
    assert calls["render"]["backend_version"] == "9.9.9-does-not-exist"


def test_generate_naive_config_uses_perf_database_without_overrides(monkeypatch):
    calls = _patch_naive_dependencies(monkeypatch)

    def fake_latest_version(system, backend):
        assert system == "h200_sxm"
        assert backend == "vllm"
        return "0.19.0"

    monkeypatch.setattr("aiconfigurator.sdk.perf_database.get_latest_database_version", fake_latest_version)

    result = api.generate_naive_config(
        model_path="Qwen/Qwen3-32B",
        total_gpus=8,
        system="h200_sxm",
        backend="vllm",
    )

    assert result["backend_version"] == "0.19.0"
    assert calls["render"]["backend_version"] == "0.19.0"
