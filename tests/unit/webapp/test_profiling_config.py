# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for profiling config generation."""

import importlib.util
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_CONFIG_PATH = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "aiconfigurator"
    / "webapp"
    / "components"
    / "profiling"
    / "sdk"
    / "config.py"
)


def _load_profiling_config_module():
    spec = importlib.util.spec_from_file_location("profiling_config_for_test", _CONFIG_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_generate_config_yaml_supplies_required_served_model_name(monkeypatch):
    profiling_config = _load_profiling_config_module()
    captured = {}

    def fake_generate_config_from_input_dict(input_params, backend):
        captured["input_params"] = input_params
        captured["backend"] = backend
        return {"params": {}}

    def fake_generate_backend_artifacts(params, backend, backend_version):
        captured["backend_version"] = backend_version
        return {"k8s_deploy.yaml": "apiVersion: nvidia.com/v1alpha1\n"}

    monkeypatch.setattr(profiling_config, "generate_config_from_input_dict", fake_generate_config_from_input_dict)
    monkeypatch.setattr(profiling_config, "generate_backend_artifacts", fake_generate_backend_artifacts)

    result = profiling_config.generate_config_yaml(
        model_path="deepseek-ai/DeepSeek-V3",
        system="h200_sxm",
        backend="trtllm",
        version="1.3.0rc10",
        isl=2048,
        osl=128,
        num_gpus=4,
    )

    service_config = captured["input_params"]["ServiceConfig"]
    assert service_config["served_model_name"] == "deepseek-ai/DeepSeek-V3"
    assert service_config["served_model_path"] == "deepseek-ai/DeepSeek-V3"
    assert captured["backend"] == "trtllm"
    assert captured["backend_version"] == "1.3.0rc10"
    assert result == "apiVersion: nvidia.com/v1alpha1\n"
