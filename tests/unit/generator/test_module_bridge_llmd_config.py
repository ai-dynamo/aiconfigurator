# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LlmdConfig override propagation through the Task-to-generator bridge.

The naive path preserves the LlmdConfig section from --generator-set
overrides, but task_config_to_generator_config (used by cli default/sweep)
never read it, so llm-d-kustomize/llm-d-helm artifacts silently fell back to
the template defaults (vllm/vllm-openai:latest image and the default
Kustomize base) no matter what the user requested.
"""

import pandas as pd
import pytest
import yaml

from aiconfigurator.generator.api import generate_backend_artifacts
from aiconfigurator.generator.module_bridge import task_config_to_generator_config
from aiconfigurator.sdk.task_v2 import Task

_IMAGE = "nvcr.io/nvstaging/ai-dynamo/vllm-runtime:1.3.0-rc1"
_BASE = "./guides/pd-disaggregation/modelserver/gpu/vllm/base"
_LLMD_OVERRIDES = {
    "LlmdConfig": {
        "vllm_image": _IMAGE,
        "kustomize_base_path": _BASE,
    }
}


def _disagg_task() -> Task:
    return Task(
        serving_mode="disagg",
        prefill_model_path="Qwen/Qwen3-32B",
        prefill_system_name="h200_sxm",
        prefill_backend_name="vllm",
        decode_model_path="Qwen/Qwen3-32B",
        decode_system_name="h200_sxm",
        decode_backend_name="vllm",
        isl=4000,
        osl=1000,
        ttft=2000.0,
        tpot=50.0,
    )


def _bridge(overrides: dict | None = None) -> dict:
    return task_config_to_generator_config(
        task_config=_disagg_task(),
        result_df=pd.Series({"(p)tp": 1, "(d)tp": 1}),
        generator_overrides=overrides,
    )


@pytest.mark.unit
def test_llmd_config_overrides_survive_bridge():
    cfg = _bridge(_LLMD_OVERRIDES)
    assert cfg["LlmdConfig"] == {"vllm_image": _IMAGE, "kustomize_base_path": _BASE}


@pytest.mark.unit
def test_absent_llmd_override_leaves_section_out():
    assert "LlmdConfig" not in _bridge()


@pytest.mark.unit
def test_llmd_kustomize_artifacts_use_overridden_image_and_base():
    cfg = _bridge(_LLMD_OVERRIDES)
    artifacts = generate_backend_artifacts(cfg, "vllm", backend_version="0.20.1", deployment_target="llm-d-kustomize")
    assert set(artifacts) >= {"kustomization.yaml", "patch-decode.yaml", "patch-prefill.yaml"}

    kustomization = yaml.safe_load(artifacts["kustomization.yaml"])
    assert kustomization["resources"] == [_BASE]

    for patch_name in ("patch-decode.yaml", "patch-prefill.yaml"):
        container = yaml.safe_load(artifacts[patch_name])["spec"]["template"]["spec"]["containers"][0]
        assert container["image"] == _IMAGE, f"{patch_name}: image not overridden"


@pytest.mark.unit
def test_llmd_helm_values_use_overridden_image():
    cfg = _bridge(_LLMD_OVERRIDES)
    artifacts = generate_backend_artifacts(cfg, "vllm", backend_version="0.20.1", deployment_target="llm-d-helm")
    values = artifacts["llm-d-values.yaml"]
    assert _IMAGE in values
    assert "vllm/vllm-openai:latest" not in values
