# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from aiconfigurator.generator.facts.request_resolution import (
    hardware_key_for_system,
    model_profile_for_path,
    resolve_facts_for_request,
)


def test_system_name_normalizes_to_hardware_key():
    assert hardware_key_for_system("h200_sxm") == "h200"
    assert hardware_key_for_system("gb200") == "gb200"
    assert hardware_key_for_system("b200_sxm") == "b200"


def test_model_path_matches_profile():
    assert model_profile_for_path("deepseek-ai/DeepSeek-V4-Pro") == "deepseek-v4"
    assert model_profile_for_path("Qwen/Qwen3-0.6B") is None


def test_resolve_for_request_deepseek_blackwell():
    # system_name lives in K8sConfig in the real param shape (verified via canary).
    params = {
        "ServiceConfig": {"model_path": "deepseek-ai/DeepSeek-V4-Pro"},
        "K8sConfig": {"system_name": "gb200"},
        "NodeConfig": {},
    }
    facts = resolve_facts_for_request(params, "trtllm", "1.2.0")
    assert facts is not None and facts.model is not None and "moe" in facts.model["traits"]
    assert facts.hardware["moe_backend"]["trtllm"] == "WIDEEP"


def test_resolve_for_request_generic_qwen_h200_has_no_model():
    params = {
        "ServiceConfig": {"model_path": "Qwen/Qwen3-0.6B"},
        "K8sConfig": {"system_name": "h200_sxm"},
        "NodeConfig": {},
    }
    facts = resolve_facts_for_request(params, "vllm", "1.2.0")
    assert facts is not None and facts.model is None
    assert facts.hardware["moe_backend"]["trtllm"] == "CUTLASS"
