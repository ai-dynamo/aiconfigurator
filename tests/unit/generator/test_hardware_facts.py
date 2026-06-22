# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import yaml

_HW = Path(__file__).resolve().parents[3] / "src/aiconfigurator/generator/facts/hardware.yaml"


def _profiles():
    return yaml.safe_load(_HW.read_text())["profiles"]


def test_hardware_profiles_present():
    profiles = _profiles()
    assert {"h200", "b200", "gb200"} <= set(profiles)


def test_each_profile_has_node_selector_and_moe_backend():
    for name, p in _profiles().items():
        assert isinstance(p.get("node_selector"), dict), f"{name} missing node_selector"
        assert isinstance(p.get("moe_backend"), dict), f"{name} missing moe_backend (per-backend dict)"


def test_moe_backend_reflects_silicon():
    profiles = _profiles()
    # Every profile has a non-empty trtllm MoE selection (the crash-relevant one).
    for name, p in profiles.items():
        assert p["moe_backend"].get("trtllm"), f"{name} missing moe_backend.trtllm"

    # sglang deepep_moe is a Blackwell/wide-EP selection only.
    # Source: webapp/events/event_fn.py moe_backend="deepep_moe" when enable_wideep+sglang.
    blackwell = {"b200", "gb200"}
    hopper = {"h100", "h200"}
    for name in blackwell:
        assert profiles[name]["moe_backend"].get("sglang") == "deepep_moe", (
            f"{name} (Blackwell) must have sglang: deepep_moe"
        )
    for name in hopper:
        assert "sglang" not in profiles[name]["moe_backend"], (
            f"{name} (Hopper) must NOT have sglang deepep_moe — Blackwell/wide-EP only"
        )


# ---------------------------------------------------------------------------
# Toleration data tests
# ---------------------------------------------------------------------------


def test_every_profile_has_tolerations_key():
    """Every hardware profile must declare a tolerations list (may be empty)."""
    for name, p in _profiles().items():
        assert "tolerations" in p, f"{name} missing tolerations key"
        assert isinstance(p["tolerations"], list), f"{name}.tolerations must be a list, got {type(p['tolerations'])}"


def test_gb200_tolerations_include_arch_arm64():
    """GB200 (arm64) nodes carry a kubernetes.io/arch taint; the toleration is required.

    Provenance: glm-5-nvfp4/sglang/disagg/efa/deploy.yaml
    """
    gb200 = _profiles()["gb200"]
    arch_toleration = {
        "key": "kubernetes.io/arch",
        "operator": "Equal",
        "value": "arm64",
        "effect": "NoSchedule",
    }
    assert arch_toleration in gb200["tolerations"], (
        "gb200 tolerations must include kubernetes.io/arch=arm64 NoSchedule "
        "(required for Grace-Blackwell arm64 node taint)"
    )


def test_gb200_tolerations_include_gpu():
    """GB200 must also tolerate the nvidia.com/gpu NoSchedule taint."""
    gb200 = _profiles()["gb200"]
    gpu_keys = {t["key"] for t in gb200["tolerations"]}
    assert "nvidia.com/gpu" in gpu_keys, "gb200 tolerations must include nvidia.com/gpu"


def test_h200_tolerations_match_recipe():
    """H200 (Hopper x86) toleration: nvidia.com/gpu Equal true NoSchedule.

    Provenance: nemotron-3-super/vllm/agg-h200-chat/deploy.yaml and
    nemotron-3-ultra/vllm/agg-h200-chat-nomtp/deploy.yaml
    """
    h200 = _profiles()["h200"]
    assert isinstance(h200["tolerations"], list), "h200.tolerations must be a list"
    assert len(h200["tolerations"]) >= 1, "h200 must have at least one toleration per recipe"
    # The gpu toleration key must be present.
    gpu_keys = {t["key"] for t in h200["tolerations"]}
    assert "nvidia.com/gpu" in gpu_keys, "h200 must tolerate nvidia.com/gpu"
    # No arch toleration — h200 is x86_64, not arm64.
    arch_keys = {t.get("value") for t in h200["tolerations"] if t.get("key") == "kubernetes.io/arch"}
    assert "arm64" not in arch_keys, "h200 (x86_64) must NOT have kubernetes.io/arch=arm64 toleration"


def test_hopper_h100_tolerations_empty():
    """H100 Hopper x86: no recipe shows tolerations; qwen3 hopper recipe explicitly uses [].

    Provenance: qwen3-235b-a22b-fp8/trtllm/agg/hopper/deploy.yaml — tolerations: []
    """
    h100 = _profiles()["h100"]
    assert h100["tolerations"] == [], "h100 tolerations must be [] — no Hopper H100 recipe shows node taints"
