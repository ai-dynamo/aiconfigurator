from pathlib import Path

import yaml

_HW = (
    Path(__file__).resolve().parents[3]
    / "src/aiconfigurator/generator/facts/hardware.yaml"
)


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
