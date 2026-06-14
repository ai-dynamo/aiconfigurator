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
    # Blackwell/GB200 and Hopper both have a trtllm MoE selection (non-empty).
    assert profiles["gb200"]["moe_backend"].get("trtllm")
    assert profiles["h200"]["moe_backend"].get("trtllm")
