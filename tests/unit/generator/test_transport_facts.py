from pathlib import Path

import yaml

_T = (
    Path(__file__).resolve().parents[3]
    / "src/aiconfigurator/generator/facts/transport.yaml"
)


def _profiles():
    return yaml.safe_load(_T.read_text())["profiles"]


def test_transport_profiles_present():
    assert {"nvlink", "ib", "efa"} <= set(_profiles())


def test_efa_is_a_self_contained_stack():
    efa = _profiles()["efa"]
    assert isinstance(efa.get("env"), dict) and efa["env"], "efa carries its full env block"
    # the FI_PROVIDER/FI_EFA_* libfabric keys are present as one unit
    assert any(k.startswith("FI_") for k in efa["env"])
    assert "pod" in efa, "efa declares pod-level requirements (privileged/hostIPC/efa resource)"


def test_nvlink_is_the_default_baseline():
    profiles = _profiles()
    # nvlink is the simplest profile (may have minimal/empty env); it must exist
    assert "nvlink" in profiles
