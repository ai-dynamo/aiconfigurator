# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NodeConfig propagation through the naive generation path.

naive.py built a local node_config (with user NodeConfig overrides merged in)
but forwarded only the num_gpus_per_node scalar into
collect_generator_params(), which rebuilds NodeConfig from that one value.
Both the system identity and explicit --generator-set NodeConfig.* overrides
were therefore dropped: a --system b60 run rendered run_0.sh with
CUDA_VISIBLE_DEVICES instead of the B60 ONEAPI_DEVICE_SELECTOR (the template
branches on NodeConfig.system_name), while the K8s facts stayed correct
because they resolve from the separate K8sConfig.system_name.
"""

from unittest.mock import patch

import pytest

from aiconfigurator.generator.api import generate_backend_artifacts
from aiconfigurator.generator.naive import build_naive_generator_params

_SYSTEM_CONFIGS = {
    "b60": {"gpus_per_node": 8, "vram_per_gpu": 24 * 1024**3},
    "h200_sxm": {"gpus_per_node": 8, "vram_per_gpu": 141 * 1024**3},
}


def _naive_params(system_name: str, mode: str = "agg", overrides: dict | None = None) -> dict:
    with (
        patch(
            "aiconfigurator.generator.naive._get_system_config",
            side_effect=lambda name: dict(_SYSTEM_CONFIGS[name]),
        ),
        patch(
            "aiconfigurator.generator.naive._estimate_model_weight_bytes",
            return_value=10 * 1024**3,
        ),
    ):
        return build_naive_generator_params(
            model_name="Qwen/Qwen3-8B",
            total_gpus=2,
            system_name=system_name,
            backend_name="vllm",
            mode=mode,
            generator_overrides=overrides,
        )


@pytest.mark.unit
@pytest.mark.parametrize("mode", ["agg", "disagg"])
def test_system_name_propagates_to_node_config(mode):
    params = _naive_params("b60", mode=mode)
    assert params["NodeConfig"]["system_name"] == "b60"
    assert params["NodeConfig"]["num_gpus_per_node"] == 8


@pytest.mark.unit
def test_node_config_overrides_propagate():
    params = _naive_params(
        "h200_sxm",
        overrides={"NodeConfig": {"system_name": "custom-system", "num_gpus_per_node": 4}},
    )
    assert params["NodeConfig"]["system_name"] == "custom-system"
    assert params["NodeConfig"]["num_gpus_per_node"] == 4


@pytest.mark.unit
def test_b60_run_script_uses_oneapi_device_selector():
    params = _naive_params("b60")
    artifacts = generate_backend_artifacts(
        params, "vllm", backend_version="0.20.1", deployment_target="dynamo-j2"
    )
    run_scripts = {k: v for k, v in artifacts.items() if k.startswith("run_")}
    assert run_scripts, f"expected run scripts, got {sorted(artifacts)}"
    for name, content in run_scripts.items():
        assert "ONEAPI_DEVICE_SELECTOR" in content, f"{name}: missing B60 device selector"
        assert "CUDA_VISIBLE_DEVICES" not in content, f"{name}: unexpected CUDA selector"


@pytest.mark.unit
def test_non_b60_run_script_keeps_cuda_device_selector():
    params = _naive_params("h200_sxm")
    artifacts = generate_backend_artifacts(
        params, "vllm", backend_version="0.20.1", deployment_target="dynamo-j2"
    )
    run_scripts = {k: v for k, v in artifacts.items() if k.startswith("run_")}
    assert run_scripts
    for name, content in run_scripts.items():
        assert "CUDA_VISIBLE_DEVICES" in content, f"{name}: missing CUDA selector"
        assert "ONEAPI_DEVICE_SELECTOR" not in content, f"{name}: unexpected B60 selector"
