# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""P4: module_bridge.task_config_to_request emits a GeneratorRequest that lowers
to the same artifacts as the legacy dict bridge."""
from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from aiconfigurator.generator.api import generate_backend_artifacts
from aiconfigurator.generator.module_bridge import (
    task_config_to_generator_config,
    task_config_to_request,
)
from aiconfigurator.generator.request import GeneratorRequest, to_legacy_params


def _fake_task_config(mode="agg"):
    runtime = SimpleNamespace(isl=4000, osl=1000, ttft=600.0, tpot=50.0, prefix=0)
    config = SimpleNamespace(runtime_config=runtime, is_moe=False, nextn=0, nextn_accept_rates=None)
    return SimpleNamespace(
        config=config,
        model_path="Qwen/Qwen3-32B",
        serving_mode=mode,
        system_name="h200_sxm",
        backend_name="trtllm",
        backend_version="1.2.0rc5",
        total_gpus=8,
    )


def _row(prefix=""):
    return pd.Series({f"{prefix}{k}": v for k, v in
                      {"workers": 1, "tp": 1, "pp": 1, "dp": 1, "moe_tp": 1, "moe_ep": 1, "bs": 128}.items()})


def test_bridge_injects_encode_role_from_overrides():
    """EPD via `cli default`: an encode role is injected from Workers.encode.* +
    WorkerConfig.encode_workers (the sweep does not produce it)."""
    tc = _fake_task_config("disagg")
    row = pd.Series(
        {f"(p){k}": v for k, v in {"workers": 1, "tp": 2, "pp": 1, "dp": 1, "moe_tp": 1, "moe_ep": 1, "bs": 64}.items()}
        | {f"(d){k}": v for k, v in {"workers": 1, "tp": 2, "pp": 1, "dp": 1, "moe_tp": 1, "moe_ep": 1, "bs": 128}.items()}
    )
    # without the override -> no encode role
    base = task_config_to_generator_config(tc, row, num_gpus_per_node=8)
    assert "encode" not in base["params"]
    # with the override -> encode role injected
    ov = {
        "Workers": {"encode": {"tensor_parallel_size": 1, "modality": "multimodal", "max_file_size_mb": 50}},
        "WorkerConfig": {"encode_workers": 1},
    }
    cfg = task_config_to_generator_config(tc, row, generator_overrides=ov, num_gpus_per_node=8)
    enc = cfg["params"]["encode"]
    assert enc["tensor_parallel_size"] == 1 and enc["gpus_per_worker"] == 1
    assert enc["modality"] == "multimodal" and enc["max_file_size_mb"] == 50
    assert cfg["WorkerConfig"]["encode_workers"] == 1


def _disagg_task_config():
    return _fake_task_config("disagg")


def test_request_bridge_matches_dict_bridge_agg():
    tc = _fake_task_config("agg")
    row = _row("")
    req = task_config_to_request(tc, row, num_gpus_per_node=8)
    assert isinstance(req, GeneratorRequest)
    assert req.backend.name == "trtllm"
    assert req.validate() == []

    dict_out = task_config_to_generator_config(tc, row, num_gpus_per_node=8)
    arts_req = generate_backend_artifacts(to_legacy_params(req), "trtllm", backend_version="1.2.0rc5")
    arts_dict = generate_backend_artifacts(dict_out, "trtllm", backend_version="1.2.0rc5")
    assert arts_req == arts_dict
