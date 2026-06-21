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
