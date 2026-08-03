# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from aiconfigurator.generator.module_bridge import task_config_to_generator_config

pytestmark = pytest.mark.unit


class _WideEPTask(SimpleNamespace):
    def get_deployment_attention_backend(self, *, role: str) -> str:
        assert role in {"agg", "prefill", "decode"}
        return "trtllm_mla"


def _task() -> _WideEPTask:
    return _WideEPTask(
        primary_backend_name="sglang",
        primary_system_name="b200_sxm",
        primary_backend_version="0.5.10",
        primary_model_path="deepseek-ai/DeepSeek-V3",
        prefix=0,
        is_moe=True,
        nextn=0,
        nextn_accepted=None,
        serving_mode="agg",
        total_gpus=8,
        system_name="b200_sxm",
        prefill_system_name="b200_sxm",
        decode_system_name="b200_sxm",
        isl=1024,
        osl=256,
        ttft=2000.0,
        tpot=50.0,
    )


def test_resolved_backend_reaches_generated_worker_config():
    row = pd.Series({"workers": 1, "tp": 8, "pp": 1, "dp": 1, "moe_tp": 1, "moe_ep": 8, "bs": 8})

    result = task_config_to_generator_config(_task(), row, num_gpus_per_node=8)

    assert result["params"]["agg"]["attention_backend"] == "trtllm_mla"


def test_explicit_generator_override_wins_after_resolution():
    row = pd.Series({"workers": 1, "tp": 8, "pp": 1, "dp": 1, "moe_tp": 1, "moe_ep": 8, "bs": 8})
    overrides = {"Workers": {"agg": {"attention_backend": "flashinfer"}}}

    result = task_config_to_generator_config(_task(), row, generator_overrides=overrides, num_gpus_per_node=8)

    assert result["params"]["agg"]["attention_backend"] == "flashinfer"
