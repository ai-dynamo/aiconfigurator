# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The bridge must preserve the requested system into NodeConfig.

Task v2 stores the disagg system selection in phase-specific fields; the shared
top-level ``system_name`` is empty for disagg. The bridge must read
``primary_system_name`` so hardware facts (node selectors, NCCL/UCX env) resolve
for disaggregated deployments instead of silently placing workers anywhere.
"""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from aiconfigurator.generator.module_bridge import task_config_to_generator_config


def _task(
    *,
    serving_mode: str,
    system_name: str,
    primary_system_name: str,
    prefill_system_name: str | None = None,
    decode_system_name: str | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        primary_backend_name="vllm",
        primary_system_name=primary_system_name,
        primary_backend_version="0.20.1",
        primary_model_path="Qwen/Qwen3-32B-FP8",
        prefix=0,
        is_moe=False,
        nextn=0,
        nextn_accepted=None,
        serving_mode=serving_mode,
        total_gpus=0,
        system_name=system_name,
        # Disagg phase-specific systems; default to the primary so homogeneous
        # cases don't trip the heterogeneous-placement guard.
        prefill_system_name=prefill_system_name if prefill_system_name is not None else primary_system_name,
        decode_system_name=decode_system_name if decode_system_name is not None else primary_system_name,
        isl=1024,
        osl=256,
        ttft=2000.0,
        tpot=50.0,
    )


def test_disagg_preserves_system_name_from_primary():
    # Disagg: top-level system_name is empty; prefill/decode carry the value.
    task = _task(serving_mode="disagg", system_name="", primary_system_name="gb200")
    row = pd.Series({"(p)workers": 1, "(p)tp": 1, "(d)workers": 1, "(d)tp": 1})

    result = task_config_to_generator_config(task, row, num_gpus_per_node=4)

    assert result["NodeConfig"]["system_name"] == "gb200"


def test_agg_preserves_system_name():
    # Agg: primary_system_name is the top-level system_name.
    task = _task(serving_mode="agg", system_name="h200_sxm", primary_system_name="h200_sxm")
    row = pd.Series({"workers": 1, "tp": 1})

    result = task_config_to_generator_config(task, row, num_gpus_per_node=8)

    assert result["NodeConfig"]["system_name"] == "h200_sxm"


def test_disagg_heterogeneous_systems_raise():
    # NodeConfig.system_name is global; differing prefill/decode systems would
    # silently apply the prefill placement to both -> fail fast instead.
    task = _task(
        serving_mode="disagg",
        system_name="",
        primary_system_name="gb200",
        prefill_system_name="gb200",
        decode_system_name="h200_sxm",
    )
    row = pd.Series({"(p)workers": 1, "(p)tp": 1, "(d)workers": 1, "(d)tp": 1})

    with pytest.raises(ValueError, match="matching prefill/decode systems"):
        task_config_to_generator_config(task, row, num_gpus_per_node=4)
