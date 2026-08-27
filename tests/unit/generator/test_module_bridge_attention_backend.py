# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import shlex
from types import SimpleNamespace

import pandas as pd
import pytest

from aiconfigurator.generator.api import generate_backend_artifacts
from aiconfigurator.generator.module_bridge import task_config_to_generator_config

pytestmark = pytest.mark.unit


def _task(*, serving_mode: str = "agg", attention_backend: str | None = "fa3") -> SimpleNamespace:
    return SimpleNamespace(
        primary_backend_name="sglang",
        primary_system_name="gb200",
        primary_backend_version="0.5.14",
        primary_model_path="Qwen/Qwen3-32B-FP8",
        prefix=0,
        is_moe=False,
        nextn=0,
        nextn_accepted=None,
        serving_mode=serving_mode,
        total_gpus=0,
        system_name="gb200",
        prefill_system_name="gb200",
        decode_system_name="gb200",
        isl=1024,
        osl=256,
        ttft=2000.0,
        tpot=50.0,
        attention_backend=attention_backend,
    )


def _flag_value(cli_args: str, flag: str) -> str:
    tokens = shlex.split(cli_args)
    assert tokens.count(flag) == 1
    return tokens[tokens.index(flag) + 1]


def test_aggregate_worker_uses_task_attention_backend():
    row = pd.Series({"workers": 1, "tp": 1, "pp": 1, "dp": 1, "bs": 64})

    result = task_config_to_generator_config(_task(), row, num_gpus_per_node=4)

    assert result["params"]["agg"]["attention_backend"] == "fa3"


def test_disaggregated_workers_use_task_attention_backend():
    row = pd.Series(
        {
            "(p)workers": 1,
            "(p)tp": 1,
            "(p)pp": 1,
            "(p)dp": 1,
            "(p)bs": 8,
            "(d)workers": 1,
            "(d)tp": 1,
            "(d)pp": 1,
            "(d)dp": 1,
            "(d)bs": 64,
        }
    )

    result = task_config_to_generator_config(_task(serving_mode="disagg"), row, num_gpus_per_node=4)

    assert result["params"]["prefill"]["attention_backend"] == "fa3"
    assert result["params"]["decode"]["attention_backend"] == "fa3"


def test_explicit_worker_override_wins_for_its_role():
    row = pd.Series(
        {
            "(p)workers": 1,
            "(p)tp": 1,
            "(p)pp": 1,
            "(p)dp": 1,
            "(p)bs": 8,
            "(d)workers": 1,
            "(d)tp": 1,
            "(d)pp": 1,
            "(d)dp": 1,
            "(d)bs": 64,
        }
    )

    result = task_config_to_generator_config(
        _task(serving_mode="disagg"),
        row,
        generator_overrides={"Workers": {"prefill": {"attention_backend": "flashinfer"}}},
        num_gpus_per_node=4,
    )

    assert result["params"]["prefill"]["attention_backend"] == "flashinfer"
    assert result["params"]["decode"]["attention_backend"] == "fa3"


def test_sglang_artifact_uses_task_attention_backend():
    row = pd.Series({"workers": 1, "tp": 1, "pp": 1, "dp": 1, "bs": 64})
    params = task_config_to_generator_config(_task(), row, num_gpus_per_node=4)

    artifacts = generate_backend_artifacts(
        params,
        "sglang",
        backend_version="0.5.14",
        deployment_target="dynamo-j2",
    )

    assert _flag_value(artifacts["cli_args_agg"], "--attention-backend") == "fa3"


def test_unset_attention_backend_is_omitted():
    row = pd.Series({"workers": 1, "tp": 1, "pp": 1, "dp": 1, "bs": 64})
    params = task_config_to_generator_config(
        _task(attention_backend=None),
        row,
        num_gpus_per_node=4,
    )

    artifacts = generate_backend_artifacts(
        params,
        "sglang",
        backend_version="0.5.14",
        deployment_target="dynamo-j2",
    )

    assert "attention_backend" not in params["params"]["agg"]
    assert "--attention-backend" not in shlex.split(artifacts["cli_args_agg"])


@pytest.mark.parametrize(
    ("backend", "backend_version"),
    [("vllm", "0.24.0"), ("trtllm", "1.3.0rc5")],
)
def test_non_sglang_artifact_omits_attention_backend(backend: str, backend_version: str):
    task = _task()
    task.primary_backend_name = backend
    task.primary_backend_version = backend_version
    row = pd.Series({"workers": 1, "tp": 1, "pp": 1, "dp": 1, "bs": 64})
    params = task_config_to_generator_config(task, row, num_gpus_per_node=4)

    artifacts = generate_backend_artifacts(
        params,
        backend,
        backend_version=backend_version,
        deployment_target="dynamo-j2",
    )

    assert "--attention-backend" not in shlex.split(artifacts["cli_args_agg"])
