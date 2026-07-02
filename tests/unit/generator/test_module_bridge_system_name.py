# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""System-name propagation from Task v2 into generator params.

Task v2's prefix discipline keeps the top-level ``system_name`` empty for
disagg tasks (the values live in ``prefill_system_name``/``decode_system_name``),
so the bridge must read ``primary_system_name``. Reading the raw field left
``NodeConfig.system_name`` empty, facts resolution returned None, and disagg
DGDs shipped with no nodeSelector/tolerations/NCCL env — a ``--system gb200``
SGLang deployment silently scheduled onto amd64 H200 nodes.
"""

import pandas as pd
import pytest
import yaml

from aiconfigurator.generator.api import generate_backend_artifacts
from aiconfigurator.generator.module_bridge import task_config_to_generator_config
from aiconfigurator.sdk.task_v2 import Task

MODEL = "Qwen/Qwen3-32B"


def _disagg_task(prefill_system: str, decode_system: str) -> Task:
    return Task(
        serving_mode="disagg",
        prefill_model_path=MODEL,
        prefill_system_name=prefill_system,
        prefill_backend_name="sglang",
        decode_model_path=MODEL,
        decode_system_name=decode_system,
        decode_backend_name="sglang",
        isl=1024,
        osl=256,
        ttft=2000.0,
        tpot=50.0,
    )


def _result_row() -> pd.Series:
    return pd.Series(
        {
            "(p)workers": 1,
            "(p)tp": 1,
            "(d)workers": 1,
            "(d)tp": 1,
        }
    )


@pytest.mark.unit
def test_disagg_system_name_propagates_to_node_config():
    task = _disagg_task("gb200", "gb200")
    cfg = task_config_to_generator_config(task_config=task, result_df=_result_row())
    assert cfg["NodeConfig"]["system_name"] == "gb200"


@pytest.mark.unit
def test_agg_system_name_still_propagates():
    task = Task(
        serving_mode="agg",
        model_path=MODEL,
        system_name="h200_sxm",
        backend_name="sglang",
        isl=1024,
        osl=256,
        ttft=2000.0,
        tpot=50.0,
    )
    cfg = task_config_to_generator_config(task_config=task, result_df=pd.Series({"tp": 1}))
    assert cfg["NodeConfig"]["system_name"] == "h200_sxm"


@pytest.mark.unit
def test_heterogeneous_disagg_systems_rejected():
    task = _disagg_task("gb200", "h200_sxm")
    with pytest.raises(ValueError, match=r"heterogeneous"):
        task_config_to_generator_config(task_config=task, result_df=_result_row())


@pytest.mark.unit
def test_disagg_gb200_hardware_facts_reach_dgd():
    """End-to-end through the generator: the resolved system must produce GB200
    placement and NCCL/UCX environment on disagg workers in the raw DGD."""
    task = _disagg_task("gb200", "gb200")
    cfg = task_config_to_generator_config(task_config=task, result_df=_result_row())
    artifacts = generate_backend_artifacts(
        cfg, "sglang", backend_version="0.5.11", deployment_target="dynamo-j2"
    )
    dgd = yaml.safe_load(artifacts["k8s_deploy.yaml"])

    workers = {
        name: svc
        for name, svc in dgd["spec"]["services"].items()
        if svc.get("componentType") == "worker"
    }
    assert workers, "expected disagg worker services in the DGD"
    for name, svc in workers.items():
        pod = svc.get("extraPodSpec") or {}
        selector = pod.get("nodeSelector") or {}
        assert selector.get("kubernetes.io/arch") == "arm64", f"{name}: missing arm64 selector"
        assert selector.get("nvidia.com/gpu.product") == "NVIDIA-GB200", f"{name}: missing GB200 selector"
        env = {e["name"]: e.get("value") for e in (pod.get("mainContainer") or {}).get("env") or []}
        assert "NCCL_MNNVL_ENABLE" in env, f"{name}: missing GB200 NCCL facts"
        assert "UCX_CUDA_IPC_ENABLE_MNNVL" in env, f"{name}: missing GB200 UCX facts"
