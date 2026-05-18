# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from tools.support_matrix.support_matrix import (
    STATUS_HW_INCOMPATIBLE,
    STATUS_PASS,
    SupportMatrix,
    TestConstraints,
    get_hardware_incompatibility,
)

pytestmark = pytest.mark.unit


def _system_spec(*, sm_version: int, fp8: bool = False, fp4: bool = False) -> dict:
    gpu = {"sm_version": sm_version}
    if fp8:
        gpu["fp8_tc_flops"] = 1
    if fp4:
        gpu["fp4_tc_flops"] = 1
    return {"gpu": gpu}


def test_fp8_model_is_hardware_incompatible_without_fp8_support():
    incompatibility = get_hardware_incompatibility(
        model="Qwen/Qwen3-32B-FP8",
        system="a100_sxm",
        backend="trtllm",
        system_spec=_system_spec(sm_version=80),
    )

    assert incompatibility is not None
    assert incompatibility.missing_datatypes == ("FP8",)
    assert incompatibility.reason == "a100_sxm (SM80) does not support FP8 required by Qwen/Qwen3-32B-FP8"


def test_fp8_model_is_allowed_when_system_advertises_fp8_support():
    incompatibility = get_hardware_incompatibility(
        model="Qwen/Qwen3-32B-FP8",
        system="l40s",
        backend="trtllm",
        system_spec=_system_spec(sm_version=89, fp8=True),
    )

    assert incompatibility is None


def test_fp8_model_is_allowed_for_b60_software_fallback_without_native_fp8_flops():
    incompatibility = get_hardware_incompatibility(
        model="nvidia/Llama-3.1-70B-Instruct-FP8",
        system="b60",
        backend="vllm",
        system_spec={"gpu": {"bfloat16_tc_flops": 1}},
    )

    assert incompatibility is None


def test_fp4_model_is_hardware_incompatible_without_fp4_support():
    incompatibility = get_hardware_incompatibility(
        model="nvidia/Qwen3-235B-A22B-NVFP4",
        system="h100_sxm",
        backend="trtllm",
        system_spec=_system_spec(sm_version=90, fp8=True),
    )

    assert incompatibility is not None
    assert incompatibility.missing_datatypes == ("FP4",)
    assert "does not support FP4" in incompatibility.reason


@pytest.mark.parametrize("model", ["openai/gpt-oss-20b", "openai/gpt-oss-120b"])
def test_mxfp4_model_is_not_native_fp4_hardware_incompatible_on_hopper(model):
    incompatibility = get_hardware_incompatibility(
        model=model,
        system="h100_sxm",
        backend="trtllm",
        system_spec=_system_spec(sm_version=90, fp8=True),
    )

    assert incompatibility is None


def test_run_single_test_short_circuits_hardware_incompatible_model(monkeypatch):
    def fail_run_mode(**_kwargs):
        raise AssertionError("TaskRunner should not run for hardware-incompatible combinations")

    monkeypatch.setattr(SupportMatrix, "_run_mode", staticmethod(fail_run_mode))

    status_dict, error_dict = SupportMatrix.run_single_test(
        model="Qwen/Qwen3-32B-FP8",
        system="a100_sxm",
        backend="trtllm",
        version="1.0.0",
        system_spec=_system_spec(sm_version=80),
    )

    assert status_dict == {"agg": STATUS_HW_INCOMPATIBLE, "disagg": STATUS_HW_INCOMPATIBLE}
    assert "does not support FP8" in error_dict["agg"]
    assert STATUS_PASS not in status_dict.values()


def test_large_moe_support_matrix_search_enables_pp_and_caps_ep():
    task_config = SupportMatrix._create_task_config(
        mode="agg",
        model="Qwen/Qwen3-Coder-480B-A35B-Instruct",
        system="rtx_pro_6000_server",
        backend="vllm",
        version="0.19.0",
        constraints=TestConstraints(total_gpus=128, isl=256, osl=256, prefix=128, ttft=2000000.0, tpot=50000.0),
        engine_step_backend=None,
        system_spec={"node": {"num_gpus_per_node": 8}},
    )

    worker_config = task_config.config.worker_config
    assert 16 in worker_config.num_gpu_per_worker
    assert 2 in worker_config.pp_list
    assert worker_config.dp_list == [1]
    assert max(worker_config.moe_ep_list) == 8


def test_large_moe_support_matrix_search_caps_ep_to_smaller_node():
    task_config = SupportMatrix._create_task_config(
        mode="disagg",
        model="Qwen/Qwen3-Coder-480B-A35B-Instruct",
        system="rtx_pro_6000_server",
        backend="trtllm",
        version="1.3.0rc10",
        constraints=TestConstraints(total_gpus=128, isl=256, osl=256, prefix=128, ttft=2000000.0, tpot=50000.0),
        engine_step_backend=None,
        system_spec={"node": {"num_gpus_per_node": 4}},
    )

    prefill_config = task_config.config.prefill_worker_config
    decode_config = task_config.config.decode_worker_config
    assert max(prefill_config.moe_ep_list) == 4
    assert max(decode_config.moe_ep_list) == 4
    assert task_config.config.replica_config.max_gpu_per_replica == 128
