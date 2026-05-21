# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pytest

from tools.support_matrix import support_matrix as support_matrix_module
from tools.support_matrix.support_matrix import STATUS_PASS, SupportMatrix, TestConstraints

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "backend,failure_message",
    [
        pytest.param(
            "sglang",
            "No results found: the model does not fit in GPU memory for any parallel configuration. "
            "Try increasing --total-gpus.",
            id="sglang-memory",
        ),
        pytest.param(
            "sglang",
            "No results found for any parallel configuration. Showing last exception: "
            "Failed to query moe data for num_tokens=128, hidden_size=7168. "
            "Missing silicon data for the requested lookup.",
            id="sglang-missing-moe-data",
        ),
        pytest.param(
            "vllm",
            "No results found: the model does not fit in GPU memory for any parallel configuration. "
            "Try increasing --total-gpus.",
            id="vllm-memory",
        ),
        pytest.param(
            "trtllm",
            "No results found: the model does not fit in GPU memory for any parallel configuration. "
            "Try increasing --total-gpus.",
            id="trtllm-memory",
        ),
    ],
)
def test_run_single_test_retries_large_worker_after_recoverable_failure(monkeypatch, backend, failure_message):
    calls: list[dict] = []

    def fake_run_mode(**kwargs):
        calls.append(kwargs)
        if kwargs["yaml_config"] is None:
            raise RuntimeError(failure_message)
        return pd.DataFrame({"tokens/s/gpu": [1.0]})

    monkeypatch.setattr(SupportMatrix, "_run_mode", staticmethod(fake_run_mode))
    monkeypatch.setattr(
        support_matrix_module,
        "_get_test_constraints",
        lambda _model: TestConstraints(total_gpus=128, isl=256, osl=256, prefix=128, ttft=2_000_000, tpot=50_000),
    )

    statuses, errors = SupportMatrix.run_single_test(
        model="zai-org/GLM-5",
        system="b200_sxm",
        backend=backend,
        version="0.5.10",
        system_spec={"gpu": {"sm_version": 100, "fp8_tc_flops": 1, "fp4_tc_flops": 1}},
    )

    assert statuses == {"agg": STATUS_PASS, "disagg": STATUS_PASS}
    assert errors == {"agg": None, "disagg": None}
    assert [call["yaml_config"] is None for call in calls] == [True, False, True, False]
    assert calls[1]["yaml_config"]["config"]["worker_config"]["num_gpu_per_worker"] == [16]
    assert calls[3]["yaml_config"]["config"]["prefill_worker_config"]["pp_list"] == [2]
