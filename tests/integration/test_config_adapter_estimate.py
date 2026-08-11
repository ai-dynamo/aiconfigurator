# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from aiconfigurator.cli.api import EstimateResult, cli_estimate
from aiconfigurator.sdk.config_adapter import (
    AdapterOverrides,
    DynamoRecipeSource,
    adapt_config,
    to_cli_estimate_kwargs,
)

pytestmark = pytest.mark.integration


def _deployment(*, disagg: bool) -> str:
    if not disagg:
        services = """
    worker:
      componentType: worker
      resources: {limits: {gpu: 2}}
      extraPodSpec:
        mainContainer:
          args: ["python -m dynamo.trtllm --model-path QWEN/QWEN3-32B --tensor-parallel-size 2"]
"""
    else:
        services = """
    prefill:
      componentType: worker
      subComponentType: prefill
      resources: {limits: {gpu: 2}}
      extraPodSpec:
        mainContainer:
          args:
            - >-
              python -m dynamo.trtllm --model-path QWEN/QWEN3-32B
              --tensor-parallel-size 2 --max-batch-size 1
              --disaggregation-mode prefill
    decode:
      componentType: worker
      subComponentType: decode
      resources: {limits: {gpu: 2}}
      extraPodSpec:
        mainContainer:
          args:
            - >-
              python -m dynamo.trtllm --model-path QWEN/QWEN3-32B
              --tensor-parallel-size 2 --disaggregation-mode decode
"""
    return f"""
kind: DynamoGraphDeployment
metadata: {{name: estimate-integration}}
spec:
  backendFramework: trtllm
  services:
{services}
"""


@pytest.mark.parametrize(("disagg", "expected_mode"), [(False, "agg"), (True, "disagg")])
def test_adapted_request_runs_real_estimate(disagg, expected_mode):
    report = adapt_config(
        DynamoRecipeSource(_deployment(disagg=disagg)),
        AdapterOverrides(
            system_name="h100_sxm",
            database_mode="SOL",
            isl=128,
            osl=16,
            concurrency=2,
        ),
    )

    result = cli_estimate(**to_cli_estimate_kwargs(report.requests[0]))

    assert isinstance(result, EstimateResult)
    assert result.mode == expected_mode
    assert result.ttft > 0
    assert result.tpot >= 0
