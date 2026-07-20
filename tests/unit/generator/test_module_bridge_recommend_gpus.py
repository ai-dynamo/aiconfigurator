# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The bridge must use total_gpus_needed (not the escalation budget) for recommend mode.

When recommend mode escalates the GPU budget to fit a model, task.total_gpus
holds the search ceiling (8/16/32/64), not the actual recommendation.  The
per-row ``total_gpus_needed`` from load-match picking is the true result.
``task_config_to_generator_config`` must prefer ``total_gpus_needed`` from the
result row so generated deployment artifacts have the correct replica count.
"""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from aiconfigurator.generator.module_bridge import task_config_to_generator_config


def _task(*, total_gpus: int, serving_mode: str = "agg") -> SimpleNamespace:
    return SimpleNamespace(
        primary_backend_name="vllm",
        primary_system_name="h200_sxm",
        primary_backend_version="0.20.1",
        primary_model_path="Qwen/Qwen3-32B-FP8",
        prefix=0,
        is_moe=False,
        nextn=0,
        nextn_accept_rates=[],
        serving_mode=serving_mode,
        total_gpus=total_gpus,
        system_name="h200_sxm",
        prefill_system_name="h200_sxm",
        decode_system_name="h200_sxm",
        isl=1024,
        osl=256,
        ttft=2000.0,
        tpot=50.0,
    )


def _result_row(**kwargs) -> pd.Series:
    base = {"tp": 1, "pp": 1, "dp": 1, "moe_tp": 1, "moe_ep": 1, "bs": 64, "workers": 1}
    base.update(kwargs)
    return pd.Series(base)


@pytest.mark.unit
class TestRecommendGPUCount:
    def test_uses_total_gpus_needed_over_task_budget(self):
        task = _task(total_gpus=64)
        row = _result_row(total_gpus_needed=16, tp=8, pp=1, dp=1)
        cfg = task_config_to_generator_config(task, row)
        assert cfg["WorkerConfig"]["agg_workers"] == 2  # 16 // 8 = 2

    def test_falls_back_to_task_total_gpus_without_column(self):
        task = _task(total_gpus=16)
        row = _result_row(tp=8, pp=1, dp=1)
        cfg = task_config_to_generator_config(task, row)
        assert cfg["WorkerConfig"]["agg_workers"] == 2  # 16 // 8 = 2

    def test_zero_total_gpus_needed_falls_back(self):
        task = _task(total_gpus=8)
        row = _result_row(total_gpus_needed=0, tp=8, pp=1, dp=1)
        cfg = task_config_to_generator_config(task, row)
        assert cfg["WorkerConfig"]["agg_workers"] == 1  # 8 // 8 = 1
