# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Duck-typed task compatibility for the Task-to-generator bridge.

The spica replay path (cli/spica/helper.py:_spica_generator_task) feeds the
bridge an argparse.Namespace that mimics a Task with a fixed field set — it
has primary_* views, serving_mode, and workload/SLA scalars, but none of the
optional Task fields (multimodal image workload, per-role disagg system
names). New bridge reads of Task attributes must therefore stay defensive
(getattr with defaults), or the spica path crashes with AttributeError.
"""

import argparse

import pandas as pd
import pytest

from aiconfigurator.generator.module_bridge import task_config_to_generator_config


def _spica_like_namespace(serving_mode: str) -> argparse.Namespace:
    # Mirrors the exact field set of cli/spica/helper.py:_spica_generator_task.
    return argparse.Namespace(
        primary_backend_name="sglang",
        primary_system_name="h200_sxm",
        primary_backend_version="0.5.11",
        primary_model_path="Qwen/Qwen3-32B",
        prefix=0,
        is_moe=False,
        nextn=0,
        nextn_accept_rates=[0.85, 0.8, 0.6, 0.0, 0.0],
        serving_mode=serving_mode,
        total_gpus=8,
        system_name="h200_sxm",
        isl=1024,
        osl=256,
        ttft=2000.0,
        tpot=50.0,
    )


@pytest.mark.unit
@pytest.mark.parametrize("serving_mode", ["agg", "disagg"])
def test_bridge_accepts_spica_namespace(serving_mode):
    row = pd.Series({"tp": 1, "(p)tp": 1, "(d)tp": 1})
    cfg = task_config_to_generator_config(task_config=_spica_like_namespace(serving_mode), result_df=row)
    assert cfg["NodeConfig"]["system_name"] == "h200_sxm"
    # No image fields on the namespace -> image workload must stay disabled.
    assert cfg["BenchConfig"].get("image_batch_size", 0) == 0
