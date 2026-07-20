# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
from unittest.mock import patch

import pytest
import yaml

from aiconfigurator.cli.main import build_default_tasks
from aiconfigurator.cli.report_and_save import save_results

pytestmark = pytest.mark.unit


def test_glm52_explicit_nextn_zero_is_preserved_in_dumped_exp_config(tmp_path):
    """Regression: an explicit zero must not fall back to GLM-5.2's MTP default."""
    tasks = build_default_tasks(
        model_path="nvidia/GLM-5.2-NVFP4",
        total_gpus=1,
        system="gb200",
        backend="vllm",
        backend_version="0.11.0",
        database_mode="SOL",
        nextn=0,
    )
    task = tasks["agg"]
    assert task.nextn == 0

    args = argparse.Namespace(inclusive_tpot=False, deployment_target="dynamo-j2")
    with patch(
        "aiconfigurator.cli.report_and_save.get_default_dynamo_version_mapping",
        return_value=("1.0.0", {"vllm": "0.11.0"}),
    ):
        save_results(
            args=args,
            best_configs={},
            pareto_fronts={"agg": None},
            tasks=tasks,
            save_dir=str(tmp_path),
            backend="vllm",
        )

    exp_config_path = next(tmp_path.glob("**/agg/exp_config.yaml"))
    dumped = yaml.safe_load(exp_config_path.read_text(encoding="utf-8"))
    assert dumped["nextn"] == 0


def test_glm52_explicit_auto_uses_model_default(tmp_path):
    """Explicit auto/None should retain GLM-5.2's checkpoint MTP depth."""
    tasks = build_default_tasks(
        model_path="nvidia/GLM-5.2-NVFP4",
        total_gpus=1,
        system="gb200",
        backend="vllm",
        backend_version="0.11.0",
        database_mode="SOL",
        nextn=None,
    )
    task = tasks["agg"]
    assert task.nextn == 1

    args = argparse.Namespace(inclusive_tpot=False, deployment_target="dynamo-j2")
    with patch(
        "aiconfigurator.cli.report_and_save.get_default_dynamo_version_mapping",
        return_value=("1.0.0", {"vllm": "0.11.0"}),
    ):
        save_results(
            args=args,
            best_configs={},
            pareto_fronts={"agg": None},
            tasks=tasks,
            save_dir=str(tmp_path),
            backend="vllm",
        )

    exp_config_path = next(tmp_path.glob("**/agg/exp_config.yaml"))
    dumped = yaml.safe_load(exp_config_path.read_text(encoding="utf-8"))
    assert dumped["nextn"] == 1
