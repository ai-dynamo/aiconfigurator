# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Artifact-level MTP regression tests.

These go through the real ``save_results`` path and assert on the dumped
``exp_config.yaml`` — the file users actually consume for repro — rather
than on in-memory task state.
"""

import argparse
from unittest.mock import patch

import pytest
import yaml

from aiconfigurator.cli.main import build_default_tasks
from aiconfigurator.cli.report_and_save import save_results

pytestmark = pytest.mark.unit


def _dump_exp_config(tasks, tmp_path):
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
    return yaml.safe_load(exp_config_path.read_text(encoding="utf-8"))


def test_mtp_checkpoint_stays_off_in_dumped_exp_config(tmp_path):
    """GLM-5.2 ships num_nextn_predict_layers=1; without an explicit --nextn the
    dumped exp_config must record MTP off (never auto-enabled)."""
    tasks = build_default_tasks(
        model_path="nvidia/GLM-5.2-NVFP4",
        total_gpus=1,
        system="gb200",
        backend="vllm",
        backend_version="0.11.0",
        database_mode="SOL",
    )
    assert tasks["agg"].nextn == 0

    dumped = _dump_exp_config(tasks, tmp_path)
    assert dumped["nextn"] == 0


def test_explicit_nextn_is_preserved_in_dumped_exp_config(tmp_path):
    tasks = build_default_tasks(
        model_path="nvidia/GLM-5.2-NVFP4",
        total_gpus=1,
        system="gb200",
        backend="vllm",
        backend_version="0.11.0",
        database_mode="SOL",
        nextn=1,
        nextn_accepted=0.7,
    )
    assert tasks["agg"].nextn == 1
    assert tasks["agg"].nextn_accepted == 0.7

    dumped = _dump_exp_config(tasks, tmp_path)
    assert dumped["nextn"] == 1
    assert dumped["nextn_accepted"] == 0.7
