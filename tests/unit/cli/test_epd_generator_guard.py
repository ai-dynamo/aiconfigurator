# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""EPD rows must not silently produce generator artifacts: the generator
bridge does not map the dedicated encode pool yet, so the emitted deploy
configs would contradict the recommendation."""

import argparse
import logging
from unittest.mock import patch

import pandas as pd
import pytest

from aiconfigurator.cli.main import build_default_tasks
from aiconfigurator.cli.report_and_save import save_results

pytestmark = pytest.mark.unit


def test_epd_rows_skip_generator_artifacts(tmp_path, caplog):
    tasks = build_default_tasks(
        model_path="nvidia/GLM-5.2-NVFP4",
        total_gpus=1,
        system="gb200",
        backend="vllm",
        backend_version="0.11.0",
        database_mode="SOL",
    )

    def _row(e_workers: float) -> dict:
        return {
            "tp": 1,
            "pp": 1,
            "dp": 1,
            "ttft": 100.0,
            "tpot": 10.0,
            "tokens/s/gpu": 100.0,
            "power_w": 0.0,
            "(e)workers": e_workers,
        }

    args = argparse.Namespace(inclusive_tpot=False, deployment_target="dynamo-j2")
    caplog.set_level(logging.WARNING)
    with (
        patch(
            "aiconfigurator.cli.report_and_save.get_default_dynamo_version_mapping",
            return_value=("1.0.0", {"vllm": "0.11.0"}),
        ),
        patch(
            "aiconfigurator.cli.report_and_save.task_config_to_generator_config",
            return_value={},
        ) as bridge,
    ):
        save_results(
            args=args,
            best_configs={"agg": pd.DataFrame([_row(2.0), _row(0.0)])},
            pareto_fronts={"agg": None},
            tasks=tasks,
            save_dir=str(tmp_path),
            backend="vllm",
        )

    assert bridge.call_count == 1  # only the non-EPD row reaches the bridge
    assert "generator artifacts skipped" in caplog.text
