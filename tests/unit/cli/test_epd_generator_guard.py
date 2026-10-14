# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""EPD rows must not silently produce generator artifacts: the generator
bridge does not map the dedicated encode pool yet, so the emitted deploy
configs would contradict the recommendation."""

import argparse
import json
import logging
from unittest.mock import patch

import pandas as pd
import pytest

from aiconfigurator.cli.main import build_default_tasks
from aiconfigurator.cli.report_and_save import save_results
from aiconfigurator.sdk.performance_result import MOE_COMM_FALLBACKS_COLUMN, MoECommFallback

pytestmark = pytest.mark.unit


def test_epd_rows_skip_generator_artifacts(tmp_path, caplog):
    tasks = build_default_tasks(
        model_path="nvidia/GLM-5.2-NVFP4",
        total_gpus=1,
        system="gb200",
        backend="vllm",
        backend_version="current",
        database_mode="SOL",
    )

    fallback = MoECommFallback("context", "deepep_ht", 32, 8, 8, 1)

    def _row(e_workers: float, fallbacks: tuple[MoECommFallback, ...] = ()) -> dict:
        row = {
            "tp": 1,
            "pp": 1,
            "dp": 1,
            "ttft": 100.0,
            "tpot": 10.0,
            "tokens/s/gpu": 100.0,
            "tokens/s/user": 10.0,
            "power_w": 0.0,
            "(e)workers": e_workers,
        }
        row[MOE_COMM_FALLBACKS_COLUMN] = fallbacks
        return row

    args = argparse.Namespace(inclusive_tpot=False, deployment_target="dynamo-j2")
    caplog.set_level(logging.WARNING)
    with (
        patch(
            "aiconfigurator.cli.report_and_save.get_default_dynamo_version_mapping",
            return_value=("1.0.0", {"vllm": "current"}),
        ),
        patch(
            "aiconfigurator.cli.report_and_save.task_config_to_generator_config",
            return_value={},
        ) as bridge,
    ):
        save_results(
            args=args,
            best_configs={"agg": pd.DataFrame([_row(2.0, (fallback,)), _row(0.0)])},
            pareto_fronts={"agg": pd.DataFrame([_row(2.0, (fallback,))])},
            tasks=tasks,
            save_dir=str(tmp_path),
            backend="vllm",
        )

    assert bridge.call_count == 1  # only the non-EPD row reaches the bridge
    assert "generator artifacts skipped" in caplog.text
    agg_dir = next(tmp_path.glob("**/agg"))
    assert MOE_COMM_FALLBACKS_COLUMN not in pd.read_csv(agg_dir / "best_config_topn.csv").columns
    assert MOE_COMM_FALLBACKS_COLUMN not in pd.read_csv(agg_dir / "pareto.csv").columns
    sidecar = agg_dir / "top1" / "moe_comm_fallbacks.json"
    assert json.loads(sidecar.read_text()) == [
        {
            "comm_backend": "deepep_ht",
            "inference_phase": "context",
            "measurement_ep_size": 8,
            "measurement_node_num": 1,
            "requested_ep_size": 32,
            "requested_node_num": 8,
        }
    ]
    assert "requested EP32/node8; using EP8/node1 silicon data" in caplog.text
    assert str(sidecar) in caplog.text
