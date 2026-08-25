# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""EPD CLI misuse must fail loud, and EPD report arithmetic must stay
consistent with the row totals."""

import argparse
import logging

import pandas as pd
import pytest

from aiconfigurator.cli.main import _run_estimate_mode, build_default_tasks
from aiconfigurator.cli.report_and_save import _plot_worker_setup_table
from aiconfigurator.sdk.performance_result import MOE_COMM_FALLBACKS_COLUMN, MoECommFallback

pytestmark = pytest.mark.unit


def test_epd_agg_worker_table_gpu_decomposition_includes_cp():
    config_df = pd.DataFrame(
        [
            {
                "backend": "sglang",
                "tokens/s/gpu": 10.0,
                "tokens/s/user": 1.0,
                "request_rate": 1.0,
                "ttft": 100.0,
                "tpot": 10.0,
                "request_latency": 200.0,
                "concurrency": 1,
                "bs": 8,
                "tp": 1,
                "pp": 1,
                "dp": 1,
                "cp": 2,
                "num_total_gpus": 8,
                "(a)workers": 3,
                "(e)workers": 1,
                "(e)tp": 2,
                "(e)bs": 4,
            }
        ]
    )

    table = _plot_worker_setup_table(
        "agg",
        config_df,
        total_gpus=8,
        tpot_target=50.0,
        top=1,
        is_moe=False,
        request_latency_target=None,
        show_power=False,
    )

    # 3 agg workers x (tp*pp*dp*cp = 2 GPUs) + 1 encode worker x 2 = 8.
    assert "8 (=3x2+1x2(e))" in table


def test_estimate_encoder_flags_require_enable_epd():
    args = argparse.Namespace(enable_epd=False, encoder_tp=4, encoder_batch_size=None, encoder_num_workers=None)
    with pytest.raises(SystemExit, match="require --enable-epd"):
        _run_estimate_mode(args)


def test_estimate_epd_rejects_role_specific_memory_fractions(cli_parser):
    args = cli_parser.parse_args(
        [
            "estimate",
            "--model-path",
            "Qwen/Qwen3-32B",
            "--system",
            "h200_sxm",
            "--estimate-mode",
            "disagg",
            "--enable-epd",
            "--prefill-free-gpu-memory-fraction",
            "0.85",
        ]
    )

    with pytest.raises(SystemExit, match="not supported with --enable-epd"):
        _run_estimate_mode(args)


@pytest.mark.parametrize("option", ["--prefill-max-seq-len", "--decode-max-seq-len"])
def test_estimate_epd_rejects_role_specific_max_seq_len(cli_parser, option):
    args = cli_parser.parse_args(
        [
            "estimate",
            "--model-path",
            "Qwen/Qwen3-32B",
            "--system",
            "h200_sxm",
            "--estimate-mode",
            "disagg",
            "--enable-epd",
            option,
            "8192",
        ]
    )

    with pytest.raises(SystemExit, match="not supported with --enable-epd"):
        _run_estimate_mode(args)


def test_default_afd_serving_mode_rejects_enable_epd():
    with pytest.raises(ValueError, match="'afd' does not support EPD"):
        build_default_tasks(
            model_path="Qwen/Qwen3-VL-8B-Instruct",
            total_gpus=8,
            system="h200_sxm",
            backend="sglang",
            serving_mode="afd",
            enable_epd=True,
        )


def test_estimate_epd_warns_from_task_row_before_early_return(cli_parser, monkeypatch, caplog):
    from aiconfigurator.sdk.task_v2 import Task

    fallback = MoECommFallback("context", "deepep_ht", 32, 8, 8, 1)

    class FakeTask:
        def run_single_agg(self, **_kwargs):
            return {
                "ttft": 1.0,
                "tpot": 0.0,
                "request_latency": 1.0,
                "power_w": 0.0,
                "power_coverage": 1.0,
                MOE_COMM_FALLBACKS_COLUMN: (fallback,),
            }

    monkeypatch.setattr(Task, "from_cli", classmethod(lambda cls, **_kwargs: FakeTask()))
    args = cli_parser.parse_args(
        [
            "estimate",
            "--model-path",
            "Qwen/Qwen3-VL-8B-Instruct",
            "--system",
            "gb200",
            "--estimate-mode",
            "agg",
            "--enable-epd",
            "--encoder-tp",
            "1",
            "--nextn",
            "0",
        ]
    )

    with caplog.at_level(logging.WARNING, logger="aiconfigurator.cli.main"):
        _run_estimate_mode(args)

    assert (
        "Estimated MoE communication latency used fallback silicon data: "
        "context/deepep_ht: requested EP32/node8; using EP8/node1 silicon data." in caplog.messages
    )
