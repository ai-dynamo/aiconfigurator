# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for plain (non-TTY) CLI summary output without ANSI escapes."""

from __future__ import annotations

import logging
import sys
from unittest.mock import MagicMock

import pandas as pd
import pytest

from aiconfigurator.cli.report_and_save import _plot_worker_setup_table, log_final_summary
from aiconfigurator.logging_utils import setup_logging, use_plain_cli_output
from aiconfigurator.sdk.pareto_analysis import draw_pareto_to_string

pytestmark = pytest.mark.unit

_ESC = "\x1b["


@pytest.fixture(autouse=True)
def mock_stdout_isatty(monkeypatch):
    mock = MagicMock(return_value=True)
    monkeypatch.setattr(sys.stdout, "isatty", mock)
    return mock


@pytest.fixture(autouse=True)
def _reset_logging_after_test(monkeypatch):
    yield
    setup_logging(no_color=False)
    monkeypatch.delenv("NO_COLOR", raising=False)


def test_cli_parser_accepts_no_color(cli_parser):
    base = [
        "default",
        "--model-path",
        "Qwen/Qwen3-32B",
        "--total-gpus",
        "8",
        "--system",
        "h200_sxm",
    ]
    assert cli_parser.parse_args([*base, "--no-color"]).no_color is True


@pytest.mark.parametrize("isatty", [True, False])
def test_use_plain_when_stdout_not_a_tty(mock_stdout_isatty, isatty):
    mock_stdout_isatty.return_value = isatty
    assert use_plain_cli_output() is not isatty


@pytest.mark.parametrize("nocolor_env_set", [True, False])
def test_use_plain_when_no_color_env_set(monkeypatch, nocolor_env_set):
    if nocolor_env_set:
        monkeypatch.setenv("NO_COLOR", "1")
    assert use_plain_cli_output() is nocolor_env_set


def test_colored_formatter_force_no_color_disables_colors():
    setup_logging(no_color=True)
    handler = logging.getLogger().handlers[0]
    assert handler.formatter.use_colors is False
    assert handler.formatter.force_no_color is True
    assert use_plain_cli_output() is True


@pytest.mark.parametrize("use_ansi", [True, False])
def test_draw_pareto_to_string(use_ansi):
    setup_logging(no_color=not use_ansi)
    df = pd.DataFrame({"tokens/s/user": [1.0, 2.0], "tokens/s/gpu_cluster": [10.0, 20.0]})
    out = draw_pareto_to_string(
        "Test",
        [{"df": df, "label": "a"}],
        highlight={"df": df.head(1), "label": "best"},
    )
    assert (_ESC in out) == use_ansi


@pytest.mark.parametrize("use_ansi", [True, False])
def test_log_final_summary(caplog, use_ansi):
    caplog.set_level("INFO")
    setup_logging(no_color=not use_ansi)
    # setup_logging clears root handlers; reattach pytest's capture handler.
    logging.getLogger().addHandler(caplog.handler)

    model_path = "unit-test-model"
    tc = MagicMock()
    tc.config.model_path = model_path
    tc.config.is_moe = False
    tc.config.runtime_config.tpot = 50.0
    tc.config.runtime_config.request_latency = None
    tc.backend_name = "trtllm"
    tc.total_gpus = 8

    best_row = {
        "backend": "trtllm",
        "tokens/s/gpu": 100.0,
        "tokens/s/user": 50.0,
        "tokens/s/gpu_cluster": 100.0,
        "request_rate": 2.0,
        "ttft": 100.0,
        "request_latency": 200.0,
        "tpot": 10.0,
        "concurrency": 4.0,
        "num_total_gpus": 8,
        "tp": 4,
        "pp": 2,
        "dp": 1,
        "moe_tp": 1,
        "moe_ep": 1,
        "bs": 64,
        "power_w": 400.0,
    }
    best_configs = {"agg": pd.DataFrame([best_row])}
    pareto_df = pd.DataFrame(
        {
            "tokens/s/user": [1.0, 2.0],
            "tokens/s/gpu_cluster": [10.0, 15.0],
        }
    )
    pareto_fronts = {"agg": pareto_df}

    log_final_summary(
        chosen_exp="agg",
        best_throughputs={"agg": 100.0, "disagg": 0.0},
        best_configs=best_configs,
        pareto_fronts=pareto_fronts,
        task_configs={"agg": tc},
        mode="default",
        pareto_x_axis={"agg": "tokens/s/user"},
        top_n=1,
    )

    logged = "\n".join(r.message for r in caplog.records)
    assert (_ESC in logged) == use_ansi

    text = _plot_worker_setup_table(
        "agg",
        best_configs["agg"],
        total_gpus=8,
        tpot_target=50.0,
        top=3,
        is_moe=False,
        request_latency_target=None,
        show_power=True,
    )
    assert (_ESC in text) == use_ansi
    assert "tokens/s/gpu" in text


def test_draw_pareto_plain_output_is_pure_ascii():
    """Ensure piped Pareto chart output is pure ASCII (no mojibake under `cat -v`).

    Prior to this fix, plotext's Unicode box-drawing characters (U+2500-U+257F)
    and block elements (U+2580-U+259F) appeared as M-bM-^T... sequences when
    piped through `cat -v`, breaking CI logs and scripted post-processing.
    """
    setup_logging(no_color=True)
    df = pd.DataFrame({"tokens/s/user": [1.0, 2.0, 3.0], "tokens/s/gpu_cluster": [10.0, 40.0, 90.0]})
    out = draw_pareto_to_string(
        "cat-v test",
        [{"df": df, "label": "series"}],
    )
    non_ascii = [c for c in out if ord(c) > 127]
    assert non_ascii == [], (
        f"Plain output contains non-ASCII characters that would break `cat -v`: "
        f"{[f'U+{ord(c):04X}' for c in set(non_ascii)]}"
    )
