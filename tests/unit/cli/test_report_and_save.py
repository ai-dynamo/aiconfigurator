"""Unit tests for CLI summary/report rendering."""

from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest

from aiconfigurator.cli.report_and_save import _plot_worker_setup_table, log_final_summary

pytestmark = pytest.mark.unit


def _make_task_config():
    return SimpleNamespace(
        # Flat fields the new Task exposes directly.
        tpot=30.0,
        request_latency=1200.0,
        primary_model_path="Qwen/Qwen3-32B",
        is_moe=False,
        total_gpus=16,
        backend_name="trtllm",
        primary_backend_name="trtllm",
        serving_mode="agg",
    )


class TestReportAndSave:
    """Tests for CLI report rendering."""

    def test_plot_worker_setup_table_includes_request_rate_column(self):
        """Top-config table should show cluster-level request rate."""
        config_df = pd.DataFrame(
            {
                "backend": ["trtllm"],
                "tokens/s/gpu": [120.0],
                "tokens/s/user": [12.0],
                "request_rate": [5.5],
                "ttft": [100.0],
                "tpot": [20.0],
                "request_latency": [900.0],
                "concurrency": [2],
                "num_total_gpus": [8],
                "tp": [4],
                "pp": [2],
                "dp": [1],
                "bs": [8],
            }
        )

        table = _plot_worker_setup_table(
            exp_name="agg",
            config_df=config_df,
            total_gpus=16,
            tpot_target=30.0,
            top=1,
            is_moe=False,
            request_latency_target=None,
            show_power=False,
        )

        assert "req/s" in table
        assert "11.00" in table

    def test_log_final_summary_includes_request_rate(self):
        """Overall best configuration summary should include cluster-level request rate."""
        best_configs = {
            "agg": pd.DataFrame(
                {
                    "tokens/s/user": [12.0],
                    "request_rate": [5.5],
                    "num_total_gpus": [8],
                    "ttft": [100.0],
                    "tpot": [20.0],
                    "request_latency": [900.0],
                }
            )
        }

        with (
            patch("aiconfigurator.cli.report_and_save._plot_worker_setup_table", return_value=""),
            patch("aiconfigurator.cli.report_and_save.draw_pareto_to_string", return_value=""),
            patch("aiconfigurator.cli.report_and_save.logger.info") as mock_info,
        ):
            log_final_summary(
                chosen_exp="agg",
                best_throughputs={"agg": 120.0},
                best_configs=best_configs,
                pareto_fronts={},
                task_configs={"agg": _make_task_config()},
                mode="default",
            )

        logged_summary = mock_info.call_args.args[0]
        assert "Request Rate: 11.00 req/s" in logged_summary
