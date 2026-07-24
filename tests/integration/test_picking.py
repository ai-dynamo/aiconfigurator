# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
E2E tests for the three picking modes: default, load-match, and autoscale.

Each test runs the full pipeline (model + traffic + SLA -> picking result)
and optionally saves DGD configs to a temporary directory for inspection.
"""

import os

import pytest
import yaml

from aiconfigurator.cli.main import _execute_tasks, build_default_tasks
from aiconfigurator.generator.api import generate_backend_artifacts
from aiconfigurator.generator.module_bridge import task_config_to_generator_config
from aiconfigurator.sdk.task_v2 import Task

pytestmark = pytest.mark.integration

MODEL = "Qwen/Qwen3-32B"
SYSTEM = "h200_sxm"
BACKEND = "trtllm"


def _save_chosen_dgd(best_configs, tasks, chosen_exp, output_dir):
    """Save the rank-1 DGD config for the chosen experiment."""
    config_df = best_configs.get(chosen_exp)
    if config_df is None or config_df.empty:
        return

    if "backend" in config_df.columns and not config_df.empty:
        first_backend = config_df["backend"].iloc[0]
        tc_key = f"{chosen_exp}_{first_backend}"
        tc = tasks.get(tc_key, tasks.get(chosen_exp))
    else:
        tc = tasks[chosen_exp]

    row = config_df.iloc[0]

    # For load-match: use total_gpus_needed instead of sweep ceiling
    original_total_gpus = tc.total_gpus
    if "total_gpus_needed" in row.index and row["total_gpus_needed"] > 0:
        tc.total_gpus = int(row["total_gpus_needed"])

    cfg = task_config_to_generator_config(task_config=tc, result_df=row)
    tc.total_gpus = original_total_gpus

    exp_dir = os.path.join(output_dir, chosen_exp)
    os.makedirs(exp_dir, exist_ok=True)

    with open(os.path.join(exp_dir, "generator_config.yaml"), "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    try:
        generate_backend_artifacts(
            params=cfg,
            backend=tc.primary_backend_name,
            backend_version=tc.primary_backend_version,
            output_dir=exp_dir,
        )
    except Exception:
        pass  # artifact generation may fail without dynamo; config is the main output


class TestDefaultPicking:
    """Default mode: maximize throughput for N GPUs under SLA."""

    def test_agg_vs_disagg(self, tmp_path):
        """8 GPUs, compare agg vs disagg under relaxed SLA."""
        tasks = build_default_tasks(
            model_path=MODEL,
            total_gpus=8,
            system=SYSTEM,
            backend=BACKEND,
            isl=4000,
            osl=1000,
            ttft=2000,
            tpot=50,
        )
        chosen, best_configs, _, _, best_latencies = _execute_tasks(
            tasks,
            mode="default",
            top_n=3,
        )
        assert chosen in ("agg", "disagg")
        assert best_latencies[chosen]["ttft"] > 0
        assert best_latencies[chosen]["tpot"] > 0

        _save_chosen_dgd(best_configs, tasks, chosen, str(tmp_path))
        assert (tmp_path / chosen / "generator_config.yaml").exists()

    def test_request_latency_constraint(self, tmp_path):
        """Default mode with request_latency as the SLA constraint."""
        request_latency = 35000
        tasks = build_default_tasks(
            model_path=MODEL,
            total_gpus=8,
            system=SYSTEM,
            backend=BACKEND,
            isl=4000,
            osl=1000,
            ttft=request_latency,
            request_latency=request_latency,
        )
        chosen, best_configs, _, _, best_latencies = _execute_tasks(
            tasks,
            mode="default",
            top_n=3,
        )
        assert chosen in ("agg", "disagg")

        _save_chosen_dgd(best_configs, tasks, chosen, str(tmp_path))
        assert (tmp_path / chosen / "generator_config.yaml").exists()


class TestLoadMatchPicking:
    """Load-match mode: minimize GPUs for a target load under SLA."""

    def test_by_request_rate(self, tmp_path):
        """Find min GPUs for target_request_rate=5.0 req/s."""
        tasks = build_default_tasks(
            model_path=MODEL,
            total_gpus=64,
            system=SYSTEM,
            backend=BACKEND,
            isl=4000,
            osl=1000,
            ttft=2000,
            tpot=50,
        )
        chosen, best_configs, _, _, best_latencies = _execute_tasks(
            tasks,
            mode="default",
            top_n=3,
            target_request_rate=5.0,
            max_total_gpus=64,
        )
        for df in best_configs.values():
            if df is not None and not df.empty:
                assert "replicas_needed" in df.columns
                assert "total_gpus_needed" in df.columns

        assert "total_gpus_needed" in best_latencies[chosen]

        _save_chosen_dgd(best_configs, tasks, chosen, str(tmp_path))
        assert (tmp_path / chosen / "generator_config.yaml").exists()

    def test_by_concurrency(self, tmp_path):
        """Find min GPUs for target_concurrency=50."""
        tasks = build_default_tasks(
            model_path=MODEL,
            total_gpus=64,
            system=SYSTEM,
            backend=BACKEND,
            isl=4000,
            osl=1000,
            ttft=2000,
            tpot=50,
        )
        chosen, best_configs, _, _, best_latencies = _execute_tasks(
            tasks,
            mode="default",
            top_n=3,
            target_concurrency=50,
            max_total_gpus=64,
        )
        for df in best_configs.values():
            if df is not None and not df.empty:
                assert "replicas_needed" in df.columns
                assert "total_gpus_needed" in df.columns

        _save_chosen_dgd(best_configs, tasks, chosen, str(tmp_path))
        assert (tmp_path / chosen / "generator_config.yaml").exists()


class TestRecommendPicking:
    """Recommend mode: find minimum GPUs for a target load via cli_recommend."""

    def test_recommend_by_request_rate(self):
        """cli_recommend should return results with total_gpus_needed."""
        from aiconfigurator.cli.api import cli_recommend

        result = cli_recommend(
            model_path=MODEL,
            system=SYSTEM,
            backend=BACKEND,
            target_request_rate=5.0,
            isl=4000,
            osl=1000,
            ttft=2000,
            tpot=50,
        )

        assert result.chosen_exp in result.best_configs
        has_results = False
        for mode, df in result.best_configs.items():
            if df is not None and not df.empty:
                has_results = True
                assert "replicas_needed" in df.columns, f"{mode} missing replicas_needed"
                assert "total_gpus_needed" in df.columns, f"{mode} missing total_gpus_needed"
                assert (df["total_gpus_needed"] > 0).all(), f"{mode} has zero GPU recommendations"
                assert (df["replicas_needed"] >= 1).all(), f"{mode} has replicas < 1"
        assert has_results, "cli_recommend returned no results"

    def test_recommend_higher_rate_needs_more_gpus(self):
        """Doubling the target rate should need at least as many GPUs."""
        from aiconfigurator.cli.api import cli_recommend

        common = dict(
            model_path=MODEL,
            system=SYSTEM,
            backend=BACKEND,
            isl=4000,
            osl=1000,
            ttft=2000,
            tpot=50,
        )
        result_low = cli_recommend(target_request_rate=2.0, **common)
        result_high = cli_recommend(target_request_rate=20.0, **common)

        def min_gpus(cli_result):
            for df in cli_result.best_configs.values():
                if df is not None and not df.empty and "total_gpus_needed" in df.columns:
                    return int(df["total_gpus_needed"].min())
            return 0

        low_gpus = min_gpus(result_low)
        high_gpus = min_gpus(result_high)
        assert high_gpus >= low_gpus, (
            f"Higher rate (20 req/s) recommended fewer GPUs ({high_gpus}) than lower rate (2 req/s, {low_gpus})"
        )


class TestAutoscalePicking:
    """Autoscale mode: independent P/D picking, 1 replica each."""

    def test_autoscale(self, tmp_path):
        """Relaxed SLA: should find valid P and D engines."""
        task = Task(
            serving_mode="disagg",
            prefill_model_path=MODEL,
            decode_model_path=MODEL,
            prefill_system_name=SYSTEM,
            decode_system_name=SYSTEM,
            prefill_backend_name=BACKEND,
            decode_backend_name=BACKEND,
            total_gpus=8,
            isl=4000,
            osl=1000,
            ttft=2000,
            tpot=50,
        )
        pareto_df = task.run(autoscale=True)
        assert pareto_df is not None and not pareto_df.empty
        assert (pareto_df["(p)workers"] == 1).all()
        assert (pareto_df["(d)workers"] == 1).all()

        best_row = pareto_df.iloc[0]
        assert best_row["ttft"] > 0
        assert best_row["tpot"] > 0

        _save_chosen_dgd({"disagg": pareto_df.head(1)}, {"disagg": task}, "disagg", str(tmp_path))
        assert (tmp_path / "disagg" / "generator_config.yaml").exists()

    def test_autoscale_tight_sla(self):
        """Tight SLA: should still return closest-match configs (not empty)."""
        task = Task(
            serving_mode="disagg",
            prefill_model_path=MODEL,
            decode_model_path=MODEL,
            prefill_system_name=SYSTEM,
            decode_system_name=SYSTEM,
            prefill_backend_name=BACKEND,
            decode_backend_name=BACKEND,
            total_gpus=8,
            isl=4000,
            osl=1000,
            ttft=600,
            tpot=25,
        )
        pareto_df = task.run(autoscale=True)
        # With closest-match fallback, should always return something
        assert pareto_df is not None and not pareto_df.empty
        assert (pareto_df["(p)workers"] == 1).all()
        assert (pareto_df["(d)workers"] == 1).all()
