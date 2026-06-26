# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import re
import subprocess as sp
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.e2e, pytest.mark.sweep]


def _enabled() -> bool:
    return os.environ.get("AIC_RUN_SPICA_THOROUGH_E2E", "").lower() in {"1", "true", "yes", "on"}


@pytest.mark.skipif(not _enabled(), reason="set AIC_RUN_SPICA_THOROUGH_E2E=true to run real Spica thorough sweep")
@pytest.mark.timeout(300)
def test_cli_default_thorough_sweep_real_static_workload(tmp_path: Path):
    """Run a small real Spica thorough sweep from ordinary default CLI inputs."""

    env = os.environ.copy()
    env.update(
        {
            "AIC_SPICA_THOROUGH_SWEEP_ROUNDS": "1",
            "AIC_SPICA_THOROUGH_PARALLEL_EVALS": "4",
            "AIC_SPICA_THOROUGH_CANDIDATES_PER_ROUND": "4",
            "AIC_SPICA_THOROUGH_SYNTHETIC_CONCURRENCY": "1",
            "AIC_SPICA_THOROUGH_SYNTHETIC_NUM_REQUEST_RATIO": "4",
        }
    )
    for key in (
        "AIC_SPICA_TRACE_SWEEP_ROUNDS",
        "AIC_SPICA_TRACE_PARALLEL_EVALS",
        "AIC_SPICA_TRACE_CANDIDATES_PER_ROUND",
    ):
        env.pop(key, None)

    save_dir = tmp_path / "thorough-default"
    cmd = [
        sys.executable,
        "-m",
        "aiconfigurator.cli.main",
        "default",
        "--model-path",
        "meta-llama/Meta-Llama-3.1-8B",
        "--total-gpus",
        "4",
        "--system",
        "gb200",
        "--backend",
        "trtllm",
        "--isl",
        "128",
        "--osl",
        "16",
        "--ttft",
        "8000",
        "--tpot",
        "200",
        "--max-seq-len",
        "8192",
        "--top-n",
        "2",
        "--thorough-sweep",
        "--save-dir",
        str(save_dir),
    ]

    completed = sp.run(cmd, capture_output=True, text=True, timeout=300, env=env)
    combined = f"{completed.stdout}\n{completed.stderr}"
    assert completed.returncode == 0, combined

    assert "sweep={'max_rounds': 1, 'parallel_evals': 4, 'candidates_per_round': 4}" in combined
    sweep_done = re.search(
        r"smart-sweep done: (?P<feasible>\d+)/8 feasible, 0 gated, 0 backend-unsupported, "
        r"(?P<replay_failed>\d+) replay-failed",
        combined,
    )
    assert sweep_done, combined
    assert int(sweep_done.group("feasible")) > 0
    assert "AIConfigurator Final Results" in combined
    assert "Total GPUs: 4" in combined
    assert "Synthetic Workload: ISL=128, OSL=16, concurrency=1, num_request_ratio=4.0" in combined
    assert "Best Experiment Chosen:" in combined

    result_dirs = [path for path in save_dir.iterdir() if path.is_dir()]
    assert len(result_dirs) == 1
    result_dir = result_dirs[0]
    assert result_dir.name.startswith(
        "meta-llama_Meta-Llama-3.1-8B_gb200_trtllm_thorough_isl128_osl16_ttft8000_tpot200_"
    )
    assert (result_dir / "spica_candidates.yaml").is_file()
    assert (result_dir / "spica_candidates.csv").is_file()
    assert (result_dir / "pareto.csv").is_file()
    assert (result_dir / "pareto_frontier.png").is_file()
    for mode in ("agg", "disagg"):
        assert (result_dir / mode / "exp_config.yaml").is_file()
        assert (result_dir / mode / "best_config_topn.csv").is_file()
        assert (result_dir / mode / "pareto.csv").is_file()
        assert (result_dir / mode / "top1" / "generator_config.yaml").is_file()
        assert (result_dir / mode / "top1" / "bench_run.sh").is_file()
        assert (result_dir / mode / "top1" / "k8s_bench.yaml").is_file()
        assert (result_dir / mode / "top1" / "k8s_deploy.yaml").is_file()
        assert (result_dir / mode / "top1" / "run_0.sh").is_file()
        assert (result_dir / mode / "top1" / "sflow.yaml").is_file()
        assert (result_dir / mode / "top1" / "spica_candidate.yaml").is_file()

    assert (result_dir / "agg" / "top1" / "agg_config.yaml").is_file()
    assert (result_dir / "disagg" / "top1" / "prefill_config.yaml").is_file()
    assert (result_dir / "disagg" / "top1" / "decode_config.yaml").is_file()
