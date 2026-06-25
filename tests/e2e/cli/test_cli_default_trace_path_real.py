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
    return os.environ.get("AIC_RUN_SPICA_TRACE_E2E", "").lower() in {"1", "true", "yes", "on"}


@pytest.mark.skipif(not _enabled(), reason="set AIC_RUN_SPICA_TRACE_E2E=true to run real Spica trace replay")
@pytest.mark.timeout(900)
def test_cli_default_trace_path_real_default_sweep(tmp_path: Path):
    """Run the real default Spica trace sweep against an external Mooncake trace.

    This is intentionally opt-in: it requires Spica and compatible Dynamo replay
    bindings on PYTHONPATH, and the default 4-GPU Spica sweep takes minutes.
    """

    trace_path = os.environ.get("AIC_SPICA_TRACE_PATH")
    if not trace_path:
        pytest.fail("AIC_SPICA_TRACE_PATH must point to a Mooncake JSONL trace when AIC_RUN_SPICA_TRACE_E2E=true")
    if not Path(trace_path).exists():
        pytest.fail(f"AIC_SPICA_TRACE_PATH does not exist: {trace_path}")

    env = os.environ.copy()
    for key in (
        "AIC_SPICA_TRACE_SWEEP_ROUNDS",
        "AIC_SPICA_TRACE_PARALLEL_EVALS",
        "AIC_SPICA_TRACE_CANDIDATES_PER_ROUND",
    ):
        env.pop(key, None)

    save_dir = tmp_path / "trace-default"
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
        "--trace-path",
        trace_path,
        "--ttft",
        "8000",
        "--tpot",
        "200",
        "--max-seq-len",
        "8192",
        "--top-n",
        "2",
        "--save-dir",
        str(save_dir),
    ]

    completed = sp.run(cmd, capture_output=True, text=True, timeout=900, env=env)
    combined = f"{completed.stdout}\n{completed.stderr}"
    assert completed.returncode == 0, combined

    assert "sweep={'max_rounds': 3, 'parallel_evals': 16}" in combined
    sweep_done = re.search(
        r"smart-sweep done: (?P<feasible>\d+)/96 feasible, 0 gated, 0 backend-unsupported, "
        r"(?P<replay_failed>\d+) replay-failed",
        combined,
    )
    assert sweep_done, combined
    assert int(sweep_done.group("feasible")) > 0
    assert "AIConfigurator Final Results" in combined
    assert "Total GPUs: 4" in combined
    assert "Trace mode: --isl/--osl ignored; request lengths come from replay." in combined
    assert "Best Experiment Chosen:" in combined

    result_dirs = [path for path in save_dir.iterdir() if path.is_dir()]
    assert len(result_dirs) == 1
    result_dir = result_dirs[0]
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
