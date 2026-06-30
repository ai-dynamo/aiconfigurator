# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess as sp
import sys
from pathlib import Path

import pytest
import yaml

pytestmark = [pytest.mark.e2e, pytest.mark.sweep]


def _enabled() -> bool:
    return os.environ.get("AIC_RUN_SPICA_THOROUGH_E2E", "").lower() in {"1", "true", "yes", "on"}


def _yaml(path: Path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


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

    artifact_dir = Path(os.environ.get("AIC_SPICA_THOROUGH_E2E_ARTIFACT_DIR", tmp_path))
    artifact_dir.mkdir(parents=True, exist_ok=True)
    save_dir = artifact_dir / "save_dir"
    if save_dir.exists():
        shutil.rmtree(save_dir)
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

    (artifact_dir / "command.txt").write_text(f"{shlex.join(cmd)}\n", encoding="utf-8")
    completed = sp.run(cmd, capture_output=True, text=True, timeout=300, env=env)
    (artifact_dir / "stdout.txt").write_text(completed.stdout, encoding="utf-8")
    (artifact_dir / "stderr.txt").write_text(completed.stderr, encoding="utf-8")
    combined = f"{completed.stdout}\n{completed.stderr}"
    assert completed.returncode == 0, combined

    assert "sweep={'max_rounds': 1, 'parallel_evals': 4, 'candidates_per_round': 4}" in combined
    sweep_done = re.search(
        r"smart-sweep done: (?P<feasible>\d+)/(?P<attempts>\d+) replay attempt\(s\) feasible, "
        r"(?P<gated>\d+) gated, (?P<unsupported>\d+) backend-unsupported, "
        r"(?P<replay_failed>\d+) replay-failed, (?P<cache_hits>\d+) cache hit\(s\)",
        combined,
    )
    assert sweep_done, combined
    assert int(sweep_done.group("feasible")) == 4
    assert int(sweep_done.group("attempts")) == 4
    assert int(sweep_done.group("gated")) == 0
    assert int(sweep_done.group("unsupported")) == 0
    assert int(sweep_done.group("replay_failed")) == 0
    assert int(sweep_done.group("cache_hits")) == 0
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
    feasible_modes = [mode for mode in ("agg", "disagg") if (result_dir / mode / "exp_config.yaml").is_file()]
    assert feasible_modes
    for mode in feasible_modes:
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

        if mode == "agg":
            assert (result_dir / mode / "top1" / "agg_config.yaml").is_file()
        else:
            assert (result_dir / mode / "top1" / "prefill_config.yaml").is_file()
            assert (result_dir / mode / "top1" / "decode_config.yaml").is_file()


@pytest.mark.skipif(not _enabled(), reason="set AIC_RUN_SPICA_THOROUGH_E2E=true to run real Spica thorough sweep")
@pytest.mark.timeout(300)
def test_cli_native_thorough_config_preserves_dynamo_features(tmp_path: Path):
    """Run one real native-config candidate and validate its deployable feature contract."""

    env = os.environ.copy()
    env.update(
        {
            # The YAML is already bounded; these also prevent an inherited CI
            # override from expanding the run if native configs gain env overrides.
            "AIC_SPICA_THOROUGH_SWEEP_ROUNDS": "1",
            "AIC_SPICA_THOROUGH_PARALLEL_EVALS": "1",
            "AIC_SPICA_THOROUGH_CANDIDATES_PER_ROUND": "1",
            # The runtime's definitive KVBM-attachment signal is debug-level.
            "DYN_LOG": ("info,dynamo_mocker::scheduler=debug,dynamo_mocker::kvbm_offload::engine=debug"),
        }
    )
    artifact_dir = Path(os.environ.get("AIC_SPICA_THOROUGH_E2E_ARTIFACT_DIR", tmp_path))
    artifact_dir.mkdir(parents=True, exist_ok=True)
    save_dir = artifact_dir / "native_save_dir"
    if save_dir.exists():
        shutil.rmtree(save_dir)

    config_path = Path(__file__).parent / "data" / "spica_native_features.yaml"
    cmd = [
        sys.executable,
        "-m",
        "aiconfigurator.cli.main",
        "default",
        "--thorough-config",
        str(config_path),
        "--top-n",
        "1",
        "--generator-set",
        "ServiceConfig.port=8123",
        "--save-dir",
        str(save_dir),
    ]

    (artifact_dir / "native_command.txt").write_text(f"{shlex.join(cmd)}\n", encoding="utf-8")
    completed = sp.run(cmd, capture_output=True, text=True, timeout=300, env=env)
    (artifact_dir / "native_stdout.txt").write_text(completed.stdout, encoding="utf-8")
    (artifact_dir / "native_stderr.txt").write_text(completed.stderr, encoding="utf-8")
    combined = f"{completed.stdout}\n{completed.stderr}"
    assert completed.returncode == 0, combined

    assert "sweep={'max_rounds': 1, 'parallel_evals': 1, 'candidates_per_round': 1}" in combined
    assert "Synthetic Workload: ISL=128, OSL=16, request_rate=1.0, num_request_ratio=3.0" in combined
    assert re.search(
        r"smart-sweep done: 1/1 replay attempt\(s\) feasible, 0 gated, "
        r"0 backend-unsupported, 0 replay-failed, 0 cache hit\(s\)",
        combined,
    )
    assert combined.count("kvbm-offload: init_kvbm_offline attaching engine") == 2
    assert combined.count("kvbm-offload: building mock offload engine") == 2
    assert "kvbm-offload offline init failed" not in combined
    assert "Could not compute kv_bytes_per_token" not in combined

    result_dirs = [path for path in save_dir.iterdir() if path.is_dir()]
    assert len(result_dirs) == 1
    result_dir = result_dirs[0]
    assert "_thorough_spica_native_features_ttft0_tpot0_" in result_dir.name

    exp_config = _yaml(result_dir / "agg" / "exp_config.yaml")
    assert exp_config["ttft"] == 0.0
    assert exp_config["tpot"] == 0.0
    assert exp_config["objective_target"] == "throughput"
    assert exp_config["planner_optimization_target"] == "throughput"
    assert exp_config["primary_backend_version"] == "1.3.0rc10"
    assert exp_config["total_gpus"] == 2
    assert exp_config["min_gpu_budget"] == 2
    assert exp_config["min_endpoint"] == 2

    top_dir = result_dir / "agg" / "top1"
    candidate = _yaml(top_dir / "spica_candidate.yaml")
    candidate_config = candidate["config"]
    assert candidate_config["objective_target"] == "throughput"
    assert candidate_config["planner_optimization_target"] == "throughput"
    assert candidate_config["backend_version"] == "1.3.0rc10"
    assert candidate_config["planner_ttft_ms"] is None
    assert candidate_config["planner_itl_ms"] is None
    assert candidate_config["context_length"] == 4096
    assert candidate_config["gpu_budget"] == 2
    assert candidate_config["min_gpu_budget"] == 2
    assert candidate_config["min_endpoint"] == 2
    assert candidate_config["replicas"] == 2
    assert candidate_config["router_mode"] == "kv_router"
    assert candidate_config["host_cache_hit_weight"] == pytest.approx(0.75)
    assert candidate_config["disk_cache_hit_weight"] == pytest.approx(0.25)
    assert candidate_config["num_g2_blocks"] == 128
    assert candidate_config["kv_bytes_per_token"] == 131072
    assert candidate_config["offload_batch_size"] == 4
    assert candidate["metrics"]["planner_total_ticks"] >= 1

    generator_config = _yaml(top_dir / "generator_config.yaml")
    assert generator_config["ServiceConfig"]["port"] == 8123
    assert generator_config["WorkerConfig"]["agg_workers"] == 2
    assert generator_config["params"]["agg"]["max_seq_len"] == 4096
    assert _yaml(top_dir / "agg_config.yaml")["max_seq_len"] == 4096
    dyn_config = generator_config["DynConfig"]
    assert dyn_config["router_mode"] == "kv"
    router_config = dyn_config["router_config"]
    assert router_config["host_cache_hit_weight"] == pytest.approx(0.75)
    assert router_config["disk_cache_hit_weight"] == pytest.approx(0.25)

    planner_config = dyn_config["planner_config"]
    assert planner_config["optimization_target"] == "throughput"
    assert planner_config["max_gpu_budget"] == 2
    assert planner_config["min_gpu_budget"] == 2
    assert planner_config["min_endpoint"] == 2
    assert planner_config["enable_load_scaling"] is True
    assert planner_config["enable_throughput_scaling"] is False
    assert planner_config["load_adjustment_interval_seconds"] == 1
    assert "ttft_ms" not in planner_config
    assert "itl_ms" not in planner_config

    kvbm_config = dyn_config["kvbm_config"]
    assert kvbm_config["cpu_cache_override_num_blocks"] == 128
    assert kvbm_config["max_transfer_batch_size"] == 4
    assert kvbm_config["max_concurrent_transfers"] == 4

    k8s_deploy = _yaml(top_dir / "k8s_deploy.yaml")
    services = k8s_deploy["spec"]["services"]
    frontend_args = services["Frontend"]["extraPodSpec"]["mainContainer"]["args"]
    assert frontend_args[:4] == ["--http-port", "8123", "--router-mode", "kv"]

    # Parse exactly what the generated pod will pass to Dynamo.
    from dynamo.frontend.frontend_args import FrontendArgGroup

    frontend_parser = argparse.ArgumentParser()
    FrontendArgGroup().add_arguments(frontend_parser)
    parsed_frontend = frontend_parser.parse_args(frontend_args)
    assert parsed_frontend.http_port == 8123
    assert parsed_frontend.router_mode == "kv"
    assert parsed_frontend.host_cache_hit_weight == pytest.approx(0.75)
    assert parsed_frontend.disk_cache_hit_weight == pytest.approx(0.25)

    planner_container = services["Planner"]["extraPodSpec"]["mainContainer"]
    planner_payload = json.loads(planner_container["args"][planner_container["args"].index("--config") + 1])
    assert planner_payload == planner_config
    from dynamo.planner.config.planner_config import PlannerConfig

    validated_planner = PlannerConfig.model_validate(planner_payload)
    assert validated_planner.optimization_target == "throughput"
    assert validated_planner.max_gpu_budget == 2
    assert validated_planner.min_gpu_budget == 2
    assert validated_planner.min_endpoint == 2

    worker = services["TRTLLMWorker"]
    worker_env = {item["name"]: item["value"] for item in worker["envs"]}
    assert worker_env["DYN_KVBM_CPU_CACHE_OVERRIDE_NUM_BLOCKS"] == "128"
    assert worker_env["DYN_KVBM_MAX_TRANSFER_BATCH_SIZE"] == "4"
    assert worker_env["DYN_KVBM_MAX_CONCURRENT_TRANSFERS"] == "4"
    worker_script = worker["extraPodSpec"]["mainContainer"]["args"][0]
    assert "args+=(--connector kvbm)" in worker_script

    run_script = (top_dir / "run_0.sh").read_text(encoding="utf-8")
    assert '--http-port "8123"' in run_script
    assert "--router-host-cache-hit-weight 0.75" in run_script
    assert "--router-disk-cache-hit-weight 0.25" in run_script
    assert "--connector kvbm" in run_script
    assert "DYN_KVBM_MAX_TRANSFER_BATCH_SIZE=4" in run_script
    assert "DYN_KVBM_MAX_CONCURRENT_TRANSFERS=4" in run_script
