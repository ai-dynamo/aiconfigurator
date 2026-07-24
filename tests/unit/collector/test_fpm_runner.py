# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

import collector.fpm_forward.runner as fpm_runner
from collector.fpm_forward.config import PrefillSamplingProfile
from collector.fpm_forward.planner import BackendPolicy, FPMCell
from collector.fpm_forward.runner import (
    REMOTE_EXIT_MARKER,
    REMOTE_WORKDIR,
    KubernetesCellRunner,
    _cell_generator_overrides,
    _configured_sampling_metadata,
    _render_cell,
    _required_attempt_id,
    _runtime_collection_summary,
    _runtime_timing_summary,
    _validate_runtime_collection,
    run_collection,
)
from collector.fpm_forward.types import ParallelTopology

pytestmark = pytest.mark.unit


def _runner(tmp_path) -> KubernetesCellRunner:
    runner = object.__new__(KubernetesCellRunner)
    runner.namespace = "test"
    runner.cell_dir = tmp_path
    runner.expected_labels = {}
    return runner


def _write_provenance(path: Path, *, cell_id: str, plan_sha256: str = "plan-sha", attempt_id: str = "attempt"):
    path.write_text(
        json.dumps(
            {
                "schema_name": "aic_fpm_collector_provenance",
                "schema_version": 1,
                "cell_id": cell_id,
                "plan_sha256": plan_sha256,
                "attempt_id": attempt_id,
                "runtime": {"backend": "vllm", "backend_version": "0.24.0"},
            }
        )
    )


def _cell(*, phase: str = "prefill", dp: int = 1, strategy: str = "dep") -> FPMCell:
    return FPMCell(
        cell_id=f"cell-{phase}",
        workload_kind=phase,
        topology=ParallelTopology(tp=1, pp=1, dp=dp, moe_tp=1, moe_ep=dp, cp=1),
        weight_quantization="nvfp4",
        kv_cache_dtype="fp8",
        backend_policy=BackendPolicy("baseline", "baseline", {}, {}),
        parallel_strategy=strategy,
        gemm_quant_mode="nvfp4",
        moe_quant_mode="nvfp4",
        fmha_quant_mode="fp8",
        comm_quant_mode="half",
    )


def _plan(cell: FPMCell):
    prefill_sampling = PrefillSamplingProfile.build(max_isl=8192, max_batch_size=None)
    return SimpleNamespace(
        sha256="plan-sha",
        model_path="nvidia/GLM-5.2-NVFP4",
        system="b200_sxm",
        backend="vllm",
        cells=(cell,),
        options=SimpleNamespace(
            warmup_iterations=3,
            vllm_max_model_len=-1,
            prefill_sampling=prefill_sampling,
        ),
        capability=SimpleNamespace(architecture="GlmMoeDsaForCausalLM"),
        to_dict=lambda: {"sha256": "plan-sha"},
    )


def _native_payload(*, phase: str, rank: int, dp: int, run_id: str = "run") -> dict:
    point = {
        "point_type": phase,
        "benchmark_id": 1,
        "total_prefill_tokens": 257 if phase == "prefill" else 0,
        "total_kv_read_tokens": 128,
        "batch_size": 4,
        "expected_cudagraph_mode": "PIECEWISE" if phase == "prefill" else "FULL",
        "expected_capture_size": 272 if phase == "prefill" else 4,
        "padding_tokens": 15 if phase == "prefill" else 0,
        "sample_reasons": ["post_capture"] if phase == "prefill" else ["capture"],
    }
    scheduled = {
        "num_prefill_requests": 4 if phase == "prefill" else 0,
        "sum_prefill_tokens": 257 if phase == "prefill" else 0,
        "sum_prefill_kv_tokens": 128 if phase == "prefill" else 0,
        "num_decode_requests": 0 if phase == "prefill" else 4,
        "sum_decode_kv_tokens": 0 if phase == "prefill" else 128,
    }
    rank_results = []
    for dp_rank in range(dp):
        rank_fpm = {
            "counter_id": 1,
            "dp_rank": dp_rank,
            "wall_time": 0.01 + dp_rank / 1000,
            "scheduled_requests": scheduled,
        }
        rank_results.append({"dp_rank": dp_rank, "fpms": [rank_fpm]})
    local_fpm = rank_results[rank]["fpms"][0]
    group_wall_time = max(item["fpms"][0]["wall_time"] for item in rank_results)
    return {
        "schema_version": 2,
        "artifact_type": "rank",
        "status": "complete",
        "valid": True,
        "usable": True,
        "timing_valid": True,
        "run_id": run_id,
        "grid_digest": "grid",
        "config": {"mode": phase},
        "coverage": {"expected_points": 1, "completed_points": 1, "skipped_points": 0},
        "dp": {"rank": rank, "size": dp},
        "results": [
            {
                "point": point,
                "fpms": [local_fpm],
            }
        ],
        "iteration_groups": [
            {
                "benchmark_id": 1,
                "point": point,
                "expected_dp_ranks": list(range(dp)),
                "complete": True,
                "wall_time": group_wall_time,
                "rank_results": rank_results,
            }
        ],
        "skipped_points": [],
        "missing_phases": [],
        "timing": {
            "benchmark_elapsed_seconds": 10.0 + rank,
            "measured_iteration_seconds": group_wall_time,
        },
    }


def test_apply_skips_client_validation_and_verifies_created_object(tmp_path):
    runner = _runner(tmp_path)
    runner.manifest = tmp_path / "k8s_deploy.yaml"
    runner.manifest.write_text("apiVersion: v1\nkind: Pod\n")
    runner.kind = "Pod"
    runner.name = "pod-0"
    calls = []

    def kubectl(*args, **kwargs):
        calls.append(args)
        return subprocess.CompletedProcess(
            args,
            0,
            stdout=json.dumps({"kind": "Pod", "metadata": {"name": "pod-0", "labels": {}}}),
            stderr="",
        )

    runner._kubectl = kubectl
    runner.apply()

    assert calls[0][:2] == ("apply", "--validate=false")
    assert len(calls) == 1
    assert calls[0][-2:] == ("-o", "json")


def test_apply_rejects_masked_failure_without_created_object(tmp_path):
    runner = _runner(tmp_path)
    runner.manifest = tmp_path / "k8s_deploy.yaml"
    runner.manifest.write_text("apiVersion: v1\nkind: Pod\n")
    runner.kind = "Pod"
    runner.name = "pod-0"
    runner._kubectl = lambda *args, **kwargs: subprocess.CompletedProcess(
        args,
        0,
        stdout="",
        stderr="error: context deadline exceeded\n",
    )

    with pytest.raises(RuntimeError, match="no verifiable object"):
        runner.apply()


def test_exec_checked_rejects_masked_remote_failure(tmp_path):
    runner = _runner(tmp_path)
    runner._kubectl = lambda *args, **kwargs: subprocess.CompletedProcess(
        args,
        0,
        stdout=f"{REMOTE_EXIT_MARKER}127\n",
        stderr="command terminated with exit code 127\n",
    )

    with pytest.raises(RuntimeError, match="remote_exit=127"):
        runner._exec_checked("pod-0", ["bash", f"{REMOTE_WORKDIR}/run_with_etcd.sh"], timeout=10)


def test_prepare_attempt_clears_results_and_writes_runtime_provenance(tmp_path):
    runner = _runner(tmp_path)
    calls = []
    runner._exec_checked = lambda pod, command, timeout: calls.append((pod, command, timeout))

    runner.prepare_attempt(
        ["pod-0"],
        cell_id="cell-prefill",
        plan_sha256="plan-sha",
        attempt_id="attempt-1",
    )

    assert calls[0][1][:2] == ["find", "/results"]
    assert calls[1][1][:2] == ["python3", "-c"]
    payload = json.loads(calls[1][1][3])
    assert payload == {
        "schema_name": "aic_fpm_collector_provenance",
        "schema_version": 1,
        "cell_id": "cell-prefill",
        "plan_sha256": "plan-sha",
        "attempt_id": "attempt-1",
    }
    assert calls[1][1][4] == "collector-provenance.json"


def test_passed_checkpoint_requires_attempt_identity():
    with pytest.raises(ValueError, match="has no attempt identity"):
        _required_attempt_id({"status": "passed"}, "cell-prefill")


def test_cleanup_verifies_workload_and_pods_are_deleted(tmp_path):
    runner = _runner(tmp_path)
    runner.manifest = tmp_path / "k8s_deploy.yaml"
    runner.kind = "Pod"
    runner.name = "pod-0"
    responses = iter(
        [
            subprocess.CompletedProcess([], 0, stdout="pod/pod-0 deleted\n", stderr=""),
            subprocess.CompletedProcess(
                [],
                1,
                stdout="",
                stderr='Error from server (NotFound): pods "pod-0" not found',
            ),
        ]
    )
    calls = []

    def kubectl(*args, **kwargs):
        calls.append(args)
        return next(responses)

    runner._kubectl = kubectl
    runner.pods = lambda **_kwargs: []

    runner.cleanup()

    assert "--cascade=foreground" in calls[0]


def test_cleanup_ignores_pods_already_terminating_after_foreground_delete(tmp_path):
    runner = _runner(tmp_path)
    runner.manifest = tmp_path / "k8s_deploy.yaml"
    runner.kind = "LeaderWorkerSet"
    runner.name = "cell"
    runner.selector = "app.kubernetes.io/name=cell"
    responses = iter(
        [
            subprocess.CompletedProcess([], 0, stdout="leaderworkerset/cell deleted\n", stderr=""),
            subprocess.CompletedProcess(
                [],
                1,
                stdout="",
                stderr='Error from server (NotFound): leaderworkersets "cell" not found',
            ),
            subprocess.CompletedProcess(
                [],
                0,
                stdout=json.dumps(
                    {
                        "items": [
                            {
                                "metadata": {
                                    "name": "cell-0",
                                    "deletionTimestamp": "2026-07-19T00:00:00Z",
                                }
                            }
                        ]
                    }
                ),
                stderr="",
            ),
        ]
    )
    runner._kubectl = lambda *args, **kwargs: next(responses)

    runner.cleanup()


def test_cleanup_rejects_nonterminating_pod_after_foreground_delete(tmp_path):
    runner = _runner(tmp_path)
    runner.manifest = tmp_path / "k8s_deploy.yaml"
    runner.kind = "LeaderWorkerSet"
    runner.name = "cell"
    runner.selector = "app.kubernetes.io/name=cell"
    responses = iter(
        [
            subprocess.CompletedProcess([], 0, stdout="leaderworkerset/cell deleted\n", stderr=""),
            subprocess.CompletedProcess(
                [],
                1,
                stdout="",
                stderr='Error from server (NotFound): leaderworkersets "cell" not found',
            ),
            subprocess.CompletedProcess(
                [],
                0,
                stdout=json.dumps({"items": [{"metadata": {"name": "cell-0"}}]}),
                stderr="",
            ),
        ]
    )
    runner._kubectl = lambda *args, **kwargs: next(responses)

    with pytest.raises(RuntimeError, match="owned FPM pods remain"):
        runner.cleanup()


def test_cleanup_rejects_a_remaining_workload(tmp_path):
    runner = _runner(tmp_path)
    runner.manifest = tmp_path / "k8s_deploy.yaml"
    runner.kind = "Pod"
    runner.name = "pod-0"
    responses = iter(
        [
            subprocess.CompletedProcess([], 0, stdout="pod/pod-0 deleted\n", stderr=""),
            subprocess.CompletedProcess(
                [],
                0,
                stdout=json.dumps({"kind": "Pod", "metadata": {"name": "pod-0"}}),
                stderr="",
            ),
        ]
    )
    runner._kubectl = lambda *args, **kwargs: next(responses)

    with pytest.raises(RuntimeError, match="resource remains"):
        runner.cleanup()


def test_collect_rejects_masked_copy_without_benchmark_files(tmp_path):
    runner = _runner(tmp_path)
    runner._exec_checked = lambda *args, **kwargs: subprocess.CompletedProcess(args, 0, stdout="", stderr="")
    runner._kubectl = lambda *args, **kwargs: subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    with pytest.raises(RuntimeError, match="did not return a /results file manifest"):
        runner.collect(["pod-0"])


def test_stage_rejects_masked_truncated_copy(tmp_path):
    runner = _runner(tmp_path)
    source = tmp_path / "run.sh"
    source.write_text("#!/bin/sh\n")
    calls = []

    def exec_checked(pod, command, *, timeout):
        calls.append(command)
        if command[:2] == ["python3", "-c"]:
            raise RuntimeError("remote hash mismatch")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    runner._exec_checked = exec_checked
    runner._kubectl = lambda *args, **kwargs: subprocess.CompletedProcess(args, 0, stdout="", stderr="reset\n")

    with pytest.raises(RuntimeError, match="failed to stage an exact copy"):
        runner.stage(["pod-0"], [source])
    assert hashlib.sha256(source.read_bytes()).hexdigest() in calls[-1]


def test_collect_rejects_masked_partial_copy(tmp_path):
    # exercise the raw-copy path: these fixtures model uncompressed transfers
    runner = _runner(tmp_path)
    runner._exec_checked = lambda pod, command, timeout: (_ for _ in ()).throw(RuntimeError("remote_exit=127"))
    payloads = {
        "benchmark.json": b'{"status":"complete"}\n',
        "benchmark_dp1.json": b'{"status":"complete"}\n',
    }
    runner._remote_result_manifest = lambda pod: {
        name: {"size": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
        for name, payload in payloads.items()
    }

    def kubectl(*args, **kwargs):
        if args[0] == "cp":
            name = args[1].split(":/results/", 1)[1]
            destination = Path(args[2])
            if name == "benchmark.json":
                destination.write_bytes(payloads[name])
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="connection reset by peer\n")

    runner._kubectl = kubectl

    with pytest.raises(RuntimeError, match=r"failed to collect exact result file 'benchmark_dp1\.json'"):
        runner.collect(["pod-0"])


def test_collect_retries_one_file_without_recopying_verified_files(tmp_path):
    runner = _runner(tmp_path)
    payloads = {
        "benchmark.json": b'{"status":"complete"}\n',
        "engine.stderr.log": b"done\n",
    }
    runner._remote_result_manifest = lambda pod: {
        name: {"size": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
        for name, payload in payloads.items()
    }
    attempts: dict[str, int] = {}

    def kubectl(*args, **kwargs):
        if args[0] == "cp":
            name = args[1].split(":/results/", 1)[1]
            attempts[name] = attempts.get(name, 0) + 1
            payload = payloads[name]
            Path(args[2]).write_bytes(payload[:-1] if attempts[name] == 1 else payload)
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="connection reset by peer\n")

    runner._kubectl = kubectl
    runner.collect(["pod-0"])

    assert attempts == {"benchmark.json": 2, "engine.stderr.log": 2}
    runner.collect(["pod-0"])
    assert attempts == {"benchmark.json": 2, "engine.stderr.log": 2}


def test_collect_accepts_exact_remote_manifest(tmp_path):
    # exercise the raw-copy path: these fixtures model uncompressed transfers
    runner = _runner(tmp_path)
    runner._exec_checked = lambda pod, command, timeout: (_ for _ in ()).throw(RuntimeError("remote_exit=127"))
    payloads = {"benchmark.json": b'{"status":"complete"}\n', "engine.stderr.log": b"done\n"}
    runner._remote_result_manifest = lambda pod: {
        name: {"size": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
        for name, payload in payloads.items()
    }

    def kubectl(*args, **kwargs):
        if args[0] == "cp":
            name = args[1].split(":/results/", 1)[1]
            destination = Path(args[2])
            destination.write_bytes(payloads[name])
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    runner._kubectl = kubectl
    runner.collect(["pod-0"])
    assert (tmp_path / "raw" / "pod-0" / "benchmark.json").read_bytes() == payloads["benchmark.json"]


def test_collect_accepts_headless_worker_without_benchmark_artifact(tmp_path):
    runner = _runner(tmp_path)
    payloads = {
        "pod-0": {"benchmark.json": b'{"status":"complete"}\n', "engine.log": b"leader\n"},
        "pod-1": {"engine.log": b"headless\n"},
    }
    runner._remote_result_manifest = lambda pod: {
        name: {"size": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
        for name, payload in payloads[pod].items()
    }

    def kubectl(*args, **kwargs):
        if args[0] == "cp":
            pod = args[1].split("/", 1)[1].split(":", 1)[0]
            name = args[1].split(":/results/", 1)[1]
            destination = Path(args[2])
            destination.write_bytes(payloads[pod][name])
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    runner._kubectl = kubectl
    runner.collect(["pod-0", "pod-1"])

    assert (tmp_path / "raw" / "pod-0" / "benchmark.json").is_file()
    assert not (tmp_path / "raw" / "pod-1" / "benchmark.json").exists()


def test_formal_prefill_render_preserves_the_complete_collector_axis():
    cell = _cell()
    plan = _plan(cell)
    overrides = _cell_generator_overrides(plan, cell, {})
    args = overrides["params"]["agg"]["extra_cli_args"]

    assert args[args.index("--benchmark-mode") + 1] == "prefill"
    assert args[args.index("--benchmark-warmup-iterations") + 1] == "3"
    assert args[args.index("--max-model-len") + 1] == "-1"
    assert args[args.index("--max-num-batched-tokens") + 1] == "8192"
    assert args[args.index("--prefill-max-new-token-samples") + 1] == "199"
    assert "--prefix-max-batch-size-samples" not in args
    assert int(args[args.index("--prefill-max-kv-read-token-samples") + 1]) > 16
    assert "--scheduler-cls" not in args
    assert "--max-num-seqs" not in args
    assert "--gpu-memory-utilization" not in args
    compilation = json.loads(args[args.index("--compilation-config") + 1])
    assert compilation["max_cudagraph_capture_size"] == 2048
    assert len(compilation["cudagraph_capture_sizes"]) == 99
    env_names = {item["name"] for item in overrides["K8sConfig"]["extra_env"]}
    assert "DYN_FPM_CASE_CONFIG" not in env_names
    assert "PYTHONPATH" not in env_names


def test_decode_render_keeps_dynamo_runtime_limits_and_capture_axis():
    cell = _cell(phase="decode")
    args = _cell_generator_overrides(_plan(cell), cell, {})["params"]["agg"]["extra_cli_args"]

    assert args[args.index("--benchmark-mode") + 1] == "decode"
    assert args[args.index("--max-model-len") + 1] == "-1"
    assert "--max-num-batched-tokens" not in args
    assert "--max-num-seqs" not in args
    assert "--compilation-config" not in args
    assert "--prefill-max-new-token-samples" not in args


def test_formal_prefill_metadata_records_candidate_axis_counts():
    cell = _cell()

    assert _configured_sampling_metadata(_plan(cell), cell, smoke=False) == {
        "prefill_cudagraph_capture_size_count": 99,
        "prefill_requested_new_token_axis_count": 199,
        "prefill_max_new_token_samples": 199,
    }


def test_cell_render_adds_one_collector_owned_benchmark_timeout():
    cell = _cell()
    plan = _plan(cell)
    plan.options.warmup_iterations = 2

    args = _cell_generator_overrides(plan, cell, {})["params"]["agg"]["extra_cli_args"]

    assert args.count("--benchmark-timeout") == 1
    assert args[args.index("--benchmark-timeout") + 1] == "3600"
    assert args[args.index("--benchmark-warmup-iterations") + 1] == "2"


def test_explicit_prefill_batch_limit_does_not_override_dynamo_sampling():
    cell = _cell()
    plan = _plan(cell)
    plan.options.prefill_sampling = PrefillSamplingProfile.build(max_isl=1000, max_batch_size=16)

    args = _cell_generator_overrides(plan, cell, {})["params"]["agg"]["extra_cli_args"]

    assert args[args.index("--max-num-seqs") + 1] == "16"
    assert "--prefix-max-batch-size-samples" not in args
    assert args[args.index("--prefill-max-new-token-samples") + 1] == "132"
    compilation = json.loads(args[args.index("--compilation-config") + 1])
    assert compilation["max_cudagraph_capture_size"] == 1000
    assert compilation["cudagraph_capture_sizes"][-1] == 1000


def test_smoke_uses_native_axis_limits_instead_of_explicit_cases():
    cell = _cell()
    args = _cell_generator_overrides(_plan(cell), cell, {}, smoke=True)["params"]["agg"]["extra_cli_args"]

    assert args[args.index("--max-model-len") + 1] == "-1"
    assert args[args.index("--prefill-max-new-token-samples") + 1] == "2"
    assert args[args.index("--prefill-max-kv-read-token-samples") + 1] == "2"
    assert args[args.index("--prefix-max-batch-size-samples") + 1] == "1"


def test_native_collection_validation_accepts_balanced_total_points(tmp_path):
    cell = _cell(dp=2)
    for rank in range(2):
        path = tmp_path / f"pod-{rank}" / ("benchmark.json" if rank == 0 else "benchmark_dp1.json")
        path.parent.mkdir()
        _write_provenance(path.parent / "collector-provenance.json", cell_id=cell.cell_id)
        path.write_text(json.dumps(_native_payload(phase="prefill", rank=rank, dp=2)))

    assert _validate_runtime_collection(cell, tmp_path) == 1
    assert _runtime_collection_summary(cell, tmp_path) == {
        "measured_point_count": 1,
        "measured_batch_size_axis_count": 1,
        "measured_kv_read_axis_count": 1,
        "measured_new_token_axis_count": 1,
    }

    second_rank = tmp_path / "pod-1" / "benchmark_dp1.json"
    payload = json.loads(second_rank.read_text())
    payload["grid_digest"] = "different"
    second_rank.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="different run identities"):
        _validate_runtime_collection(cell, tmp_path)


def test_native_collection_validation_rejects_group_local_divergence(tmp_path):
    cell = _cell(dp=2)
    for rank in range(2):
        payload = _native_payload(phase="prefill", rank=rank, dp=2)
        path = tmp_path / f"pod-{rank}" / ("benchmark.json" if rank == 0 else "benchmark_dp1.json")
        path.parent.mkdir()
        _write_provenance(path.parent / "collector-provenance.json", cell_id=cell.cell_id)
        path.write_text(json.dumps(payload))

    path = tmp_path / "pod-1" / "benchmark_dp1.json"
    payload = json.loads(path.read_text())
    payload["results"][0]["fpms"][0]["wall_time"] = 0.5
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="differs from synchronized group"):
        _validate_runtime_collection(cell, tmp_path)


def test_native_collection_validation_rejects_wrong_cell_provenance(tmp_path):
    cell = _cell()
    path = tmp_path / "pod-0" / "benchmark.json"
    path.parent.mkdir()
    _write_provenance(path.parent / "collector-provenance.json", cell_id="other-cell")
    path.write_text(json.dumps(_native_payload(phase="prefill", rank=0, dp=1)))

    with pytest.raises(ValueError, match="provenance cell mismatch"):
        _validate_runtime_collection(cell, tmp_path)


def test_runtime_summaries_use_native_rank_artifacts_and_skip_merged(tmp_path):
    for rank in range(2):
        payload = _native_payload(phase="decode", rank=rank, dp=2)
        (tmp_path / f"benchmark_dp{rank}.json").write_text(json.dumps(payload))
    merged = _native_payload(phase="decode", rank=0, dp=2)
    merged["artifact_type"] = "merged"
    (tmp_path / "benchmark_merged.json").write_text(json.dumps(merged))

    assert _runtime_timing_summary(tmp_path) == {
        "runtime_rank_count": 2,
        "benchmark_elapsed_seconds": 11.0,
        "measured_iteration_seconds": 0.011,
    }


def test_run_collection_stages_no_explicit_scheduler_or_case_manifest(monkeypatch, tmp_path):
    cell = _cell()
    plan = _plan(cell)
    events = []
    staged_names = []

    def render_cell(*args, **kwargs):
        cell_dir = args[2]
        (cell_dir / "k8s_deploy.yaml").write_text("apiVersion: v1\nkind: Pod\nmetadata:\n  name: cell\n")
        (cell_dir / "run.sh").write_text("#!/bin/sh\n")

    class FakeResource:
        def __init__(self, _manifest, _cell_dir):
            events.append("init")

        def apply(self):
            events.append("apply")

        def wait_ready(self, _expected_nodes):
            return ["pod-0"]

        def stage(self, _pods, files):
            staged_names.extend(path.name for path in files)

        def prepare_attempt(self, _pods, **_kwargs):
            events.append("prepare_attempt")

        def execute(self, _pods):
            events.append("execute")

        def collect(self, _pods, *, require_benchmark=True):
            events.append("collect")

        def cleanup(self):
            events.append("cleanup")

    monkeypatch.setattr(fpm_runner, "_render_cell", render_cell)
    monkeypatch.setattr(fpm_runner, "KubernetesCellRunner", FakeResource)
    monkeypatch.setattr(
        fpm_runner,
        "_runtime_collection_summary",
        lambda *_args, **_kwargs: {
            "measured_point_count": 7,
            "measured_batch_size_axis_count": 1,
            "measured_kv_read_axis_count": 2,
            "measured_new_token_axis_count": 2,
        },
    )
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()

    errors = run_collection(
        plan,
        generator_overrides={},
        checkpoint_dir=str(checkpoint_dir),
        artifact_root=str(tmp_path / "artifacts"),
        resume=False,
        retry_failed=False,
        smoke=True,
        cell_limit=1,
    )

    assert errors == []
    assert events.count("execute") == 1
    assert events.count("cleanup") == 1
    assert set(staged_names) == {"run.sh", "run_with_etcd.sh", "preflight.py"}
    assert "cases.json" not in staged_names
    assert "fpm_scheduler.py" not in staged_names
    checkpoint = json.loads((checkpoint_dir / "fpm_forward_smoke.json").read_text())
    assert checkpoint["cells"][cell.cell_id]["prefill_max_new_token_samples"] == 2
    assert checkpoint["cells"][cell.cell_id]["measured_new_token_axis_count"] == 2
    assert "prefill_requested_new_token_axis_count" not in checkpoint["cells"][cell.cell_id]


def test_resume_retry_recovers_complete_salvaged_attempt_without_rerun(monkeypatch, tmp_path):
    cell = _cell()
    plan = _plan(cell)
    artifact_root = tmp_path / "artifacts"
    raw = artifact_root / plan.sha256[:16] / "smoke" / "cells" / cell.cell_id / "raw" / "pod-0"
    raw.mkdir(parents=True)
    _write_provenance(
        raw / "collector-provenance.json",
        cell_id=cell.cell_id,
        plan_sha256=plan.sha256,
        attempt_id="attempt-1",
    )
    (raw / "benchmark.json").write_text(json.dumps(_native_payload(phase="prefill", rank=0, dp=1)))

    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    checkpoint_path = checkpoint_dir / "fpm_forward_smoke.json"
    checkpoint_path.write_text(
        json.dumps(
            {
                "schema": fpm_runner.CHECKPOINT_SCHEMA,
                "plan_sha256": plan.sha256,
                "cells": {
                    cell.cell_id: {
                        "status": "failed",
                        "attempt_id": "attempt-1",
                        "error_type": "RuntimeError",
                        "error": "result copy was interrupted",
                    }
                },
            }
        )
    )

    def reject_rerun(*_args, **_kwargs):
        raise AssertionError("a complete salvaged attempt must not rerun on the cluster")

    monkeypatch.setattr(fpm_runner, "_render_cell", reject_rerun)

    errors = run_collection(
        plan,
        generator_overrides={},
        checkpoint_dir=str(checkpoint_dir),
        artifact_root=str(artifact_root),
        resume=True,
        retry_failed=True,
        smoke=True,
        cell_limit=1,
    )

    assert errors == []
    checkpoint = json.loads(checkpoint_path.read_text())
    record = checkpoint["cells"][cell.cell_id]
    assert record["status"] == "passed"
    assert record["attempt_id"] == "attempt-1"
    assert record["measured_point_count"] == 1
    assert record["artifact_recovery"] == {
        "error": "result copy was interrupted",
        "error_type": "RuntimeError",
        "original_status": "failed",
        "recovered_at": record["artifact_recovery"]["recovered_at"],
        "validation": "native_collection_plan_and_attempt_identity",
    }
    assert checkpoint["smoke"]["status"] == "passed"


def test_plain_resume_recovers_interrupted_attempt_without_rerun(monkeypatch, tmp_path):
    cell = _cell()
    plan = _plan(cell)
    artifact_root = tmp_path / "artifacts"
    raw = artifact_root / plan.sha256[:16] / "smoke" / "cells" / cell.cell_id / "raw" / "pod-0"
    raw.mkdir(parents=True)
    _write_provenance(
        raw / "collector-provenance.json",
        cell_id=cell.cell_id,
        plan_sha256=plan.sha256,
        attempt_id="attempt-1",
    )
    (raw / "benchmark.json").write_text(json.dumps(_native_payload(phase="prefill", rank=0, dp=1)))

    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    checkpoint_path = checkpoint_dir / "fpm_forward_smoke.json"
    checkpoint_path.write_text(
        json.dumps(
            {
                "schema": fpm_runner.CHECKPOINT_SCHEMA,
                "plan_sha256": plan.sha256,
                "cells": {cell.cell_id: {"status": "interrupted", "attempt_id": "attempt-1"}},
            }
        )
    )

    def reject_rerun(*_args, **_kwargs):
        raise AssertionError("a complete interrupted attempt must not rerun on the cluster")

    monkeypatch.setattr(fpm_runner, "_render_cell", reject_rerun)

    errors = run_collection(
        plan,
        generator_overrides={},
        checkpoint_dir=str(checkpoint_dir),
        artifact_root=str(artifact_root),
        resume=True,
        retry_failed=False,
        smoke=True,
        cell_limit=1,
    )

    assert errors == []
    record = json.loads(checkpoint_path.read_text())["cells"][cell.cell_id]
    assert record["status"] == "passed"
    assert record["artifact_recovery"]["original_status"] == "interrupted"


def test_recovery_rejects_entries_with_cleanup_error(tmp_path):
    cell = _cell()
    plan = _plan(cell)
    root = tmp_path / "artifacts" / plan.sha256[:16] / "smoke"
    raw = root / "cells" / cell.cell_id / "raw" / "pod-0"
    raw.mkdir(parents=True)
    _write_provenance(
        raw / "collector-provenance.json",
        cell_id=cell.cell_id,
        plan_sha256=plan.sha256,
        attempt_id="attempt-1",
    )
    (raw / "benchmark.json").write_text(json.dumps(_native_payload(phase="prefill", rank=0, dp=1)))
    entry = {
        "status": "failed",
        "attempt_id": "attempt-1",
        "error_type": "RuntimeError",
        "error": "result copy was interrupted",
        "cleanup_error": "kubectl delete timed out",
    }

    # Identical artifacts, but the unverified teardown quarantines the entry:
    # only a rerun re-drives the verified deletion of the leaked resource.
    assert fpm_runner._recover_completed_attempt(plan, cell, root, entry) is None
    del entry["cleanup_error"]
    recovered = fpm_runner._recover_completed_attempt(plan, cell, root, entry)
    assert recovered is not None and recovered["status"] == "passed"


def test_cleanup_failure_marks_passed_cell_retryable(monkeypatch, tmp_path):
    cell = _cell()
    plan = _plan(cell)

    def render_cell(*args, **kwargs):
        cell_dir = args[2]
        (cell_dir / "k8s_deploy.yaml").write_text("apiVersion: v1\nkind: Pod\nmetadata:\n  name: cell\n")
        (cell_dir / "run.sh").write_text("#!/bin/sh\n")

    class FakeResource:
        def __init__(self, _manifest, _cell_dir):
            pass

        def apply(self):
            pass

        def wait_ready(self, _expected_nodes):
            return ["pod-0"]

        def stage(self, _pods, _files):
            pass

        def prepare_attempt(self, _pods, **_kwargs):
            pass

        def execute(self, _pods):
            pass

        def collect(self, _pods, *, require_benchmark=True):
            pass

        def cleanup(self):
            raise RuntimeError("owned pod remains")

    monkeypatch.setattr(fpm_runner, "_render_cell", render_cell)
    monkeypatch.setattr(fpm_runner, "KubernetesCellRunner", FakeResource)
    monkeypatch.setattr(
        fpm_runner,
        "_runtime_collection_summary",
        lambda *_args, **_kwargs: {
            "measured_point_count": 1,
            "measured_batch_size_axis_count": 1,
            "measured_kv_read_axis_count": 1,
            "measured_new_token_axis_count": 1,
        },
    )
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()

    errors = run_collection(
        plan,
        generator_overrides={},
        checkpoint_dir=str(checkpoint_dir),
        artifact_root=str(tmp_path / "artifacts"),
        resume=False,
        retry_failed=False,
        smoke=True,
        cell_limit=1,
    )

    assert [error["classification"] for error in errors] == ["resource_cleanup_failed"]
    checkpoint = json.loads((checkpoint_dir / "fpm_forward_smoke.json").read_text())
    record = checkpoint["cells"][cell.cell_id]
    assert record["status"] == "cleanup_failed"
    assert record["cleanup_error"] == "owned pod remains"
    assert "smoke" not in checkpoint


def test_typed_generator_render_uses_collector_prefill_axis(tmp_path):
    cell = _cell(dp=4)
    plan = _plan(cell)
    base = {
        "K8sConfig": {
            "k8s_image": "nvcr.io/nvidia/ai-dynamo/vllm-runtime:test",
            "k8s_pvc_mount_path": "/model-cache",
            "k8s_model_path_in_pvc": "models--nvidia--GLM-5.2-NVFP4",
        }
    }

    _render_cell(plan, cell, tmp_path, base)

    script = (tmp_path / "run.sh").read_text()
    assert "--benchmark-mode prefill" in script
    assert "--benchmark-warmup-iterations 3" in script
    assert "--scheduler-cls fpm_scheduler" not in script
    assert "DYN_FPM_CASE_CONFIG" not in script
    assert "--max-model-len -1" in script
    assert "--max-num-seqs" not in script
    assert "--max-num-batched-tokens 8192" in script
    assert "--gpu-memory-utilization" not in script
    assert "--compilation-config" in script
    assert "--prefill-max-new-token-samples 199" in script
    assert "--prefix-max-batch-size-samples" not in script
    assert "--cudagraph-capture-sizes" not in script
    assert "--enable-expert-parallel" in script
    assert "--model /model-cache/models--nvidia--GLM-5.2-NVFP4" in script
    assert 'value.get("schema_version") != 2' in script
    subprocess.run(["bash", "-n", str(tmp_path / "run.sh")], check=True)


@pytest.mark.parametrize(
    ("model_path", "weight_quantization"),
    [
        ("nvidia/GLM-5.2-NVFP4", "nvfp4"),
        ("sgl-project/DeepSeek-V4-Flash-FP8", "fp8_block"),
    ],
)
def test_pure_tp_render_uses_shared_vllm_tp_without_expert_parallel(
    tmp_path,
    model_path,
    weight_quantization,
):
    cell = FPMCell(
        cell_id="pure-tp",
        workload_kind="prefill",
        topology=ParallelTopology(tp=4, pp=1, dp=1, moe_tp=4, moe_ep=1, cp=1),
        weight_quantization=weight_quantization,
        kv_cache_dtype="fp8",
        backend_policy=BackendPolicy("baseline", "baseline", {}, {}),
        parallel_strategy="pure_tp",
    )
    plan = _plan(cell)
    plan.model_path = model_path
    plan.capability = SimpleNamespace(
        architecture="GlmMoeDsaForCausalLM" if "GLM" in model_path else "DeepseekV4ForCausalLM"
    )

    _render_cell(plan, cell, tmp_path, {})

    script = (tmp_path / "run.sh").read_text()
    assert "--tensor-parallel-size 4" in script
    assert "--data-parallel-size 1" in script
    assert "--enable-expert-parallel" not in script


def _expected_meta(data: bytes) -> dict:
    import hashlib

    return {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}


def test_copy_result_file_uses_gzip_transfer(tmp_path):
    import gzip as _gzip

    runner = _runner(tmp_path)
    payload = b'{"results": [1, 2, 3]}' * 50000  # >1MiB engages compression
    exec_calls = []
    runner._exec_checked = lambda pod, command, timeout: exec_calls.append(command)

    def kubectl(*args, **kwargs):
        if args[0] == "cp":
            assert args[1].endswith(".__xfer.gz"), args[1]
            # The transfer temp must live outside the manifested /results tree.
            assert f":{REMOTE_WORKDIR}/" in args[1], args[1]
            Path(args[2]).write_bytes(_gzip.compress(payload))
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    runner._kubectl = kubectl
    runner._copy_result_file("pod-0", "benchmark.json", _expected_meta(payload), tmp_path / "raw")

    assert (tmp_path / "raw" / "benchmark.json").read_bytes() == payload
    assert any("gzip -cf" in " ".join(c) for c in exec_calls)
    assert any(c[:2] == ["rm", "-f"] for c in exec_calls), "transfer temp must be removed"


def test_copy_result_file_retries_truncated_gzip(tmp_path):
    import gzip as _gzip

    runner = _runner(tmp_path)
    payload = b"x" * (2 * 1024 * 1024)
    blob = _gzip.compress(payload)
    attempts = iter([blob[: len(blob) // 2], blob])
    runner._exec_checked = lambda pod, command, timeout: None

    def kubectl(*args, **kwargs):
        if args[0] == "cp":
            Path(args[2]).write_bytes(next(attempts))
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    runner._kubectl = kubectl
    runner._copy_result_file("pod-0", "benchmark.json", _expected_meta(payload), tmp_path / "raw")
    assert (tmp_path / "raw" / "benchmark.json").read_bytes() == payload


def test_copy_result_file_falls_back_to_raw_when_gzip_unavailable(tmp_path):
    runner = _runner(tmp_path)
    payload = b"raw-path-data" * 100000  # >1MiB so compression is attempted
    copies = []

    def exec_checked(pod, command, timeout):
        raise RuntimeError("remote_exit=127")

    runner._exec_checked = exec_checked

    def kubectl(*args, **kwargs):
        if args[0] == "cp":
            copies.append(args[1])
            Path(args[2]).write_bytes(payload)
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    runner._kubectl = kubectl
    runner._copy_result_file("pod-0", "benchmark.json", _expected_meta(payload), tmp_path / "raw")
    assert copies and copies[0].endswith(":/results/benchmark.json")
    assert (tmp_path / "raw" / "benchmark.json").read_bytes() == payload


def test_copy_result_file_skips_compression_for_small_files(tmp_path):
    runner = _runner(tmp_path)
    payload = b"raw-path-data"
    exec_calls = []
    runner._exec_checked = lambda pod, command, timeout: exec_calls.append(command)

    def kubectl(*args, **kwargs):
        if args[0] == "cp":
            assert args[1].endswith(":/results/benchmark.json"), args[1]
            Path(args[2]).write_bytes(payload)
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    runner._kubectl = kubectl
    runner._copy_result_file("pod-0", "benchmark.json", _expected_meta(payload), tmp_path / "raw")
    assert exec_calls == [], "small files must not pay the in-pod gzip round-trips"
    assert (tmp_path / "raw" / "benchmark.json").read_bytes() == payload
