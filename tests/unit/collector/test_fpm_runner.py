# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

import collector.fpm_forward.runner as fpm_runner
from collector.fpm_forward.capacity import FPMExecutionProfile, _cudagraph_capture_sizes
from collector.fpm_forward.planner import BackendPolicy, FPMCell
from collector.fpm_forward.runner import (
    REMOTE_EXIT_MARKER,
    REMOTE_WORKDIR,
    KubernetesCellRunner,
    _case_payload,
    _cell_generator_overrides,
    _render_cell,
    _validate_runtime_collection,
    run_collection,
)
from collector.fpm_forward.types import FPMPoint, ParallelTopology

pytestmark = pytest.mark.unit


def _profile(
    points: tuple[FPMPoint, ...],
    *,
    max_batch_size: int | None = None,
    max_num_tokens: int | None = None,
    max_seq_len: int | None = None,
    gpu_memory_utilization: float = 0.86,
) -> FPMExecutionProfile:
    capture_sizes = _cudagraph_capture_sizes(points)
    return FPMExecutionProfile(
        ordered_points=points,
        selected_point_count=len(points),
        max_batch_size=max_batch_size or max(point.batch_size for point in points),
        max_num_tokens=max_num_tokens or max(point.batch_size * point.suffix_length for point in points),
        max_seq_len=max_seq_len or max(point.prefix_length + point.suffix_length + 1 for point in points),
        gpu_memory_utilization=gpu_memory_utilization,
        cudagraph_capture_sizes=capture_sizes,
        memory_source="test",
        memory_fraction_ceiling=0.9,
        kv_tolerance_fraction=0.05,
        total_gpu_capacity_bytes=1,
        non_kv_bytes=1,
        required_kv_bytes=1,
        decisions=(),
    )


def _runner(tmp_path) -> KubernetesCellRunner:
    runner = object.__new__(KubernetesCellRunner)
    runner.namespace = "test"
    runner.cell_dir = tmp_path
    return runner


def test_apply_skips_client_validation_and_verifies_created_object(tmp_path):
    runner = _runner(tmp_path)
    runner.manifest = tmp_path / "k8s_deploy.yaml"
    runner.manifest.write_text("apiVersion: v1\nkind: Pod\n")
    runner.kind = "Pod"
    runner.name = "pod-0"
    calls = []

    def kubectl(*args, **kwargs):
        calls.append(args)
        if args[0] == "get":
            return subprocess.CompletedProcess(args, 0, stdout=json.dumps({"metadata": {"name": "pod-0"}}), stderr="")
        return subprocess.CompletedProcess(args, 0, stdout="pod/pod-0 created\n", stderr="")

    runner._kubectl = kubectl

    runner.apply()

    assert calls[0][:2] == ("apply", "--validate=false")
    assert calls[1][:2] == ("get", "Pod/pod-0")


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

    with pytest.raises(RuntimeError, match="did not create Pod/pod-0"):
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


def test_collect_rejects_masked_copy_without_benchmark_files(tmp_path):
    runner = _runner(tmp_path)
    runner._exec_checked = lambda *args, **kwargs: subprocess.CompletedProcess(args, 0, stdout="", stderr="")
    runner._kubectl = lambda *args, **kwargs: subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    with pytest.raises(RuntimeError, match="did not return a /results file manifest"):
        runner.collect(["pod-0"])


def test_stage_rejects_masked_truncated_copy(tmp_path):
    runner = _runner(tmp_path)
    source = tmp_path / "cases.json"
    source.write_text('{"selected_point_count": 1}\n')
    calls = []

    def exec_checked(pod, command, *, timeout):
        calls.append(command)
        if command[:2] == ["python3", "-c"]:
            raise RuntimeError("remote hash mismatch")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    runner._exec_checked = exec_checked
    runner._kubectl = lambda *args, **kwargs: subprocess.CompletedProcess(
        args,
        0,
        stdout="",
        stderr="connection reset by peer\n",
    )

    with pytest.raises(RuntimeError, match="failed to stage an exact copy"):
        runner.stage(["pod-0"], [source])

    assert hashlib.sha256(source.read_bytes()).hexdigest() in calls[-1]


def test_collect_rejects_masked_partial_copy(tmp_path):
    runner = _runner(tmp_path)
    payloads = {
        "benchmark.json": b'{"status":"complete"}\n',
        "benchmark_dp1.json": b'{"status":"complete"}\n',
    }
    runner._exec_checked = lambda *args, **kwargs: subprocess.CompletedProcess(args, 0, stdout="", stderr="")
    runner._remote_result_manifest = lambda pod: {
        name: {"size": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
        for name, payload in payloads.items()
    }

    def kubectl(*args, **kwargs):
        if args[0] == "cp":
            destination = Path(args[2])
            (destination / "benchmark.json").write_bytes(payloads["benchmark.json"])
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="connection reset by peer\n")

    runner._kubectl = kubectl

    with pytest.raises(RuntimeError, match=r"missing=\['benchmark_dp1.json'\]"):
        runner.collect(["pod-0"])


def test_collect_accepts_exact_remote_manifest(tmp_path):
    runner = _runner(tmp_path)
    payloads = {
        "benchmark.json": b'{"status":"complete"}\n',
        "engine.stderr.log": b"benchmark complete\n",
    }
    runner._exec_checked = lambda *args, **kwargs: subprocess.CompletedProcess(args, 0, stdout="", stderr="")
    runner._remote_result_manifest = lambda pod: {
        name: {"size": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
        for name, payload in payloads.items()
    }

    def kubectl(*args, **kwargs):
        if args[0] == "cp":
            destination = Path(args[2])
            for name, payload in payloads.items():
                (destination / name).write_bytes(payload)
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    runner._kubectl = kubectl

    runner.collect(["pod-0"])

    assert (tmp_path / "raw" / "pod-0" / "benchmark.json").read_bytes() == payloads["benchmark.json"]


def test_cell_render_preserves_capacity_planned_engine_limits():
    point = FPMPoint("prefill", batch_size=1, suffix_length=1, prefix_length=0)
    plan = SimpleNamespace(
        sha256="plan-sha",
        model_path="nvidia/GLM-5.2-NVFP4",
        options=SimpleNamespace(kv_block_size=64),
        capability=SimpleNamespace(architecture="GlmMoeDsaForCausalLM"),
        prefill_design=SimpleNamespace(selected=(point,)),
        decode_design=SimpleNamespace(selected=()),
    )
    cell = FPMCell(
        cell_id="cell",
        workload_kind="prefill",
        topology=ParallelTopology(tp=1, pp=1, dp=4, moe_tp=1, moe_ep=4, cp=1),
        weight_quantization="nvfp4",
        kv_cache_dtype="fp8_e4m3",
        backend_policy=BackendPolicy("baseline", "baseline", {}, {}),
        sampling_sha256="sampling",
        execution_profile=_profile(
            (point,),
            max_batch_size=256,
            max_num_tokens=16384,
            max_seq_len=65537,
        ),
    )
    base = {
        "K8sConfig": {
            "k8s_image": "example/runtime:test",
            "k8s_pvc_mount_path": "/model-cache",
            "k8s_model_path_in_pvc": "models--nvidia--GLM-5.2-NVFP4",
        }
    }

    overrides = _cell_generator_overrides(plan, cell, base, selected_point_count=1)

    assert overrides["preserve_engine_limits"] is True
    assert overrides["params"]["agg"]["max_batch_size"] == 256
    assert overrides["params"]["agg"]["max_num_tokens"] == 16384
    assert overrides["params"]["agg"]["max_seq_len"] == 65537
    assert overrides["params"]["agg"]["gpu_memory_utilization"] == 0.86
    assert overrides["params"]["agg"]["compilation_config"]["cudagraph_capture_sizes"] == [1]
    assert overrides["K8sConfig"]["k8s_image"] == "example/runtime:test"
    assert overrides["ServiceConfig"]["model_path"] == "/model-cache/models--nvidia--GLM-5.2-NVFP4"
    assert overrides["ServiceConfig"]["served_model_name"] == "nvidia/GLM-5.2-NVFP4"
    assert overrides["params"]["agg"]["extra_cli_args"].count("--benchmark-timeout") == 1
    assert "--enable-prefix-caching" in overrides["params"]["agg"]["extra_cli_args"]
    assert "--no-scheduler-reserve-full-isl" in overrides["params"]["agg"]["extra_cli_args"]
    assert "--no-async-scheduling" in overrides["params"]["agg"]["extra_cli_args"]
    assert "--trust-remote-code" in overrides["params"]["agg"]["extra_cli_args"]
    assert "--reasoning-parser=glm45" in overrides["params"]["agg"]["extra_cli_args"]
    executor_index = overrides["params"]["agg"]["extra_cli_args"].index("--distributed-executor-backend")
    assert overrides["params"]["agg"]["extra_cli_args"][executor_index + 1] == "mp"
    timeout_index = overrides["params"]["agg"]["extra_cli_args"].index("--benchmark-timeout")
    assert overrides["params"]["agg"]["extra_cli_args"][timeout_index + 1] == "3600"


def test_cell_render_adds_one_collector_owned_benchmark_timeout():
    point = FPMPoint("prefill", batch_size=1, suffix_length=1, prefix_length=0)
    plan = SimpleNamespace(
        sha256="plan-sha",
        model_path="model",
        options=SimpleNamespace(kv_block_size=64),
        prefill_design=SimpleNamespace(selected=(point,)),
        decode_design=SimpleNamespace(selected=()),
    )
    cell = FPMCell(
        cell_id="cell",
        workload_kind="prefill",
        topology=ParallelTopology(tp=1, pp=1, dp=1, moe_tp=1, moe_ep=1, cp=1),
        weight_quantization="nvfp4",
        kv_cache_dtype="fp8_e4m3",
        backend_policy=BackendPolicy("baseline", "baseline", {}, {}),
        sampling_sha256="sampling",
        execution_profile=_profile((point,)),
    )
    base = {}

    overrides = _cell_generator_overrides(plan, cell, base, selected_point_count=1)

    args = overrides["params"]["agg"]["extra_cli_args"]
    assert args.count("--benchmark-timeout") == 1
    timeout_index = args.index("--benchmark-timeout")
    assert args[timeout_index + 1] == "3600"


def test_smoke_case_payload_freezes_requested_point_count():
    points = (
        FPMPoint("prefill", batch_size=1, suffix_length=1, prefix_length=0),
        FPMPoint("prefill", batch_size=1, suffix_length=1, prefix_length=64),
    )
    plan = SimpleNamespace(
        sha256="plan-sha",
        options=SimpleNamespace(warmup_repeats=0, measurement_repeats=1),
        prefill_design=SimpleNamespace(selected=points, ordered_points=points, sha256="sampling"),
        decode_design=SimpleNamespace(selected=(), ordered_points=(), sha256="decode"),
    )
    cell = FPMCell(
        cell_id="cell",
        workload_kind="prefill",
        topology=ParallelTopology(tp=4, pp=1, dp=1, moe_tp=4, moe_ep=1, cp=1),
        weight_quantization="nvfp4",
        kv_cache_dtype="fp8_e4m3",
        backend_policy=BackendPolicy("baseline", "baseline", {}, {}),
        sampling_sha256="sampling",
        execution_profile=_profile(points),
    )

    payload = _case_payload(plan, cell, selected_point_count=2)

    assert payload["selected_point_count"] == 2
    assert len(payload["ordered_shapes"]) == 2
    assert payload["warmup_repeats"] == 0
    assert payload["measured_repeats"] == 1


def test_integrated_collection_requires_runtime_capacity_for_full_profile(tmp_path):
    point = FPMPoint("prefill", batch_size=1, suffix_length=1, prefix_length=0)
    cell = FPMCell(
        cell_id="cell",
        workload_kind="prefill",
        topology=ParallelTopology(tp=1, pp=1, dp=2, moe_tp=1, moe_ep=2, cp=1),
        weight_quantization="nvfp4",
        kv_cache_dtype="fp8_e4m3",
        backend_policy=BackendPolicy("baseline", "baseline", {}, {}),
        sampling_sha256="sampling",
        execution_profile=_profile((point,)),
    )
    for rank, eligible in ((0, 1), (1, 0)):
        path = tmp_path / f"pod-{rank}" / f"benchmark_dp{rank}.json"
        path.parent.mkdir(parents=True)
        path.write_text(
            json.dumps(
                {
                    "status": "complete",
                    "valid": True,
                    "collector": {
                        "cell_id": "cell",
                        "selected_point_count": 1,
                        "capacity_eligible_count": eligible,
                        "capacity_cancelled_count": 0 if eligible else 1,
                    },
                    "cancelled_points": [] if eligible else [{"point": point.to_dict()}],
                    "campaign_results": [
                        {
                            "point": point.to_dict(),
                            "warmup_fpms": [],
                            "fpms": [{"dp_rank": rank}],
                        }
                    ],
                }
            )
        )

    with pytest.raises(ValueError, match="runtime capacity admits only 0/1"):
        _validate_runtime_collection(cell, tmp_path)


def test_integrated_collection_requires_identical_dp_rank_selection(tmp_path):
    points = (
        FPMPoint("prefill", batch_size=1, suffix_length=1, prefix_length=0),
        FPMPoint("prefill", batch_size=1, suffix_length=1, prefix_length=64),
    )
    cell = FPMCell(
        cell_id="cell",
        workload_kind="prefill",
        topology=ParallelTopology(tp=1, pp=1, dp=2, moe_tp=1, moe_ep=2, cp=1),
        weight_quantization="nvfp4",
        kv_cache_dtype="fp8_e4m3",
        backend_policy=BackendPolicy("baseline", "baseline", {}, {}),
        sampling_sha256="sampling",
        execution_profile=_profile(points),
    )
    for rank, selected, cancelled in (
        (0, points[0], []),
        (1, points[1], [{"point": points[0].to_dict()}]),
    ):
        path = tmp_path / f"pod-{rank}" / f"benchmark_dp{rank}.json"
        path.parent.mkdir(parents=True)
        path.write_text(
            json.dumps(
                {
                    "status": "complete",
                    "valid": True,
                    "collector": {
                        "cell_id": "cell",
                        "selected_point_count": 1,
                        "capacity_eligible_count": len(points) - len(cancelled),
                        "capacity_cancelled_count": len(cancelled),
                    },
                    "cancelled_points": cancelled,
                    "campaign_results": [
                        {
                            "point": selected.to_dict(),
                            "warmup_fpms": [],
                            "fpms": [{"dp_rank": rank}],
                        }
                    ],
                }
            )
        )

    with pytest.raises(ValueError, match="selected different formal point sets"):
        _validate_runtime_collection(cell, tmp_path, selected_point_count=1)


def test_run_collection_starts_engine_once_per_cell(monkeypatch, tmp_path):
    point = FPMPoint("prefill", batch_size=1, suffix_length=1, prefix_length=0)
    cell = FPMCell(
        cell_id="cell",
        workload_kind="prefill",
        topology=ParallelTopology(tp=1, pp=1, dp=1, moe_tp=1, moe_ep=1, cp=1),
        weight_quantization="nvfp4",
        kv_cache_dtype="fp8_e4m3",
        backend_policy=BackendPolicy("baseline", "baseline", {}, {}),
        sampling_sha256="sampling",
        execution_profile=_profile((point,)),
    )
    plan = SimpleNamespace(
        sha256="plan-sha",
        cells=(cell,),
        runtime_overlay_files=(),
        runtime_overlay_base_files=(),
        options=SimpleNamespace(smoke_points=1, warmup_repeats=0, measurement_repeats=1),
        to_dict=lambda: {"sha256": "plan-sha"},
    )
    events = []

    def render_cell(*_args, **_kwargs):
        cell_dir = _args[2]
        (cell_dir / "k8s_deploy.yaml").write_text("apiVersion: v1\nkind: Pod\n")
        (cell_dir / "run.sh").write_text("#!/bin/sh\n")

    class FakeResource:
        def __init__(self, _manifest, _cell_dir):
            events.append("init")

        def apply(self):
            events.append("apply")

        def wait_ready(self, _expected_nodes):
            events.append("ready")
            return ["pod-0"]

        def stage(self, _pods, _files):
            events.append("stage")

        def execute(self, _pods):
            events.append("execute")

        def collect(self, _pods, *, require_benchmark=True):
            assert require_benchmark is True
            events.append("collect")

        def cleanup(self):
            events.append("cleanup")

    monkeypatch.setattr(fpm_runner, "_render_cell", render_cell)
    monkeypatch.setattr(fpm_runner, "KubernetesCellRunner", FakeResource)
    monkeypatch.setattr(
        fpm_runner,
        "_validate_runtime_collection",
        lambda *_args, **_kwargs: events.append("validate"),
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
    assert events.count("collect") == 1
    assert events.count("stage") == 1
    assert events.count("validate") == 1
    assert events.count("cleanup") == 1


def test_typed_generator_render_keeps_frozen_fpm_limits(tmp_path):
    point = FPMPoint("prefill", batch_size=1, suffix_length=1, prefix_length=0)
    plan = SimpleNamespace(
        sha256="plan-sha",
        model_path="nvidia/GLM-5.2-NVFP4",
        system="b200_sxm",
        backend="vllm",
        options=SimpleNamespace(kv_block_size=64),
        capability=SimpleNamespace(architecture="GlmMoeDsaForCausalLM"),
        prefill_design=SimpleNamespace(selected=(point,)),
        decode_design=SimpleNamespace(selected=()),
    )
    cell = FPMCell(
        cell_id="cell",
        workload_kind="prefill",
        topology=ParallelTopology(tp=1, pp=1, dp=4, moe_tp=1, moe_ep=4, cp=1),
        weight_quantization="nvfp4",
        kv_cache_dtype="fp8_e4m3",
        backend_policy=BackendPolicy("baseline", "baseline", {}, {}),
        sampling_sha256="sampling",
        execution_profile=_profile(
            (point,),
            max_batch_size=256,
            max_num_tokens=16384,
            max_seq_len=65537,
            gpu_memory_utilization=0.86,
        ),
    )
    base = {
        "K8sConfig": {
            "k8s_image": "nvcr.io/nvidia/ai-dynamo/vllm-runtime:test",
            "k8s_pvc_mount_path": "/model-cache",
            "k8s_model_path_in_pvc": "models--nvidia--GLM-5.2-NVFP4",
        },
    }

    _render_cell(plan, cell, tmp_path, base, selected_point_count=1)

    script = (tmp_path / "run.sh").read_text()
    assert "--max-model-len 65537" in script
    assert "--max-num-seqs 256" in script
    assert "--max-num-batched-tokens 16384" in script
    assert "--gpu-memory-utilization 0.86" in script
    assert "--compilation-config" in script
    assert "--enable-expert-parallel" in script
    assert "--model /model-cache/models--nvidia--GLM-5.2-NVFP4" in script
    assert "--served-model-name nvidia/GLM-5.2-NVFP4" in script


@pytest.mark.parametrize(
    ("model_path", "weight_quantization"),
    [
        ("nvidia/GLM-5.2-NVFP4", "nvfp4"),
        ("sgl-project/DeepSeek-V4-Flash-FP8", "fp8_block"),
    ],
)
def test_pure_tp_render_uses_shared_vllm_tp_without_expert_parallel(tmp_path, model_path, weight_quantization):
    point = FPMPoint("prefill", batch_size=1, suffix_length=1, prefix_length=0)
    plan = SimpleNamespace(
        sha256="plan-sha",
        model_path=model_path,
        system="b200_sxm",
        backend="vllm",
        options=SimpleNamespace(kv_block_size=64),
        prefill_design=SimpleNamespace(selected=(point,)),
        decode_design=SimpleNamespace(selected=()),
    )
    cell = FPMCell(
        cell_id="pure-tp",
        workload_kind="prefill",
        topology=ParallelTopology(tp=4, pp=1, dp=1, moe_tp=4, moe_ep=1, cp=1),
        weight_quantization=weight_quantization,
        kv_cache_dtype="fp8",
        backend_policy=BackendPolicy("baseline", "baseline", {}, {}),
        sampling_sha256="sampling",
        execution_profile=_profile((point,)),
        parallel_strategy="pure_tp",
    )

    _render_cell(plan, cell, tmp_path, {}, selected_point_count=1)

    script = (tmp_path / "run.sh").read_text()
    assert "--tensor-parallel-size 4" in script
    assert "--data-parallel-size 1" in script
    assert "--enable-expert-parallel" not in script


def test_runtime_overlay_installer_rejects_wrong_base_hash(tmp_path, monkeypatch):
    module_path = Path(__file__).parents[3] / "collector" / "fpm_forward" / "runtime" / "install_overlay.py"
    spec = importlib.util.spec_from_file_location("fpm_install_overlay_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    workdir = tmp_path / "work"
    results = tmp_path / "results"
    package = tmp_path / "package"
    workdir.mkdir()
    results.mkdir()
    package.mkdir()
    source = workdir / "runtime-overlay-example.py"
    source.write_text("VALUE = 'overlay'\n")
    target = package / "example.py"
    target.write_text("VALUE = 'base'\n")
    manifest = {
        "schema_version": 1,
        "files": {
            "example.py": {
                "source_name": source.name,
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "original_sha256": "0" * 64,
            }
        },
    }
    manifest_path = workdir / "runtime-overlay-manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    monkeypatch.setattr(module, "WORKDIR", workdir)
    monkeypatch.setattr(module, "MANIFEST", manifest_path)
    monkeypatch.setattr(module, "RESULTS_DIR", results)
    monkeypatch.setattr(
        module.importlib.util,
        "find_spec",
        lambda _name: SimpleNamespace(submodule_search_locations=[str(package)]),
    )

    with pytest.raises(RuntimeError, match="base hash mismatch"):
        module.main()
    assert target.read_text() == "VALUE = 'base'\n"


def test_runtime_overlay_installer_is_idempotent_for_pinned_overlay(tmp_path, monkeypatch):
    module_path = Path(__file__).parents[3] / "collector" / "fpm_forward" / "runtime" / "install_overlay.py"
    spec = importlib.util.spec_from_file_location("fpm_install_overlay_idempotent_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    workdir = tmp_path / "work"
    results = tmp_path / "results"
    package = tmp_path / "package"
    workdir.mkdir()
    results.mkdir()
    package.mkdir()
    source = workdir / "runtime-overlay-example.py"
    source.write_text("VALUE = 'overlay'\n")
    target = package / "example.py"
    target.write_text("VALUE = 'base'\n")
    manifest = {
        "schema_version": 1,
        "files": {
            "example.py": {
                "source_name": source.name,
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "original_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            }
        },
    }
    manifest_path = workdir / "runtime-overlay-manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    monkeypatch.setattr(module, "WORKDIR", workdir)
    monkeypatch.setattr(module, "MANIFEST", manifest_path)
    monkeypatch.setattr(module, "RESULTS_DIR", results)
    monkeypatch.setattr(
        module.importlib.util,
        "find_spec",
        lambda _name: SimpleNamespace(submodule_search_locations=[str(package)]),
    )

    module.main()
    assert target.read_text() == "VALUE = 'overlay'\n"
    audit = json.loads((results / "runtime-overlay-audit.json").read_text())
    assert audit["files"]["example.py"]["status"] == "installed"

    shutil.rmtree(results)
    results.mkdir()
    module.main()
    assert target.read_text() == "VALUE = 'overlay'\n"
    audit = json.loads((results / "runtime-overlay-audit.json").read_text())
    assert audit["files"]["example.py"]["status"] == "already_installed"
