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

from collector.fpm_forward.planner import BackendPolicy, FPMCell
from collector.fpm_forward.runner import (
    REMOTE_EXIT_MARKER,
    REMOTE_WORKDIR,
    KubernetesCellRunner,
    _case_payload,
    _cell_generator_overrides,
    _render_cell,
)
from collector.fpm_forward.types import FPMPoint, ParallelTopology

pytestmark = pytest.mark.unit


def _runner(tmp_path) -> KubernetesCellRunner:
    runner = object.__new__(KubernetesCellRunner)
    runner.namespace = "test"
    runner.cell_dir = tmp_path
    return runner


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

    with pytest.raises(RuntimeError, match="failed to collect benchmark results"):
        runner.collect(["pod-0"])


def test_cell_render_preserves_capacity_planned_engine_limits():
    point = FPMPoint("prefill", batch_size=1, suffix_length=1, prefix_length=0)
    plan = SimpleNamespace(
        sha256="plan-sha",
        options=SimpleNamespace(kv_block_size=64),
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
    )
    base = {
        "params": {
            "agg": {
                "max_batch_size": 256,
                "max_num_tokens": 16384,
                "max_seq_len": 65537,
                "compilation_config": {"cudagraph_capture_sizes": [1]},
            }
        }
    }

    overrides = _cell_generator_overrides(plan, cell, base, selected_point_count=1)

    assert overrides["preserve_engine_limits"] is True
    assert overrides["params"]["agg"]["max_batch_size"] == 256
    assert overrides["params"]["agg"]["max_num_tokens"] == 16384
    assert overrides["params"]["agg"]["max_seq_len"] == 65537
    assert overrides["params"]["agg"]["compilation_config"] == {"cudagraph_capture_sizes": [1]}
    assert overrides["params"]["agg"]["extra_cli_args"].count("--benchmark-timeout") == 1
    timeout_index = overrides["params"]["agg"]["extra_cli_args"].index("--benchmark-timeout")
    assert overrides["params"]["agg"]["extra_cli_args"][timeout_index + 1] == "3600"


def test_cell_render_preserves_explicit_benchmark_timeout():
    point = FPMPoint("prefill", batch_size=1, suffix_length=1, prefix_length=0)
    plan = SimpleNamespace(
        sha256="plan-sha",
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
    )
    base = {"params": {"agg": {"extra_cli_args": ["--benchmark-timeout=900"]}}}

    overrides = _cell_generator_overrides(plan, cell, base, selected_point_count=1)

    args = overrides["params"]["agg"]["extra_cli_args"]
    assert "--benchmark-timeout=900" in args
    assert "--benchmark-timeout" not in args


def test_smoke_case_payload_freezes_requested_point_count():
    points = (
        FPMPoint("prefill", batch_size=1, suffix_length=1, prefix_length=0),
        FPMPoint("prefill", batch_size=1, suffix_length=1, prefix_length=64),
    )
    plan = SimpleNamespace(
        sha256="plan-sha",
        options=SimpleNamespace(warmup_repeats=1, measurement_repeats=1),
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
    )

    payload = _case_payload(plan, cell, selected_point_count=2)

    assert payload["selected_point_count"] == 2
    assert len(payload["ordered_shapes"]) == 2


def test_typed_generator_render_keeps_frozen_fpm_limits(tmp_path):
    point = FPMPoint("prefill", batch_size=1, suffix_length=1, prefix_length=0)
    plan = SimpleNamespace(
        sha256="plan-sha",
        model_path="nvidia/GLM-5.2-NVFP4",
        system="b200_sxm",
        backend="vllm",
        options=SimpleNamespace(kv_block_size=64),
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
    )
    base = {
        "generated_config_version": "0.20.1",
        "params": {
            "agg": {
                "max_batch_size": 256,
                "max_num_tokens": 16384,
                "max_seq_len": 65537,
                "gpu_memory_utilization": 0.86,
                "compilation_config": {
                    "cudagraph_mode": "FULL_AND_PIECEWISE",
                    "cudagraph_capture_sizes": [1],
                    "max_cudagraph_capture_size": 1,
                },
            }
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


def test_pure_tp_render_uses_shared_vllm_tp_without_expert_parallel(tmp_path):
    point = FPMPoint("prefill", batch_size=1, suffix_length=1, prefix_length=0)
    plan = SimpleNamespace(
        sha256="plan-sha",
        model_path="nvidia/GLM-5.2-NVFP4",
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
        weight_quantization="nvfp4",
        kv_cache_dtype="fp8",
        backend_policy=BackendPolicy("baseline", "baseline", {}, {}),
        sampling_sha256="sampling",
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
