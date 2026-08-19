# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import collector.fpm_forward.runner as fpm_runner
from aiconfigurator.fpm_contract import FPM_CELL_LABEL
from collector.fpm_forward.config import FPMCollectionOptions, PrefillSamplingProfile
from collector.fpm_forward.model_capability import load_model_config
from collector.fpm_forward.planner import BackendPolicy, FPMCell, build_collection_plan
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


def _multinode_manifest(path: Path) -> Path:
    path.write_text(
        """\
apiVersion: resource.nvidia.com/v1beta1
kind: ComputeDomain
metadata:
  name: cell-compute-domain
  namespace: test
spec:
  channel:
    resourceClaimTemplate:
      name: cell-compute-domain-channel
  numNodes: 0
---
apiVersion: leaderworkerset.x-k8s.io/v1
kind: LeaderWorkerSet
metadata:
  name: cell
  namespace: test
  labels:
    app.kubernetes.io/name: cell
    aiconfigurator.nvidia.com/cell: cell
spec:
  leaderWorkerTemplate:
    size: 2
"""
    )
    return path


def _podcliqueset_manifest(
    path: Path,
    *,
    spec: dict | None = None,
    include_compute_domain: bool = False,
) -> Path:
    workload = {
        "apiVersion": "grove.io/v1alpha1",
        "kind": "PodCliqueSet",
        "metadata": {
            "name": "cell",
            "namespace": "test",
            "labels": {
                "app.kubernetes.io/name": "cell",
                "aiconfigurator.nvidia.com/cell": "cell",
            },
        },
        "spec": spec
        or {
            "replicas": 1,
            "template": {
                "cliques": [
                    {"name": "leader", "spec": {"replicas": 1}},
                    {"name": "worker", "spec": {"replicas": 3}},
                ]
            },
        },
    }
    documents = [workload]
    if include_compute_domain:
        documents.insert(
            0,
            {
                "apiVersion": "resource.nvidia.com/v1beta1",
                "kind": "ComputeDomain",
                "metadata": {"name": "cell-compute-domain", "namespace": "test"},
                "spec": {
                    "channel": {
                        "resourceClaimTemplate": {
                            "name": "cell-compute-domain-channel",
                        }
                    },
                    "numNodes": 0,
                },
            },
        )
    path.write_text("\n---\n".join(json.dumps(document) for document in documents) + "\n")
    return path


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
        backend_policy=BackendPolicy("baseline", {}, {}),
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
        capability=SimpleNamespace(
            architecture="GlmMoeDsaForCausalLM",
            model_config=load_model_config("nvidia/GLM-5.2-NVFP4"),
            support_level="native",
            template_id=None,
            template_version=None,
            aic_database_version="test",
        ),
        aic_revision="test-revision",
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
    assert len(calls) == 2
    assert calls[1][:2] == ("get", "Pod/pod-0")


def test_multidoc_manifest_tracks_compute_domain_and_workload(monkeypatch, tmp_path):
    manifest = _multinode_manifest(tmp_path / "k8s_deploy.yaml")
    monkeypatch.setattr(fpm_runner, "_kubectl_command", lambda: ["kubectl"])

    runner = KubernetesCellRunner(manifest, tmp_path)

    assert runner.kind == "LeaderWorkerSet"
    assert runner.name == "cell"
    assert runner.namespace == "test"
    assert runner.resources == [
        ("ComputeDomain", "cell-compute-domain", "test"),
        ("LeaderWorkerSet", "cell", "test"),
    ]
    assert fpm_runner._expected_nodes(manifest) == 2


def test_podcliqueset_manifest_tracks_resource_and_uses_cell_label_selector(monkeypatch, tmp_path):
    manifest = _podcliqueset_manifest(tmp_path / "k8s_deploy.yaml")
    monkeypatch.setattr(fpm_runner, "_kubectl_command", lambda: ["kubectl"])

    documents = fpm_runner._manifest_documents(manifest)
    runner = KubernetesCellRunner(manifest, tmp_path)

    assert [document["kind"] for document in documents] == ["PodCliqueSet"]
    assert runner.kind == "PodCliqueSet"
    assert runner.name == "cell"
    assert runner.namespace == "test"
    assert runner.selector == f"{FPM_CELL_LABEL}=cell"
    assert runner.resources == [("PodCliqueSet", "cell", "test")]


def test_selector_uses_the_same_cell_label_for_every_workload_kind(monkeypatch, tmp_path):
    """Contract: pods are selected exclusively through FPM_CELL_LABEL with one
    spelling, whatever the workload kind."""

    monkeypatch.setattr(fpm_runner, "_kubectl_command", lambda: ["kubectl"])
    pod_manifest = tmp_path / "pod.yaml"
    pod_manifest.write_text(
        json.dumps(
            {
                "apiVersion": "v1",
                "kind": "Pod",
                "metadata": {
                    "name": "cell",
                    "namespace": "test",
                    "labels": {FPM_CELL_LABEL: "cell"},
                },
            }
        )
    )
    manifests = [
        pod_manifest,
        _multinode_manifest(tmp_path / "lws.yaml"),
        _podcliqueset_manifest(tmp_path / "pcs.yaml"),
    ]

    selectors = {KubernetesCellRunner(manifest, tmp_path).selector for manifest in manifests}

    assert selectors == {f"{FPM_CELL_LABEL}=cell"}


def test_workload_missing_the_cell_label_is_rejected_as_contract_violation(monkeypatch, tmp_path):
    monkeypatch.setattr(fpm_runner, "_kubectl_command", lambda: ["kubectl"])
    manifest = tmp_path / "pod.yaml"
    manifest.write_text(
        json.dumps(
            {
                "apiVersion": "v1",
                "kind": "Pod",
                "metadata": {
                    "name": "cell",
                    "namespace": "test",
                    "labels": {"app.kubernetes.io/name": "cell"},
                },
            }
        )
    )

    with pytest.raises(ValueError, match="does not carry the collector cell label"):
        KubernetesCellRunner(manifest, tmp_path)


def test_multidoc_manifest_tracks_compute_domain_and_podcliqueset(monkeypatch, tmp_path):
    manifest = _podcliqueset_manifest(
        tmp_path / "k8s_deploy.yaml",
        include_compute_domain=True,
    )
    monkeypatch.setattr(fpm_runner, "_kubectl_command", lambda: ["kubectl"])

    runner = KubernetesCellRunner(manifest, tmp_path)

    assert runner.kind == "PodCliqueSet"
    assert runner.resources == [
        ("ComputeDomain", "cell-compute-domain", "test"),
        ("PodCliqueSet", "cell", "test"),
    ]
    assert fpm_runner._expected_nodes(manifest) == 4


def test_expected_nodes_sums_podcliqueset_clique_replicas(tmp_path):
    manifest = _podcliqueset_manifest(
        tmp_path / "k8s_deploy.yaml",
        spec={
            "replicas": 1,
            "template": {
                "cliques": [
                    {"name": "leader", "spec": {"replicas": 1}},
                    {"name": "worker-a", "spec": {"replicas": 2}},
                    {"name": "worker-b", "spec": {"replicas": 4}},
                ]
            },
        },
    )

    assert fpm_runner._expected_nodes(manifest) == 7


@pytest.mark.parametrize(
    "replicas",
    [None, 0, 2, True],
    ids=["missing", "zero", "multiple", "boolean"],
)
def test_expected_nodes_rejects_podcliqueset_replicas_other_than_one(tmp_path, replicas):
    spec = {
        "template": {
            "cliques": [
                {"name": "leader", "spec": {"replicas": 1}},
            ]
        }
    }
    if replicas is not None:
        spec["replicas"] = replicas
    manifest = _podcliqueset_manifest(tmp_path / "k8s_deploy.yaml", spec=spec)

    with pytest.raises(ValueError, match=r"PodCliqueSet spec\.replicas must be 1"):
        fpm_runner._expected_nodes(manifest)


@pytest.mark.parametrize(
    "cliques",
    [[], {}, "leader"],
    ids=["empty", "mapping", "string"],
)
def test_expected_nodes_rejects_empty_or_non_list_podcliqueset_cliques(tmp_path, cliques):
    manifest = _podcliqueset_manifest(
        tmp_path / "k8s_deploy.yaml",
        spec={"replicas": 1, "template": {"cliques": cliques}},
    )

    with pytest.raises(ValueError, match="requires at least one clique"):
        fpm_runner._expected_nodes(manifest)


@pytest.mark.parametrize(
    ("clique", "error_type"),
    [
        (None, TypeError),
        ({}, TypeError),
        ({"name": "worker", "spec": {}}, ValueError),
        ({"name": "worker", "spec": {"replicas": 0}}, ValueError),
        ({"name": "worker", "spec": {"replicas": -1}}, ValueError),
        ({"name": "worker", "spec": {"replicas": True}}, ValueError),
        ({"name": "worker", "spec": {"replicas": "2"}}, ValueError),
        ("worker", TypeError),
    ],
    ids=[
        "null",
        "missing-spec",
        "missing-replicas",
        "zero",
        "negative",
        "boolean",
        "string-replicas",
        "non-mapping",
    ],
)
def test_expected_nodes_rejects_invalid_podcliqueset_clique(tmp_path, clique, error_type):
    manifest = _podcliqueset_manifest(
        tmp_path / "k8s_deploy.yaml",
        spec={"replicas": 1, "template": {"cliques": [clique]}},
    )

    with pytest.raises(
        error_type,
        match=r"(cliques must be mappings with spec|clique replicas must be positive integers)",
    ):
        fpm_runner._expected_nodes(manifest)


def test_apply_multidoc_manifest_verifies_every_resource(monkeypatch, tmp_path):
    manifest = _multinode_manifest(tmp_path / "k8s_deploy.yaml")
    monkeypatch.setattr(fpm_runner, "_kubectl_command", lambda: ["kubectl"])
    runner = KubernetesCellRunner(manifest, tmp_path)
    responses = iter(
        [
            subprocess.CompletedProcess([], 0, stdout="resources configured\n", stderr=""),
            subprocess.CompletedProcess(
                [],
                0,
                stdout=json.dumps(
                    {
                        "apiVersion": "resource.nvidia.com/v1beta1",
                        "kind": "ComputeDomain",
                        "metadata": {"name": "cell-compute-domain", "namespace": "test"},
                    }
                ),
                stderr="",
            ),
            subprocess.CompletedProcess(
                [],
                0,
                stdout=json.dumps(
                    {
                        "apiVersion": "leaderworkerset.x-k8s.io/v1",
                        "kind": "LeaderWorkerSet",
                        "metadata": {
                            "name": "cell",
                            "namespace": "test",
                            "labels": {
                                "app.kubernetes.io/name": "cell",
                                "aiconfigurator.nvidia.com/cell": "cell",
                            },
                        },
                    }
                ),
                stderr="",
            ),
        ]
    )
    calls = []

    def kubectl(*args, **kwargs):
        calls.append(args)
        return next(responses)

    runner._kubectl = kubectl
    runner.apply()

    assert calls[0][:2] == ("apply", "--validate=false")
    assert calls[1][:2] == ("get", "ComputeDomain/cell-compute-domain")
    assert calls[2][:2] == ("get", "LeaderWorkerSet/cell")


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
        runner._exec_checked("pod-0", ["bash", f"{REMOTE_WORKDIR}/fpm_exec.sh"], timeout=10)


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


def test_cleanup_verifies_every_multidoc_resource_is_deleted(monkeypatch, tmp_path):
    manifest = _multinode_manifest(tmp_path / "k8s_deploy.yaml")
    monkeypatch.setattr(fpm_runner, "_kubectl_command", lambda: ["kubectl"])
    runner = KubernetesCellRunner(manifest, tmp_path)
    responses = iter(
        [
            subprocess.CompletedProcess([], 0, stdout="resources deleted\n", stderr=""),
            subprocess.CompletedProcess(
                [],
                1,
                stdout="",
                stderr='Error from server (NotFound): computedomains "cell-compute-domain" not found',
            ),
            subprocess.CompletedProcess(
                [],
                1,
                stdout="",
                stderr='Error from server (NotFound): leaderworkersets "cell" not found',
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

    assert calls[1][:2] == ("get", "ComputeDomain/cell-compute-domain")
    assert calls[2][:2] == ("get", "LeaderWorkerSet/cell")


def test_cleanup_escalates_stuck_foreground_delete_to_background(monkeypatch, tmp_path):
    """A controller that keeps reconciling children of a terminating parent can
    stall the foreground cascade forever (observed with LWS v0.6.0 while the
    workload was still unscheduled); cleanup must re-drive the stuck parent
    with a background delete and still verify NotFound before declaring the
    cell clean."""

    monkeypatch.setattr(fpm_runner, "CLEANUP_PROBE_INTERVAL_SECONDS", 0.0)
    runner = _runner(tmp_path)
    runner.manifest = tmp_path / "k8s_deploy.yaml"
    runner.kind = "LeaderWorkerSet"
    runner.name = "cell-agg"
    responses = iter(
        [
            subprocess.CompletedProcess([], 1, stdout="", stderr="error: timed out waiting for the condition\n"),
            subprocess.CompletedProcess([], 0, stdout=json.dumps({"metadata": {"name": "cell-agg"}}), stderr=""),
            subprocess.CompletedProcess([], 0, stdout="leaderworkerset.x-k8s.io/cell-agg deleted\n", stderr=""),
            subprocess.CompletedProcess(
                [],
                1,
                stdout="",
                stderr='Error from server (NotFound): leaderworkersets "cell-agg" not found',
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
    assert calls[2][:2] == ("delete", "LeaderWorkerSet/cell-agg")
    assert "--cascade=background" in calls[2]


def test_cleanup_fails_closed_when_background_escalation_cannot_converge(monkeypatch, tmp_path):
    monkeypatch.setattr(fpm_runner, "CLEANUP_PROBE_INTERVAL_SECONDS", 0.0)
    monkeypatch.setattr(fpm_runner, "CLEANUP_ESCALATION_TIMEOUT_SECONDS", 0.0)
    runner = _runner(tmp_path)
    runner.manifest = tmp_path / "k8s_deploy.yaml"
    runner.kind = "LeaderWorkerSet"
    runner.name = "cell-agg"
    present = json.dumps({"metadata": {"name": "cell-agg"}})

    def kubectl(*args, **kwargs):
        if args[0] == "delete":
            failed = 1 if "--cascade=foreground" in args else 0
            return subprocess.CompletedProcess([], failed, stdout="", stderr="")
        return subprocess.CompletedProcess([], 0, stdout=present, stderr="")

    runner._kubectl = kubectl
    runner.pods = lambda **_kwargs: []

    with pytest.raises(RuntimeError, match="owned FPM resource remains after cleanup: LeaderWorkerSet/cell-agg"):
        runner.cleanup()


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


def test_cleanup_rejects_nonterminating_pod_after_foreground_delete(monkeypatch, tmp_path):
    monkeypatch.setattr(fpm_runner, "CLEANUP_PROBE_INTERVAL_SECONDS", 0.0)
    monkeypatch.setattr(fpm_runner, "CLEANUP_ESCALATION_TIMEOUT_SECONDS", 0.0)
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


def test_cleanup_rejects_a_remaining_workload(monkeypatch, tmp_path):
    monkeypatch.setattr(fpm_runner, "CLEANUP_PROBE_INTERVAL_SECONDS", 0.0)
    monkeypatch.setattr(fpm_runner, "CLEANUP_ESCALATION_TIMEOUT_SECONDS", 0.0)
    runner = _runner(tmp_path)
    runner.manifest = tmp_path / "k8s_deploy.yaml"
    runner.kind = "Pod"
    runner.name = "pod-0"
    present = subprocess.CompletedProcess(
        [],
        0,
        stdout=json.dumps({"kind": "Pod", "metadata": {"name": "pod-0"}}),
        stderr="",
    )
    responses = iter(
        [
            subprocess.CompletedProcess([], 0, stdout="pod/pod-0 deleted\n", stderr=""),
            present,
            subprocess.CompletedProcess([], 0, stdout="pod/pod-0 deleted\n", stderr=""),
            present,
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


def test_cell_generator_overrides_preserve_explicit_scheduler_and_merge_resource_labels():
    cell = _cell()
    base = {
        "K8sConfig": {
            "fpm_resource_labels": {"kai.scheduler/queue": "team-a"},
            "worker_extra_pod_spec": {"schedulerName": "kai-scheduler"},
        }
    }

    overrides = _cell_generator_overrides(_plan(cell), cell, base)
    k8s = overrides["K8sConfig"]

    assert k8s["worker_extra_pod_spec"]["schedulerName"] == "kai-scheduler"
    assert k8s["fpm_resource_labels"] == {
        "kai.scheduler/queue": "team-a",
        "aiconfigurator.nvidia.com/owned-by": "fpm-forward-collector",
        "aiconfigurator.nvidia.com/plan": "plan-sha",
        "aiconfigurator.nvidia.com/cell": cell.cell_id,
    }


def test_policy_override_cannot_change_the_cell_identity_label():
    """Pod-selection isolation rests on FPM_CELL_LABEL equaling the cell id;
    a backend policy that remaps it after the merge must fail the render."""

    cell = _cell()
    hijacking_policy = BackendPolicy(
        "hijack",
        {"K8sConfig": {"fpm_resource_labels": {FPM_CELL_LABEL: "another-cell"}}},
        {},
    )
    cell = dataclasses.replace(cell, backend_policy=hijacking_policy)

    with pytest.raises(ValueError, match="must equal the cell id after overrides"):
        _cell_generator_overrides(_plan(cell), cell, {})


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
    assert args[args.index("--benchmark-timeout") + 1] == "10800"
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


def test_native_collection_validation_accepts_execution_order_decoupled_from_ids(tmp_path):
    """KV warm-up reorders execution (batch/kv-depth descending, fake points
    last) and the engine writes results in execution order, so file order is
    NOT benchmark_id order (engine contract: the two are decoupled). The
    validator must accept any permutation whose ID set is exactly 1..N and
    canonicalize by ID."""

    cell = _cell(phase="decode")
    execution_order = [3, 1, 4, 2]
    rows = []
    groups = []
    for benchmark_id in execution_order:
        point = {
            "point_type": "decode",
            "benchmark_id": benchmark_id,
            "total_prefill_tokens": 0,
            "total_kv_read_tokens": 128,
            "batch_size": 4,
            "expected_cudagraph_mode": "FULL",
            "expected_capture_size": 4,
            "padding_tokens": 0,
            "sample_reasons": ["capture"],
        }
        fpm = {
            "counter_id": benchmark_id,
            "dp_rank": 0,
            "wall_time": 0.01 * benchmark_id,
            "scheduled_requests": {
                "num_prefill_requests": 0,
                "sum_prefill_tokens": 0,
                "sum_prefill_kv_tokens": 0,
                "num_decode_requests": 4,
                "sum_decode_kv_tokens": 128,
            },
        }
        rows.append({"point": point, "fpms": [fpm]})
        groups.append(
            {
                "benchmark_id": benchmark_id,
                "point": point,
                "expected_dp_ranks": [0],
                "complete": True,
                "wall_time": fpm["wall_time"],
                "rank_results": [{"dp_rank": 0, "fpms": [fpm]}],
            }
        )
    measured = sum(0.01 * benchmark_id for benchmark_id in execution_order)
    payload = {
        "schema_version": 2,
        "artifact_type": "rank",
        "status": "complete",
        "valid": True,
        "usable": True,
        "timing_valid": True,
        "stop_reason": None,
        "error": None,
        "run_id": "run",
        "grid_digest": "grid",
        "config": {"mode": "decode"},
        "coverage": {"expected_points": 4, "completed_points": 4, "skipped_points": 0},
        "dp": {"rank": 0, "size": 1},
        "results": rows,
        "iteration_groups": groups,
        "skipped_points": [],
        "missing_phases": [],
        "timing": {"benchmark_elapsed_seconds": measured + 1.0, "measured_iteration_seconds": measured},
    }
    path = tmp_path / "pod-0" / "benchmark.json"
    path.parent.mkdir()
    _write_provenance(path.parent / "collector-provenance.json", cell_id=cell.cell_id)
    path.write_text(json.dumps(payload))

    assert _validate_runtime_collection(cell, tmp_path) == 4

    # A permutation with a gap (1,2,4,5) must still fail as non-contiguous.
    # counter_id tracks benchmark_id by contract, so mutate both (the row and
    # its group share the fpm object, one update covers each).
    for row, group, wrong_id in zip(rows, groups, [5, 1, 4, 2], strict=True):
        row["point"]["benchmark_id"] = wrong_id
        row["fpms"][0]["counter_id"] = wrong_id
        group["benchmark_id"] = wrong_id
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="not contiguous"):
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
        (cell_dir / "fpm_env.sh").write_text("#!/bin/sh\n")

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
    # One verified delete BEFORE apply (workload names are deterministic, so a
    # fresh checkpoint proves nothing about the cluster) and one after.
    assert events.count("cleanup") == 2
    assert events.index("cleanup") < events.index("apply")
    # Contract: the staged set is exactly the two rendered runtime artifacts
    # plus the collector's own in-pod runtime and preflight.
    assert set(staged_names) == {"run.sh", "fpm_env.sh", "fpm_exec.sh", "preflight.py"}
    assert "cases.json" not in staged_names
    assert "fpm_scheduler.py" not in staged_names
    assert "run_with_etcd.sh" not in staged_names
    checkpoint = json.loads((checkpoint_dir / "fpm_forward_smoke.json").read_text())
    assert checkpoint["cells"][cell.cell_id]["prefill_max_new_token_samples"] == 2
    assert checkpoint["cells"][cell.cell_id]["measured_new_token_axis_count"] == 2
    assert "prefill_requested_new_token_axis_count" not in checkpoint["cells"][cell.cell_id]


def test_partial_formal_run_is_campaign_incomplete_not_database_failure(monkeypatch, tmp_path):
    """A formal run limited below the frozen plan passes its targeted cells
    but cannot publish; it must report the honest campaign_incomplete
    classification and never record a failed database publication."""

    first = _cell()
    second = dataclasses.replace(first, cell_id="cell-prefill-second")
    plan = _plan(first)
    plan.cells = (first, second)

    def render_cell(*args, **_kwargs):
        cell_dir = args[2]
        (cell_dir / "k8s_deploy.yaml").write_text("apiVersion: v1\nkind: Pod\nmetadata:\n  name: cell\n")
        (cell_dir / "run.sh").write_text("#!/bin/sh\n")
        (cell_dir / "fpm_env.sh").write_text("#!/bin/sh\n")

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
            pass

    monkeypatch.setattr(fpm_runner, "_render_cell", render_cell)
    monkeypatch.setattr(fpm_runner, "KubernetesCellRunner", FakeResource)
    monkeypatch.setattr(
        fpm_runner,
        "_runtime_collection_summary",
        lambda *_args, **_kwargs: {"measured_point_count": 1},
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
        smoke=False,
        cell_limit=1,
    )

    assert [error["classification"] for error in errors] == ["campaign_incomplete"]
    checkpoint = json.loads((checkpoint_dir / "fpm_forward.json").read_text())
    assert checkpoint["cells"][first.cell_id]["status"] == "passed"
    assert second.cell_id not in checkpoint["cells"]
    assert "database" not in checkpoint


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
        (cell_dir / "fpm_env.sh").write_text("#!/bin/sh\n")

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

        cleanup_calls = 0

        def cleanup(self):
            # The pre-apply verified delete must succeed (a failure there
            # fails the cell before any work); only the post-run teardown
            # raises, which is the state this test pins.
            FakeResource.cleanup_calls += 1
            if FakeResource.cleanup_calls > 1:
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

    # The render call returns exactly the contract's three artifacts.
    for artifact in ("k8s_deploy.yaml", "fpm_env.sh", "run.sh"):
        assert (tmp_path / artifact).exists(), artifact

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

    # Thin run.sh contract face: no collection logic, ends with the setsid
    # foreground exec (the collector's fpm_exec.sh owns the checker and reads
    # the schema version from fpm_env.sh).
    assert "check_result_files" not in script
    assert 'value.get("schema_version")' not in script
    last_line = script.rstrip().splitlines()[-1]
    assert last_line.startswith("exec python3 -c")
    assert "os.setsid(); os.execvp" in last_line
    env_script = (tmp_path / "fpm_env.sh").read_text()
    assert "export FPM_RESULT_SCHEMA_VERSION=2" in env_script
    assert "export FPM_BENCHMARK_MODE=prefill" in env_script
    subprocess.run(["bash", "-n", str(tmp_path / "run.sh")], check=True)
    subprocess.run(["bash", "-n", str(tmp_path / "fpm_env.sh")], check=True)


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
        backend_policy=BackendPolicy("baseline", {}, {}),
        parallel_strategy="pure_tp",
        gemm_quant_mode=weight_quantization,
    )
    plan = _plan(cell)
    plan.model_path = model_path
    plan.capability = SimpleNamespace(
        architecture="GlmMoeDsaForCausalLM" if "GLM" in model_path else "DeepseekV4ForCausalLM",
        model_config=load_model_config(model_path),
    )

    _render_cell(plan, cell, tmp_path, {})

    script = (tmp_path / "run.sh").read_text()
    assert "--tensor-parallel-size 4" in script
    assert "--data-parallel-size 1" in script
    assert "--enable-expert-parallel" not in script


@pytest.mark.parametrize(
    ("workload_kind", "strategy", "topology", "flag_expected"),
    [
        # kvwarm regime (EP-sharded MoE): warm-depth reuse requires prefix
        # caching; disabling it collapses decode into the fake-KV fallback
        # (skip_reason="prefix_caching_disabled").
        pytest.param(
            "decode", "tep", ParallelTopology(tp=4, pp=1, dp=1, moe_tp=1, moe_ep=4, cp=1), False, id="decode-tep"
        ),
        pytest.param(
            "decode", "dep", ParallelTopology(tp=1, pp=1, dp=4, moe_tp=1, moe_ep=4, cp=1), False, id="decode-dep"
        ),
        # pure_tp joined the warm set (A2, 2026-08-20): expert-activation
        # dispersion is content-driven even with cross-GPU balance by
        # construction, so warm coverage applies (engine gc-warmtp+).
        pytest.param(
            "decode",
            "pure_tp",
            ParallelTopology(tp=4, pp=1, dp=1, moe_tp=4, moe_ep=1, cp=1),
            False,
            id="decode-pure-tp",
        ),
        # prefill never disables prefix caching regardless of strategy.
        pytest.param(
            "prefill",
            "pure_tp",
            ParallelTopology(tp=4, pp=1, dp=1, moe_tp=4, moe_ep=1, cp=1),
            False,
            id="prefill-pure-tp",
        ),
        pytest.param(
            "prefill", "tep", ParallelTopology(tp=4, pp=1, dp=1, moe_tp=1, moe_ep=4, cp=1), False, id="prefill-tep"
        ),
    ],
)
def test_decode_prefix_caching_follows_kvwarm_regime(tmp_path, workload_kind, strategy, topology, flag_expected):
    """Prefix caching is regime-conditional, mirroring the engine's KV
    warm-up eligibility (tep/dep warm; pure_tp/dense stay fake-KV): warm
    cells must keep it on, fake-KV decode must keep it off, prefill never
    disables it."""

    cell = FPMCell(
        cell_id=f"prefix-cache-{workload_kind}-{strategy}",
        workload_kind=workload_kind,
        topology=topology,
        weight_quantization="nvfp4",
        kv_cache_dtype="fp8",
        backend_policy=BackendPolicy("baseline", {}, {}),
        parallel_strategy=strategy,
        gemm_quant_mode="nvfp4",
    )
    plan = _plan(cell)
    plan.model_path = "nvidia/GLM-5.2-NVFP4"
    plan.capability = SimpleNamespace(
        architecture="GlmMoeDsaForCausalLM",
        model_config=load_model_config("nvidia/GLM-5.2-NVFP4"),
    )

    _render_cell(plan, cell, tmp_path, {})

    script = (tmp_path / "run.sh").read_text()
    if flag_expected:
        assert "--no-enable-prefix-caching" in script
    else:
        assert "--no-enable-prefix-caching" not in script


def test_render_uses_frozen_model_config_without_resolving_model_path(tmp_path, monkeypatch):
    """A --fpm-model-config campaign identifies a checkpoint that exists only
    inside the cluster: render must be a pure function of the frozen plan.
    Freeze a real config under a nonexistent model_path, poison every model
    resolution entry point, and require the three artifacts plus a
    generator-request whose ModelConfig agrees with the frozen capability."""

    import aiconfigurator.sdk.utils as sdk_utils
    from aiconfigurator.generator import naive as generator_naive
    from collector.fpm_forward import planner as planner_module

    monkeypatch.setattr(planner_module, "_git_revision", lambda: "test-revision")

    def estimator_unavailable(*_args, **_kwargs):
        raise RuntimeError("the checkpoint is accessible only inside the runtime Pod")

    monkeypatch.setattr(
        "collector.fpm_forward.memory_admission.KVCacheEstimator.from_request",
        estimator_unavailable,
    )

    cache_root = Path(fpm_runner.__file__).resolve().parents[2] / "src" / "aiconfigurator" / "model_configs"
    config_dir = tmp_path / "private-model-config"
    config_dir.mkdir()
    shutil.copyfile(cache_root / "nvidia--GLM-5.2-NVFP4_config.json", config_dir / "config.json")
    shutil.copyfile(cache_root / "nvidia--GLM-5.2-NVFP4_hf_quant_config.json", config_dir / "hf_quant_config.json")

    args = argparse.Namespace(
        fpm_max_gpus=4,
        fpm_gpu_counts=[4],
        fpm_parallel_presets=None,
        fpm_parallel_axes=None,
        fpm_moe_backend=None,
        fpm_attention_backend=None,
        fpm_enable_wideep=None,
        fpm_enable_eplb=None,
        fpm_weight_quantizations=None,
        fpm_kv_cache_dtypes=None,
        fpm_tp_sizes=None,
        fpm_pp_sizes=None,
        fpm_dp_sizes=None,
        fpm_moe_tp_sizes=None,
        fpm_moe_ep_sizes=None,
        fpm_cp_sizes=None,
        fpm_warmup_iterations=None,
        fpm_max_prefill_isl=None,
        fpm_max_prefill_batch_size=None,
    )
    plan = build_collection_plan(
        backend="vllm",
        model_path="private-org/pod-only-checkpoint",
        model_architecture="GlmMoeDsaForCausalLM",
        system="b200_sxm",
        selected_ops={"dsa_context_module", "dsa_generation_module"},
        options=FPMCollectionOptions.from_args(args),
        model_config_path=str(config_dir),
    )
    assert plan.capability.model_config.source_kind == "explicit"

    def fail_resolution(*_args, **_kwargs):
        raise AssertionError("render must not re-resolve model metadata from plan.model_path")

    monkeypatch.setattr(sdk_utils, "get_model_config_from_model_path", fail_resolution)
    monkeypatch.setattr(generator_naive, "get_model_config_from_model_path", fail_resolution)

    cell = plan.cells[0]
    cell_dir = tmp_path / "cell"
    cell_dir.mkdir()
    _render_cell(plan, cell, cell_dir, {})

    for artifact in ("k8s_deploy.yaml", "fpm_env.sh", "run.sh"):
        assert (cell_dir / artifact).exists(), artifact
    request = json.loads((cell_dir / "generator-request.json").read_text())
    assert plan.capability.is_moe is True
    assert request["ModelConfig"]["is_moe"] is plan.capability.is_moe


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


def _running_cell_fixture(tmp_path, plan, cell):
    artifact_root = tmp_path / "artifacts"
    cell_dir = artifact_root / plan.sha256[:16] / "smoke" / "cells" / cell.cell_id
    raw = cell_dir / "raw" / "pod-0"
    raw.mkdir(parents=True)
    _write_provenance(
        raw / "collector-provenance.json",
        cell_id=cell.cell_id,
        plan_sha256=plan.sha256,
        attempt_id="attempt-1",
    )
    (raw / "benchmark.json").write_text(json.dumps(_native_payload(phase="prefill", rank=0, dp=1)))
    (cell_dir / "k8s_deploy.yaml").write_text("apiVersion: v1\nkind: Pod\nmetadata:\n  name: cell\n")
    (cell_dir / "run.sh").write_text("#!/bin/sh\n")
    (cell_dir / "fpm_env.sh").write_text("#!/bin/sh\n")

    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    checkpoint_path = checkpoint_dir / "fpm_forward_smoke.json"
    checkpoint_path.write_text(
        json.dumps(
            {
                "schema": fpm_runner.CHECKPOINT_SCHEMA,
                "plan_sha256": plan.sha256,
                "cells": {cell.cell_id: {"status": "running", "attempt_id": "attempt-1"}},
            }
        )
    )
    return artifact_root, checkpoint_dir, checkpoint_path


def test_running_recovery_verifies_teardown_of_the_abandoned_workload(monkeypatch, tmp_path):
    """A persistent status=running means the finally block never ran, so the
    workload may still be alive holding GPUs; recovery must drive a verified
    delete before the entry can flip to passed (after which the cell is
    skipped forever and nothing else would)."""

    cell = _cell()
    plan = _plan(cell)
    artifact_root, checkpoint_dir, checkpoint_path = _running_cell_fixture(tmp_path, plan, cell)

    cleanups = []

    class FakeResource:
        def __init__(self, manifest, _cell_dir):
            self.manifest = manifest

        def cleanup(self):
            cleanups.append(str(self.manifest))

    monkeypatch.setattr(fpm_runner, "KubernetesCellRunner", FakeResource)

    def reject_rerun(*_args, **_kwargs):
        raise AssertionError("a recovered running cell must not rerun on the cluster")

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
    assert len(cleanups) == 1
    record = json.loads(checkpoint_path.read_text())["cells"][cell.cell_id]
    assert record["status"] == "passed"
    assert record["artifact_recovery"]["original_status"] == "running"


def test_running_entry_with_failed_teardown_reruns_instead_of_passing(monkeypatch, tmp_path):
    cell = _cell()
    plan = _plan(cell)
    artifact_root, checkpoint_dir, checkpoint_path = _running_cell_fixture(tmp_path, plan, cell)

    class FakeResource:
        def __init__(self, _manifest, _cell_dir):
            pass

        def cleanup(self):
            raise RuntimeError("kubectl unreachable")

    monkeypatch.setattr(fpm_runner, "KubernetesCellRunner", FakeResource)
    render_calls = []
    monkeypatch.setattr(fpm_runner, "_render_cell", lambda *args, **kwargs: render_calls.append(args))

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

    # Recovery was refused, the cell reran (and failed on the same unreachable
    # cluster) — it must never silently become a passed entry.
    assert render_calls
    record = json.loads(checkpoint_path.read_text())["cells"][cell.cell_id]
    assert record["status"] == "failed"
    assert record["cleanup_error"] == "kubectl unreachable"
    assert [error["classification"] for error in errors] == [
        "campaign_cell_failed",
        "resource_cleanup_failed",
    ]


def test_sigterm_is_routed_through_the_interrupt_path():
    before = signal.getsignal(signal.SIGTERM)
    with pytest.raises(KeyboardInterrupt), fpm_runner._sigterm_as_interrupt():
        os.kill(os.getpid(), signal.SIGTERM)
        for _ in range(10_000):
            time.sleep(0.001)
        raise AssertionError("SIGTERM was not converted to KeyboardInterrupt")
    assert signal.getsignal(signal.SIGTERM) is before


def test_terminate_active_commands_unblocks_run_command():
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(fpm_runner._run_command, ["sleep", "30"], check=False)
        deadline = time.monotonic() + 10
        while fpm_runner.terminate_active_commands() == 0:
            if time.monotonic() > deadline:
                raise AssertionError("the command never registered as terminable")
            time.sleep(0.01)
        completed = future.result(timeout=10)
    assert completed.returncode != 0


def test_run_collection_fails_cell_when_a_rendered_artifact_is_missing(monkeypatch, tmp_path):
    """The render-existence check covers all three contract artifacts; a
    Generator that stops emitting fpm_env.sh must fail the cell up front."""

    cell = _cell()
    plan = _plan(cell)

    def render_cell(*args, **kwargs):
        cell_dir = args[2]
        (cell_dir / "k8s_deploy.yaml").write_text("apiVersion: v1\nkind: Pod\nmetadata:\n  name: cell\n")
        (cell_dir / "run.sh").write_text("#!/bin/sh\n")

    monkeypatch.setattr(fpm_runner, "_render_cell", render_cell)
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

    assert errors
    record = json.loads((checkpoint_dir / "fpm_forward_smoke.json").read_text())["cells"][cell.cell_id]
    assert record["status"] == "failed"
    assert "did not emit" in record["error"]
    assert "fpm_env.sh" in record["error"]


def test_run_command_kills_child_on_interrupt():
    """A KeyboardInterrupt landing while the main thread is blocked in the
    child wait must kill the child (Popen.__exit__ would otherwise block
    forever on a hung kubectl that is no longer reachable via the registry)."""

    def _raise(signum, frame):
        raise KeyboardInterrupt

    previous = signal.signal(signal.SIGALRM, _raise)
    try:
        signal.setitimer(signal.ITIMER_REAL, 0.2)
        start = time.monotonic()
        with pytest.raises(KeyboardInterrupt):
            fpm_runner._run_command(["bash", "-c", "sleep 30 & exec sleep 30"], check=False)
        assert time.monotonic() - start < 15
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def test_run_command_timeout_does_not_drain_orphaned_pipe_holders():
    """tsh kubectl re-execs a pipe-sharing grandchild that survives SIGKILL on
    the wrapper; the timeout path must not drain pipes it cannot close (the
    background sleep here plays the orphan holding stdout open)."""

    start = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired):
        fpm_runner._run_command(["bash", "-c", "sleep 30 & exec sleep 30"], check=False, timeout=1)
    assert time.monotonic() - start < 15


def test_pre_apply_cleanup_failure_blocks_apply(monkeypatch, tmp_path):
    """The unconditional pre-apply verified delete must fail CLOSED: if the
    stale-workload delete cannot be verified, apply() must never run (applying
    would adopt the possibly-live workload)."""

    cell = _cell()
    plan = _plan(cell)

    def render_cell(*args, **kwargs):
        cell_dir = args[2]
        (cell_dir / "k8s_deploy.yaml").write_text("apiVersion: v1\nkind: Pod\nmetadata:\n  name: cell\n")
        (cell_dir / "run.sh").write_text("#!/bin/sh\n")
        (cell_dir / "fpm_env.sh").write_text("#!/bin/sh\n")

    applied = []

    class FakeResource:
        def __init__(self, _manifest, _cell_dir):
            pass

        def cleanup(self):
            raise RuntimeError("delete verification failed")

        def apply(self):
            applied.append(True)

    monkeypatch.setattr(fpm_runner, "_render_cell", render_cell)
    monkeypatch.setattr(fpm_runner, "KubernetesCellRunner", FakeResource)
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

    assert applied == []
    record = json.loads((checkpoint_dir / "fpm_forward_smoke.json").read_text())["cells"][cell.cell_id]
    assert record["status"] == "failed"
    assert "delete verification failed" in record["error"]
    assert errors


def test_running_entry_without_manifest_reruns_instead_of_passing(monkeypatch, tmp_path):
    """No manifest means the abandoned workload's teardown cannot be verified;
    the running entry must fall through to a rerun, never flip to passed."""

    cell = _cell()
    plan = _plan(cell)
    artifact_root, checkpoint_dir, checkpoint_path = _running_cell_fixture(tmp_path, plan, cell)
    cell_dir = artifact_root / plan.sha256[:16] / "smoke" / "cells" / cell.cell_id
    (cell_dir / "k8s_deploy.yaml").unlink()
    (cell_dir / "run.sh").unlink()

    render_calls = []
    monkeypatch.setattr(fpm_runner, "_render_cell", lambda *args, **kwargs: render_calls.append(args))

    run_collection(
        plan,
        generator_overrides={},
        checkpoint_dir=str(checkpoint_dir),
        artifact_root=str(artifact_root),
        resume=True,
        retry_failed=False,
        smoke=True,
        cell_limit=1,
    )

    assert render_calls
    record = json.loads(checkpoint_path.read_text())["cells"][cell.cell_id]
    assert record["status"] == "failed"


def test_resume_survives_pruned_raw_artifacts_of_passed_cell(monkeypatch, tmp_path, caplog):
    # The passed-cell metadata refresh re-reads raw artifacts that operators
    # legitimately prune after publication; their absence must not abort the
    # resume or demote the cell.
    cell = _cell()
    plan = _plan(cell)
    artifact_root = tmp_path / "artifacts"

    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    checkpoint_path = checkpoint_dir / "fpm_forward_smoke.json"
    checkpoint_path.write_text(
        json.dumps(
            {
                "schema": fpm_runner.CHECKPOINT_SCHEMA,
                "plan_sha256": plan.sha256,
                "cells": {cell.cell_id: {"status": "passed", "attempt_id": "attempt-1"}},
            }
        )
    )

    def reject_rerun(*_args, **_kwargs):
        raise AssertionError("a passed cell must not rerun when its raw artifacts were pruned")

    monkeypatch.setattr(fpm_runner, "_render_cell", reject_rerun)

    with caplog.at_level("WARNING", logger=fpm_runner.logger.name):
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
    assert record["attempt_id"] == "attempt-1"
    assert any("Skipping metadata refresh" in message for message in caplog.messages)


def test_user_extra_env_reaches_the_render_request_alongside_collector_identities():
    cell = _cell()
    plan = _plan(cell)
    base = {"K8sConfig": {"extra_env": [{"name": "VLLM_FLASHINFER_ALLREDUCE_BACKEND", "value": "trtllm"}]}}

    merged = _cell_generator_overrides(plan, cell, base)

    resolved = {item["name"]: item["value"] for item in merged["K8sConfig"]["extra_env"]}
    assert resolved["VLLM_FLASHINFER_ALLREDUCE_BACKEND"] == "trtllm"
    assert resolved["FPM_RUN_ID"] == cell.cell_id


def test_user_extra_env_conflicting_with_collector_identity_fails_closed():
    cell = _cell()
    plan = _plan(cell)
    base = {"K8sConfig": {"extra_env": [{"name": "FPM_RUN_ID", "value": "not-the-cell"}]}}

    with pytest.raises(ValueError, match="conflicting FPM environment value"):
        _cell_generator_overrides(plan, cell, base)


def test_publish_partial_ships_passed_cells_and_records_the_missing(tmp_path, monkeypatch):
    """--fpm-publish-partial: a plan with a legitimately-failed cell can still
    publish its passed cells; the checkpoint records exactly what is missing
    so coverage is auditable, and nothing is re-collected."""

    passed_cell = _cell()
    failed_cell = dataclasses.replace(passed_cell, cell_id="cell-prefill-failed")
    plan = _plan(passed_cell)
    plan.cells = (passed_cell, failed_cell)

    artifact_root = tmp_path / "artifacts"
    raw = artifact_root / plan.sha256[:16] / "cells" / passed_cell.cell_id / "raw" / "pod-0"
    raw.mkdir(parents=True)
    _write_provenance(
        raw / "collector-provenance.json",
        cell_id=passed_cell.cell_id,
        plan_sha256=plan.sha256,
        attempt_id="attempt-1",
    )
    (raw / "benchmark.json").write_text(json.dumps(_native_payload(phase="prefill", rank=0, dp=1)))

    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "fpm_forward.json").write_text(
        json.dumps(
            {
                "schema": fpm_runner.CHECKPOINT_SCHEMA,
                "plan_sha256": plan.sha256,
                "cells": {
                    passed_cell.cell_id: {"status": "passed", "attempt_id": "attempt-1"},
                    failed_cell.cell_id: {
                        "status": "failed",
                        "attempt_id": "attempt-2",
                        "error_type": "RuntimeError",
                        "error": "engine rejected the topology",
                    },
                },
            }
        )
    )

    # The database writer has its own coverage; stub it so this test isolates
    # the publication gate and the aggregation over passed cells.
    published = {}

    def fake_writer(plan_arg, rows, *, systems_root=None):
        published["rows"] = rows
        parquet = tmp_path / "db" / "fpm_forward_perf.parquet"
        parquet.parent.mkdir(parents=True, exist_ok=True)
        parquet.write_bytes(b"parquet")
        metadata = parquet.with_suffix(".metadata.json")
        metadata.write_text("{}")
        return parquet, metadata, ()

    monkeypatch.setattr("collector.fpm_forward.database.write_formal_database", fake_writer)

    errors = run_collection(
        plan,
        generator_overrides={},
        checkpoint_dir=str(checkpoint_dir),
        artifact_root=str(artifact_root),
        resume=True,
        retry_failed=False,
        smoke=False,
        database_root=str(tmp_path / "db"),
        publish_partial=True,
    )

    assert not [e for e in errors if e["classification"] == "formal_database_failed"]
    # Partial publication must not hide the failure: the run records the
    # missing cells as an error so the exit code stays nonzero.
    incomplete = [e for e in errors if e["classification"] == "campaign_incomplete"]
    assert len(incomplete) == 1
    assert failed_cell.cell_id in incomplete[0]["error_message"]
    checkpoint = json.loads((checkpoint_dir / "fpm_forward.json").read_text())
    database = checkpoint["database"]
    assert database["status"] == "passed"
    assert database["published_cells"] == 1
    assert database["plan_cells"] == 2
    assert database["missing_cells"] == [failed_cell.cell_id]
    assert database["row_count"] == len(published["rows"]) > 0
    assert Path(database["parquet"]).exists()


def test_publish_partial_never_publishes_smoke_rows(tmp_path, monkeypatch):
    """--fpm-publish-partial is a formal-run escape hatch: combined with
    --smoke it must not open a path for smoke rows into the formal database.

    The failed cell is first so the smoke target set is not all-passed: that
    is exactly the state where the partial-publication arm would otherwise
    pick up the passed cell's smoke rows and publish them."""

    passed_cell = _cell()
    failed_cell = dataclasses.replace(passed_cell, cell_id="cell-prefill-failed")
    plan = _plan(passed_cell)
    plan.cells = (failed_cell, passed_cell)

    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "fpm_forward_smoke.json").write_text(
        json.dumps(
            {
                "schema": fpm_runner.CHECKPOINT_SCHEMA,
                "plan_sha256": plan.sha256,
                "cells": {
                    passed_cell.cell_id: {"status": "passed", "attempt_id": "attempt-1"},
                    failed_cell.cell_id: {"status": "failed", "attempt_id": "attempt-2"},
                },
            }
        )
    )

    def forbidden_writer(*_args, **_kwargs):
        raise AssertionError("smoke rows must never reach write_formal_database")

    monkeypatch.setattr("collector.fpm_forward.database.write_formal_database", forbidden_writer)

    errors = run_collection(
        plan,
        generator_overrides={},
        checkpoint_dir=str(checkpoint_dir),
        artifact_root=str(tmp_path / "artifacts"),
        resume=True,
        retry_failed=False,
        smoke=True,
        database_root=str(tmp_path / "db"),
        publish_partial=True,
    )

    assert [e["classification"] for e in errors] == ["campaign_incomplete"]
    checkpoint = json.loads((checkpoint_dir / "fpm_forward_smoke.json").read_text())
    assert "database" not in checkpoint


def test_partial_run_without_the_flag_still_refuses_to_publish(tmp_path):
    """Default stays all-or-nothing: same setup, no flag, no database."""

    passed_cell = _cell()
    failed_cell = dataclasses.replace(passed_cell, cell_id="cell-prefill-failed")
    plan = _plan(passed_cell)
    plan.cells = (passed_cell, failed_cell)

    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "fpm_forward.json").write_text(
        json.dumps(
            {
                "schema": fpm_runner.CHECKPOINT_SCHEMA,
                "plan_sha256": plan.sha256,
                "cells": {
                    passed_cell.cell_id: {"status": "passed", "attempt_id": "attempt-1"},
                    failed_cell.cell_id: {"status": "failed", "attempt_id": "attempt-2"},
                },
            }
        )
    )

    errors = run_collection(
        plan,
        generator_overrides={},
        checkpoint_dir=str(checkpoint_dir),
        artifact_root=str(tmp_path / "artifacts"),
        resume=True,
        retry_failed=False,
        smoke=False,
        database_root=str(tmp_path / "db"),
    )

    assert [e["classification"] for e in errors] == ["campaign_incomplete"]
    checkpoint = json.loads((checkpoint_dir / "fpm_forward.json").read_text())
    assert "database" not in checkpoint


def test_run_manifest_records_collector_phases_and_engine_interface(monkeypatch, tmp_path):
    """R16 §3: every run writes a machine-readable timing manifest - collector
    phase segments now, engine phase fields as a declared interface until
    dynamo-fpm phase instrumentation lands."""

    cell = _cell()
    plan = _plan(cell)

    def render_cell(*args, **_kwargs):
        cell_dir = args[2]
        (cell_dir / "k8s_deploy.yaml").write_text("apiVersion: v1\nkind: Pod\nmetadata:\n  name: cell\n")
        (cell_dir / "run.sh").write_text("#!/bin/sh\n")
        (cell_dir / "fpm_env.sh").write_text("#!/bin/sh\n")

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
            pass

    monkeypatch.setattr(fpm_runner, "_render_cell", render_cell)
    monkeypatch.setattr(fpm_runner, "KubernetesCellRunner", FakeResource)
    monkeypatch.setattr(
        fpm_runner,
        "_runtime_collection_summary",
        lambda *_args, **_kwargs: {"measured_point_count": 1},
    )
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    artifact_root = tmp_path / "artifacts"

    run_collection(
        plan,
        generator_overrides={},
        checkpoint_dir=str(checkpoint_dir),
        artifact_root=str(artifact_root),
        resume=False,
        retry_failed=False,
        smoke=True,
    )

    manifest = json.loads((artifact_root / plan.sha256[:16] / "smoke" / "run-manifest.json").read_text())
    assert manifest["schema_name"] == "aic_fpm_run_manifest"
    assert manifest["run_total_s"] > 0
    entry = manifest["cells"][cell.cell_id]
    phases = entry["collector_phase_seconds"]
    assert set(phases) == {"render_s", "schedule_s", "stage_s", "execute_wall_s", "collect_s"}
    assert entry["engine_phase_seconds"]["kvwarm_warmup_s"] is None
    assert "pending dynamo-fpm" in entry["engine_phase_seconds"]["note"]
