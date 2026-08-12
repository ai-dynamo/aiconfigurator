# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct unit tests for the shared FPM contract functions.

Both sides of the Generator/collector boundary replicate these conventions
(in shell and in Python); the vectors here pin the reference implementation
that those replicas are contract-tested against.
"""

from __future__ import annotations

import pytest

from aiconfigurator.fpm_contract import (
    fpm_benchmark_result_name,
    fpm_expected_result_paths,
    fpm_validate_benchmark_output_path,
    fpm_workload_node_count,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("output_path", "dp_rank", "expected"),
    [
        pytest.param("run.v1/metrics.final.json", 0, "run.v1/metrics.final.json", id="rank0-dotted-passthrough"),
        pytest.param("run.v1/metrics.final.json", 1, "run.v1/metrics.final_dp1.json", id="dotted-nonzero-rank"),
        pytest.param("plain-noext", 3, "plain-noext_dp3", id="extensionless"),
        # Hidden files follow the engine's Path.stem/.suffix primitive: the
        # whole dotted name is the stem, so the rank suffix lands at the end.
        pytest.param("/results/.json", 1, "/results/.json_dp1", id="hidden-file-json-basename"),
        pytest.param("a/b/c/benchmark.json", 2, "a/b/c/benchmark_dp2.json", id="nested-directory"),
        pytest.param("/results/benchmark.smoke.json", 1, "/results/benchmark.smoke_dp1.json", id="multi-dot"),
    ],
)
def test_fpm_benchmark_result_name(output_path, dp_rank, expected):
    assert fpm_benchmark_result_name(output_path, dp_rank) == expected


@pytest.mark.parametrize("dp_rank", [-1, True, 1.5, "1"], ids=["negative", "bool", "float", "string"])
def test_fpm_benchmark_result_name_rejects_invalid_ranks(dp_rank):
    with pytest.raises(ValueError, match="dp_rank must be a non-negative integer"):
        fpm_benchmark_result_name("/results/benchmark.json", dp_rank)


@pytest.mark.parametrize(
    "output_path",
    [
        pytest.param("/results/benchmark.json", id="default"),
        pytest.param("/results/benchmark_smoke.json", id="custom-conforming"),
        pytest.param("/results/benchmark.smoke.json", id="multi-dot-conforming"),
    ],
)
def test_fpm_validate_benchmark_output_path_accepts_discoverable_paths(output_path):
    fpm_validate_benchmark_output_path(output_path)


@pytest.mark.parametrize(
    ("output_path", "match"),
    [
        pytest.param("/results/custom.json", "basename must match", id="glob-mismatch"),
        pytest.param("/tmp/benchmark.json", "must live under /results", id="outside-results"),
        pytest.param("/results/../etc/benchmark.json", "without '.' or '..'", id="traversal"),
        pytest.param("relative/benchmark.json", "must be absolute", id="relative"),
        pytest.param("/results/benchmark", "basename must match", id="extensionless"),
        pytest.param("/results", "must live under /results", id="results-dir-itself"),
    ],
)
def test_fpm_validate_benchmark_output_path_rejects_undiscoverable_paths(output_path, match):
    with pytest.raises(ValueError, match=match):
        fpm_validate_benchmark_output_path(output_path)


def test_every_gate_accepted_path_is_discoverable_by_every_glob_consumer():
    # Cross-boundary invariant (per review): any output path the render
    # validator accepts must yield per-rank result names that every
    # glob-driven collector discovery site can find under FPM_RESULTS_DIR.
    import fnmatch
    from pathlib import PurePosixPath

    from aiconfigurator.fpm_contract import FPM_BENCHMARK_RESULT_GLOB, FPM_RESULTS_DIR

    accepted = ["/results/benchmark.json", "/results/benchmark_smoke.json", "/results/benchmark.v2.json"]
    for base in accepted:
        fpm_validate_benchmark_output_path(base)
        for dp_rank in range(0, 5):
            name = PurePosixPath(fpm_benchmark_result_name(base, dp_rank))
            assert fnmatch.fnmatch(name.name, FPM_BENCHMARK_RESULT_GLOB), name
            assert PurePosixPath(FPM_RESULTS_DIR) in name.parents, name


def test_fpm_expected_result_paths_window_covers_the_node_local_rank_range():
    # Global rank zero keeps the unsuffixed benchmark.json.
    assert fpm_expected_result_paths("/results/benchmark.json", 0, 2) == [
        "/results/benchmark.json",
        "/results/benchmark_dp1.json",
    ]
    assert fpm_expected_result_paths("/results/benchmark.json", 1, 2) == [
        "/results/benchmark_dp2.json",
        "/results/benchmark_dp3.json",
    ]


@pytest.mark.parametrize(
    ("node_rank", "local_data_parallel_size"),
    [
        pytest.param(-1, 2, id="negative-node-rank"),
        pytest.param(0, 0, id="local-size-below-one"),
        pytest.param(True, 2, id="bool-node-rank"),
        pytest.param(False, 2, id="false-node-rank"),
        pytest.param(0, True, id="bool-local-size"),
        pytest.param(1.0, 2, id="float-node-rank"),
        pytest.param(0, 2.0, id="float-local-size"),
    ],
)
def test_fpm_expected_result_paths_rejects_invalid_window(node_rank, local_data_parallel_size):
    with pytest.raises(ValueError, match="node_rank must be a non-negative integer"):
        fpm_expected_result_paths("/results/benchmark.json", node_rank, local_data_parallel_size)


def _podcliqueset(spec: object) -> dict:
    return {"kind": "PodCliqueSet", "spec": spec}


@pytest.mark.parametrize(
    ("workload", "expected"),
    [
        pytest.param({"kind": "Pod"}, 1, id="pod-is-one-node"),
        pytest.param(
            {"kind": "LeaderWorkerSet", "spec": {"leaderWorkerTemplate": {"size": 4}}},
            4,
            id="lws-size",
        ),
        pytest.param(
            _podcliqueset(
                {
                    "replicas": 1,
                    "template": {
                        "cliques": [
                            {"name": "leader", "spec": {"replicas": 1}},
                            {"name": "worker", "spec": {"replicas": 3}},
                        ]
                    },
                }
            ),
            4,
            id="pcs-sums-clique-replicas",
        ),
    ],
)
def test_fpm_workload_node_count(workload, expected):
    assert fpm_workload_node_count(workload) == expected


@pytest.mark.parametrize(
    "replicas",
    [pytest.param(2, id="multiple"), pytest.param(True, id="boolean")],
)
def test_fpm_workload_node_count_rejects_podcliqueset_replicas_other_than_one(replicas):
    workload = _podcliqueset(
        {
            "replicas": replicas,
            "template": {"cliques": [{"name": "worker", "spec": {"replicas": 1}}]},
        }
    )

    with pytest.raises(ValueError, match=r"PodCliqueSet spec\.replicas must be 1"):
        fpm_workload_node_count(workload)


@pytest.mark.parametrize(
    "template",
    [pytest.param({}, id="missing-cliques"), pytest.param({"cliques": []}, id="empty-cliques")],
)
def test_fpm_workload_node_count_requires_at_least_one_clique(template):
    workload = _podcliqueset({"replicas": 1, "template": template})

    with pytest.raises(ValueError, match="requires at least one clique"):
        fpm_workload_node_count(workload)


@pytest.mark.parametrize(
    "clique",
    [
        pytest.param("worker", id="non-mapping-clique"),
        pytest.param({"name": "worker", "spec": "x"}, id="non-mapping-spec"),
    ],
)
def test_fpm_workload_node_count_rejects_non_mapping_clique_spec(clique):
    workload = _podcliqueset({"replicas": 1, "template": {"cliques": [clique]}})

    with pytest.raises(TypeError, match="cliques must be mappings with spec"):
        fpm_workload_node_count(workload)


def test_fpm_workload_node_count_rejects_unknown_kind():
    with pytest.raises(ValueError, match="unsupported generated FPM workload kind: Deployment"):
        fpm_workload_node_count({"kind": "Deployment"})
