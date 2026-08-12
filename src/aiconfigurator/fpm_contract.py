# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared contract between the Generator FPM target and the FPM collector.

This module is the single home for every fact the Generator's FPM deployment
target (``src/aiconfigurator/generator/builders/fpm_builder.py``) and the FPM
collector (``collector/fpm_forward/``) must agree on. Neither side may spell
one of these facts locally; both import it from here, so a change is visible
to both owners in review.

The complete coupling surface between the two modules is:

1. **Render call** -- the collector invokes the public Generator API once per
   cell with the declared deployment-input schema (``deployment_config.yaml``
   keys plus ``params.<role>``); the Generator returns the artifacts named
   below.
2. **Rendered artifacts** -- ``k8s_deploy.yaml`` (only the resource kinds
   below, every pod template carrying the collector's identity label),
   ``fpm_env.sh`` (exports exactly :data:`FPM_ENV_EXPORTED_VARS`, plus
   :data:`FPM_BARRIER_TIMEOUT_ENV` when the operator supplies it through
   ``extra_env``), and ``run.sh`` (launches the engine in its own session
   via ``setsid`` and exits with the engine's exit code; contains no
   collection logic).
3. **This constants table.**
4. **The engine's result-file convention** -- Dynamo's native self-benchmark
   names per-DP-rank outputs by suffixing the configured output path
   (:func:`fpm_benchmark_result_name`). That convention belongs to the
   engine; both sides only replicate it.

In-pod consumers (``fpm_env.sh``, ``run.sh``, and the collector's staged
``fpm_exec.sh``) cannot import this module because the deployed image does
not ship ``aiconfigurator``. Their replicas of these facts are pinned by
contract tests on both sides instead:

* generator tests assert the rendered ``fpm_env.sh`` exports exactly
  :data:`FPM_ENV_EXPORTED_VARS` (plus the forwarded barrier tunable when
  supplied) and that ``run.sh`` consumes no ``FPM_*`` variables beyond
  :data:`FPM_ENV_EXPORTED_VARS` and :data:`FPM_RUN_ID_ENV`;
* collector tests assert ``fpm_exec.sh`` consumes only
  :data:`FPM_ENV_EXPORTED_VARS` (plus the :data:`FPM_BARRIER_TIMEOUT_ENV`
  tunable it defaults in-script) and that its result-naming shell function
  matches :func:`fpm_benchmark_result_name` on a shared vector set.
"""

from __future__ import annotations

import fnmatch
from pathlib import PurePosixPath
from typing import Any

# Schema version of the native benchmark result envelope written by Dynamo's
# InstrumentedScheduler (PR 11509). The collector's validator and the staged
# runtime's completion checker accept exactly this version.
FPM_NATIVE_BENCHMARK_RESULT_SCHEMA_VERSION = 2

# Artifact names the Generator FPM target emits and the collector stages into
# every pod's working directory.
FPM_MANIFEST_FILENAME = "k8s_deploy.yaml"
FPM_RUN_SCRIPT_FILENAME = "run.sh"
FPM_ENV_FILENAME = "fpm_env.sh"

# Resource kinds a generated FPM manifest may contain. The collector applies
# the manifest blind and verifies each document individually; any other kind
# fails closed on both sides.
FPM_WORKLOAD_KINDS = frozenset({"Pod", "LeaderWorkerSet", "PodCliqueSet"})
FPM_AUXILIARY_KINDS = frozenset({"ComputeDomain"})

# Collector-owned pod identity label. The collector passes it (valued with the
# cell id) through ``K8sConfig.fpm_resource_labels``; the Generator must apply
# workload metadata labels to every pod template of every workload kind, and
# the collector selects pods exclusively through this label -- it never
# depends on Generator-chosen labels or on the workload kind.
FPM_CELL_LABEL = "aiconfigurator.nvidia.com/cell"

# Environment variables exported by the rendered ``fpm_env.sh``. Everything
# the collector's in-pod runtime needs from the Generator travels through
# these exports; the runtime consumes no other render-time fact.
#
# FPM_NODE_RANK / FPM_MASTER_ADDR are derived from the orchestrator's pod
# environment (LWS_WORKER_INDEX / LWS_LEADER_ADDRESS for LeaderWorkerSet,
# GROVE_PCLQ_POD_INDEX / GROVE_PCLQ_NAME + GROVE_HEADLESS_SERVICE for Grove).
# Pre-set FPM_NODE_RANK / FPM_MASTER_ADDR values take priority over both, so
# an operator-less environment can inject them explicitly. Discovery fails
# closed (exit 2) on a multinode cell whose environment yields no complete
# rank + leader answer, and on a non-numeric or out-of-range rank.
FPM_ENV_EXPORTED_VARS = (
    "FPM_NODE_COUNT",
    "FPM_DATA_PARALLEL_SIZE",
    "FPM_LOCAL_DATA_PARALLEL_SIZE",
    "FPM_BENCHMARK_MODE",
    "FPM_BENCHMARK_OUTPUT_PATH",
    "FPM_WAIT_TIMEOUT_SECONDS",
    "FPM_RESULT_SCHEMA_VERSION",
    "FPM_NODE_RANK",
    "FPM_MASTER_ADDR",
)

# Cross-boundary environment names the collector injects through the declared
# ``K8sConfig.extra_env`` input and the rendered scripts consume.
FPM_RUN_ID_ENV = "FPM_RUN_ID"
FPM_ENGINE_BENCHMARK_OUTPUT_ENV = "DYN_FPM_BENCHMARK_OUTPUT_PATH"

# Operator-tunable override for the DP completion barrier deadlines. Not a
# render fact: the collection runtime defaults it in-script. When supplied
# through ``K8sConfig.extra_env`` the Generator forwards it via ``fpm_env.sh``
# (never ``run.sh``) so it reaches the runtime that owns the barrier. Every
# other name in :data:`FPM_ENV_EXPORTED_VARS` is rejected in ``extra_env``:
# an export in ``run.sh`` would shadow the contract values for the engine
# while the collection runtime keeps the originals.
FPM_BARRIER_TIMEOUT_ENV = "FPM_COMPLETION_BARRIER_TIMEOUT_SECONDS"

# In-pod results directory: the Generator mounts it and defaults the engine's
# benchmark output under it; the collector's runtime and transfer own its
# contents. Shell/in-pod replicas of this path are pinned by tests.
FPM_RESULTS_DIR = "/results"

# Result files the engine writes under /results. The exact per-rank names come
# from :func:`fpm_benchmark_result_name`; this glob is the documented sweep
# pattern for sites that scan a results tree (the engine may also emit merged
# artifacts such as ``benchmark_merged.json`` alongside the per-rank files).
FPM_BENCHMARK_RESULT_GLOB = "benchmark*.json"


def fpm_validate_benchmark_output_path(output_path: str) -> None:
    """Reject benchmark output paths the collector could never discover.

    The in-pod completion gate waits on the configured path itself, but every
    collector discovery site (transfer observation, timing summary, native
    validation) sweeps :data:`FPM_BENCHMARK_RESULT_GLOB` under
    :data:`FPM_RESULTS_DIR`. A path outside that surface renders and passes
    the gate, then burns the whole measurement before failing at collection
    -- so the render must fail closed instead.
    """

    path = PurePosixPath(output_path)
    if not path.is_absolute() or any(part in (".", "..") for part in path.parts):
        raise ValueError(f"benchmark output path must be absolute without '.' or '..' segments: {output_path!r}")
    if PurePosixPath(FPM_RESULTS_DIR) not in path.parents:
        raise ValueError(f"benchmark output path must live under {FPM_RESULTS_DIR}: {output_path!r}")
    if not fnmatch.fnmatch(path.name, FPM_BENCHMARK_RESULT_GLOB):
        raise ValueError(
            f"benchmark output basename must match {FPM_BENCHMARK_RESULT_GLOB!r} "
            f"so collector discovery can find it: {output_path!r}"
        )


def fpm_benchmark_result_name(output_path: str, dp_rank: int) -> str:
    """Return the engine's result path for one data-parallel rank.

    Mirrors Dynamo's native self-benchmark naming primitive (``Path.stem`` /
    ``Path.suffix``): rank 0 writes the configured output path unchanged;
    every other rank suffixes the file's stem with ``_dp<rank>`` before the
    suffix. Using the same primitive keeps edge inputs (hidden files,
    multi-dot names) byte-identical to the engine.
    """

    if not isinstance(dp_rank, int) or isinstance(dp_rank, bool) or dp_rank < 0:
        raise ValueError("dp_rank must be a non-negative integer")
    if dp_rank == 0:
        return output_path
    path = PurePosixPath(output_path)
    return str(path.with_name(f"{path.stem}_dp{dp_rank}{path.suffix}"))


def fpm_expected_result_paths(
    output_path: str,
    node_rank: int,
    local_data_parallel_size: int,
) -> list[str]:
    """Return the result paths one node's gate waits for.

    Node ``k`` hosts data-parallel ranks ``[k * local, (k + 1) * local)``;
    each writes exactly one result file on its own node.
    """

    if (
        not isinstance(node_rank, int)
        or isinstance(node_rank, bool)
        or not isinstance(local_data_parallel_size, int)
        or isinstance(local_data_parallel_size, bool)
        or node_rank < 0
        or local_data_parallel_size < 1
    ):
        raise ValueError("node_rank must be a non-negative integer and local_data_parallel_size a positive integer")
    start = node_rank * local_data_parallel_size
    return [
        fpm_benchmark_result_name(output_path, dp_rank) for dp_rank in range(start, start + local_data_parallel_size)
    ]


def fpm_workload_node_count(workload: dict[str, Any]) -> int:
    """Return the pod count a generated FPM workload document schedules.

    This enumerates the only manifest-internal fields the collector may read:
    ``spec.leaderWorkerTemplate.size`` for a LeaderWorkerSet, and
    ``spec.replicas`` (pinned to 1) times the clique ``spec.replicas`` sum for
    a PodCliqueSet. Everything else inside the manifest is opaque to the
    collector.
    """

    kind = workload.get("kind")
    if kind == "Pod":
        return 1
    if kind == "LeaderWorkerSet":
        return int(workload["spec"]["leaderWorkerTemplate"]["size"])
    if kind == "PodCliqueSet":
        spec = workload.get("spec") or {}
        if not isinstance(spec, dict):
            raise ValueError("FPM PodCliqueSet spec must be a mapping")
        replicas = spec.get("replicas")
        if not isinstance(replicas, int) or isinstance(replicas, bool) or replicas != 1:
            raise ValueError("FPM PodCliqueSet spec.replicas must be 1")
        template = spec.get("template") or {}
        if not isinstance(template, dict):
            raise ValueError("FPM PodCliqueSet spec.template must be a mapping")
        cliques = template.get("cliques") or []
        if not isinstance(cliques, list) or not cliques:
            raise ValueError("FPM PodCliqueSet requires at least one clique")
        total = 0
        for clique in cliques:
            if not isinstance(clique, dict) or not isinstance(clique.get("spec"), dict):
                raise TypeError("FPM PodCliqueSet cliques must be mappings with spec")
            clique_replicas = clique["spec"].get("replicas")
            if not isinstance(clique_replicas, int) or isinstance(clique_replicas, bool) or clique_replicas < 1:
                raise ValueError("FPM PodCliqueSet clique replicas must be positive integers")
            total += clique_replicas
        return total
    raise ValueError(f"unsupported generated FPM workload kind: {kind}")
