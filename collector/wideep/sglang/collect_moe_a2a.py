# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standalone multi-node MoE all-to-all (DeepEP) comm collector.

Produces the unified ``moe_a2a_perf`` table consumed by
``aiconfigurator_core.sdk.operations.moe_comm.load_moe_a2a_data``. Two DeepEP
kernel families are measured, both under ``comm_dtype="default"`` (the dtype
key the SDK's DeepEP legs use — moe_comm.py:191, :205):

* ``deepep_ht`` — the "normal"/high-throughput kernels, swept over the
  declared SM budgets, with the per-phase transmit/notify split and the
  NVL/RDMA chunk-size tuning ported from ``deepep/test_internode.py``.
* ``deepep_ll`` — the low-latency decode kernels, ported from
  ``deepep/test_low_latency.py``.

This module REPLACES the manual ``deepep/`` script + ``extract_data.py``
log-scraping pipeline, whose identity columns were module-level constants
(``extract_data.py:14-21``: framework/version/device hardcoded, ``node_num``
parsed out of a log FILENAME). Here every identity column is live: the
distributed world supplies ``ep_size``/``node_num``, ``torch`` supplies the
device name, and the framework version is the installed one, cross-checked
against the ``wideep_sglang`` manifest pin.

It is a STANDALONE collector (``provenance.STANDALONE_COLLECTOR_MODULES``),
not a ``collect.py`` ``OpEntry``: the executor is single-host, this benchmark
is inherently multi-node. Consequently the module also plays the executor's
role — it owns its case plan, its classified failure log, its parquet
finalization and its ``collection_meta.yaml`` sidecar (the
``collect_dsv4_megamoe`` lifecycle).

Launch (see ``collector/network/slurm/submit_moe_a2a.sh``)::

    torchrun --nnodes 2 --nproc-per-node 8 \\
        collector/wideep/sglang/collect_moe_a2a.py --gpus-per-node 8

Emission order (D5): rows are emitted ascending on every non-token key axis
— ``(comm_backend, sms, hidden_size, topk, num_experts, num_tokens)`` for the
case sweep, and ``combine`` before ``dispatch`` within a case — so the
consumer's nested store is built in ascending insertion order at every level
and Rust-sorted vs Python-insertion-order iteration cannot diverge.

Deliberate deviations from the source scripts, beyond the dropped correctness
cross-products documented on each bench function:

* **Per-case reseeding.** ``run_ht_case``/``run_ll_case`` call
  ``torch.manual_seed(rank)`` (and ``random.seed(rank)``) at the top of every
  case; the sources seed once per process
  (``test_internode.py:381``, ``test_low_latency.py:34``). Cases here are
  emitted rows rather than steps of one script run, so each row's synthetic
  routing must be reproducible from its own case identity instead of from how
  many cases happened to run before it. The cost is that consecutive cases
  sharing a shape see the same routing draw.
* **No power column.** See :func:`_power_columns`.
* **Buffer lifetime.** HT and LL DeepEP Buffers are never co-resident; see
  the run loop in :func:`main`.

Known limitation (rank failure inside a case): a rank that fails inside a
DeepEP collective leaves its peers blocked until the NCCL watchdog aborts the
job. The fallback is an aborted run with NO sidecar written — never a false
'complete'. Cross-case all_reduce(MAX) agreement keeps ranks in lockstep
between cases; within-case desync is accepted as watchdog-terminated.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

# Standalone entry point: `python collector/wideep/sglang/collect_moe_a2a.py`
# must be able to import the `collector` package (provenance/helper/case
# generation all live there).
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from collector import provenance
from collector.framework_manifest import get_collector_runtime
from collector.helper import finalize_perf_files, log_perf, stale_output_artifacts
from collector.registry_types import PerfFile
from collector.wideep.distributed_lifecycle import agree_stage, raise_for_stage

MODULE_NAME = "collector.wideep.sglang.collect_moe_a2a"
OP_NAME = "moe_a2a"
TABLE_STEM = Path(PerfFile.MOE_A2A.value).stem

#: Framework identity recorded in the perf rows.
FRAMEWORK = "SGLang"
#: The manifest entry that pins the DeepEP-capable runtime for this collector.
MANIFEST_FRAMEWORK = "sglang"
MANIFEST_WORKLOAD = "wideep"

#: Ground truth for ``kernel_source``: the benchmark invokes deep_ep's own
#: Buffer kernels directly. Declared in collector/kernel_source_backends.yaml
#: (sglang -> deepep).
KERNEL_SOURCE = "deepep"

COMM_BACKEND_HT = "deepep_ht"
COMM_BACKEND_LL = "deepep_ll"
#: The DeepEP legs' dtype key. The legacy DeepEP tables had no dtype axis and
#: the SDK's adapters store them under "default" (moe_comm.py:191, :205); the
#: new-schema rows must key identically or they cannot overwrite/extend those
#: leaves.
COMM_DTYPE = "default"

#: LL rows carry no SM budget. 0 is the value the SDK's legacy LL adapter
#: assigns (moe_comm.py:191) and what ``_normalize_sms`` maps null to.
LL_SMS = 0

PHASE_DISPATCH = "dispatch"
PHASE_COMBINE = "combine"

#: Ported verbatim from ``deepep/test_internode.py:127``.
HT_RDMA_BUFFER_SIZE = 128

#: LL warmup iterations before the profiled region, matching
#: ``deepep/utils.py:bench``'s ``num_warmups`` default.
LL_WARMUP_ITERS = 50

#: Classified failure log. Rank-scoped: every rank records its own view of a
#: failed case, and the output dir is typically shared storage, so a single
#: file would be corrupted by concurrent writers.
ERRORS_FILENAME_TEMPLATE = "errors_moe_a2a.rank{rank}.json"


class MoeA2ADeclarationError(RuntimeError):
    """A declared input (shape, runtime pin, world layout) does not resolve."""


class MoeA2ABenchmarkError(RuntimeError):
    """A queued case failed to execute. Classified failure record, never a skip."""


class MoeA2APeerError(MoeA2ABenchmarkError):
    """Another distributed rank failed the same lifecycle stage."""


# ---------------------------------------------------------------------------
# Distributed identity — from the environment, never from filenames
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DistIdentity:
    """The world this process runs in. ``ep_size == world_size``."""

    rank: int
    world_size: int
    local_rank: int
    gpus_per_node: int
    node_num: int
    master_addr: str
    master_port: str

    @property
    def ep_size(self) -> int:
        return self.world_size


def derive_dist_identity(
    env: dict[str, str],
    *,
    gpus_per_node: int,
    visible_device_count: int | None = None,
) -> DistIdentity:
    """Derive the distributed identity from launcher environment variables.

    ``gpus_per_node`` is an EXPLICIT argument (never inferred from a filename,
    a system name table, or a stale default): it is a property of the machine
    the job landed on and is the divisor that turns ``WORLD_SIZE`` into
    ``node_num``, a persisted key column. When ``visible_device_count`` is
    known (a live ``torch.cuda.device_count()``) it is cross-checked against
    the declared value, because a mismatch silently mislabels every row's
    ``node_num``.

    Raises:
        MoeA2ADeclarationError: on a non-positive ``gpus_per_node``, a
            device-count mismatch, or a ``WORLD_SIZE`` that is not an integral
            number of nodes.
    """
    if gpus_per_node <= 0:
        raise MoeA2ADeclarationError(f"--gpus-per-node must be positive, got {gpus_per_node}")

    rank = int(env.get("RANK", env.get("SLURM_PROCID", "0")))
    world_size = int(env.get("WORLD_SIZE", env.get("SLURM_NTASKS", "1")))
    if "LOCAL_RANK" in env:
        local_rank = int(env["LOCAL_RANK"])
    elif "SLURM_LOCALID" in env:
        local_rank = int(env["SLURM_LOCALID"])
    else:
        local_rank = rank % gpus_per_node

    if visible_device_count is not None and visible_device_count != gpus_per_node:
        raise MoeA2ADeclarationError(
            f"--gpus-per-node={gpus_per_node} does not match the {visible_device_count} CUDA device(s) "
            f"visible to rank {rank} on {socket.gethostname()}. node_num is derived from this divisor "
            "and is a persisted key column, so a mismatch would mislabel every collected row."
        )
    if world_size % gpus_per_node != 0:
        raise MoeA2ADeclarationError(
            f"WORLD_SIZE={world_size} is not an integral number of nodes at "
            f"--gpus-per-node={gpus_per_node}; moe_a2a rows persist node_num = WORLD_SIZE // gpus_per_node."
        )

    return DistIdentity(
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        gpus_per_node=gpus_per_node,
        node_num=world_size // gpus_per_node,
        master_addr=env.get("MASTER_ADDR", "127.0.0.1"),
        master_port=env.get("MASTER_PORT", "29500"),
    )


def init_distributed(identity: DistIdentity):
    """Initialize the NCCL process group for ``identity`` and return its group.

    Mirrors ``deepep/utils.py:init_dist`` (tcp:// init method off
    MASTER_ADDR/MASTER_PORT, bf16 default dtype, cuda default device) but
    takes ranks straight from the launcher rather than reconstructing them
    from a node rank times a spawn count.
    """
    import inspect

    import torch
    import torch.distributed as dist

    torch.cuda.set_device(identity.local_rank)
    if not dist.is_initialized():
        params: dict[str, Any] = {
            "backend": "nccl",
            "init_method": f"tcp://{identity.master_addr}:{identity.master_port}",
            "world_size": identity.world_size,
            "rank": identity.rank,
        }
        if "device_id" in inspect.signature(dist.init_process_group).parameters:
            params["device_id"] = torch.device(f"cuda:{identity.local_rank}")
        dist.init_process_group(**params)
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda")
    return dist.new_group(list(range(identity.world_size)))


# ---------------------------------------------------------------------------
# Case plan — declared shapes x declared workload grid
# ---------------------------------------------------------------------------


@dataclass(frozen=True, order=True)
class MoeA2ARoutingSpec:
    """Serving routing declaration attached to one physical MoE shape."""

    method_type: str = "Renormalize"
    scoring_func: str = "softmax"
    routed_scaling_factor: float | None = None
    renormalize: bool = True
    has_correction_bias: bool = False
    num_expert_group: int = 1
    topk_group: int = 1


@dataclass(frozen=True, order=True)
class MoeA2AShape:
    """One correlated MoE geometry: the tuple is declared together, never crossed."""

    hidden_size: int
    topk: int
    num_experts: int
    routing: MoeA2ARoutingSpec = MoeA2ARoutingSpec()


def _routing_spec_from_recipe(recipe: Any) -> MoeA2ARoutingSpec:
    num_groups = recipe.sglang_moe_num_expert_group
    topk_groups = recipe.sglang_moe_topk_group
    if (num_groups is None) != (topk_groups is None):
        raise MoeA2ADeclarationError(f"{recipe.model_name}: num_expert_group and topk_group must be declared together")
    num_groups = 1 if num_groups is None else int(num_groups)
    topk_groups = 1 if topk_groups is None else int(topk_groups)
    if num_groups <= 0 or topk_groups <= 0 or topk_groups > num_groups:
        raise MoeA2ADeclarationError(f"{recipe.model_name}: invalid routing groups {topk_groups}/{num_groups}")
    if int(recipe.num_experts) % num_groups:
        raise MoeA2ADeclarationError(
            f"{recipe.model_name}: num_experts={recipe.num_experts} is not divisible by num_expert_group={num_groups}"
        )
    selected_experts = (int(recipe.num_experts) // num_groups) * topk_groups
    if int(recipe.topk) > selected_experts:
        raise MoeA2ADeclarationError(
            f"{recipe.model_name}: topk={recipe.topk} exceeds {selected_experts} experts "
            "available in the selected routing groups"
        )
    scoring_func = str(recipe.sglang_moe_scoring_func)
    if scoring_func not in {"softmax", "sigmoid", "sqrtsoftplus"}:
        raise MoeA2ADeclarationError(f"{recipe.model_name}: unsupported scoring_func={scoring_func!r}")
    if bool(recipe.sglang_moe_has_correction_bias) and int(recipe.num_experts) // num_groups < 2:
        raise MoeA2ADeclarationError(f"{recipe.model_name}: correction-bias routing requires two experts per group")
    return MoeA2ARoutingSpec(
        method_type=str(recipe.sglang_moe_routing_method_type or "Renormalize"),
        scoring_func=scoring_func,
        routed_scaling_factor=recipe.sglang_moe_routed_scaling_factor,
        renormalize=bool(recipe.sglang_moe_renormalize),
        has_correction_bias=bool(recipe.sglang_moe_has_correction_bias),
        num_expert_group=num_groups,
        topk_group=topk_groups,
    )


def resolve_moe_a2a_shapes(recipes: list[Any]) -> list[MoeA2AShape]:
    """Resolve declarations and reject routing collisions hidden by the parquet key."""
    from collector.case_generator import is_wideep_moe_model

    declarations: dict[tuple[int, int, int], tuple[MoeA2AShape, set[str]]] = {}
    for recipe in recipes:
        if not is_wideep_moe_model(recipe.model_name):
            continue
        physical_key = (int(recipe.hidden_size), int(recipe.topk), int(recipe.num_experts))
        shape = MoeA2AShape(*physical_key, routing=_routing_spec_from_recipe(recipe))
        previous = declarations.get(physical_key)
        if previous is not None and previous[0].routing != shape.routing:
            raise MoeA2ADeclarationError(
                "routing collision for persisted shape "
                f"{physical_key}: {sorted(previous[1])} declare {previous[0].routing}, "
                f"but {recipe.model_name!r} declares {shape.routing}"
            )
        if previous is None:
            declarations[physical_key] = (shape, {str(recipe.model_name)})
        else:
            previous[1].add(str(recipe.model_name))
    return sorted(value[0] for value in declarations.values())


@dataclass(frozen=True)
class MoeA2ACase:
    comm_backend: str
    shape: MoeA2AShape
    num_tokens: int
    sms: int

    def sort_key(self) -> tuple:
        """D5: ascending on every non-token key axis, token last.

        ``sms`` sorts above the shape axes so one DeepEP Buffer serves a whole
        SM budget (buffer creation is a collective allocation). Insertion order
        into the consumer's nested store stays ascending at every level either
        way — the store nests hidden/topk/num_experts above sms, and each of
        those children is still first-inserted in ascending order.
        """
        return (
            self.comm_backend,
            self.sms,
            self.shape.hidden_size,
            self.shape.topk,
            self.shape.num_experts,
            self.num_tokens,
        )


def get_moe_a2a_shapes(*, required_expert_parallel_size: int | None = None) -> list[MoeA2AShape]:
    """The declared MoE geometries this comm table is collected for.

    Sourced from ``cases/models/*_cases.yaml`` ``model_case_values.moe`` rows
    that declare ``wideep: true`` — the same declaration home the large-EP
    compute collector uses. The persisted comm key has no model column, so
    rows sharing ``(hidden_size, topk, num_experts)`` are one case; the drop
    counts are logged so a zero-case plan is always explainable.
    """
    from collector.case_generator import get_common_moe_test_cases, is_wideep_moe_model

    recipes = get_common_moe_test_cases(
        backend="sglang",
        required_expert_parallel_size=required_expert_parallel_size,
    )
    ordered = resolve_moe_a2a_shapes(recipes)
    dropped_not_declared = sum(1 for recipe in recipes if not is_wideep_moe_model(recipe.model_name))
    print(
        f"moe_a2a: {len(ordered)} declared shapes from {len(recipes)} moe recipes "
        f"(dropped: {dropped_not_declared} not declared wideep; "
        f"{len(recipes) - dropped_not_declared - len(ordered)} deduplicated onto "
        "(hidden_size, topk, num_experts))",
        flush=True,
    )
    return ordered


def get_moe_a2a_workload_grid() -> dict[str, list[int]]:
    """The declared workload axes from ``cases/base_ops/moe_a2a.yaml``."""
    from collector.case_generator import get_base_common_case_values

    values = get_base_common_case_values("moe_a2a")
    if not values:
        raise MoeA2ADeclarationError(
            "collector/cases/base_ops/moe_a2a.yaml declares no common_case_values.moe_a2a grid"
        )
    grid = {}
    for axis in ("ht_token_counts", "ll_token_counts", "sms"):
        raw = values.get(axis)
        if not isinstance(raw, list) or not raw:
            raise MoeA2ADeclarationError(f"common_case_values.moe_a2a.{axis} must be a non-empty list")
        grid[axis] = [int(item) for item in raw]
    return grid


def build_case_plan(
    *,
    shapes: list[MoeA2AShape],
    grid: dict[str, list[int]],
    ep_size: int,
    node_num: int,
    modes: tuple[str, ...] = (COMM_BACKEND_HT, COMM_BACKEND_LL),
) -> list[MoeA2ACase]:
    """Expand the declared grid against the declared shapes for THIS world.

    Two structural identities decide which (shape, world) pairs exist at all.
    Both are the op's own universal math — declared once here, counted, and
    logged — never a per-model exception:

    * ``num_experts % ep_size == 0`` — DeepEP shards experts across ranks.
    * ``num_experts % node_num == 0`` — HT group-limited routing buckets the
      experts per RDMA node (``scores.view(num_tokens, num_nodes, -1)``).

    Emitted in D5 sort order.
    """
    invalid_shapes = [shape for shape in shapes if shape.num_experts % ep_size != 0]
    if invalid_shapes:
        raise MoeA2ADeclarationError(
            f"declared moe_a2a shapes are not divisible by ep_size={ep_size}: {invalid_shapes}; "
            "request the EP constraint from get_moe_a2a_shapes instead of filtering a generated plan"
        )
    invalid_node_shapes = [shape for shape in shapes if shape.num_experts % node_num != 0]
    if invalid_node_shapes:
        raise MoeA2ADeclarationError(
            f"declared moe_a2a shapes are not divisible by node_num={node_num}: {invalid_node_shapes}"
        )

    cases: list[MoeA2ACase] = []
    for shape in shapes:
        if COMM_BACKEND_HT in modes:
            for sms in grid["sms"]:
                for num_tokens in grid["ht_token_counts"]:
                    cases.append(MoeA2ACase(COMM_BACKEND_HT, shape, num_tokens, sms))
        if COMM_BACKEND_LL in modes:
            for num_tokens in grid["ll_token_counts"]:
                cases.append(MoeA2ACase(COMM_BACKEND_LL, shape, num_tokens, LL_SMS))

    cases.sort(key=MoeA2ACase.sort_key)
    print(
        f"moe_a2a: {len(cases)} cases for ep_size={ep_size} node_num={node_num} "
        f"from {len(shapes)} declared shapes x {len(grid['sms'])} sms x "
        f"{len(grid['ht_token_counts'])} HT tokens + {len(grid['ll_token_counts'])} LL tokens",
        flush=True,
    )
    if not cases:
        raise MoeA2ADeclarationError(
            f"moe_a2a expanded to zero cases for ep_size={ep_size} node_num={node_num}; "
            "collecting nothing is never a clean completion"
        )
    return cases


def case_plan_ids(cases: list[MoeA2ACase], *, ep_size: int, node_num: int) -> list[str]:
    """Stable case identifiers for ``provenance.case_plan_hash``.

    Format mirrors the ``dsv4_megamoe`` standalone precedent
    (``<module>:<entry>:<json>``, sorted keys). ``ep_size``/``node_num`` are
    part of the identity: the same grid collected on a different world is a
    different plan.
    """
    ids = []
    for case in cases:
        payload = {
            "comm_backend": case.comm_backend,
            "ep_size": ep_size,
            "hidden_size": case.shape.hidden_size,
            "node_num": node_num,
            "num_experts": case.shape.num_experts,
            "num_tokens": case.num_tokens,
            "routing": {
                "has_correction_bias": case.shape.routing.has_correction_bias,
                "method_type": case.shape.routing.method_type,
                "num_expert_group": case.shape.routing.num_expert_group,
                "renormalize": case.shape.routing.renormalize,
                "routed_scaling_factor": case.shape.routing.routed_scaling_factor,
                "scoring_func": case.shape.routing.scoring_func,
                "topk_group": case.shape.routing.topk_group,
            },
            "sms": case.sms,
            "topk": case.shape.topk,
        }
        ids.append(f"{MODULE_NAME}:run_case:" + json.dumps(payload, sort_keys=True, separators=(",", ":")))
    return ids


# ---------------------------------------------------------------------------
# Row construction — the frozen consumer contract
# ---------------------------------------------------------------------------


def _build_moe_a2a_row(
    *,
    comm_backend: str,
    phase: str,
    ep_size: int,
    node_num: int,
    shape: MoeA2AShape,
    num_tokens: int,
    sms: int,
    transmit_us: float,
    notify_us: float,
    comm_dtype: str = COMM_DTYPE,
) -> dict:
    """One ``moe_a2a_perf`` row.

    ``comm_dtype`` defaults to the DeepEP legs' ``"default"`` key; the trtllm
    alltoall writer (``collector/network/slurm/collect_trtllm_alltoall.py``)
    shares this builder and passes its per-row dtype key instead.

    UNITS: ``latency`` is MICROSECONDS. ``load_moe_a2a_data`` divides the
    column by 1000 to reach its milliseconds leaves
    (``moe_comm.py:393``: ``latency = float(row["latency"]) / 1000.0  #
    collector records us``), exactly like the legacy DeepEP adapter sums its
    ``*_transmit_us``/``*_notify_us`` columns in microseconds before the same
    division (``moe_comm.py:200-204``).

    ``latency = transmit_us + notify_us`` reproduces that legacy definition:
    the SDK's HT adapter sums ``(<phase>_transmit_us, <phase>_notify_us)``
    per phase. ``transmit_us``/``notify_us`` themselves are informational —
    the loader reads only ``latency`` — but keeping the split makes the new
    table a strict superset of the retired one.
    """
    return {
        "comm_backend": comm_backend,
        "phase": phase,
        "comm_dtype": comm_dtype,
        "ep_size": int(ep_size),
        "node_num": int(node_num),
        "hidden_size": int(shape.hidden_size),
        "topk": int(shape.topk),
        "num_experts": int(shape.num_experts),
        "num_tokens": int(num_tokens),
        "sms": int(sms),
        "transmit_us": float(transmit_us),
        "notify_us": float(notify_us),
        "latency": float(transmit_us) + float(notify_us),
    }


def _power_columns() -> None:
    """D7: this table emits NO power column, and that is a measurement fact.

    ``None`` is the loader's supported absent case
    (``has_power = "power" in rows[0]``) — never a fabricated 0.0, never a
    present-but-null column (which would crash ``float(row.get("power", 0.0))``).

    Why absent rather than sampled: there is no region whose wall-clock power
    average corresponds to a single emitted row's workload. An HT row's
    latency is one winning config extracted from a kineto profile of a
    116-configuration tuning sweep, and an LL row's latency is one phase of a
    combined dispatch+combine round trip. Sampling NVML across either region
    would hand the loader ``energy = power x latency`` computed from two
    different workloads. And because ``helper.log_perf`` writes the header
    from the first row while HT and LL share one file, power cannot be
    emitted for one family only.

    Adding real power here means per-phase, dedicated timed re-runs of the
    winning configuration under the sampler — a change to the measurement
    method that must be designed and validated on hardware, not bolted on
    blind.
    """
    return None


# ---------------------------------------------------------------------------
# Kineto timing — ported from deepep/utils.py
# ---------------------------------------------------------------------------


@contextmanager
def _suppressed_stdio(enabled: bool):
    """Silence the kineto C++ chatter around a profiled region."""
    if not enabled:
        yield
        return
    with open(os.devnull, "w") as devnull:
        stdout_fd, stderr_fd = sys.stdout.fileno(), sys.stderr.fileno()
        saved_out, saved_err = os.dup(stdout_fd), os.dup(stderr_fd)
        old_stdout, old_stderr = sys.stdout, sys.stderr
        try:
            os.dup2(devnull.fileno(), stdout_fd)
            os.dup2(devnull.fileno(), stderr_fd)
            sys.stdout = sys.stderr = devnull
            yield
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr
            os.dup2(saved_out, stdout_fd)
            os.dup2(saved_err, stderr_fd)
            os.close(saved_out)
            os.close(saved_err)


def bench_kineto(
    fn,
    kernel_names: tuple[str, ...],
    *,
    num_tests: int = 30,
    barrier_comm_profiling: bool = False,
    num_kernels_per_period: int = 1,
) -> list[float]:
    """Per-kernel average durations in SECONDS, via the torch CUDA profiler.

    Ported from ``deepep/utils.py:bench_kineto`` (the retired manual pipeline
    measured with exactly this; keeping it identical keeps the new rows
    comparable with the shipped legacy tables). Vendored rather than imported:
    the ``deepep/`` scripts are deprecated by this module and are deleted in
    the collection-validation follow-up (plan D4).
    """
    import torch
    import torch.distributed as dist

    with _suppressed_stdio(True):
        schedule = torch.profiler.schedule(wait=0, warmup=1, active=1, repeat=1)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA], schedule=schedule) as prof:
            for _ in range(2):
                # A large kernel plus a barrier absorbs the unbalanced CPU
                # launch overhead across ranks (deepep/utils.py:187-193).
                if barrier_comm_profiling:
                    lhs = torch.randn((8192, 8192), dtype=torch.float, device="cuda")
                    rhs = torch.randn((8192, 8192), dtype=torch.float, device="cuda")
                    lhs @ rhs
                    dist.all_reduce(torch.ones(1, dtype=torch.float, device="cuda"))
                for _ in range(num_tests):
                    fn()
                torch.cuda.synchronize()
                prof.step()

    prof_lines = prof.key_averages().table(sort_by="cuda_time_total", max_name_column_width=100).split("\n")
    for name in kernel_names:
        if sum(name in line for line in prof_lines) != 1:
            raise MoeA2ABenchmarkError(
                f"kineto profile does not contain exactly one row for kernel {name!r}; "
                "the measured region did not run the expected DeepEP kernel"
            )

    units = {"ms": 1e3, "us": 1e6}
    durations: list[Any] = []
    for name in kernel_names:
        for line in prof_lines:
            if name in line:
                time_str = line.split()[-2]
                for unit, scale in units.items():
                    if unit in time_str:
                        durations.append(float(time_str.replace(unit, "")) / scale)
                        break
                break

    if num_kernels_per_period > 1:
        with tempfile.NamedTemporaryFile(suffix=".json") as tmp:
            prof.export_chrome_trace(tmp.name)
            profile_data = json.loads(Path(tmp.name).read_text())
        for index, kernel_name in enumerate(kernel_names):
            events = [event for event in profile_data["traceEvents"] if f"::{kernel_name}" in event["name"]]
            events = sorted(events, key=lambda event: event["ts"])
            per_event = [event["dur"] / 1e6 for event in events]
            num_patterns = len(per_event) // num_kernels_per_period
            durations[index] = [
                sum(per_event[offset::num_kernels_per_period]) / num_patterns
                for offset in range(num_kernels_per_period)
            ]
    return durations


# ---------------------------------------------------------------------------
# HT (normal) benchmark — ported from deepep/test_internode.py
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PhaseTiming:
    """Transmit/notify split for one comm phase, in microseconds."""

    transmit_us: float
    notify_us: float


def _ht_nvl_buffer_size(num_ranks: int) -> int:
    """``deepep/test_internode.py:127`` verbatim."""
    return 720 if num_ranks in (144, 160) else 512


def create_ht_buffer(group, num_sms: int):
    """One HT DeepEP Buffer per SM budget (``test_internode.py:372-379``)."""
    import deep_ep

    return deep_ep.Buffer(
        group,
        int(2e9),
        int(1e9),
        low_latency_mode=False,
        num_qps_per_rank=num_sms,
        explicitly_destroy=True,
    )


def run_ht_case(
    *,
    buffer,
    group,
    case: MoeA2ACase,
    identity: DistIdentity,
) -> dict[str, PhaseTiming]:
    """Measure one HT dispatch+combine case; returns per-phase timings in us.

    Ported from ``deepep/test_internode.py:test_main``. **Kept**: the routing
    construction (the model-declared grouped/global routing identity), the
    NVL/RDMA buffer sizes, the FP8 dispatch tuning sweep
    (nvl 4..44 step 4 x rdma 4..32 step 4), the rank-0 config broadcast, the
    combine tuning sweep (nvl 1..7 x rdma 8/12..32 step 4), and the
    ``("dispatch"|"combine", "notify")`` kineto kernel split.

    **Dropped**: the correctness cross-product (previous_event x async x
    {bf16, ones, fp8} x with_topk, plus ``calc_diff`` tolerance checks) —
    that is a DeepEP unit test, not a measurement, and it dominated the
    runtime. One structural check survives as a classified raise: the
    dispatch must deliver the globally expected token count. Also dropped:
    the manual re-derivation of the dispatch layout — ``get_dispatch_layout``
    is the framework's own path and is used directly.
    """
    import deep_ep
    import torch
    import torch.distributed as dist

    shape = case.shape
    num_ranks = identity.world_size
    num_nodes = identity.node_num
    routing = shape.routing
    nvl_buffer_size = _ht_nvl_buffer_size(num_ranks)

    torch.manual_seed(identity.rank)
    x = torch.ones((case.num_tokens, shape.hidden_size), dtype=torch.bfloat16, device="cuda") * identity.rank
    x_e4m3 = _per_token_cast_to_fp8(x)
    x_e4m3 = (x_e4m3[0], x_e4m3[1].T.contiguous().T)

    logits = torch.randn((case.num_tokens, shape.num_experts), dtype=torch.float32, device="cuda")
    if routing.scoring_func == "sigmoid":
        scores = torch.sigmoid(logits)
    elif routing.scoring_func == "sqrtsoftplus":
        scores = torch.nn.functional.softplus(logits).sqrt()
    else:
        scores = torch.softmax(logits, dim=-1)
    selection_scores = scores
    if routing.has_correction_bias:
        selection_scores = scores + torch.randn((shape.num_experts,), dtype=torch.float32, device="cuda")
    if routing.num_expert_group > 1:
        grouped = selection_scores.view(case.num_tokens, routing.num_expert_group, -1)
        if routing.has_correction_bias:
            group_scores = grouped.topk(2, dim=-1).values.sum(dim=-1)
        else:
            group_scores = grouped.amax(dim=-1)
        group_idx = torch.topk(group_scores, k=routing.topk_group, dim=-1, sorted=False).indices
        selection_scores = _create_grouped_scores(selection_scores, group_idx, routing.num_expert_group)
    topk_idx = torch.topk(selection_scores, shape.topk, dim=-1, largest=True, sorted=False)[1]
    topk_weights = torch.ones((case.num_tokens, shape.topk), dtype=torch.float32, device="cuda") * identity.rank

    (
        num_tokens_per_rank,
        num_tokens_per_rdma_rank,
        num_tokens_per_expert,
        is_token_in_rank,
        _,
    ) = buffer.get_dispatch_layout(topk_idx, shape.num_experts)

    gbl_num_tokens_per_rank = num_tokens_per_rank.clone()
    dist.all_reduce(gbl_num_tokens_per_rank, group=group)

    # Settle the ranks before anything is measured: the layout work above is
    # not uniform across ranks, and residual launch skew shows up inside the
    # first profiled dispatches (deepep/test_internode.py:123-124).
    group.barrier()
    time.sleep(1)

    base_config = deep_ep.Config(case.sms, 8, nvl_buffer_size, 16, HT_RDMA_BUFFER_SIZE)
    layout_args = {
        "num_tokens_per_rank": num_tokens_per_rank,
        "num_tokens_per_rdma_rank": num_tokens_per_rdma_rank,
        "is_token_in_rank": is_token_in_rank,
        "num_tokens_per_expert": num_tokens_per_expert,
    }
    recv_x, _, _, _, handle, _ = buffer.dispatch(
        x=x, topk_idx=topk_idx, topk_weights=topk_weights, config=base_config, **layout_args
    )
    if gbl_num_tokens_per_rank[identity.rank].item() != recv_x.size(0):
        raise MoeA2ABenchmarkError(
            f"moe_a2a HT dispatch delivered {recv_x.size(0)} tokens, expected "
            f"{gbl_num_tokens_per_rank[identity.rank].item()} (case={case})"
        )

    # --- dispatch tuning (FP8 payload, cached handle) -------------------
    best_transmit, best_notify, best_config_values = 1e10, None, None
    for nvl_chunk_size in range(4, 45, 4):
        for rdma_chunk_size in range(4, 33, 4):
            config = deep_ep.Config(case.sms, nvl_chunk_size, nvl_buffer_size, rdma_chunk_size, HT_RDMA_BUFFER_SIZE)
            tune_args = {"x": x_e4m3, "handle": handle, "config": config}
            transmit, notify = bench_kineto(lambda args=tune_args: buffer.dispatch(**args), ("dispatch", "notify"))
            if transmit < best_transmit:
                best_transmit, best_notify = transmit, notify
                best_config_values = (nvl_chunk_size, rdma_chunk_size)

    # Every rank uses rank 0's winning config so the measured configuration
    # is one global fact, not 8..N divergent local ones
    # (test_internode.py:291-302).
    chosen = torch.tensor(list(best_config_values), dtype=torch.int32, device="cuda")
    gathered = [torch.zeros_like(chosen) for _ in range(num_ranks)]
    dist.all_gather(gathered, chosen, group=group)
    nvl_chunk_size, rdma_chunk_size = gathered[0].tolist()
    dispatch_timing = PhaseTiming(best_transmit * 1e6, best_notify * 1e6)

    dispatch_config = deep_ep.Config(case.sms, nvl_chunk_size, nvl_buffer_size, rdma_chunk_size, HT_RDMA_BUFFER_SIZE)
    recv_x, _, _, _, handle, _ = buffer.dispatch(x=x, config=dispatch_config, **layout_args)

    # --- combine tuning --------------------------------------------------
    best_transmit, best_notify = 1e10, None
    for nvl_chunk_size in range(1, 8, 1):
        for rdma_chunk_size in range(12 if num_nodes == 2 else 8, 33, 4):
            config = deep_ep.Config(case.sms, nvl_chunk_size, nvl_buffer_size, rdma_chunk_size, HT_RDMA_BUFFER_SIZE)
            tune_args = {"x": recv_x, "handle": handle, "config": config}
            transmit, notify = bench_kineto(lambda args=tune_args: buffer.combine(**args), ("combine", "notify"))
            if transmit < best_transmit:
                best_transmit, best_notify = transmit, notify

    return {
        PHASE_DISPATCH: dispatch_timing,
        PHASE_COMBINE: PhaseTiming(best_transmit * 1e6, best_notify * 1e6),
    }


def _per_token_cast_to_fp8(x):
    """``deepep/utils.py:per_token_cast_to_fp8`` verbatim."""
    import torch

    if x.dim() != 2 or x.size(1) % 128 != 0:
        raise MoeA2ABenchmarkError(f"fp8 cast needs a 2D tensor with hidden % 128 == 0, got {tuple(x.shape)}")
    m, n = x.shape
    x_view = x.view(m, -1, 128)
    x_amax = x_view.abs().float().amax(dim=2).view(m, -1).clamp(1e-4)
    return (x_view * (448.0 / x_amax.unsqueeze(2))).to(torch.float8_e4m3fn).view(m, n), (x_amax / 448.0).view(m, -1)


def _create_grouped_scores(scores, group_idx, num_groups: int):
    """``deepep/utils.py:create_grouped_scores`` verbatim."""
    import torch

    num_tokens, num_experts = scores.shape
    scores = scores.view(num_tokens, num_groups, -1)
    mask = torch.zeros((num_tokens, num_groups), dtype=torch.bool, device=scores.device)
    mask = mask.scatter_(1, group_idx, True).unsqueeze(-1).expand_as(scores)
    return (scores * mask).view(num_tokens, num_experts)


# ---------------------------------------------------------------------------
# LL (low-latency) benchmark — ported from deepep/test_low_latency.py
# ---------------------------------------------------------------------------


def create_ll_buffer(
    group, *, identity: DistIdentity, cases: list[MoeA2ACase], allow_mnnvl: bool, disable_nvlink: bool
):
    """One LL Buffer sized for the largest LL case in the plan.

    ``get_low_latency_rdma_size_hint`` scales with (tokens, hidden, ranks,
    experts), so the buffer is sized by the max hint across every LL case;
    ``clean_low_latency_buffer`` re-scopes it per case
    (``test_internode.py:402-418``).
    """
    import deep_ep

    hints = [
        deep_ep.Buffer.get_low_latency_rdma_size_hint(
            case.num_tokens, case.shape.hidden_size, identity.world_size, case.shape.num_experts
        )
        for case in cases
    ]
    max_experts = max(case.shape.num_experts for case in cases)
    return deep_ep.Buffer(
        group,
        num_rdma_bytes=max(hints),
        low_latency_mode=True,
        num_qps_per_rank=max(24, max_experts // identity.world_size),
        allow_nvlink_for_low_latency_mode=not disable_nvlink,
        explicitly_destroy=True,
        allow_mnnvl=allow_mnnvl,
    )


def run_ll_case(*, buffer, group, case: MoeA2ACase, identity: DistIdentity) -> dict[str, PhaseTiming]:
    """Measure one LL dispatch+combine case; returns per-phase timings in us.

    Ported from ``deepep/test_low_latency.py:test_main``. **Kept**: the
    synthetic routing (random scores, top-k, a few masked positions), the FP8
    dispatch + combine round trip, and the separate-profiling call that yields
    per-kernel dispatch/combine averages with ``return_recv_hook=False``.

    **Dropped**: the correctness cross-product over
    {x variants} x return_recv_hook x fp8 x round_scale x ue8m0 x zero_copy,
    the per-expert layout assertions, the LogFMT variants and the hash-based
    pressure test.

    ``notify_us`` is 0.0 for LL: the low-latency kernels expose no separate
    notify kernel to the profiler, exactly as the legacy LL table carried only
    per-phase averages with no transmit/notify split (moe_comm.py:232-234).
    ``latency`` therefore equals the measured per-phase average — the same
    quantity the SDK's legacy LL adapter stored.
    """
    import random

    import torch

    shape = case.shape
    torch.manual_seed(identity.rank)
    random.seed(identity.rank)

    buffer.clean_low_latency_buffer(case.num_tokens, shape.hidden_size, shape.num_experts)

    num_local_experts = shape.num_experts // identity.world_size
    x = torch.randn((case.num_tokens, shape.hidden_size), dtype=torch.bfloat16, device="cuda") * 0.1
    routing = shape.routing
    logits = torch.randn((case.num_tokens, shape.num_experts), dtype=torch.float32, device="cuda")
    if routing.scoring_func == "sigmoid":
        scores = torch.sigmoid(logits)
    elif routing.scoring_func == "sqrtsoftplus":
        scores = torch.nn.functional.softplus(logits).sqrt()
    else:
        scores = torch.softmax(logits, dim=-1)
    selection_scores = scores
    if routing.has_correction_bias:
        selection_scores = scores + torch.randn((shape.num_experts,), dtype=torch.float32, device="cuda")
    if routing.num_expert_group > 1:
        grouped = selection_scores.view(case.num_tokens, routing.num_expert_group, -1)
        if routing.has_correction_bias:
            group_scores = grouped.topk(2, dim=-1).values.sum(dim=-1)
        else:
            group_scores = grouped.amax(dim=-1)
        group_idx = torch.topk(group_scores, k=routing.topk_group, dim=-1, sorted=False).indices
        selection_scores = _create_grouped_scores(selection_scores, group_idx, routing.num_expert_group)
    topk_idx = torch.topk(selection_scores, shape.topk, dim=-1, largest=True, sorted=True)[1]
    topk_weights = torch.randn((case.num_tokens, shape.topk), dtype=torch.float32, device="cuda").abs()
    for _ in range(10):
        topk_idx[random.randint(0, case.num_tokens - 1), random.randint(0, shape.topk - 1)] = -1

    # One priming dispatch supplies the GEMM-output stand-in that the timed
    # round trip combines; the timed region re-dispatches and owns its own
    # handle (test_low_latency.py:195-215).
    recv_stats = torch.zeros((num_local_experts,), dtype=torch.int, device="cuda")
    packed_recv_x, _, _, event, _ = buffer.low_latency_dispatch(
        x,
        topk_idx,
        case.num_tokens,
        shape.num_experts,
        use_fp8=True,
        cumulative_local_expert_recv_stats=recv_stats,
        async_finish=True,
        return_recv_hook=False,
    )
    event.current_stream_wait()
    simulated_gemm_x = _per_token_cast_back(
        packed_recv_x[0].view(-1, shape.hidden_size),
        packed_recv_x[1].contiguous().view(-1, shape.hidden_size // 128),
    ).view(packed_recv_x[0].shape)

    def round_trip():
        _, _, inner_handle, inner_event, _ = buffer.low_latency_dispatch(
            x,
            topk_idx,
            case.num_tokens,
            shape.num_experts,
            cumulative_local_expert_recv_stats=recv_stats,
            use_fp8=True,
            async_finish=False,
            return_recv_hook=False,
        )
        buffer.low_latency_combine(simulated_gemm_x, topk_idx, topk_weights, inner_handle, return_recv_hook=False)

    # Warm up before profiling. test_low_latency.py reaches its reported
    # measurement only after utils.bench's 50 warmup + 50 timed iterations
    # (:227) and a full return_recv_hook=True bench_kineto pass (:236); the
    # port must be at least as warm as the rows it has to stay comparable
    # with.
    for _ in range(LL_WARMUP_ITERS):
        round_trip()
    torch.cuda.synchronize()

    group.barrier()
    dispatch_t, combine_t = bench_kineto(
        round_trip,
        ("dispatch", "combine"),
        barrier_comm_profiling=True,
        num_kernels_per_period=1,
    )
    return {
        PHASE_DISPATCH: PhaseTiming(dispatch_t * 1e6, 0.0),
        PHASE_COMBINE: PhaseTiming(combine_t * 1e6, 0.0),
    }


def _per_token_cast_back(x_fp8, x_scales):
    """``deepep/utils.py:per_token_cast_back`` verbatim."""
    import torch

    if x_fp8.numel() == 0:
        return x_fp8.to(torch.bfloat16)
    if x_scales.dtype == torch.int:
        x_scales = x_scales.view(dtype=torch.uint8).to(torch.int) << 23
        x_scales = x_scales.view(dtype=torch.float)
    x_fp32 = x_fp8.to(torch.float32).view(x_fp8.size(0), -1, 128)
    x_scales = x_scales.view(x_fp8.size(0), -1, 1)
    return (x_fp32 * x_scales).view(x_fp8.shape).to(torch.bfloat16)


# ---------------------------------------------------------------------------
# Provenance sidecar
# ---------------------------------------------------------------------------


def _git_collector_ref(repo_root: Path) -> str:
    """The repo SHA the collector ran from (design §5), "unknown" outside a repo.

    Local copy of ``collect.py:_git_collector_ref``: this module runs inside
    the wideep container without importing the single-host executor.
    """
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        print("[moe_a2a] collector_ref: `git rev-parse HEAD` failed; recording 'unknown'", flush=True)
        return "unknown"


def _split_image_digest(image_ref: str) -> tuple[str, str | None]:
    """Split ``repo/image:tag@sha256:<hex>``; mirrors ``collect.py``."""
    image, sep, digest = image_ref.partition("@")
    return image, (digest if sep else None)


def resolve_runtime_meta(installed_version: str, image_ref: str | None) -> dict[str, Any]:
    """The sidecar ``runtime`` block, from the ``wideep_sglang`` manifest pin.

    The INSTALLED version is what actually produced the data, so it is what is
    recorded — but it must equal the pin, or the collected rows would be
    attributed to a runtime that never ran them. The same holds for the image:
    ``image_ref`` is the reference the launcher actually passed to
    ``srun --container-image`` (the manifest holds several variants, e.g. the
    GB200 launcher runs ``grace_blackwell``, not ``default``), so the sidecar
    records the launched variant instead of assuming one — and refuses a ref
    the manifest does not pin at all.
    """
    from packaging.version import InvalidVersion, Version

    runtime = get_collector_runtime(MANIFEST_FRAMEWORK, workload=MANIFEST_WORKLOAD)
    try:
        installed_public = Version(installed_version).public
    except InvalidVersion as error:
        raise MoeA2ADeclarationError(f"invalid installed sglang version {installed_version!r}") from error
    if installed_public != Version(runtime.version).public:
        raise MoeA2ADeclarationError(
            f"moe_a2a collection requires sglang {runtime.version} (manifest wideep_sglang pin), "
            f"found {installed_version}; use {runtime.image()}"
        )
    if not image_ref:
        raise MoeA2ADeclarationError(
            "moe_a2a collection requires --image-ref with the container image the job was launched "
            'with (the launchers pass --image-ref "${CONTAINER_IMAGE}"); runtime provenance must '
            "attest the image that actually ran, not a manifest default"
        )
    variant = next((name for name, ref in sorted(runtime.images.items()) if ref == image_ref), None)
    if variant is None:
        raise MoeA2ADeclarationError(
            f"moe_a2a was launched with image {image_ref!r}, which is not a manifest wideep_sglang "
            f"image variant ({runtime.images}); rows from an unpinned image are not publishable"
        )
    image, image_digest = _split_image_digest(image_ref)
    meta = {"framework": runtime.framework, "version": installed_version, "image": image, "image_variant": variant}
    if image_digest:
        meta["image_digest"] = image_digest
    return meta


def write_moe_a2a_sidecar(
    output_dir: str | Path,
    *,
    runtime_meta: dict[str, Any],
    case_ids: list[str],
    parquet_path: Path,
    failure_count: int,
    repo_root: Path = _REPO_ROOT,
    module_name: str = MODULE_NAME,
) -> Path:
    """Write ``collection_meta.yaml`` for the finalized ``moe_a2a_perf`` table.

    ``status`` follows ``provenance.derive_table_status``: recorded per-case
    failures are observability-only (complete stays complete); only a
    ModuleCollectionFailure demotes, and a module-level death never reaches
    this writer because dying runs do not finalize a sidecar. ``module_name`` selects whose hash closure is
    attested — the trtllm alltoall writer produces the same table and shares
    this finalizer.
    """
    import pyarrow.parquet as pq

    if not case_ids:
        raise MoeA2ADeclarationError(
            "refusing to attest a moe_a2a collection with an empty case plan: a case_plan_hash "
            "over zero cases cannot explain the produced parquet"
        )
    closures = provenance.load_closures(repo_root / "collector" / "hash_closures.yaml")
    table_entry = {
        "collector_ref": _git_collector_ref(repo_root),
        "collector_hash": provenance.collector_hash(module_name, repo_root, closures),
        "case_plan_hash": provenance.case_plan_hash(case_ids),
        "collected_at": date.today().isoformat(),
        "rows": pq.read_metadata(parquet_path).num_rows,
        "status": provenance.derive_table_status(unresolved_failed_count=failure_count, had_module_failure=False),
    }
    return provenance.write_collection_meta(output_dir, runtime_meta, {TABLE_STEM: table_entry})


def record_failure(output_dir: Path, case: MoeA2ACase, error: BaseException, identity: DistIdentity) -> None:
    """Append one classified failure record (the standalone ``errors_*.json``)."""
    record = {
        "module": MODULE_NAME,
        "op": OP_NAME,
        "classification": "unexpected",
        "error_type": type(error).__name__,
        "error": str(error),
        "rank": identity.rank,
        "case": {
            "comm_backend": case.comm_backend,
            "ep_size": identity.ep_size,
            "node_num": identity.node_num,
            "hidden_size": case.shape.hidden_size,
            "topk": case.shape.topk,
            "num_experts": case.shape.num_experts,
            "num_tokens": case.num_tokens,
            "sms": case.sms,
        },
    }
    path = output_dir / ERRORS_FILENAME_TEMPLATE.format(rank=identity.rank)
    existing = json.loads(path.read_text()) if path.exists() else []
    existing.append(record)
    path.write_text(json.dumps(existing, indent=2))
    print(
        f"[moe_a2a] FAILED {case.comm_backend} {case.shape} tokens={case.num_tokens} sms={case.sms}: {error}",
        flush=True,
    )


def record_stage_failure(
    output_dir: Path,
    error: BaseException,
    identity: DistIdentity,
    *,
    stage: str,
) -> Path:
    """Append a fatal non-case lifecycle failure to this rank's log."""

    path = output_dir / ERRORS_FILENAME_TEMPLATE.format(rank=identity.rank)
    existing = json.loads(path.read_text()) if path.exists() else []
    existing.append(
        {
            "module": MODULE_NAME,
            "op": OP_NAME,
            "classification": "unexpected",
            "stage": stage,
            "error_type": type(error).__name__,
            "error": str(error),
            "rank": identity.rank,
            "case": None,
        }
    )
    path.write_text(json.dumps(existing, indent=2))
    return path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _emit_case_rows(
    *,
    timings: dict[str, PhaseTiming],
    case: MoeA2ACase,
    identity: DistIdentity,
    perf_path: str,
    version: str,
    device_name: str,
) -> None:
    """Emit the case's two rows, ``combine`` before ``dispatch`` (D5).

    The consumer's store nests ``phase`` directly under ``comm_backend``, so
    alphabetical phase emission keeps that level's insertion order ascending
    too.
    """
    for phase in (PHASE_COMBINE, PHASE_DISPATCH):
        timing = timings[phase]
        if not log_perf(
            item_list=[
                _build_moe_a2a_row(
                    comm_backend=case.comm_backend,
                    phase=phase,
                    ep_size=identity.ep_size,
                    node_num=identity.node_num,
                    shape=case.shape,
                    num_tokens=case.num_tokens,
                    sms=case.sms,
                    transmit_us=timing.transmit_us,
                    notify_us=timing.notify_us,
                )
            ],
            framework=FRAMEWORK,
            version=version,
            device_name=device_name,
            op_name=OP_NAME,
            kernel_source=KERNEL_SOURCE,
            perf_filename=perf_path,
            power_stats=_power_columns(),
        ):
            raise RuntimeError(f"log_perf refused to write {perf_path} for {phase}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect the unified MoE all-to-all (DeepEP HT + LL) comm table across nodes."
    )
    parser.add_argument(
        "--framework",
        choices=["sglang", "vllm"],
        default="sglang",
        help="inference framework whose DeepEP integration is benchmarked (vllm is dormant, see D3)",
    )
    parser.add_argument(
        "--gpus-per-node",
        type=int,
        required=True,
        help="GPUs per node in this allocation; node_num = WORLD_SIZE // gpus-per-node (a persisted key column)",
    )
    parser.add_argument(
        "--modes",
        default=f"{COMM_BACKEND_HT},{COMM_BACKEND_LL}",
        help="comma-separated DeepEP kernel families to collect",
    )
    parser.add_argument("--output-path", default=os.getcwd())
    parser.add_argument(
        "--image-ref",
        default=None,
        help="container image the job was launched with (the launcher's ${CONTAINER_IMAGE}); must be a "
        "manifest wideep_sglang image variant — recorded in the sidecar runtime block",
    )
    parser.add_argument(
        "--allow-mnnvl",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="allow MNNVL for the low-latency buffer (publishable default: enabled)",
    )
    parser.add_argument("--disable-nvlink", action="store_true", help="disable NVLink for the low-latency buffer")
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="expand and print the case plan (needs --world-size when not launched under torchrun), then exit",
    )
    parser.add_argument(
        "--world-size",
        type=int,
        default=None,
        help="override WORLD_SIZE for --plan-only expansion outside a launcher",
    )
    return parser.parse_args(argv)


def resolve_modes(raw: str) -> tuple[str, ...]:
    modes = tuple(item.strip() for item in raw.split(",") if item.strip())
    unknown = sorted(set(modes) - {COMM_BACKEND_HT, COMM_BACKEND_LL})
    if unknown:
        raise MoeA2ADeclarationError(f"unsupported --modes {unknown}; known: [{COMM_BACKEND_HT}, {COMM_BACKEND_LL}]")
    if not modes:
        raise MoeA2ADeclarationError("--modes selected no DeepEP kernel family")
    return modes


def transport_is_default(*, allow_mnnvl: bool, disable_nvlink: bool) -> bool:
    """Return whether LL transport flags match the sole publishable mode."""

    return allow_mnnvl and not disable_nvlink


def require_supported_framework(framework: str) -> None:
    """D3: the vLLM leg is declared but dormant — no pinned image exists yet."""
    if framework == "vllm":
        raise NotImplementedError(
            "moe_a2a collection for vLLM is not implemented: there is no pinned vLLM+DeepEP runtime "
            "(plan D3). Activation is a manifest change — add a `wideep_vllm` entry to "
            "collector/framework_manifest.yaml with a digest-pinned image, then implement this leg "
            "against that exact version. No fake pin, no unverified placeholder against a "
            "nonexistent runtime."
        )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    require_supported_framework(args.framework)
    modes = resolve_modes(args.modes)

    if args.plan_only:
        env = dict(os.environ)
        if args.world_size is not None:
            env["WORLD_SIZE"] = str(args.world_size)
        identity = derive_dist_identity(env, gpus_per_node=args.gpus_per_node)
        cases = build_case_plan(
            shapes=get_moe_a2a_shapes(required_expert_parallel_size=identity.ep_size),
            grid=get_moe_a2a_workload_grid(),
            ep_size=identity.ep_size,
            node_num=identity.node_num,
            modes=modes,
        )
        ids = case_plan_ids(cases, ep_size=identity.ep_size, node_num=identity.node_num)
        print(json.dumps({"cases": len(cases), "case_plan_hash": provenance.case_plan_hash(ids)}, indent=2))
        return

    from importlib.metadata import version as get_version

    import torch
    import torch.distributed as dist

    identity = derive_dist_identity(
        dict(os.environ), gpus_per_node=args.gpus_per_node, visible_device_count=torch.cuda.device_count()
    )
    runtime_meta = resolve_runtime_meta(get_version("sglang"), args.image_ref)
    runtime_meta["transport"] = {
        "allow_mnnvl": args.allow_mnnvl,
        "allow_nvlink": not args.disable_nvlink,
    }
    group = init_distributed(identity)
    print(
        f"[moe_a2a] host={socket.gethostname()} rank={identity.rank}/{identity.world_size} "
        f"local_rank={identity.local_rank} gpus_per_node={identity.gpus_per_node} "
        f"node_num={identity.node_num} ep_size={identity.ep_size}",
        flush=True,
    )

    cases = build_case_plan(
        shapes=get_moe_a2a_shapes(required_expert_parallel_size=identity.ep_size),
        grid=get_moe_a2a_workload_grid(),
        ep_size=identity.ep_size,
        node_num=identity.node_num,
        modes=modes,
    )
    case_ids = case_plan_ids(cases, ep_size=identity.ep_size, node_num=identity.node_num)

    output_dir = Path(args.output_path)
    perf_path = str(output_dir / PerfFile.MOE_A2A.value)
    device_name = torch.cuda.get_device_name(torch.device("cuda", identity.local_rank))

    ht_buffer = None
    ll_buffer = None
    current_sms = None
    failure_count = 0
    ll_cases = [case for case in cases if case.comm_backend == COMM_BACKEND_LL]
    run_failed = False
    try:

        def agree(_stage: str, failed: bool) -> bool:
            failure = torch.tensor([int(failed)], dtype=torch.int32, device="cuda")
            dist.all_reduce(failure, op=dist.ReduceOp.MAX, group=group)
            return bool(failure.item())

        preflight_error: BaseException | None = None
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            stale = stale_output_artifacts(output_dir, PerfFile.MOE_A2A.value)
            if stale:
                raise MoeA2ADeclarationError(
                    f"moe_a2a refuses to run into {output_dir}: it holds artifacts from a previous "
                    f"attempt ({', '.join(stale)}). Use a fresh --output-path."
                )
        except BaseException as error:
            preflight_error = error
        raise_for_stage(
            agree_stage(
                "preflight",
                preflight_error,
                agreement=agree,
                peer_error_type=MoeA2APeerError,
            )
        )

        for case_index, case in enumerate(cases):
            timings = None
            benchmark_error: BaseException | None = None
            try:
                if case.comm_backend == COMM_BACKEND_HT:
                    if case.sms != current_sms:
                        if ht_buffer is not None:
                            ht_buffer.destroy()
                        ht_buffer = create_ht_buffer(group, case.sms)
                        current_sms = case.sms
                    timings = run_ht_case(buffer=ht_buffer, group=group, case=case, identity=identity)
                else:
                    if ll_buffer is None:
                        if ht_buffer is not None:
                            ht_buffer.destroy()
                            ht_buffer = None
                        ll_buffer = create_ll_buffer(
                            group,
                            identity=identity,
                            cases=ll_cases,
                            allow_mnnvl=args.allow_mnnvl,
                            disable_nvlink=args.disable_nvlink,
                        )
                    timings = run_ll_case(buffer=ll_buffer, group=group, case=case, identity=identity)
            except BaseException as error:
                benchmark_error = error

            benchmark_outcome = agree_stage(
                f"case:{case_index}:benchmark",
                benchmark_error,
                agreement=agree,
                peer_error_type=MoeA2APeerError,
            )
            if benchmark_outcome.failed:
                failure_count += 1
                assert benchmark_outcome.error is not None
                record_error: BaseException | None = None
                try:
                    record_failure(output_dir, case, benchmark_outcome.error, identity)
                except BaseException as error:
                    record_error = error
                raise_for_stage(
                    agree_stage(
                        f"case:{case_index}:failure_record",
                        record_error,
                        agreement=agree,
                        peer_error_type=MoeA2APeerError,
                    )
                )
                continue

            row_error: BaseException | None = None
            if identity.rank == 0:
                try:
                    assert timings is not None
                    _emit_case_rows(
                        timings=timings,
                        case=case,
                        identity=identity,
                        perf_path=perf_path,
                        version=runtime_meta["version"],
                        device_name=device_name,
                    )
                except BaseException as error:
                    row_error = error
            raise_for_stage(
                agree_stage(
                    f"case:{case_index}:row_write",
                    row_error,
                    agreement=agree,
                    peer_error_type=MoeA2APeerError,
                )
            )

        buffer_error: BaseException | None = None
        try:
            if ht_buffer is not None:
                ht_buffer.destroy()
                ht_buffer = None
            if ll_buffer is not None:
                ll_buffer.destroy()
                ll_buffer = None
        except BaseException as error:
            buffer_error = error
        raise_for_stage(
            agree_stage(
                "buffer_teardown",
                buffer_error,
                agreement=agree,
                peer_error_type=MoeA2APeerError,
            )
        )

        publishable = COMM_BACKEND_LL not in modes or transport_is_default(
            allow_mnnvl=args.allow_mnnvl,
            disable_nvlink=args.disable_nvlink,
        )
        parquet_path: Path | None = None
        finalize_error: BaseException | None = None
        if publishable and identity.rank == 0:
            try:
                converted = finalize_perf_files([perf_path])
                if len(converted) != 1:
                    raise MoeA2ABenchmarkError(
                        f"moe_a2a produced no finalized parquet; {failure_count}/{len(cases)} cases failed"
                    )
                parquet_path = Path(converted[0])
            except BaseException as error:
                finalize_error = error
        raise_for_stage(
            agree_stage(
                "parquet_finalize",
                finalize_error,
                agreement=agree,
                peer_error_type=MoeA2APeerError,
            )
        )

        meta_path: Path | None = None
        sidecar_error: BaseException | None = None
        if publishable and identity.rank == 0:
            try:
                assert parquet_path is not None
                meta_path = write_moe_a2a_sidecar(
                    output_dir,
                    runtime_meta=runtime_meta,
                    case_ids=case_ids,
                    parquet_path=parquet_path,
                    failure_count=failure_count,
                )
            except BaseException as error:
                sidecar_error = error
        raise_for_stage(
            agree_stage(
                "sidecar_write",
                sidecar_error,
                agreement=agree,
                peer_error_type=MoeA2APeerError,
            )
        )
        raise_for_stage(
            agree_stage(
                "final_ready",
                None,
                agreement=agree,
                peer_error_type=MoeA2APeerError,
            )
        )
        dist.barrier(group=group)

        if identity.rank == 0:
            if publishable:
                print(
                    f"[moe_a2a] wrote {parquet_path} and {meta_path} ({failure_count} classified failures)",
                    flush=True,
                )
            else:
                print(
                    f"[moe_a2a] diagnostic transport staged rows at {perf_path}; "
                    "parquet and collection_meta.yaml will not be finalized",
                    flush=True,
                )
    except BaseException as error:
        run_failed = True
        try:
            record_stage_failure(output_dir, error, identity, stage="fatal_runtime")
        except Exception as record_error:
            print(
                f"[moe_a2a] failed to record fatal failure {error!r}: {record_error}",
                file=sys.stderr,
                flush=True,
            )
        raise
    finally:
        for buffer in (ht_buffer, ll_buffer):
            if buffer is not None:
                try:
                    buffer.destroy()
                except Exception as error:
                    if not run_failed:
                        raise
                    print(f"[moe_a2a] buffer destroy after failure also failed: {error}", file=sys.stderr)
        if dist.is_initialized():
            try:
                dist.destroy_process_group()
            except Exception as error:
                if not run_failed:
                    raise
                print(f"[moe_a2a] process-group destroy after failure also failed: {error}", file=sys.stderr)


if __name__ == "__main__":
    main()
