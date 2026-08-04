# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SGLang WideEP DeepEP low-latency dispatch/combine collector.

Runs the DeepEP low-latency (decode EP) dispatch/combine micro-benchmark once
across all visible GPUs of a single node (full node: 4 on GB200, 8 on
B200/H200). It sweeps the MoE ``(hidden_size, num_experts, topk)`` shapes used
by the MoE models under ``src/aiconfigurator/model_configs`` against a token
list, and writes ``wideep_deepep_ll_perf.txt`` rows (``node_num=1``) that match
the schema consumed by ``operations.moe.load_wideep_deepep_ll_data``.

Unlike the per-GPU collectors, DeepEP LL is a single collective NCCL/DeepEP job
that must own every GPU at once, so ``collect.py`` invokes
``run_deepep_ll_fullnode`` directly (bypassing the per-GPU worker pool) and then
finalizes the produced CSV as parquet.
"""

import json
import os
import socket
import sys
import traceback
from pathlib import Path

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
COLLECTOR_ROOT = os.path.dirname(os.path.dirname(THIS_DIR))  # collector/
REPO_ROOT = os.path.dirname(COLLECTOR_ROOT)
DEEPEP_DIR = os.path.join(THIS_DIR, "deepep")
MODEL_CONFIGS_DIR = os.path.join(REPO_ROOT, "src", "aiconfigurator", "model_configs")

try:
    from collector.registry_types import PerfFile
except ModuleNotFoundError:  # running with collector/ on sys.path
    if COLLECTOR_ROOT not in sys.path:
        sys.path.append(COLLECTOR_ROOT)
    from registry_types import PerfFile

# Pinned to the WideEP SGLang runtime (collector/framework_manifest.yaml -> 0.5.10).
# DeepEP LL has been collected on >= 0.5.0 runtimes, so allow the family.
__compat__ = ">=0.5.0"

# Token sweep: powers of two up to 1024.
DEEPEP_LL_TOKENS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

# `num_max_dispatch_tokens_per_rank` is the receive-buffer slot capacity, which
# SGLang pins to a constant independent of the batch size
# (SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK, default 128). Sizing it to the
# swept token count instead misrepresents the deployed configuration, and it also
# breaks FP8: at capacities 1/2/4/8 every scale read back for a received token is
# 0.0 while its FP8 payload is intact, so the dequantised value is -0.0 and the
# dispatch correctness check fails. Measured on GB200 across two DeepEP 1.2.1
# builds; the upstream scale-store arithmetic does not explain it, so treat 128 as
# the validated configuration rather than a derived lower bound.
DEEPEP_LL_DISPATCH_CAPACITY = int(os.environ.get("DEEPEP_LL_DISPATCH_CAPACITY", "128"))


def _dispatch_capacity(num_tokens: int) -> int:
    """Slot capacity for a token count, mirroring SGLang's fixed-capacity buffer."""
    return max(DEEPEP_LL_DISPATCH_CAPACITY, num_tokens)


def _ensure_low_latency_qp_depth(max_tokens: int) -> None:
    """Keep DeepEP LL's NVSHMEM queue deep enough for the dispatch capacity.

    DeepEP asserts `nvshmem_qp_depth >= (num_max_dispatch_tokens_per_rank + 1) * 2`.
    """
    min_qp_depth = max(128, 2 * (_dispatch_capacity(max_tokens) + 1))
    current = os.environ.get("NVSHMEM_QP_DEPTH")
    if current is None or int(current) < min_qp_depth:
        os.environ["NVSHMEM_QP_DEPTH"] = str(min_qp_depth)


def _iter_moe_configs():
    """Yield (hidden_size, num_experts, topk) from every MoE model config.

    Handles the field-name variants across DeepSeek / Qwen / Llama / MiniMax /
    GPT-OSS, including VL models that nest the language config under
    ``text_config``.
    """
    if not os.path.isdir(MODEL_CONFIGS_DIR):
        return
    for filename in sorted(os.listdir(MODEL_CONFIGS_DIR)):
        if not filename.endswith("_config.json"):
            continue
        try:
            with open(os.path.join(MODEL_CONFIGS_DIR, filename)) as f:
                cfg = json.load(f)
        except (OSError, ValueError):
            continue
        text_cfg = cfg.get("text_config") or {}
        num_experts = (
            cfg.get("n_routed_experts")
            or cfg.get("num_experts")
            or cfg.get("num_local_experts")
            or text_cfg.get("n_routed_experts")
            or text_cfg.get("num_experts")
            or text_cfg.get("num_local_experts")
        )
        topk = (
            cfg.get("num_experts_per_tok")
            or cfg.get("moe_topk")
            or cfg.get("num_experts_per_token")
            or cfg.get("topk")
            or text_cfg.get("num_experts_per_tok")
        )
        hidden = cfg.get("hidden_size") or text_cfg.get("hidden_size")
        if num_experts and topk and hidden:
            yield int(hidden), int(num_experts), int(topk)


def get_deepep_ll_test_cases():
    """Return unique structural ``(hidden_size, num_experts, topk)`` MoE combos.

    Tokens are swept internally by the benchmark, so each case is one structural
    shape. The public plan intentionally includes every discovered MoE shape.
    Framework kernel limitations are observed as per-case failures at runtime,
    rather than silently changing the plan.
    """
    combos = set(_iter_moe_configs())
    return [
        {"hidden_size": hidden, "num_experts": num_experts, "topk": topk}
        for hidden, num_experts, topk in sorted(combos)
    ]


def _verify_token_coverage(output_path, tokens, cases):
    """Assert every successful shape wrote every requested token point."""
    import csv

    wanted = {int(num_token) for num_token in tokens}
    expected_shapes = {(int(case["hidden_size"]), int(case["num_experts"]), int(case["topk"])) for case in cases}
    seen: dict[tuple[int, int, int], set[int]] = {}
    with open(output_path, newline="") as handle:
        for row in csv.DictReader(handle):
            key = (int(row["hidden_size"]), int(row["num_experts"]), int(row["num_topk"]))
            seen.setdefault(key, set()).add(int(row["num_token"]))

    if not seen:
        raise RuntimeError(f"DeepEP LL collection wrote no data rows to {output_path}")

    short = []
    for hidden, num_experts, num_topk in sorted(expected_shapes):
        shape = (hidden, num_experts, num_topk)
        if shape not in seen:
            short.append(f"hidden={hidden} experts={num_experts} topk={num_topk}: missing all {len(wanted)} tokens")
            continue
        missing = sorted(wanted - seen[shape])
        if missing:
            short.append(f"hidden={hidden} experts={num_experts} topk={num_topk}: missing tokens={missing}")
    if short:
        raise RuntimeError("DeepEP LL collection under-covers the requested token sweep:\n  " + "\n  ".join(short))
    return len(expected_shapes)


def _validate_expert_partition(hidden: int, num_experts: int, num_topk: int, num_ranks: int) -> None:
    """Fail a queued shape whose experts cannot be partitioned across ranks."""
    if num_experts % num_ranks != 0:
        raise RuntimeError(
            f"DeepEP LL requires num_experts divisible by num_ranks; "
            f"hidden={hidden} experts={num_experts} topk={num_topk} ranks={num_ranks}"
        )


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return int(sock.getsockname()[1])


def _prime_cuda_profiler() -> None:
    """Burn the process's first ``torch.profiler`` session on a throwaway kernel.

    On the WideEP SGLang 0.5.10 runtime (torch 2.9.1 / CUDA 13) the first
    profiler session of a process yields an empty table, so ``bench_kineto``
    trips its "exactly one matching kernel" assertion and loses the first
    benchmarked token of every shape. Priming costs one trivial kernel and
    leaves every measured session untouched.
    """
    import torch

    try:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]):
            torch.ones(1, device="cuda").mul_(2)
        torch.cuda.synchronize()
    except Exception:  # profiling is best-effort priming; never fail collection here
        print(f"[deepep_ll] profiler priming failed (continuing):\n{traceback.format_exc()}", flush=True)


def _ll_worker(local_rank, num_gpus, cases, tokens, output_path, device_name, version):
    """Per-rank entrypoint spawned across all GPUs of the single node."""
    if DEEPEP_DIR not in sys.path:
        sys.path.insert(0, DEEPEP_DIR)

    import deep_ep
    import test_low_latency
    import torch
    import torch.distributed as dist

    from utils import init_dist

    try:
        from helper import log_perf
    except ModuleNotFoundError:
        if COLLECTOR_ROOT not in sys.path:
            sys.path.append(COLLECTOR_ROOT)
        from helper import log_perf

    rank, num_ranks, group = init_dist(local_rank, num_gpus)
    if device_name is None:
        device_name = torch.cuda.get_device_name(local_rank)
    _prime_cuda_profiler()
    written = 0

    def flush(shape_metrics):
        """Append one shape's rows to the perf CSV (rank 0 only).

        Writing per-shape keeps already-collected data on disk even if a later
        shape aborts the run.
        """
        nonlocal written
        if rank != 0 or not shape_metrics:
            return
        item_list = [
            {
                "node_num": 1,
                "hidden_size": m["hidden"],
                "num_token": m["num_tokens"],
                "num_topk": m["num_topk"],
                "num_experts": m["num_experts"],
                "combine_avg_t_us": m["combine_avg_t_us"],
                "combine_bandwidth_gbps": m["combine_bandwidth_gbps"],
                "dispatch_avg_t_us": m["dispatch_avg_t_us"],
                "dispatch_bandwidth_gbps": m["dispatch_bandwidth_gbps"],
            }
            for m in shape_metrics
        ]
        log_perf(
            item_list=item_list,
            framework="sglang",
            version=version,
            device_name=device_name,
            op_name="ll",
            kernel_source="deepep",
            perf_filename=output_path,
        )
        written += len(item_list)
        print(f"[deepep_ll] wrote {len(item_list)} rows (total {written}) to {output_path}", flush=True)

    for case in cases:
        hidden = int(case["hidden_size"])
        num_experts = int(case["num_experts"])
        num_topk = int(case["topk"])
        _validate_expert_partition(hidden, num_experts, num_topk, num_ranks)

        # Buffer allocation failure is a failed case. The parent isolates every
        # shape in its own spawn, so raising records the exact case and leaves the
        # following shape a fresh process group.
        max_capacity = _dispatch_capacity(max(tokens))
        num_rdma_bytes = deep_ep.Buffer.get_low_latency_rdma_size_hint(max_capacity, hidden, num_ranks, num_experts)
        if rank == 0:
            print(
                f"[deepep_ll] hidden={hidden} experts={num_experts} topk={num_topk} "
                f"ranks={num_ranks} buffer={num_rdma_bytes / 1e6:.0f}MB",
                flush=True,
            )
        try:
            buffer = deep_ep.Buffer(
                group,
                num_rdma_bytes=num_rdma_bytes,
                low_latency_mode=True,
                num_qps_per_rank=num_experts // num_ranks,
                allow_nvlink_for_low_latency_mode=True,
                explicitly_destroy=True,
                allow_mnnvl=False,
            )
        except Exception:
            if rank == 0:
                print(
                    f"[deepep_ll] FAILED (buffer alloc) hidden={hidden} experts={num_experts} "
                    f"topk={num_topk}:\n{traceback.format_exc()}",
                    flush=True,
                )
            raise

        # A failure inside the benchmark leaves the process group in an undefined
        # state (a half-finished collective), so we cannot safely continue: log the
        # full traceback and re-raise. The orchestrator isolates each shape in its
        # own spawn, so this only tears down the current shape. Rows are flushed
        # per token so a mid-shape crash keeps everything collected so far.
        try:
            for num_tokens in tokens:
                token_metrics: list[dict] = []
                capacity = _dispatch_capacity(num_tokens)
                buffer.clean_low_latency_buffer(capacity, hidden, num_experts)
                # FIXME(kernel-limit): DeepEP 1.2.1 low-latency dispatch only
                # instantiates selected hidden sizes (SWITCH_HIDDEN), caps top-k
                # at 11 (internode_ll.cu kNumMaxTopK), and its FP8 cast expects a
                # 128-divisible hidden size. These source-derived claims remain
                # unverified for the next runtime; invoke the framework and record
                # any affected shape as a failure instead of filtering the plan.
                test_low_latency.test_main(
                    num_tokens,
                    hidden,
                    num_experts,
                    num_topk,
                    rank,
                    num_ranks,
                    group,
                    buffer,
                    seed=1,
                    do_check=True,
                    enable_route_masking=False,
                    dispatch_capacity=capacity,
                    metrics_out=token_metrics if rank == 0 else None,
                )
                dist.barrier()
                flush(token_metrics)
        except Exception:
            if rank == 0:
                print(
                    f"[deepep_ll] FAILED hidden={hidden} experts={num_experts} topk={num_topk}:\n"
                    f"{traceback.format_exc()}",
                    flush=True,
                )
            try:
                buffer.destroy()
            except Exception:
                pass
            raise
        buffer.destroy()
        dist.barrier()

    if rank == 0:
        print(f"[deepep_ll] worker done, {written} rows this shape-group in {output_path}", flush=True)

    dist.barrier()
    dist.destroy_process_group()


def run_deepep_ll_fullnode(perf_filename=PerfFile.WIDEEP_DEEPEP_LL, *, device=None, limit=None, cases=None):
    """Run the single-node, full-node DeepEP LL sweep.

    Spawns one process per visible GPU (the full node), sweeps MoE shapes x
    tokens, and writes ``wideep_deepep_ll_perf.txt`` to the current directory so
    ``collect.py`` can finalize it as parquet. ``device`` is accepted for
    registry-signature compatibility and ignored (the job owns all GPUs).
    """
    import torch
    import torch.multiprocessing as mp

    num_gpus = torch.cuda.device_count()
    if num_gpus < 1:
        raise RuntimeError("DeepEP LL collection requires at least one visible CUDA device")

    if cases is None:
        cases = get_deepep_ll_test_cases()
        # Optional single-shape selection (DEEPEP_LL_SHAPE_INDEX) so a SLURM job array
        # can run one shape per fresh node. A hard CUDA fault leaves node-level
        # NVSHMEM/IBGDA state dirty and cascades into later shapes on the same node;
        # isolating one shape per node sidesteps that entirely.
        shape_index = os.environ.get("DEEPEP_LL_SHAPE_INDEX")
        if shape_index is not None and shape_index != "":
            idx = int(shape_index)
            if idx < 0 or idx >= len(cases):
                raise RuntimeError(f"DEEPEP_LL_SHAPE_INDEX={idx} out of range (0..{len(cases) - 1})")
            cases = [cases[idx]]
        elif limit is not None:
            cases = cases[:limit]
    else:
        cases = list(cases)
    if not cases:
        raise RuntimeError("No MoE shapes resolved for DeepEP LL collection")

    # Resolve the device name inside the worker (rank 0) so the parent never
    # creates a CUDA context that would persist across the per-shape spawns.
    device_name = None
    try:
        from collector.wideep.sglang import dataset_version_label
    except ModuleNotFoundError:  # running with collector/ on sys.path
        from wideep.sglang import dataset_version_label

    version = dataset_version_label("DEEPEP_LL_VERSION", "deepep_ll")

    output_path = os.path.join(os.getcwd(), str(perf_filename))
    _ensure_low_latency_qp_depth(max(DEEPEP_LL_TOKENS))

    # Single-node distributed bootstrap (WORLD_SIZE = number of NODES = 1).
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["WORLD_SIZE"] = "1"
    os.environ["RANK"] = "0"

    print(
        f"[deepep_ll] starting full-node collection: {num_gpus} GPUs, {len(cases)} shapes, tokens={DEEPEP_LL_TOKENS}",
        flush=True,
    )

    # Isolate each shape in its own spawn job: a hard CUDA fault (e.g. illegal
    # memory access) in one shape corrupts the context for every rank and cannot
    # be caught in-process, so a single mp.spawn over all shapes would lose every
    # later shape. One spawn per shape contains the blast radius; rows are flushed
    # per token, so completed work always lands on disk.
    successful_cases, failed = [], []
    for idx, case in enumerate(cases, start=1):
        os.environ["MASTER_PORT"] = str(_free_port())
        print(
            f"[deepep_ll] === shape {idx}/{len(cases)}: hidden={case['hidden_size']} "
            f"experts={case['num_experts']} topk={case['topk']} ===",
            flush=True,
        )
        try:
            mp.spawn(
                _ll_worker,
                args=(num_gpus, [case], list(DEEPEP_LL_TOKENS), output_path, device_name, version),
                nprocs=num_gpus,
                join=True,
            )
            _verify_token_coverage(output_path, DEEPEP_LL_TOKENS, [case])
            successful_cases.append(case)
        except Exception:
            failed.append(case)
            print(
                f"[deepep_ll] shape FAILED (isolated, continuing): {case}\n{traceback.format_exc()}",
                flush=True,
            )

    succeeded = len(successful_cases)
    print(f"[deepep_ll] done: {succeeded}/{len(cases)} shapes ok, {len(failed)} failed", flush=True)
    if failed:
        print(f"[deepep_ll] failed shapes: {failed}", flush=True)

    if not Path(output_path).exists() and not failed:
        raise RuntimeError(f"DeepEP LL collection produced no output at {output_path}")

    status = "complete" if not failed else "partial"
    output = output_path if Path(output_path).exists() else "no output rows"
    print(f"[deepep_ll] collection {status}: {output}", flush=True)
    return {"succeeded": succeeded, "failed": failed}
