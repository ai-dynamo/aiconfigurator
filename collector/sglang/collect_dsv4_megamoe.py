# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cross-rank DeepSeek-V4 MegaMoE module collector.

The collection boundary is the MegaMoE routed path:

    prepared hidden_states + prepared topk_ids/topk_weights
      -> SGLang cached symmetric buffer lookup
      -> SGLang/DeepGEMM pre-dispatch into the symmetric buffer
      -> deep_gemm.fp8_fp4_mega_moe

Gate, top-k selection, routing generation, source-rank assignment, validation,
and distributed setup are outside the timed region.  The cold symmetric buffer
allocation/rendezvous path is also outside per-module latency, matching SGLang's
cached-buffer steady state.
"""

from __future__ import annotations

import argparse
import inspect
import os
import socket
from dataclasses import dataclass

import torch
import torch.distributed as dist

try:
    from collector.helper import benchmark_with_power, log_perf
    from collector.registry_types import PerfFile
    from collector.sglang.dsv4_megamoe_workload import (
        SUPPORTED_SOURCE_POLICIES,
        append_fused_shared_experts,
        build_routing_plan,
        parse_distribution,
        parse_int_list,
    )
except ImportError:
    from dsv4_megamoe_workload import (
        SUPPORTED_SOURCE_POLICIES,
        append_fused_shared_experts,
        build_routing_plan,
        parse_distribution,
        parse_int_list,
    )
    from registry_types import PerfFile

    from helper import benchmark_with_power, log_perf


DEFAULT_MODEL_CONFIGS = {
    "dsv4_flash": {
        "model": "deepseek-ai/DeepSeek-V4-Flash",
        "hidden_size": 4096,
        "inter_size": 2048,
        "routed_num_experts": 256,
        "routed_topk": 6,
        "routed_scaling_factor": 1.5,
        "norm_topk_prob": True,
    },
    "dsv4_pro": {
        "model": "deepseek-ai/DeepSeek-V4-Pro",
        "hidden_size": 7168,
        "inter_size": 3072,
        "routed_num_experts": 384,
        "routed_topk": 6,
        "routed_scaling_factor": 2.5,
        "norm_topk_prob": True,
    },
}

DEFAULT_GPUS_PER_NODE = {
    "B200": 8,
    "B200_SXM": 8,
    "GB200": 4,
    "B300": 8,
    "B300_SXM": 8,
    "GB300": 4,
}

DEFAULT_DISTRIBUTIONS = "balanced,power_law_1.01,power_law_1.2,power_law_sampled_1.9"
DEFAULT_PREFILL_TOKENS = "1024,2048,4096,8192,16384,32768"
DEFAULT_DECODE_TOKENS = "1,2,4,8,16,32,64,128,256,512"
DEFAULT_NUM_MAX_TOKENS_PER_RANK = 0
DEFAULT_CAP_POLICY = "case_tokens"
DEFAULT_SAMPLED_POWER_LAW_SEED_COUNT = 10
DEFAULT_MODULE_PERF = PerfFile.DSV4_MEGAMOE_MODULE.value
DEFAULT_MOE_DTYPE = "w4a8_mxfp4_mxfp8"
DEFAULT_KERNEL_DTYPE = "fp8_fp4"
BUFFER_POLICY = "cached_sglang"
_MEGA_MOE_BUFFER_CACHE = {}


@dataclass(frozen=True)
class DistInfo:
    rank: int
    world_size: int
    local_rank: int
    device: torch.device
    gpus_per_node: int
    num_nodes: int


@dataclass(frozen=True)
class MegaMoECase:
    phase: str
    tokens_per_rank: int
    distribution: str
    ep_size: int
    routing_seed: int


@dataclass(frozen=True)
class CaseRunResult:
    row: dict[str, object]
    power_stats: dict[str, object] | None


def _init_process_group_with_device(device: torch.device) -> None:
    kwargs = {"backend": "nccl"}
    sig = inspect.signature(dist.init_process_group)
    if "device_id" in sig.parameters:
        kwargs["device_id"] = device
    dist.init_process_group(**kwargs)


def init_distributed(gpus_per_node: int | None) -> DistInfo:
    rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0")))
    world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", "1")))
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
    elif "SLURM_LOCALID" in os.environ:
        local_rank = int(os.environ["SLURM_LOCALID"])
    else:
        local_rank = rank % max(1, gpus_per_node or torch.cuda.device_count())

    inferred_gpus_per_node = int(
        gpus_per_node
        or os.environ.get("GPUS_PER_NODE", os.environ.get("SLURM_NTASKS_PER_NODE", torch.cuda.device_count()))
    )
    if inferred_gpus_per_node <= 0:
        raise ValueError("gpus_per_node must be positive")

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(local_rank)

    if world_size > 1 and not dist.is_initialized():
        _init_process_group_with_device(device)

    num_nodes = (world_size + inferred_gpus_per_node - 1) // inferred_gpus_per_node
    print(
        f"[dsv4-megamoe] host={socket.gethostname()} rank={rank}/{world_size} "
        f"local_rank={local_rank} device={device} gpus_per_node={inferred_gpus_per_node} "
        f"master={os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}",
        flush=True,
    )
    return DistInfo(rank, world_size, local_rank, device, inferred_gpus_per_node, num_nodes)


def _barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def _all_reduce_max(value: float, device: torch.device) -> float:
    tensor = torch.tensor([value], dtype=torch.float64, device=device)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor.item())


def _all_reduce_min_int(value: int, device: torch.device) -> int:
    tensor = torch.tensor([value], dtype=torch.int32, device=device)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.MIN)
    return int(tensor.item())


def _parse_distributions(value: str) -> list[str]:
    distributions = [item.strip() for item in value.split(",") if item.strip()]
    for distribution in distributions:
        if distribution == "balanced":
            continue
        parse_distribution(distribution)
    return distributions


def _parse_routing_seeds(args: argparse.Namespace) -> list[int]:
    raw = str(getattr(args, "routing_seeds", "") or "").strip()
    seeds = parse_int_list(raw) if raw else [int(args.routing_seed)]
    if not seeds:
        raise ValueError("at least one routing seed is required")
    return seeds


def _routing_seeds_explicit(args: argparse.Namespace) -> bool:
    return bool(str(getattr(args, "routing_seeds", "") or "").strip())


def _seeds_for_distribution(
    distribution: str,
    *,
    args: argparse.Namespace,
    routing_seeds: list[int],
    routing_seeds_explicit: bool,
) -> list[int]:
    spec = parse_distribution(distribution)
    if spec.kind == "sampled_power_law" and not routing_seeds_explicit:
        start_seed = int(args.routing_seed)
        return list(range(start_seed, start_seed + DEFAULT_SAMPLED_POWER_LAW_SEED_COUNT))
    return routing_seeds


def get_cached_mega_moe_buffer(
    *,
    group,
    total_num_experts: int,
    num_max_tokens_per_rank: int,
    total_topk: int,
    hidden_size: int,
    inter_size: int,
):
    """Return the cached DeepGEMM MegaMoE symmetric buffer for this shape.

    This mirrors SGLang's `_get_mega_moe_symm_buffer`: buffer allocation and
    symmetric-memory rendezvous are one-time shape setup, while steady-state
    forward reuses the cached buffer.
    """

    import deep_gemm

    key = (
        id(group),
        num_max_tokens_per_rank,
        total_num_experts,
        total_topk,
        hidden_size,
        inter_size,
    )
    buffer = _MEGA_MOE_BUFFER_CACHE.get(key)
    created = False
    if buffer is None:
        buffer = deep_gemm.get_symm_buffer_for_mega_moe(
            group,
            total_num_experts,
            num_max_tokens_per_rank,
            total_topk,
            hidden_size,
            inter_size,
            use_fp8_dispatch=True,
            activation="swiglu",
        )
        _MEGA_MOE_BUFFER_CACHE[key] = buffer
        created = True
    return buffer, created


def destroy_cached_mega_moe_buffers() -> None:
    for buffer in list(_MEGA_MOE_BUFFER_CACHE.values()):
        buffer.destroy()
    _MEGA_MOE_BUFFER_CACHE.clear()


def build_cases(args: argparse.Namespace, ep_size: int) -> list[MegaMoECase]:
    phases = [item.strip() for item in args.phases.split(",") if item.strip()]
    distributions = _parse_distributions(args.distributions)
    routing_seeds = _parse_routing_seeds(args)
    routing_seeds_explicit = _routing_seeds_explicit(args)
    cases: list[MegaMoECase] = []

    if "context" in phases:
        for tokens in parse_int_list(args.prefill_tokens):
            for distribution in distributions:
                seeds = _seeds_for_distribution(
                    distribution,
                    args=args,
                    routing_seeds=routing_seeds,
                    routing_seeds_explicit=routing_seeds_explicit,
                )
                for routing_seed in seeds:
                    cases.append(MegaMoECase("context", tokens, distribution, ep_size, routing_seed))
    if "generation" in phases:
        for tokens in parse_int_list(args.decode_tokens):
            for distribution in distributions:
                seeds = _seeds_for_distribution(
                    distribution,
                    args=args,
                    routing_seeds=routing_seeds,
                    routing_seeds_explicit=routing_seeds_explicit,
                )
                for routing_seed in seeds:
                    cases.append(MegaMoECase("generation", tokens, distribution, ep_size, routing_seed))

    unknown = sorted(set(phases) - {"context", "generation"})
    if unknown:
        raise ValueError(f"unsupported phases: {unknown}")
    return cases


def _case_log_key(case: MegaMoECase) -> tuple[str, int, str, int]:
    return (case.phase, case.tokens_per_rank, case.distribution, case.ep_size)


def group_cases_for_logging(cases: list[MegaMoECase]) -> list[list[MegaMoECase]]:
    """Group seed variants that should collapse into one perf row."""
    groups: list[list[MegaMoECase]] = []
    group_by_key: dict[tuple[str, int, str, int], list[MegaMoECase]] = {}
    for case in cases:
        key = _case_log_key(case)
        if key not in group_by_key:
            group_by_key[key] = []
            groups.append(group_by_key[key])
        group_by_key[key].append(case)
    return groups


def _mean_power_stats(results: list[CaseRunResult]) -> dict[str, float] | None:
    averaged: dict[str, float] = {}
    for key in ("power", "power_limit"):
        values = []
        for result in results:
            if not result.power_stats:
                continue
            value = result.power_stats.get(key)
            if value in (None, ""):
                continue
            values.append(float(value))
        if values:
            averaged[key] = sum(values) / len(values)
    return averaged or None


def aggregate_case_run_results(results: list[CaseRunResult]) -> CaseRunResult | None:
    """Average seed samples into the single row consumed by AIC."""
    if not results:
        return None
    if len(results) == 1:
        return results[0]

    row = dict(results[0].row)
    latencies = [float(result.row["latency"]) for result in results]
    row["latency"] = f"{sum(latencies) / len(latencies):.6f}"
    return CaseRunResult(row=row, power_stats=_mean_power_stats(results))


def case_num_max_tokens_per_rank(args: argparse.Namespace, case: MegaMoECase) -> int:
    if args.cap_policy == "case_tokens":
        return int(case.tokens_per_rank)
    if args.cap_policy == "fixed":
        return int(args.num_max_tokens_per_rank)
    raise ValueError(f"unsupported cap policy: {args.cap_policy}")


def _cast_grouped_weights_to_fp4(bf16_weights: torch.Tensor):
    import deep_gemm
    from deep_gemm.utils import per_token_cast_to_fp4

    num_groups, n, k = bf16_weights.shape
    weight = torch.empty((num_groups, n, k // 2), device=bf16_weights.device, dtype=torch.int8)
    scale = torch.empty((num_groups, n, k // 32), device=bf16_weights.device, dtype=torch.float32)
    for group_idx in range(num_groups):
        weight[group_idx], scale[group_idx] = per_token_cast_to_fp4(
            bf16_weights[group_idx],
            use_ue8m0=True,
            gran_k=32,
        )
    scale = deep_gemm.transform_sf_into_required_layout(scale, n, k, (1, 32), num_groups)
    return weight, scale


def build_transformed_weights(
    *,
    num_local_experts: int,
    hidden_size: int,
    inter_size: int,
    device: torch.device,
    seed: int,
):
    import deep_gemm

    torch.manual_seed(seed)
    l1_bf16 = torch.randn((num_local_experts, inter_size * 2, hidden_size), dtype=torch.bfloat16, device=device)
    l2_bf16 = torch.randn((num_local_experts, hidden_size, inter_size), dtype=torch.bfloat16, device=device)
    l1_fp4 = _cast_grouped_weights_to_fp4(l1_bf16)
    l2_fp4 = _cast_grouped_weights_to_fp4(l2_bf16)
    return deep_gemm.transform_weights_for_mega_moe(l1_fp4, l2_fp4)


def make_pre_dispatch(pre_dispatch: str):
    if pre_dispatch == "copy":
        from deep_gemm.utils import per_token_cast_to_fp8

        def copy_pre_dispatch(hidden_states, topk_ids, topk_weights, buffer, num_tokens: int):
            x_fp8, x_sf = per_token_cast_to_fp8(
                hidden_states,
                use_ue8m0=True,
                gran_k=32,
                use_packed_ue8m0=True,
            )
            buffer.x[:num_tokens].copy_(x_fp8)
            buffer.x_sf[:num_tokens].copy_(x_sf)
            buffer.topk_idx[:num_tokens].copy_(topk_ids)
            buffer.topk_weights[:num_tokens].copy_(topk_weights)
            if num_tokens < buffer.topk_idx.shape[0]:
                buffer.topk_idx[num_tokens:].fill_(-1)
                buffer.topk_weights[num_tokens:].zero_()

        return copy_pre_dispatch

    if pre_dispatch == "sglang_jit":
        from sglang.jit_kernel.deepseek_v4 import mega_moe_pre_dispatch

        def sglang_jit_pre_dispatch(hidden_states, topk_ids, topk_weights, buffer, num_tokens: int):
            del num_tokens
            mega_moe_pre_dispatch(
                hidden_states,
                topk_ids,
                topk_weights,
                buffer.x,
                buffer.x_sf,
                buffer.topk_idx,
                buffer.topk_weights,
                quant_group_size=32,
            )

        return sglang_jit_pre_dispatch

    raise ValueError(f"unsupported pre_dispatch: {pre_dispatch}")


def run_case(
    *,
    case: MegaMoECase,
    args: argparse.Namespace,
    dist_info: DistInfo,
    model_config: dict[str, int | float | bool | str],
    transformed_weights,
) -> CaseRunResult | None:
    import deep_gemm

    rank = dist_info.rank
    device = dist_info.device
    ep_size = case.ep_size
    hidden_size = int(model_config["hidden_size"])
    inter_size = int(model_config["inter_size"])
    routed_num_experts = int(model_config["routed_num_experts"])
    routed_topk = int(model_config["routed_topk"])
    num_fused_shared_experts = int(args.num_fused_shared_experts)
    total_num_experts = routed_num_experts + num_fused_shared_experts
    total_topk = routed_topk + num_fused_shared_experts
    routed_scaling_factor = float(
        args.routed_scaling_factor if args.routed_scaling_factor is not None else model_config["routed_scaling_factor"]
    )
    norm_topk_prob = bool(
        args.renormalize_topk_weights if args.renormalize_topk_weights is not None else model_config["norm_topk_prob"]
    )
    print(
        f"[dsv4-megamoe] rank={rank} case-start phase={case.phase} "
        f"tokens_per_rank={case.tokens_per_rank} distribution={case.distribution} "
        f"routing_seed={case.routing_seed} "
        f"ep={ep_size} hidden={hidden_size} inter={inter_size} "
        f"num_experts={routed_num_experts} topk={routed_topk} "
        f"cap_policy={args.cap_policy}",
        flush=True,
    )

    if ep_size != dist_info.world_size:
        raise ValueError(f"case ep_size={ep_size} must match WORLD_SIZE={dist_info.world_size}")
    if total_num_experts % ep_size != 0:
        raise ValueError(
            f"total_num_experts={total_num_experts} must be divisible by ep_size={ep_size}; "
            "DSv4 FP4 MegaMoE normally has num_fused_shared_experts=0."
        )

    tokens_per_rank = [case.tokens_per_rank for _ in range(ep_size)]
    plan = build_routing_plan(
        distribution=case.distribution,
        tokens_per_rank=tokens_per_rank,
        routed_num_experts=routed_num_experts,
        routed_topk=routed_topk,
        ep_size=ep_size,
        rank=rank,
        source_policy=args.source_policy,
        routing_seed=case.routing_seed,
        norm_topk_prob=norm_topk_prob,
    )
    print(
        f"[dsv4-megamoe] rank={rank} routing-ready phase={case.phase} "
        f"distribution={case.distribution} routing_seed={case.routing_seed} "
        f"local_topk_shape={tuple(plan.local_topk_ids.shape)}",
        flush=True,
    )

    local_topk_ids, local_topk_weights = append_fused_shared_experts(
        plan.local_topk_ids,
        plan.local_topk_weights,
        routed_num_experts=routed_num_experts,
        num_fused_shared_experts=num_fused_shared_experts,
        routed_scaling_factor=routed_scaling_factor,
    )
    local_topk_ids = local_topk_ids.to(device=device, dtype=torch.int32, non_blocking=True).contiguous()
    local_topk_weights = local_topk_weights.to(device=device, dtype=torch.float32, non_blocking=True).contiguous()
    hidden_states = torch.randn(
        (case.tokens_per_rank, hidden_size),
        dtype=torch.bfloat16,
        device=device,
    )
    num_max_tokens_per_rank = case_num_max_tokens_per_rank(args, case)

    buffer_kwargs = {
        "group": dist.group.WORLD if dist.is_initialized() else None,
        "total_num_experts": total_num_experts,
        "num_max_tokens_per_rank": num_max_tokens_per_rank,
        "total_topk": total_topk,
        "hidden_size": hidden_size,
        "inter_size": inter_size,
    }
    buffer, buffer_created = get_cached_mega_moe_buffer(**buffer_kwargs)
    effective_num_max_tokens_per_rank = int(getattr(buffer, "num_max_tokens_per_rank", num_max_tokens_per_rank))
    pre_dispatch = make_pre_dispatch(args.pre_dispatch)
    output = torch.empty((case.tokens_per_rank, hidden_size), dtype=torch.bfloat16, device=device)
    print(
        f"[dsv4-megamoe] rank={rank} buffer-ready policy={BUFFER_POLICY} "
        f"created={str(buffer_created).lower()} pre_dispatch={args.pre_dispatch} "
        f"num_max_tokens_per_rank={num_max_tokens_per_rank} "
        f"effective_num_max_tokens_per_rank={effective_num_max_tokens_per_rank} "
        "kernel=deep_gemm.fp8_fp4_mega_moe",
        flush=True,
    )

    def timed_megamoe():
        with torch.no_grad():
            buffer, _ = get_cached_mega_moe_buffer(**buffer_kwargs)
            pre_dispatch(hidden_states, local_topk_ids, local_topk_weights, buffer, case.tokens_per_rank)
            deep_gemm.fp8_fp4_mega_moe(
                output,
                transformed_weights[0],
                transformed_weights[1],
                buffer,
                recipe=(1, 1, 32),
                activation="swiglu",
                activation_clamp=args.activation_clamp,
                fast_math=bool(args.fast_math),
            )
            if args.include_routed_scale and routed_scaling_factor != 1.0:
                output.mul_(routed_scaling_factor)

    _barrier()
    print(
        f"[dsv4-megamoe] rank={rank} benchmark-start phase={case.phase} "
        f"distribution={case.distribution} routing_seed={case.routing_seed}",
        flush=True,
    )
    with benchmark_with_power(
        device=device,
        kernel_func=timed_megamoe,
        num_warmups=args.num_warmup,
        num_runs=args.num_iterations,
        repeat_n=1,
        allow_graph_fail=False,
        use_cuda_graph=True,
    ) as bench:
        pass
    used_cuda_graph = bool(bench.get("used_cuda_graph", False))
    if not used_cuda_graph:
        raise RuntimeError("benchmark_with_power did not use CUDA Graph")
    local_latency = float(bench["latency_ms"])
    latency = _all_reduce_max(local_latency, device)
    graph_ok = _all_reduce_min_int(1 if used_cuda_graph else 0, device)
    if graph_ok != 1:
        raise RuntimeError("not all ranks used CUDA Graph")
    _barrier()
    print(
        f"[dsv4-megamoe] rank={rank} benchmark-done local_latency_ms={local_latency:.6f} max_latency_ms={latency:.6f}",
        flush=True,
    )

    if rank != 0:
        return None

    row = {
        "phase": case.phase,
        "moe_dtype": DEFAULT_MOE_DTYPE,
        "kernel_dtype": DEFAULT_KERNEL_DTYPE,
        "num_tokens": case.tokens_per_rank,
        "global_num_tokens": plan.global_num_tokens,
        "hidden_size": hidden_size,
        "inter_size": inter_size,
        "topk": routed_topk,
        "num_experts": routed_num_experts,
        "num_fused_shared_experts": num_fused_shared_experts,
        "moe_tp_size": 1,
        "moe_ep_size": ep_size,
        "distribution": case.distribution,
        "source_policy": args.source_policy,
        "pre_dispatch": args.pre_dispatch,
        "num_max_tokens_per_rank": num_max_tokens_per_rank,
        "effective_num_max_tokens_per_rank": effective_num_max_tokens_per_rank,
        "routed_scaling_factor": routed_scaling_factor,
        "includes_routed_scale": str(bool(args.include_routed_scale)).lower(),
        "includes_gate_topk": "false",
        "buffer_policy": BUFFER_POLICY,
        "includes_buffer_init": "false",
        "used_cuda_graph": "true",
        "latency": f"{latency:.6f}",
    }
    return CaseRunResult(row=row, power_stats=bench.get("power_stats"))


def log_case_run_result(
    *,
    result: CaseRunResult,
    args: argparse.Namespace,
    dist_info: DistInfo,
    sample_count: int,
) -> None:
    log_perf(
        item_list=[result.row],
        framework="SGLang",
        version=args.sglang_version,
        device_name=torch.cuda.get_device_name(dist_info.device),
        op_name="dsv4_megamoe_module",
        kernel_source="deepgemm_megamoe",
        perf_filename=os.path.join(args.output_path, args.perf_file),
        power_stats=result.power_stats,
    )
    print(
        f"[dsv4-megamoe] logged dsv4_megamoe_module samples={sample_count} {result.row}",
        flush=True,
    )


def _default_gpus_per_node(system_name: str) -> int | None:
    return DEFAULT_GPUS_PER_NODE.get(system_name.upper())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect DSv4 DeepGEMM MegaMoE full-module latency.")
    parser.add_argument("--model-config", choices=sorted(DEFAULT_MODEL_CONFIGS), default="dsv4_pro")
    parser.add_argument("--system-name", default=os.environ.get("AIC_SYSTEM_NAME", "gb200"))
    parser.add_argument("--gpus-per-node", type=int, default=None)
    parser.add_argument("--phases", default="context,generation")
    parser.add_argument("--prefill-tokens", default=DEFAULT_PREFILL_TOKENS)
    parser.add_argument("--decode-tokens", default=DEFAULT_DECODE_TOKENS)
    parser.add_argument("--distributions", default=DEFAULT_DISTRIBUTIONS)
    parser.add_argument("--source-policy", choices=SUPPORTED_SOURCE_POLICIES, default="random")
    parser.add_argument("--routing-seed", type=int, default=0)
    parser.add_argument(
        "--routing-seeds",
        default=os.environ.get("ROUTING_SEEDS", ""),
        help=(
            "comma-separated routing seeds to collect and average into one perf row; defaults to ten consecutive "
            "seeds for power_law_sampled_1.9 and --routing-seed for other synthetic distributions"
        ),
    )
    parser.add_argument("--num-fused-shared-experts", type=int, default=0)
    parser.add_argument("--routed-scaling-factor", type=float, default=None)
    parser.add_argument("--include-routed-scale", type=int, choices=[0, 1], default=1)
    parser.add_argument("--renormalize-topk-weights", type=int, choices=[0, 1], default=None)
    parser.add_argument("--num-max-tokens-per-rank", type=int, default=DEFAULT_NUM_MAX_TOKENS_PER_RANK)
    parser.add_argument(
        "--cap-policy",
        choices=["fixed", "case_tokens"],
        default=os.environ.get("CAP_POLICY", DEFAULT_CAP_POLICY),
        help=(
            "fixed uses --num-max-tokens-per-rank for every case; "
            "case_tokens uses each case's local token count as the requested DeepGEMM cap"
        ),
    )
    parser.add_argument("--pre-dispatch", choices=["sglang_jit", "copy"], default="sglang_jit")
    parser.add_argument("--activation-clamp", type=float, default=10.0)
    parser.add_argument("--fast-math", type=int, choices=[0, 1], default=1)
    parser.add_argument("--num-warmup", type=int, default=5)
    parser.add_argument("--num-iterations", type=int, default=20)
    parser.add_argument("--output-path", default=os.getcwd())
    parser.add_argument("--perf-file", default=DEFAULT_MODULE_PERF)
    parser.add_argument("--sglang-version", default=os.environ.get("SGLANG_VERSION", "unknown"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gpus_per_node = args.gpus_per_node or _default_gpus_per_node(args.system_name)
    dist_info = init_distributed(gpus_per_node)
    model_config = DEFAULT_MODEL_CONFIGS[args.model_config]
    ep_size = dist_info.world_size
    cases = build_cases(args, ep_size)

    if ep_size <= 1:
        raise ValueError("DSv4 MegaMoE collection requires EP/world_size > 1")
    if not cases:
        raise ValueError("no DSv4 MegaMoE cases were requested")
    max_case_tokens = max(case.tokens_per_rank for case in cases)
    if args.num_max_tokens_per_rank <= 0:
        args.num_max_tokens_per_rank = max_case_tokens
    if args.cap_policy == "fixed" and args.num_max_tokens_per_rank < max_case_tokens:
        raise ValueError("--num-max-tokens-per-rank must cover the largest local token case")

    total_num_experts = int(model_config["routed_num_experts"]) + int(args.num_fused_shared_experts)
    if total_num_experts % ep_size != 0:
        raise ValueError("total experts must be divisible by EP size")
    num_local_experts = total_num_experts // ep_size

    os.makedirs(args.output_path, exist_ok=True)
    print(
        f"[dsv4-megamoe] rank={dist_info.rank} build-transformed-weights "
        f"num_local_experts={num_local_experts} hidden={model_config['hidden_size']} "
        f"inter={model_config['inter_size']}",
        flush=True,
    )
    transformed_weights = build_transformed_weights(
        num_local_experts=num_local_experts,
        hidden_size=int(model_config["hidden_size"]),
        inter_size=int(model_config["inter_size"]),
        device=dist_info.device,
        seed=1234 + dist_info.rank,
    )
    print(f"[dsv4-megamoe] rank={dist_info.rank} transformed-weights-ready", flush=True)

    try:
        cached_cap: int | None = None
        for case_group in group_cases_for_logging(cases):
            results: list[CaseRunResult] = []
            sample_count = 0
            for case in case_group:
                case_cap = case_num_max_tokens_per_rank(args, case)
                if cached_cap is not None and case_cap != cached_cap:
                    _barrier()
                    destroy_cached_mega_moe_buffers()
                    _barrier()
                cached_cap = case_cap
                sample_count += 1
                result = run_case(
                    case=case,
                    args=args,
                    dist_info=dist_info,
                    model_config=model_config,
                    transformed_weights=transformed_weights,
                )
                if result is not None:
                    results.append(result)
            aggregated = aggregate_case_run_results(results)
            if aggregated is not None:
                log_case_run_result(
                    result=aggregated,
                    args=args,
                    dist_info=dist_info,
                    sample_count=sample_count,
                )
    finally:
        _barrier()
        destroy_cached_mega_moe_buffers()
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
