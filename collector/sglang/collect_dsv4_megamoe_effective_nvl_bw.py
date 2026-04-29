#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Collect DeepGEMM MegaMoE effective NVLink bandwidth.

This intentionally follows DeepGEMM's ``tests/test_mega_moe.py`` accounting:
``num_recv_tokens * hidden * 3 / fused_kernel_time``.  It also reports an AIC
remote-only variant that excludes local source-to-local-owner routes.

For multi-node collection, launch this script on each node with the same
environment convention used by ``collector/deep_collector`` and DeepGEMM:
``MASTER_ADDR``/``MASTER_PORT`` identify the rank-0 node, ``WORLD_SIZE`` is the
number of nodes, ``RANK`` is the node rank, and ``--num-processes`` is the
number of local GPU ranks spawned per node.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import statistics
from pathlib import Path
from typing import Any


def _sample_power_law(
    size: int,
    *,
    alpha: float,
    xmin: float,
    xmax: float,
    generator: Any,
) -> Any:
    """Sample the same inverse-CDF power-law shape used by AIC MoE collectors."""
    import torch

    if xmax <= xmin:
        raise ValueError(f"xmax must be > xmin for power-law sampling, got xmin={xmin}, xmax={xmax}")
    u = torch.rand(size, dtype=torch.float64, device="cpu", generator=generator)
    if abs(alpha - 1.0) < 1e-12:
        return xmin * torch.pow(torch.tensor(xmax / xmin, dtype=torch.float64, device="cpu"), u)
    return ((xmax ** (1 - alpha) - xmin ** (1 - alpha)) * u + xmin ** (1 - alpha)) ** (1 / (1 - alpha))


def _round_robin_adjust_per_rank(
    counts_2d: Any,
    *,
    remaining: int,
    is_valid: Any,
    pick_local_index: Any,
    step: int,
) -> Any:
    import torch

    while remaining > 0:
        progressed = False
        for rank_idx in range(counts_2d.size(0)):
            local_counts = counts_2d[rank_idx]
            valid_local = torch.nonzero(is_valid(local_counts)).flatten()
            if valid_local.numel() == 0:
                continue
            chosen_local_idx = valid_local[pick_local_index(local_counts[valid_local])].item()
            counts_2d[rank_idx, chosen_local_idx] += step
            remaining -= 1
            progressed = True
            if remaining == 0:
                break
        if not progressed:
            break
    return counts_2d


def _assign_experts_from_counts(num_tokens_per_expert: Any, *, num_tokens: int, topk: int) -> Any:
    import torch

    target_sum = num_tokens * topk
    actual_sum = int(num_tokens_per_expert.sum().item())
    if actual_sum != target_sum:
        raise ValueError(f"expert-count sum mismatch: expected {target_sum}, got {actual_sum}")
    if int(num_tokens_per_expert.max().item()) > num_tokens:
        raise ValueError(
            f"expert count {int(num_tokens_per_expert.max().item())} exceeds per-expert upper bound {num_tokens}"
        )

    sorted_experts = torch.argsort(num_tokens_per_expert, descending=True)
    sorted_counts = num_tokens_per_expert[sorted_experts].to(dtype=torch.long)
    expert_ids_flat = torch.repeat_interleave(sorted_experts, sorted_counts)
    h_selected = expert_ids_flat.reshape(topk, num_tokens).t().contiguous()

    sorted_rows = torch.sort(h_selected, dim=1).values
    if bool((sorted_rows[:, 1:] == sorted_rows[:, :-1]).any()):
        raise ValueError("power-law assignment produced duplicate experts within a token")
    return h_selected


def _generate_power_law_assignment(
    *,
    num_tokens: int,
    num_experts: int,
    topk: int,
    ep: int,
    alpha: float,
    generator: Any,
    remap_hot_rank_to_zero: bool,
) -> tuple[Any, Any, dict[str, Any]]:
    """Generate a full-EP power-law expert assignment.

    This follows collector/helper.py::_generate_power_law_distribution's count
    construction, but returns the full assignment instead of filtering rank0.
    """
    import torch

    if num_experts % ep != 0:
        raise ValueError("num_experts must be divisible by ep")
    if topk > num_experts:
        raise ValueError("topk must be <= num_experts")

    if num_tokens * topk > num_experts:
        num_tokens_per_expert = _sample_power_law(
            num_experts,
            alpha=alpha,
            xmin=1,
            xmax=max(1.000001, num_tokens * 0.8),
            generator=generator,
        )
    else:
        num_tokens_per_expert = _sample_power_law(
            num_experts,
            alpha=alpha,
            xmin=0.01,
            xmax=2,
            generator=generator,
        )

    target_sum = num_tokens * topk
    target_distribution = num_tokens_per_expert / num_tokens_per_expert.sum() * target_sum
    num_tokens_per_expert = torch.round(target_distribution).to(torch.int64)

    upper_bound = num_tokens
    overflow = int((num_tokens_per_expert - upper_bound).clamp(min=0).sum().item())
    num_tokens_per_expert = num_tokens_per_expert.clamp(max=upper_bound)
    experts_per_rank = num_experts // ep

    if overflow > 0:
        counts_2d = num_tokens_per_expert.view(ep, experts_per_rank)
        counts_2d = _round_robin_adjust_per_rank(
            counts_2d,
            remaining=overflow,
            is_valid=lambda local_counts: local_counts < upper_bound,
            pick_local_index=torch.argmin,
            step=1,
        )
        num_tokens_per_expert = counts_2d.reshape(-1)

    delta = target_sum - int(num_tokens_per_expert.sum().item())
    if delta != 0:
        counts_2d = num_tokens_per_expert.view(ep, experts_per_rank)
        if delta > 0:
            counts_2d = _round_robin_adjust_per_rank(
                counts_2d,
                remaining=int(delta),
                is_valid=lambda local_counts: local_counts < upper_bound,
                pick_local_index=torch.argmin,
                step=1,
            )
        else:
            counts_2d = _round_robin_adjust_per_rank(
                counts_2d,
                remaining=int(-delta),
                is_valid=lambda local_counts: local_counts > 0,
                pick_local_index=torch.argmax,
                step=-1,
            )
        num_tokens_per_expert = counts_2d.reshape(-1)

    pre_remap_rank_loads = num_tokens_per_expert.view(ep, experts_per_rank).sum(dim=1)
    pre_remap_hot_rank = int(torch.argmax(pre_remap_rank_loads).item())

    if remap_hot_rank_to_zero and pre_remap_hot_rank != 0:
        counts_2d = num_tokens_per_expert.view(ep, experts_per_rank).clone()
        rank0 = counts_2d[0].clone()
        counts_2d[0] = counts_2d[pre_remap_hot_rank].clone()
        counts_2d[pre_remap_hot_rank] = rank0
        num_tokens_per_expert = counts_2d.reshape(-1)

    final_rank_loads = num_tokens_per_expert.view(ep, experts_per_rank).sum(dim=1)
    h_selected_experts = _assign_experts_from_counts(
        num_tokens_per_expert,
        num_tokens=num_tokens,
        topk=topk,
    )

    permutation = torch.randperm(num_tokens, device="cpu", generator=generator)
    h_selected_experts = h_selected_experts[permutation].contiguous()

    metadata = {
        "rank_expert_loads_before_remap": [int(v) for v in pre_remap_rank_loads.tolist()],
        "rank_expert_loads": [int(v) for v in final_rank_loads.tolist()],
        "hot_rank_before_remap": pre_remap_hot_rank,
        "hot_rank_after_remap": int(torch.argmax(final_rank_loads).item()),
        "remap_hot_rank_to_zero": remap_hot_rank_to_zero,
        "max_expert_load": int(num_tokens_per_expert.max().item()),
        "min_expert_load": int(num_tokens_per_expert.min().item()),
    }
    return num_tokens_per_expert, h_selected_experts, metadata


def _make_topk_inputs(
    *,
    args: argparse.Namespace,
    rank_idx: int,
    num_ranks: int,
    num_tokens: int,
    num_experts: int,
    num_topk: int,
    sample_seed: int,
    routing_seed: int,
) -> tuple[Any, Any, dict[str, Any]]:
    import torch

    if args.routing_mode == "random":
        scores = torch.randn((num_tokens, num_experts), dtype=torch.float, device="cuda")
        topk_weights, topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)
        return topk_weights, topk_idx, {"routing_mode": "random", "routing_seed": sample_seed}

    if args.routing_mode != "power-law":
        raise ValueError(f"unknown routing mode: {args.routing_mode}")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(routing_seed)
    global_num_tokens = num_tokens * num_ranks
    _, global_topk_idx, metadata = _generate_power_law_assignment(
        num_tokens=global_num_tokens,
        num_experts=num_experts,
        topk=num_topk,
        ep=num_ranks,
        alpha=args.power_law_alpha,
        generator=generator,
        remap_hot_rank_to_zero=args.power_law_remap_hot_rank_to_zero,
    )
    rank_start = rank_idx * num_tokens
    topk_idx = global_topk_idx[rank_start : rank_start + num_tokens].to(device="cuda", dtype=torch.long)

    weights_generator = torch.Generator(device="cuda")
    weights_generator.manual_seed(sample_seed + 19013)
    topk_weights = torch.randn(
        (num_tokens, num_topk),
        dtype=torch.float,
        device="cuda",
        generator=weights_generator,
    )

    metadata.update(
        {
            "routing_mode": "power-law",
            "routing_seed": routing_seed,
            "power_law_alpha": args.power_law_alpha,
            "global_num_tokens": global_num_tokens,
        }
    )
    return topk_weights, topk_idx, metadata


def _build_route_matrix(topk_by_rank: list[list[list[int]]], *, num_experts: int, num_ranks: int) -> list[list[int]]:
    if num_experts % num_ranks != 0:
        raise ValueError("num_experts must be divisible by num_ranks")
    experts_per_rank = num_experts // num_ranks
    routes = [[0 for _ in range(num_ranks)] for _ in range(num_ranks)]
    for src_rank, rank_rows in enumerate(topk_by_rank):
        for row in rank_rows:
            for expert_id in row:
                expert_id = int(expert_id)
                if expert_id < 0:
                    continue
                dst_rank = expert_id // experts_per_rank
                routes[src_rank][dst_rank] += 1
    return routes


def _summarize_rank(
    *,
    rank_idx: int,
    num_ranks: int,
    routes: list[list[int]],
    num_recv_tokens: int,
    hidden: int,
    num_tokens: int,
    num_topk: int,
    t_fused_s: float,
) -> dict[str, Any]:
    recv_edges = sum(routes[src][rank_idx] for src in range(num_ranks))
    remote_recv_edges = sum(routes[src][rank_idx] for src in range(num_ranks) if src != rank_idx)
    local_recv_edges = routes[rank_idx][rank_idx]
    if recv_edges != num_recv_tokens:
        raise RuntimeError(
            f"rank {rank_idx}: route recv edges ({recv_edges}) != DeepGEMM recv tokens ({num_recv_tokens})"
        )

    deepgemm_bytes = num_recv_tokens * hidden * 3
    remote_only_bytes = remote_recv_edges * hidden * 3
    reduction_s = num_tokens * hidden * 2 * (1 + num_topk) / 6.5e12
    approx_factor = t_fused_s / (t_fused_s - reduction_s) if t_fused_s > reduction_s else float("nan")
    safe_div = lambda numerator, denominator: float("nan") if denominator == 0 else numerator / denominator

    return {
        "rank": rank_idx,
        "t_fused_s": t_fused_s,
        "t_fused_ms": t_fused_s * 1e3,
        "num_recv_tokens_deepgemm": num_recv_tokens,
        "recv_edges": recv_edges,
        "remote_recv_edges": remote_recv_edges,
        "local_recv_edges": local_recv_edges,
        "deepgemm_nvlink_bytes": deepgemm_bytes,
        "remote_only_nvlink_bytes": remote_only_bytes,
        "deepgemm_effective_nvl_gbs": safe_div(deepgemm_bytes, t_fused_s) / 1e9,
        "remote_only_effective_nvl_gbs": safe_div(remote_only_bytes, t_fused_s) / 1e9,
        "reduction_s": reduction_s,
        "reduction_us": reduction_s * 1e6,
        "approx_factor": approx_factor,
        "deepgemm_overlap_nvl_gbs": safe_div(deepgemm_bytes, t_fused_s) / 1e9 * approx_factor,
        "remote_only_overlap_nvl_gbs": safe_div(remote_only_bytes, t_fused_s) / 1e9 * approx_factor,
    }


def _percentile(values: list[float], percent: float) -> float:
    if not values:
        return float("nan")
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    idx = (len(ordered) - 1) * percent / 100.0
    lower = int(idx)
    upper = min(lower + 1, len(ordered) - 1)
    weight = idx - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _stat_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            "mean": float("nan"),
            "median": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "p10": float("nan"),
            "p90": float("nan"),
        }
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
        "p10": _percentile(values, 10),
        "p90": _percentile(values, 90),
    }


def _aggregate_sample_rows(sample_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_tokens: dict[int, list[dict[str, Any]]] = {}
    for row in sample_rows:
        by_tokens.setdefault(int(row["tokens"]), []).append(row)

    metrics = [
        "ep8_effective_tps",
        "cluster_remote_only_effective_nvl_gbs",
        "cluster_deepgemm_effective_nvl_gbs",
        "max_t_fused_ms",
        "max_recv_edges",
        "max_remote_recv_edges",
        "max_remote_only_nvlink_bytes",
        "max_deepgemm_nvlink_bytes",
        "reduction_fraction_of_max_fused",
        "overlap_correction_factor",
    ]

    aggregated = []
    for tokens in sorted(by_tokens):
        rows = by_tokens[tokens]
        item: dict[str, Any] = {
            "tokens": tokens,
            "num_samples": len(rows),
            "samples": rows,
        }
        for metric in metrics:
            values = [float(row[metric]) for row in rows]
            item[metric] = _stat_summary(values)
        aggregated.append(item)
    return aggregated


def _run_worker(local_rank: int, num_local_ranks: int, args: argparse.Namespace) -> None:
    import random

    import deep_gemm
    import torch
    import torch.distributed as dist
    from deep_gemm.testing import bench_kineto
    from deep_gemm.utils import per_token_cast_to_fp4, per_token_cast_to_fp8
    from deep_gemm.utils.dist import dist_print, init_dist, uneven_all_gather

    rank_idx, num_ranks, group = init_dist(local_rank, num_local_ranks)
    torch.manual_seed(args.seed + rank_idx)
    random.seed(args.seed + rank_idx)

    token_cases = args.num_tokens_list or [args.num_tokens]
    num_max_tokens_per_rank = args.num_max_tokens_per_rank
    hidden = args.hidden
    intermediate_hidden = args.intermediate_hidden
    num_experts = args.num_experts
    num_topk = args.num_topk
    num_experts_per_rank = num_experts // num_ranks

    if not token_cases or any(num_tokens <= 0 for num_tokens in token_cases):
        raise ValueError("use explicit positive --num-tokens or --num-tokens-list for reproducible collection")
    if any(num_tokens > num_max_tokens_per_rank for num_tokens in token_cases):
        raise ValueError("all token counts must be <= num_max_tokens_per_rank")
    if hidden % 128 != 0 or intermediate_hidden % 128 != 0:
        raise ValueError("hidden and intermediate_hidden must be divisible by 128")
    if num_experts % num_ranks != 0:
        raise ValueError("num_experts must be divisible by num_ranks")

    buffer = deep_gemm.get_symm_buffer_for_mega_moe(
        group,
        num_experts,
        num_max_tokens_per_rank,
        num_topk,
        hidden,
        intermediate_hidden,
    )

    def cast_grouped_weights_to_fp4(bf16_weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        num_groups, n, k = bf16_weights.shape
        w = torch.empty((num_groups, n, k // 2), device="cuda", dtype=torch.int8)
        w_sf = torch.empty((num_groups, n, k // 32), device="cuda", dtype=torch.float)
        for group_idx in range(num_groups):
            w[group_idx], w_sf[group_idx] = per_token_cast_to_fp4(bf16_weights[group_idx], use_ue8m0=True, gran_k=32)
        w_sf = deep_gemm.transform_sf_into_required_layout(w_sf, n, k, (1, 32), num_groups)
        return w, w_sf

    l1_weights = torch.randn(
        (num_experts_per_rank, intermediate_hidden * 2, hidden),
        dtype=torch.bfloat16,
        device="cuda",
    )
    l2_weights = torch.randn(
        (num_experts_per_rank, hidden, intermediate_hidden),
        dtype=torch.bfloat16,
        device="cuda",
    )
    l1_weights = cast_grouped_weights_to_fp4(l1_weights)
    l2_weights = cast_grouped_weights_to_fp4(l2_weights)
    transformed_l1_weights, transformed_l2_weights = deep_gemm.transform_weights_for_mega_moe(l1_weights, l2_weights)
    mega_moe_kwargs: dict[str, Any] = {
        "activation_clamp": args.activation_clamp,
        "fast_math": bool(args.fast_math),
    }
    if "cumulative_local_expert_recv_stats" in inspect.signature(deep_gemm.fp8_fp4_mega_moe).parameters:
        mega_moe_kwargs["cumulative_local_expert_recv_stats"] = torch.zeros(
            (num_experts_per_rank,), dtype=torch.int, device="cuda"
        )

    if rank_idx == 0:
        print("Config:", flush=True)
        print(f" > Processes: {num_ranks}", flush=True)
        print(f" > Tokens per rank: {token_cases}/{num_max_tokens_per_rank}", flush=True)
        print(f" > Hidden: {hidden}", flush=True)
        print(f" > Intermediate: {intermediate_hidden}", flush=True)
        print(f" > Experts: {num_topk}/{num_experts}", flush=True)
        print(f" > Routing: {args.routing_mode}", flush=True)
        if args.routing_mode == "power-law":
            print(f" > Power-law alpha: {args.power_law_alpha}", flush=True)
        print("", flush=True)

    multi_case = len(token_cases) > 1
    output_dir = args.output
    if (multi_case or args.repeat_samples > 1) and output_dir.suffix:
        output_dir = output_dir.with_suffix("")
    if (multi_case or args.repeat_samples > 1) and rank_idx == 0:
        output_dir.mkdir(parents=True, exist_ok=True)

    sample_summary_rows: list[dict[str, Any]] = []

    for case_idx, num_tokens in enumerate(token_cases):
        for sample_idx in range(args.repeat_samples):
            routing_seed = args.seed + case_idx * 1000003 + sample_idx * 9176
            sample_seed = routing_seed + rank_idx
            torch.manual_seed(sample_seed)
            random.seed(sample_seed)

            x_bf16 = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device="cuda")
            x = per_token_cast_to_fp8(x_bf16, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)
            topk_weights, topk_idx, routing_metadata = _make_topk_inputs(
                args=args,
                rank_idx=rank_idx,
                num_ranks=num_ranks,
                num_tokens=num_tokens,
                num_experts=num_experts,
                num_topk=num_topk,
                sample_seed=sample_seed,
                routing_seed=routing_seed,
            )
            if args.masked_ratio > 0:
                rand_mask = torch.rand_like(topk_idx, dtype=torch.float)
                topk_idx.masked_fill_(rand_mask < args.masked_ratio, -1)
                topk_weights.masked_fill_(topk_idx < 0, 0)

            def run_fused():
                buffer.x[:num_tokens].copy_(x[0])
                buffer.x_sf[:num_tokens].copy_(x[1])
                buffer.topk_idx[:num_tokens].copy_(topk_idx)
                buffer.topk_weights[:num_tokens].copy_(topk_weights)
                y = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device="cuda")
                deep_gemm.fp8_fp4_mega_moe(
                    y,
                    transformed_l1_weights,
                    transformed_l2_weights,
                    buffer,
                    **mega_moe_kwargs,
                )
                return y

            if rank_idx == 0:
                print(f"Collecting tokens={num_tokens} sample={sample_idx}", flush=True)

            # Count local received token/expert selections exactly like DeepGEMM's test.
            gathered_topk_idx = uneven_all_gather(topk_idx, group=group)
            gathered_topk_idx[
                (gathered_topk_idx < rank_idx * num_experts_per_rank)
                | (gathered_topk_idx >= (rank_idx + 1) * num_experts_per_rank)
            ] = -1
            num_recv_tokens = int((gathered_topk_idx != -1).sum().item())

            trace_path = None
            if args.dump_profile_traces:
                trace_path = f"{args.dump_profile_traces}/tokens_{num_tokens}_sample{sample_idx}_rank{rank_idx}.json"

            t_fused_s = float(
                bench_kineto(
                    run_fused,
                    "mega_moe",
                    barrier=lambda: dist.barrier(),
                    trace_path=trace_path,
                )
            )

            topk_by_rank: list[list[list[int]] | None] = [None for _ in range(num_ranks)]
            dist.all_gather_object(topk_by_rank, topk_idx.detach().cpu().tolist())
            routes = _build_route_matrix(
                topk_by_rank, num_experts=num_experts, num_ranks=num_ranks  # type: ignore[arg-type]
            )
            rank_summary = _summarize_rank(
                rank_idx=rank_idx,
                num_ranks=num_ranks,
                routes=routes,
                num_recv_tokens=num_recv_tokens,
                hidden=hidden,
                num_tokens=num_tokens,
                num_topk=num_topk,
                t_fused_s=t_fused_s,
            )

            summaries: list[dict[str, Any] | None] = [None for _ in range(num_ranks)]
            dist.all_gather_object(summaries, rank_summary)

            if rank_idx == 0:
                per_rank = [summary for summary in summaries if summary is not None]
                aggregate = {
                    "bottleneck_rank_by_time": max(per_rank, key=lambda item: item["t_fused_s"])["rank"],
                    "bottleneck_rank_by_deepgemm_bytes": max(per_rank, key=lambda item: item["deepgemm_nvlink_bytes"])[
                        "rank"
                    ],
                    "bottleneck_rank_by_remote_only_bytes": max(
                        per_rank, key=lambda item: item["remote_only_nvlink_bytes"]
                    )["rank"],
                    "max_t_fused_ms": max(item["t_fused_ms"] for item in per_rank),
                    "max_recv_edges": max(item["recv_edges"] for item in per_rank),
                    "max_remote_recv_edges": max(item["remote_recv_edges"] for item in per_rank),
                    "max_deepgemm_nvlink_bytes": max(item["deepgemm_nvlink_bytes"] for item in per_rank),
                    "max_remote_only_nvlink_bytes": max(item["remote_only_nvlink_bytes"] for item in per_rank),
                }
                aggregate["cluster_deepgemm_effective_nvl_gbs"] = (
                    aggregate["max_deepgemm_nvlink_bytes"] / (aggregate["max_t_fused_ms"] / 1e3) / 1e9
                )
                aggregate["cluster_remote_only_effective_nvl_gbs"] = (
                    aggregate["max_remote_only_nvlink_bytes"] / (aggregate["max_t_fused_ms"] / 1e3) / 1e9
                )
                aggregate["ep8_effective_tps"] = num_tokens * num_ranks / (aggregate["max_t_fused_ms"] / 1e3)
                reduction_s = num_tokens * hidden * 2 * (1 + num_topk) / 6.5e12
                aggregate["reduction_us"] = reduction_s * 1e6
                aggregate["reduction_fraction_of_max_fused"] = reduction_s / (aggregate["max_t_fused_ms"] / 1e3)
                aggregate["overlap_correction_factor"] = (
                    1.0 / (1.0 - aggregate["reduction_fraction_of_max_fused"])
                    if aggregate["reduction_fraction_of_max_fused"] < 1.0
                    else float("nan")
                )
                payload = {
                    "schema": "aic-deepgemm-megamoe-effective-nvl-bw-v1",
                    "config": {
                        "num_processes": num_ranks,
                        "num_tokens": num_tokens,
                        "num_max_tokens_per_rank": num_max_tokens_per_rank,
                        "hidden": hidden,
                        "intermediate_hidden": intermediate_hidden,
                        "num_experts": num_experts,
                        "num_topk": num_topk,
                        "masked_ratio": args.masked_ratio,
                        "activation_clamp": args.activation_clamp,
                        "fast_math": args.fast_math,
                        "seed": args.seed,
                        "routing_seed": routing_seed,
                        "routing_mode": args.routing_mode,
                        "power_law_alpha": args.power_law_alpha,
                        "power_law_remap_hot_rank_to_zero": args.power_law_remap_hot_rank_to_zero,
                        "sample_idx": sample_idx,
                    },
                    "routing": routing_metadata,
                    "routes": routes,
                    "per_rank": per_rank,
                    "aggregate": aggregate,
                }
                if multi_case or args.repeat_samples > 1:
                    suffix = f"tokens_{num_tokens}_sample{sample_idx}.json"
                    output_path = output_dir / suffix
                else:
                    output_path = args.output
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                print(f"Wrote {output_path}", flush=True)
                sample_summary_rows.append({"tokens": num_tokens, "sample_idx": sample_idx, **aggregate})

    if (multi_case or args.repeat_samples > 1) and rank_idx == 0:
        aggregate_summary_rows = _aggregate_sample_rows(sample_summary_rows)
        (output_dir / "summary.json").write_text(json.dumps(aggregate_summary_rows, indent=2, sort_keys=True) + "\n")
        (output_dir / "summary_samples.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in sample_summary_rows)
        )
        print(f"Wrote {output_dir / 'summary.json'}", flush=True)

    if args.hard_exit_after_write:
        os._exit(0)

    dist.barrier()
    buffer.destroy()
    dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect DeepGEMM MegaMoE effective NVLink bandwidth")
    parser.add_argument("--num-processes", type=int, default=8)
    parser.add_argument("--local-rank-idx", type=int, default=None)
    parser.add_argument("--num-max-tokens-per-rank", type=int, default=8192)
    parser.add_argument("--num-tokens", type=int, default=None)
    parser.add_argument(
        "--num-tokens-list",
        type=str,
        default="",
        help="Comma-separated token counts to collect in one process group, reusing the same symmetric buffer.",
    )
    parser.add_argument("--hidden", type=int, default=7168)
    parser.add_argument("--intermediate-hidden", type=int, default=3072)
    parser.add_argument("--num-experts", type=int, default=384)
    parser.add_argument("--num-topk", type=int, default=6)
    parser.add_argument("--masked-ratio", type=float, default=0.0)
    parser.add_argument(
        "--routing-mode",
        choices=("random", "power-law"),
        default="random",
        help="random matches DeepGEMM's original test; power-law matches AIC MoE collector routing.",
    )
    parser.add_argument("--power-law-alpha", type=float, default=1.01)
    parser.add_argument(
        "--power-law-remap-hot-rank-to-zero",
        action="store_true",
        help="Match AIC's single-rank collector convention by moving the hottest expert group to EP rank 0.",
    )
    parser.add_argument("--activation-clamp", type=float, default=10.0)
    parser.add_argument("--fast-math", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeat-samples", type=int, default=1)
    parser.add_argument("--dump-profile-traces", type=str, default="")
    parser.add_argument("--output", type=Path, default=Path("artifacts/deepgemm_effective_nvl_bw/result.json"))
    parser.add_argument(
        "--hard-exit-after-write",
        action="store_true",
        help="Exit worker processes immediately after writing results; useful when symmetric-memory cleanup hangs.",
    )
    args = parser.parse_args()
    args.num_tokens_list = [int(item.strip()) for item in args.num_tokens_list.split(",") if item.strip()]
    if args.num_tokens is None and not args.num_tokens_list:
        parser.error("one of --num-tokens or --num-tokens-list is required")
    if args.repeat_samples <= 0:
        parser.error("--repeat-samples must be positive")

    if args.dump_profile_traces:
        os.makedirs(args.dump_profile_traces, exist_ok=True)

    if args.local_rank_idx is not None:
        _run_worker(args.local_rank_idx, args.num_processes, args)
    else:
        import torch

        torch.multiprocessing.spawn(_run_worker, args=(args.num_processes, args), nprocs=args.num_processes)


if __name__ == "__main__":
    main()
