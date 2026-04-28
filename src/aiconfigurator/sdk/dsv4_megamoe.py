# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V4 MegaMoE traffic and overlap helpers.

These helpers deliberately model communication from routing assignments instead
of measuring a distributed run.  DeepGEMM MegaMoE uses symmetric-memory remote
loads/stores inside the fused kernel, so AIC keeps the collector single-rank and
derives inter-rank movement here.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class MegaMoETrafficEstimate:
    route_matrix: tuple[tuple[int, ...], ...]
    hidden_size: int
    total_remote_edges: int
    primary_bytes_per_remote_selection: int
    total_primary_bytes: int
    owner_primary_bytes: tuple[int, ...]
    endpoint_primary_bytes: tuple[int, ...]
    bottleneck_primary_bytes: int
    metadata_bytes_per_remote_selection: int
    total_metadata_bytes: int
    owner_metadata_bytes: tuple[int, ...]
    bottleneck_metadata_bytes: int


@dataclass(frozen=True)
class MegaMoECommunicationEstimate:
    traffic: MegaMoETrafficEstimate
    effective_nvlink_bandwidth_bps: float
    nvl_barrier_latency_us: float
    nvl_barrier_count: int
    include_metadata: bool
    data_bytes: int
    data_ms: float
    barrier_ms: float
    total_ms: float


@dataclass(frozen=True)
class Dsv4MegaMoEWorkload:
    """Synthetic DSv4 MegaMoE routing workload for target-EP modeling.

    ``num_global_tokens`` is the number of synthetic routed token rows.
    ``num_tokens_per_rank`` is exact for per-rank builders and the per-rank cap
    for global-token builders, where the final rank may have fewer rows.
    Shared expert selections are kept out of ``route_matrix`` because they are
    local replicated work, not inter-rank routed traffic.
    """

    num_tokens_per_rank: int
    num_global_tokens: int
    routed_topk: int
    mega_topk: int
    routed_num_experts: int
    num_fused_shared_experts: int
    moe_ep_size: int
    experts_per_rank: int
    bottleneck_rank_before_remap: int
    routed_expert_counts: tuple[int, ...]
    routed_rank_loads: tuple[int, ...]
    routed_topk_ids_by_src_rank: tuple[tuple[tuple[int, ...], ...], ...]
    mega_topk_ids_by_src_rank: tuple[tuple[tuple[int, ...], ...], ...]
    route_matrix: tuple[tuple[int, ...], ...]
    rank0_local_token_indices: tuple[int, ...]
    rank0_local_topk_ids: tuple[tuple[int, ...], ...]
    rank0_masked_m: tuple[int, ...]
    traffic: MegaMoETrafficEstimate | None = None


def owner_rank_for_expert(expert_id: int, *, num_experts: int, moe_ep_size: int) -> int:
    """Return contiguous expert owner rank."""
    if moe_ep_size <= 0:
        raise ValueError("moe_ep_size must be positive")
    if num_experts <= 0:
        raise ValueError("num_experts must be positive")
    if num_experts % moe_ep_size != 0:
        raise ValueError("num_experts must be divisible by moe_ep_size")
    if expert_id < 0 or expert_id >= num_experts:
        raise ValueError(f"expert_id out of range: {expert_id}")

    experts_per_rank = num_experts // moe_ep_size
    return expert_id // experts_per_rank


def _argmin_index(values: Sequence[int]) -> int:
    return min(range(len(values)), key=lambda idx: values[idx])


def _argmax_index(values: Sequence[int]) -> int:
    return max(range(len(values)), key=lambda idx: values[idx])


def _round_robin_adjust_per_rank(
    counts_by_rank: list[list[int]],
    *,
    remaining: int,
    is_valid,
    pick_local_index,
    step: int,
) -> None:
    """Mirror collector.helper's per-rank rounding adjustment semantics."""
    while remaining > 0:
        progressed = False
        for local_counts in counts_by_rank:
            valid_local = [idx for idx, value in enumerate(local_counts) if is_valid(value)]
            if not valid_local:
                continue
            chosen_valid_idx = pick_local_index([local_counts[idx] for idx in valid_local])
            local_counts[valid_local[chosen_valid_idx]] += step
            remaining -= 1
            progressed = True
            if remaining == 0:
                break
        if not progressed:
            break


def _sample_power_law_value(rng: random.Random, alpha: float, xmin: float, xmax: float) -> float:
    if xmax <= xmin:
        return xmin
    u = rng.random()
    if math.isclose(alpha, 1.0):
        return xmin * ((xmax / xmin) ** u)
    return ((xmax ** (1.0 - alpha) - xmin ** (1.0 - alpha)) * u + xmin ** (1.0 - alpha)) ** (
        1.0 / (1.0 - alpha)
    )


def _generate_power_law_counts_with_bottleneck_remap(
    *,
    num_tokens: int,
    num_experts: int,
    topk: int,
    ep: int,
    alpha: float,
    seed: int | None,
) -> tuple[tuple[int, ...], int]:
    """Generate routed expert counts using AIC's target-EP power-law semantics.

    This is a torch-free counterpart of ``collector.helper``'s
    ``_generate_power_law_distribution``: counts are generated for the target EP
    size, adjusted rank-by-rank, then the max-load EP expert block is swapped
    into rank 0.
    """
    if num_tokens <= 0:
        raise ValueError("num_tokens must be positive")
    if num_experts <= 0:
        raise ValueError("num_experts must be positive")
    if topk <= 0:
        raise ValueError("topk must be positive")
    if topk > num_experts:
        raise ValueError("topk must not exceed num_experts")
    if ep <= 0:
        raise ValueError("ep must be positive")
    if num_experts % ep != 0:
        raise ValueError("num_experts must be divisible by ep")

    rng = random.Random(seed)
    if num_tokens * topk > num_experts:
        samples = [_sample_power_law_value(rng, alpha, 1.0, num_tokens * 0.8) for _ in range(num_experts)]
    else:
        samples = [_sample_power_law_value(rng, alpha, 0.01, 2.0) for _ in range(num_experts)]

    target_sum = num_tokens * topk
    sample_sum = sum(samples)
    if sample_sum <= 0:
        raise ValueError("invalid power-law samples")

    counts = [int(round(value / sample_sum * target_sum)) for value in samples]
    upper_bound = num_tokens
    overflow = sum(max(0, value - upper_bound) for value in counts)
    counts = [min(value, upper_bound) for value in counts]

    experts_per_rank = num_experts // ep
    counts_by_rank = [counts[start : start + experts_per_rank] for start in range(0, num_experts, experts_per_rank)]

    if overflow > 0:
        _round_robin_adjust_per_rank(
            counts_by_rank,
            remaining=int(overflow),
            is_valid=lambda value: value < upper_bound,
            pick_local_index=_argmin_index,
            step=1,
        )

    current_sum = sum(sum(row) for row in counts_by_rank)
    delta = target_sum - current_sum
    if delta > 0:
        _round_robin_adjust_per_rank(
            counts_by_rank,
            remaining=int(delta),
            is_valid=lambda value: value < upper_bound,
            pick_local_index=_argmin_index,
            step=1,
        )
    elif delta < 0:
        _round_robin_adjust_per_rank(
            counts_by_rank,
            remaining=int(-delta),
            is_valid=lambda value: value > 0,
            pick_local_index=_argmax_index,
            step=-1,
        )

    rank_loads = [sum(row) for row in counts_by_rank]
    bottleneck_rank = max(range(ep), key=lambda rank: rank_loads[rank])
    if bottleneck_rank != 0:
        counts_by_rank[0], counts_by_rank[bottleneck_rank] = counts_by_rank[bottleneck_rank], counts_by_rank[0]

    final_counts = tuple(value for row in counts_by_rank for value in row)
    if sum(final_counts) != target_sum:
        raise ValueError(f"power-law count sum mismatch: expected {target_sum}, got {sum(final_counts)}")
    if max(final_counts, default=0) > upper_bound:
        raise ValueError("power-law count exceeds per-expert upper bound")

    return final_counts, int(bottleneck_rank)


def _generate_uniform_counts(
    *,
    num_tokens: int,
    num_experts: int,
    topk: int,
) -> tuple[int, ...]:
    if num_tokens <= 0:
        raise ValueError("num_tokens must be positive")
    if num_experts <= 0:
        raise ValueError("num_experts must be positive")
    if topk <= 0:
        raise ValueError("topk must be positive")
    if topk > num_experts:
        raise ValueError("topk must not exceed num_experts")

    target_sum = num_tokens * topk
    base = target_sum // num_experts
    remainder = target_sum % num_experts
    counts = [base + (1 if expert_id < remainder else 0) for expert_id in range(num_experts)]
    if max(counts, default=0) > num_tokens:
        raise ValueError("uniform count exceeds per-expert upper bound")
    return tuple(counts)


def _assign_experts_from_counts(
    expert_counts: Sequence[int],
    *,
    num_tokens: int,
    topk: int,
) -> tuple[tuple[int, ...], ...]:
    """Assign count-level expert demand to token-major top-k IDs.

    This follows the same column-major fill as ``collector.helper``:
    sort experts by descending count, repeat IDs by count, reshape to
    ``(topk, num_tokens)``, then transpose.
    """
    if num_tokens <= 0:
        raise ValueError("num_tokens must be positive")
    if topk <= 0:
        raise ValueError("topk must be positive")
    if sum(expert_counts) != num_tokens * topk:
        raise ValueError("expert counts must sum to num_tokens * topk")
    if any(count < 0 for count in expert_counts):
        raise ValueError("expert counts must be non-negative")
    if max(expert_counts, default=0) > num_tokens:
        raise ValueError("each expert count must be <= num_tokens")

    sorted_experts = sorted(range(len(expert_counts)), key=lambda expert_id: (-expert_counts[expert_id], expert_id))
    flat: list[int] = []
    for expert_id in sorted_experts:
        flat.extend([expert_id] * int(expert_counts[expert_id]))
    if len(flat) != num_tokens * topk:
        raise ValueError("flat assignment length mismatch")

    return tuple(
        tuple(flat[topk_idx * num_tokens + token_idx] for topk_idx in range(topk))
        for token_idx in range(num_tokens)
    )


def _partition_token_rows_by_source_rank(
    token_rows: Sequence[Sequence[int]],
    *,
    moe_ep_size: int,
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    """Partition global token rows evenly across source ranks.

    The expert-assignment rows are generated from sorted counts, so their row
    order is not meaningful.  A round-robin partition avoids introducing a
    synthetic source-rank hotspot from that row order.
    """
    if moe_ep_size <= 0:
        raise ValueError("moe_ep_size must be positive")
    rows_by_rank: list[list[tuple[int, ...]]] = [[] for _ in range(moe_ep_size)]
    for token_idx, row in enumerate(token_rows):
        rows_by_rank[token_idx % moe_ep_size].append(tuple(int(value) for value in row))
    return tuple(tuple(rows) for rows in rows_by_rank)


def _append_shared_expert_ids(
    routed_rows_by_src_rank: Sequence[Sequence[Sequence[int]]],
    *,
    routed_num_experts: int,
    num_fused_shared_experts: int,
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    if num_fused_shared_experts < 0:
        raise ValueError("num_fused_shared_experts must be non-negative")
    if num_fused_shared_experts == 0:
        return tuple(
            tuple(tuple(int(value) for value in row) for row in rank_rows)
            for rank_rows in routed_rows_by_src_rank
        )

    shared_ids = tuple(routed_num_experts + idx for idx in range(num_fused_shared_experts))
    return tuple(
        tuple(tuple(int(value) for value in row) + shared_ids for row in rank_rows)
        for rank_rows in routed_rows_by_src_rank
    )


def _rank0_local_workload_from_global_assignment(
    routed_rows_by_src_rank: Sequence[Sequence[Sequence[int]]],
    *,
    routed_num_experts: int,
    moe_ep_size: int,
    num_fused_shared_experts: int,
) -> tuple[tuple[int, ...], tuple[tuple[int, ...], ...], tuple[int, ...]]:
    """Return rank0-local token rows, local top-k IDs, and per-expert counts."""
    experts_per_rank = routed_num_experts // moe_ep_size
    rank0_token_indices: list[int] = []
    rank0_rows: list[tuple[int, ...]] = []
    local_expert_counts = [0 for _ in range(experts_per_rank + num_fused_shared_experts)]

    for src_rank, rank_rows in enumerate(routed_rows_by_src_rank):
        for local_token_idx, routed_row in enumerate(rank_rows):
            global_token_idx = local_token_idx * moe_ep_size + src_rank
            local_row: list[int] = []
            for expert_id in routed_row:
                owner_rank = owner_rank_for_expert(
                    int(expert_id),
                    num_experts=routed_num_experts,
                    moe_ep_size=moe_ep_size,
                )
                if owner_rank == 0:
                    local_id = int(expert_id)
                    local_row.append(local_id)
                    local_expert_counts[local_id] += 1
                else:
                    local_row.append(-1)

            for shared_idx in range(num_fused_shared_experts):
                if src_rank == 0:
                    local_id = experts_per_rank + shared_idx
                    local_row.append(local_id)
                    local_expert_counts[local_id] += 1
                else:
                    local_row.append(-1)

            if any(expert_id >= 0 for expert_id in local_row):
                rank0_token_indices.append(global_token_idx)
                rank0_rows.append(tuple(local_row))

    return tuple(rank0_token_indices), tuple(rank0_rows), tuple(local_expert_counts)


def build_dsv4_power_law_megamoe_workload(
    *,
    num_tokens_per_rank: int,
    routed_num_experts: int,
    routed_topk: int,
    moe_ep_size: int,
    alpha: float,
    num_fused_shared_experts: int = 0,
    hidden_size: int | None = None,
    quant_group_size: int = 32,
    seed: int | None = 0,
) -> Dsv4MegaMoEWorkload:
    """Build a DSv4 MegaMoE target-EP power-law workload.

    The routed expert counts are generated for ``moe_ep_size`` first, and the
    bottleneck expert-owner rank is remapped into rank 0 before assignment.
    This preserves the existing AIC MoE collector convention while also
    retaining the full source-rank route matrix needed by MegaMoE communication
    modeling.
    """
    if num_tokens_per_rank <= 0:
        raise ValueError("num_tokens_per_rank must be positive")
    if moe_ep_size <= 0:
        raise ValueError("moe_ep_size must be positive")
    if routed_num_experts % moe_ep_size != 0:
        raise ValueError("routed_num_experts must be divisible by moe_ep_size")
    if num_fused_shared_experts < 0:
        raise ValueError("num_fused_shared_experts must be non-negative")

    num_global_tokens = num_tokens_per_rank * moe_ep_size
    experts_per_rank = routed_num_experts // moe_ep_size
    routed_counts, bottleneck_rank = _generate_power_law_counts_with_bottleneck_remap(
        num_tokens=num_global_tokens,
        num_experts=routed_num_experts,
        topk=routed_topk,
        ep=moe_ep_size,
        alpha=alpha,
        seed=seed,
    )
    routed_rows = _assign_experts_from_counts(
        routed_counts,
        num_tokens=num_global_tokens,
        topk=routed_topk,
    )
    routed_rows_by_src_rank = _partition_token_rows_by_source_rank(
        routed_rows,
        moe_ep_size=moe_ep_size,
    )
    mega_rows_by_src_rank = _append_shared_expert_ids(
        routed_rows_by_src_rank,
        routed_num_experts=routed_num_experts,
        num_fused_shared_experts=num_fused_shared_experts,
    )
    routes = build_route_matrix(
        routed_rows_by_src_rank,
        num_experts=routed_num_experts,
        moe_ep_size=moe_ep_size,
    )
    rank0_token_indices, rank0_topk_ids, rank0_masked_m = _rank0_local_workload_from_global_assignment(
        routed_rows_by_src_rank,
        routed_num_experts=routed_num_experts,
        moe_ep_size=moe_ep_size,
        num_fused_shared_experts=num_fused_shared_experts,
    )
    rank_loads = tuple(
        sum(routed_counts[rank * experts_per_rank : (rank + 1) * experts_per_rank])
        for rank in range(moe_ep_size)
    )
    traffic = (
        estimate_megamoe_traffic(routes, hidden_size=hidden_size, quant_group_size=quant_group_size)
        if hidden_size is not None
        else None
    )

    return Dsv4MegaMoEWorkload(
        num_tokens_per_rank=int(num_tokens_per_rank),
        num_global_tokens=int(num_global_tokens),
        routed_topk=int(routed_topk),
        mega_topk=int(routed_topk + num_fused_shared_experts),
        routed_num_experts=int(routed_num_experts),
        num_fused_shared_experts=int(num_fused_shared_experts),
        moe_ep_size=int(moe_ep_size),
        experts_per_rank=int(experts_per_rank),
        bottleneck_rank_before_remap=int(bottleneck_rank),
        routed_expert_counts=routed_counts,
        routed_rank_loads=rank_loads,
        routed_topk_ids_by_src_rank=routed_rows_by_src_rank,
        mega_topk_ids_by_src_rank=mega_rows_by_src_rank,
        route_matrix=routes,
        rank0_local_token_indices=rank0_token_indices,
        rank0_local_topk_ids=rank0_topk_ids,
        rank0_masked_m=rank0_masked_m,
        traffic=traffic,
    )


def build_dsv4_power_law_megamoe_workload_from_global_tokens(
    *,
    num_global_tokens: int,
    routed_num_experts: int,
    routed_topk: int,
    moe_ep_size: int,
    alpha: float,
    num_fused_shared_experts: int = 0,
    hidden_size: int | None = None,
    quant_group_size: int = 32,
    seed: int | None = 0,
) -> Dsv4MegaMoEWorkload:
    """Build a target-EP power-law workload from global token count."""
    if num_global_tokens <= 0:
        raise ValueError("num_global_tokens must be positive")
    if moe_ep_size <= 0:
        raise ValueError("moe_ep_size must be positive")
    if routed_num_experts % moe_ep_size != 0:
        raise ValueError("routed_num_experts must be divisible by moe_ep_size")
    if num_fused_shared_experts < 0:
        raise ValueError("num_fused_shared_experts must be non-negative")

    experts_per_rank = routed_num_experts // moe_ep_size
    routed_counts, bottleneck_rank = _generate_power_law_counts_with_bottleneck_remap(
        num_tokens=num_global_tokens,
        num_experts=routed_num_experts,
        topk=routed_topk,
        ep=moe_ep_size,
        alpha=alpha,
        seed=seed,
    )
    routed_rows = _assign_experts_from_counts(
        routed_counts,
        num_tokens=num_global_tokens,
        topk=routed_topk,
    )
    routed_rows_by_src_rank = _partition_token_rows_by_source_rank(
        routed_rows,
        moe_ep_size=moe_ep_size,
    )
    mega_rows_by_src_rank = _append_shared_expert_ids(
        routed_rows_by_src_rank,
        routed_num_experts=routed_num_experts,
        num_fused_shared_experts=num_fused_shared_experts,
    )
    routes = build_route_matrix(
        routed_rows_by_src_rank,
        num_experts=routed_num_experts,
        moe_ep_size=moe_ep_size,
    )
    rank0_token_indices, rank0_topk_ids, rank0_masked_m = _rank0_local_workload_from_global_assignment(
        routed_rows_by_src_rank,
        routed_num_experts=routed_num_experts,
        moe_ep_size=moe_ep_size,
        num_fused_shared_experts=num_fused_shared_experts,
    )
    rank_loads = tuple(
        sum(routed_counts[rank * experts_per_rank : (rank + 1) * experts_per_rank])
        for rank in range(moe_ep_size)
    )
    traffic = (
        estimate_megamoe_traffic(routes, hidden_size=hidden_size, quant_group_size=quant_group_size)
        if hidden_size is not None
        else None
    )

    return Dsv4MegaMoEWorkload(
        num_tokens_per_rank=int(math.ceil(num_global_tokens / moe_ep_size)),
        num_global_tokens=int(num_global_tokens),
        routed_topk=int(routed_topk),
        mega_topk=int(routed_topk + num_fused_shared_experts),
        routed_num_experts=int(routed_num_experts),
        num_fused_shared_experts=int(num_fused_shared_experts),
        moe_ep_size=int(moe_ep_size),
        experts_per_rank=int(experts_per_rank),
        bottleneck_rank_before_remap=int(bottleneck_rank),
        routed_expert_counts=routed_counts,
        routed_rank_loads=rank_loads,
        routed_topk_ids_by_src_rank=routed_rows_by_src_rank,
        mega_topk_ids_by_src_rank=mega_rows_by_src_rank,
        route_matrix=routes,
        rank0_local_token_indices=rank0_token_indices,
        rank0_local_topk_ids=rank0_topk_ids,
        rank0_masked_m=rank0_masked_m,
        traffic=traffic,
    )


def build_dsv4_uniform_megamoe_workload(
    *,
    num_tokens_per_rank: int,
    routed_num_experts: int,
    routed_topk: int,
    moe_ep_size: int,
    num_fused_shared_experts: int = 0,
    hidden_size: int | None = None,
    quant_group_size: int = 32,
) -> Dsv4MegaMoEWorkload:
    """Build a target-EP uniform DSv4 MegaMoE local-compute workload."""
    if num_tokens_per_rank <= 0:
        raise ValueError("num_tokens_per_rank must be positive")
    if moe_ep_size <= 0:
        raise ValueError("moe_ep_size must be positive")
    if routed_num_experts % moe_ep_size != 0:
        raise ValueError("routed_num_experts must be divisible by moe_ep_size")
    if num_fused_shared_experts < 0:
        raise ValueError("num_fused_shared_experts must be non-negative")

    num_global_tokens = num_tokens_per_rank * moe_ep_size
    experts_per_rank = routed_num_experts // moe_ep_size
    routed_counts = _generate_uniform_counts(
        num_tokens=num_global_tokens,
        num_experts=routed_num_experts,
        topk=routed_topk,
    )
    routed_rows = _assign_experts_from_counts(
        routed_counts,
        num_tokens=num_global_tokens,
        topk=routed_topk,
    )
    routed_rows_by_src_rank = _partition_token_rows_by_source_rank(
        routed_rows,
        moe_ep_size=moe_ep_size,
    )
    mega_rows_by_src_rank = _append_shared_expert_ids(
        routed_rows_by_src_rank,
        routed_num_experts=routed_num_experts,
        num_fused_shared_experts=num_fused_shared_experts,
    )
    routes = build_route_matrix(
        routed_rows_by_src_rank,
        num_experts=routed_num_experts,
        moe_ep_size=moe_ep_size,
    )
    rank0_token_indices, rank0_topk_ids, rank0_masked_m = _rank0_local_workload_from_global_assignment(
        routed_rows_by_src_rank,
        routed_num_experts=routed_num_experts,
        moe_ep_size=moe_ep_size,
        num_fused_shared_experts=num_fused_shared_experts,
    )
    rank_loads = tuple(
        sum(routed_counts[rank * experts_per_rank : (rank + 1) * experts_per_rank])
        for rank in range(moe_ep_size)
    )
    traffic = (
        estimate_megamoe_traffic(routes, hidden_size=hidden_size, quant_group_size=quant_group_size)
        if hidden_size is not None
        else None
    )

    return Dsv4MegaMoEWorkload(
        num_tokens_per_rank=int(num_tokens_per_rank),
        num_global_tokens=int(num_global_tokens),
        routed_topk=int(routed_topk),
        mega_topk=int(routed_topk + num_fused_shared_experts),
        routed_num_experts=int(routed_num_experts),
        num_fused_shared_experts=int(num_fused_shared_experts),
        moe_ep_size=int(moe_ep_size),
        experts_per_rank=int(experts_per_rank),
        bottleneck_rank_before_remap=0,
        routed_expert_counts=routed_counts,
        routed_rank_loads=rank_loads,
        routed_topk_ids_by_src_rank=routed_rows_by_src_rank,
        mega_topk_ids_by_src_rank=mega_rows_by_src_rank,
        route_matrix=routes,
        rank0_local_token_indices=rank0_token_indices,
        rank0_local_topk_ids=rank0_topk_ids,
        rank0_masked_m=rank0_masked_m,
        traffic=traffic,
    )


def build_dsv4_uniform_megamoe_workload_from_global_tokens(
    *,
    num_global_tokens: int,
    routed_num_experts: int,
    routed_topk: int,
    moe_ep_size: int,
    num_fused_shared_experts: int = 0,
    hidden_size: int | None = None,
    quant_group_size: int = 32,
) -> Dsv4MegaMoEWorkload:
    """Build a target-EP uniform workload from global token count."""
    if num_global_tokens <= 0:
        raise ValueError("num_global_tokens must be positive")
    if moe_ep_size <= 0:
        raise ValueError("moe_ep_size must be positive")
    if routed_num_experts % moe_ep_size != 0:
        raise ValueError("routed_num_experts must be divisible by moe_ep_size")
    if num_fused_shared_experts < 0:
        raise ValueError("num_fused_shared_experts must be non-negative")

    experts_per_rank = routed_num_experts // moe_ep_size
    routed_counts = _generate_uniform_counts(
        num_tokens=num_global_tokens,
        num_experts=routed_num_experts,
        topk=routed_topk,
    )
    routed_rows = _assign_experts_from_counts(
        routed_counts,
        num_tokens=num_global_tokens,
        topk=routed_topk,
    )
    routed_rows_by_src_rank = _partition_token_rows_by_source_rank(
        routed_rows,
        moe_ep_size=moe_ep_size,
    )
    mega_rows_by_src_rank = _append_shared_expert_ids(
        routed_rows_by_src_rank,
        routed_num_experts=routed_num_experts,
        num_fused_shared_experts=num_fused_shared_experts,
    )
    routes = build_route_matrix(
        routed_rows_by_src_rank,
        num_experts=routed_num_experts,
        moe_ep_size=moe_ep_size,
    )
    rank0_token_indices, rank0_topk_ids, rank0_masked_m = _rank0_local_workload_from_global_assignment(
        routed_rows_by_src_rank,
        routed_num_experts=routed_num_experts,
        moe_ep_size=moe_ep_size,
        num_fused_shared_experts=num_fused_shared_experts,
    )
    rank_loads = tuple(
        sum(routed_counts[rank * experts_per_rank : (rank + 1) * experts_per_rank])
        for rank in range(moe_ep_size)
    )
    traffic = (
        estimate_megamoe_traffic(routes, hidden_size=hidden_size, quant_group_size=quant_group_size)
        if hidden_size is not None
        else None
    )

    return Dsv4MegaMoEWorkload(
        num_tokens_per_rank=int(math.ceil(num_global_tokens / moe_ep_size)),
        num_global_tokens=int(num_global_tokens),
        routed_topk=int(routed_topk),
        mega_topk=int(routed_topk + num_fused_shared_experts),
        routed_num_experts=int(routed_num_experts),
        num_fused_shared_experts=int(num_fused_shared_experts),
        moe_ep_size=int(moe_ep_size),
        experts_per_rank=int(experts_per_rank),
        bottleneck_rank_before_remap=0,
        routed_expert_counts=routed_counts,
        routed_rank_loads=rank_loads,
        routed_topk_ids_by_src_rank=routed_rows_by_src_rank,
        mega_topk_ids_by_src_rank=mega_rows_by_src_rank,
        route_matrix=routes,
        rank0_local_token_indices=rank0_token_indices,
        rank0_local_topk_ids=rank0_topk_ids,
        rank0_masked_m=rank0_masked_m,
        traffic=traffic,
    )


def build_route_matrix(
    topk_ids_by_src_rank: Sequence[Sequence[Sequence[int]]],
    *,
    num_experts: int,
    moe_ep_size: int,
) -> tuple[tuple[int, ...], ...]:
    """Build routes[src_rank][dst_rank] from per-source-rank topk IDs.

    ``topk_ids_by_src_rank[src]`` is a token-major matrix of expert IDs for the
    tokens originating on ``src``.  Expert ID ``-1`` is treated as masked and is
    ignored.
    """
    if len(topk_ids_by_src_rank) != moe_ep_size:
        raise ValueError("topk_ids_by_src_rank length must equal moe_ep_size")

    routes = [[0 for _ in range(moe_ep_size)] for _ in range(moe_ep_size)]
    for src_rank, rank_topk in enumerate(topk_ids_by_src_rank):
        for token_topk in rank_topk:
            for expert_id in token_topk:
                expert_id = int(expert_id)
                if expert_id == -1:
                    continue
                dst_rank = owner_rank_for_expert(
                    expert_id,
                    num_experts=num_experts,
                    moe_ep_size=moe_ep_size,
                )
                routes[src_rank][dst_rank] += 1
    return tuple(tuple(row) for row in routes)


def normalize_route_matrix(route_matrix: Sequence[Sequence[int]]) -> tuple[tuple[int, ...], ...]:
    if not route_matrix:
        raise ValueError("route_matrix must be non-empty")

    width = len(route_matrix)
    normalized: list[tuple[int, ...]] = []
    for row in route_matrix:
        if len(row) != width:
            raise ValueError("route_matrix must be square")
        normalized_row = tuple(int(value) for value in row)
        if any(value < 0 for value in normalized_row):
            raise ValueError("route_matrix entries must be non-negative")
        normalized.append(normalized_row)
    return tuple(normalized)


def estimate_megamoe_traffic(
    route_matrix: Sequence[Sequence[int]],
    *,
    hidden_size: int,
    quant_group_size: int = 32,
) -> MegaMoETrafficEstimate:
    """Estimate DSv4 MegaMoE inter-rank traffic from a route matrix.

    Primary bytes follow DeepGEMM's benchmark convention:
    dispatch pull FP8 activation (``hidden`` bytes) plus combine write-back BF16
    output (``2 * hidden`` bytes), or ``3 * hidden`` per remote token/expert
    selection.
    """
    if hidden_size <= 0:
        raise ValueError("hidden_size must be positive")
    if quant_group_size <= 0:
        raise ValueError("quant_group_size must be positive")

    routes = normalize_route_matrix(route_matrix)
    ep_size = len(routes)
    primary_per_edge = 3 * hidden_size
    metadata_per_edge = (hidden_size + quant_group_size - 1) // quant_group_size + 8

    owner_edges = [0 for _ in range(ep_size)]
    endpoint_edges = [0 for _ in range(ep_size)]
    total_remote_edges = 0
    for src_rank, row in enumerate(routes):
        for dst_rank, edges in enumerate(row):
            if src_rank == dst_rank:
                continue
            total_remote_edges += edges
            owner_edges[dst_rank] += edges
            endpoint_edges[src_rank] += edges

    owner_primary = tuple(edges * primary_per_edge for edges in owner_edges)
    endpoint_primary = tuple(edges * primary_per_edge for edges in endpoint_edges)
    owner_metadata = tuple(edges * metadata_per_edge for edges in owner_edges)

    return MegaMoETrafficEstimate(
        route_matrix=routes,
        hidden_size=hidden_size,
        total_remote_edges=total_remote_edges,
        primary_bytes_per_remote_selection=primary_per_edge,
        total_primary_bytes=total_remote_edges * primary_per_edge,
        owner_primary_bytes=owner_primary,
        endpoint_primary_bytes=endpoint_primary,
        bottleneck_primary_bytes=max(owner_primary, default=0),
        metadata_bytes_per_remote_selection=metadata_per_edge,
        total_metadata_bytes=total_remote_edges * metadata_per_edge,
        owner_metadata_bytes=owner_metadata,
        bottleneck_metadata_bytes=max(owner_metadata, default=0),
    )


def estimate_megamoe_communication_ms(
    route_matrix: Sequence[Sequence[int]],
    *,
    hidden_size: int,
    effective_nvlink_bandwidth_bps: float,
    nvl_barrier_latency_us: float,
    nvl_barrier_count: int = 2,
    include_metadata: bool = False,
    quant_group_size: int = 32,
) -> MegaMoECommunicationEstimate:
    if effective_nvlink_bandwidth_bps <= 0:
        raise ValueError("effective_nvlink_bandwidth_bps must be positive")
    if nvl_barrier_latency_us < 0:
        raise ValueError("nvl_barrier_latency_us must be non-negative")
    if nvl_barrier_count < 0:
        raise ValueError("nvl_barrier_count must be non-negative")

    traffic = estimate_megamoe_traffic(
        route_matrix,
        hidden_size=hidden_size,
        quant_group_size=quant_group_size,
    )
    data_bytes = traffic.bottleneck_primary_bytes
    if include_metadata:
        data_bytes += traffic.bottleneck_metadata_bytes

    data_ms = data_bytes / effective_nvlink_bandwidth_bps * 1000.0
    barrier_ms = nvl_barrier_count * nvl_barrier_latency_us / 1000.0
    return MegaMoECommunicationEstimate(
        traffic=traffic,
        effective_nvlink_bandwidth_bps=float(effective_nvlink_bandwidth_bps),
        nvl_barrier_latency_us=float(nvl_barrier_latency_us),
        nvl_barrier_count=int(nvl_barrier_count),
        include_metadata=include_metadata,
        data_bytes=data_bytes,
        data_ms=data_ms,
        barrier_ms=barrier_ms,
        total_ms=data_ms + barrier_ms,
    )


def compose_megamoe_routed_latency_ms(
    *,
    local_routing_prep_ms: float,
    core_compute_ms: float,
    comm_ms: float,
    sync_tail_ms: float = 0.0,
) -> float:
    """Compose routed MegaMoE latency with communication-compute overlap."""
    values = {
        "local_routing_prep_ms": local_routing_prep_ms,
        "core_compute_ms": core_compute_ms,
        "comm_ms": comm_ms,
        "sync_tail_ms": sync_tail_ms,
    }
    for name, value in values.items():
        if value < 0:
            raise ValueError(f"{name} must be non-negative")

    return local_routing_prep_ms + max(core_compute_ms, comm_ms) + sync_tail_ms


def route_matrix_from_flat_assignments(
    assignments: Iterable[tuple[int, int]],
    *,
    num_experts: int,
    moe_ep_size: int,
) -> tuple[tuple[int, ...], ...]:
    """Build a route matrix from ``(src_rank, expert_id)`` assignments."""
    if moe_ep_size <= 0:
        raise ValueError("moe_ep_size must be positive")
    routes = [[0 for _ in range(moe_ep_size)] for _ in range(moe_ep_size)]
    for src_rank, expert_id in assignments:
        src_rank = int(src_rank)
        if src_rank < 0 or src_rank >= moe_ep_size:
            raise ValueError(f"src_rank out of range: {src_rank}")
        expert_id = int(expert_id)
        if expert_id == -1:
            continue
        dst_rank = owner_rank_for_expert(
            expert_id,
            num_experts=num_experts,
            moe_ep_size=moe_ep_size,
        )
        routes[src_rank][dst_rank] += 1
    return tuple(tuple(row) for row in routes)
