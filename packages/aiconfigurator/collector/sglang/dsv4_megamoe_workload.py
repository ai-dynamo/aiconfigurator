# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Workload helpers for DeepSeek-V4 MegaMoE collection.

This module intentionally reuses AIC's existing MoE distribution helpers instead
of reimplementing power-law or balanced routing.  It adds only the source-rank
placement layer needed by real EP collection.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

try:
    from collector.helper import balanced_logits, power_law_logits_v3, sample_power_law
except ImportError:
    from helper import balanced_logits, power_law_logits_v3, sample_power_law


SAMPLED_POWER_LAW_DISTRIBUTION = "power_law_sampled_1.9"
SAMPLED_POWER_LAW_ALPHA = 1.9
SAMPLED_POWER_LAW_XMAX = 1024.0
SUPPORTED_DISTRIBUTIONS = (
    "balanced",
    "power_law_<alpha>",
    SAMPLED_POWER_LAW_DISTRIBUTION,
)
SUPPORTED_SOURCE_POLICIES = ("random",)


@dataclass(frozen=True)
class DistributionSpec:
    name: str
    kind: str
    alpha: float | None = None


@dataclass(frozen=True)
class RoutingPlan:
    distribution: str
    source_policy: str
    global_num_tokens: int
    local_topk_ids: torch.Tensor
    local_topk_weights: torch.Tensor


def parse_distribution(distribution: str) -> DistributionSpec:
    if distribution == "balanced":
        return DistributionSpec(name=distribution, kind="balanced")
    if distribution == SAMPLED_POWER_LAW_DISTRIBUTION:
        return DistributionSpec(name=distribution, kind="sampled_power_law", alpha=SAMPLED_POWER_LAW_ALPHA)
    if distribution.startswith("power_law_sampled_"):
        raise ValueError(
            f"unsupported sampled power-law distribution: {distribution}; "
            f"only {SAMPLED_POWER_LAW_DISTRIBUTION} has been collected and validated"
        )
    if distribution.startswith("power_law_"):
        alpha = float(distribution.removeprefix("power_law_"))
        if alpha <= 0:
            raise ValueError(f"power-law alpha must be positive, got {alpha}")
        return DistributionSpec(name=distribution, kind="power_law", alpha=alpha)
    raise ValueError(f"unsupported distribution: {distribution}; expected one of {SUPPORTED_DISTRIBUTIONS}")


def parse_int_list(value: str | Sequence[int]) -> list[int]:
    if isinstance(value, str):
        if not value:
            return []
        return [int(item.strip()) for item in value.split(",") if item.strip()]
    return [int(item) for item in value]


def _validate_common(
    *,
    tokens_per_rank: Sequence[int],
    routed_num_experts: int,
    routed_topk: int,
    ep_size: int,
    rank: int,
) -> tuple[int, ...]:
    if ep_size <= 0:
        raise ValueError("ep_size must be positive")
    if rank < 0 or rank >= ep_size:
        raise ValueError(f"rank {rank} must be within [0, {ep_size})")
    if routed_topk <= 0:
        raise ValueError("routed_topk must be positive")
    if routed_num_experts <= 0:
        raise ValueError("routed_num_experts must be positive")
    if routed_num_experts % ep_size != 0:
        raise ValueError("routed_num_experts must be divisible by ep_size")
    normalized = tuple(int(item) for item in tokens_per_rank)
    if len(normalized) != ep_size:
        raise ValueError("tokens_per_rank length must equal ep_size")
    if any(item <= 0 for item in normalized):
        raise ValueError("all tokens_per_rank entries must be positive")
    return normalized


def _logits_for_distribution(
    *,
    spec: DistributionSpec,
    global_num_tokens: int,
    routed_num_experts: int,
    routed_topk: int,
    ep_size: int,
) -> torch.Tensor:
    if spec.kind == "balanced":
        return balanced_logits(global_num_tokens, routed_num_experts, routed_topk).to(dtype=torch.float32, device="cpu")
    if spec.kind == "power_law":
        if spec.alpha is None:
            raise ValueError("power-law distribution requires alpha")
        return power_law_logits_v3(
            global_num_tokens,
            routed_num_experts,
            routed_topk,
            ep_size,
            spec.alpha,
            use_eplb=False,
            return_rank0_info=False,
        ).to(dtype=torch.float32, device="cpu")
    if spec.kind == "sampled_power_law":
        if spec.alpha is None:
            raise ValueError("sampled power-law distribution requires alpha")
        return sampled_power_law_logits(
            global_num_tokens,
            routed_num_experts,
            routed_topk,
            ep_size,
            spec.alpha,
        ).to(dtype=torch.float32, device="cpu")
    raise ValueError(f"unsupported distribution kind: {spec.kind}")


def _swap_max_rank_to_rank0(topk_ids: torch.Tensor, *, num_experts: int, ep: int) -> torch.Tensor:
    experts_per_rank = num_experts // ep
    rank_loads = torch.bincount(
        torch.div(topk_ids.reshape(-1).to(dtype=torch.int64), experts_per_rank, rounding_mode="floor"),
        minlength=ep,
    )[:ep]
    max_rank = int(torch.argmax(rank_loads).item())
    if max_rank == 0:
        return topk_ids

    swapped = topk_ids.clone()
    rank0_start = 0
    rank0_end = experts_per_rank
    max_start = max_rank * experts_per_rank
    max_end = max_start + experts_per_rank
    rank0_mask = (topk_ids >= rank0_start) & (topk_ids < rank0_end)
    max_mask = (topk_ids >= max_start) & (topk_ids < max_end)
    swapped[rank0_mask] = topk_ids[rank0_mask] + max_start
    swapped[max_mask] = topk_ids[max_mask] - max_start
    return swapped


def _sampled_power_law_xmax(_global_num_tokens: int) -> float:
    """Fixed xmax for top-k sampled synthetic routing."""
    return SAMPLED_POWER_LAW_XMAX


def sampled_power_law_logits(num_tokens: int, num_experts: int, topk: int, ep: int, alpha: float) -> torch.Tensor:
    """Generate power-law routing by sampling per-token top-k rows.

    AIC's existing ``power_law_logits_v3`` creates a per-expert count vector by
    rounding continuous weights, then assigns those counts to token rows.  That
    can over-spread low-token decode workloads.  This variant keeps the same
    inverse-CDF power-law weight family but samples each token's top-k experts
    without replacement, matching the discrete shape produced by a real router
    without adding latency correction factors.
    """
    import torch.nn.functional as F

    if topk > num_experts:
        raise ValueError(f"topk={topk} cannot exceed num_experts={num_experts}")
    if num_tokens <= 0:
        raise ValueError("num_tokens must be positive")

    expert_weights = sample_power_law(num_experts, alpha, 1, _sampled_power_law_xmax(num_tokens))
    expert_weights = expert_weights.to(dtype=torch.float64, device="cpu").clamp_min(1e-20)

    selected_batches: list[torch.Tensor] = []
    batch_size = min(num_tokens, 8192)
    for offset in range(0, num_tokens, batch_size):
        current = min(batch_size, num_tokens - offset)
        batch_weights = expert_weights.unsqueeze(0).expand(current, -1).contiguous()
        selected_batches.append(torch.multinomial(batch_weights, topk, replacement=False))
    selected_experts = torch.cat(selected_batches, dim=0).to(dtype=torch.int64, device="cpu")
    selected_experts = _swap_max_rank_to_rank0(selected_experts, num_experts=num_experts, ep=ep)
    expert_map = F.one_hot(selected_experts, num_classes=num_experts).sum(1)
    return F.softmax(expert_map.bfloat16(), dim=1)


def _validate_plan(
    *,
    topk_ids_by_rank: Sequence[torch.Tensor],
    topk_weights_by_rank: Sequence[torch.Tensor],
    routed_num_experts: int,
    routed_topk: int,
    tokens_per_rank: Sequence[int],
    expected_expert_counts: torch.Tensor,
) -> None:
    if len(topk_ids_by_rank) != len(tokens_per_rank):
        raise ValueError("rank count mismatch")
    for rank, (topk_ids, topk_weights, tokens) in enumerate(
        zip(topk_ids_by_rank, topk_weights_by_rank, tokens_per_rank, strict=True)
    ):
        if tuple(topk_ids.shape) != (tokens, routed_topk):
            raise ValueError(f"rank {rank} topk_ids shape mismatch: {tuple(topk_ids.shape)}")
        if tuple(topk_weights.shape) != (tokens, routed_topk):
            raise ValueError(f"rank {rank} topk_weights shape mismatch: {tuple(topk_weights.shape)}")
        if topk_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("topk_ids must be integer")
        if torch.any(topk_ids < 0) or torch.any(topk_ids >= routed_num_experts):
            raise ValueError("topk_ids contain invalid routed expert ids")
        sorted_ids = torch.sort(topk_ids.to(dtype=torch.int64), dim=1).values
        if torch.any(sorted_ids[:, 1:] == sorted_ids[:, :-1]):
            raise ValueError("a token row contains duplicate expert ids")

    merged = torch.cat([item.reshape(-1).to(dtype=torch.int64) for item in topk_ids_by_rank])
    actual_counts = torch.bincount(merged, minlength=routed_num_experts).to(dtype=torch.int64)
    if not torch.equal(actual_counts[:routed_num_experts], expected_expert_counts.to(dtype=torch.int64)):
        raise ValueError("expert counts changed while assigning source ranks")


def build_routing_plan(
    *,
    distribution: str,
    tokens_per_rank: Sequence[int],
    routed_num_experts: int,
    routed_topk: int,
    ep_size: int,
    rank: int,
    source_policy: str = "random",
    routing_seed: int = 0,
    norm_topk_prob: bool = True,
) -> RoutingPlan:
    """Build a local routing plan from an AIC distribution helper.

    ``distribution`` controls destination expert load.  ``source_policy`` controls
    source-rank placement of the generated token rows.

    ``random`` shuffles whole token rows before assigning them to source ranks,
    preserving both expert counts and per-token top-k structure.

    """
    tokens_per_rank = _validate_common(
        tokens_per_rank=tokens_per_rank,
        routed_num_experts=routed_num_experts,
        routed_topk=routed_topk,
        ep_size=ep_size,
        rank=rank,
    )
    if source_policy != "random":
        raise ValueError(f"synthetic routing only supports source_policy=random, got {source_policy}")

    spec = parse_distribution(distribution)
    global_num_tokens = int(sum(tokens_per_rank))
    rng_state = torch.random.get_rng_state()
    torch.manual_seed(int(routing_seed))
    try:
        logits = _logits_for_distribution(
            spec=spec,
            global_num_tokens=global_num_tokens,
            routed_num_experts=routed_num_experts,
            routed_topk=routed_topk,
            ep_size=ep_size,
        )
    finally:
        torch.random.set_rng_state(rng_state)
    topk_weights, topk_ids = torch.topk(logits, k=routed_topk, dim=-1, largest=True, sorted=False)
    if norm_topk_prob:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True).clamp_min(1e-20)
    topk_ids = topk_ids.to(dtype=torch.int32, device="cpu").contiguous()
    topk_weights = topk_weights.to(dtype=torch.float32, device="cpu").contiguous()
    expected_expert_counts = torch.bincount(
        topk_ids.reshape(-1).to(dtype=torch.int64),
        minlength=routed_num_experts,
    )

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(routing_seed))
    permutation = torch.randperm(global_num_tokens, generator=generator)
    topk_ids = topk_ids[permutation].contiguous()
    topk_weights = topk_weights[permutation].contiguous()

    topk_ids_by_rank = []
    topk_weights_by_rank = []
    offset = 0
    for tokens in tokens_per_rank:
        end = offset + tokens
        topk_ids_by_rank.append(topk_ids[offset:end].contiguous())
        topk_weights_by_rank.append(topk_weights[offset:end].contiguous())
        offset = end

    _validate_plan(
        topk_ids_by_rank=topk_ids_by_rank,
        topk_weights_by_rank=topk_weights_by_rank,
        routed_num_experts=routed_num_experts,
        routed_topk=routed_topk,
        tokens_per_rank=tokens_per_rank,
        expected_expert_counts=expected_expert_counts[:routed_num_experts],
    )

    return RoutingPlan(
        distribution=distribution,
        source_policy=source_policy,
        global_num_tokens=global_num_tokens,
        local_topk_ids=topk_ids_by_rank[rank],
        local_topk_weights=topk_weights_by_rank[rank],
    )


def append_fused_shared_experts(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    routed_num_experts: int,
    num_fused_shared_experts: int,
    routed_scaling_factor: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Append SGLang-style fused shared expert slots to a routed top-k plan."""
    if num_fused_shared_experts == 0:
        return topk_ids.contiguous(), topk_weights.contiguous()
    if num_fused_shared_experts < 0:
        raise ValueError("num_fused_shared_experts must be non-negative")
    if routed_scaling_factor <= 0:
        raise ValueError("routed_scaling_factor must be positive")

    num_tokens = topk_ids.shape[0]
    shared_ids = torch.arange(
        routed_num_experts,
        routed_num_experts + num_fused_shared_experts,
        dtype=topk_ids.dtype,
        device=topk_ids.device,
    ).expand(num_tokens, num_fused_shared_experts)
    routed_sum = topk_weights.sum(dim=-1, keepdim=True).clamp_min(1e-20)
    normalized_routed_weights = topk_weights / routed_sum
    shared_weights = torch.full(
        (num_tokens, num_fused_shared_experts),
        1.0 / routed_scaling_factor,
        dtype=topk_weights.dtype,
        device=topk_weights.device,
    )
    return (
        torch.cat([topk_ids, shared_ids], dim=1).contiguous(),
        torch.cat([normalized_routed_weights, shared_weights], dim=1).contiguous(),
    )
