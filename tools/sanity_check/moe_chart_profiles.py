# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Select queryable MoE shapes for the database sanity-check chart."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True, order=True)
class MoeChartProfile:
    topk: int
    num_experts: int
    hidden_size: int
    inter_size: int

    @property
    def label(self) -> str:
        return f"k={self.topk} e={self.num_experts} h={self.hidden_size} i={self.inter_size}"


def _mapping_at(mapping: Mapping[object, object], key: object) -> Mapping[object, object]:
    value = mapping.get(key, {})
    return value if isinstance(value, Mapping) else {}


def select_moe_chart_profiles(
    quant_data: Mapping[object, object],
    *,
    workload_distribution: str,
    moe_tp_size: int,
    moe_ep_size: int,
    target_tokens: Sequence[int],
    preferred: MoeChartProfile | None = None,
    max_profiles: int = 2,
) -> list[MoeChartProfile]:
    """Return available shapes for one distribution and TP/EP topology.

    The historical chart shape remains the sole choice when it is available.
    Otherwise, rank real table shapes by coverage of the chart's token probes
    and return up to ``max_profiles`` deterministic fallbacks.
    """
    if max_profiles <= 0:
        return []

    distribution_data = _mapping_at(quant_data, workload_distribution)
    candidates: list[tuple[int, int, MoeChartProfile]] = []
    for topk, topk_data in distribution_data.items():
        if not isinstance(topk_data, Mapping):
            continue
        for num_experts, expert_data in topk_data.items():
            if not isinstance(expert_data, Mapping):
                continue
            for hidden_size, hidden_data in expert_data.items():
                if not isinstance(hidden_data, Mapping):
                    continue
                for inter_size, inter_data in hidden_data.items():
                    if not isinstance(inter_data, Mapping):
                        continue
                    tp_data = _mapping_at(inter_data, moe_tp_size)
                    token_data = _mapping_at(tp_data, moe_ep_size)
                    if not token_data:
                        continue
                    profile = MoeChartProfile(
                        topk=int(topk),
                        num_experts=int(num_experts),
                        hidden_size=int(hidden_size),
                        inter_size=int(inter_size),
                    )
                    target_coverage = sum(token in token_data for token in target_tokens)
                    candidates.append((target_coverage, len(token_data), profile))

    if preferred is not None and any(profile == preferred for _, _, profile in candidates):
        return [preferred]

    candidates.sort(key=lambda item: (-item[0], -item[1], item[2]))
    return [profile for _, _, profile in candidates[:max_profiles]]
