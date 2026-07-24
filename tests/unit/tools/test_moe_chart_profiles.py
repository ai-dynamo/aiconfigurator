# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from tools.sanity_check.moe_chart_profiles import MoeChartProfile, select_moe_chart_profiles

pytestmark = pytest.mark.unit


def _add_profile(data, profile, *, distribution="balanced", tp=1, ep=4, tokens=(1, 16, 32)):
    data.setdefault(distribution, {}).setdefault(profile.topk, {}).setdefault(profile.num_experts, {}).setdefault(
        profile.hidden_size, {}
    ).setdefault(profile.inter_size, {}).setdefault(tp, {})[ep] = dict.fromkeys(tokens, 1.0)


def test_prefers_historical_profile_when_available():
    preferred = MoeChartProfile(topk=8, num_experts=256, hidden_size=7168, inter_size=2048)
    fallback = MoeChartProfile(topk=6, num_experts=256, hidden_size=4096, inter_size=2048)
    data = {}
    _add_profile(data, fallback)
    _add_profile(data, preferred, tokens=(1,))

    profiles = select_moe_chart_profiles(
        data,
        workload_distribution="balanced",
        moe_tp_size=1,
        moe_ep_size=4,
        target_tokens=(1, 16, 32),
        preferred=preferred,
    )

    assert profiles == [preferred]


def test_selects_both_dsv4_profiles_when_preferred_is_absent():
    flash = MoeChartProfile(topk=6, num_experts=256, hidden_size=4096, inter_size=2048)
    pro = MoeChartProfile(topk=6, num_experts=384, hidden_size=7168, inter_size=3072)
    data = {}
    _add_profile(data, pro)
    _add_profile(data, flash)

    profiles = select_moe_chart_profiles(
        data,
        workload_distribution="balanced",
        moe_tp_size=1,
        moe_ep_size=4,
        target_tokens=(1, 16, 32),
        preferred=MoeChartProfile(topk=8, num_experts=256, hidden_size=7168, inter_size=2048),
    )

    assert profiles == [flash, pro]


def test_filters_topology_and_ranks_by_target_token_coverage():
    best = MoeChartProfile(topk=6, num_experts=256, hidden_size=4096, inter_size=2048)
    second = MoeChartProfile(topk=6, num_experts=384, hidden_size=7168, inter_size=3072)
    wrong_distribution = MoeChartProfile(topk=4, num_experts=128, hidden_size=2880, inter_size=2880)
    wrong_topology = MoeChartProfile(topk=2, num_experts=64, hidden_size=2048, inter_size=4096)
    data = {}
    _add_profile(data, second, tokens=(1, 16, 999))
    _add_profile(data, best, tokens=(1, 16, 32))
    _add_profile(data, wrong_distribution, distribution="uniform")
    _add_profile(data, wrong_topology, tp=2, ep=2)

    profiles = select_moe_chart_profiles(
        data,
        workload_distribution="balanced",
        moe_tp_size=1,
        moe_ep_size=4,
        target_tokens=(1, 16, 32),
    )

    assert profiles == [best, second]
