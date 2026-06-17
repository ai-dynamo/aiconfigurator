# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from argparse import Namespace
from unittest.mock import MagicMock

import pytest

_saved_mock = sys.modules.get("torch")
_restore_mock = isinstance(_saved_mock, MagicMock)
if _restore_mock:
    sys.modules.pop("torch")

try:
    import torch as _real_torch
except ImportError:
    if _restore_mock:
        sys.modules["torch"] = _saved_mock
    pytest.skip("real torch required for tensor operations", allow_module_level=True)

try:
    from collector.sglang.collect_dsv4_megamoe import build_cases
    from collector.sglang.dsv4_megamoe_workload import (
        _sampled_power_law_xmax,
        build_routing_plan,
        parse_distribution,
    )
finally:
    if _restore_mock:
        sys.modules["torch"] = _saved_mock

torch = _real_torch


@pytest.mark.unit
def test_random_routing_plan_is_deterministic_and_preserves_counts():
    ep_size = 4
    tokens_per_rank = [16] * ep_size
    routed_num_experts = 16
    routed_topk = 2

    plan = build_routing_plan(
        distribution="balanced",
        tokens_per_rank=tokens_per_rank,
        routed_num_experts=routed_num_experts,
        routed_topk=routed_topk,
        ep_size=ep_size,
        rank=0,
        source_policy="random",
        routing_seed=123,
    )
    plan_again = build_routing_plan(
        distribution="balanced",
        tokens_per_rank=tokens_per_rank,
        routed_num_experts=routed_num_experts,
        routed_topk=routed_topk,
        ep_size=ep_size,
        rank=0,
        source_policy="random",
        routing_seed=123,
    )

    assert plan.source_policy == "random"
    assert plan.global_num_tokens == sum(tokens_per_rank)
    assert tuple(plan.local_topk_ids.shape) == (tokens_per_rank[0], routed_topk)
    assert tuple(plan.local_topk_weights.shape) == (tokens_per_rank[0], routed_topk)
    assert torch.equal(plan.local_topk_ids, plan_again.local_topk_ids)
    assert torch.equal(plan.local_topk_weights, plan_again.local_topk_weights)
    assert torch.all(plan.local_topk_ids >= 0)
    assert torch.all(plan.local_topk_ids < routed_num_experts)


@pytest.mark.unit
def test_unknown_source_policy_is_rejected():
    with pytest.raises(ValueError, match="synthetic routing only supports source_policy=random"):
        build_routing_plan(
            distribution="power_law_1.01",
            tokens_per_rank=[16] * 4,
            routed_num_experts=16,
            routed_topk=2,
            ep_size=4,
            rank=0,
            source_policy="unsupported",
        )


@pytest.mark.unit
def test_sampled_power_law_builds_valid_routing_plan():
    plan = build_routing_plan(
        distribution="power_law_sampled_1.9",
        tokens_per_rank=[16] * 8,
        routed_num_experts=384,
        routed_topk=6,
        ep_size=8,
        rank=0,
        source_policy="random",
        routing_seed=123,
    )

    assert plan.distribution == "power_law_sampled_1.9"
    assert plan.global_num_tokens == 16 * 8
    assert tuple(plan.local_topk_ids.shape) == (16, 6)
    assert torch.all(plan.local_topk_ids >= 0)
    assert torch.all(plan.local_topk_ids < 384)


@pytest.mark.unit
def test_sampled_power_law_xmax_is_fixed_to_collected_default():
    assert _sampled_power_law_xmax(64) == pytest.approx(1024.0)
    assert _sampled_power_law_xmax(1024) == pytest.approx(1024.0)
    assert _sampled_power_law_xmax(131072) == pytest.approx(1024.0)


@pytest.mark.unit
def test_only_validated_sampled_power_law_distribution_is_accepted():
    assert parse_distribution("power_law_sampled_1.9").kind == "sampled_power_law"
    with pytest.raises(ValueError, match=r"only power_law_sampled_1\.9"):
        parse_distribution("power_law_sampled_1.2")


def _case_args(**overrides):
    args = Namespace(
        phases="context",
        prefill_tokens="8192",
        decode_tokens="",
        distributions="power_law_sampled_1.9",
        routing_seed=0,
        routing_seeds="",
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


@pytest.mark.unit
def test_build_cases_expands_sampled_power_law_to_ten_default_seeds():
    cases = build_cases(_case_args(), ep_size=16)

    assert [case.routing_seed for case in cases] == list(range(10))
    assert {case.distribution for case in cases} == {"power_law_sampled_1.9"}


@pytest.mark.unit
def test_build_cases_uses_explicit_routing_seeds_for_sampled_power_law():
    cases = build_cases(_case_args(routing_seed=100, routing_seeds="3,7"), ep_size=16)

    assert [case.routing_seed for case in cases] == [3, 7]


@pytest.mark.unit
def test_build_cases_keeps_single_seed_for_non_sampled_power_law():
    cases = build_cases(_case_args(distributions="power_law_1.2", routing_seed=5), ep_size=16)

    assert [case.routing_seed for case in cases] == [5]
