# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Vizier-backed sampler. Needs the (pinned) vizier+jax stack; skips otherwise."""

import pytest

pytest.importorskip("vizier")

from spica.parallel_enum import ParallelShape, ReplicaParallelConfig  # noqa: E402
from spica.sampler import Suggestion, make_branch_sampler  # noqa: E402
from spica.search_space import BranchSpace  # noqa: E402


def _branch() -> BranchSpace:
    configs = tuple(ReplicaParallelConfig(ParallelShape(tp=4, dp=1, moe_tp=1, moe_ep=4), replicas=r) for r in (1, 2, 4))
    return BranchSpace(
        deployment_mode="agg",
        backend="trtllm",
        parallel_configs=configs,
        knob_choices={
            "router_mode": ["round_robin"],  # single choice -> constant, not a param
            "planner_scaling_policy": ["disabled", "throughput_180_5"],  # categorical
            "planner_fpm_sampling": ["default", "large"],
            "planner_load_sensitivity": ["default"],  # single -> constant
            "agg_max_num_batched_tokens": [8192, 16384],  # discrete int
            "agg_max_num_seqs": [256, 512, 1024],  # discrete int
            "overlap_score_credit": [0.0, 0.5, 1.0],  # discrete float (ignored under round_robin)
        },
    )


def test_suggest_produces_valid_selections():
    branch = _branch()
    sampler = make_branch_sampler(branch, study_id="test_valid")
    suggestions = sampler.suggest(count=4)
    assert len(suggestions) == 4
    for s in suggestions:
        assert isinstance(s, Suggestion)
        # constants injected; branch identity present
        assert s.selection["deployment_mode"] == "agg" and s.selection["backend"] == "trtllm"
        assert s.selection["router_mode"] == "round_robin"
        assert s.selection["planner_load_sensitivity"] == "default"
        # searched knobs land within their choice sets, native types preserved
        assert s.selection["planner_scaling_policy"] in {"disabled", "throughput_180_5"}
        assert s.selection["agg_max_num_seqs"] in {256, 512, 1024}
        assert isinstance(s.selection["agg_max_num_seqs"], int)
        assert s.selection["overlap_score_credit"] in {0.0, 0.5, 1.0}
        # the chosen parallel config is one of the branch's
        assert s.parallel_config in branch.parallel_configs


def test_suggest_observe_round_trips():
    # Verify the ask/tell round-trip feeds back without error and the study
    # tracks the best observed score. (Convergence quality isn't asserted —
    # Vizier GP-bandit is slow, ~seconds per suggest, so keep trial counts low.)
    branch = _branch()
    sampler = make_branch_sampler(branch, study_id="test_round_trip")
    scores = []
    for _ in range(2):
        for s in sampler.suggest(count=2):
            score = float(s.selection["agg_max_num_seqs"])
            scores.append(score)
            sampler.observe(s, score)
    assert len(scores) == 4
    best = list(sampler._study.optimal_trials())[0].materialize()
    assert best.final_measurement.metrics["objective"].value == max(scores)
