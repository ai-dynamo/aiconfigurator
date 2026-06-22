# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-branch candidate space. The branch enumeration calls the KV-feasibility
path, so it needs aiconfigurator + a perf DB (skips otherwise)."""

import pytest

pytest.importorskip("aiconfigurator")

from spica.config import SmartSearchConfig  # noqa: E402
from spica.kv_estimate import NoPerfDatabase  # noqa: E402
from spica.model_hw import NoViableParallelConfig  # noqa: E402
from spica.search_space import branch_knob_choices, enumerate_branches  # noqa: E402


def _config(**ss_overrides) -> SmartSearchConfig:
    ss = {
        "model_name": "deepseek-ai/DeepSeek-V3",
        "hardware_sku": "gb200",
        "backend": ["trtllm"],
        "deployment_mode": ["agg"],
        "gpu_budget": 16,
    }
    ss.update(ss_overrides)
    return SmartSearchConfig(search_space=ss, workload={"trace_path": "/tmp/t.jsonl"})


def test_branch_knob_choices_by_mode():
    ss = _config().search_space
    agg = branch_knob_choices(ss, "agg")
    assert "agg_max_num_seqs" in agg and "prefill_max_num_seqs" not in agg
    assert "router_mode" in agg and "planner_scaling_policy" in agg
    disagg = branch_knob_choices(ss, "disagg")
    assert "prefill_max_num_seqs" in disagg and "decode_max_num_seqs" in disagg
    assert "agg_max_num_seqs" not in disagg


def test_enumerate_branches_deepseek_gb200():
    cfg = _config(deployment_mode=["agg", "disagg"], backend=["trtllm"], gpu_budget=16)
    try:
        branches = enumerate_branches(cfg)
    except (NoPerfDatabase, NoViableParallelConfig):
        pytest.skip("no gb200/trtllm perf DB")
    except ValueError as exc:
        if "unsupported model/backend/GPU" in str(exc):
            pytest.skip(f"native KV build unavailable: {exc}")
        raise
    # one branch per deployment_mode (backend is a searched knob, not a branch)
    assert {b.deployment_mode for b in branches} == {"agg", "disagg"}
    for b in branches:
        assert b.knob_choices["backend"] == ["trtllm"]  # only the viable backend(s)
        assert len(b.parallel_configs) > 0  # KV-feasible configs exist
        assert all(c.total_gpus <= 16 for c in b.parallel_configs)
        # every config is tagged with the backends that support it
        assert all(b.supported_backends[c] == frozenset({"trtllm"}) for c in b.parallel_configs)
        # planner + router knobs always present; engine knobs match the mode
        assert "planner_scaling_policy" in b.knob_choices
        key = "agg_max_num_seqs" if b.deployment_mode == "agg" else "decode_max_num_seqs"
        assert key in b.knob_choices


def test_pinned_parallel_configs_replace_the_menu():
    # a dense, KV-trivial model so the pinned shapes are guaranteed feasible
    cfg = _config(
        model_name="meta-llama/Meta-Llama-3.1-8B",
        deployment_mode=["agg"],
        gpu_budget=32,
        parallel_configs=[{"tp": 4, "replicas": 2}, {"tp": 8, "replicas": 1}],
    )
    try:
        branches = enumerate_branches(cfg)
    except NoPerfDatabase:
        pytest.skip("no gb200/trtllm perf DB")
    (branch,) = branches
    menu = {(c.shape.tp, c.replicas) for c in branch.parallel_configs}
    assert menu == {(4, 2), (8, 1)}  # exactly the pinned set, not the full enumeration
    assert all(c.total_gpus == 8 for c in branch.parallel_configs)


def test_pinned_parallel_config_illegal_is_rejected():
    cfg = _config(
        model_name="meta-llama/Meta-Llama-3.1-8B",
        deployment_mode=["agg"],
        gpu_budget=32,
        parallel_configs=[{"tp": 3, "replicas": 1}],  # tp=3 not on the GPU ladder
    )
    try:
        with pytest.raises(NoViableParallelConfig):
            enumerate_branches(cfg)
    except NoPerfDatabase:
        pytest.skip("no gb200/trtllm perf DB")
