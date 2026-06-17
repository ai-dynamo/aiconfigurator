# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-branch candidate space. The branch enumeration calls the KV-feasibility
path, so it needs aiconfigurator + a perf DB (skips otherwise)."""

import pytest

pytest.importorskip("aiconfigurator")

from spica.config import SmartSearchConfig  # noqa: E402
from spica.kv_estimate import NoPerfDatabase  # noqa: E402
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
    except NoPerfDatabase:
        pytest.skip("no gb200/trtllm perf DB")
    except ValueError as exc:
        if "unsupported model/backend/GPU" in str(exc):
            pytest.skip(f"native KV build unavailable: {exc}")
        raise
    assert {b.deployment_mode for b in branches} == {"agg", "disagg"}
    for b in branches:
        assert b.backend == "trtllm"
        assert len(b.parallel_configs) > 0  # KV-feasible configs exist
        assert all(c.total_gpus <= 16 for c in b.parallel_configs)
        # planner + router knobs always present; engine knobs match the mode
        assert "planner_scaling_policy" in b.knob_choices
        key = "agg_max_num_seqs" if b.deployment_mode == "agg" else "decode_max_num_seqs"
        assert key in b.knob_choices
