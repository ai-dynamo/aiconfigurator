# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the large-EP MoE comm-backend registry (operations/moe_comm.py)."""

import pytest

from aiconfigurator.sdk import common
from aiconfigurator_core.sdk.operations.moe_comm import MOE_A2A_BACKENDS, nodes_for

pytestmark = pytest.mark.unit


def test_perf_data_filename_enum_entries():
    assert common.PerfDataFilename.moe_a2a.value == "moe_a2a_perf.parquet"


def test_registry_has_all_framework_backend_identities():
    assert set(MOE_A2A_BACKENDS) == {
        "deepep_ht",
        "deepep_ll",
        "deepep_v2_context",
        "deepep_v2_generation",
        "trtllm_deepep_ht",
        "trtllm_deepep_ll",
        "nvlink_two_sided",
        "nvlink_one_sided",
    }


@pytest.mark.parametrize(
    ("name", "frameworks", "inference_phases", "comm_phases", "min_sm"),
    [
        ("deepep_ht", ("sglang", "vllm"), ("context",), ("dispatch", "combine"), 0),
        ("deepep_ll", ("sglang", "vllm"), ("generation",), ("dispatch", "combine"), 0),
        ("deepep_v2_context", ("vllm",), ("context",), ("dispatch", "combine"), 0),
        ("deepep_v2_generation", ("vllm",), ("generation",), ("dispatch", "combine"), 0),
        ("trtllm_deepep_ht", ("trtllm",), ("context",), ("dispatch", "combine"), 0),
        ("trtllm_deepep_ll", ("trtllm",), ("generation",), ("dispatch", "combine"), 0),
        ("nvlink_two_sided", ("trtllm",), ("context", "generation"), ("prepare", "dispatch", "combine"), 100),
        ("nvlink_one_sided", ("trtllm",), ("context", "generation"), ("dispatch", "combine"), 100),
    ],
)
def test_backend_spec_table(name, frameworks, inference_phases, comm_phases, min_sm):
    spec = MOE_A2A_BACKENDS[name]
    assert spec.name == name
    assert spec.frameworks == frameworks
    assert spec.inference_phases == inference_phases
    assert spec.comm_phases == comm_phases
    assert spec.min_sm == min_sm
    assert spec.max_topk == 8


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        # All rules satisfied.
        (dict(topk=8, num_experts=128, moe_tp_size=1, moe_ep_size=16), True),
        # topk=1 passes as well.
        (dict(topk=1, num_experts=128, moe_tp_size=1, moe_ep_size=16), True),
        # topk above max_topk (8).
        (dict(topk=16, num_experts=128, moe_tp_size=1, moe_ep_size=16), False),
        # moe_tp_size must be 1.
        (dict(topk=8, num_experts=128, moe_tp_size=2, moe_ep_size=16), False),
        # moe_ep_size must be > 1.
        (dict(topk=8, num_experts=128, moe_tp_size=1, moe_ep_size=1), False),
        # num_experts must be divisible by moe_ep_size (100 % 16 != 0).
        (dict(topk=8, num_experts=100, moe_tp_size=1, moe_ep_size=16), False),
        # moe_ep_size must not exceed num_experts.
        (dict(topk=8, num_experts=128, moe_tp_size=1, moe_ep_size=256), False),
    ],
)
def test_feasible_rules(kwargs, expected):
    assert MOE_A2A_BACKENDS["deepep_ht"].feasible(**kwargs) is expected


@pytest.mark.parametrize("name", ["nvlink_two_sided", "nvlink_one_sided"])
@pytest.mark.parametrize(("sm_version", "expected"), [(90, False), (100, True)])
def test_feasible_min_sm(name, sm_version, expected):
    spec = MOE_A2A_BACKENDS[name]
    assert spec.feasible(topk=8, num_experts=128, moe_tp_size=1, moe_ep_size=16, sm_version=sm_version) is expected


@pytest.mark.parametrize(
    ("ep_size", "num_gpus_per_node", "expected"),
    [(16, 8, 2), (12, 8, 2), (8, 8, 1), (32, 4, 8)],
)
def test_nodes_for(ep_size, num_gpus_per_node, expected):
    assert nodes_for(ep_size, num_gpus_per_node) == expected
