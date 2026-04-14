# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# test_parallel_run.py injects a MagicMock as "torch" into sys.modules so
# collector code can be imported without CUDA.  This test needs real tensors,
# so evict the mock and attempt a real import.  If torch is not installed
# (CI Docker image), restore the mock and skip the entire module.
_saved_mock = None
if isinstance(sys.modules.get("torch"), MagicMock):
    _saved_mock = sys.modules.pop("torch")

try:
    import torch
except ImportError:
    if _saved_mock is not None:
        sys.modules["torch"] = _saved_mock
    pytest.skip("real torch required for tensor operations", allow_module_level=True)


def _import_helper_module():
    module_name = "collector.helper_test_copy"
    helper_path = Path(__file__).resolve().parents[3] / "collector" / "helper.py"
    spec = importlib.util.spec_from_file_location(module_name, helper_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.unit
def test_build_rank0_local_workload_masks_remote_experts():
    helper = _import_helper_module()

    rank0_info = {
        "rank0_logits": torch.tensor(
            [
                [0.5, 0.0, 0.5, 0.0],
                [0.0, 0.5, 0.0, 0.5],
            ],
            dtype=torch.float32,
        ),
        "rank0_selected_slots": torch.tensor(
            [
                [0, 2],
                [1, 3],
            ],
            dtype=torch.int64,
        ),
        "rank0_num_tokens": 2,
        "slots_per_rank": 2,
        "rank0_total_selections": 2,
    }

    workload = helper.build_rank0_local_workload(rank0_info)

    assert workload["num_tokens"] == 2
    assert torch.equal(workload["topk_ids"], torch.tensor([[0, -1], [1, -1]], dtype=torch.int32))
    assert torch.allclose(workload["topk_weights"], torch.tensor([[0.5, 0.0], [0.5, 0.0]], dtype=torch.float32))
    assert torch.equal(workload["masked_m"], torch.tensor([1, 1], dtype=torch.int32))


@pytest.mark.unit
def test_global_power_law_rank0_workload_keeps_global_distribution():
    helper = _import_helper_module()

    torch.manual_seed(0)
    _, rank0_info = helper.power_law_logits_v3(
        num_tokens=1024,
        num_experts=128,
        topk=8,
        ep=8,
        alpha=1.01,
        return_rank0_info=True,
    )

    workload = helper.build_rank0_local_workload(rank0_info)
    local_ids = workload["topk_ids"][workload["topk_ids"] >= 0]

    assert workload["num_tokens"] == rank0_info["rank0_num_tokens"]
    assert torch.all(local_ids < rank0_info["slots_per_rank"])
    assert torch.all(workload["topk_weights"][workload["topk_ids"] == -1] == 0)
    assert torch.all(workload["topk_weights"] >= 0)
    assert workload["masked_m"].sum().item() == rank0_info["rank0_total_selections"]
    assert (workload["topk_ids"] == -1).any()
    assert rank0_info["rank0_total_selections"] < workload["num_tokens"] * workload["topk_ids"].shape[1]
