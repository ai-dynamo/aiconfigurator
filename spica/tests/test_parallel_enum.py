# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from spica.parallel_enum import (
    enumerate_disagg_configs,
    enumerate_parallel_configs,
    enumerate_worker_shapes,
)


def _tuples(shapes):
    return {(s.tp, s.dp, s.moe_tp, s.moe_ep) for s in shapes}


def test_dense_shape_is_plain_tp():
    shapes = enumerate_worker_shapes(is_moe=False, mla=False, backend="vllm", gpus_per_worker=4)
    assert _tuples(shapes) == {(4, 1, 1, 1)}
    s = shapes[0]
    assert s.pp == 1
    assert s.gpus_per_worker == 4
    assert s.strategy == "tp"


def test_moe_non_mla_gives_tep_dep_and_pure_tp():
    shapes = enumerate_worker_shapes(is_moe=True, mla=False, backend="vllm", gpus_per_worker=4)
    # TEP (4,1,1,4), DEP (1,4,1,4), pure-TP (4,1,4,1)
    assert _tuples(shapes) == {(4, 1, 1, 4), (1, 4, 1, 4), (4, 1, 4, 1)}
    assert {s.strategy for s in shapes} == {"tep", "dep", "tp"}
    for s in shapes:  # width constraint dp*tp == moe_tp*moe_ep == gpus_per_worker
        assert s.dp * s.tp == s.moe_tp * s.moe_ep == 4
        assert s.gpus_per_worker == 4


def test_moe_mla_drops_pure_tp():
    shapes = enumerate_worker_shapes(is_moe=True, mla=True, backend="vllm", gpus_per_worker=4)
    assert _tuples(shapes) == {(4, 1, 1, 4), (1, 4, 1, 4)}  # TEP, DEP only
    assert all(s.strategy in {"tep", "dep"} for s in shapes)


def test_sglang_wideep_forbids_pure_tp():
    shapes = enumerate_worker_shapes(is_moe=True, mla=False, backend="sglang", gpus_per_worker=4, enable_wideep=True)
    assert _tuples(shapes) == {(4, 1, 1, 4), (1, 4, 1, 4)}  # moe_tp>1 dropped


def test_moe_has_no_shape_below_two_gpus():
    assert enumerate_worker_shapes(is_moe=True, mla=False, backend="vllm", gpus_per_worker=1) == []


def test_replica_iteration_within_budget():
    cfgs = enumerate_parallel_configs(is_moe=False, mla=False, backend="vllm", gpu_budget=8)
    by_g: dict[int, set[int]] = {}
    for c in cfgs:
        by_g.setdefault(c.shape.gpus_per_worker, set()).add(c.replicas)
    assert by_g[8] == {1}
    assert by_g[4] == {1, 2}
    assert by_g[2] == {1, 2, 3, 4}
    assert by_g[1] == set(range(1, 9))
    assert all(c.total_gpus <= 8 for c in cfgs)


def test_min_gpus_per_worker_floor():
    # memory-fit floor: workers smaller than 4 GPUs are excluded
    cfgs = enumerate_parallel_configs(is_moe=False, mla=False, backend="vllm", gpu_budget=8, min_gpus_per_worker=4)
    assert {c.shape.gpus_per_worker for c in cfgs} == {4, 8}


def test_min_gpu_budget_floor():
    cfgs = enumerate_parallel_configs(is_moe=False, mla=False, backend="vllm", gpu_budget=8, min_gpu_budget=4)
    assert all(4 <= c.total_gpus <= 8 for c in cfgs)
    g2_replicas = sorted(c.replicas for c in cfgs if c.shape.gpus_per_worker == 2)
    assert g2_replicas == [2, 3, 4]  # total 4,6,8


def test_moe_mla_budget_only_ep_sharding():
    cfgs = enumerate_parallel_configs(is_moe=True, mla=True, backend="trtllm", gpu_budget=16)
    assert cfgs
    assert all(c.total_gpus <= 16 for c in cfgs)
    assert all(c.shape.moe_tp == 1 for c in cfgs)  # MLA: no pure expert-TP


def test_disagg_pairs_share_budget():
    cfgs = enumerate_disagg_configs(is_moe=True, mla=True, backend="trtllm", gpu_budget=8)
    assert len(cfgs) == 44
    for c in cfgs:
        assert c.total_gpus == c.prefill.total_gpus + c.decode.total_gpus <= 8
        assert c.prefill.total_gpus >= 2 and c.decode.total_gpus >= 2  # each role >= 1 worker
    # prefill and decode may differ in shape
    assert any(c.prefill.shape != c.decode.shape for c in cfgs)


def test_disagg_min_gpu_budget_full_utilization():
    cfgs = enumerate_disagg_configs(is_moe=True, mla=True, backend="trtllm", gpu_budget=8, min_gpu_budget=8)
    assert len(cfgs) == 24
    assert all(c.total_gpus == 8 for c in cfgs)


def test_disagg_dense_small_budget():
    cfgs = enumerate_disagg_configs(is_moe=False, mla=False, backend="vllm", gpu_budget=4)
    assert cfgs
    assert all(c.total_gpus <= 4 for c in cfgs)
    assert all(c.prefill.shape.strategy == "tp" for c in cfgs)
