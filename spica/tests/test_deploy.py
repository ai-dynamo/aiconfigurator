# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""sample -> deployment payload translation (pure; no dynamo import).

Feeds real ``unroll_sample`` output into ``build_deployment`` so the field names
the two modules agree on are exercised."""

from spica.config import SearchSpace, SLATarget
from spica.deploy import build_deployment
from spica.load_predictor_sweep import LoadPredictorResult
from spica.parallel_enum import DisaggParallelConfig, ParallelShape, ReplicaParallelConfig
from spica.sample import unroll_sample

BV = "1.3.0rc10"  # backend_version (normally from kv_estimate.resolve_backend_version)


def _space(**ov):
    base = {"model_name": "deepseek-ai/DeepSeek-V3", "hardware_sku": "gb200"}
    base.update(ov)
    return SearchSpace(**base)


def _agg_sel(**ov):
    sel = {
        "deployment_mode": "agg",
        "backend": "trtllm",
        "agg_max_num_batched_tokens": 16384,
        "agg_max_num_seqs": 512,
        "router_mode": "round_robin",
        "planner_scaling_policy": "disabled",
        "planner_fpm_sampling": "default",
        "planner_load_sensitivity": "default",
    }
    sel.update(ov)
    return sel


AGG_MOE = ReplicaParallelConfig(ParallelShape(tp=4, dp=1, moe_tp=1, moe_ep=4), replicas=2)


def test_agg_static_disabled_uses_plain_path():
    sample = unroll_sample(search_space=_space(), selection=_agg_sel(), parallel_config=AGG_MOE)
    plan = build_deployment(sample, backend_version=BV)
    assert plan.deployment_mode == "agg" and plan.is_static
    assert plan.planner_config is None  # disabled -> plain replay
    assert plan.num_workers == 2
    ea = plan.agg_engine_args
    assert ea["aic_backend"] == "trtllm" and ea["aic_backend_version"] == BV
    assert ea["aic_model_path"] == "deepseek-ai/DeepSeek-V3" and ea["aic_system"] == "gb200"
    assert ea["aic_tp_size"] == 4 and ea["aic_attention_dp_size"] == 1
    assert ea["aic_moe_tp_size"] == 1 and ea["aic_moe_ep_size"] == 4  # MoE shape
    assert ea["max_num_seqs"] == 512 and ea["worker_type"] == "aggregated"
    assert "num_gpu_blocks" not in ea  # replay estimates it


def test_agg_scaling_builds_planner_config():
    lp = LoadPredictorResult(best_by_interval={180: "prophet_w20_log1p"}, reason="swept")
    sample = unroll_sample(
        search_space=_space(),
        selection=_agg_sel(planner_scaling_policy="throughput_180_5"),
        parallel_config=AGG_MOE,
        load_predictor=lp,
    )
    plan = build_deployment(sample, backend_version=BV, planner_sla=SLATarget(ttft_ms=2000.0, itl_ms=30.0))
    assert not plan.is_static
    pc = plan.planner_config
    assert pc["mode"] == "agg"
    assert pc["optimization_target"] == "sla"  # throughput scaling -> sla target
    assert pc["enable_throughput_scaling"] is True
    assert pc["throughput_adjustment_interval_seconds"] == 180
    assert pc["decode_engine_num_gpu"] == 4  # tp(4) * attention_dp(1)
    assert pc["load_predictor"] == "prophet"  # resolved from the sweep winner
    assert pc["ttft_ms"] == 2000.0 and pc["itl_ms"] == 30.0  # planner SLA seeded from goal


def test_disagg_builds_both_roles():
    cfg = DisaggParallelConfig(
        prefill=ReplicaParallelConfig(ParallelShape(tp=8, dp=1, moe_tp=1, moe_ep=8), 1),
        decode=ReplicaParallelConfig(ParallelShape(tp=1, dp=8, moe_tp=1, moe_ep=8), 2),
    )
    sel = _agg_sel(
        deployment_mode="disagg",
        planner_scaling_policy="load_180_5",
        prefill_max_num_batched_tokens=32768,
        prefill_max_num_seqs=4,
        decode_max_num_batched_tokens=8192,
        decode_max_num_seqs=1024,
    )
    sample = unroll_sample(search_space=_space(), selection=sel, parallel_config=cfg)
    plan = build_deployment(sample, backend_version=BV)
    assert plan.deployment_mode == "disagg"
    assert plan.num_prefill_workers == 1 and plan.num_decode_workers == 2
    assert plan.prefill_engine_args["aic_tp_size"] == 8 and plan.prefill_engine_args["worker_type"] == "prefill"
    assert plan.decode_engine_args["aic_attention_dp_size"] == 8 and plan.decode_engine_args["worker_type"] == "decode"
    # load policy -> load target; engine GPU counts per role
    assert plan.planner_config["optimization_target"] == "load"
    assert plan.planner_config["prefill_engine_num_gpu"] == 8 and plan.planner_config["decode_engine_num_gpu"] == 8


def test_kv_router_emits_router_config():
    sel = _agg_sel(
        router_mode="kv_router",
        overlap_score_credit=0.5,
        prefill_load_scale=1.0,
        host_cache_hit_weight=0.75,
        disk_cache_hit_weight=0.25,
        router_temperature=0.2,
    )
    sample = unroll_sample(search_space=_space(), selection=sel, parallel_config=AGG_MOE)
    plan = build_deployment(sample, backend_version=BV)
    assert plan.router_mode == "kv_router"
    assert plan.router_config["overlap_score_credit"] == 0.5 and plan.router_config["router_temperature"] == 0.2

    rr = build_deployment(
        unroll_sample(search_space=_space(), selection=_agg_sel(), parallel_config=AGG_MOE), backend_version=BV
    )
    assert rr.router_config is None  # round_robin


def test_dense_shape_omits_moe_sizes():
    dense = ReplicaParallelConfig(ParallelShape(tp=2, dp=1, moe_tp=1, moe_ep=1), replicas=1)
    sample = unroll_sample(
        search_space=_space(model_name="Qwen/Qwen3-32B"), selection=_agg_sel(), parallel_config=dense
    )
    ea = build_deployment(sample, backend_version=BV).agg_engine_args
    assert ea["aic_tp_size"] == 2
    assert "aic_moe_tp_size" not in ea and "aic_moe_ep_size" not in ea  # dense
