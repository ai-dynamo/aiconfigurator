# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SGLang DeepSeek-V4 MegaMoE target-EP local-compute collector.

This module is intentionally not registered as a default SGLang collector yet.
It runs a single-rank DeepGEMM MegaMoE kernel with the target EP rank-local
expert count and target-EP bottleneck-rank token workload.  Local pre-dispatch
is measured separately with a source-rank-local token workload because real
MegaMoE pre-dispatch happens before EP exchange.  The runtime has no cross-rank
communication because ``num_ranks == 1``; ``moe_ep_size`` in the task remains
the target EP size recorded for AIC modeling.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import traceback
from dataclasses import dataclass
from inspect import Parameter, signature
from pathlib import Path
from typing import Any

try:
    from helper import benchmark_with_power, log_perf
except ModuleNotFoundError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from helper import benchmark_with_power, log_perf

try:
    from aiconfigurator.sdk.dsv4_megamoe import (
        Dsv4MegaMoEWorkload,
        build_dsv4_power_law_megamoe_workload_from_global_tokens,
        build_dsv4_uniform_megamoe_workload_from_global_tokens,
    )
except ModuleNotFoundError:
    _REPO_ROOT = Path(__file__).resolve().parents[2]
    sys.path.append(str(_REPO_ROOT / "src"))
    from aiconfigurator.sdk.dsv4_megamoe import (
        Dsv4MegaMoEWorkload,
        build_dsv4_power_law_megamoe_workload_from_global_tokens,
        build_dsv4_uniform_megamoe_workload_from_global_tokens,
    )

try:
    from collector.sglang.collect_dsv4_megamoe import (
        DEFAULT_DSV4_HIDDEN_SIZE,
        DEFAULT_DSV4_MOE_INTER_SIZE,
        DEFAULT_DSV4_NUM_EXPERTS,
        DEFAULT_DSV4_TOPK,
        DEFAULT_MEGAMOE_TOKEN_CAP,
        Dsv4MegaMoETask,
        _apply_dsv4_megamoe_env_defaults,
        _base_report,
        _find_megamoe_layer,
        _load_model_runner,
        _parse_bool_env,
        _report_path,
        _safe_version,
        _write_json,
        dsv4_megamoe_env_defaults,
    )
except ModuleNotFoundError:
    from collect_dsv4_megamoe import (
        DEFAULT_DSV4_HIDDEN_SIZE,
        DEFAULT_DSV4_MOE_INTER_SIZE,
        DEFAULT_DSV4_NUM_EXPERTS,
        DEFAULT_DSV4_TOPK,
        DEFAULT_MEGAMOE_TOKEN_CAP,
        Dsv4MegaMoETask,
        _apply_dsv4_megamoe_env_defaults,
        _base_report,
        _find_megamoe_layer,
        _load_model_runner,
        _parse_bool_env,
        _report_path,
        _safe_version,
        _write_json,
        dsv4_megamoe_env_defaults,
    )


PERF_FILENAME = "dsv4_megamoe_compute_perf.txt"
REPORT_SCHEMA_VERSION = "aic-dsv4-megamoe-compute-v1"
DEFAULT_POWER_LAW_ALPHA = 1.2
SUPPORTED_DISTRIBUTIONS = {"uniform", "power_law"}
DEFAULT_TARGET_EP_LIST = [1, 2, 4, 8, 16, 32, 64, 128, 256]


@dataclass(frozen=True)
class Dsv4MegaMoEComputeTask:
    moe_type: str
    num_tokens: int
    hidden_size: int
    inter_size: int
    topk: int
    num_experts: int
    moe_tp_size: int
    moe_ep_size: int
    model_name: str
    perf_filename: str
    distribution: str
    power_law_alpha: float | None

    def as_params(self) -> list[Any]:
        return [
            self.moe_type,
            self.num_tokens,
            self.hidden_size,
            self.inter_size,
            self.topk,
            self.num_experts,
            self.moe_tp_size,
            self.moe_ep_size,
            self.model_name,
            self.perf_filename,
            self.distribution,
            self.power_law_alpha,
        ]


def dsv4_megamoe_compute_env_defaults() -> dict[str, str]:
    return dsv4_megamoe_env_defaults()


def _compute_report_path(task: Dsv4MegaMoEComputeTask, device_id: int, suffix: str) -> Path:
    path = _report_path(_probe_compatible_task(task), device_id, suffix=suffix)
    safe_distribution = task.distribution.replace("/", "_").replace(":", "_")
    name = (
        f"dsv4_megamoe_compute_{suffix}_{task.num_tokens}_"
        f"ep{task.moe_ep_size}_{safe_distribution}_gpu{device_id}.json"
    )
    return path.with_name(name)


def _probe_compatible_task(task: Dsv4MegaMoEComputeTask):
    return Dsv4MegaMoETask(
        task.moe_type,
        task.num_tokens,
        task.hidden_size,
        task.inter_size,
        task.topk,
        task.num_experts,
        task.moe_tp_size,
        task.moe_ep_size,
        task.model_name,
        task.perf_filename,
        task.distribution,
        task.power_law_alpha,
    )


def _base_compute_report(task: Dsv4MegaMoEComputeTask, device_id: int, status: str) -> dict[str, Any]:
    report = _base_report(_probe_compatible_task(task), device_id, status=status)
    report["schema"] = REPORT_SCHEMA_VERSION
    return report


def get_dsv4_megamoe_compute_test_cases() -> list[list[Any]]:
    if _parse_bool_env("AIC_DSV4_MEGAMOE_FULL", default=False):
        num_tokens_list = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    else:
        num_tokens_list = [1, 8, 32, 128, 512, 1024]

    raw_distributions = os.environ.get("AIC_DSV4_MEGAMOE_COMPUTE_DISTRIBUTIONS")
    if raw_distributions:
        distributions = [item.strip() for item in raw_distributions.split(",") if item.strip()]
    else:
        distributions = ["uniform", "power_law"]

    raw_ep_list = os.environ.get("AIC_DSV4_MEGAMOE_TARGET_EP_LIST")
    if raw_ep_list:
        target_ep_list = [int(item.strip()) for item in raw_ep_list.split(",") if item.strip()]
    else:
        target_ep_list = DEFAULT_TARGET_EP_LIST

    tasks: list[Dsv4MegaMoEComputeTask] = []
    for distribution in distributions:
        if distribution not in SUPPORTED_DISTRIBUTIONS:
            raise ValueError(f"Unsupported DSv4 MegaMoE compute distribution: {distribution}")
        for target_ep_size in target_ep_list:
            if target_ep_size <= 0:
                raise ValueError(f"Invalid DSv4 MegaMoE target EP size: {target_ep_size}")
            if DEFAULT_DSV4_NUM_EXPERTS % target_ep_size != 0:
                continue
            for num_tokens in num_tokens_list:
                tasks.append(
                    Dsv4MegaMoEComputeTask(
                        moe_type="mxfp4",
                        num_tokens=num_tokens,
                        hidden_size=DEFAULT_DSV4_HIDDEN_SIZE,
                        inter_size=DEFAULT_DSV4_MOE_INTER_SIZE,
                        topk=DEFAULT_DSV4_TOPK,
                        num_experts=DEFAULT_DSV4_NUM_EXPERTS,
                        moe_tp_size=1,
                        moe_ep_size=target_ep_size,
                        model_name="deepseek-ai/DeepSeek-V4",
                        perf_filename=PERF_FILENAME,
                        distribution=distribution,
                        power_law_alpha=DEFAULT_POWER_LAW_ALPHA if distribution == "power_law" else None,
                    )
                )
    return [task.as_params() for task in tasks]


def _coerce_compute_task(
    moe_type: str,
    num_tokens: int,
    hidden_size: int,
    inter_size: int,
    topk: int,
    num_experts: int,
    moe_tp_size: int,
    moe_ep_size: int,
    model_name: str,
    perf_filename: str,
    distribution: str,
    power_law_alpha: float | None,
) -> Dsv4MegaMoEComputeTask:
    task = Dsv4MegaMoEComputeTask(
        moe_type=str(moe_type),
        num_tokens=int(num_tokens),
        hidden_size=int(hidden_size),
        inter_size=int(inter_size),
        topk=int(topk),
        num_experts=int(num_experts),
        moe_tp_size=int(moe_tp_size),
        moe_ep_size=int(moe_ep_size),
        model_name=str(model_name),
        perf_filename=str(perf_filename),
        distribution=str(distribution),
        power_law_alpha=None if power_law_alpha is None else float(power_law_alpha),
    )
    if task.moe_type != "mxfp4":
        raise ValueError(f"Unsupported DSv4 MegaMoE compute dtype: {task.moe_type}")
    if task.num_tokens <= 0:
        raise ValueError("num_tokens must be positive")
    if task.moe_tp_size != 1:
        raise ValueError("DSv4 MegaMoE local-compute collector requires moe_tp_size == 1")
    if task.moe_ep_size <= 0:
        raise ValueError("moe_ep_size must be positive")
    if task.num_experts % task.moe_ep_size != 0:
        raise ValueError("num_experts must be divisible by target moe_ep_size")
    if task.distribution not in SUPPORTED_DISTRIBUTIONS:
        raise ValueError(f"Unsupported DSv4 MegaMoE compute distribution: {task.distribution}")
    if task.distribution == "power_law" and task.power_law_alpha is None:
        raise ValueError("power_law distribution requires power_law_alpha")

    token_cap = int(
        os.environ.get(
            "SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK",
            DEFAULT_MEGAMOE_TOKEN_CAP,
        )
    )
    if task.num_tokens > token_cap:
        raise ValueError("num_tokens exceeds MegaMoE token cap")
    return task


def _local_expert_indices(
    *,
    routed_num_experts: int,
    target_moe_ep_size: int,
    num_fused_shared_experts: int,
) -> tuple[int, ...]:
    if routed_num_experts % target_moe_ep_size != 0:
        raise ValueError("routed_num_experts must be divisible by target_moe_ep_size")
    local_routed_experts = routed_num_experts // target_moe_ep_size
    routed_indices = tuple(range(local_routed_experts))
    shared_indices = tuple(routed_num_experts + idx for idx in range(num_fused_shared_experts))
    return routed_indices + shared_indices


def _build_local_mega_weight_pairs(
    experts: Any,
    expert_indices: tuple[int, ...],
) -> tuple[tuple[Any, Any], tuple[Any, Any]]:
    import torch

    from deep_gemm import transform_sf_into_required_layout, transform_weights_for_mega_moe

    required_attrs = ("w13_weight", "w13_weight_scale_inv", "w2_weight", "w2_weight_scale_inv")
    missing = [name for name in required_attrs if not hasattr(experts, name)]
    if missing:
        raise RuntimeError(f"SGLang FusedMoE experts are missing required raw weight attrs: {missing}")
    if not expert_indices:
        raise ValueError("expert_indices must be non-empty")

    index = torch.tensor(expert_indices, dtype=torch.long, device=experts.w13_weight.data.device)
    w13 = experts.w13_weight.data.index_select(0, index).contiguous()
    w13_sf_fp32 = experts.w13_weight_scale_inv.data.index_select(0, index).contiguous()
    w2 = experts.w2_weight.data.index_select(0, index).contiguous()
    w2_sf_fp32 = experts.w2_weight_scale_inv.data.index_select(0, index).contiguous()

    num_groups, n1, half_k1 = w13.shape
    k1 = half_k1 * 2
    num_groups_w2, n2, half_k2 = w2.shape
    if num_groups_w2 != num_groups:
        raise RuntimeError("SGLang FusedMoE w13/w2 expert dimensions do not match")
    k2 = half_k2 * 2

    w13_sf = transform_sf_into_required_layout(
        w13_sf_fp32,
        mn=n1,
        k=k1,
        recipe=(1, 32),
        num_groups=num_groups,
        disable_ue8m0_cast=False,
    )
    w2_sf = transform_sf_into_required_layout(
        w2_sf_fp32,
        mn=n2,
        k=k2,
        recipe=(1, 32),
        num_groups=num_groups,
        disable_ue8m0_cast=False,
    )
    return transform_weights_for_mega_moe((w13, w13_sf), (w2, w2_sf))


def _build_target_ep_workload(
    task: Dsv4MegaMoEComputeTask,
    *,
    routed_topk: int,
    routed_num_experts: int,
    num_fused_shared_experts: int,
    hidden_size: int | None = None,
) -> Dsv4MegaMoEWorkload:
    if task.distribution == "uniform":
        return build_dsv4_uniform_megamoe_workload_from_global_tokens(
            num_global_tokens=task.num_tokens,
            routed_num_experts=routed_num_experts,
            routed_topk=routed_topk,
            moe_ep_size=task.moe_ep_size,
            num_fused_shared_experts=num_fused_shared_experts,
            hidden_size=hidden_size,
        )
    elif task.distribution == "power_law":
        if task.power_law_alpha is None:
            raise ValueError("power_law distribution requires power_law_alpha")
        return build_dsv4_power_law_megamoe_workload_from_global_tokens(
            num_global_tokens=task.num_tokens,
            routed_num_experts=routed_num_experts,
            routed_topk=routed_topk,
            moe_ep_size=task.moe_ep_size,
            alpha=task.power_law_alpha,
            num_fused_shared_experts=num_fused_shared_experts,
            hidden_size=hidden_size,
        )
    else:
        raise ValueError(f"Unsupported distribution: {task.distribution}")


def _make_local_topk_tensors(workload: Dsv4MegaMoEWorkload, *, device: str):
    import torch

    if workload.rank0_local_topk_ids:
        topk_ids = torch.tensor(workload.rank0_local_topk_ids, dtype=torch.int32, device=device)
    else:
        topk_ids = torch.empty((0, workload.mega_topk), dtype=torch.int32, device=device)
    topk_weights = torch.zeros_like(topk_ids, dtype=torch.float32)
    topk_weights[topk_ids >= 0] = 1.0 / float(workload.mega_topk)
    masked_m = torch.tensor(workload.rank0_masked_m, dtype=torch.int32, device=device)
    return topk_ids.contiguous(), topk_weights.contiguous(), masked_m.contiguous()


def _make_source_predispatch_topk_tensors(workload: Dsv4MegaMoEWorkload, *, device: str):
    import torch

    if not workload.mega_topk_ids_by_src_rank:
        raise ValueError("workload has no source-rank topk rows")

    source_rank = max(
        range(len(workload.mega_topk_ids_by_src_rank)),
        key=lambda rank: len(workload.mega_topk_ids_by_src_rank[rank]),
    )
    source_rows = workload.mega_topk_ids_by_src_rank[source_rank]
    if source_rows:
        topk_ids = torch.tensor(source_rows, dtype=torch.int32, device=device)
    else:
        topk_ids = torch.empty((0, workload.mega_topk), dtype=torch.int32, device=device)
    topk_weights = torch.zeros_like(topk_ids, dtype=torch.float32)
    topk_weights[topk_ids >= 0] = 1.0 / float(workload.mega_topk)
    return source_rank, topk_ids.contiguous(), topk_weights.contiguous()


def _deep_gemm_supports_recv_stats(deep_gemm_module: Any) -> bool:
    try:
        params = signature(deep_gemm_module.fp8_fp4_mega_moe).parameters
    except (TypeError, ValueError):
        return False
    if "cumulative_local_expert_recv_stats" in params:
        return True
    return any(param.kind == Parameter.VAR_KEYWORD for param in params.values())


def _prepare_runtime(task: Dsv4MegaMoEComputeTask, moe_layer: Any, device: str) -> dict[str, Any]:
    import torch

    from sglang.jit_kernel.deepseek_v4 import mega_moe_pre_dispatch
    from sglang.srt.distributed.parallel_state import get_moe_ep_group
    from sglang.srt.models.deepseek_v2 import _get_mega_moe_symm_buffer

    import deep_gemm

    supports_recv_stats = _deep_gemm_supports_recv_stats(deep_gemm)

    routed_topk = int(moe_layer.config.num_experts_per_tok)
    routed_num_experts = int(moe_layer.config.n_routed_experts)
    num_fused_shared_experts = int(getattr(moe_layer, "num_fused_shared_experts", 0) or 0)
    global_mega_topk = routed_topk + num_fused_shared_experts
    global_mega_num_experts = int(moe_layer.experts.num_experts)
    target_moe_ep_size = int(task.moe_ep_size)
    local_routed_num_experts = routed_num_experts // target_moe_ep_size
    runtime_mega_num_experts = local_routed_num_experts + num_fused_shared_experts
    token_cap = int(os.environ.get("SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK", DEFAULT_MEGAMOE_TOKEN_CAP))

    workload = _build_target_ep_workload(
        task,
        routed_topk=routed_topk,
        routed_num_experts=routed_num_experts,
        num_fused_shared_experts=num_fused_shared_experts,
        hidden_size=task.hidden_size,
    )
    core_num_tokens = len(workload.rank0_local_token_indices)
    if core_num_tokens <= 0:
        raise RuntimeError("target-EP workload produced no rank0-local tokens")
    core_hidden_states = torch.randn(core_num_tokens, task.hidden_size, dtype=torch.bfloat16, device=device)
    core_topk_ids, core_topk_weights, expected_masked_m = _make_local_topk_tensors(workload, device=device)

    source_predispatch_rank, source_topk_ids, source_topk_weights = _make_source_predispatch_topk_tensors(
        workload,
        device=device,
    )
    source_predispatch_num_tokens = int(source_topk_ids.shape[0])
    if source_predispatch_num_tokens <= 0:
        raise RuntimeError("target-EP workload produced no source-local pre-dispatch tokens")
    source_hidden_states = torch.randn(
        source_predispatch_num_tokens,
        task.hidden_size,
        dtype=torch.bfloat16,
        device=device,
    )

    local_expert_indices = _local_expert_indices(
        routed_num_experts=routed_num_experts,
        target_moe_ep_size=target_moe_ep_size,
        num_fused_shared_experts=num_fused_shared_experts,
    )
    local_mega_l1_weights, local_mega_l2_weights = _build_local_mega_weight_pairs(
        moe_layer.experts,
        local_expert_indices,
    )

    ep_group = get_moe_ep_group().device_group
    if ep_group.size() != 1:
        raise RuntimeError(
            f"DSv4 MegaMoE local-compute collector requires a single-rank runtime EP group, got {ep_group.size()}"
        )
    core_token_cap = max(token_cap, core_num_tokens)
    core_buf = _get_mega_moe_symm_buffer(
        ep_group,
        num_experts=runtime_mega_num_experts,
        num_max_tokens_per_rank=core_token_cap,
        num_topk=global_mega_topk,
        hidden=task.hidden_size,
        intermediate_hidden=task.inter_size,
    )
    source_predispatch_token_cap = max(token_cap, source_predispatch_num_tokens)
    source_predispatch_buf = _get_mega_moe_symm_buffer(
        ep_group,
        num_experts=global_mega_num_experts,
        num_max_tokens_per_rank=source_predispatch_token_cap,
        num_topk=global_mega_topk,
        hidden=task.hidden_size,
        intermediate_hidden=task.inter_size,
    )

    def prepare_core_func():
        mega_moe_pre_dispatch(
            core_hidden_states,
            core_topk_ids,
            core_topk_weights,
            core_buf.x,
            core_buf.x_sf,
            core_buf.topk_idx,
            core_buf.topk_weights,
            quant_group_size=32,
        )

    def predispatch_func():
        mega_moe_pre_dispatch(
            source_hidden_states,
            source_topk_ids,
            source_topk_weights,
            source_predispatch_buf.x,
            source_predispatch_buf.x_sf,
            source_predispatch_buf.topk_idx,
            source_predispatch_buf.topk_weights,
            quant_group_size=32,
        )

    swiglu_limit = getattr(moe_layer.config, "swiglu_limit", None)

    y = torch.empty((core_num_tokens, task.hidden_size), dtype=torch.bfloat16, device=device)

    def core_func(cumulative_local_expert_recv_stats=None):
        kwargs = {
            "recipe": (1, 1, 32),
            "activation": "swiglu",
            "activation_clamp": swiglu_limit,
            "fast_math": True,
        }
        if cumulative_local_expert_recv_stats is not None:
            if not supports_recv_stats:
                raise RuntimeError("Installed DeepGEMM fp8_fp4_mega_moe does not support recv-stats validation")
            kwargs["cumulative_local_expert_recv_stats"] = cumulative_local_expert_recv_stats
        deep_gemm.fp8_fp4_mega_moe(y, local_mega_l1_weights, local_mega_l2_weights, core_buf, **kwargs)

    validation_stats = (
        torch.empty(runtime_mega_num_experts, dtype=torch.int32, device=device)
        if supports_recv_stats
        else None
    )

    def validate_workload_func():
        if int(expected_masked_m.numel()) != runtime_mega_num_experts:
            raise RuntimeError("rank0_masked_m length does not match runtime MegaMoE expert count")
        valid_topk = core_topk_ids >= 0
        if bool((core_topk_ids[valid_topk] >= runtime_mega_num_experts).any().item()):
            raise RuntimeError("local topk IDs exceed runtime MegaMoE expert count")
        if int(valid_topk.sum().item()) != int(expected_masked_m.sum().item()):
            raise RuntimeError("rank0_masked_m sum does not match local topk valid selection count")

        prepare_core_func()
        if supports_recv_stats:
            validation_stats.zero_()
            core_func(validation_stats)
            torch.cuda.synchronize()
            if not torch.equal(validation_stats.cpu(), expected_masked_m.cpu()):
                raise RuntimeError(
                    "DeepGEMM MegaMoE local recv stats do not match target-EP workload: "
                    f"actual={validation_stats.cpu().tolist()}, expected={expected_masked_m.cpu().tolist()}"
                )
        else:
            core_func()
            torch.cuda.synchronize()

    metadata = {
        "routed_topk": routed_topk,
        "routed_num_experts": routed_num_experts,
        "num_fused_shared_experts": num_fused_shared_experts,
        "mega_topk": global_mega_topk,
        "mega_num_experts": global_mega_num_experts,
        "runtime_moe_ep_size": 1,
        "target_moe_ep_size": target_moe_ep_size,
        "local_routed_num_experts": local_routed_num_experts,
        "runtime_mega_num_experts": runtime_mega_num_experts,
        "local_expert_indices": list(local_expert_indices),
        "rank0_local_num_tokens": core_num_tokens,
        "core_num_tokens": core_num_tokens,
        "source_predispatch_rank": source_predispatch_rank,
        "source_predispatch_num_tokens": source_predispatch_num_tokens,
        "rank0_total_local_selections": int(sum(workload.rank0_masked_m)),
        "rank0_masked_m": list(workload.rank0_masked_m),
        "workload_num_tokens_per_rank": workload.num_tokens_per_rank,
        "workload_num_global_tokens": workload.num_global_tokens,
        "workload_bottleneck_rank_before_remap": workload.bottleneck_rank_before_remap,
        "workload_routed_rank_loads": list(workload.routed_rank_loads),
        "traffic_total_remote_edges": None if workload.traffic is None else workload.traffic.total_remote_edges,
        "traffic_bottleneck_primary_bytes": (
            None if workload.traffic is None else workload.traffic.bottleneck_primary_bytes
        ),
        "buffer_num_max_tokens_per_rank": int(core_buf.num_max_tokens_per_rank),
        "core_buffer_num_max_tokens_per_rank": int(core_buf.num_max_tokens_per_rank),
        "source_predispatch_buffer_num_max_tokens_per_rank": int(source_predispatch_buf.num_max_tokens_per_rank),
        "topk_ids_dtype": str(core_topk_ids.dtype),
        "topk_weights_dtype": str(core_topk_weights.dtype),
        "core_topk_ids_dtype": str(core_topk_ids.dtype),
        "core_topk_weights_dtype": str(core_topk_weights.dtype),
        "source_predispatch_topk_ids_dtype": str(source_topk_ids.dtype),
        "source_predispatch_topk_weights_dtype": str(source_topk_weights.dtype),
        "buf_topk_idx_dtype": str(core_buf.topk_idx.dtype),
        "buf_topk_weights_dtype": str(core_buf.topk_weights.dtype),
        "model_mega_moe_weights_built": bool(getattr(moe_layer.experts, "_mega_moe_weights_built", False)),
        "local_mega_moe_weights_built": True,
        "local_l1_weight_shape": [list(tensor.shape) for tensor in local_mega_l1_weights],
        "local_l1_weight_stride": [list(tensor.stride()) for tensor in local_mega_l1_weights],
        "local_l2_weight_shape": [list(tensor.shape) for tensor in local_mega_l2_weights],
        "local_l2_weight_stride": [list(tensor.stride()) for tensor in local_mega_l2_weights],
        "deep_gemm_supports_recv_stats": supports_recv_stats,
        "workload_validation_mode": "kernel_recv_stats" if supports_recv_stats else "input_invariants_and_core_smoke",
    }
    return {
        "predispatch_func": predispatch_func,
        "prepare_core_func": prepare_core_func,
        "core_func": core_func,
        "validate_workload_func": validate_workload_func,
        "metadata": metadata,
    }


def _benchmark_runtime(runtime: dict[str, Any], device: str) -> tuple[dict[str, Any], dict[str, Any] | None]:
    import torch

    torch_device = torch.device(device)
    predispatch_func = runtime["predispatch_func"]
    prepare_core_func = runtime["prepare_core_func"]
    core_func = runtime["core_func"]
    validate_workload_func = runtime["validate_workload_func"]

    validate_workload_func()
    prepare_core_func()
    torch.cuda.synchronize()

    predispatch_latency_ms: float | None = None
    if _parse_bool_env("AIC_DSV4_MEGAMOE_MEASURE_PREDISPATCH", default=True):
        with benchmark_with_power(
            device=torch_device,
            kernel_func=predispatch_func,
            num_warmups=int(os.environ.get("AIC_DSV4_MEGAMOE_NUM_WARMUPS", "3")),
            num_runs=int(os.environ.get("AIC_DSV4_MEGAMOE_NUM_RUNS", "10")),
            repeat_n=1,
            measure_power=False,
            allow_graph_fail=True,
        ) as predispatch_results:
            pass
        predispatch_latency_ms = float(predispatch_results["latency_ms"])

    prepare_core_func()
    torch.cuda.synchronize()

    with benchmark_with_power(
        device=torch_device,
        kernel_func=core_func,
        num_warmups=int(os.environ.get("AIC_DSV4_MEGAMOE_NUM_WARMUPS", "3")),
        num_runs=int(os.environ.get("AIC_DSV4_MEGAMOE_NUM_RUNS", "10")),
        repeat_n=1,
        allow_graph_fail=True,
    ) as core_results:
        pass

    core_latency_ms = float(core_results["latency_ms"])
    local_compute_latency_ms = core_latency_ms + (predispatch_latency_ms or 0.0)
    return (
        {
            "latency_ms": local_compute_latency_ms,
            "core_latency_ms": core_latency_ms,
            "local_predispatch_latency_ms": predispatch_latency_ms,
            "used_cuda_graph": core_results.get("used_cuda_graph"),
            "num_runs_executed": core_results.get("num_runs_executed"),
        },
        core_results.get("power_stats"),
    )


def _run_dsv4_megamoe_compute_inprocess(task: Dsv4MegaMoEComputeTask, device_id: int) -> None:
    env_snapshot = _apply_dsv4_megamoe_env_defaults()

    if _parse_bool_env("AIC_DSV4_MEGAMOE_DRY_RUN", default=False):
        report = _base_compute_report(task, device_id, status="dry_run")
        report["env"] = env_snapshot
        report["message"] = "Dry run only; SGLang runtime was not imported."
        _write_json(_compute_report_path(task, device_id, suffix="dry_run"), report)
        return

    import torch

    if not torch.cuda.is_available():
        report = _base_compute_report(task, device_id, status="skipped")
        report["env"] = env_snapshot
        report["error"] = "CUDA is not available"
        _write_json(_compute_report_path(task, device_id, suffix="compute"), report)
        raise RuntimeError("CUDA is not available")

    torch.cuda.set_device("cuda:0")
    model_runner = None
    report = _base_compute_report(task, device_id, status="running")
    try:
        model_runner, _server_args, model_path = _load_model_runner(task, device_id)
        layer_id, moe_layer = _find_megamoe_layer(model_runner)
        runtime = _prepare_runtime(task, moe_layer, "cuda:0")
        timings, power_stats = _benchmark_runtime(runtime, "cuda:0")

        observed = {
            **runtime["metadata"],
            **timings,
            "layer_id": layer_id,
            "moe_layer_class": moe_layer.__class__.__name__,
            "experts_class": moe_layer.experts.__class__.__name__,
            "is_hash_layer": bool(getattr(moe_layer, "is_hash", False)),
            "is_nextn": bool(getattr(moe_layer, "is_nextn", False)),
        }
        report.update(
            {
                "status": "ok",
                "model_path": model_path,
                "sglang_version": _safe_version("sglang"),
                "layer_id": layer_id,
                "observed": observed,
                "power_stats": power_stats,
            }
        )
        _write_json(_compute_report_path(task, device_id, suffix="compute"), report)

        device_name = torch.cuda.get_device_name("cuda:0")
        log_perf(
            item_list=[
                {
                    "moe_dtype": task.moe_type,
                    "num_tokens": observed["rank0_local_num_tokens"],
                    "core_num_tokens": observed["core_num_tokens"],
                    "source_predispatch_num_tokens": observed["source_predispatch_num_tokens"],
                    "source_num_tokens": task.num_tokens,
                    "workload_num_global_tokens": observed["workload_num_global_tokens"],
                    "hidden_size": task.hidden_size,
                    "inter_size": task.inter_size,
                    "topk": observed["mega_topk"],
                    "num_experts": observed["mega_num_experts"],
                    "runtime_num_experts": observed["runtime_mega_num_experts"],
                    "routed_topk": observed["routed_topk"],
                    "routed_num_experts": observed["routed_num_experts"],
                    "local_routed_num_experts": observed["local_routed_num_experts"],
                    "num_fused_shared_experts": observed["num_fused_shared_experts"],
                    "moe_tp_size": task.moe_tp_size,
                    "moe_ep_size": task.moe_ep_size,
                    "runtime_moe_ep_size": observed["runtime_moe_ep_size"],
                    "distribution": task.distribution,
                    "latency": timings["latency_ms"],
                    "core_latency": timings["core_latency_ms"],
                    "local_predispatch_latency": timings["local_predispatch_latency_ms"],
                    "traffic_bottleneck_primary_bytes": observed["traffic_bottleneck_primary_bytes"],
                }
            ],
            framework="SGLang",
            version=_safe_version("sglang"),
            device_name=device_name,
            op_name="moe",
            kernel_source="deep_gemm_fp8_fp4_mega_moe_compute",
            perf_filename=task.perf_filename,
            power_stats=power_stats,
        )
    except Exception as e:
        report.update(
            {
                "status": "error",
                "error_type": type(e).__name__,
                "error": str(e),
                "traceback": traceback.format_exc(),
            }
        )
        _write_json(_compute_report_path(task, device_id, suffix="compute"), report)
        raise
    finally:
        if model_runner is not None:
            del model_runner
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


def _run_dsv4_megamoe_compute_from_env() -> None:
    payload = json.loads(os.environ["AIC_DSV4_MEGAMOE_COMPUTE_TASK_JSON"])
    task = Dsv4MegaMoEComputeTask(**payload["task"])
    _run_dsv4_megamoe_compute_inprocess(task, int(payload["device_id"]))


def _run_subprocess(task: Dsv4MegaMoEComputeTask, device_id: int) -> None:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(device_id)
    env["AIC_DSV4_MEGAMOE_COMPUTE_TASK_JSON"] = json.dumps(
        {
            "task": task.__dict__,
            "device_id": device_id,
        }
    )
    module_dir = os.path.dirname(os.path.abspath(__file__))
    code = f"""
import sys
sys.path.insert(0, {module_dir!r})
from collect_dsv4_megamoe_compute import _run_dsv4_megamoe_compute_from_env
_run_dsv4_megamoe_compute_from_env()
"""
    proc = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        cwd=module_dir,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=int(os.environ.get("AIC_DSV4_MEGAMOE_TIMEOUT_SEC", "900")),
    )
    if proc.stdout:
        print(proc.stdout)
    if proc.returncode != 0:
        raise RuntimeError(f"DSv4 MegaMoE compute subprocess failed with exit code {proc.returncode}")


def run_dsv4_megamoe_compute(
    moe_type,
    num_tokens,
    hidden_size,
    inter_size,
    topk,
    num_experts,
    moe_tp_size,
    moe_ep_size,
    model_name,
    perf_filename,
    distribution,
    power_law_alpha=None,
    device="cuda:0",
):
    _apply_dsv4_megamoe_env_defaults()
    task = _coerce_compute_task(
        moe_type,
        num_tokens,
        hidden_size,
        inter_size,
        topk,
        num_experts,
        moe_tp_size,
        moe_ep_size,
        model_name,
        perf_filename,
        distribution,
        power_law_alpha,
    )
    device_str = str(device)
    device_id = int(device_str.split(":")[-1]) if ":" in device_str else 0

    if _parse_bool_env("AIC_DSV4_MEGAMOE_DRY_RUN", default=False):
        _run_dsv4_megamoe_compute_inprocess(task, device_id)
        return

    _run_subprocess(task, device_id)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SGLang DeepSeek-V4 MegaMoE compute collector")
    parser.add_argument("--dry-run", action="store_true", help="Write compute plan JSON without importing SGLang")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=0, help="Run only the first N cases")
    args = parser.parse_args()

    if args.dry_run:
        os.environ["AIC_DSV4_MEGAMOE_DRY_RUN"] = "1"

    cases = get_dsv4_megamoe_compute_test_cases()
    if args.limit:
        cases = cases[: args.limit]
    for case in cases:
        run_dsv4_megamoe_compute(*case, device=args.device)
