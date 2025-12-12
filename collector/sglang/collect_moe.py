# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import itertools
import os
from typing import TypedDict
from unittest.mock import MagicMock

import pkg_resources

# Mock global server args before importing MOE modules (required by SGLang 0.5.5+)
# The fused_moe_triton_config module now requires get_global_server_args() to be set
import sglang.srt.server_args as _server_args_module
import torch

if _server_args_module._global_server_args is None:
    _mock_server_args = MagicMock()
    _mock_server_args.enable_deterministic_inference = False
    _server_args_module._global_server_args = _mock_server_args

from sglang.srt.layers.moe.fused_moe_triton.fused_moe import fused_moe
from sglang.srt.layers.moe.fused_moe_triton.fused_moe_triton_config import (
    get_config_dtype_str,
    get_default_config,
    get_moe_configs,
)
from sglang.srt.layers.moe.topk import TopKConfig, select_experts
from sglang.srt.utils import is_hip

try:
    from common_test_cases import get_common_moe_test_cases

    from helper import balanced_logits, benchmark_with_power, get_sm_version, log_perf, power_law_logits_v3
except ModuleNotFoundError:
    import os
    import sys

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from common_test_cases import get_common_moe_test_cases

    from helper import balanced_logits, benchmark_with_power, get_sm_version, log_perf, power_law_logits_v3


_is_hip = is_hip()


def get_moe_test_cases():
    # fp8_block MOE requires SM90+ due to shared memory requirements
    # L40S (SM89) has 100KB shared memory, fp8_block kernel needs ~144KB
    sm_version = get_sm_version()
    if sm_version < 90:
        moe_list = ["float16"]
    else:
        moe_list = ["float16", "fp8_block"]

    test_cases = []

    for common_moe_testcase in get_common_moe_test_cases():
        if common_moe_testcase.token_expert_distribution != "power_law":
            continue

        # Skip EP > 1 test cases - this collector only supports single-GPU MOE (ep_size=1)
        # For EP > 1 (multi-GPU expert parallelism), use collect_wideep_deepep_moe.py instead
        if common_moe_testcase.ep != 1:
            continue

        model_name = common_moe_testcase.model_name
        if model_name in ["GPT_OSS_20B", "GPT_OSS_120B"]:
            continue

        num_tokens_list = [num_tokens for num_tokens in common_moe_testcase.num_tokens_list if num_tokens <= 20480]

        for moe_type, num_tokens in itertools.product(moe_list, num_tokens_list):
            test_cases.append(
                [
                    moe_type,
                    num_tokens,
                    common_moe_testcase.hidden_size,
                    common_moe_testcase.inter_size,
                    common_moe_testcase.topk,
                    common_moe_testcase.num_experts,
                    common_moe_testcase.tp,
                    common_moe_testcase.ep,
                    common_moe_testcase.model_name,
                    "moe_perf.txt",
                    common_moe_testcase.token_expert_distribution,
                    common_moe_testcase.power_law_alpha,
                ]
            )

    return test_cases


class BenchmarkConfig(TypedDict):
    BLOCK_SIZE_M: int
    BLOCK_SIZE_N: int
    BLOCK_SIZE_K: int
    GROUP_SIZE_M: int
    num_warps: int
    num_stages: int


def benchmark_config(
    config: BenchmarkConfig,
    num_tokens: int,
    num_experts: int,
    shard_intermediate_size: int,
    hidden_size: int,
    topk: int,
    dtype: torch.dtype,
    use_fp8_w8a8: bool,
    use_int8_w8a8: bool,
    use_int8_w8a16: bool,
    block_shape: list[int] | None = None,
    num_iters: int = 10,
    distributed: str = "power_law",
    power_law_alpha: float = 0,
) -> float:
    init_dtype = torch.float16 if use_fp8_w8a8 else dtype
    x = torch.randn(num_tokens, hidden_size, dtype=dtype)
    if use_int8_w8a16 or use_int8_w8a8:
        w1 = torch.randint(
            -127,
            127,
            (
                num_experts,
                shard_intermediate_size,
                hidden_size,
            ),
            dtype=torch.int8,
        )
        w2 = torch.randint(
            -127,
            127,
            (
                num_experts,
                hidden_size,
                shard_intermediate_size // 2,
            ),
            dtype=torch.int8,
        )
    else:
        w1 = torch.randn(num_experts, shard_intermediate_size, hidden_size, dtype=init_dtype)
        w2 = torch.randn(num_experts, hidden_size, shard_intermediate_size // 2, dtype=init_dtype)
    if distributed == "uniform":
        gating_output = torch.randn(num_iters, num_tokens, num_experts, dtype=torch.float32)
    elif distributed == "balanced":
        gating_output = [balanced_logits(num_tokens, num_experts, topk) for _ in range(num_iters)]
    elif distributed == "power_law":
        # only support ep=1 for sglang
        gating_output = [
            power_law_logits_v3(num_tokens, num_experts, topk, 1, power_law_alpha) for _ in range(num_iters)
        ]
    else:
        raise ValueError(f"Unsupported distributed mode: {distributed}")

    w1_scale = None
    w2_scale = None
    a1_scale = None
    a2_scale = None
    if use_int8_w8a16:
        w1_scale = torch.randn((num_experts, 2 * shard_intermediate_size), dtype=torch.float32)
        w2_scale = torch.randn((hidden_size, num_experts), dtype=torch.float32)
    if use_fp8_w8a8 or use_int8_w8a8:
        if use_int8_w8a8 and block_shape is None:
            w1_scale = torch.randn(num_experts, shard_intermediate_size, dtype=torch.float32)
            w2_scale = torch.randn(num_experts, hidden_size, dtype=torch.float32)
        elif block_shape is None:
            w1_scale = torch.randn(num_experts, dtype=torch.float32)
            w2_scale = torch.randn(num_experts, dtype=torch.float32)
            a1_scale = torch.randn(1, dtype=torch.float32)
            a2_scale = torch.randn(1, dtype=torch.float32)
        else:
            block_n, block_k = block_shape[0], block_shape[1]
            n_tiles_w1 = (shard_intermediate_size + block_n - 1) // block_n
            n_tiles_w2 = (hidden_size + block_n - 1) // block_n
            k_tiles_w1 = (hidden_size + block_k - 1) // block_k
            k_tiles_w2 = (shard_intermediate_size // 2 + block_k - 1) // block_k
            w1_scale = torch.rand((num_experts, n_tiles_w1, k_tiles_w1), dtype=torch.float32)
            w2_scale = torch.rand((num_experts, n_tiles_w2, k_tiles_w2), dtype=torch.float32)

    if use_fp8_w8a8:
        w1 = w1.to(torch.float8_e4m3fnuz if _is_hip else torch.float8_e4m3fn)
        w2 = w2.to(torch.float8_e4m3fnuz if _is_hip else torch.float8_e4m3fn)

    input_gating = torch.randn(num_tokens, num_experts, dtype=torch.float32)
    topk_output = select_experts(x, input_gating, TopKConfig(top_k=topk))

    def prepare(i: int):
        input_gating = gating_output[i]
        new_topk_output = select_experts(x, input_gating, TopKConfig(top_k=topk))
        topk_output.topk_weights.copy_(new_topk_output.topk_weights)
        topk_output.topk_ids.copy_(new_topk_output.topk_ids)
        topk_output.router_logits.copy_(new_topk_output.router_logits)

    def run():
        from sglang.srt.layers.moe.fused_moe_triton import override_config

        with override_config(config):
            fused_moe(
                x,
                w1,
                w2,
                topk_output,
                use_fp8_w8a8=use_fp8_w8a8,
                use_int8_w8a8=use_int8_w8a8,
                use_int8_w8a16=use_int8_w8a16,
                w1_scale=w1_scale,
                w2_scale=w2_scale,
                a1_scale=a1_scale,
                a2_scale=a2_scale,
                block_shape=block_shape,
            )

    # Use benchmark_with_power context manager
    device = torch.device(x.device)

    def kernel_func():
        for _ in range(10):
            run()

    with benchmark_with_power(
        device=device,
        kernel_func=kernel_func,
        num_warmups=5,
        num_runs=num_iters,
        repeat_n=1,
    ) as results:
        pass

    return results["latency_ms"], results["power_stats"]


def benchmark(
    num_tokens: int,
    num_experts: int,
    shard_intermediate_size: int,
    hidden_size: int,
    topk: int,
    dtype: torch.dtype,
    use_fp8_w8a8: bool,
    use_int8_w8a8: bool,
    use_int8_w8a16: bool,
    block_shape: list[int],
    distributed: str = "power_law",
    power_law_alpha: float = 0,
) -> tuple[dict[str, int], float]:
    torch.cuda.manual_seed_all(0)
    dtype_str = get_config_dtype_str(dtype, use_int8_w8a16=use_int8_w8a16, use_fp8_w8a8=use_fp8_w8a8)
    # NOTE(woosuk): The current naming convention uses w2.shape[2], which
    # is the intermediate size after silu_and_mul.
    block_n = block_shape[0] if block_shape else 0
    block_k = block_shape[1] if block_shape else 0
    op_config = get_moe_configs(num_experts, shard_intermediate_size // 2, dtype_str, block_n, block_k)
    if op_config is None:
        config = get_default_config(
            num_tokens,
            num_experts,
            shard_intermediate_size,
            hidden_size,
            topk,
            dtype_str,
            False,
            block_shape,
        )
    else:
        config = op_config[min(op_config.keys(), key=lambda x: abs(x - num_tokens))]
    kernel_time, power_stats = benchmark_config(
        config,
        num_tokens,
        num_experts,
        shard_intermediate_size,
        hidden_size,
        topk,
        dtype,
        use_fp8_w8a8,
        use_int8_w8a8,
        use_int8_w8a16,
        block_shape,
        distributed=distributed,
        power_law_alpha=power_law_alpha,
    )
    return kernel_time, power_stats


def run_moe_torch(
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
    distributed="power_law",
    power_law_alpha=0,
    device="cuda:0",
):
    torch.cuda.set_device(device)
    torch.set_default_device(device)

    assert moe_ep_size == 1, "only support moe ep size = 1"
    assert moe_type == "fp8_block" or moe_type == "float16", "only support moe type = fp8_block or float16"
    assert inter_size % moe_tp_size == 0, "inter_size % moe_tp_size must be 0"

    latency, power_stats = benchmark(
        num_tokens,
        num_experts,
        2 * inter_size // moe_tp_size,
        hidden_size,
        topk,
        torch.bfloat16,
        moe_type == "fp8_block",
        False,
        False,
        None,
        distributed=distributed,
        power_law_alpha=power_law_alpha,
    )

    log_perf(
        item_list=[
            {
                "moe_dtype": moe_type,
                "num_tokens": num_tokens,
                "hidden_size": hidden_size,
                "inter_size": inter_size,
                "topk": topk,
                "num_experts": num_experts,
                "moe_tp_size": moe_tp_size,
                "moe_ep_size": moe_ep_size,
                "distribution": "power_law_" + str(power_law_alpha) if distributed == "power_law" else distributed,
                "latency": latency,
            }
        ],
        framework="SGLang",
        version=pkg_resources.get_distribution("sglang").version,
        device_name=torch.cuda.get_device_name(device),
        op_name="moe",
        kernel_source="sglang_fused_moe_triton",
        perf_filename=perf_filename,
        power_stats=power_stats,
    )
