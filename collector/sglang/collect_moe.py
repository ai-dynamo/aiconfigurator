# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    from flashinfer import fp4_quantize
    from sglang.srt.layers.moe.flashinfer_cutedsl_moe import (
        flashinfer_cutedsl_moe_masked,
    )

    HAS_FLASHINFER_CUTE = True
except ImportError:
    HAS_FLASHINFER_CUTE = False

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
    elif sm_version < 100:
        moe_list = ["float16", "fp8_block"]
    else:
        moe_list = ["float16", "fp8_block", "nvfp4"]

    test_cases = []

    for common_moe_testcase in get_common_moe_test_cases():
        if common_moe_testcase.token_expert_distribution != "power_law":
            continue

        # Skip EP > 1 test cases - this collector only supports single-GPU MOE (ep_size=1)
        # For EP > 1 (multi-GPU expert parallelism), use collect_wideep_deepep_moe.py instead
        if common_moe_testcase.ep != 1:
            continue

        model_name = common_moe_testcase.model_name
        if model_name in ["openai/gpt-oss-20b", "openai/gpt-oss-120b"]:
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
    use_nvfp4: bool = False,
    block_shape: list[int] | None = None,
    num_iters: int = 10,
    distributed: str = "power_law",
    power_law_alpha: float = 0,
) -> float:
    device = torch.device("cuda")

    # 1. Gating Output Generation (Common)
    if distributed == "uniform":
        gating_output = torch.randn(num_iters, num_tokens, num_experts, dtype=torch.float32, device=device)
    elif distributed == "balanced":
        gating_output = [balanced_logits(num_tokens, num_experts, topk).to(device) for _ in range(num_iters)]
    elif distributed == "power_law":
        gating_output = [
            power_law_logits_v3(num_tokens, num_experts, topk, 1, power_law_alpha).to(device) for _ in range(num_iters)
        ]
    else:
        raise ValueError(f"Unsupported distributed mode: {distributed}")

    # 2. Setup based on Path
    if use_nvfp4:
        if not HAS_FLASHINFER_CUTE:
            raise ImportError("FlashInfer CuteDSL not available")

        # Global scales and Alpha
        input_gs = torch.ones(num_experts, device=device, dtype=torch.float32)
        w1_gs = torch.ones(num_experts, device=device, dtype=torch.float32)
        a2_gs = torch.ones(num_experts, device=device, dtype=torch.float32)
        w2_gs = torch.ones(num_experts, device=device, dtype=torch.float32)
        w1_alpha = torch.ones(num_experts, device=device, dtype=torch.float32)
        w2_alpha = torch.ones(num_experts, device=device, dtype=torch.float32)

        # Weight quantization
        w1_bf16 = torch.randn(num_experts, shard_intermediate_size, hidden_size, device=device, dtype=dtype)
        w2_bf16 = torch.randn(num_experts, hidden_size, shard_intermediate_size // 2, device=device, dtype=dtype)

        w1_gs_exp = w1_gs.repeat_interleave(shard_intermediate_size).view(-1, 1)
        w2_gs_exp = w2_gs.repeat_interleave(hidden_size).view(-1, 1)

        w1_fp4, w1_sf = fp4_quantize(w1_bf16.reshape(-1, hidden_size), w1_gs_exp, is_sf_swizzled_layout=False)
        w2_fp4, w2_sf = fp4_quantize(
            w2_bf16.reshape(-1, shard_intermediate_size // 2), w2_gs_exp, is_sf_swizzled_layout=False
        )

        w1 = w1_fp4.reshape(num_experts, shard_intermediate_size, hidden_size // 2)
        w1_bs = w1_sf.view(torch.float8_e4m3fn).reshape(num_experts, shard_intermediate_size, -1).contiguous()
        w2 = w2_fp4.reshape(num_experts, hidden_size, shard_intermediate_size // 4)
        w2_bs = w2_sf.view(torch.float8_e4m3fn).reshape(num_experts, hidden_size, -1).contiguous()

        def get_masked_m(logits):
            _, topk_idx = torch.topk(torch.softmax(logits, dim=1), topk, dim=-1)
            counts = [(topk_idx.view(-1) == i).sum() for i in range(num_experts)]
            return torch.tensor(counts, dtype=torch.int32, device=device)

        masked_m_list = [get_masked_m(logits) for logits in gating_output]

        # Calculate the maximum tokens any single expert will handle across all iterations
        max_m = 0
        for counts in masked_m_list:
            max_m = max(max_m, counts.max().item())
        # Align to 128 for kernel efficiency and safety
        max_m = (max_m + 127) // 128 * 128

        x_dispatched = torch.randn(num_experts, max_m, hidden_size, device=device, dtype=dtype)

        def run_op(i):
            flashinfer_cutedsl_moe_masked(
                hidden_states=(x_dispatched, None),
                input_global_scale=input_gs,
                w1=w1,
                w1_blockscale=w1_bs,
                w1_alpha=w1_alpha,
                w2=w2,
                a2_global_scale=a2_gs,
                w2_blockscale=w2_bs,
                w2_alpha=w2_alpha,
                masked_m=masked_m_list[i % num_iters],
            )
    else:
        init_dtype = torch.float16 if use_fp8_w8a8 else dtype
        x = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)
        if use_int8_w8a16 or use_int8_w8a8:
            w1 = torch.randint(
                -127, 127, (num_experts, shard_intermediate_size, hidden_size), dtype=torch.int8, device=device
            )
            w2 = torch.randint(
                -127, 127, (num_experts, hidden_size, shard_intermediate_size // 2), dtype=torch.int8, device=device
            )
        else:
            w1 = torch.randn(num_experts, shard_intermediate_size, hidden_size, dtype=init_dtype, device=device)
            w2 = torch.randn(num_experts, hidden_size, shard_intermediate_size // 2, dtype=init_dtype, device=device)

        w1_scale = w2_scale = a1_scale = a2_scale = None
        if use_int8_w8a16:
            w1_scale = torch.randn((num_experts, 2 * shard_intermediate_size), dtype=torch.float32, device=device)
            w2_scale = torch.randn((hidden_size, num_experts), dtype=torch.float32, device=device)
        elif use_fp8_w8a8 or use_int8_w8a8:
            if use_int8_w8a8 and block_shape is None:
                w1_scale = torch.randn(num_experts, shard_intermediate_size, dtype=torch.float32, device=device)
                w2_scale = torch.randn(num_experts, hidden_size, dtype=torch.float32, device=device)
            elif block_shape is None:
                w1_scale = torch.randn(num_experts, dtype=torch.float32, device=device)
                w2_scale = torch.randn(num_experts, dtype=torch.float32, device=device)
                a1_scale = torch.randn(1, dtype=torch.float32, device=device)
                a2_scale = torch.randn(1, dtype=torch.float32, device=device)
            else:
                bn, bk = block_shape
                w1_scale = torch.rand(
                    (num_experts, (shard_intermediate_size + bn - 1) // bn, (hidden_size + bk - 1) // bk),
                    dtype=torch.float32,
                    device=device,
                )
                w2_scale = torch.rand(
                    (num_experts, (hidden_size + bn - 1) // bn, (shard_intermediate_size // 2 + bk - 1) // bk),
                    dtype=torch.float32,
                    device=device,
                )

        if use_fp8_w8a8:
            f8_type = torch.float8_e4m3fnuz if _is_hip else torch.float8_e4m3fn
            w1, w2 = w1.to(f8_type), w2.to(f8_type)

        topk_output = select_experts(x, torch.randn(num_tokens, num_experts, device=device), TopKConfig(top_k=topk))

        def run_op(i):
            from sglang.srt.layers.moe.fused_moe_triton import override_config

            input_gating = gating_output[i % num_iters]
            new_topk = select_experts(x, input_gating, TopKConfig(top_k=topk))
            topk_output.topk_weights.copy_(new_topk.topk_weights)
            topk_output.topk_ids.copy_(new_topk.topk_ids)
            topk_output.router_logits.copy_(new_topk.router_logits)

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

    # 3. Unified Execution Loop
    outside_loop_count = 5  # Repeat ops within kernel_func to increase accuracy for fast kernels

    def kernel_func():
        for i in range(outside_loop_count):
            run_op(i)

    with benchmark_with_power(
        device=device,
        kernel_func=kernel_func,
        num_warmups=5,
        num_runs=num_iters,
        repeat_n=1,
    ) as results:
        pass

    return results["latency_ms"] / outside_loop_count, results["power_stats"]


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
    use_nvfp4: bool = False,
    block_shape: list[int] | None = None,
    distributed: str = "power_law",
    power_law_alpha: float = 0,
) -> tuple[dict[str, int], float]:
    torch.cuda.manual_seed_all(0)

    if use_nvfp4:
        # nvfp4 uses flashinfer cutedsl backend, which doesn't need triton configs
        kernel_time, power_stats = benchmark_config(
            None,
            num_tokens,
            num_experts,
            shard_intermediate_size,
            hidden_size,
            topk,
            dtype,
            use_fp8_w8a8,
            use_int8_w8a8,
            use_int8_w8a16,
            use_nvfp4,
            block_shape,
            distributed=distributed,
            power_law_alpha=power_law_alpha,
        )
        return kernel_time, power_stats

    dtype_str = get_config_dtype_str(
        dtype,
        use_int8_w8a16=use_int8_w8a16,
        use_fp8_w8a8=use_fp8_w8a8,
    )
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
        use_nvfp4,
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
    assert moe_type in [
        "fp8_block",
        "float16",
        "nvfp4",
    ], "only support moe type = fp8_block, float16 or nvfp4"
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
        use_nvfp4=moe_type == "nvfp4",
        block_shape=None,
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
        kernel_source="sglang_flashinfer_cutedsl_moe" if moe_type == "nvfp4" else "sglang_fused_moe_triton",
        perf_filename=perf_filename,
        power_stats=power_stats,
    )


if __name__ == "__main__":
    test_cases = get_moe_test_cases()
    for test_case in test_cases:
        print(test_case)
        run_moe_torch(*test_case)
