# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

import torch
import torch.nn.functional as F
from common_test_cases import get_common_moe_test_cases
from vllm.model_executor.layers.fused_moe import fused_experts
from vllm.model_executor.layers.fused_moe.config import fp8_w8a8_moe_quant_config
from vllm.model_executor.layers.fused_moe.layer import determine_expert_map
from vllm.version import __version__ as vllm_version

from helper import balanced_logits, benchmark_with_power, get_sm_version, log_perf, power_law_logits_v3

aic_debug = int(os.getenv("aic_moe_debug", "0"))  # noqa: SIM112


def get_moe_test_cases():
    """Generate MoE test cases"""

    # Quantization types supported by vLLM
    moe_list = ["float16"]
    if get_sm_version() > 86:
        moe_list += ["fp8"]

    test_cases = []

    for common_moe_testcase in get_common_moe_test_cases():
        if common_moe_testcase.token_expert_distribution != "power_law":
            continue

        model_name = common_moe_testcase.model_name
        if model_name in ["GPT_OSS_20B", "GPT_OSS_120B"]:
            continue

        # vllm does not support TP when EP is enabled.
        if common_moe_testcase.tp > 1 and common_moe_testcase.ep > 1:
            continue

        for moe_type in moe_list:
            test_cases.append(
                [
                    moe_type,
                    common_moe_testcase.num_tokens_list,
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


def run_moe_torch(
    moe_type,
    num_tokens_lists,
    hidden_size,
    inter_size,
    topk,
    num_experts,
    moe_tp_size,
    moe_ep_size,
    model_name,
    perf_filename,
    distributed="power_law",
    power_law_alpha=0.0,
    device="cuda:0",
):
    """Run vLLM MoE performance benchmarking"""
    torch.cuda.set_device(device)
    torch.set_default_device(device)

    # Configure quantization parameters
    dtype = torch.float16
    quant_config = None

    if moe_type == "fp8":
        w1_scale = torch.randn(num_experts, dtype=torch.float32)
        w2_scale = torch.randn(num_experts, dtype=torch.float32)
        a1_scale = torch.randn(1, dtype=torch.float32)
        a2_scale = torch.randn(1, dtype=torch.float32)
        block_shape = None

        quant_config = fp8_w8a8_moe_quant_config(
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            a1_scale=a1_scale,
            a2_scale=a2_scale,
            block_shape=block_shape,
        )
        dtype = torch.float8_e4m3fn

    # Calculate local number of experts
    local_num_experts = num_experts // moe_ep_size
    local_inter_size = inter_size // moe_tp_size

    # Create weight tensors
    # w1: gate + up projection weights [num_experts, 2 * inter_size, hidden_size]
    # w2: down projection weights [num_experts, hidden_size, inter_size]
    w1 = torch.randn(
        local_num_experts,
        2 * local_inter_size,
        hidden_size,
        dtype=torch.float16,
        device=device,
    )
    w2 = torch.randn(
        local_num_experts,
        hidden_size,
        local_inter_size,
        dtype=torch.float16,
        device=device,
    )

    # Maps global expert index to local expert index.
    _, expert_map = determine_expert_map(moe_ep_size, 0, num_experts)

    if dtype == torch.float8_e4m3fn:
        w1 = w1.to(dtype)
        w2 = w2.to(dtype)

    # Performance testing for each token count
    for num_tokens_idx, num_tokens in enumerate(num_tokens_lists):
        print("num_tokens", num_tokens)
        print("topk", topk)
        hidden_states = torch.randn([num_tokens, hidden_size]).half().to(device)

        # Generate topk_weights and topk_ids
        num_iter = 10 if distributed == "power_law" else 1
        if distributed == "power_law":
            topk_weights_list = []
            topk_ids_list = []

            for _ in range(num_iter):
                logits = (
                    power_law_logits_v3(
                        num_tokens,
                        num_experts,
                        topk,
                        moe_ep_size,
                        power_law_alpha,
                    )
                    .half()
                    .to(device)
                )
                weights, ids = torch.topk(logits, topk, dim=-1)
                topk_weights_list.append(F.softmax(weights, dim=-1))
                topk_ids_list.append(ids)

            print("actual num_tokens: ", [topk_ids.shape[0] for topk_ids in topk_ids_list])

        elif distributed == "balanced":
            actual_logits = balanced_logits(num_tokens, num_experts, topk).half().to(device)
            topk_weights, topk_ids = torch.topk(actual_logits, topk, dim=-1)
            topk_weights = F.softmax(topk_weights, dim=-1)

        else:
            raise ValueError(f"Unsupported distributed mode: {distributed}")

        num_warmups = 3
        num_runs = 6
        if distributed == "power_law":
            num_warmups = 1
            num_runs = 1

        def run_single_iteration():
            if distributed == "power_law":
                for i, (tw, ti) in enumerate(zip(topk_weights_list, topk_ids_list)):
                    local_num_tokens = tw.shape[0]
                    _ = fused_experts(
                        hidden_states[:local_num_tokens],
                        w1,
                        w2,
                        tw,
                        ti,
                        inplace=True,
                        quant_config=quant_config,
                        global_num_experts=num_experts,
                        expert_map=expert_map,
                    )
            else:
                _ = fused_experts(
                    hidden_states,
                    w1,
                    w2,
                    topk_weights,
                    topk_ids,
                    inplace=True,
                    quant_config=quant_config,
                    global_num_experts=num_experts,
                    expert_map=expert_map,
                )

        def run_iterations(use_cuda_graph=False):
            # Use benchmark_with_power context manager
            with benchmark_with_power(
                device=device,
                kernel_func=run_single_iteration,
                num_warmups=num_warmups,
                num_runs=num_runs,
                repeat_n=1,
            ) as results:
                pass

            return results["latency_ms"] / num_iter, results["power_stats"]

        try:
            latency, power_stats = run_iterations(use_cuda_graph=False)
        except torch.OutOfMemoryError:
            # If OOM, check if we had at least one successful run.
            if num_tokens_idx > 0:
                break
            raise

        print(f"moe latency: {latency}")

        source = "vllm_fused_moe"

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
            framework="VLLM",
            version=vllm_version,
            device_name=torch.cuda.get_device_name(device),
            op_name="moe",
            kernel_source=source,
            perf_filename=perf_filename,
            power_stats=power_stats,
        )


if __name__ == "__main__":
    test_cases = get_moe_test_cases()
    print(f"Total test cases: {len(test_cases)}")

    for test_case in test_cases:
        print(f"Running test case: {test_case}")
        try:
            run_moe_torch(*test_case)
        except Exception as e:
            print(f"Test case failed: {test_case}")
            print(f"Error: {e}")
            continue
