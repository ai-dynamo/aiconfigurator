# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import os

import tensorrt_llm
import torch
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.modules.fused_moe import FusedMoE, RenormalizeMoeRoutingMethod
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig
from torch.nn.parameter import Parameter

try:
    from common_test_cases import get_common_moe_test_cases

    from helper import (
        balanced_logits,
        benchmark_with_power,
        get_sm_version,
        log_perf,
    )
except ModuleNotFoundError:
    import os
    import sys

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from common_test_cases import get_common_moe_test_cases

    from helper import (
        balanced_logits,
        benchmark_with_power,
        get_sm_version,
        log_perf,
    )


def get_moe_test_cases():
    moe_list = ["float16"]
    if get_sm_version() > 86:
        moe_list += ["fp8"]
        if get_sm_version() < 100:
            moe_list += ["w4afp8", "fp8_block"]

    if get_sm_version() >= 100:
        moe_list += ["nvfp4"]

    test_cases = []

    for common_moe_testcase in get_common_moe_test_cases():
        model_name = common_moe_testcase.model_name
        inter_s = common_moe_testcase.inter_size
        moe_tp = common_moe_testcase.tp

        if common_moe_testcase.token_expert_distribution != "balanced":
            continue

        for moe_type in moe_list:
            if model_name in ["openai/gpt-oss-20b", "openai/gpt-oss-120b"]:
                if moe_type != "w4a16_mxfp4":
                    continue
            else:
                if moe_type == "w4a16_mxfp4":
                    continue

            # w4afp8 requires k shape to be multiple of 128
            if moe_type == "w4afp8" and inter_s // moe_tp % 128 != 0:
                continue

            min_latency_mode_options = [False]

            if moe_type == "nvfp4":
                if inter_s // moe_tp % 128 != 0:
                    continue
                # FIXME: recent version only supports SM100 for min-latency mode.
                # current support, DS router only support up to 256 experts.
                # Renormalize router only support <=128 experts. trtllmgen kernels only
                # support renormalize, ds and llama router.
                if get_sm_version() == 100 and common_moe_testcase.num_experts <= 256:
                    min_latency_mode_options.append(True)

            for min_latency_mode in min_latency_mode_options:
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
                        min_latency_mode,
                        "moe_perf.txt",
                    ]
                )

    return test_cases


def run_moe_torch(
    moe_type,
    num_tokens,
    hidden_size,
    inter_size,
    topk,
    num_experts,
    moe_tp_size,
    moe_ep_size,
    cutlass_min_latency_mode,
    perf_filename,
    device="cuda:0",
):
    torch.cuda.set_device(device)
    torch.set_default_device(device)

    # moe type support float16, fp8_qdq, fp8_block, w4a8, nvfp4(not implemented yet)
    dtype = torch.bfloat16
    quant_algo = None
    if moe_type == "fp8_block":
        quant_algo = QuantAlgo.FP8_BLOCK_SCALES
        dtype = torch.float8_e4m3fn
    elif moe_type == "w4afp8":
        quant_algo = QuantAlgo.W4A8_AWQ
        dtype = torch.float8_e4m3fn
    elif moe_type == "fp8":
        quant_algo = QuantAlgo.FP8
        dtype = torch.float8_e4m3fn
    elif moe_type == "nvfp4":
        quant_algo = QuantAlgo.NVFP4

    quant_group_size = 128
    if moe_type == "nvfp4":
        quant_group_size = 16

    quant_config = QuantConfig(
        quant_algo=quant_algo,
        kv_cache_quant_algo=None,
        group_size=quant_group_size,  # need to evaluate the impact of group size
        smoothquant_val=0.5,
        clamp_val=None,
        use_meta_recipe=False,
        has_zero_point=False,
        pre_quant_scale=False,
        exclude_modules=None,
    )

    # parallel mapping
    mapping = Mapping()
    mapping.moe_ep_size = moe_ep_size
    mapping.moe_tp_size = moe_tp_size

    model_config = ModelConfig()
    model_config.mapping = mapping
    model_config.quant_config = quant_config
    model_config.moe_max_num_tokens = 65536  # to avoid multi-chunk auxi stream in cuda-graph mode.

    routing_method = RenormalizeMoeRoutingMethod(topk)

    moe = FusedMoE(
        routing_method=routing_method,
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=inter_size,
        dtype=dtype,
        # In both low latency and attention dp scenarios, FusedMoE needs not to do allreduce
        # inside op.
        reduce_results=False,
        model_config=model_config,
    )

    hidden_states = torch.randn([num_tokens, hidden_size]).bfloat16().to(torch.device(device))
    router_logits = balanced_logits(num_tokens, num_experts, topk).bfloat16().to(torch.device(device))

    ffn1_weights = Parameter(
        torch.randn(moe.w3_w1_weight.shape, dtype=torch.bfloat16, device=torch.device(device)).to(
            dtype=moe.w3_w1_weight.dtype
        ),
        requires_grad=False,
    )
    ffn2_weights = Parameter(
        torch.randn(moe.w2_weight.shape, dtype=torch.bfloat16, device=torch.device(device)).to(
            dtype=moe.w2_weight.dtype
        ),
        requires_grad=False,
    )

    moe.w3_w1_weight = ffn1_weights
    moe.w2_weight = ffn2_weights

    # Define kernel function for benchmarking
    def kernel_func():
        moe.forward(hidden_states, router_logits, cutlass_min_latency_mode=cutlass_min_latency_mode)

    # Benchmark with automatic power measurement and graph fallback
    with benchmark_with_power(
        device=device,
        kernel_func=kernel_func,
        num_warmups=3,
        num_runs=6,
        repeat_n=1,
        allow_graph_fail=True,  # Enable graceful fallback to eager execution
    ) as results:
        latency = results["latency_ms"]
        power_stats = results["power_stats"]

    if cutlass_min_latency_mode:
        source = "moe_torch_flow_min_latency"
    else:
        source = "moe_torch_flow"

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
                "distribution": "uniform",
                "latency": latency,
            }
        ],
        framework="TRTLLM",
        version=tensorrt_llm.__version__,
        device_name=torch.cuda.get_device_name(device),
        op_name="moe",
        kernel_source=source,
        perf_filename=perf_filename,
        power_stats=power_stats,
    )
