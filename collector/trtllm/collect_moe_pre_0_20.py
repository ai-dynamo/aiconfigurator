# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import tensorrt_llm
import torch
import torch.nn.functional as F
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.modules.fused_moe import FusedMoE, RenormalizeMoeRoutingMethod
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig
from torch.nn.parameter import Parameter

from helper import get_sm_version, log_perf


def balanced_logits(num_tokens, num_experts, topk):
    h_selected_experts = -torch.ones([num_tokens, topk])
    stride = math.ceil(num_experts / topk)

    for token_i in range(num_tokens):
        for i in range(topk):
            if num_tokens >= stride:
                h_selected_experts[token_i][i] = (token_i + i * stride) % num_experts
            else:
                h_selected_experts[token_i][i] = (token_i * stride / num_tokens + i * stride) % num_experts

    expert_map = F.one_hot(h_selected_experts.long(), num_classes=num_experts).sum(1)
    router_logits = F.softmax(expert_map.bfloat16(), dim=1)
    return router_logits


def get_moe_test_cases():
    num_tokens = [
        1,
        2,
        4,
        8,
        16,
        32,
        48,
        64,
        80,
        96,
        128,
        160,
        192,
        256,
        320,
        384,
        512,
        768,
        1024,
        1536,
        2048,
        3072,
        4096,
        6144,
        8192,
        12288,
        16384,
        20480,
    ]
    tp_list = [1, 2, 4, 8, 16, 32]
    ep_list = [1, 2, 4, 8, 16, 32, 64, 128, 160, 256]
    num_gpu_list = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    # hidden_size,inter_s,topk,num_expert, gated act
    # [15360,30720,2,16],# GPT-MOE-1.8T
    # [15360,3840,16,128],# GPT-MOE-1.8T-FineGrained
    # [3584,2560,8,64],# Qwen2-57B
    # [2048,1408,4,60], #qwen1.5_moe
    # [2048,1408,6,64], #deepseekv1_moe
    # [5120,1536,6,160], #deepseekv2
    model_config_list = [
        [4096, 14336, 2, 8],  # mixtral_8x7b
        [6144, 16384, 2, 8],  # mixtral_8x22b
        [7168, 2048, 8, 256],  # deepseekv3, will have 1 shared expert
        [4096, 1536, 8, 128],  # qwen3-moe, 235b-a22b
        [6144, 2560, 8, 160],  # qwen3-moe, 480b-a35b
        [7168, 2048, 8, 384],  # kimi k2
    ]
    moe_list = ["float16"]

    if get_sm_version() > 86:
        moe_list += ["fp8"]
        if get_sm_version() < 100:
            moe_list += ["w4afp8", "fp8_block"]

    if get_sm_version() >= 100:
        moe_list += ["nvfp4"]

    test_cases = []

    for num_gpu in num_gpu_list:  # starting from fewer gpus. workaround for potential buffer bug in moe impl.
        for moe_type in moe_list:
            for num_token in num_tokens:
                for model_config in model_config_list:
                    hs, inter_s, topk, num_experts = model_config
                    for tp in tp_list:
                        for ep in ep_list:
                            if tp * ep != num_gpu:
                                continue
                            if ep > num_experts:
                                continue
                            # we need to ensure inter_s can be divided by tp.
                            if inter_s % tp != 0:
                                continue
                            # w4afp8 requires k shape to be multiple of 128
                            if moe_type == "w4afp8" and inter_s // tp % 128 != 0:
                                continue
                            if moe_type == "nvfp4":
                                if inter_s // tp % 128 == 0:
                                    test_cases.append(
                                        [
                                            moe_type,
                                            num_token,
                                            hs,
                                            inter_s,
                                            topk,
                                            num_experts,
                                            tp,
                                            ep,
                                            True,
                                            "moe_perf.txt",
                                        ]
                                    )
                                    test_cases.append(
                                        [
                                            moe_type,
                                            num_token,
                                            hs,
                                            inter_s,
                                            topk,
                                            num_experts,
                                            tp,
                                            ep,
                                            False,
                                            "moe_perf.txt",
                                        ]
                                    )
                                continue
                            test_cases.append(
                                [
                                    moe_type,
                                    num_token,
                                    hs,
                                    inter_s,
                                    topk,
                                    num_experts,
                                    tp,
                                    ep,
                                    False,
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

    num_warmups = 3
    num_runs = 6

    # capture
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        moe.forward(hidden_states, router_logits, cutlass_min_latency_mode=cutlass_min_latency_mode)
    # warmup
    for i in range(num_warmups):
        g.replay()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for i in range(num_runs):
        g.replay()
    end_event.record()
    torch.cuda.synchronize()
    latency = start_event.elapsed_time(end_event) / num_runs

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
    )
