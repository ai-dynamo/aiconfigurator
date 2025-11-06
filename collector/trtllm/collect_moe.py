# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
import os

import tensorrt_llm
import torch
import torch.nn.functional as F
from tensorrt_llm._torch.autotuner import AutoTuner, autotune
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_deepseekv3 import DeepseekV3Gate
from tensorrt_llm._torch.modules.fused_moe import RenormalizeMoeRoutingMethod, create_moe
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig
from torch.nn.parameter import Parameter

from helper import get_sm_version, log_perf

aic_debug = int(os.getenv("aic_moe_debug", "0"))  # noqa: SIM112


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


def sample_power_law(size, alpha, xmin, xmax):
    u = torch.rand(size)
    inv_cdf = ((xmax ** (1 - alpha) - xmin ** (1 - alpha)) * u + xmin ** (1 - alpha)) ** (1 / (1 - alpha))
    return inv_cdf


def power_law_logits_v3(num_tokens, num_experts, topk, ep, alpha):
    if num_tokens * topk > num_experts:
        num_tokens_per_expert = sample_power_law(num_experts, alpha, 1, num_tokens * 0.8)
    else:
        num_tokens_per_expert = sample_power_law(num_experts, alpha, 0.01, 2)

    target_sum = num_tokens * topk

    original_distribution = num_tokens_per_expert / num_tokens_per_expert.sum()

    target_distribution = original_distribution * target_sum

    num_tokens_per_expert = torch.round(target_distribution).to(torch.int64)

    current_sum = num_tokens_per_expert.sum().item()
    delta = target_sum - current_sum
    if delta != 0:
        sorted_indices = torch.argsort(num_tokens_per_expert, descending=True)

        if delta > 0:
            for i in range(delta):
                expert_idx = sorted_indices[i % len(sorted_indices)]
                num_tokens_per_expert[expert_idx] += 1
        else:
            for i in range(-delta):
                expert_idx = sorted_indices[-(i % len(sorted_indices)) - 1]
                if num_tokens_per_expert[expert_idx] > 0:
                    num_tokens_per_expert[expert_idx] -= 1
                else:
                    num_tokens_per_expert[torch.argmax(num_tokens_per_expert)] -= 1

    if len(num_tokens_per_expert) > 1:
        sorted_tokens = torch.sort(num_tokens_per_expert, descending=True)[0]
        assert sorted_tokens[0] >= sorted_tokens[-1], "Power law distribution pattern disrupted"

    with torch.no_grad():
        conv1d = torch.nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=num_experts // ep,
            stride=num_experts // ep,
            padding=0,
            bias=False,
        )
        conv1d_weights = torch.tensor([1 for _ in range(num_experts // ep)])
        conv1d.weight.copy_(conv1d_weights)

    res = conv1d(num_tokens_per_expert.unsqueeze(0).unsqueeze(0).float())
    max_ep_idx = torch.argmax(res).item()

    if max_ep_idx != 0:
        ep_group_size = num_experts // ep
        num_tokens_per_expert_reshaped = num_tokens_per_expert.view(ep, ep_group_size)
        num_tokens_per_expert_reshaped[0], num_tokens_per_expert_reshaped[max_ep_idx] = (
            num_tokens_per_expert_reshaped[max_ep_idx].clone(),
            num_tokens_per_expert_reshaped[0].clone(),
        )
        num_tokens_per_expert = num_tokens_per_expert_reshaped.view(-1)

    if aic_debug == 2:
        print("num_tokens_per_expert", num_tokens_per_expert, num_tokens_per_expert.sum().item())

    _, num_tokens_per_expert_sorted_index = torch.sort(num_tokens_per_expert, descending=True)
    expert_assignments = []
    num_tokens_per_expert_sorted_index_lists = num_tokens_per_expert_sorted_index.tolist()
    for expert_id in num_tokens_per_expert_sorted_index_lists:
        expert_assignments.extend([expert_id] * num_tokens_per_expert[expert_id])

    expert_assignments = torch.tensor(expert_assignments, dtype=torch.long)
    h_selected_experts = expert_assignments.reshape(topk, num_tokens).T

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
        # 32768,
        # 65536,
    ]
    tp_list = [1, 2, 4, 8, 16, 32]
    ep_list = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    num_gpu_list = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    alpha_list = [1.01, 1.2]
    # hidden_size,inter_s,topk,num_expert, gated act
    # [15360,30720,2,16],# GPT-MOE-1.8T
    # [15360,3840,16,128],# GPT-MOE-1.8T-FineGrained
    # [3584,2560,8,64],# Qwen2-57B
    # [2048,1408,4,60], #qwen1.5_moe
    # [2048,1408,6,64], #deepseekv1_moe
    # [5120,1536,6,160], #deepseekv2
    model_config_list = [
        [4096, 14336, 2, 8, "MOE_Mixtral8x7B"],  # mixtral_8x7b
        [6144, 16384, 2, 8, "MOE_Mixtral8x22B"],  # mixtral_8x22b
        [7168, 2048, 8, 256, "DEEPSEEK_V3"],  # deepseekv3, will have 1 shared expert
        [2048, 768, 8, 128, "QWEN3_30B_A3B"],  # qwen3-moe, 30b-a3b
        [4096, 1536, 8, 128, "QWEN3_235B"],  # qwen3-moe, 235b-a22b
        [6144, 2560, 8, 160, "QWEN3_480B"],  # qwen3-moe, 480b-a35b
        [7168, 2048, 8, 384, "KIMI_K2"],  # kimi k2
        [2880, 2880, 4, 128, "GPT_OSS_120B"],
        [2880, 2880, 4, 32, "GPT_OSS_20B"],
    ]

    moe_list = ["float16"]

    if get_sm_version() > 86:
        moe_list += ["fp8"]
        if get_sm_version() < 100:
            # though trtllm gen kernel source supports fp8_block, it only provides min-latency
            # data. not practical.
            moe_list += [
                "w4afp8",
                "fp8_block",
                "w4a16_mxfp4",
            ]

    if get_sm_version() >= 100:
        moe_list += ["nvfp4"]

    test_cases = []

    # currently, we support max-throughput for typical quantizations. support min-latency for nvfp4.
    for num_gpu in num_gpu_list:  # starting from fewer gpus. workaround for potential buffer bug in moe impl.
        for moe_type in moe_list:
            for model_config in model_config_list:
                hs, inter_s, topk, num_experts, model_name = model_config
                if model_name in ["GPT_OSS_20B", "GPT_OSS_120B"]:
                    if moe_type != "w4a16_mxfp4":
                        continue
                else:
                    if moe_type == "w4a16_mxfp4":
                        continue
                for tp in tp_list:
                    # QWEN3_30B_A3B: exclude tp >= 8 as they are not used for actual deployments
                    if model_name == "QWEN3_30B_A3B" and tp >= 8:
                        continue
                    for ep in ep_list:
                        if tp * ep != num_gpu:
                            continue
                        if ep > num_experts:
                            continue
                        if num_experts % ep != 0:
                            continue
                        # we need to ensure inter_s can be divided by tp.
                        if inter_s % tp != 0:
                            continue
                        # w4afp8 requires k shape to be multiple of 128
                        if moe_type == "w4afp8" and inter_s // tp % 128 != 0:
                            continue
                        if moe_type == "nvfp4":
                            if inter_s // tp % 128 != 0:
                                continue
                            # FIXME: recent version only supports SM100 for min-latency mode.
                            # current support, DS router only support up to 256 experts.
                            # Renormalize router only support <=128 experts. trtllmgen kernels only
                            # support renormalize, ds and llama router.
                            if get_sm_version() == 100 and num_experts <= 256:
                                for power_law_alpha in alpha_list:
                                    test_cases.append(
                                        [
                                            moe_type,
                                            num_tokens,
                                            hs,
                                            inter_s,
                                            topk,
                                            num_experts,
                                            tp,
                                            ep,
                                            True,
                                            model_name,
                                            "moe_perf.txt",
                                            "power_law",
                                            power_law_alpha,
                                        ]
                                    )
                                    # test_cases.append(
                                    #     [
                                    #         moe_type,
                                    #         num_tokens,
                                    #         hs,
                                    #         inter_s,
                                    #         topk,
                                    #         num_experts,
                                    #         tp,
                                    #         ep,
                                    #         True,
                                    #         model_name,
                                    #         "moe_perf.txt",
                                    #         "balanced",
                                    #         0,
                                    #     ]
                                    # )

                        for power_law_alpha in alpha_list:
                            test_cases.append(
                                [
                                    moe_type,
                                    num_tokens,
                                    hs,
                                    inter_s,
                                    topk,
                                    num_experts,
                                    tp,
                                    ep,
                                    False,
                                    model_name,
                                    "moe_perf.txt",
                                    "power_law",
                                    power_law_alpha,
                                ]
                            )
                        # test_cases.append(
                        #     [
                        #         moe_type,
                        #         num_tokens,
                        #         hs,
                        #         inter_s,
                        #         topk,
                        #         num_experts,
                        #         tp,
                        #         ep,
                        #         False,
                        #         model_name,
                        #         "moe_perf.txt",
                        #         "balanced",
                        #         0,
                        #     ]
                        # )
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
    min_latency_mode,
    model_name,
    perf_filename,
    distributed="power_law",
    power_law_alpha=0.0,
    device="cuda:0",
):
    torch.cuda.set_device(device)
    torch.set_default_device(device)

    # moe type support float16, fp8_qdq, fp8_block, w4a8, nvfp4(not implemented yet)
    dtype = torch.bfloat16
    quant_group_size = 128
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
        quant_group_size = 16
    elif moe_type == "w4a16_mxfp4":
        quant_algo = QuantAlgo.W4A16_MXFP4
        quant_group_size = 32

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
    model_config.moe_max_num_tokens = num_tokens_lists[-1]  # to avoid multi-chunk auxi stream in cuda-graph mode.
    swiglu_alpha = None
    swiglu_beta = None
    swiglu_limit = None

    if model_name in ["GPT_OSS_120B", "GPT_OSS_20B"]:
        # use triton backend for best performance on Hopper
        model_config.moe_backend = "triton"
        swiglu_alpha = torch.tensor([1.702] * (num_experts // moe_ep_size), dtype=torch.float32).cuda()
        swiglu_beta = torch.tensor([1.0] * (num_experts // moe_ep_size), dtype=torch.float32).cuda()
        swiglu_limit = torch.tensor([7.0] * (num_experts // moe_ep_size), dtype=torch.float32).cuda()
        if 86 < get_sm_version() < 100:
            model_config.moe_backend = "triton"
        else:
            model_config.moe_backend = "cutlass" if not min_latency_mode else "trtllm"
    else:
        model_config.moe_backend = "cutlass" if not min_latency_mode else "trtllm"

    router_logits_dtype = torch.bfloat16
    # current min_latency mode only support experts <= 256. Thus K2 will not have min_latency mode.
    if min_latency_mode:
        # FIXME: all use deepseek setting for now.
        n_group = 8
        topk_group = 4
        routed_scaling_factor = 2.5

        routing_method = DeepseekV3Gate(
            hidden_size,
            num_experts,
            top_k=topk,
            n_group=n_group,
            topk_group=topk_group,
            routed_scaling_factor=routed_scaling_factor,
            dtype=dtype,
            moe_backend="TRTLLM",
        ).routing_method
        router_logits_dtype = torch.float32
    else:
        # for low latency mode in fp4, experts > 128 is not supported.
        routing_method = RenormalizeMoeRoutingMethod(topk)

    moe = create_moe(
        routing_method=routing_method,
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=inter_size,
        dtype=dtype,
        # In both low latency and attention dp scenarios, create_moe needs not to do allreduce
        # inside op.
        reduce_results=False,
        model_config=model_config,
        swiglu_alpha=swiglu_alpha,
        swiglu_beta=swiglu_beta,
        swiglu_limit=swiglu_limit,
    )

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

    max_index = -1
    while True:
        try:
            hidden_states_max_tokens = (
                torch.randn([num_tokens_lists[max_index], hidden_size]).bfloat16().to(torch.device(device))
            )
            logits_max_tokens = (
                torch.randn([num_tokens_lists[max_index], num_experts]).to(router_logits_dtype).to(torch.device(device))
            )
            torch.cuda.synchronize()
            AutoTuner.get().clear_cache()
            with torch.inference_mode(), autotune():
                moe.forward(hidden_states_max_tokens, logits_max_tokens, do_finalize=not min_latency_mode)
            torch.cuda.synchronize()
            if aic_debug == 1:
                print(f"tune success for tokens size {num_tokens_lists[max_index]}")
            break
        except Exception as e:
            if aic_debug == 1:
                print(
                    f"tune failed for tokens size {num_tokens_lists[max_index]}, fallback to "
                    f"tokens size {num_tokens_lists[max_index - 1]}"
                )
            max_index -= -3
            if max_index == -len(num_tokens_lists):
                raise ValueError("tune failed") from e
            continue

    for num_tokens in num_tokens_lists:
        hidden_states = torch.randn([num_tokens, hidden_size]).bfloat16().to(torch.device(device))

        num_iter = 5 if distributed == "power_law" else 1
        if distributed == "power_law":
            actual_logits_list = [
                power_law_logits_v3(num_tokens, num_experts, topk, moe_ep_size, power_law_alpha)
                .to(router_logits_dtype)
                .to(torch.device(device))
                for _ in range(num_iter)
            ]
        elif distributed == "balanced":
            actual_logits = (
                balanced_logits(num_tokens, num_experts, topk).to(router_logits_dtype).to(torch.device(device))
            )
        else:
            raise ValueError(f"Unsupported distributed mode: {distributed}")

        num_warmups = 3
        num_runs = 6
        if distributed == "power_law":
            num_warmups = 1
            num_runs = 1

        # capture
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            if distributed == "power_law":
                for actual_logits in actual_logits_list:
                    moe.forward(hidden_states, actual_logits, do_finalize=not min_latency_mode)
            else:
                moe.forward(hidden_states, actual_logits, do_finalize=not min_latency_mode)
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
        latency = start_event.elapsed_time(end_event) / num_runs / num_iter

        if min_latency_mode:
            source = "moe_torch_flow_min_latency"  # trtllm gen
        else:
            source = "moe_torch_flow"  # cutlass

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
            framework="TRTLLM",
            version=tensorrt_llm.__version__,
            device_name=torch.cuda.get_device_name(device),
            op_name="moe",
            kernel_source=source,
            perf_filename=perf_filename,
        )

    del moe, ffn1_weights, ffn2_weights, hidden_states, actual_logits, actual_logits_list
    torch.cuda.empty_cache()
