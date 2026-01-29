# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

import tensorrt_llm
import torch
from tensorrt_llm._torch.autotuner import AutoTuner, autotune
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_deepseekv3 import DeepseekV3Gate
from tensorrt_llm._torch.modules.fused_moe import RenormalizeMoeRoutingMethod, create_moe
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
        power_law_logits_v3,
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
        power_law_logits_v3,
    )

aic_debug = int(os.getenv("aic_moe_debug", "0"))  # noqa: SIM112


def get_moe_test_cases():
    moe_list = ["float16"]
    if get_sm_version() > 86:
        moe_list += ["fp8"]
        if get_sm_version() < 100:
            # though trtllm gen kernel source supports fp8_block, it only provides min-latency
            # data. not practical.
            moe_list += [
                "w4afp8",
                "fp8_block",
            ]

    if get_sm_version() >= 100:
        moe_list += ["nvfp4"]

    test_cases = []

    for common_moe_testcase in get_common_moe_test_cases():
        model_name = common_moe_testcase.model_name
        inter_s = common_moe_testcase.inter_size
        moe_tp = common_moe_testcase.tp

        if common_moe_testcase.token_expert_distribution != "power_law":
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
    min_latency_mode,
    model_name,
    perf_filename,
    distributed="power_law",
    power_law_alpha=0.0,
    device="cuda:0",
):
    device = torch.device(device)
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
    model_config.moe_max_num_tokens = num_tokens_lists[-1]  # to avoid multi-chunk auxi stream in cuda-graph mode.
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
    )

    ffn1_weights = Parameter(
        torch.randn(moe.w3_w1_weight.shape, dtype=torch.bfloat16, device=device).to(dtype=moe.w3_w1_weight.dtype),
        requires_grad=False,
    )
    ffn2_weights = Parameter(
        torch.randn(moe.w2_weight.shape, dtype=torch.bfloat16, device=device).to(dtype=moe.w2_weight.dtype),
        requires_grad=False,
    )

    moe.w3_w1_weight = ffn1_weights
    moe.w2_weight = ffn2_weights

    max_index = -1
    while True:
        try:
            hidden_states_max_tokens = torch.randn([num_tokens_lists[max_index], hidden_size], device=device).bfloat16()
            logits_max_tokens = torch.randn([num_tokens_lists[max_index], num_experts], device=device).to(
                router_logits_dtype
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
            max_index -= 1
            if max_index == -len(num_tokens_lists):
                raise ValueError("tune failed for all tokens sizes") from e
            continue

    for num_tokens in num_tokens_lists:
        hidden_states = torch.randn([num_tokens, hidden_size], device=device).bfloat16()

        num_iter = 5 if distributed == "power_law" else 1
        if distributed == "power_law":
            actual_logits_list = [
                power_law_logits_v3(num_tokens, num_experts, topk, moe_ep_size, power_law_alpha).to(router_logits_dtype)
                for _ in range(num_iter)
            ]
        elif distributed == "balanced":
            actual_logits = balanced_logits(num_tokens, num_experts, topk).to(router_logits_dtype)
        else:
            raise ValueError(f"Unsupported distributed mode: {distributed}")

        # Helper closure to encapsulate forward pass logic (reduces duplication)
        def run_forward_pass():
            """Execute one forward pass through MOE, handling both power_law and balanced modes."""
            if distributed == "power_law":
                for logits in actual_logits_list:
                    moe.forward(hidden_states, logits, do_finalize=not min_latency_mode)
            else:
                moe.forward(hidden_states, actual_logits, do_finalize=not min_latency_mode)

        # Benchmark with automatic power measurement and graph fallback
        num_warmups = 1 if distributed == "power_law" else 3
        num_runs = 1 if distributed == "power_law" else 6

        # Use benchmark_with_power with graceful graph fallback
        with benchmark_with_power(
            device=device,
            kernel_func=run_forward_pass,
            num_warmups=num_warmups,
            num_runs=num_runs,
            repeat_n=1,
            allow_graph_fail=True,  # Enable graceful fallback to eager execution
        ) as results:
            # Calculate per-iteration latency (accounting for internal iterations)
            latency = results["latency_ms"] / num_iter
            power_stats = results["power_stats"]

            # Log if CUDA graph capture failed (for debugging)
            if not results["used_cuda_graph"] and aic_debug == 1:
                print(f"CUDA graph capture failed for {num_tokens} tokens, used eager execution fallback")

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
            power_stats=power_stats,
        )
