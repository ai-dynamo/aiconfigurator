# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import gc
import glob
import inspect
import json
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from types import SimpleNamespace

import tensorrt_llm
import torch
from tensorrt_llm._torch.autotuner import AutoTuner, autotune
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_deepseekv3 import DeepseekV3Gate
from tensorrt_llm._torch.modules.fused_moe import RenormalizeMoeRoutingMethod, create_moe
from tensorrt_llm._torch.utils import ActivationType
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig

# Models that use non-gated MoE (Relu2 activation instead of SwiGLU)
# These are substring patterns that will be matched against the full model name
NON_GATED_MOE_MODELS = ["Nemotron-3"]

from collector.common_test_cases import get_common_moe_test_cases
from collector.helper import (
    EXIT_CODE_RESTART,
    balanced_logits,
    benchmark_with_power,
    get_sm_version,
    log_perf,
    power_law_logits_v3,
)

aic_debug = int(os.getenv("aic_moe_debug", "0"))  # noqa: SIM112
AIC_RESTART_WORKER = int(os.getenv("AIC_RESTART_WORKER", "0"))

moe_tune_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "moe_tuned_cache_path")


def gc_collect():
    for _ in range(2):
        gc.collect()
        torch.cuda.empty_cache()


def _process_json_file(file_path):
    """Process a single JSON file, returning (deleted, message) tuple."""
    try:
        if os.path.getsize(file_path) == 0:
            os.remove(file_path)
            return (True, f"Deleted empty file: {file_path}")
        else:
            with open(file_path) as f:
                data = json.load(f)
                if not data:
                    os.remove(file_path)
                    return (True, f"Deleted empty JSON content: {file_path}")
        return (False, None)
    except (OSError, json.JSONDecodeError) as e:
        try:
            os.remove(file_path)
            return (True, f"Deleted invalid file: {file_path} (Error: {e})")
        except OSError:
            return (False, None)


def cleanup_empty_json_files(directory):
    if not os.path.exists(directory):
        return

    json_files = glob.glob(os.path.join(directory, "*.json"))
    deleted_count = 0

    # Parallelize io operations
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(_process_json_file, fp): fp for fp in json_files}
        for future in as_completed(futures):
            deleted, message = future.result()
            if deleted:
                deleted_count += 1
            if message:
                print(message)

    if deleted_count > 0:
        print(f"Total deleted {deleted_count} invalid JSON files from {directory}")


def get_moe_test_cases():
    moe_list = ["float16"]
    if get_sm_version() > 86:
        moe_list += ["fp8"]
        if get_sm_version() < 100:
            # though trtllm gen kernel source supports fp8_block, it only provides min-latency
            # data. not practical.
            moe_list += [
                # "w4afp8", FIXME: trtllm 1.2 has bugs for w4afp8
                "fp8_block",
                "w4a16_mxfp4",
            ]

    if get_sm_version() >= 100:
        moe_list += ["nvfp4"]

    test_cases = []

    for common_moe_testcase in get_common_moe_test_cases():
        model_name = common_moe_testcase.model_name
        inter_s = common_moe_testcase.inter_size
        moe_tp = common_moe_testcase.tp

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

            # TLLM_CHECK_WITH_INFO(inter_size % (256 / sizeof_bits<WeightType>::value) == 0
            weight_bits = {
                "float16": 16,
                "fp8": 8,
                "fp8_block": 8,
                "w4a16_mxfp4": 4,
                "w4afp8": 4,
                "nvfp4": 4,
            }[moe_type]
            if (inter_s // moe_tp) % (256 // weight_bits) != 0:
                continue

            min_latency_mode_options = [False]

            if moe_type == "nvfp4" and get_sm_version() == 100 and common_moe_testcase.num_experts <= 256:
                # FIXME: recent version only supports SM100 for min-latency mode.
                # current support, DS router only support up to 256 experts.
                # Renormalize router only support <=128 experts. trtllmgen kernels only
                # support renormalize, ds and llama router.
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

    # Try to optimize number of autotune cache hits by shuffling test cases.
    # This makes sure the same cache keys are far apart from each other.
    random.seed(42)
    random.shuffle(test_cases)

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

    gc_collect()

    if aic_debug == 1:
        print("MOE Allocated GDRAM:", torch.cuda.memory_allocated(device.index) / 1024**2, "MB")
        print("MOE Reserved GDRAM:", torch.cuda.memory_reserved(device) / 1024**2, "MB")
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

    if power_law_alpha - 0.0 < 1e-6:
        distributed = "balanced"

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

    # Create a minimal pretrained_config with required attributes for TensorRT-LLM 1.3+
    # The CommunicationFactory.create_strategy() accesses model_config.pretrained_config.hidden_size
    pretrained_config = SimpleNamespace(
        hidden_size=hidden_size,
        intermediate_size=inter_size,
        num_experts=num_experts,
        torch_dtype=torch.bfloat16,
    )

    model_config = ModelConfig(pretrained_config=pretrained_config)
    model_config.mapping = mapping
    model_config.quant_config = quant_config
    model_config.moe_max_num_tokens = num_tokens_lists[-1]  # to avoid multi-chunk auxi stream in cuda-graph mode.
    swiglu_alpha = None
    swiglu_beta = None
    swiglu_limit = None

    # Determine activation type based on model
    # Nemotron-3 Nano uses non-gated MoE with Relu2 activation
    # Other models (DeepSeek, Qwen, Mixtral) use gated MoE with SwiGLU activation
    if any(pattern in model_name for pattern in NON_GATED_MOE_MODELS):
        activation_type = ActivationType.Relu2
        is_gated = False
    else:
        activation_type = ActivationType.Swiglu
        is_gated = True

    if model_name in ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]:
        # use triton backend for best performance on Hopper
        model_config.moe_backend = "triton"
        swiglu_alpha = torch.tensor([1.702] * (num_experts // moe_ep_size), dtype=torch.float32).to(
            torch.device(device)
        )
        swiglu_beta = torch.tensor([1.0] * (num_experts // moe_ep_size), dtype=torch.float32).to(torch.device(device))
        swiglu_limit = torch.tensor([7.0] * (num_experts // moe_ep_size), dtype=torch.float32).to(torch.device(device))
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
        activation_type=activation_type,
    )
    moe.to(torch.device(device))

    if moe_type == "w4a16_mxfp4":
        w1_weight = torch.randn((num_experts, inter_size, hidden_size), dtype=dtype).cuda()
        w2_weight = torch.randn((num_experts, hidden_size, inter_size), dtype=dtype).cuda()
        w3_weight = torch.randn((num_experts, inter_size, hidden_size), dtype=dtype).cuda()
        w1_bias = torch.randn((num_experts, inter_size), dtype=dtype).cuda()
        w2_bias = torch.randn((num_experts, hidden_size), dtype=dtype).cuda()
        w3_bias = torch.randn((num_experts, inter_size), dtype=dtype).cuda()

        from triton_kernels.numerics_details.mxfp import downcast_to_mxfp_torch

        def fp32_to_mxfp4(tensor):
            tensor = tensor.transpose(1, 2).contiguous()
            tensor_fp4, tensor_scales = downcast_to_mxfp_torch(tensor, torch.uint8, axis=1)
            tensor_fp4 = tensor_fp4.transpose(1, 2).contiguous()
            tensor_scales = tensor_scales.transpose(1, 2).contiguous()
            return tensor_fp4, tensor_scales

        w1_weight_fp4, w1_weight_scale = fp32_to_mxfp4(w1_weight)
        w2_weight_fp4, w2_weight_scale = fp32_to_mxfp4(w2_weight)
        w3_weight_fp4, w3_weight_scale = fp32_to_mxfp4(w3_weight)

        weights = {}
        for expert_id in range(num_experts):
            weights[f"{expert_id}.w1.weight"] = w1_weight_fp4[expert_id]
            weights[f"{expert_id}.w2.weight"] = w2_weight_fp4[expert_id]
            weights[f"{expert_id}.w3.weight"] = w3_weight_fp4[expert_id]
            weights[f"{expert_id}.w1.weight_scale"] = w1_weight_scale[expert_id]
            weights[f"{expert_id}.w2.weight_scale"] = w2_weight_scale[expert_id]
            weights[f"{expert_id}.w3.weight_scale"] = w3_weight_scale[expert_id]
            weights[f"{expert_id}.w1.bias"] = w1_bias[expert_id]
            weights[f"{expert_id}.w2.bias"] = w2_bias[expert_id]
            weights[f"{expert_id}.w3.bias"] = w3_bias[expert_id]
        moe.load_weights([weights])

    # dry run
    torch.cuda.synchronize()
    max_tokens = num_tokens_lists[-1]
    for i in range(len(num_tokens_lists)):
        max_tokens = num_tokens_lists[-i - 1]
        try:
            hidden_states_max_tokens = torch.randn([max_tokens, hidden_size]).bfloat16().to(torch.device(device))
            logits_max_tokens = balanced_logits(max_tokens, num_experts, topk).to(router_logits_dtype)
            moe.forward(hidden_states_max_tokens, logits_max_tokens, do_finalize=not min_latency_mode)
            torch.cuda.synchronize()
            if aic_debug == 1:
                print(f"Successfully dry run for {max_tokens} tokens")
            break
        except Exception as e:
            if i == len(num_tokens_lists) - 1:
                RuntimeError(f"dry run failed for {max_tokens} tokens: {e}")
            else:
                continue

    if moe_type != "w4a16_mxfp4":
        cleanup_empty_json_files(moe_tune_path)
        cache_path = (
            f"{moe_tune_path}/{moe_type}_{hidden_size}_{inter_size // moe_tp_size}_{num_experts // moe_ep_size}"
        )
        existing_files = glob.glob(f"{cache_path}*")
        cache_loaded = False
        if existing_files:
            json_path = existing_files[0]
            try:
                AutoTuner.get().profiling_cache.load_cache(json_path)
                cache_loaded = True
                print(f"Loaded profiling cache from {json_path}")
            except (OSError, json.JSONDecodeError):
                pass

        if not cache_loaded:
            torch.cuda.synchronize()
            for i in range(len(num_tokens_lists)):
                max_tokens_for_tuning = num_tokens_lists[-i - 1]
                if max_tokens_for_tuning > max_tokens:
                    continue
                else:
                    try:
                        # Check if autotune() accepts "rank" kwarg.
                        # It was removed in trtllm 1.2.0rc6.
                        autotune_kwargs = {"cache_path": cache_path}
                        if "rank" in inspect.signature(autotune).parameters:
                            autotune_kwargs["rank"] = torch.device(device).index

                        with torch.inference_mode(), autotune(**autotune_kwargs):
                            moe.forward(
                                hidden_states_max_tokens[:max_tokens_for_tuning],
                                logits_max_tokens[:max_tokens_for_tuning],
                                do_finalize=not min_latency_mode,
                            )
                        torch.cuda.synchronize()
                    except Exception as e:
                        print(f"tune failed for {max_tokens_for_tuning} tokens: {e}, fallback to samller tokens")
                        continue

    del hidden_states_max_tokens, logits_max_tokens

    for num_tokens in num_tokens_lists:
        # gc_collect()

        if num_tokens > max_tokens:
            continue
        hidden_states = torch.randn([num_tokens, hidden_size]).bfloat16().to(torch.device(device))
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

        # ═══════════════════════════════════════════════════════════════════════════════
        # Helper closure to encapsulate forward pass logic (reduces duplication)
        # ═══════════════════════════════════════════════════════════════════════════════
        def run_forward_pass():
            """Execute one forward pass through MOE, handling both power_law and balanced modes."""
            if distributed == "power_law":
                for logits in actual_logits_list:
                    moe.forward(hidden_states, logits, do_finalize=not min_latency_mode)  # noqa: F821
            else:
                moe.forward(hidden_states, actual_logits, do_finalize=not min_latency_mode)  # noqa: F821

        # ═══════════════════════════════════════════════════════════════════════════════
        # Benchmark with automatic power measurement and graph fallback
        # ═══════════════════════════════════════════════════════════════════════════════
        # Determine base warmups and runs based on distribution mode
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
        elif not is_gated:
            source = "moe_torch_flow_nongated"  # non-gated MoE (relu2)
        else:
            source = "moe_torch_flow"  # cutlass (gated SwiGLU)

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
        if distributed == "power_law":
            del actual_logits_list
        else:
            del actual_logits

        del hidden_states

    # Exit the worker process after completing MOE task to ensure complete resource cleanup
    # This forces OS to reclaim all GPU memory, CUDA context, and other resources
    if AIC_RESTART_WORKER:
        sys.exit(EXIT_CODE_RESTART)


if __name__ == "__main__":
    test_cases = get_moe_test_cases()
    for test_case in test_cases:
        run_moe_torch(*test_case)
