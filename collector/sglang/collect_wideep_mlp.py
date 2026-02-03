# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import os

import torch

try:
    from helper import _get_deepseek_model_path, log_perf
except ModuleNotFoundError:
    import os
    import sys

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from helper import _get_deepseek_model_path, log_perf
from importlib.metadata import version as get_version

from sglang.srt.distributed import (
    init_distributed_environment,
    initialize_model_parallel,
)
from sglang.srt.layers.quantization import Fp8Config
from sglang.srt.models.deepseek_v2 import DeepseekV2MLP
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

DEEPSEEK_MODEL_PATH = _get_deepseek_model_path()


def get_mlp_prefill_test_cases():
    """Get test cases for MLP prefill phase
    Returns: list of [quant_type, num_token, hidden_size, intermediate_size]
    """
    test_cases = []

    num_tokens = [
        1,
        2,
        4,
        8,
        16,
        32,
        64,
        128,
        256,
        512,
        1024,
        2048,
        4096,
        8192,
        16384,
        32768,
        65536,
        131072,
    ]
    quant_types = ["fp8_block"]
    hidden_size = 7168
    intermediate_size = 2048

    for quant_type in quant_types:
        for num_token in num_tokens:
            test_cases.append([quant_type, num_token, hidden_size, intermediate_size])

    return test_cases


def get_mlp_decode_test_cases():
    """Get test cases for MLP decode phase
    Returns: list of [quant_type, num_token, hidden_size, intermediate_size]
    """
    test_cases = []

    num_tokens = [
        1,
        2,
        4,
        8,
        16,
        32,
        64,
        128,
        256,
        512,
        1024,
        2048,
        4096,
        8192,
        16384,
        32768,
        65536,
        131072,
    ]
    quant_types = ["fp8_block"]
    hidden_size = 7168
    intermediate_size = 2048

    for quant_type in quant_types:
        for num_token in num_tokens:
            test_cases.append([quant_type, num_token, hidden_size, intermediate_size])

    return test_cases


def cleanup_distributed():
    """Clean up distributed environment if it exists"""
    import sglang.srt.distributed.parallel_state as parallel_state

    # Reset all global group variables
    for var_name in ["_TP", "_PP", "_MOE_EP", "_MOE_TP", "_WORLD", "_PDMUX_PREFILL_TP_GROUP"]:
        if hasattr(parallel_state, var_name):
            setattr(parallel_state, var_name, None)

    import sglang.srt.eplb.expert_location as expert_location

    if hasattr(expert_location, "_global_expert_location_metadata"):
        expert_location._global_expert_location_metadata = None

    try:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
    except Exception as e:
        print(f"Warning: Could not clean up torch.distributed: {e}")


def initialize_distributed(model_path, port=29500):
    """Initialize distributed environment for MLP benchmarking"""
    dist_init_method = f"tcp://127.0.0.1:{port}"
    init_distributed_environment(
        backend="nccl",
        world_size=1,
        rank=0,
        local_rank=0,
        distributed_init_method=dist_init_method,
        timeout=10,
    )

    initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=1,
        duplicate_tp_group=False,
        backend="nccl",
    )

    # Set global server args (required by DeepseekV2MLP)
    server_args = ServerArgs(
        model_path=model_path,
        dtype="auto",
        device="cuda",
        load_format="dummy",
        tp_size=1,
        trust_remote_code=True,
        mem_fraction_static=0.5,
        disable_radix_cache=True,
    )
    set_global_server_args_for_scheduler(server_args)


def run_mlp_torch(
    quant_type,
    num_token,
    hidden_size,
    intermediate_size,
    is_context,
    num_warmup,
    num_iterations,
    device,
    output_path,
):
    """Run MLP benchmark for both context and generation phases"""

    torch.cuda.set_device(device)
    phase = "Context" if is_context else "Decode"
    print(f"\n{phase}: quant_type={quant_type}, num_token={num_token}")

    try:
        quant_config = Fp8Config(
            is_checkpoint_fp8_serialized=True,
            activation_scheme="dynamic",
            ignored_layers=None,
            weight_block_size=[128, 128],
        )

        mlp = DeepseekV2MLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            hidden_act="silu",
            quant_config=quant_config,
            reduce_results=True,
            prefix="",
            tp_rank=0,
            tp_size=1,
        ).to(device)

        input_tensor = torch.randn((num_token, hidden_size), dtype=torch.bfloat16, device=device)

        # Import at function level to avoid scoping issues with ruff
        from helper import benchmark_with_power

        if is_context:
            # Context phase: use benchmark_with_power
            def kernel_func():
                _ = mlp(input_tensor)  # noqa: F821

            with benchmark_with_power(
                device=device,
                kernel_func=kernel_func,
                num_warmups=num_warmup,
                num_runs=num_iterations,
                repeat_n=1,
            ) as results:
                pass

            avg_time = results["latency_ms"]
            power_stats = results["power_stats"]
            # Save to collector/ directory (parent of sglang/) to match non-wideep behavior
            collector_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            perf_filename = (
                os.path.join(collector_dir, "wideep_context_mlp_perf.txt")
                if output_path is None
                else os.path.join(output_path, "wideep_context_mlp_perf.txt")
            )
            kernel_source = "deepseek_v3"

            print(f"  {phase} MLP time: {avg_time:.3f} ms")
        else:
            # Decode phase: use benchmark_with_power
            def kernel_func():
                _ = mlp(input_tensor)  # noqa: F821

            with benchmark_with_power(
                device=device,
                kernel_func=kernel_func,
                num_warmups=num_warmup,
                num_runs=num_iterations,
                repeat_n=1,
            ) as results:
                pass

            avg_time = results["latency_ms"]
            power_stats = results["power_stats"]
            # Save to collector/ directory (parent of sglang/) to match non-wideep behavior
            collector_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            perf_filename = (
                os.path.join(collector_dir, "wideep_generation_mlp_perf.txt")
                if output_path is None
                else os.path.join(output_path, "wideep_generation_mlp_perf.txt")
            )
            kernel_source = "deepseek_v3"

            print(f"  {phase} MLP time: {avg_time:.3f} ms")

        # Save via log_perf
        try:
            if output_path is not None:
                os.makedirs(os.path.dirname(perf_filename), exist_ok=True)
            device_name = torch.cuda.get_device_name(device)
            version = get_version("sglang")
            log_perf(
                item_list=[
                    {
                        "quant_type": quant_type,
                        "num_token": num_token,
                        "hidden_size": hidden_size,
                        "intermediate_size": intermediate_size,
                        "avg_ms": avg_time,
                    }
                ],
                framework="SGLang",
                version=version,
                device_name=device_name,
                op_name="mlp",
                kernel_source=kernel_source,
                perf_filename=perf_filename,
                power_stats=power_stats,
            )
        except Exception as e:
            print(f"  Warning: failed to log MLP metrics: {e}")

        del mlp, input_tensor
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"  {phase} test failed: {e!s}")
        print("  Skipping this configuration...")


# ============================================================================
# Functions for collect.py framework (trtllm style: direct params, not index)
# ============================================================================


def get_wideep_mlp_context_test_cases():
    """Returns list of [quant_type, num_tokens_list, hidden_size, inter_size, perf_filename]."""
    # Group all num_tokens into one test case per quant_type (like trtllm moe pattern)
    num_tokens = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
    return [["fp8_block", num_tokens, 7168, 2048, "wideep_context_mlp_perf.txt"]]


def get_wideep_mlp_generation_test_cases():
    """Returns list of [quant_type, num_tokens_list, hidden_size, inter_size, perf_filename]."""
    num_tokens = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
    return [["fp8_block", num_tokens, 7168, 2048, "wideep_generation_mlp_perf.txt"]]


def run_wideep_mlp_context(quant_type, num_tokens_list, hidden_size, intermediate_size, perf_filename, device="cuda:0"):
    """Run wideep MLP context benchmark.

    Compatible with collect.py framework - accepts direct params like trtllm collectors.
    """
    device_str = str(device) if not isinstance(device, str) else device
    torch.cuda.set_device(device_str)

    print(f"\n{'=' * 60}")
    print(f"MLP Context: quant={quant_type}, tokens={len(num_tokens_list)} configs")
    print(f"{'=' * 60}")

    # Initialize distributed with device-specific port
    if not torch.distributed.is_initialized():
        cleanup_distributed()
        device_id = int(device_str.split(":")[-1]) if ":" in device_str else 0
        port = 29500 + device_id
        initialize_distributed(DEEPSEEK_MODEL_PATH, port=port)

    for num_token in num_tokens_list:
        run_mlp_torch(
            quant_type,
            num_token,
            hidden_size,
            intermediate_size,
            is_context=True,
            num_warmup=3,
            num_iterations=10,
            device=device_str,
            output_path=None,
        )


def run_wideep_mlp_generation(
    quant_type, num_tokens_list, hidden_size, intermediate_size, perf_filename, device="cuda:0"
):
    """Run wideep MLP generation benchmark.

    Compatible with collect.py framework - accepts direct params like trtllm collectors.
    """
    device_str = str(device) if not isinstance(device, str) else device
    torch.cuda.set_device(device_str)

    print(f"\n{'=' * 60}")
    print(f"MLP Generation: quant={quant_type}, tokens={len(num_tokens_list)} configs")
    print(f"{'=' * 60}")

    # Initialize distributed with device-specific port
    if not torch.distributed.is_initialized():
        cleanup_distributed()
        device_id = int(device_str.split(":")[-1]) if ":" in device_str else 0
        initialize_distributed(DEEPSEEK_MODEL_PATH, port=29500 + device_id)

    for num_token in num_tokens_list:
        run_mlp_torch(
            quant_type,
            num_token,
            hidden_size,
            intermediate_size,
            is_context=False,
            num_warmup=3,
            num_iterations=10,
            device=device_str,
            output_path=None,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SGLang Wideep MLP Benchmark")
    parser.add_argument("--output-path", default=None, help="Output directory for perf files")
    parser.add_argument("--device", default="cuda:0", help="CUDA device")
    args = parser.parse_args()

    print("Starting SGLang MLP Benchmark")
    print(f"Model path: {DEEPSEEK_MODEL_PATH}")
    print(f"Device: {torch.cuda.get_device_name()}")

    # Run all context and generation test cases
    for test_case in get_wideep_mlp_context_test_cases():
        run_wideep_mlp_context(*test_case, device=args.device)

    for test_case in get_wideep_mlp_generation_test_cases():
        run_wideep_mlp_generation(*test_case, device=args.device)

    cleanup_distributed()
    torch.cuda.empty_cache()

    print("\n" + "=" * 50)
    print("MLP BENCHMARK COMPLETED")
    print("=" * 50)
