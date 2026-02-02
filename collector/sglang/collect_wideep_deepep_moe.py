# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
import logging
import os

import numpy as np
import torch
import torch.distributed as dist
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.entrypoints.engine import _set_envs_and_config
from sglang.srt.layers.moe import initialize_moe_config
from sglang.srt.layers.moe.token_dispatcher.deepep import (
    DeepEPLLDispatchOutput,
    DeepEPNormalDispatchOutput,
)
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.utils import (
    configure_logger,
    get_bool_env_var,
    set_gpu_proc_affinity,
    suppress_other_loggers,
)

try:
    from helper import _get_deepseek_model_path, log_perf, power_law_deepep_decode, power_law_deepep_prefill
except ModuleNotFoundError:
    import os
    import sys

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from helper import _get_deepseek_model_path, log_perf, power_law_deepep_decode, power_law_deepep_prefill
from importlib.metadata import version as get_version

DEEPSEEK_MODEL_PATH = _get_deepseek_model_path()


def get_moe_prefill_test_cases(rank):
    """Get test cases for MoE prefill phase including distribution and alpha.

    Returns a list of dicts with keys: 'num_tokens', 'distributed', 'power_law_alpha'.
    For uniform distribution, 'power_law_alpha' is None.
    """
    test_cases = []
    num_tokens = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    power_law_alphas = [0.6, 0.8, 1.01, 1.02, 1.2]

    for num_token in sorted(num_tokens):
        if num_token * 8 < 128:
            continue
        if num_token * rank > 256 * 2048:
            continue
        # Uniform
        test_cases.append({"num_tokens": num_token, "distributed": "uniform", "power_law_alpha": None})
        # Power-law variants
        for alpha in power_law_alphas:
            test_cases.append(
                {
                    "num_tokens": num_token,
                    "distributed": "power_law",
                    "power_law_alpha": alpha,
                }
            )

    return test_cases


def get_moe_decode_test_cases():
    """Get test cases for MoE decode phase including distribution and alpha.

    Returns a list of dicts with keys: 'num_tokens', 'distributed', 'power_law_alpha'.
    For uniform distribution, 'power_law_alpha' is None.
    """
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
    power_law_alphas = [0.6, 0.8, 1.01, 1.02, 1.2]
    test_cases = []
    # Uniform cases
    for bs in batch_sizes:
        test_cases.append(
            {
                "num_tokens": bs,
                "distributed": "uniform",
                "power_law_alpha": None,
            }
        )
    # Power-law cases
    for bs in batch_sizes:
        for alpha in power_law_alphas:
            test_cases.append(
                {
                    "num_tokens": bs,
                    "distributed": "power_law",
                    "power_law_alpha": alpha,
                }
            )
    return test_cases


def load_model_with_dummy_weights(server_args, port_args, tp_rank):
    """Load model with dummy weights and limited layers for MoE testing"""
    suppress_other_loggers()
    rank_print = print if tp_rank == 0 else lambda *args, **kwargs: None

    if server_args.load_format == "dummy":
        existing_override = {}
        if server_args.json_model_override_args:
            existing_override = json.loads(server_args.json_model_override_args)

        existing_override["num_hidden_layers"] = 4
        server_args.json_model_override_args = json.dumps(existing_override)

    model_config = ModelConfig.from_server_args(server_args)
    rank_print(f"Loading model with {model_config.num_hidden_layers} layers")
    rank_print("Will test MoE module from layer 3 (4th layer, 0-indexed)")

    model_runner = ModelRunner(
        model_config=model_config,
        mem_fraction_static=server_args.mem_fraction_static,
        gpu_id=tp_rank,
        tp_rank=tp_rank,
        tp_size=server_args.tp_size,
        pp_rank=0,
        pp_size=1,
        moe_ep_rank=tp_rank,
        moe_ep_size=server_args.ep_size,
        nccl_port=port_args.nccl_port,
        server_args=server_args,
    )

    rank_print("Model loaded successfully.")

    if server_args.tp_size > 1:
        dist.barrier()

    return model_runner


def benchmark_moe_layer_prefill(
    model_runner,
    server_args,
    port_args,
    num_warmup,
    num_iterations,
    test_layer,
    rank_print,
    device,
    tp_rank,
    prefill_test_cases,
    moe_layer,
    num_local_experts,
    simulated_ep_size,
    output_path,
):
    """Benchmark MoE layer in prefill phase

    Args:
        num_local_experts: Number of experts on this GPU (= model's n_routed_experts)
        simulated_ep_size: The EP size being simulated (= 256 / num_local_experts)
    """

    for case in prefill_test_cases:
        try:
            # Backward compatible: old format was just an int
            if isinstance(case, dict):
                num_token = case["num_tokens"]
                distributed = case.get("distributed", "uniform")
                power_law_alpha = case.get("power_law_alpha", 0.8) if distributed == "power_law" else None
            else:
                num_token = int(case)
                distributed = "uniform"
                power_law_alpha = None

            model_runner.req_to_token_pool.clear()
            model_runner.token_to_kv_pool_allocator.clear()

            # Fake dispatch outputs with random data
            hidden_states_per_token_iter = torch.randn(
                int(num_token * simulated_ep_size),
                model_runner.model.config.hidden_size,
                dtype=torch.bfloat16,
                device=device,
            )

            if hidden_states_per_token_iter.shape[1] % 128 != 0:
                pad_size = 128 - (hidden_states_per_token_iter.shape[1] % 128)
                hidden_states_per_token_iter = torch.nn.functional.pad(hidden_states_per_token_iter, (0, pad_size))

            hidden_states_fp8_tensor_iter = hidden_states_per_token_iter.to(torch.float8_e4m3fn)
            scale_tensor_iter = torch.ones(
                hidden_states_per_token_iter.shape[0],
                hidden_states_per_token_iter.shape[1] // 128,
                device=hidden_states_per_token_iter.device,
                dtype=torch.float32,
            )

            num_tokens_iter = hidden_states_per_token_iter.shape[0]
            topk = 8
            topk_idx_iter = torch.full((num_tokens_iter, topk), -1, device=device, dtype=torch.int32)
            topk_weights_iter = torch.zeros((num_tokens_iter, topk), device=device, dtype=torch.float32)

            if distributed == "uniform":
                tokens_per_local_expert = int(num_token * topk * simulated_ep_size // 256)
                rank_print(f"tokens_per_local_expert: {tokens_per_local_expert}")
                if tokens_per_local_expert <= 0:
                    continue
                num_recv = [tokens_per_local_expert] * num_local_experts

                total_valid_positions = sum(num_recv)
                expert_indices_list = []
                for expert_id in range(num_local_experts):
                    expert_indices_list.extend([expert_id] * tokens_per_local_expert)

                expert_indices_tensor = torch.tensor(expert_indices_list, device=device, dtype=torch.int32)
                shuffled_indices = torch.randperm(len(expert_indices_tensor), device=device)
                expert_indices_tensor = expert_indices_tensor[shuffled_indices]

                positions_per_row = total_valid_positions // num_tokens_iter
                extra_positions = total_valid_positions % num_tokens_iter

                valid_positions_count = 0
                for i in range(num_tokens_iter):
                    current_row_positions = positions_per_row + (1 if i < extra_positions else 0)
                    for j in range(current_row_positions):
                        if valid_positions_count < total_valid_positions:
                            topk_idx_iter[i, j % topk] = expert_indices_tensor[valid_positions_count]
                            valid_positions_count += 1
                        else:
                            break

                # Uniform weights across used columns
                for i in range(num_tokens_iter):
                    used_mask = topk_idx_iter[i] != -1
                    if used_mask.any():
                        topk_weights_iter[i, used_mask] = 1.0 / topk

            elif distributed == "power_law":
                # Use power_law_deepep_prefill to generate router logits for local experts
                # Generate multiple samples to avoid outliers from a single sampling
                power_law_samples = []
                for _ in range(5):
                    topk_idx_sample, topk_weights_sample, num_recv_tensor = power_law_deepep_prefill(
                        num_tokens_iter,
                        num_local_experts * simulated_ep_size,
                        topk,
                        simulated_ep_size,
                        power_law_alpha if power_law_alpha is not None else 0.8,
                    )
                    topk_idx_sample = topk_idx_sample.to(device).contiguous()
                    topk_weights_sample = topk_weights_sample.to(device).contiguous()
                    topk_weights_sample = torch.nan_to_num(topk_weights_sample, nan=0.0, posinf=0.0, neginf=0.0)
                    num_recv = num_recv_tensor.tolist()
                    power_law_samples.append((topk_idx_sample, topk_weights_sample, num_recv))

            else:
                raise ValueError(f"Unsupported distributed mode: {distributed}")

            # For uniform distribution, create a single-element list for unified processing
            if distributed == "uniform":
                # Safety clamp for weights
                topk_weights_iter = torch.nan_to_num(topk_weights_iter, nan=0.0, posinf=0.0, neginf=0.0)
                power_law_samples = [(topk_idx_iter, topk_weights_iter, num_recv)]

            # Warmup
            for _ in range(num_warmup):
                for topk_idx_sample, topk_weights_sample, num_recv_sample in power_law_samples:
                    hidden_states_fp8_tensor_iter = hidden_states_per_token_iter.to(torch.float8_e4m3fn)
                    scale_tensor_iter = torch.ones(
                        hidden_states_per_token_iter.shape[0],
                        hidden_states_per_token_iter.shape[1] // 128,
                        device=hidden_states_per_token_iter.device,
                        dtype=torch.float32,
                    )
                    dispatch_output = DeepEPNormalDispatchOutput(
                        hidden_states=hidden_states_fp8_tensor_iter,
                        hidden_states_scale=scale_tensor_iter,
                        topk_ids=topk_idx_sample.clone(),
                        topk_weights=topk_weights_sample.clone(),
                        num_recv_tokens_per_expert=num_recv_sample,
                    )
                    _ = moe_layer.experts.run_moe_core(dispatch_output)

            torch.get_device_module(device).synchronize()
            torch.cuda.empty_cache()

            gemm_latencies = []

            for i in range(num_iterations):
                for topk_idx_sample, topk_weights_sample, num_recv_sample in power_law_samples:
                    hidden_states_fp8_tensor_iter = hidden_states_per_token_iter.to(torch.float8_e4m3fn)
                    scale_tensor_iter = torch.ones(
                        hidden_states_per_token_iter.shape[0],
                        hidden_states_per_token_iter.shape[1] // 128,
                        device=hidden_states_per_token_iter.device,
                        dtype=torch.float32,
                    )
                    dispatch_output = DeepEPNormalDispatchOutput(
                        hidden_states=hidden_states_fp8_tensor_iter,
                        hidden_states_scale=scale_tensor_iter,
                        topk_ids=topk_idx_sample.clone(),
                        topk_weights=topk_weights_sample.clone(),
                        num_recv_tokens_per_expert=num_recv_sample,
                    )
                    torch.get_device_module(device).synchronize()
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    start_event.record()

                    _ = moe_layer.experts.run_moe_core(dispatch_output)

                    end_event.record()
                    end_event.synchronize()
                    latency_ms = start_event.elapsed_time(end_event)
                    if i > 2:
                        gemm_latencies.append(latency_ms)

            torch.cuda.empty_cache()

            avg_latency_ms = np.mean(gemm_latencies)

            if tp_rank == 0:
                rank_print("DeepEP MoE GEMM Results (Prefill):")
                rank_print(f"  Average latency: {avg_latency_ms:.3f}ms")
            if tp_rank == 0:
                try:
                    moe_tp_size = 1
                    moe_ep_size = simulated_ep_size
                    num_tokens_log = num_token * simulated_ep_size
                    device_name = torch.cuda.get_device_name(server_args.device)
                    version = get_version("sglang")
                    # Save to collector/ directory to match non-wideep behavior
                    collector_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                    perf_filename = (
                        os.path.join(collector_dir, "wideep_context_moe_perf.txt")
                        if output_path is None
                        else os.path.join(output_path, "wideep_context_moe_perf.txt")
                    )
                    distribution_str = f"power_law_{power_law_alpha}" if distributed == "power_law" else distributed
                    log_perf(
                        item_list=[
                            {
                                "moe_dtype": "fp8_block",
                                "num_tokens": num_tokens_log,
                                "hidden_size": 7168,
                                "inter_size": 2048,
                                "topk": 8,
                                "num_experts": 256,
                                "moe_tp_size": moe_tp_size,
                                "moe_ep_size": moe_ep_size,
                                "distribution": distribution_str,
                                "latency": avg_latency_ms,
                            }
                        ],
                        framework="SGLang",
                        version=version,
                        device_name=device_name,
                        op_name="moe_context",
                        kernel_source="deepepmoe",
                        perf_filename=perf_filename,
                    )
                except Exception as e:
                    rank_print(f"  Warning: failed to log prefill MoE metrics: {e}")
            del (
                hidden_states_per_token_iter,
                hidden_states_fp8_tensor_iter,
                scale_tensor_iter,
                topk_idx_iter,
                topk_weights_iter,
                num_recv,
                dispatch_output,
            )
            torch.cuda.empty_cache()

        except Exception as e:
            rank_print(f"Prefill case failed: {e}, skipping...")
            # Check if this is a CUDA error - if so, the context is corrupted and we should exit
            if "CUDA error" in str(e) or "illegal memory access" in str(e).lower():
                rank_print("CUDA error detected, exiting prefill benchmark early to avoid cascading failures")
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                break
            try:
                torch.cuda.empty_cache()
            except Exception:
                # If empty_cache fails, CUDA context is corrupted
                rank_print("CUDA context corrupted, exiting prefill benchmark early")
                break
            continue


def benchmark_moe_layer_decode(
    model_runner,
    server_args,
    port_args,
    num_warmup,
    num_iterations,
    test_layer,
    rank_print,
    device,
    tp_rank,
    decode_test_cases,
    moe_layer,
    num_local_experts,
    simulated_ep_size,
    output_path=None,
):
    """Benchmark MoE layer in decode phase

    Args:
        num_local_experts: Number of experts on this GPU (= model's n_routed_experts)
        simulated_ep_size: The EP size being simulated (= 256 / num_local_experts)
    """
    model_runner.req_to_token_pool.clear()
    model_runner.token_to_kv_pool_allocator.clear()
    top_k = moe_layer.topk.topk_config.top_k

    for case in decode_test_cases:
        try:
            num_token = case["num_tokens"]
            distributed = case["distributed"]
            power_law_alpha = case.get("power_law_alpha", 0.8) if distributed == "power_law" else None
            num_max_dispatch_tokens_per_rank = 128

            if num_token > num_max_dispatch_tokens_per_rank:
                print(
                    f"num_token {num_token} > num_max_dispatch_tokens_per_rank "
                    f"{num_max_dispatch_tokens_per_rank}, skipping"
                )
                continue

            hidden_size = model_runner.model.config.hidden_size

            if hidden_size % 128 != 0:
                pad_size = 128 - (hidden_size % 128)
                hidden_size += pad_size

            hidden_states = torch.randn(
                num_local_experts,
                num_max_dispatch_tokens_per_rank * simulated_ep_size,
                hidden_size,
                dtype=torch.bfloat16,
                device="cuda",
            )

            scale_hidden_size = hidden_size // 128
            scale_tensor = torch.ones(
                num_local_experts,
                num_max_dispatch_tokens_per_rank * simulated_ep_size,
                scale_hidden_size,
                device=hidden_states.device,
                dtype=torch.float32,
            )
            hidden_states_fp8_tensor = hidden_states.to(torch.float8_e4m3fn)

            masked_m = torch.zeros(num_local_experts, device=device, dtype=torch.int32)

            # support two distributed mode: power_law and uniform
            if distributed == "power_law":
                masked_m_list = [
                    power_law_deepep_decode(
                        num_token * simulated_ep_size,
                        num_local_experts * simulated_ep_size,
                        top_k,
                        simulated_ep_size,
                        power_law_alpha,
                    )
                    .to(masked_m.dtype)
                    .to(torch.device(device))
                    for _ in range(5)
                ]
            elif distributed == "uniform":
                # Total experts is 256, simulated_ep_size = 256 / num_local_experts
                base_tokens_per_expert = int(num_token * top_k) * simulated_ep_size // 256
                if base_tokens_per_expert == 0:
                    # Each expert that receives tokens gets exactly 1 token
                    # Number of experts with tokens on this card = total_calls / simulated_ep_size
                    # = (num_token * top_k * num_rank) / num_rank = num_token * top_k
                    masked_m[: int(num_token * top_k)] = 1
                else:
                    masked_m[:] = base_tokens_per_expert
                masked_m_list = [masked_m]
            else:
                raise ValueError(f"Unsupported distributed mode: {distributed}")
            max_masked_m = int(torch.stack([mm.max() for mm in masked_m_list]).max().item())
            if max_masked_m > hidden_states.shape[1]:
                print(
                    f"  Skipping: max(masked_m_list) {max_masked_m} > hidden_states.shape[1] {hidden_states.shape[1]}"
                )
                continue
            scale_tensor = torch.ones(
                num_local_experts,
                num_max_dispatch_tokens_per_rank * simulated_ep_size,
                scale_hidden_size,
                device=hidden_states.device,
                dtype=torch.float32,
            )
            hidden_states_fp8_tensor = hidden_states.to(torch.float8_e4m3fn)

            topk_idx_empty = torch.empty(0, device=device, dtype=torch.int32)
            topk_weights_empty = torch.empty(0, device=device, dtype=torch.float32)

            torch.get_device_module(device).synchronize()
            torch.cuda.empty_cache()

            for _ in range(num_warmup):
                dispatch_output_list = []
                for masked_m in masked_m_list:
                    hidden_states_fp8_tensor_copy = hidden_states_fp8_tensor.clone()
                    scale_tensor_copy = scale_tensor.clone()

                    output = DeepEPLLDispatchOutput(
                        hidden_states=hidden_states_fp8_tensor_copy,
                        hidden_states_scale=scale_tensor_copy,
                        topk_ids=topk_idx_empty,
                        topk_weights=topk_weights_empty,
                        masked_m=masked_m,
                        expected_m=int(torch.ceil(masked_m.float().mean()).item()),
                    )
                    dispatch_output_list.append(output)

                for dispatch_output in dispatch_output_list:
                    _ = moe_layer.experts.run_moe_core(dispatch_output)

            torch.get_device_module(device).synchronize()
            torch.cuda.empty_cache()

            # Use benchmark_with_power for timing
            from helper import benchmark_with_power

            # Pre-compute expected_m values outside of kernel_func to avoid .item() during CUDA graph capture
            expected_m_list = [int(torch.ceil(masked_m_item.float().mean()).item()) for masked_m_item in masked_m_list]

            # Pre-clone masked_m tensors (they won't be disposed by run_moe_core)
            masked_m_clones = [m.clone() for m in masked_m_list]

            # Pre-create enough tensor copies to avoid clone() inside kernel_func
            # run_moe_core disposes hidden_states and hidden_states_scale via dispose_tensor()
            # Estimate: kernel_func called ~4 times (warmup 3 + capture 1) in graph mode
            # Each call iterates len(masked_m_list) times (max 5 for power_law)
            # Total: 4 * 5 = 20 tensor sets needed, use 50 for safety
            num_masked_m = len(masked_m_list)
            num_kernel_calls = 20  # Conservative estimate for kernel_func invocations
            num_tensor_sets = num_kernel_calls * num_masked_m

            hidden_states_copies = []
            scale_copies = []
            for _ in range(num_tensor_sets):
                hidden_states_copies.append(
                    torch.randn(
                        num_local_experts,
                        num_max_dispatch_tokens_per_rank * simulated_ep_size,
                        hidden_size,
                        dtype=torch.bfloat16,
                        device=device,
                    ).to(torch.float8_e4m3fn)
                )
                scale_copies.append(
                    torch.ones(
                        num_local_experts,
                        num_max_dispatch_tokens_per_rank * simulated_ep_size,
                        scale_hidden_size,
                        device=device,
                        dtype=torch.float32,
                    )
                )

            # Use a mutable container to track tensor index across all run_moe_core calls
            tensor_idx = [0]

            def kernel_func():
                for masked_m_clone, expected_m_val in zip(masked_m_clones, expected_m_list):
                    idx = tensor_idx[0] % num_tensor_sets
                    tensor_idx[0] += 1
                    dispatch_output = DeepEPLLDispatchOutput(
                        hidden_states=hidden_states_copies[idx],
                        hidden_states_scale=scale_copies[idx],
                        topk_ids=torch.empty(0, device=device, dtype=torch.int32),
                        topk_weights=torch.empty(0, device=device, dtype=torch.float32),
                        masked_m=masked_m_clone,
                        expected_m=expected_m_val,
                    )
                    _ = moe_layer.experts.run_moe_core(dispatch_output)

            with benchmark_with_power(
                device=device,
                kernel_func=kernel_func,
                num_warmups=3,
                num_runs=num_iterations,
                repeat_n=1,
            ) as results:
                pass

            avg_latency_ms = results["latency_ms"] / len(masked_m_list)
            power_stats = results["power_stats"]

            if tp_rank == 0:
                rank_print("DeepEP MoE GEMM Results (Decode) - CUDA Graph Enabled:")
                rank_print(f"  Average latency: {avg_latency_ms:.3f}ms")
            if tp_rank == 0:
                try:
                    moe_tp_size = 1
                    moe_ep_size = simulated_ep_size
                    num_tokens_log = num_token * simulated_ep_size
                    device_name = torch.cuda.get_device_name(server_args.device)
                    version = get_version("sglang")
                    distribution_str = f"power_law_{power_law_alpha}" if distributed == "power_law" else distributed
                    # Save to collector/ directory to match non-wideep behavior
                    collector_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                    perf_filename = (
                        os.path.join(collector_dir, "wideep_generation_moe_perf.txt")
                        if output_path is None
                        else os.path.join(output_path, "wideep_generation_moe_perf.txt")
                    )
                    log_perf(
                        item_list=[
                            {
                                "moe_dtype": "fp8_block",
                                "num_tokens": num_tokens_log,
                                "hidden_size": 7168,
                                "inter_size": 2048,
                                "topk": 8,
                                "num_experts": 256,
                                "moe_tp_size": moe_tp_size,
                                "moe_ep_size": moe_ep_size,
                                "distribution": distribution_str,
                                "latency": avg_latency_ms,
                            }
                        ],
                        framework="SGLang",
                        version=version,
                        device_name=device_name,
                        op_name="moe_generation",
                        kernel_source="deepepmoe",
                        perf_filename=perf_filename,
                        power_stats=power_stats,
                    )
                except Exception as e:
                    rank_print(f"  Warning: failed to log decode MoE metrics: {e}")
            del hidden_states, hidden_states_fp8_tensor, scale_tensor, dispatch_output_list
            torch.cuda.empty_cache()

        except Exception as e:
            rank_print(f"Decode case failed: {e}, skipping...")
            # Check if this is a CUDA error - if so, the context is corrupted and we should exit
            if "CUDA error" in str(e) or "illegal memory access" in str(e).lower():
                rank_print("CUDA error detected, exiting decode benchmark early to avoid cascading failures")
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                break
            try:
                torch.cuda.empty_cache()
            except Exception:
                # If empty_cache fails, CUDA context is corrupted
                rank_print("CUDA context corrupted, exiting decode benchmark early")
                break
            continue


def run_moe(
    server_args,
    port_args,
    num_warmup,
    num_iterations,
    test_layer,
    num_experts,
    tp_rank,
    output_path=None,
):
    """Run the complete MoE benchmark"""

    if get_bool_env_var("SGLANG_SET_CPU_AFFINITY"):
        set_gpu_proc_affinity(server_args.tp_size, server_args.nnodes, tp_rank)

    configure_logger(server_args, prefix=f" TP{tp_rank}")

    # Initialize MoE config in subprocess (required for DeepEP + DeepGEMM backend)
    _set_envs_and_config(server_args)
    initialize_moe_config(server_args)

    rank_print = print if tp_rank == 0 else lambda *args, **kwargs: None

    rank_print(f"\n{'=' * 60}")
    rank_print(f"Testing MoE Layer {test_layer}")
    rank_print(f"{'=' * 60}")

    try:
        rank_print(f"\n{'=' * 50}")
        rank_print(f"Testing with {num_experts} experts")
        rank_print(f"{'=' * 50}")

        original_json_override = server_args.json_model_override_args
        server_args.json_model_override_args = json.dumps({"num_hidden_layers": 4, "n_routed_experts": num_experts})

        model_runner = load_model_with_dummy_weights(server_args, port_args, tp_rank)

        moe_layer = model_runner.model.model.layers[test_layer].mlp
        actual_num_experts = moe_layer.config.n_routed_experts

        rank_print(f"Loaded model with {actual_num_experts} experts")

        server_args.json_model_override_args = original_json_override

        # Calculate simulated EP size: 256 total experts / num_local_experts
        num_local_experts = actual_num_experts  # With ep_size=1, all experts are local
        simulated_ep_size = 256 // num_local_experts
        rank_print(f"Simulating EP size: {simulated_ep_size} (num_local_experts={num_local_experts})")

        prefill_test_cases = get_moe_prefill_test_cases(simulated_ep_size)
        rank_print(f"Testing {len(prefill_test_cases)} prefill configurations...")

        # Use deepep_mode="normal" for prefill
        server_args.deepep_mode = "normal"
        benchmark_moe_layer_prefill(
            model_runner,
            server_args,
            port_args,
            num_warmup,
            num_iterations,
            test_layer,
            rank_print,
            server_args.device,
            tp_rank,
            prefill_test_cases,
            moe_layer,
            num_local_experts,
            simulated_ep_size,
            output_path,
        )

        decode_test_cases = get_moe_decode_test_cases()
        rank_print(f"Testing {len(decode_test_cases)} decode configurations...")
        # Use deepep_mode="low_latency" for decode
        server_args.deepep_mode = "low_latency"
        benchmark_moe_layer_decode(
            model_runner,
            server_args,
            port_args,
            num_warmup,
            num_iterations,
            test_layer,
            rank_print,
            server_args.device,
            tp_rank,
            decode_test_cases,
            moe_layer,
            num_local_experts,
            simulated_ep_size,
            output_path=output_path,
        )

        del model_runner, moe_layer
        torch.cuda.empty_cache()

    except Exception as e:
        rank_print(f"Error during MoE benchmark: {e}")
        import traceback

        rank_print(f"Traceback: {traceback.format_exc()}")
        return

    torch.cuda.empty_cache()

    rank_print(f"\n{'=' * 60}")
    rank_print("BENCHMARK COMPLETED SUCCESSFULLY")
    rank_print(f"{'=' * 60}")


# ============================================================================
# Functions for collect.py framework (trtllm style: direct params, not index)
# ============================================================================


def get_wideep_moe_test_cases():
    """Returns list of [num_experts, perf_filename] for MOE collection.

    Each num_experts value simulates a different EP size:
    - num_experts=128 → EP=2
    - num_experts=64 → EP=4
    - num_experts=32 → EP=8
    - num_experts=16 → EP=16
    - num_experts=8 → EP=32
    - num_experts=4 → EP=64
    - num_experts=2 → EP=128
    - num_experts=1 → EP=256

    Formula: simulated_ep_size = 256 / num_experts
    """
    return [[n, "wideep_moe_perf.txt"] for n in [128, 64, 32, 16, 8, 4, 2, 1]]


def run_moe_benchmark(num_experts, gpu_id, output_path=None):
    """Run MOE benchmark - called in subprocess with CUDA_VISIBLE_DEVICES set.

    This function contains all the initialization logic that must happen
    after CUDA_VISIBLE_DEVICES is set.
    """
    # In subprocess, always use cuda:0 since CUDA_VISIBLE_DEVICES isolates the GPU
    torch.cuda.set_device("cuda:0")

    server_port = 30000 + gpu_id * 100
    server_args = ServerArgs(
        model_path=DEEPSEEK_MODEL_PATH,
        dtype="auto",
        device="cuda",
        load_format="dummy",
        tp_size=1,
        trust_remote_code=True,
        mem_fraction_static=0.3,
        moe_a2a_backend="deepep",
        moe_runner_backend="deep_gemm",
        deepep_mode="auto",
        ep_size=1,
        node_rank=0,
        host="localhost",
        port=server_port,
        cuda_graph_max_bs=4,
        disable_cuda_graph=True,
    )

    logging.basicConfig(level=getattr(logging, server_args.log_level.upper()), format="%(message)s")
    _set_envs_and_config(server_args)

    # PortArgs.init_new() must be called in subprocess for proper isolation
    port_args = PortArgs.init_new(server_args)

    simulated_ep_size = 256 // num_experts * server_args.ep_size
    print(f"\n{'=' * 60}")
    print(f"MOE Benchmark: num_experts={num_experts}, EP_size={simulated_ep_size}, GPU={gpu_id}")
    print(f"{'=' * 60}")

    # Run the actual benchmark
    run_moe(server_args, port_args, 3, 10, 3, num_experts, 0, output_path)

    torch.cuda.empty_cache()
    print(f"Completed num_experts={num_experts} (EP size {simulated_ep_size})")


def _run_moe_subprocess(num_experts, gpu_id, output_path=None):
    """Helper to run MOE in subprocess with CUDA_VISIBLE_DEVICES isolation."""
    import subprocess
    import sys

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    code = f'''
import sys
sys.path.insert(0, "{os.path.dirname(os.path.abspath(__file__))}")
from collect_wideep_deepep_moe import run_moe_benchmark
run_moe_benchmark({num_experts}, {gpu_id}, {output_path!r})
'''

    proc = subprocess.Popen(
        [sys.executable, "-c", code],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=os.path.dirname(os.path.abspath(__file__)),
    )

    try:
        stdout, _ = proc.communicate(timeout=600)  # 10 min timeout per MOE config
        if stdout:
            print(stdout.decode("utf-8", errors="replace"))
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        print(f"MOE subprocess timed out for num_experts={num_experts}")

    if proc.returncode != 0:
        raise RuntimeError(f"MOE subprocess failed with exit code {proc.returncode}")


def run_wideep_moe(num_experts, perf_filename, device="cuda:0"):
    """Run wideep DeepEP MOE benchmark.

    Compatible with collect.py framework - uses subprocess for GPU isolation.
    """
    device_str = str(device) if not isinstance(device, str) else device
    gpu_id = int(device_str.split(":")[-1]) if ":" in device_str else 0

    simulated_ep_size = 256 // num_experts
    print("\n" + "=" * 60)
    print(f"MOE: num_experts={num_experts} (EP size {simulated_ep_size}), GPU={gpu_id}")
    print("=" * 60)

    _run_moe_subprocess(num_experts, gpu_id, None)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SGLang Wideep DeepEP MOE Benchmark")
    parser.add_argument("--output-path", default=None, help="Output directory for perf files")
    args = parser.parse_args()

    print(f"Model path: {DEEPSEEK_MODEL_PATH}")

    # Run all MOE test cases
    for test_case in get_wideep_moe_test_cases():
        run_wideep_moe(*test_case)

    print("\n" + "=" * 60)
    print("SCRIPT COMPLETED SUCCESSFULLY")
    print("=" * 60)
