# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
import logging
import multiprocessing
import os

import numpy as np
import torch
import torch.distributed as dist
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.entrypoints.engine import _set_envs_and_config
from sglang.srt.layers.moe.token_dispatcher.deepep import (
    DeepEPLLDispatchOutput,
    DeepEPNormalDispatchOutput,
)

# Aliases for backwards compatibility with collector code
DeepEPLLOutput = DeepEPLLDispatchOutput
DeepEPNormalOutput = DeepEPNormalDispatchOutput
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.utils import (
    configure_logger,
    get_bool_env_var,
    set_gpu_proc_affinity,
    suppress_other_loggers,
)

try:
    from helper import log_perf, power_law_logits_v4, sample_power_law
except ModuleNotFoundError:
    import os
    import sys

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from helper import log_perf, power_law_logits_v4, sample_power_law
import pkg_resources

DEEPSEEK_MODEL_PATH = os.environ.get("DEEPSEEK_MODEL_PATH", "/deepseek-v3")

aic_debug = int(os.getenv("aic_moe_debug", "0"))  # noqa: SIM112


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

    expert_assignments = torch.tensor(expert_assignments, dtype=torch.int64)
    h_selected_experts = expert_assignments.reshape(topk, num_tokens).T

    # New logic: return topk_idx, topk_weights, num_recv_tokens_per_expert
    num_local_experts = num_experts // ep
    topk_idx = h_selected_experts.clone().contiguous()
    topk_weights = torch.full_like(topk_idx, 0.1, dtype=torch.float32)

    # Mask experts not in rank 0
    mask = topk_idx >= num_local_experts
    topk_idx[mask] = -1
    topk_weights[mask] = 0.0

    # num_recv for rank 0 experts
    num_recv_tokens_per_expert = num_tokens_per_expert[:num_local_experts]
    num_recv_tokens_per_expert = (num_recv_tokens_per_expert + 127) // 128 * 128

    return topk_idx, topk_weights, num_recv_tokens_per_expert


# NOTE: power_law_logits_v4 was copied from aiconfigurator/collector/trtllm/collect_moe.py and
# modified to restrict max tokens per expert to be less than num_tokens
def power_law_logits_v4(num_tokens, num_experts, topk, ep, alpha):
    """Generate power law distribution for token assignment to experts"""
    while True:
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
        num_tokens_per_expert_rank0 = num_tokens_per_expert.view(ep, num_experts // ep)[max_ep_idx].view(-1)
        if max(num_tokens_per_expert_rank0) <= num_tokens:
            return num_tokens_per_expert_rank0


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
    num_experts,
    ep_size,
    num_rank,
    output_path,
):
    """Benchmark MoE layer in prefill phase"""
    num_local_experts = num_experts // ep_size

    for case in prefill_test_cases:
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
            int(num_token * num_rank),
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
            tokens_per_local_expert = int(num_token * topk * num_rank // 256)
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
                    topk_weights_iter[i, used_mask] = 1.0 / ep_size / (topk // ep_size)

        elif distributed == "power_law":
            # Use v3 to generate router logits for local experts, then take per-token top-k
            # Generate multiple samples to avoid outliers from a single sampling
            power_law_samples = []
            for _ in range(5):
                topk_idx_sample, topk_weights_sample, num_recv_tensor = power_law_logits_v3(
                    num_tokens_iter,
                    num_local_experts * num_rank,
                    topk,
                    num_rank,
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
                dispatch_output = DeepEPNormalOutput(
                    hidden_states=(hidden_states_fp8_tensor_iter, scale_tensor_iter),
                    topk_idx=topk_idx_sample.clone(),
                    topk_weights=topk_weights_sample.clone(),
                    num_recv_tokens_per_expert=num_recv_sample,
                )
                _ = moe_layer.experts.moe_impl(dispatch_output)

        torch.get_device_module(device).synchronize()
        torch.cuda.empty_cache()

        gemm_latencies = []

        # Use profiler for synchronization
        profiler = torch.profiler.profile(
            with_stack=True,
        )
        profiler.start()

        for i in range(num_iterations):
            for topk_idx_sample, topk_weights_sample, num_recv_sample in power_law_samples:
                hidden_states_fp8_tensor_iter = hidden_states_per_token_iter.to(torch.float8_e4m3fn)
                scale_tensor_iter = torch.ones(
                    hidden_states_per_token_iter.shape[0],
                    hidden_states_per_token_iter.shape[1] // 128,
                    device=hidden_states_per_token_iter.device,
                    dtype=torch.float32,
                )
                dispatch_output = DeepEPNormalOutput(
                    hidden_states=(hidden_states_fp8_tensor_iter, scale_tensor_iter),
                    topk_idx=topk_idx_sample.clone(),
                    topk_weights=topk_weights_sample.clone(),
                    num_recv_tokens_per_expert=num_recv_sample,
                )
                torch.get_device_module(device).synchronize()
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()

                _ = moe_layer.experts.moe_impl(dispatch_output)

                torch.get_device_module(device).synchronize()
                end_event.record()
                latency_ms = start_event.elapsed_time(end_event)
                if i > 2:
                    gemm_latencies.append(latency_ms)

        profiler.stop()
        torch.cuda.empty_cache()

        avg_latency_ms = np.mean(gemm_latencies)

        if tp_rank == 0:
            rank_print("DeepEP MoE GEMM Results (Prefill):")
            rank_print(f"  Average latency: {avg_latency_ms:.3f}ms")
        if tp_rank == 0:
            try:
                moe_tp_size = 1
                moe_ep_size = (
                    server_args.ep_size if num_experts == 256 else int(server_args.ep_size * 256 // num_experts)
                )
                num_tokens_log = num_token * moe_ep_size
                device_name = torch.cuda.get_device_name(server_args.device)
                version = pkg_resources.get_distribution("sglang").version
                perf_filename = os.path.join(output_path, "wideep_context_moe_perf.txt")
                os.makedirs(os.path.dirname(perf_filename), exist_ok=True)
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
    num_experts,
    ep_size,
    num_rank,
    output_path=None,
):
    """Benchmark MoE layer in decode phase"""

    model_runner.req_to_token_pool.clear()
    model_runner.token_to_kv_pool_allocator.clear()
    top_k = moe_layer.topk.top_k
    num_local_experts = int(num_experts // ep_size)

    for case in decode_test_cases:
        num_token = case["num_tokens"]
        distributed = case["distributed"]
        power_law_alpha = case.get("power_law_alpha", 0.8) if distributed == "power_law" else None
        num_max_dispatch_tokens_per_rank = 128

        if num_token > num_max_dispatch_tokens_per_rank:
            print(
                f"num_token {num_token} > num_max_dispatch_tokens_per_rank {num_max_dispatch_tokens_per_rank}, skipping"
            )
            continue

        hidden_size = model_runner.model.config.hidden_size

        if hidden_size % 128 != 0:
            pad_size = 128 - (hidden_size % 128)
            hidden_size += pad_size

        hidden_states = torch.randn(
            num_local_experts,
            num_max_dispatch_tokens_per_rank * num_rank,
            hidden_size,
            dtype=torch.bfloat16,
            device="cuda",
        )

        scale_hidden_size = hidden_size // 128
        scale_tensor = torch.ones(
            num_local_experts,
            num_max_dispatch_tokens_per_rank * num_rank,
            scale_hidden_size,
            device=hidden_states.device,
            dtype=torch.float32,
        )
        hidden_states_fp8_tensor = hidden_states.to(torch.float8_e4m3fn)

        masked_m = torch.zeros(num_local_experts, device=device, dtype=torch.int32)

        # support two distributed mode: power_law and uniform
        if distributed == "power_law":
            masked_m_list = [
                power_law_logits_v4(
                    num_token * num_rank,
                    num_local_experts * num_rank,
                    top_k,
                    num_rank,
                    power_law_alpha,
                )
                .to(masked_m.dtype)
                .to(torch.device(device))
                for _ in range(5)
            ]
        elif distributed == "uniform":
            # expert size is 256
            base_tokens_per_expert = int(num_token * top_k) * num_rank // 256
            if base_tokens_per_expert == 0:
                masked_m[: int(num_token * top_k) * num_rank // ep_size] = 1
            else:
                masked_m[:] = base_tokens_per_expert
            masked_m_list = [masked_m]
        else:
            raise ValueError(f"Unsupported distributed mode: {distributed}")
        max_masked_m = int(torch.stack([mm.max() for mm in masked_m_list]).max().item())
        assert max_masked_m <= hidden_states.shape[1], (
            f"max(masked_m_list) {max_masked_m} > hidden_states.shape[1] {hidden_states.shape[1]}"
        )
        scale_tensor = torch.ones(
            num_local_experts,
            num_max_dispatch_tokens_per_rank * num_rank,
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

                output = DeepEPLLOutput(
                    hidden_states_fp8=(
                        hidden_states_fp8_tensor_copy,
                        scale_tensor_copy,
                    ),
                    topk_idx=topk_idx_empty,
                    topk_weights=topk_weights_empty,
                    masked_m=masked_m,
                    expected_m=int(torch.ceil(masked_m.float().mean()).item()),
                )
                dispatch_output_list.append(output)

            for dispatch_output in dispatch_output_list:
                _ = moe_layer.experts.moe_impl(dispatch_output)

        torch.get_device_module(device).synchronize()
        torch.cuda.empty_cache()

        dispatch_output_list = []
        for masked_m in masked_m_list:
            hidden_states_fp8_tensor_copy = hidden_states_fp8_tensor.clone()
            scale_tensor_copy = scale_tensor.clone()

            output = DeepEPLLOutput(
                hidden_states_fp8=(hidden_states_fp8_tensor_copy, scale_tensor_copy),
                topk_idx=topk_idx_empty,
                topk_weights=topk_weights_empty,
                masked_m=masked_m,
                expected_m=int(torch.ceil(masked_m.float().mean()).item()),
            )
            dispatch_output_list.append(output)

        # Use benchmark_with_power for timing
        from collector.helper import benchmark_with_power

        def kernel_func():
            for dispatch_output in dispatch_output_list:  # noqa: F821
                _ = moe_layer.experts.moe_impl(dispatch_output)

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
                moe_ep_size = (
                    server_args.ep_size if num_experts == 256 else int(server_args.ep_size * 256 // num_experts)
                )
                num_tokens_log = num_token * moe_ep_size
                device_name = torch.cuda.get_device_name(server_args.device)
                version = pkg_resources.get_distribution("sglang").version
                distribution_str = f"power_law_{power_law_alpha}" if distributed == "power_law" else distributed
                perf_filename = os.path.join(output_path, "wideep_generation_moe_perf.txt")
                os.makedirs(os.path.dirname(perf_filename), exist_ok=True)
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

        ep_size = server_args.ep_size
        num_rank = ep_size if actual_num_experts == 256 else int(256 // actual_num_experts * ep_size)
        prefill_test_cases = get_moe_prefill_test_cases(num_rank)
        rank_print(f"Testing {len(prefill_test_cases)} prefill configurations...")

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
            actual_num_experts,
            ep_size,
            num_rank,
            output_path,
        )

        decode_test_cases = get_moe_decode_test_cases()
        rank_print(f"Testing {len(decode_test_cases)} decode configurations...")
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
            actual_num_experts,
            ep_size,
            num_rank,
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


if __name__ == "__main__":
    model_path = DEEPSEEK_MODEL_PATH
    output_path = "/aiconfigurator/src/aiconfigurator/systems/data/h100_sxm/sglang/0.5.0/"
    num_warmup = 3
    num_iterations = 10
    test_layer = 3

    # num_experts list to simulate different EP sizes
    # num_experts=256 -> EP 2, num_experts=128 -> EP 4, ..., num_experts=2 -> EP 256
    num_experts_list = [256, 128, 64, 32, 16, 8, 4, 2]

    server_args = ServerArgs(
        model_path=model_path,
        dtype="auto",
        device="cuda",
        load_format="dummy",
        tp_size=2,
        trust_remote_code=True,
        mem_fraction_static=0.3,
        enable_deepep_moe=True,
        enable_ep_moe=True,
        ep_size=2,
        node_rank=0,
        host="localhost",
        port=30000,
        cuda_graph_max_bs=4,
        disable_cuda_graph=True,
    )

    logging.basicConfig(
        level=getattr(logging, server_args.log_level.upper()),
        format="%(message)s",
    )

    _set_envs_and_config(server_args)
    port_args = PortArgs.init_new(server_args)

    for num_experts in num_experts_list:
        simulated_ep_size = 256 // num_experts * server_args.ep_size
        print("\n" + "=" * 60)
        print(f"Testing num_experts={num_experts} (simulating EP size {simulated_ep_size})")
        print("=" * 60)

        workers = []
        for tp_rank in range(server_args.tp_size):
            proc = multiprocessing.Process(
                target=run_moe,
                args=(
                    server_args,
                    port_args,
                    num_warmup,
                    num_iterations,
                    test_layer,
                    num_experts,
                    tp_rank,
                    output_path,
                ),
            )
            proc.start()
            workers.append(proc)

        for proc in workers:
            proc.join()

        for i, proc in enumerate(workers):
            if proc.exitcode != 0:
                print(f"Process {i} (tp_rank={i}) failed with exit code {proc.exitcode}")

        for proc in workers:
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=5)
                if proc.is_alive():
                    proc.kill()

        print(f"Completed testing num_experts={num_experts} (EP size {simulated_ep_size})")

    print("\n" + "=" * 60)
    print("SCRIPT COMPLETED SUCCESSFULLY")
    print("=" * 60)
