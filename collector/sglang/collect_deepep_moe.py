#!/usr/bin/env python3
"""
Benchmark DeepEP MoE layers with dummy weights.

This script loads the first 4 layers of DeepSeek model with dummy weights
and benchmarks the DeepEP MoE layers for different batch sizes and sequence lengths.

Usage:
    # Basic test with dummy weights
    python collect_deepep_moe.py --model-path deepseek-ai/deepseek-coder-7b-instruct --load-format dummy

    # Test specific batch sizes and sequence lengths
    python collect_deepep_moe.py --model-path deepseek-ai/deepseek-coder-7b-instruct --load-format dummy --batch-size 1 4 8 --input-len 128 512 1024

    # Test with profiling
    python collect_deepep_moe.py --model-path deepseek-ai/deepseek-coder-7b-instruct --load-format dummy --profile

Debugging CUDA errors:
    # Set environment variables for debugging
    export CUDA_LAUNCH_BLOCKING=1
    export TORCH_USE_CUDA_DSA=1
    python collect_deepep_moe.py --model-path deepseek-ai/deepseek-coder-7b-instruct --load-format dummy

Multi-GPU timing:
    # The script now properly handles multi-GPU timing with distributed barriers
    # to ensure accurate latency measurements across all GPUs
    # All ranks participate in the computation, but only rank 0 reports results
"""

import argparse
import copy
import dataclasses
import itertools
import json
import logging
import multiprocessing
import os
import time
from typing import Tuple, List, Optional
import numpy as np
import torch
import torch.distributed as dist
from dataclasses import dataclass, field

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.distributed.parallel_state import destroy_distributed_environment
from sglang.srt.entrypoints.engine import _set_envs_and_config
from sglang.srt.hf_transformers_utils import get_tokenizer
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.model_executor.cuda_graph_runner import rank0_log
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils import (
    configure_logger,
    get_bool_env_var,
    kill_process_tree,
    require_mlp_sync,
    require_mlp_tp_gather,
    set_gpu_proc_affinity,
    suppress_other_loggers,
    BumpAllocator,
)
from sglang.srt.layers.moe.utils import DeepEPMode
from sglang.srt.layers.moe.token_dispatcher.deepep import DeepEPNormalOutput, DeepEPLLOutput


# Define MoE test case functions similar to collect_attn.py
def get_moe_prefill_test_cases(ep_size, rank):
    """Get test cases for MoE prefill phase - batch_size and input_len dimensions"""
    test_cases = []
    # batch_sizes = [1, 2]  
    # # input_lens = [512]  
    input_lens = [2] #4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]  
    batch_sizes = [1] #, 2, 4, 8, 16, 32, 64, 128, 256, 512] 
    # batch_sizes = [1/16, 1/8, 1/4, 1/2, 1, 2]
    
    ## batch 4, 8, 16, 32, 64, 128, 256, 512

    for batch_size in sorted(batch_sizes):
        for input_len in sorted(input_lens):
            # Simple memory limit checks
            if batch_size * input_len * 8 * rank // ep_size < 128:  
                continue
            if batch_size * input_len * rank > 256*2048: 
                continue
            
            # Add test cases with batch_size and input_len
            test_cases.append([batch_size, input_len])
    
    return test_cases


def get_moe_decode_test_cases():
    """Get test cases for MoE decode phase - batch_size and kv_len dimensions"""
    test_cases = []
    
    # Decode parameters - batch_size and kv_len dimensions
    batch_sizes = [1] #, 2, 4, 8, 16, 32, 64]  # More reasonable batch sizes
    # batch_sizes = [1/8, 1/4, 1/2]  # More reasonable batch sizes
    
    test_cases = batch_sizes
    
    return test_cases


@dataclass
class MoEBenchArgs:
    run_name: str = "deepep_moe_test"
    batch_size: Tuple[int] = (1, 4, 8, 16)
    input_len: Tuple[int] = (128, 512, 1024, 2048)
    output_len: Tuple[int] = (16,)
    result_filename: str = "deepep_moe_results.txt"
    enable_profile: bool = False
    profile_filename_prefix: str = "deepep_moe_profile"
    num_warmup: int = 3
    num_iterations: int = 7
    test_layer: int = 3

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser):
        parser.add_argument("--run-name", type=str, default=MoEBenchArgs.run_name)
        parser.add_argument(
            "--batch-size", type=int, nargs="+", default=MoEBenchArgs.batch_size
        )
        parser.add_argument(
            "--input-len", type=int, nargs="+", default=MoEBenchArgs.input_len
        )
        parser.add_argument(
            "--output-len", type=int, nargs="+", default=MoEBenchArgs.output_len
        )
        parser.add_argument(
            "--result-filename", type=str, default=MoEBenchArgs.result_filename
        )
        parser.add_argument("--enable-profile", type=bool, default=True, help="Use Torch Profiler.")
        parser.add_argument(
            "--profile-filename-prefix",
            type=str,
            default=MoEBenchArgs.profile_filename_prefix,
            help="Prefix of the profiling file names."
        )

        parser.add_argument("--num-warmup", type=int, default=MoEBenchArgs.num_warmup)
        parser.add_argument("--num-iterations", type=int, default=MoEBenchArgs.num_iterations)

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace):
        attrs = [(attr.name, type(attr.default)) for attr in dataclasses.fields(cls)]
        # Filter out attributes that come from ServerArgs
        filtered_attrs = []
        for attr, attr_type in attrs:
            if hasattr(args, attr):
                filtered_attrs.append((attr, attr_type))
        
        return cls(
            **{attr: attr_type(getattr(args, attr)) for attr, attr_type in filtered_attrs}
        )


@dataclass
class MoEBenchResult:
    """Result for a single MoE benchmark run"""
    run_name: str
    layer_id: int
    batch_size: int
    input_len: int
    output_len: int
    phase: str  # "prefill" or "decode"
    avg_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    std_latency_ms: float
    throughput_tokens_per_sec: float
    num_iterations: int
    cuda_graph_used: bool = False


def get_gpu_device_name():
    """Get the actual GPU device name"""
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            # Map common device names to standardized names
            if "H20" in device_name:
                return "NVIDIA H20-3e"
            elif "A100" in device_name:
                return "NVIDIA A100"
            elif "V100" in device_name:
                return "NVIDIA V100"
            elif "RTX" in device_name:
                return f"NVIDIA {device_name}"
            else:
                return f"NVIDIA {device_name}"
        else:
            return "NVIDIA H20-3e"  # Default fallback
    except:
        return "NVIDIA H20-3e"  # Default fallback


def load_model_with_dummy_weights(server_args, port_args, tp_rank):
    """Load model with dummy weights and limited layers for MoE testing"""
    suppress_other_loggers()
    rank_print = print if tp_rank == 0 else lambda *args, **kwargs: None

    # Override model config to load only first 4 layers (we'll test the 4th layer)
    if server_args.load_format == "dummy":
        server_args.json_model_override_args = json.dumps({
            "num_hidden_layers": 4
        })

    model_config = ModelConfig.from_server_args(server_args)
    rank_print(f"Loading model with {model_config.num_hidden_layers} layers")
    rank_print(f"Will test MoE module from layer 3 (4th layer, 0-indexed)")
    
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
    
    rank_print(f"Model loaded successfully. Max total num tokens: {model_runner.max_total_num_tokens}")
    
    tokenizer = get_tokenizer(
        server_args.tokenizer_path,
        tokenizer_mode=server_args.tokenizer_mode,
        trust_remote_code=server_args.trust_remote_code,
    )
    
    if server_args.tp_size > 1:
        dist.barrier()
    
    return model_runner, tokenizer


def prepare_synthetic_inputs(batch_size, input_len):
    """Prepare synthetic inputs for benchmarking"""
    reqs = []
    for i in range(batch_size):
        req = Req(
            rid=str(i),
            origin_input_text="",
            origin_input_ids=list(torch.randint(0, 10000, (input_len,)).tolist()),
            sampling_params=SamplingParams(temperature=0, max_new_tokens=1),
        )
        req.prefix_indices = []
        req.fill_ids = req.origin_input_ids
        req.extend_input_len = len(req.fill_ids)
        req.logprob_start_len = 0
        req.cached_tokens = 0  
        req.already_computed = 0  
        reqs.append(req)

    return reqs


def _maybe_prepare_mlp_sync_batch(batch: ScheduleBatch, model_runner):
    """Prepare MLP sync batch for DeepEP MoE"""
    if require_mlp_sync(model_runner.server_args):
        Scheduler.prepare_mlp_sync_batch_raw(
            batch,
            dp_size=model_runner.server_args.dp_size,
            attn_tp_size=model_runner.server_args.tp_size,
            tp_group=model_runner.tp_group,
            get_idle_batch=None,
            disable_cuda_graph=True,
            spec_algorithm=SpeculativeAlgorithm.NONE,
            speculative_num_draft_tokens=None,
            require_mlp_tp_gather=require_mlp_tp_gather(model_runner.server_args),
            disable_overlap_schedule=model_runner.server_args.disable_overlap_schedule,
            enable_two_batch_overlap=False,
            enable_deepep_moe=True,
            deepep_mode=DeepEPMode.AUTO,
        )


def benchmark_moe_layer_prefill(
    model_runner,
    server_args,
    port_args,
    bench_args: MoEBenchArgs,
    rank_print,
    device: str,
    tp_rank: int,
    enable_profile: bool = False,
    profile_filename_prefix: str = ""
) -> List[MoEBenchResult]:

    # ===== PREFILL PHASE ===== (GEMM)
    results = []

    # Get prefill test cases
    moe_layer = model_runner.model.model.layers[bench_args.test_layer].mlp
    num_experts = moe_layer.config.n_routed_experts  # Total number of experts
    ep_size = server_args.ep_size
    num_local_experts = num_experts // ep_size

    if num_experts  != 256:
        sim_ep_size = int(256//num_experts * ep_size)
        sim_num_experts = 256
        # sim_batch = int(batch_size * 256//num_experts)
    else:
        sim_ep_size = ep_size
        sim_num_experts = num_experts
        # sim_batch = batch_size
    num_rank = sim_ep_size
    prefill_test_cases = get_moe_prefill_test_cases(ep_size, num_rank)
    rank_print(f"Testing {len(prefill_test_cases)} prefill configurations...")
   

    for batch_size, input_len in prefill_test_cases:
        # Check memory limits
        max_batch_size = model_runner.max_total_num_tokens // input_len
        if batch_size > max_batch_size:
            rank_print(f"Skipping prefill ({batch_size}, {input_len}) due to memory limit")
            continue
        
        # Use the simplified MoE prefill test from latency_test_run_once
        rank_print(f"Testing prefill: batch_size={batch_size}, input_len={input_len}")
        
        if num_experts != 256:
            sim_batch = int(batch_size * 256//num_experts)
        else:
            sim_batch = batch_size

        # Clear the pools.
        model_runner.req_to_token_pool.clear()
        model_runner.token_to_kv_pool_allocator.clear()


        total_tokens = batch_size * input_len * 8
        # Fake dispatch outputs with random data
        hidden_states_per_token_iter = torch.randn(
                    int(batch_size * input_len * num_rank), model_runner.model.config.hidden_size,
                    dtype=torch.bfloat16, device=device
                )
                
                # Ensure hidden_size is divisible by 128 for FP8 quantization
        if hidden_states_per_token_iter.shape[1] % 128 != 0:
            pad_size = 128 - (hidden_states_per_token_iter.shape[1] % 128)
            hidden_states_per_token_iter = torch.nn.functional.pad(hidden_states_per_token_iter, (0, pad_size))
        
        hidden_states_fp8_tensor_iter = hidden_states_per_token_iter.to(torch.float8_e4m3fn)
        scale_tensor_iter = torch.ones(
            hidden_states_per_token_iter.shape[0], hidden_states_per_token_iter.shape[1] // 128, 
            device=hidden_states_per_token_iter.device, dtype=torch.float32
        )
        
        tokens_per_local_expert = int(batch_size * input_len * 8 * num_rank // num_experts)
        if tokens_per_local_expert > 0:
            num_recv = [tokens_per_local_expert] * num_local_experts
        else:
            continue 

        # Reinitialize topk_idx and topk_weights for each iteration
        num_tokens_iter = hidden_states_per_token_iter.shape[0]
        topk_idx_iter = torch.full((num_tokens_iter, 8), -1, device=device, dtype=torch.int32)
        
        # Calculate total valid positions needed
        total_valid_positions = sum(num_recv)

        # Create a list of expert indices where each expert appears tokens_per_local_expert times
        expert_indices_list = []
        for expert_id in range(num_local_experts):
            expert_indices_list.extend([expert_id] * tokens_per_local_expert)

        # Shuffle the expert indices to randomize their distribution
        expert_indices_tensor = torch.tensor(expert_indices_list, device=device, dtype=torch.int32)
        shuffled_indices = torch.randperm(len(expert_indices_tensor), device=device)
        expert_indices_tensor = expert_indices_tensor[shuffled_indices]

        # Distribute the expert indices across the topk_idx_iter tensor more evenly
        # Calculate how many positions each row should have
        positions_per_row = total_valid_positions // num_tokens_iter
        extra_positions = total_valid_positions % num_tokens_iter

        valid_positions_count = 0
        for i in range(num_tokens_iter):
            # Determine how many positions this row should have
            current_row_positions = positions_per_row + (1 if i < extra_positions else 0)
            
            # Fill this row with expert indices
            for j in range(current_row_positions):
                if valid_positions_count < total_valid_positions:
                    topk_idx_iter[i, j] = expert_indices_tensor[valid_positions_count]
                    valid_positions_count += 1
                else:
                    break
            
            if valid_positions_count >= total_valid_positions:
                break

        topk_idx_shuffled = topk_idx_iter.clone()

        # First, collect the number of non-negative values per row
        non_negative_counts_per_row = (topk_idx_iter != -1).sum(dim=1)

        # Collect all non-negative values
        all_non_negative_values = []
        for i in range(num_tokens_iter):
            for j in range(8):
                if topk_idx_iter[i, j] != -1:
                    all_non_negative_values.append(topk_idx_iter[i, j])

        # Shuffle all non-negative values
        shuffled_values = torch.tensor(all_non_negative_values, device=device)[torch.randperm(len(all_non_negative_values), device=device)]

        # Calculate target distribution per column
        target_per_col = len(all_non_negative_values) // 8
        extra_per_col = len(all_non_negative_values) % 8

        # Clear the tensor
        topk_idx_shuffled.fill_(-1)

        # Redistribute values while preserving row structure
        value_idx = 0
        for col in range(8):
            target_count = target_per_col + (1 if col < extra_per_col else 0)
            positions_filled = 0
            
            # Fill this column with values, but respect row constraints
            for row in range(num_tokens_iter):
                if positions_filled < target_count and value_idx < len(shuffled_values):
                    # Check if this row can accept more values
                    current_row_count = (topk_idx_shuffled[row, :] != -1).sum().item()
                    original_row_count = non_negative_counts_per_row[row].item()
                    
                    if current_row_count < original_row_count:
                        topk_idx_shuffled[row, col] = shuffled_values[value_idx]
                        positions_filled += 1
                        value_idx += 1
                elif positions_filled >= target_count:
                    break

        topk_idx_iter = topk_idx_shuffled
        
        # Initialize topk_weights based on topk_idx
        topk_weights_iter = torch.zeros(num_tokens_iter, 8, device=device, dtype=torch.float32)
        for i in range(num_tokens_iter):
            # Count non-negative values in this row
            valid_count = (topk_idx_iter[i] != -1).sum().item()
            if valid_count > 0:
                # Set weight for valid positions
                weight_value = 1.0 / ep_size / (8 // ep_size)
                topk_weights_iter[i, topk_idx_iter[i] != -1] = weight_value
        
        
        # Assign tokens to local experts (assuming this card handles experts 0 to num_local_experts-1)
       

        dispatch_output = DeepEPNormalOutput(
            hidden_states=(hidden_states_fp8_tensor_iter, scale_tensor_iter),
            topk_idx=topk_idx_iter,
            topk_weights=topk_weights_iter,
            num_recv_tokens_per_expert=num_recv
        )

 
        if tp_rank == 0:
            rank_print(f"Prefill setup:")
            rank_print(f"  topk_idx: {topk_idx_iter.shape}, device: {topk_idx_iter.device}, dtype: {topk_idx_iter.dtype}")
            rank_print(f"  topk_weights: {topk_weights_iter.shape}, device: {topk_weights_iter.device}, dtype: {topk_weights_iter.dtype}")
            rank_print(f"  num_recv: {num_recv}")
            rank_print(f"  ep_size: {sim_ep_size}, total_tokens: {total_tokens}")
            rank_print(f"  num_local_experts: {num_local_experts}")
            rank_print(f"  Sample topk_idx[0]: {topk_idx_iter[0]}")
            rank_print(f"  Sample topk_weights[0]: {topk_weights_iter[0]}")
            rank_print(f"  topk_id sum: {sum(sum(topk_idx_iter!=-1))}")
            rank_print(f"  topk_id expert 0: {sum(sum(topk_idx_iter==1))}")
        
        if batch_size == 32 and input_len == 8:
            print(topk_idx_iter)
            print(topk_weights_iter)
            print(num_recv)
            print(num_local_experts)
   
        # Warmup iterations0
        for _ in range(bench_args.num_warmup):
            hidden_states_fp8_tensor_iter = hidden_states_per_token_iter.to(torch.float8_e4m3fn)
            scale_tensor_iter = torch.ones(
                hidden_states_per_token_iter.shape[0], hidden_states_per_token_iter.shape[1] // 128, 
                device=hidden_states_per_token_iter.device, dtype=torch.float32
            )
            dispatch_output = DeepEPNormalOutput(
                hidden_states=(hidden_states_fp8_tensor_iter, scale_tensor_iter),
                topk_idx=topk_idx_iter.clone(),
                topk_weights=topk_weights_iter.clone(),
                num_recv_tokens_per_expert=num_recv
            )
            # print("before moe_impl",topk_idx_iter)
            _ = moe_layer.experts.moe_impl(dispatch_output)
            # print(t)

        torch.get_device_module(device).synchronize()
        torch.cuda.empty_cache()
        
        # Benchmark GEMM computation time
        gemm_latencies = []
        profiler = None
        if enable_profile:
            profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                with_stack=True,
            )
            
            profiler.start()

        for i in range(bench_args.num_iterations):
            hidden_states_fp8_tensor_iter = hidden_states_per_token_iter.to(torch.float8_e4m3fn)
            scale_tensor_iter = torch.ones(
                hidden_states_per_token_iter.shape[0], hidden_states_per_token_iter.shape[1] // 128, 
                device=hidden_states_per_token_iter.device, dtype=torch.float32
            )
            dispatch_output = DeepEPNormalOutput(
                hidden_states=(hidden_states_fp8_tensor_iter, scale_tensor_iter),
                topk_idx=topk_idx_iter.clone(),
                topk_weights=topk_weights_iter.clone(),
                num_recv_tokens_per_expert=num_recv
            )
            # Prefill
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

        if enable_profile:
            profiler.stop()
            profile_filename = f"/profiler_output/{profile_filename_prefix}_batch{batch_size}_input{input_len}.trace.json.gz"
            parent_dir = os.path.dirname(os.path.abspath(profile_filename))
            os.makedirs(parent_dir, exist_ok=True)
            profiler.export_chrome_trace(profile_filename)
            torch.cuda.empty_cache()

        # Calculate statistics
        avg_latency_ms = np.mean(gemm_latencies)
        min_latency_ms = np.min(gemm_latencies)
        max_latency_ms = np.max(gemm_latencies)
        std_latency_ms = np.std(gemm_latencies)
        throughput = hidden_states_per_token_iter.shape[0] / (avg_latency_ms / 1000)
        
        if tp_rank == 0:
            rank_print(f"DeepEP MoE GEMM Results (Prefill):")
            rank_print(f"  Average latency: {avg_latency_ms:.3f}ms ± {std_latency_ms:.3f}ms")
            rank_print(f"  Min latency: {min_latency_ms:.3f}ms")
            rank_print(f"  Max latency: {max_latency_ms:.3f}ms")
            rank_print(f"  Throughput: {throughput:.2f} tokens/s")
            
        prefill_result = MoEBenchResult(
            run_name=bench_args.run_name,
            layer_id=bench_args.test_layer,
            batch_size=sim_batch,
            input_len=input_len,
            output_len=1,
            phase="prefill_gemm_test",
            avg_latency_ms=avg_latency_ms,
            min_latency_ms=min_latency_ms,
            max_latency_ms=max_latency_ms,
            std_latency_ms=std_latency_ms,
            throughput_tokens_per_sec=0,
            num_iterations=bench_args.num_iterations,
            cuda_graph_used=False
        )
        results.append(prefill_result)
        del hidden_states_per_token_iter, hidden_states_fp8_tensor_iter, scale_tensor_iter, topk_idx_iter, topk_weights_iter, num_recv, dispatch_output
        torch.cuda.empty_cache()

    return results

def benchmark_moe_layer_decode(
    model_runner,
    server_args,
    port_args,
    bench_args: MoEBenchArgs,
    rank_print,
    device: str,
    tp_rank: int,
    enable_profile: bool = False,
    profile_filename_prefix: str = ""
) -> List[MoEBenchResult]:
    results = []

    decode_test_cases = get_moe_decode_test_cases()
    rank_print(f"Testing {len(decode_test_cases)} decode configurations...")
    
    # Clear the pools
    model_runner.req_to_token_pool.clear()
    model_runner.token_to_kv_pool_allocator.clear()
    
    for batch_size in decode_test_cases:

        moe_layer = model_runner.model.model.layers[bench_args.test_layer].mlp
        
        # _ = moe_layer(hidden_states, forward_batch)

        num_experts = moe_layer.config.n_routed_experts  # Total number of experts
        ep_size = server_args.ep_size
        top_k = moe_layer.topk.top_k  # Get top_k from the moe layer
        total_tokens = batch_size * top_k  # For decode, we have batch_size tokens with top_k experts each
        num_local_experts  = int(num_experts/ep_size)

        if num_experts  != 256:
            sim_ep_size = int(256//num_experts * ep_size)
            sim_num_experts = 256
            sim_batch = int(batch_size * 256//num_experts)
        else:
            sim_ep_size = ep_size
            sim_num_experts = num_experts
            sim_batch = batch_size

        rank_print(f"Warming up decode: batch={batch_size}, sim_batch={sim_batch}, sim_ep_size={sim_ep_size}")

        num_ranks = sim_ep_size  # num_ranks equals ep_size       
        num_max_dispatch_tokens_per_rank = 128
        hidden_size = model_runner.model.config.hidden_size
        
        # Ensure hidden_size is divisible by 128 for FP8 quantization
        if hidden_size % 128 != 0:
            pad_size = 128 - (hidden_size % 128)
            hidden_size += pad_size
        
        # Create hidden_states with the correct shape
        hidden_states = torch.randn(
            num_local_experts, num_max_dispatch_tokens_per_rank * num_ranks, hidden_size,
            dtype=torch.bfloat16, device="cuda"
        )
        
        scale_hidden_size = hidden_size // 128
        scale_tensor = torch.ones(
            num_local_experts, num_max_dispatch_tokens_per_rank * num_ranks, scale_hidden_size, 
            device=hidden_states.device, dtype=torch.float32
        )
        hidden_states_fp8_tensor = hidden_states.to(torch.float8_e4m3fn)
        
        masked_m = torch.zeros(num_local_experts, device=device, dtype=torch.int32)
        
        # Calculate base tokens per expert and remainder
        base_tokens_per_expert = int(batch_size * top_k) * num_ranks // num_experts
        if base_tokens_per_expert == 0:
            masked_m[:int(batch_size * top_k) * num_ranks//ep_size] = 1
        else:
            masked_m[:] = base_tokens_per_expert
        
        # Calculate expected_m for DeepEPLLOutput
        expected_m = int(max(masked_m)) #.to('cpu')
        # print(f"expected_m: {expected_m.dtype}, {expected_m.device}")

        # Debug information (only for rank 0)
        if tp_rank == 0:
            print(f"DeepEP Low Latency setup:")
            print(f"  num_experts: {num_experts}")
            print(f"  ep_size: {sim_ep_size}")
            print(f"  num_ranks: {num_ranks}")
            print(f"  num_local_experts: {num_local_experts}")
            print(f"  num_max_dispatch_tokens_per_rank: {num_max_dispatch_tokens_per_rank}")
            print(f"  hidden_states shape: {hidden_states.shape}")
            print(f"  scale_tensor shape: {scale_tensor.shape}")
            print(f"  masked_m shape: {masked_m.shape}")
            print(f"  masked_m values: {masked_m}")
            print(f"  masked_m sum: {masked_m.sum()}")
            print(f"  expected_m: {expected_m}")
            print(f"  Token distribution: {[f'Expert {i}: {masked_m[i]} tokens' for i in range(min(5, num_local_experts))]}")
            if num_local_experts > 5:
                print(f"  ... and {num_local_experts - 5} more experts")
            print(f"  CUDA Graph will be used for benchmarking")
        
        torch.get_device_module(device).synchronize()
        torch.cuda.empty_cache()
        
        # Warmup iterations to capture CUDA Graph
        for _ in range(bench_args.num_warmup):
            scale_tensor = torch.ones(
                num_local_experts, num_max_dispatch_tokens_per_rank * num_ranks, scale_hidden_size, 
                device=hidden_states.device, dtype=torch.float32
            )
            hidden_states_fp8_tensor = hidden_states.to(torch.float8_e4m3fn)
            dispatch_output = DeepEPLLOutput(
                hidden_states_fp8=(hidden_states_fp8_tensor, scale_tensor),
                topk_idx=torch.empty(0, device=device, dtype=torch.int32),  # Empty tensor for low latency mode
                topk_weights=torch.empty(0, device=device, dtype=torch.float32),  # Empty tensor for low latency mode
                masked_m=masked_m,
                expected_m=expected_m
            )
            _ = moe_layer.experts.moe_impl(dispatch_output)
        
        # Capture CUDA Graph using best practices
        torch.get_device_module(device).synchronize()
        torch.cuda.empty_cache()
        
        # Create CUDA Graph with proper memory pool and stream management
        graph = torch.cuda.CUDAGraph()

        scale_tensor = torch.ones(
                num_local_experts, num_max_dispatch_tokens_per_rank * num_ranks, scale_hidden_size, 
                device=hidden_states.device, dtype=torch.float32
            )
        hidden_states_fp8_tensor = hidden_states.to(torch.float8_e4m3fn)
        dispatch_output = DeepEPLLOutput(
            hidden_states_fp8=(hidden_states_fp8_tensor, scale_tensor),
            topk_idx=torch.empty(0, device=device, dtype=torch.int32),  # Empty tensor for low latency mode
            topk_weights=torch.empty(0, device=device, dtype=torch.float32),  # Empty tensor for low latency mode
            masked_m=masked_m,
            expected_m=expected_m
        )
        
        # Capture the graph
        with torch.cuda.graph(graph):
            _ = moe_layer.experts.moe_impl(dispatch_output)
        
        # Replay the graph for benchmarking
        graph.replay()
            
        torch.get_device_module(device).synchronize()

        profiler = None
        if enable_profile:
            profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=True,  # Disable shape recording
                profile_memory=True,  # Disable memory profiling
                with_stack=True,    
            )
            
            profiler.start()
        
        # Benchmark GEMM computation time using CUDA Graph
        gemm_latencies = []
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        graph.replay()
       
        start_event.record()
        torch.get_device_module(device).synchronize()
        for i in range(bench_args.num_iterations):
 
            graph.replay()

        torch.get_device_module(device).synchronize()
        end_event.record()
        latency_ms = start_event.elapsed_time(end_event)

        gemm_latencies.append(latency_ms/bench_args.num_iterations)
       

        if enable_profile:
            profiler.stop()
            profile_filename = f"{profile_filename_prefix}_batch{batch_size}.trace.json.gz"
            parent_dir = os.path.dirname(os.path.abspath(profile_filename))
            os.makedirs(parent_dir, exist_ok=True)
            profiler.export_chrome_trace(profile_filename)
            rank_print(f"torch profiler chrome trace saved to {profile_filename}") 
        
    # Calculate statistics
        avg_latency_ms = np.mean(gemm_latencies)
        min_latency_ms = np.min(gemm_latencies)
        max_latency_ms = np.max(gemm_latencies)
        std_latency_ms = np.std(gemm_latencies)
     
        if tp_rank == 0:
            rank_print(f"DeepEP MoE GEMM Results (Decode) - CUDA Graph Enabled:")
            rank_print(f"  Average latency: {avg_latency_ms:.3f}ms ± {std_latency_ms:.3f}ms")
            rank_print(f"  Min latency: {min_latency_ms:.3f}ms")
            rank_print(f"  Max latency: {max_latency_ms:.3f}ms")
            
        decode_result = MoEBenchResult(
            run_name=bench_args.run_name,
            layer_id=bench_args.test_layer,
            batch_size=sim_batch,
            input_len=0,
            output_len=0,
            phase="decode_gemm_test",
            avg_latency_ms=avg_latency_ms,
            min_latency_ms=min_latency_ms,
            max_latency_ms=max_latency_ms,
            std_latency_ms=std_latency_ms,
            throughput_tokens_per_sec=0,
            num_iterations=bench_args.num_iterations,
            cuda_graph_used=True
        )
        results.append(decode_result)
        del hidden_states, hidden_states_fp8_tensor, scale_tensor, dispatch_output
        torch.cuda.empty_cache()

    return results


def run_moe_benchmark(
    server_args,
    port_args,
    bench_args: MoEBenchArgs,
    tp_rank: int,
):
    """Run the complete MoE benchmark"""
    
    # Set CPU affinity
    if get_bool_env_var("SGLANG_SET_CPU_AFFINITY"):
        set_gpu_proc_affinity(server_args.tp_size, server_args.nnodes, tp_rank)

    # Configure the logger
    configure_logger(server_args, prefix=f" TP{tp_rank}")
    rank_print = print if tp_rank == 0 else lambda *args, **kwargs: None

    # Check if model has MoE layers and get the specified layer
    moe_layer_id = bench_args.test_layer  # Use the test_layer from bench_args
    
    # Run benchmarks for all configurations
    all_results = []
    
    rank_print(f"\n{'='*60}")
    rank_print(f"Testing MoE Layer {moe_layer_id}")
    rank_print(f"{'='*60}")

    # Load model for decode
    model_runner, _ = load_model_with_dummy_weights(server_args, port_args, tp_rank)
    reqs = prepare_synthetic_inputs(128,16)  # Fixed output_len=16
    
    batch = ScheduleBatch.init_new(
        reqs=reqs,
        req_to_token_pool=model_runner.req_to_token_pool,
        token_to_kv_pool_allocator=model_runner.token_to_kv_pool_allocator,
        tree_cache=None,
        model_config=model_runner.model_config,
        enable_overlap=False,
        spec_algorithm=SpeculativeAlgorithm.NONE,
    )
    batch.prepare_for_extend()
    _maybe_prepare_mlp_sync_batch(batch, model_runner)
    
    try:

        results = benchmark_moe_layer_prefill(
            model_runner,
            server_args,
            port_args,
            bench_args,
            rank_print,
            server_args.device,
            tp_rank,
            bench_args.enable_profile if tp_rank == 0 else False,  # Always pass the profile setting
            bench_args.profile_filename_prefix,
        )
        all_results.extend(results)


        results = benchmark_moe_layer_decode(
            model_runner,
            server_args,
            port_args,
            bench_args,
            rank_print,
            server_args.device,
            tp_rank,
            bench_args.enable_profile if tp_rank == 0 else False,  # Always pass the profile setting
            bench_args.profile_filename_prefix,
        )
        all_results.extend(results)

    except Exception as e:
        rank_print(f"Error during MoE benchmark: {e}")
        import traceback
        rank_print(f"Traceback: {traceback.format_exc()}")
        return
    
    # Write results in CSV format on rank 0 only
    if tp_rank == 0 and bench_args.result_filename:
        try:
            with open(bench_args.result_filename, "w") as fout:
                # Write header
                fout.write("framework,version,device,op_name,kernel_source,moe_dtype,num_tokens,hidden_size,inter_size,topk,num_experts,moe_tp_size,moe_ep_size,distribution,latency\n")
                
                for result in all_results:
                    # Get model configuration for additional parameters
                    moe_layer = model_runner.model.model.layers[result.layer_id].mlp
                    num_experts = moe_layer.config.n_routed_experts
                    top_k = moe_layer.topk.top_k
                    hidden_size = model_runner.model.config.hidden_size
                    inter_size = moe_layer.config.intermediate_size
                    
                    # Calculate num_tokens based on phase
                    if "prefill" in result.phase:
                        num_tokens = result.batch_size * result.input_len
                    else:  # decode phase
                        num_tokens = result.batch_size * top_k
                    
                    # Map phase to op_name
                    op_name = "moe"
                    
                    # Determine kernel_source based on phase
                    kernel_source = "deepepmoe"
                    
                    # Get moe_dtype (assuming fp8 based on the context file)
                    moe_dtype = "fp8"
                    
                    # Get device info
                    device_name = get_gpu_device_name()
                    
                    # Get moe_tp_size and moe_ep_size
                    moe_tp_size = server_args.tp_size
                    moe_ep_size = server_args.ep_size
                    
                    # Distribution type
                    distribution = "uniform"
                    
                    # Write CSV line
                    fout.write(f"SGLang,1.0.0,{device_name},{op_name},{kernel_source},{moe_dtype},{num_tokens},{hidden_size},{inter_size},{top_k},{num_experts},{moe_tp_size},{moe_ep_size},{distribution},{result.avg_latency_ms}\n")
            
            rank_print(f"\nResults saved to {bench_args.result_filename}")
        except Exception as e:
            rank_print(f"Error writing results to file: {e}")
    
    # Clean up distributed environment if needed
    # if server_args.tp_size > 1:
    #     destroy_distributed_environment()
    
    # Clean up CUDA memory
    torch.cuda.empty_cache()
    
    rank_print(f"\n{'='*60}")
    rank_print("BENCHMARK COMPLETED SUCCESSFULLY")
    rank_print(f"{'='*60}")


def main(server_args, bench_args: MoEBenchArgs):
    """Main function"""
    server_args.cuda_graph_max_bs = max(bench_args.batch_size)
    
    # DeepEP MoE settings are already configured in server_args
    # No need to set them again

    _set_envs_and_config(server_args)

    if not server_args.model_path:
        raise ValueError("Provide --model-path for running the tests")

    port_args = PortArgs.init_new(server_args)

    if server_args.tp_size == 1:
        run_moe_benchmark(server_args, port_args, bench_args, 0)
    else:
        workers = []
        for tp_rank in range(server_args.tp_size):
            proc = multiprocessing.Process(
                target=run_moe_benchmark,
                args=(
                    server_args,
                    port_args,
                    bench_args,
                    tp_rank,
                ),
            )
            proc.start()
            workers.append(proc)

        # Wait for all processes to complete
        for proc in workers:
            proc.join()
            
        # Check if any process failed
        for i, proc in enumerate(workers):
            if proc.exitcode != 0:
                print(f"Process {i} (tp_rank={i}) failed with exit code {proc.exitcode}")
                
        # Clean up processes
        for proc in workers:
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=5)
                if proc.is_alive():
                    proc.kill()

DEEPSEEK_MODEL_PATH = os.environ.get("DEEPSEEK_MODEL_PATH", "/root/fac/deepseek-v3")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark DeepEP MoE layers with dummy weights")
    parser.add_argument("--test-layer", type=int, 
                        default=int(os.environ.get("SGLANG_TEST_LAYER", "3")),
                        help="Which layer's MoE module to test (default: 3 for 4th layer)")
    parser.add_argument("--num-warmup", type=int, default=3,
                        help="Number of warmup iterations")
    parser.add_argument("--num-iterations", type=int, default=10,
                        help="Number of benchmark iterations")
    parser.add_argument("--output", type=str, default="/root/fac/llm-pet/deepep_moe_benchmark_results_ep8_test.txt",
                        help="Output file for results")
    parser.add_argument("--model-path", type=str, default=DEEPSEEK_MODEL_PATH,
                        help="Model path or HuggingFace model ID")
    parser.add_argument("--dtype", type=str, default="auto",
                        help="Model dtype (auto, float16, bfloat16)")
    parser.add_argument("--quick-test", action="store_true",
                        help="Run quick test with small configurations")
    parser.add_argument("--enable-deepep-moe", action="store_true", default=True,
                        help="Enable DeepEP MoE")
    parser.add_argument("--batch-size", type=int, nargs="+", default=[1, 4, 8, 16],
                        help="Batch sizes to test")
    parser.add_argument("--input-len", type=int, nargs="+", default=[128, 512, 1024, 2048],
                        help="Input sequence lengths to test")
    parser.add_argument("--output-len", type=int, nargs="+", default=[16],
                        help="Output sequence lengths to test")
    
    args = parser.parse_args()
    
    # Create ServerArgs manually with the available arguments
    server_args = ServerArgs(
        model_path=args.model_path,
        dtype=args.dtype,
        device="cuda",
        load_format="dummy",
        tp_size=4,  # Use single process for simplicity
        trust_remote_code=True,
        mem_fraction_static=0.3,
        enable_deepep_moe=args.enable_deepep_moe,
        enable_ep_moe=True,
        ep_size=4,
        node_rank=0,
        host="localhost",
        port=30000,
    )
    
    # Create BenchArgs manually with the available arguments
    if args.quick_test:
        # Use smaller configurations for quick test
        batch_sizes = [1, 2]
        input_lens = [64, 128]
    else:
        # Use command-line arguments or defaults
        batch_sizes = args.batch_size
        input_lens = args.input_len
    
    bench_args = MoEBenchArgs(
        run_name="deepep_moe_test",
        batch_size=tuple(batch_sizes),
        input_len=tuple(input_lens),
        output_len=tuple(args.output_len),
        result_filename=args.output,
        enable_profile=True,
        profile_filename_prefix="deepep_moe_profile",
        num_warmup=args.num_warmup,
        num_iterations=args.num_iterations,
        test_layer=args.test_layer,
    )

    logging.basicConfig(
        level=getattr(logging, server_args.log_level.upper()),
        format="%(message)s",
    )

    try:
        main(server_args, bench_args)
        print("\n" + "="*60)
        print("SCRIPT COMPLETED SUCCESSFULLY")
        print("="*60)
    except Exception as e:
        print(f"\nScript failed with error: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        raise
    finally:
        # Only kill process tree if we're using multiple processes
        if server_args.tp_size > 1:
            kill_process_tree(os.getpid(), include_parent=False) 