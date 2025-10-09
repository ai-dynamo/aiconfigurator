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


def get_moe_prefill_test_cases(ep_size, rank):
    """Get test cases for MoE prefill phase - batch_size and input_len dimensions"""
    test_cases = []

    num_tokens = [4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384] #4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]  

    for num_token in sorted(num_tokens):

        # rank = 2 if num_expert == 256 else 2 * 256//num_expert
        if num_token * 8  < 128:  # tokens per rank
            continue
        if num_token * rank > 256*2048: 
            continue
        
        # Add test cases with batch_size and input_len
        test_cases.append(num_token)
    
    return test_cases

def get_moe_decode_test_cases():
    """Get test cases for MoE decode phase - batch_size and kv_len dimensions"""
    test_cases = []
    
    # Decode parameters - batch_size and kv_len dimensions
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]  # More reasonable batch sizes
    # batch_sizes = [1/8, 1/4, 1/2]  # More reasonable batch sizes
    
    test_cases = batch_sizes
    
    return test_cases

def sample_power_law(size, alpha, xmin, xmax):
    u = torch.rand(size)
    inv_cdf = ((xmax**(1-alpha) - xmin**(1-alpha)) * u + xmin**(1-alpha))**(1/(1-alpha))
    return inv_cdf

# NOTE: power_law_logits_v3 was copied from aiconfigurator/collector/trtllm/collect_moe.py and modified
# TODO: restrict the max token per expert to be num_local_token * num_rank 
def power_law_logits_v3(num_tokens, num_experts, topk, ep, alpha):
    if num_tokens*topk > num_experts:
        num_tokens_per_expert = sample_power_law(num_experts, alpha, 1, num_tokens*0.8)
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
                    num_tokens_per_expert[torch.argmax(num_tokens_per_expert)] -=1
    
    if len(num_tokens_per_expert) > 1:
        sorted_tokens = torch.sort(num_tokens_per_expert, descending=True)[0]
        assert sorted_tokens[0] >= sorted_tokens[-1], "Power law distribution pattern disrupted"

    with torch.no_grad():
        conv1d = torch.nn.Conv1d(in_channels=1, out_channels=1, kernel_size=num_experts//ep, stride=num_experts//ep, padding=0, bias=False)
        conv1d_weights = torch.tensor([1 for _ in range(num_experts//ep)])
        conv1d.weight.copy_(conv1d_weights)

    res = conv1d(num_tokens_per_expert.unsqueeze(0).unsqueeze(0).float())
    max_ep_idx = torch.argmax(res).item()
    return num_tokens_per_expert.view(ep, num_experts // ep)[max_ep_idx].view(-1)

@dataclass
class MoEBenchArgs:
    run_name: str = "deepep_moe_test"
    num_warmup: int = 3
    num_iterations: int = 10
    test_layer: int = 3
    num_experts: int = 128


@dataclass
class MoEBenchResult:
    """Result for a single MoE benchmark run"""
    run_name: str
    layer_id: int
    num_token: int
    phase: str  # "prefill" or "decode"
    avg_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    std_latency_ms: float
    num_iterations: int
    num_experts: int
    cuda_graph_used: bool = False

def load_model_with_dummy_weights(server_args, port_args, tp_rank):
    """Load model with dummy weights and limited layers for MoE testing"""
    suppress_other_loggers()
    rank_print = print if tp_rank == 0 else lambda *args, **kwargs: None

    # Override model config to load only first 4 layers (we'll test the 4th layer)
    # Note: n_routed_experts should already be set in server_args.json_model_override_args
    if server_args.load_format == "dummy":
        # Parse existing override args to preserve n_routed_experts
        existing_override = {}
        if server_args.json_model_override_args:
            existing_override = json.loads(server_args.json_model_override_args)
        
        # Only override num_hidden_layers, preserve other settings like n_routed_experts
        existing_override["num_hidden_layers"] = 4
        server_args.json_model_override_args = json.dumps(existing_override)

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
    
    rank_print(f"Model loaded successfully.")
    

    
    if server_args.tp_size > 1:
        dist.barrier()
    
    return model_runner


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
    prefill_test_cases: List[int],
    moe_layer,
    num_experts: int,
    ep_size: int,
    num_rank: int
) -> List[MoEBenchResult]:

    results = []
    num_local_experts = num_experts // ep_size
   

    for num_token in prefill_test_cases:

        model_runner.req_to_token_pool.clear()
        model_runner.token_to_kv_pool_allocator.clear()

        total_tokens = num_token * 8
        
        # Fake dispatch outputs with random data
        hidden_states_per_token_iter = torch.randn(
                    int(num_token * num_rank), model_runner.model.config.hidden_size,
                    dtype=torch.bfloat16, device=device
                )
                
        if hidden_states_per_token_iter.shape[1] % 128 != 0:
            pad_size = 128 - (hidden_states_per_token_iter.shape[1] % 128)
            hidden_states_per_token_iter = torch.nn.functional.pad(hidden_states_per_token_iter, (0, pad_size))
        
        hidden_states_fp8_tensor_iter = hidden_states_per_token_iter.to(torch.float8_e4m3fn)
        scale_tensor_iter = torch.ones(
            hidden_states_per_token_iter.shape[0], hidden_states_per_token_iter.shape[1] // 128, 
            device=hidden_states_per_token_iter.device, dtype=torch.float32
        )
        
        tokens_per_local_expert = int(num_token * 8 * num_rank // 256)
        print(f"tokens_per_local_expert: {tokens_per_local_expert}")
        if tokens_per_local_expert > 0:
            num_recv = [tokens_per_local_expert] * num_local_experts
        else:
            continue 

        num_tokens_iter = hidden_states_per_token_iter.shape[0]
        topk_idx_iter = torch.full((num_tokens_iter, 8), -1, device=device, dtype=torch.int32)
        
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
                    topk_idx_iter[i, j] = expert_indices_tensor[valid_positions_count]
                    valid_positions_count += 1
                else:
                    break
            
            if valid_positions_count >= total_valid_positions:
                break

        topk_idx_shuffled = topk_idx_iter.clone()

        non_negative_counts_per_row = (topk_idx_iter != -1).sum(dim=1)

        all_non_negative_values = []
        for i in range(num_tokens_iter):
            for j in range(8):
                if topk_idx_iter[i, j] != -1:
                    all_non_negative_values.append(topk_idx_iter[i, j])

        shuffled_values = torch.tensor(all_non_negative_values, device=device)[torch.randperm(len(all_non_negative_values), device=device)]

        target_per_col = len(all_non_negative_values) // 8
        extra_per_col = len(all_non_negative_values) % 8

        topk_idx_shuffled.fill_(-1)

        value_idx = 0
        for col in range(8):
            target_count = target_per_col + (1 if col < extra_per_col else 0)
            positions_filled = 0
            
            for row in range(num_tokens_iter):
                if positions_filled < target_count and value_idx < len(shuffled_values):
                    current_row_count = (topk_idx_shuffled[row, :] != -1).sum().item()
                    original_row_count = non_negative_counts_per_row[row].item()
                    
                    if current_row_count < original_row_count:
                        topk_idx_shuffled[row, col] = shuffled_values[value_idx]
                        positions_filled += 1
                        value_idx += 1
                elif positions_filled >= target_count:
                    break

        topk_idx_iter = topk_idx_shuffled
        
        topk_weights_iter = torch.zeros(num_tokens_iter, 8, device=device, dtype=torch.float32)
        for i in range(num_tokens_iter):
            valid_count = (topk_idx_iter[i] != -1).sum().item()
            if valid_count > 0:
                weight_value = 1.0 / ep_size / (8 // ep_size)
                topk_weights_iter[i, topk_idx_iter[i] != -1] = weight_value
        
        dispatch_output = DeepEPNormalOutput(
            hidden_states=(hidden_states_fp8_tensor_iter, scale_tensor_iter),
            topk_idx=topk_idx_iter,
            topk_weights=topk_weights_iter,
            num_recv_tokens_per_expert=num_recv
        )


        # if tp_rank == 0:
        #     rank_print(f"Prefill setup:")
        #     rank_print(f"  topk_idx: {topk_idx_iter.shape}, device: {topk_idx_iter.device}, dtype: {topk_idx_iter.dtype}")
        #     rank_print(f"  topk_weights: {topk_weights_iter.shape}, device: {topk_weights_iter.device}, dtype: {topk_weights_iter.dtype}")
        #     rank_print(f"  num_recv: {num_recv}")
        #     rank_print(f"  ep_size: {ep_size}, total_tokens: {total_tokens}")
        #     rank_print(f"  num_local_experts: {num_local_experts}")
        #     rank_print(f"  Sample topk_idx[0]: {topk_idx_iter[0]}")
        #     rank_print(f"  Sample topk_weights[0]: {topk_weights_iter[0]}")
        #     rank_print(f"  topk_id sum: {sum(sum(topk_idx_iter!=-1))}")
        #     rank_print(f"  topk_id expert 0: {sum(sum(topk_idx_iter==1))}")
        
        # Warmup 
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
        
        gemm_latencies = []
        
        # Use profiler for synchronization 
        profiler = torch.profiler.profile(
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

        profiler.stop()
        torch.cuda.empty_cache()

        # Calculate statistics
        avg_latency_ms = np.mean(gemm_latencies)
        min_latency_ms = np.min(gemm_latencies)
        max_latency_ms = np.max(gemm_latencies)
        std_latency_ms = np.std(gemm_latencies)
        
        if tp_rank == 0:
            rank_print(f"DeepEP MoE GEMM Results (Prefill):")
            rank_print(f"  Average latency: {avg_latency_ms:.3f}ms ± {std_latency_ms:.3f}ms")
            rank_print(f"  Min latency: {min_latency_ms:.3f}ms")
            rank_print(f"  Max latency: {max_latency_ms:.3f}ms")
            
        prefill_result = MoEBenchResult(
            run_name=bench_args.run_name,
            layer_id=bench_args.test_layer,
            num_token=num_token,
            phase="prefill_gemm_test",
            avg_latency_ms=avg_latency_ms,
            min_latency_ms=min_latency_ms,
            max_latency_ms=max_latency_ms,
            std_latency_ms=std_latency_ms,
            num_iterations=bench_args.num_iterations,
            num_experts=num_experts,
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
    decode_test_cases: List[int],
    moe_layer,
    num_experts: int,
    ep_size: int,
    num_rank: int,
    distributed = "power_law", 
    power_law_alpha = 0.8,
) -> List[MoEBenchResult]:
    results = []
    
    # Clear the pools
    model_runner.req_to_token_pool.clear()
    model_runner.token_to_kv_pool_allocator.clear()
    top_k = moe_layer.topk.top_k  # 
    num_local_experts  = int(num_experts//ep_size)
    
    for num_token in decode_test_cases:

        num_local_experts  = int(num_experts/ep_size)
    
        num_max_dispatch_tokens_per_rank = 128
        hidden_size = model_runner.model.config.hidden_size
        
        if hidden_size % 128 != 0:
            pad_size = 128 - (hidden_size % 128)
            hidden_size += pad_size
        
        hidden_states = torch.randn(
            num_local_experts, num_max_dispatch_tokens_per_rank * num_rank, hidden_size,
            dtype=torch.bfloat16, device="cuda"
        )
        
        scale_hidden_size = hidden_size // 128
        scale_tensor = torch.ones(
            num_local_experts, num_max_dispatch_tokens_per_rank * num_rank, scale_hidden_size, 
            device=hidden_states.device, dtype=torch.float32
        )
        hidden_states_fp8_tensor = hidden_states.to(torch.float8_e4m3fn)
        
        masked_m = torch.zeros(num_local_experts, device=device, dtype=torch.int32)

        # support two distributed mode: power_law and uniform
        if distributed == "power_law":
            masked_m_list = [power_law_logits_v3(num_token * num_rank, num_experts, top_k, ep_size, power_law_alpha).to(masked_m.dtype).to(torch.device(device)) for _ in range(5)]
        elif distributed == "uniform":
            # expert size is 256
            base_tokens_per_expert = int(num_token * top_k) * num_rank // 256
            if base_tokens_per_expert == 0:
                masked_m[:int(num_token * top_k) * num_rank//ep_size] = 1
            else:
                masked_m[:] = base_tokens_per_expert
            masked_m_list = [masked_m]
        else:
            raise ValueError(f"Unsupported distributed mode: {distributed}")        
        
        scale_tensor = torch.ones(
            num_local_experts, num_max_dispatch_tokens_per_rank * num_rank, scale_hidden_size, 
            device=hidden_states.device, dtype=torch.float32
        )
        hidden_states_fp8_tensor = hidden_states.to(torch.float8_e4m3fn)

        # Create dispatch_output_list with independent tensor copies for each entry
        # Each dispatch_output needs its own copy because moe_impl will dispose/delete the tensors
        topk_idx_empty = torch.empty(0, device=device, dtype=torch.int32)
        topk_weights_empty = torch.empty(0, device=device, dtype=torch.float32)
        

        # Debug information (only for rank 0)
        # if tp_rank == 0:
        #     print(f"DeepEP Low Latency setup:")
        #     print(f"  num_experts: {num_experts}")
        #     print(f"  ep_size: {ep_size}")
        #     print(f"  num_rank: {num_rank}")
        #     print(f"  num_local_experts: {num_local_experts}")
        #     print(f"  num_max_dispatch_tokens_per_rank: {num_max_dispatch_tokens_per_rank}")
        #     print(f"  hidden_states shape: {hidden_states.shape}")
        #     print(f"  scale_tensor shape: {scale_tensor.shape}")
        #     print(f"  masked_m shape: {masked_m.shape}")
        #     print(f"  masked_m values: {masked_m}")
        #     print(f"  masked_m sum: {masked_m.sum()}")
        #     print(f"  expected_m: {expected_m}")
        #     print(f"  Token distribution: {[f'Expert {i}: {masked_m[i]} tokens' for i in range(min(5, num_local_experts))]}")
        #     if num_local_experts > 5:
        #         print(f"  ... and {num_local_experts - 5} more experts")
        #     print(f"  CUDA Graph will be used for benchmarking")
        
        torch.get_device_module(device).synchronize()
        torch.cuda.empty_cache()
        
        # Warmup iterations to capture CUDA Graph
        # Need to recreate dispatch_output_list for each iteration because moe_impl disposes tensors
        for _ in range(bench_args.num_warmup):
            dispatch_output_list = []
            for masked_m in masked_m_list:
                hidden_states_fp8_tensor_copy = hidden_states_fp8_tensor.clone()
                scale_tensor_copy = scale_tensor.clone()
                
                output = DeepEPLLOutput(
                    hidden_states_fp8=(hidden_states_fp8_tensor_copy, scale_tensor_copy),
                    topk_idx=topk_idx_empty,
                    topk_weights=topk_weights_empty,
                    masked_m=masked_m,
                    expected_m=int(max(masked_m))
                )
                dispatch_output_list.append(output)
            
            for dispatch_output in dispatch_output_list:
                _ = moe_layer.experts.moe_impl(dispatch_output)
        
        torch.get_device_module(device).synchronize()
        torch.cuda.empty_cache()
        
        # Create fresh dispatch_output_list for CUDA Graph capture
        dispatch_output_list = []
        for masked_m in masked_m_list:
            hidden_states_fp8_tensor_copy = hidden_states_fp8_tensor.clone()
            scale_tensor_copy = scale_tensor.clone()
            
            output = DeepEPLLOutput(
                hidden_states_fp8=(hidden_states_fp8_tensor_copy, scale_tensor_copy),
                topk_idx=topk_idx_empty,
                topk_weights=topk_weights_empty,
                masked_m=masked_m,
                expected_m=int(max(masked_m))
            )
            dispatch_output_list.append(output)
        
        graph = torch.cuda.CUDAGraph()
        
        with torch.cuda.graph(graph):
            for dispatch_output in dispatch_output_list:
                _ = moe_layer.experts.moe_impl(dispatch_output)
        
        graph.replay()
            
        torch.get_device_module(device).synchronize()

        profiler = torch.profiler.profile(
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
        )
        profiler.start()
        
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

        gemm_latencies.append(latency_ms/bench_args.num_iterations/len(masked_m_list))
       
        profiler.stop() 
        
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
            num_token=num_token,
            phase="decode_gemm_test",
            avg_latency_ms=avg_latency_ms,
            min_latency_ms=min_latency_ms,
            max_latency_ms=max_latency_ms,
            std_latency_ms=std_latency_ms,
            num_iterations=bench_args.num_iterations,
            num_experts=num_experts,
            cuda_graph_used=True
        )
        results.append(decode_result)
        del hidden_states, hidden_states_fp8_tensor, scale_tensor, dispatch_output_list
        torch.cuda.empty_cache()

    return results


def write_results_to_file(all_results, bench_args, server_args, model_config, rank_print, tp_rank, distributed, power_law_alpha):
    """Write benchmark results to separate prefill and decode files"""
    if tp_rank != 0:
        return
    
    try:
        prefill_results = [r for r in all_results if "prefill" in r.phase]
        decode_results = [r for r in all_results if "decode" in r.phase]
        
        output_path = "/lustre/raplab/client/xutingz/workspace/gitsrc/aiconfigurator/src/aiconfigurator/systems/data/h200_sxm/sglang/0.5.0/"
        context_output_path = os.path.join(output_path, "context_moe_perf.txt")
        generation_output_path = os.path.join(output_path, "generation_moe_perf.txt")
        
        os.makedirs(output_path, exist_ok=True)
        
        if prefill_results:
            file_exists = os.path.exists(context_output_path)
            
            with open(context_output_path, 'a' if file_exists else 'w') as f:
                # Write header only if file doesn't exist
                if not file_exists:
                    f.write("framework,version,device,op_name,kernel_source,moe_dtype,num_tokens,hidden_size,inter_size,topk,num_experts,moe_tp_size,moe_ep_size,distribution,latency\n")
                
                for result in prefill_results:
                    # Use configuration values directly
                    num_experts = result.num_experts
                    hidden_size = 7168
                    inter_size = 2048
                    top_k = 8  # Default top_k value
                    op_name = "moe_context"
                    kernel_source = "deepepmoe"
                    moe_dtype = "fp8"
                    device_name = "NVIDIA H20-3e"  # Default device name
                    moe_tp_size = 1
                    moe_ep_size = server_args.ep_size if num_experts==256 else int(server_args.ep_size*256//num_experts)
                    num_tokens = result.num_token * moe_ep_size
                    num_experts = 256

                    f.write(f"SGLang,1.0.0,{device_name},{op_name},{kernel_source},{moe_dtype},{num_tokens},{hidden_size},{inter_size},{top_k},{num_experts},{moe_tp_size},{moe_ep_size},{"uniform"},{result.avg_latency_ms}\n")
            
            if file_exists:
                rank_print(f"\nPrefill results appended to {context_output_path}")
            else:
                rank_print(f"\nPrefill results saved to {context_output_path}")
        
        # Write decode results to generation_moe_perf.txt
        if decode_results:
            file_exists = os.path.exists(generation_output_path)
            
            with open(generation_output_path, 'a' if file_exists else 'w') as f:
                if not file_exists:
                    f.write("framework,version,device,op_name,kernel_source,moe_dtype,num_tokens,hidden_size,inter_size,topk,num_experts,moe_tp_size,moe_ep_size,distribution,latency\n")
                
                for result in decode_results:
                    num_experts = result.num_experts
                    hidden_size = 7168
                    inter_size = 2048
                    top_k = 8  
                    op_name = "moe_generation"
                    kernel_source = "deepepmoe"
                    moe_dtype = "fp8"
                    device_name = "NVIDIA H20-3e"  
                    moe_tp_size = 1
                    moe_ep_size = server_args.ep_size if num_experts==256 else int(server_args.ep_size*256//num_experts)
                    num_tokens = result.num_token * moe_ep_size
                    num_experts = 256
                    
                    f.write(f"SGLang,1.0.0,{device_name},{op_name},{kernel_source},{moe_dtype},{num_tokens},{hidden_size},{inter_size},{top_k},{num_experts},{moe_tp_size},{moe_ep_size},{"power_law_" + str(power_law_alpha) if distributed == "power_law" else distributed},{result.avg_latency_ms}\n")
            
            if file_exists:
                rank_print(f"\nDecode results appended to {generation_output_path}")
            else:
                rank_print(f"\nDecode results saved to {generation_output_path}")

        
    except Exception as e:
        rank_print(f"Error writing results to file: {e}")


def run_moe_benchmark(
    server_args,
    port_args,
    bench_args: MoEBenchArgs,
    tp_rank: int,
    distributed = "power_law", 
    power_law_alpha = 0.8,
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

    # Get the expert number from bench_args
    num_expert = bench_args.num_experts
    
    try:
        rank_print(f"\n{'='*50}")
        rank_print(f"Testing with {num_expert} experts")
        rank_print(f"{'='*50}")
        
        # Override model config to set the expert number
        original_json_override = server_args.json_model_override_args
        server_args.json_model_override_args = json.dumps({
            "num_hidden_layers": 4,
            "n_routed_experts": num_expert
        })
        
        # Load model with the specific expert number
        model_runner = load_model_with_dummy_weights(server_args, port_args, tp_rank)
        reqs = prepare_synthetic_inputs(128, 16)  # Fixed output_len=16
        
        batch = ScheduleBatch.init_new(
            reqs=reqs,
            req_to_token_pool=model_runner.req_to_token_pool,
            token_to_kv_pool_allocator=model_runner.token_to_kv_pool_allocator,
            tree_cache=None,
            model_config=model_runner.model_config,
            enable_overlap=False,
            spec_algorithm=SpeculativeAlgorithm.NONE,
            enable_custom_logit_processor=False,  
        )
        batch.prepare_for_extend()
        _maybe_prepare_mlp_sync_batch(batch, model_runner)
        
        # Get the MoE layer for this expert number
        moe_layer = model_runner.model.model.layers[bench_args.test_layer].mlp
        actual_num_experts = moe_layer.config.n_routed_experts  # Actual number of experts loaded
        
        rank_print(f"Loaded model with {actual_num_experts} experts")
        
        # Restore original json_override
        server_args.json_model_override_args = original_json_override

        # Calculate prefill test cases
        ep_size = server_args.ep_size
        num_local_experts = actual_num_experts // ep_size
        num_rank = ep_size if actual_num_experts == 256 else int(256//actual_num_experts * ep_size)
        prefill_test_cases = get_moe_prefill_test_cases(ep_size, num_rank)
        rank_print(f"Testing {len(prefill_test_cases)} prefill configurations...")

        results = benchmark_moe_layer_prefill(
            model_runner,
            server_args,
            port_args,
            bench_args,
            rank_print,
            server_args.device,
            tp_rank,
            prefill_test_cases,
            moe_layer,
            actual_num_experts,
            ep_size,
            num_rank,
        )
        all_results.extend(results)

        # Calculate decode test cases
        decode_test_cases = get_moe_decode_test_cases()
        rank_print(f"Testing {len(decode_test_cases)} decode configurations...")

        results = benchmark_moe_layer_decode(
            model_runner,
            server_args,
            port_args,
            bench_args,
            rank_print,
            server_args.device,
            tp_rank,
            decode_test_cases,
            moe_layer,
            actual_num_experts,
            ep_size,
            num_rank,
            distributed, 
            power_law_alpha,
        )
        all_results.extend(results)
        
        # Save model_config before cleaning up model_runner
        model_config = model_runner.model_config
        
        # Clean up model_runner
        del model_runner, moe_layer, batch, reqs
        torch.cuda.empty_cache()

    except Exception as e:
        rank_print(f"Error during MoE benchmark: {e}")
        import traceback
        rank_print(f"Traceback: {traceback.format_exc()}")
        return
    
    # Write results to separate prefill and decode files
    write_results_to_file(all_results, bench_args, server_args, model_config, rank_print, tp_rank, distributed, power_law_alpha)
    torch.cuda.empty_cache()
    
    rank_print(f"\n{'='*60}")
    rank_print("BENCHMARK COMPLETED SUCCESSFULLY")
    rank_print(f"{'='*60}")


def main(server_args, bench_args: MoEBenchArgs):
    """Main function"""

    _set_envs_and_config(server_args)

    port_args = PortArgs.init_new(server_args)

    workers = []
    for tp_rank in range(server_args.tp_size):
        proc = multiprocessing.Process(
            target=run_moe_benchmark,
            args=(
                server_args,
                port_args,
                bench_args,
                tp_rank,
                "uniform", 
                0.8,
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

DEEPSEEK_MODEL_PATH = os.environ.get("DEEPSEEK_MODEL_PATH", "/lustre/raplab/client/xutingz/workspace/model/DeepSeek-V3/")

if __name__ == "__main__":
    # Fixed configuration values
    model_path = DEEPSEEK_MODEL_PATH
    output_path = "/lustre/raplab/client/xutingz/workspace/gitsrc/aiconfigurator/src/aiconfigurator/systems/data/h200_sxm/sglang/0.5.0/"
    
    # Create ServerArgs manually with fixed values
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
    
    # Create BenchArgs with fixed values
    bench_args = MoEBenchArgs(
        run_name="deepep_moe_test",
        num_warmup=3,
        num_iterations=10,
        test_layer=3,
        num_experts=128,
    )

    logging.basicConfig(
        level=getattr(logging, server_args.log_level.upper()),
        format="%(message)s",
    )

    main(server_args, bench_args)

    print("\n" + "="*60)
    print("SCRIPT COMPLETED SUCCESSFULLY")
    print("="*60)
   