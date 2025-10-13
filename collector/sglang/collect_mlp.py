# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import os
import torch
import math
from typing import List
try:
    from helper import log_perf
except ModuleNotFoundError:
    import os, sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from helper import log_perf
import pkg_resources

from sglang.srt.models.deepseek_v2 import DeepseekV2MLP
from sglang.srt.distributed.parallel_state import destroy_model_parallel
from sglang.srt.layers.quantization import Fp8Config
from sglang.srt.distributed import (
    initialize_model_parallel,
    init_distributed_environment,
)

DEEPSEEK_MODEL_PATH = os.environ.get("DEEPSEEK_MODEL_PATH", "/deepseek-v3")


def get_mlp_prefill_test_cases():
    """Get test cases for MLP prefill phase
    Returns: list of [quant_type, num_token, hidden_size, intermediate_size]
    """
    test_cases = []

    num_tokens = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
    quant_types = ['fp8_block']
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

    num_tokens = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
    quant_types = ['fp8_block']
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
    for var_name in ['_TP', '_PP', '_MOE_EP', '_MOE_TP', '_WORLD', '_PDMUX_PREFILL_TP_GROUP']:
        if hasattr(parallel_state, var_name):
            setattr(parallel_state, var_name, None)

    import sglang.srt.eplb.expert_location as expert_location
    if hasattr(expert_location, '_global_expert_location_metadata'):
        expert_location._global_expert_location_metadata = None
    
    try:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
    except Exception as e:
        print(f"Warning: Could not clean up torch.distributed: {e}")

def initialize_distributed():
    """Initialize distributed environment for MLP benchmarking"""
    dist_init_method = f"tcp://127.0.0.1:29500"
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

def run_mlp_torch(quant_type, num_token, hidden_size, intermediate_size, is_context, num_warmup, num_iterations, device, output_path):
    """Run MLP benchmark for both context and generation phases"""
    
    torch.cuda.set_device(device)
    phase = "Context" if is_context else "Decode"
    print(f"\n{phase}: quant_type={quant_type}, num_token={num_token}")
    
    try:
        quant_config = Fp8Config(
            is_checkpoint_fp8_serialized=True,
            activation_scheme="dynamic",
            ignored_layers=None,
            weight_block_size=[128, 128]
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

        input_tensor = torch.randn(
            (num_token, hidden_size), 
            dtype=torch.bfloat16, 
            device=device
        )
        
        if is_context:
            # Context phase: no CUDA graph
            with torch.no_grad():
                for _ in range(num_warmup):
                    _ = mlp(input_tensor)
            
            torch.cuda.synchronize()
            
            times = []
            with torch.no_grad():
                for i in range(num_iterations):
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    
                    start_event.record()
                    _ = mlp(input_tensor)
                    end_event.record()
                    
                    torch.cuda.synchronize()
                    times.append(start_event.elapsed_time(end_event))
            
            avg_time = sum(times) / len(times)
            perf_filename = os.path.join(output_path, "context_mlp_perf.txt")
            kernel_source = 'deepseek_v3'
            
            print(f"  {phase} MLP time: {avg_time:.3f} ms "
                    f"(min: {min(times):.3f}, max: {max(times):.3f}, std: {math.sqrt(sum((t - avg_time) ** 2 for t in times) / len(times)):.3f})")
        else:
            # Decode phase: use CUDA graph
            torch.cuda.synchronize()
            
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                _ = mlp(input_tensor)
            
            for _ in range(num_warmup):
                g.replay()
            
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            start_event.record()
            for _ in range(num_iterations):
                g.replay()
            end_event.record()
            
            torch.cuda.synchronize()
            avg_time = start_event.elapsed_time(end_event) / num_iterations
            perf_filename = os.path.join(output_path, "generation_mlp_perf.txt")
            kernel_source = 'deepseek_v3_cuda_graph'
            
            print(f"  {phase} MLP time: {avg_time:.3f} ms")
        
        # Save via log_perf
        try:
            os.makedirs(os.path.dirname(perf_filename), exist_ok=True)
            device_name = torch.cuda.get_device_name(device)
            version = pkg_resources.get_distribution('sglang').version
            log_perf(
                item_list=[{
                    'quant_type': quant_type,
                    'num_token': num_token,
                    'hidden_size': hidden_size,
                    'intermediate_size': intermediate_size,
                    'avg_ms': avg_time
                }],
                framework='SGLang',
                version=version,
                device_name=device_name,
                op_name='mlp',
                kernel_source=kernel_source,
                perf_filename=perf_filename
            )
        except Exception as e:
            print(f"  Warning: failed to log MLP metrics: {e}")
        
        del mlp, input_tensor
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"  {phase} test failed: {str(e)}")
        print(f"  Skipping this configuration...")

if __name__ == "__main__":
    output_path = "/aiconfigurator/src/aiconfigurator/systems/data/h100_sxm/sglang/0.5.0/"
    model_path = DEEPSEEK_MODEL_PATH
    hidden_size = 7168
    intermediate_size = 2048
    num_warmup = 3
    num_iterations = 10
    dtype = "auto"
    device = "cuda:0"
    
    cleanup_distributed()
    
    print(f"Starting SGLang MLP Benchmark")
    print(f"Model path: {model_path}")
    print(f"Device: {torch.cuda.get_device_name()}")
    
    prefill_test_cases = get_mlp_prefill_test_cases()
    decode_test_cases = get_mlp_decode_test_cases()
    print(f"Running {len(prefill_test_cases)} prefill test cases and {len(decode_test_cases)} decode test cases...")
    
    # Process prefill test cases
    print(f"\n{'='*60}")
    print(f"TESTING PREFILL")
    print(f"Test cases: {len(prefill_test_cases)}")
    print(f"{'='*60}")
    cleanup_distributed()
    
    torch.cuda.empty_cache()
    initialize_distributed()
    
    for test_case in prefill_test_cases:
        quant_type, num_token, hs, inter_s = test_case
        run_mlp_torch(quant_type, num_token, hs, inter_s, True, num_warmup, num_iterations, device, output_path)

    # Process decode test cases
    print(f"\n{'='*60}")
    print(f"TESTING DECODE")
    print(f"Test cases: {len(decode_test_cases)}")
    print(f"{'='*60}")
    
    torch.cuda.empty_cache()
    
    for test_case in decode_test_cases:
        quant_type, num_token, hs, inter_s = test_case
        run_mlp_torch(quant_type, num_token, hs, inter_s, False, num_warmup, num_iterations, device, output_path)

    cleanup_distributed()
    torch.cuda.empty_cache()
    
    print("\n" + "="*50)
    print("MLP BENCHMARK COMPLETED")
    print("="*50)
    print("Output files saved to:")
    print(f"  - Context results: {os.path.join(output_path, 'context_mlp_perf.txt')}")
    print(f"  - Generation results: {os.path.join(output_path, 'generation_mlp_perf.txt')}")
    print("="*50) 