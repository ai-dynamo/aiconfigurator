# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import torch
import torch.nn as nn
import time
import math
import json
import argparse
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import concurrent.futures
import fcntl

# Add sglang to path
sys.path.insert(0, "/sgl-workspace/sglang/python")

# Add project root to path to import helper module
sys.path.insert(0, "/root/fac/llm-pet")

from sglang.srt.models.deepseek_v2 import DeepseekV2MLP
from sglang.srt.distributed.parallel_state import destroy_model_parallel
from sglang.srt.layers.quantization import (
    Fp8Config
)
from sglang.srt.distributed import (
    initialize_model_parallel,
    init_distributed_environment,
)


# Default model path
DEEPSEEK_MODEL_PATH = "/root/fac/deepseek-v3"

@dataclass
class MLPBenchResult:
    """Result for a single MLP benchmark run"""
    quant_type: str
    num_token: int
    hidden_size: int
    intermediate_size: int
    avg_time_ms: float
    min_time_ms: float
    max_time_ms: float
    std_time_ms: float
    num_iterations: int
    device: str
    kernel_source: str

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


def log_perf(item_list: list[dict], 
             framework: str, 
             version: str, 
             device_name: str, 
             op_name: str,
             kernel_source: str,
             perf_filename: str):
    
    # Use standardized device name
    device_name = get_gpu_device_name()
    
    # Fixed header matching the reference format
    header = 'framework,version,device,op_name,kernel_source,quant_type,num_token,hidden_size,intermediate_size,avg_ms'
    
    with open(perf_filename, 'a') as f:
        fcntl.flock(f, fcntl.LOCK_EX)

        if f.tell() == 0:
            f.write(header + '\n')

        for item in item_list:
            # Extract values in the exact order of the reference format
            quant_type = item.get('quant_type', '')
            num_token = item.get('num_token', '')
            hidden_size = item.get('hidden_size', '')
            intermediate_size = item.get('intermediate_size', '')
            avg_ms = item.get('avg_ms', '')
            
            # Write line in the exact format as reference files
            line = f'{framework},{version},{device_name},{op_name},{kernel_source},{quant_type},{num_token},{hidden_size},{intermediate_size},{avg_ms}'
            f.write(line + '\n')

def export_profiler_data(profiler, quant_type: str, num_token: int, hidden_size: int, intermediate_size: int):
    """Export profiler data in multiple formats"""
    # Create output directory
    os.makedirs("profiler_outputs", exist_ok=True)
    
    # Generate filename prefix
    filename_prefix = f"mlp_profile_{quant_type}_tokens{num_token}_hidden{hidden_size}_intermediate{intermediate_size}"
    
    # Export as JSON (Chrome trace format)
    json_path = f"profiler_outputs/{filename_prefix}.json"
    try:
        profiler.export_chrome_trace(json_path)
        print(f"Profiler data exported to {json_path}")
    except RuntimeError as e:
        if "Trace is already saved" in str(e):
            print(f"Profiler trace already saved to TensorBoard logs")
        else:
            print(f"Warning: Could not export Chrome trace: {e}")
    
    # Export detailed statistics
    stats_path = f"profiler_outputs/{filename_prefix}_stats.txt"
    try:
        with open(stats_path, 'w') as f:
            f.write(f"MLP Profiler Statistics\n")
            f.write(f"Quantization Type: {quant_type}\n")
            f.write(f"Number of Tokens: {num_token}\n")
            f.write(f"Hidden Size: {hidden_size}\n")
            f.write(f"Intermediate Size: {intermediate_size}\n")
            f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Write profiler table
            f.write("Profiler Table (sorted by CUDA time):\n")
            f.write(str(profiler.key_averages().table(sort_by="cuda_time_total", row_limit=50)))
            f.write("\n\n")
            
            # Write memory statistics
            f.write("Memory Statistics:\n")
            f.write(str(profiler.key_averages().table(sort_by="cuda_memory_usage", row_limit=50)))
            f.write("\n\n")
            
            # Write CPU time statistics
            f.write("CPU Time Statistics:\n")
            f.write(str(profiler.key_averages().table(sort_by="cpu_time_total", row_limit=50)))
        
        print(f"Profiler statistics exported to {stats_path}")
    except Exception as e:
        print(f"Warning: Could not export profiler statistics: {e}")

# Removed save_results_to_json function - only using txt output format

def cleanup_distributed():
    """Clean up distributed environment if it exists"""
    try:
        destroy_model_parallel()
        print("Cleaned up existing distributed environment")
    except Exception as e:
        print(f"Warning: Could not clean up distributed environment: {e}")
    
    # Also clean up torch.distributed if it exists
    try:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
            print("Cleaned up torch.distributed process group")
    except Exception as e:
        print(f"Warning: Could not clean up torch.distributed: {e}")

def get_mlp_test_cases():
    """Get test cases for MLP benchmarking"""
    # Test different batch sizes and sequence lengths
    num_tokens = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]
    
    # Test different quantization types
    quant_types = ['fp8']  # Start with float16 for basic testing
    
    # DeepSeek V2/V3 model parameters
    hidden_size = 7168
    intermediate_size = 2048  # 4x expansion ratio
    
    test_cases = []
    for quant_type in quant_types:
        for num_token in num_tokens:
                # Skip some combinations to reduce test time
                test_cases.append({
                    'quant_type': quant_type,
                    'num_token': num_token,
                    'hidden_size': hidden_size,
                    'intermediate_size': intermediate_size
                })
    
    return test_cases

def run_mlp_benchmark(
    quant_type: str,
    num_token: int,
    hidden_size: int,
    intermediate_size: int,
    device: str = 'cuda:0',
    num_warmup: int = 3,
    num_iterations: int = 10,
    enable_profile: bool = False
):
    """Run MLP benchmark with given parameters"""
    torch.cuda.set_device(device)
    
    # Clean up any existing distributed environment
    cleanup_distributed()
    
    # Initialize distributed environment for single GPU
    dist_init_method = f"tcp://127.0.0.1:29500"
    init_distributed_environment(
        backend="nccl",
        world_size=1,
        rank=0,
        local_rank=0,
        distributed_init_method=dist_init_method,
        timeout=10,
    )
    
    # Initialize model parallel groups (single GPU = no parallelism)
    initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=1,
        duplicate_tp_group=False,
        backend="nccl",
    )
    
    # Create quantization config
    quant_config = Fp8Config(is_checkpoint_fp8_serialized=True,
        activation_scheme="dynamic",
        ignored_layers=None,
        weight_block_size=[128, 128])
    # quant_config = None

    # Create MLP module
    mlp = DeepseekV2MLP(
        hidden_size=hidden_size,
        intermediate_size=2048,
        hidden_act="silu",
        quant_config=quant_config,
        reduce_results=True,
        prefix="",
        tp_rank=0,  # Set tp_rank for single GPU
        tp_size=1,  # Set tp_size for single GPU
    ).to(device)
    
    # Create input tensor
    input_tensor = torch.randn(
        (num_token, hidden_size), 
        dtype=torch.bfloat16, 
        device=device
    )
    
    # Warmup runs
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = mlp(input_tensor)
    
    torch.cuda.synchronize()
    
    # Benchmark runs
    times = []
    profiler = None
    
    if enable_profile:
        # Create profiler for detailed analysis
        profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./profiler_logs'),
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        )
        profiler.start()
    
    with torch.no_grad():
        for i in range(num_iterations):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            start_event.record()
            output = mlp(input_tensor)
            end_event.record()
            
            torch.cuda.synchronize()
            elapsed_time = start_event.elapsed_time(end_event)
            times.append(elapsed_time)
            
            if enable_profile and profiler:
                profiler.step()
    
    if enable_profile and profiler:
        profiler.stop()
        # Export profiler data
        export_profiler_data(profiler, quant_type, num_token, hidden_size, intermediate_size)
    
    # Calculate statistics
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    std_time = math.sqrt(sum((t - avg_time) ** 2 for t in times) / len(times))
    
    # No FLOPs calculations needed for the output format
    
    print(f"MLP Benchmark - {quant_type}: "
          f"num_token={num_token}, "
          f"avg_time={avg_time:.2f}ms")
    
    # Return MLPBenchResult object
    return MLPBenchResult(
        quant_type=quant_type,
        num_token=num_token,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        avg_time_ms=avg_time,
        min_time_ms=min_time,
        max_time_ms=max_time,
        std_time_ms=std_time,
        num_iterations=num_iterations,
        device=torch.cuda.get_device_name(device),
        kernel_source='deepseek_v2'
    )

def run_mlp_benchmark_cuda_graph(
    quant_type: str,
    num_token: int,
    hidden_size: int,
    intermediate_size: int,
    device: str = 'cuda:0',
    num_warmup: int = 3,
    num_iterations: int = 10,
    enable_profile: bool = False
):
    """Run MLP benchmark using CUDA graph for more accurate timing"""
    torch.cuda.set_device(device)
    
    # Clean up any existing distributed environment
    cleanup_distributed()
    
    # Create quantization config
    quant_config = Fp8Config(is_checkpoint_fp8_serialized=True,
        activation_scheme="dynamic",
        ignored_layers=None,
        weight_block_size=[128, 128])
    
    # quant_config = None
    dist_init_method = f"tcp://127.0.0.1:29500"
    # Initialize model
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
        backend="nccl",
        duplicate_tp_group=False,
    )
    
    # Create multiple MLP modules to avoid L2 cache effects
    
    mlp = DeepseekV2MLP(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        hidden_act="silu",
        quant_config=quant_config,
        reduce_results=True,
        prefix=f"mlp",
        tp_rank=0,
        tp_size=1,
    ).to(device)
    
    
    # Create input tensor
    input_tensor = torch.randn(
        (num_token, hidden_size), 
        dtype=torch.bfloat16, 
        device=device
    )
    
    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
          
            _ = mlp(input_tensor)
    
    torch.cuda.synchronize()
    
    # Create CUDA graph
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
   
        _ = mlp(input_tensor)
    
    # Warmup with graph
    for _ in range(num_warmup):
        g.replay()
    
    # Setup profiler if enabled
    profiler = None
    if enable_profile:
        profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./profiler_logs'),
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        )
        profiler.start()
    
    # Benchmark with graph
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for i in range(num_iterations):
        g.replay()
        if enable_profile and profiler:
            profiler.step()
    end_event.record()
    
    torch.cuda.synchronize()
    total_time = start_event.elapsed_time(end_event)
    avg_time = total_time / num_iterations
    
    if enable_profile and profiler:
        profiler.stop()
        # Export profiler data
        export_profiler_data(profiler, quant_type, num_token, hidden_size, intermediate_size) 

    
    print(f"MLP Benchmark (CUDA Graph) - {quant_type}: "
          f"num_token={num_token}, "
          f"avg_time={avg_time:.2f}ms")
    
    # Return MLPBenchResult object
    return MLPBenchResult(
        quant_type=quant_type,
        num_token=num_token,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        avg_time_ms=avg_time,
        min_time_ms=avg_time,  # CUDA graph gives consistent timing
        max_time_ms=avg_time,
        std_time_ms=0.0,
        num_iterations=num_iterations,
        device=torch.cuda.get_device_name(device),
        kernel_source='deepseek_v2_cuda_graph'
    )

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='SGLang MLP Benchmark with Profiling Support')
    parser.add_argument('--enable-profile', action='store_true', 
                       help='Enable PyTorch profiler to export detailed performance analysis')
    parser.add_argument('--cuda-graph-only', action='store_true',
                       help='Run only CUDA graph benchmarks (skip regular benchmarks)')
    parser.add_argument('--regular-only', action='store_true',
                       help='Run only regular benchmarks (skip CUDA graph benchmarks)')
    parser.add_argument('--num-iterations', type=int, default=10,
                       help='Number of benchmark iterations (default: 10)')
    parser.add_argument('--num-warmup', type=int, default=3,
                       help='Number of warmup iterations (default: 3)')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='CUDA device to use (default: cuda:0)')
    parser.add_argument('--output', type=str, default="/home/scratch.aichenf_wwfo/aiconfigurator/src/aiconfigurator/systems/data/h200_sxm/sglang/0.5.0/",
                       help='Output directory for results ')
    
    return parser.parse_args()

def main():
    """Main function to run MLP benchmarks"""
    args = parse_args()
    
    print("Starting SGLang MLP Benchmark")
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"Profiling enabled: {args.enable_profile}")
    print(f"Number of iterations: {args.num_iterations}")
    print(f"Number of warmup: {args.num_warmup}")
    
    # Clean up any existing distributed environment at the start
    cleanup_distributed()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    print(f"Output directory: {args.output}")
    
    # Get test cases
    test_cases = get_mlp_test_cases()
    print(f"Total test cases: {len(test_cases)}")
    
    # Run CUDA graph benchmarks (generation)
    generation_results = []
    if not args.regular_only:
        print("\n=== Running CUDA Graph Benchmarks (Generation) ===")
        for test_case in test_cases:
            # Use CUDA graph version for more accurate timing
            result = run_mlp_benchmark_cuda_graph(
                **test_case,
                device=args.device,
                num_warmup=args.num_warmup,
                num_iterations=args.num_iterations,
                enable_profile=args.enable_profile
            )
            generation_results.append(result)
    
    # Run regular benchmarks (prefill)
    prefill_results = []
    if not args.cuda_graph_only:
        print("\n=== Running Regular Benchmarks (Prefill) ===")
        for test_case in test_cases:
            result = run_mlp_benchmark(
                **test_case,
                device=args.device,
                num_warmup=args.num_warmup,
                num_iterations=args.num_iterations,
                enable_profile=args.enable_profile
            )
            prefill_results.append(result)
    
    # Write results to files after all tests are completed
    if generation_results:
        generation_filename = os.path.join(args.output, "generation_mlp_perf.txt")
        # Convert results to the format expected by log_perf
        generation_items = []
        for result in generation_results:
            generation_items.append({
                'quant_type': result.quant_type,
                'num_token': result.num_token,
                'hidden_size': result.hidden_size,
                'intermediate_size': result.intermediate_size,
                'avg_ms': result.avg_time_ms
            })
        
        log_perf(
            item_list=generation_items,
            framework='SGLang',
            version='1.0.0',
            device_name=get_gpu_device_name(),
            op_name='mlp',
            kernel_source='deepseek_v2_cuda_graph',
            perf_filename=generation_filename
        )
        print(f"Generation results saved to: {generation_filename}")
    
    if prefill_results:
        prefill_filename = os.path.join(args.output, "prefill_mlp_perf.txt")
        # Convert results to the format expected by log_perf
        prefill_items = []
        for result in prefill_results:
            prefill_items.append({
                'quant_type': result.quant_type,
                'num_token': result.num_token,
                'hidden_size': result.hidden_size,
                'intermediate_size': result.intermediate_size,
                'avg_ms': result.avg_time_ms
            })
        
        log_perf(
            item_list=prefill_items,
            framework='SGLang',
            version='1.0.0',
            device_name=get_gpu_device_name(),
            op_name='mlp',
            kernel_source='deepseek_v2',
            perf_filename=prefill_filename
        )
        print(f"Prefill results saved to: {prefill_filename}")
    
    if args.enable_profile:
        print(f"\nProfiler data exported to:")
        print(f"  - Chrome trace files: profiler_outputs/*.json")
        print(f"  - Statistics files: profiler_outputs/*_stats.txt")
        print(f"  - TensorBoard logs: profiler_logs/")
        print(f"\nTo view profiler data:")
        print(f"  1. Chrome trace: Open profiler_outputs/*.json in Chrome (chrome://tracing)")
        print(f"  2. TensorBoard: tensorboard --logdir=profiler_logs")
    
    print("\n" + "="*50)
    print("MLP BENCHMARK COMPLETED")
    print("="*50)
    print(f"Output files saved to:")
    print(f"  - Prefill results: {os.path.join(args.output, 'prefill_mlp_perf.txt')}")
    print(f"  - Generation results: {os.path.join(args.output, 'generation_mlp_perf.txt')}")
    print("="*50)

if __name__ == "__main__":
    main() 