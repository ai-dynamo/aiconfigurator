# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import os
import torch
import math
from dataclasses import dataclass, field
from typing import List
import fcntl

from sglang.srt.models.deepseek_v2 import DeepseekV2MLP
from sglang.srt.distributed.parallel_state import destroy_model_parallel
from sglang.srt.layers.quantization import Fp8Config
from sglang.srt.distributed import (
    initialize_model_parallel,
    init_distributed_environment,
)

DEEPSEEK_MODEL_PATH = os.environ.get("DEEPSEEK_MODEL_PATH", "/deepseek-v3")

@dataclass
class BenchConfig:
    # MLP parameters
    quant_types: List[str] = field(default_factory=lambda: ['fp8_block'])
    num_tokens: List[int] = field(default_factory=lambda: [])
    
    # Model parameters
    hidden_size: int = 7168
    intermediate_size: int = 2048
    
    # Common parameters
    num_warmup: int = 3
    num_iterations: int = 10
    model_path: str = DEEPSEEK_MODEL_PATH
    dtype: str = "auto"
    device: str = "cuda:0"
    enable_profiler: bool = False

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


def get_mlp_test_cases():
    """Get test cases for MLP benchmarking"""
    test_cases = []

    num_tokens = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
    quant_types = ['fp8_block']
    hidden_size = 7168
    intermediate_size = 2048  
    
    for quant_type in quant_types:
        for num_token in num_tokens:
            test_cases.append({
                'quant_type': quant_type,
                'num_token': num_token,
                'hidden_size': hidden_size,
                'intermediate_size': intermediate_size
            })
    
    return test_cases


def output_results(results: List[MLPBenchResult], output_path: str):
    """Save results to separate files for prefill and decode phases"""
    prefill_results = [r for r in results if r.kernel_source == 'deepseek_v2']
    decode_results = [r for r in results if r.kernel_source == 'deepseek_v2_cuda_graph']
    
    if prefill_results:
        prefill_filename = os.path.join(output_path, "context_mlp_perf.txt")
        prefill_items = []
        for result in prefill_results:
            prefill_items.append({
                'quant_type': result.quant_type,
                'num_token': result.num_token,
                'hidden_size': result.hidden_size,
                'intermediate_size': result.intermediate_size,
                'avg_ms': result.avg_time_ms
            })

        device_name = torch.cuda.get_device_name('cuda:0')
        save_results_to_file(
            item_list=prefill_items,
            framework='SGLang',
            version='0.5.0',
            device_name=device_name,
            op_name='mlp',
            kernel_source='deepseek_v3',
            perf_filename=prefill_filename
        )
        print(f"Prefill results saved to: {prefill_filename}")
    
    if decode_results:
        decode_filename = os.path.join(output_path, "generation_mlp_perf.txt")
        decode_items = []
        for result in decode_results:
            decode_items.append({
                'quant_type': result.quant_type,
                'num_token': result.num_token,
                'hidden_size': result.hidden_size,
                'intermediate_size': result.intermediate_size,
                'avg_ms': result.avg_time_ms
            })

        device_name = torch.cuda.get_device_name('cuda:0')
        save_results_to_file(
            item_list=decode_items,
            framework='SGLang',
            version='0.5.0',
            device_name=device_name,
            op_name='mlp',
            kernel_source='deepseek_v3_cuda_graph',
            perf_filename=decode_filename
        )
        print(f"Decode results saved to: {decode_filename}")


def save_results_to_file(item_list: list[dict], 
             framework: str, 
             version: str, 
             device_name: str, 
             op_name: str,
             kernel_source: str,
             perf_filename: str):
    
    header = 'framework,version,device,op_name,kernel_source,quant_type,num_token,hidden_size,intermediate_size,avg_ms'
    
    with open(perf_filename, 'a') as f:
        fcntl.flock(f, fcntl.LOCK_EX)

        if f.tell() == 0:
            f.write(header + '\n')

        for item in item_list:
            quant_type = item.get('quant_type', '')
            num_token = item.get('num_token', '')
            hidden_size = item.get('hidden_size', '')
            intermediate_size = item.get('intermediate_size', '')
            avg_ms = item.get('avg_ms', '')
            
            line = f'{framework},{version},{device_name},{op_name},{kernel_source},{quant_type},{num_token},{hidden_size},{intermediate_size},{avg_ms}'
            f.write(line + '\n')

def cleanup_distributed():
    """Clean up distributed environment if it exists"""
    try:
        destroy_model_parallel()
    except Exception as e:
        print(f"Warning: Could not clean up distributed environment: {e}")
    
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

def run_mlp_torch(
    cases: List,
    backend_config: BenchConfig
) -> List[MLPBenchResult]:
    """Run prefill benchmark for MLP module"""
    
    results = []
    torch.cuda.set_device(backend_config.device)
    
    for test_case in cases:
        quant_type = test_case['quant_type']
        num_token = test_case['num_token']
        hidden_size = test_case['hidden_size']
        intermediate_size = test_case['intermediate_size']
        
        print(f"\nPrefill: quant_type={quant_type}, num_token={num_token}")
        
        try:
            quant_config = Fp8Config(is_checkpoint_fp8_serialized=True,
                activation_scheme="dynamic",
                ignored_layers=None,
                weight_block_size=[128, 128])

            mlp = DeepseekV2MLP(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                hidden_act="silu",
                quant_config=quant_config,
                reduce_results=True,
                prefix="",
                tp_rank=0, 
                tp_size=1, 
            ).to(backend_config.device)

            input_tensor = torch.randn(
                (num_token, hidden_size), 
                dtype=torch.bfloat16, 
                device=backend_config.device
            )
            
            with torch.no_grad():
                for _ in range(backend_config.num_warmup):
                    _ = mlp(input_tensor)
            
            torch.cuda.synchronize()
            
            times = []
            
            with torch.no_grad():
                for i in range(backend_config.num_iterations):
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    
                    start_event.record()
                    output = mlp(input_tensor)
                    end_event.record()
                    
                    torch.cuda.synchronize()
                    elapsed_time = start_event.elapsed_time(end_event)
                    times.append(elapsed_time)
            
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            std_time = math.sqrt(sum((t - avg_time) ** 2 for t in times) / len(times))
            
            print(f"  Prefill MLP time: {avg_time:.3f} ms "
                    f"(min: {min_time:.3f}, max: {max_time:.3f}, std: {std_time:.3f})")
            
            result = MLPBenchResult(
                quant_type=quant_type,
                num_token=num_token,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                avg_time_ms=avg_time,
                min_time_ms=min_time,
                max_time_ms=max_time,
                std_time_ms=std_time,
                num_iterations=backend_config.num_iterations,
                device=torch.cuda.get_device_name(backend_config.device),
                kernel_source='deepseek_v2'
            )
            results.append(result)
            
            del mlp, input_tensor
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"  Prefill test failed: {str(e)}")
            print(f"  Skipping this configuration...")
            continue
    
    return results

def run_mlp_cuda_graph(
    cases: List,
    backend_config: BenchConfig
) -> List[MLPBenchResult]:
    """Run decode benchmark for MLP module using CUDA graph"""
    
    results = []
    torch.cuda.set_device(backend_config.device)
    
    for test_case in cases:
        quant_type = test_case['quant_type']
        num_token = test_case['num_token']
        hidden_size = test_case['hidden_size']
        intermediate_size = test_case['intermediate_size']
        
        print(f"\nDecode: quant_type={quant_type}, num_token={num_token}")
        
        try:
            quant_config = Fp8Config(is_checkpoint_fp8_serialized=True,
                activation_scheme="dynamic",
                ignored_layers=None,
                weight_block_size=[128, 128])

            mlp = DeepseekV2MLP(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                hidden_act="silu",
                quant_config=quant_config,
                reduce_results=True,
                prefix=f"mlp",
                tp_rank=0,
                tp_size=1,
            ).to(backend_config.device)

            input_tensor = torch.randn(
                (num_token, hidden_size), 
                dtype=torch.bfloat16, 
                device=backend_config.device
            )

            torch.cuda.synchronize()
            
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                _ = mlp(input_tensor)
            
            for _ in range(backend_config.num_warmup):
                g.replay()
            
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            start_event.record()
            for i in range(backend_config.num_iterations):
                g.replay()
            end_event.record()
            
            torch.cuda.synchronize()
            total_time = start_event.elapsed_time(end_event)
            avg_time = total_time / backend_config.num_iterations 

            print(f"  Decode MLP time: {avg_time:.3f} ms")
            
            result = MLPBenchResult(
                quant_type=quant_type,
                num_token=num_token,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                avg_time_ms=avg_time,
                min_time_ms=avg_time,
                max_time_ms=avg_time,
                std_time_ms=0.0,
                num_iterations=backend_config.num_iterations,
                device=torch.cuda.get_device_name(backend_config.device),
                kernel_source='deepseek_v2_cuda_graph'
            )
            results.append(result)
            
            del mlp, input_tensor
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"  Decode test failed: {str(e)}")
            print(f"  Skipping this configuration...")
            continue
    
    return results

def main(output_path: str, base_config: BenchConfig):
    """Main function to run MLP benchmarks"""
    cleanup_distributed()
    
    print(f"Starting SGLang MLP Benchmark")
    print(f"Model path: {base_config.model_path}")
    print(f"Device: {torch.cuda.get_device_name()}")
    
    all_results = []
    test_cases = get_mlp_test_cases()
    print(f"Running {len(test_cases)} test cases...")
    
    grouped_cases = {}
    for test_case in test_cases:
        quant_type = test_case['quant_type']
        if quant_type not in grouped_cases:
            grouped_cases[quant_type] = []
        grouped_cases[quant_type].append(test_case)
    
    for quant_type, cases in grouped_cases.items():
        print(f"\n{'='*60}")
        print(f"TESTING: Quant Type={quant_type}")
        print(f"Test cases: {len(cases)}")
        print(f"{'='*60}")
        cleanup_distributed()
    
        backend_config = BenchConfig(
            quant_types=[quant_type],
            num_tokens=base_config.num_tokens,
            hidden_size=base_config.hidden_size,
            intermediate_size=base_config.intermediate_size,
            num_warmup=base_config.num_warmup,
            num_iterations=base_config.num_iterations,
            model_path=base_config.model_path,
            dtype=base_config.dtype,
            device=base_config.device,
            enable_profiler=base_config.enable_profiler
        )
        
        torch.cuda.empty_cache()
        initialize_distributed()
        
        print("\n=== Running Prefill Benchmarks ===")
        prefill_results = run_mlp_torch(cases, backend_config)
        all_results.extend(prefill_results)
        
        print("\n=== Running Decode Benchmarks (CUDA Graph) ===")
        decode_results = run_mlp_cuda_graph(cases, backend_config)
        all_results.extend(decode_results)

        cleanup_distributed()
        torch.cuda.empty_cache()
    
    output_results(all_results, output_path)
    
    print("\n" + "="*50)
    print("MLP BENCHMARK COMPLETED")
    print("="*50)
    print(f"Output files saved to:")
    print(f"  - Context results: {os.path.join(output_path, 'context_mlp_perf.txt')}")
    print(f"  - Generation results: {os.path.join(output_path, 'generation_mlp_perf.txt')}")
    print("="*50)

if __name__ == "__main__":
    output_path = "aiconfigurator/src/aiconfigurator/systems/data/h100_sxm/sglang/0.5.0/"
    model_path = DEEPSEEK_MODEL_PATH
    
    base_config = BenchConfig(
        quant_types=['fp8_block'],
        num_tokens=[],
        hidden_size=7168,
        intermediate_size=2048,
        num_warmup=3,
        num_iterations=10,
        model_path=model_path,
        dtype="auto",
        device="cuda:0",
        enable_profiler=False
    )
    
    main(output_path, base_config) 