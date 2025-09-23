#!/usr/bin/env python3
"""
Collect DeepSeek attention timing by temporarily modifying SGLang source code.

This script adds precise timing measurements to DeepseekV2AttentionMLA.forward() method
in the SGLang source code, runs benchmarks, then restores the original code.

Note: This uses DeepSeek V2 model which has SGLang native implementation with DeepseekV2AttentionMLA.
"""

import os
import time
import json
import torch
import argparse
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field
import logging

# SGLang imports
from sglang.srt.server_args import ServerArgs, PortArgs
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.hf_transformers_utils import get_tokenizer
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.utils import suppress_other_loggers, get_available_gpu_memory, BumpAllocator
from sglang.srt.distributed.parallel_state import destroy_distributed_environment
from sglang.srt.entrypoints.engine import _set_envs_and_config
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
import torch.distributed as dist
from torch.profiler import profile, ProfilerActivity, record_function

# Constants
DEEPSEEK_MODEL_PATH = os.environ.get("DEEPSEEK_MODEL_PATH", "/root/fac/deepseek-v3")
CUDA_LAUNCH_BLOCKING=1
TORCH_USE_CUDA_DSA=1

tp_size=1
num_head = 128/tp_size

def calculate_mla_attention_flops(
    b : int, 
    s=None, 
    l=None,
    is_decode=0,
    hidden_size = 7168,
    q_lora_rank = 1536,
    kv_lora_rank = 512,
    qk_rope_head_dim = 64,
    qk_nope_head_dim = 128, 
    v_head_dim = 128,
    num_head = num_head,
):

    
    if is_decode == 1:
        if l is None:
            raise ValueError("在decode模式(is_decode=1)下，参数l不能为None")


        term1 = 2 * hidden_size * (q_lora_rank + kv_lora_rank + qk_rope_head_dim ) * b
        term2 = 2 * q_lora_rank * num_head * (qk_rope_head_dim + qk_nope_head_dim) * b
        term3 = 2 * num_head  * qk_nope_head_dim * kv_lora_rank * b
        term4 = 2 * b * l * num_head * (qk_rope_head_dim + kv_lora_rank)
        term5 = 2 * b * num_head * kv_lora_rank * v_head_dim
        term6 = 2 * num_head * v_head_dim * hidden_size * b
        
        flops = term1 + term2 + term3 + term4 + term5 + term6
        
        return flops
        
    else:
        if s is None:
            raise ValueError("在prefill模式(is_decode=0)下，参数s不能为None")
        
        term1 = 2 * hidden_size * (q_lora_rank + kv_lora_rank  +qk_rope_head_dim) * b*s
        term2 = 2 * q_lora_rank * num_head * (qk_rope_head_dim + qk_nope_head_dim) * b*s
        term3 = 2 * kv_lora_rank * num_head * (qk_nope_head_dim + v_head_dim) * b*s

        term4 = 2 * num_head * (qk_nope_head_dim*2 + qk_rope_head_dim) * b * s**2
        term5 = 2 * num_head * v_head_dim * hidden_size * b * s

        flops = term1 + term2 + term3 + term4 + term5
        
        return flops


def calculate_decode(b, l):
    return calculate_mla_attention_flops(b, l=l, is_decode=1)


def calculate_prefill(b, s):
    return calculate_mla_attention_flops(b, s=s, is_decode=0)

logger = logging.getLogger(__name__)

@dataclass
class BenchConfig:
    # Prefill parameters
    batch_sizes: List[int] = field(default_factory=lambda: [1, 4, 8, 16])
    seq_lengths: List[int] = field(default_factory=lambda: [128, 512, 1024, 2048])
    
    # Decode parameters
    decode_batch_sizes: List[int] = field(default_factory=lambda: [1, 4, 8, 16, 32, 64])
    decode_kv_lengths: List[int] = field(default_factory=lambda: [128, 512, 1024, 2048, 4096])
    
    # Common parameters
    test_layer: int = 0
    num_warmup: int = 3
    num_iterations: int = 10
    model_path: str = DEEPSEEK_MODEL_PATH
    dtype: str = "auto"
    device: str = "cuda"
    enable_profiler: bool = False
    attention_backend: str = "auto"  # Available options: auto, fa3, flashinfer, aiter, triton, torch_native, flashmla, cutlass_mla, intel_amx

@dataclass
class BenchResult:
    """Result for a single benchmark run"""
    batch_size: int
    seq_length: int
    phase: str  # "prefill" or "decode" 
    avg_time_ms: float
    min_time_ms: float
    max_time_ms: float
    std_time_ms: float
    num_iterations: int = 10
    flops: Optional[float] = None  # Total FLOPs
    tflops_per_sec: Optional[float] = None  # Achieved TFLOPS/s


def get_attention_backend_info(model_runner: ModelRunner) -> str:
    """Get information about the attention backend being used"""
    try:
        # Try to get the actual attention backend being used
        if hasattr(model_runner, 'attn_backend'):
            backend_type = type(model_runner.attn_backend).__name__
            return f"{backend_type} (via {model_runner.server_args.attention_backend})"
        else:
            return "Unknown"
    except Exception:
        return "Unknown"


def validate_attention_backend(attention_backend: str, model_path: str) -> bool:
    """Validate if the attention backend is compatible with the model"""
    if attention_backend == "auto":
        return True
    
    # DeepSeek models are compatible with most attention backends
    if "deepseek" in model_path.lower():
        compatible_backends = ["fa3", "flashinfer", "aiter", "triton", "torch_native", "flashmla", "cutlass_mla"]
        if attention_backend not in compatible_backends:
            print(f"Warning: Attention backend '{attention_backend}' may not be optimal for DeepSeek models.")
            print(f"Recommended backends for DeepSeek: {', '.join(compatible_backends)}")
            return False
        return True
    
    return True


def print_attention_backend_info():
    """Print information about available attention backends"""
    print("\n" + "="*60)
    print("AVAILABLE ATTENTION BACKENDS")
    print("="*60)
    backends = {
        "auto": "Automatic selection based on model and hardware",
        "fa3": "Flash Attention 3 - Fast attention implementation",
        "flashinfer": "FlashInfer - Optimized attention for inference",
        "aiter": "AITemplate - Template-based attention optimization",
        "triton": "Triton - GPU kernel optimization framework",
        "torch_native": "PyTorch native attention implementation",
        "flashmla": "Flash MLA - Multi-Latent Attention optimization",
        "cutlass_mla": "CUTLASS MLA - NVIDIA CUTLASS-based MLA",
        "intel_amx": "Intel AMX - Intel Advanced Matrix Extensions"
    }
    
    for backend, description in backends.items():
        print(f"  {backend:<15}: {description}")
    print("="*60)


def getSMVersion():
    """Get SM version for compatibility with TensorRT-LLM interface"""
    # For SGLang, we'll use a default value since we don't need SM version checks
    return 89  # Default to H200-like SM version


def get_context_attention_test_cases():
    """Get test cases for context attention - only b and s dimensions"""
    test_cases = []
    b_list = [1,2,3,4,5,6,7, 8]
    s_list = [2048]
    b_list = [1]
    s_list = [128]

    for b in sorted(b_list):
        for s in sorted(s_list):
            # Simple memory limit check
            # if b * s > 131072*2:
            #     continue
            
            # Add test cases with only b and s
            test_cases.append([b, s])

    return test_cases


def get_generation_attention_test_cases():
    """Get test cases for generation attention - only b and s dimensions"""
    test_cases = []

    # Generation parameters - only b and s dimensions
    b_list = [128]
    s_list = [1024]

    # b_list = [8]
    # s_list = [4096]

    # Simple b and s combination test cases
    for b in sorted(b_list):
        for s in sorted(s_list):
            # Simple memory limit check
            if b * s > 8192 * 1024 * 4:
                continue
            
            test_cases.append([b, s])
    
    return test_cases


def get_attention_test_cases():
    """Legacy function for backward compatibility"""
    return get_context_attention_test_cases() + get_generation_attention_test_cases()


def load_model_runner(config: BenchConfig, tp_rank: int = 0) -> Tuple[ModelRunner, object, ServerArgs]:
    """Load model runner similar to bench_one_batch.py
    
    Note: You can use --load-format dummy to avoid loading real weights.
    You can also set environment variables to control the number of layers:
    - SGLANG_TEST_NUM_LAYERS=2  # Load only 2 layers
    - SGLANG_TEST_LAYER=0       # Test layer 0's attention module
    """
    suppress_other_loggers()
    
    # Check if we should load only a few layers
    num_layers = int(os.environ.get("SGLANG_TEST_NUM_LAYERS", "2"))
    load_format = os.environ.get("SGLANG_LOAD_FORMAT", "dummy")
    
    # Create server args
    server_args = ServerArgs(
        model_path=config.model_path,
        dtype=config.dtype,
        device=config.device,
        load_format=load_format,
        tp_size=1,
        trust_remote_code=True,
        mem_fraction_static=0.7,
        disable_radix_cache=True,
    )
    
    # Set attention backend if specified
    if config.attention_backend != "auto":
        server_args.attention_backend = config.attention_backend
        print(f"Using attention backend: {config.attention_backend}")
    
    # Override number of layers if specified
    if num_layers > 0 and load_format == "dummy":
        server_args.json_model_override_args = json.dumps({
            "num_hidden_layers": num_layers
        })
    
    # Set environment
    _set_envs_and_config(server_args)
    
    # Create port args
    port_args = PortArgs.init_new(server_args)
    
    # Create model config
    model_config = ModelConfig.from_server_args(server_args)
    
    # Create model runner
    model_runner = ModelRunner(
        model_config=model_config,
        mem_fraction_static=0.7,
        gpu_id=tp_rank,
        tp_rank=tp_rank,
        tp_size=server_args.tp_size,
        pp_rank=0,
        pp_size=1,
        moe_ep_rank=0,
        moe_ep_size=1,
        nccl_port=port_args.nccl_port,
        server_args=server_args,
    )
    
    # Get tokenizer
    tokenizer = get_tokenizer(
        server_args.tokenizer_path,
        tokenizer_mode=server_args.tokenizer_mode,
        trust_remote_code=server_args.trust_remote_code,
    )
    
    return model_runner, tokenizer, server_args


def benchmark_attention_module(
    model_runner: ModelRunner,
    config: BenchConfig
) -> List[BenchResult]:
    """Benchmark attention module directly"""
    
    results = []
    
    # Get the attention module from specified layer
    attention_module = model_runner.model.model.layers[config.test_layer].self_attn
    model_runner.req_to_token_pool.clear()
    model_runner.token_to_kv_pool_allocator.clear()
    print(f"\nBenchmarking attention module from layer {config.test_layer}")
    print(f"Module type: {type(attention_module).__name__}")
    
    # Extract MLA parameters from the attention module
    mla_params = {
        'hidden_size': attention_module.hidden_size,
        'num_heads': attention_module.num_heads,
        'qk_nope_head_dim': attention_module.qk_nope_head_dim,
        'qk_rope_head_dim': attention_module.qk_rope_head_dim,
        'v_head_dim': attention_module.v_head_dim,
        'q_lora_rank': attention_module.q_lora_rank,
        'kv_lora_rank': attention_module.kv_lora_rank,
    }
    
    print("\nMLA Architecture Parameters:")
    for k, v in mla_params.items():
        print(f"  {k}: {v}")
    
    # Test Prefill Phase
    print("\n" + "="*50)
    print("PREFILL PHASE BENCHMARKING")
    print("Using direct execution (no CUDA graph) for prefill testing...")
    print("="*50)
    
    for batch_size in config.batch_sizes:
        for seq_length in config.seq_lengths:
            print(f"\nPrefill: batch_size={batch_size}, seq_length={seq_length}")
            
            # Calculate FLOPs for this configuration
            flops = calculate_prefill(batch_size, seq_length)
            
            try:
                # Create requests for prefill
                reqs = []
                for i in range(batch_size):
                    req = Req(
                        rid=str(i),
                        origin_input_text="",
                        origin_input_ids=list(torch.randint(0, 10000, (seq_length,)).tolist()),
                        sampling_params=SamplingParams(temperature=0, max_new_tokens=1),
                    )
                    req.prefix_indices = []
                    req.fill_ids = req.origin_input_ids
                    req.extend_input_len = len(req.fill_ids)
                    req.logprob_start_len = 0
                    reqs.append(req)
                
                # Create batch for prefill
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
                model_worker_batch = batch.get_model_worker_batch()
                forward_batch = ForwardBatch.init_new(model_worker_batch, model_runner)
                
                # Initialize attention backend
                model_runner.attn_backend.init_forward_metadata(forward_batch)
                
                # Create input tensors
                hidden_states = torch.randn(
                    batch_size * seq_length, model_runner.model.config.hidden_size,
                    dtype=torch.bfloat16, device="cuda"
                )
                positions = torch.arange(seq_length, device="cuda").unsqueeze(0).expand(batch_size, -1).flatten()
                # Create BumpAllocator
                zero_allocator = BumpAllocator(
                    buffer_size=256,
                    dtype=torch.float32,
                    device="cuda"
                )
                
                # For prefill phase, always use direct execution (no CUDA graph)
                use_cuda_graph = False
                current_backend = get_attention_backend_info(model_runner)
                print(f"  Using direct execution for prefill phase with backend: {current_backend}")
                
                if use_cuda_graph:
                    # CUDA Graph capture for prefill
                    g = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g):
                        with torch.no_grad():
                            _ = attention_module(
                                positions=positions,
                                hidden_states=hidden_states,
                                forward_batch=forward_batch,
                                zero_allocator=zero_allocator
                            )
                    
                    # Warmup with CUDA graph
                    for _ in range(config.num_warmup):
                        g.replay()
                    
                    # Timing with CUDA graph
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    start_event.record()
                    for _ in range(config.num_iterations):
                        g.replay()
                    end_event.record()
                    torch.cuda.synchronize()
                    
                    # Calculate timing
                    avg_cuda_time = start_event.elapsed_time(end_event) / config.num_iterations
                    cuda_times = [avg_cuda_time] * config.num_iterations  # For compatibility with existing code
                else:
                    # Direct execution without CUDA graph
                    # Warmup
                    for _ in range(config.num_warmup):
                        with torch.no_grad():
                            _ = attention_module(
                                positions=positions,
                                hidden_states=hidden_states,
                                forward_batch=forward_batch,
                                zero_allocator=zero_allocator
                            )
                    
                    # Timing with direct execution
                    cuda_times = []
                    for _ in range(config.num_iterations):
                        start_event = torch.cuda.Event(enable_timing=True)
                        end_event = torch.cuda.Event(enable_timing=True)
                        start_event.record()
                        with torch.no_grad():
                            _ = attention_module(
                                positions=positions,
                                hidden_states=hidden_states,
                                forward_batch=forward_batch,
                                zero_allocator=zero_allocator
                            )
                        end_event.record()
                        torch.cuda.synchronize()
                        cuda_times.append(start_event.elapsed_time(end_event))
                    
                    avg_cuda_time = np.mean(cuda_times)
                
                # Profiler for detailed performance analysis (optional)
                if config.enable_profiler:
                    profiler_output_dir = "/root/fac/llm-pet/profiler_output"
                    try:
                        os.makedirs(profiler_output_dir, exist_ok=True)
                        profiler_trace_path = os.path.join(
                            profiler_output_dir, 
                            f"prefill_attention_b{batch_size}_s{seq_length}_layer{config.test_layer}"
                        )
                        
                        # Lightweight profiler - only CUDA activities, no memory/shape tracking
                        with profile(
                            activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU],  # Only CUDA, no CPU
                            record_shapes=False,  # Disable shape recording
                            profile_memory=False,  # Disable memory profiling
                            with_stack=False,     # Disable stack tracing
                            schedule=torch.profiler.schedule(wait=1, warmup=1, active=10, repeat=1)
                        ) as prof:
                            for iter_idx in range(config.num_iterations):
                                with record_function("attention_prefill"):
                                    if use_cuda_graph:
                                        g.replay()
                                    else:
                                        with torch.no_grad():
                                            _ = attention_module(
                                                positions=positions,
                                                hidden_states=hidden_states,
                                                forward_batch=forward_batch,
                                                zero_allocator=zero_allocator
                                            )
                                torch.cuda.synchronize()
                                prof.step()
                        
                        # Export lightweight profiler data
                        prof.export_chrome_trace(f"{profiler_trace_path}.json")
                        print(f"  Profiler trace saved: {profiler_trace_path}.json")
                        
                    except Exception as e:
                        print(f"  Warning: Profiler failed: {str(e)}")
                else:
                    print("  Profiler disabled")
                
                # Calculate performance metrics using CUDA Events timing
                avg_time_ms = np.mean(cuda_times)
                tflops_per_sec = (flops / 1e12) / (avg_time_ms / 1000)  # TFLOPS/s
                
                # Record results
                result = BenchResult(
                    phase="prefill",
                    batch_size=batch_size,
                    seq_length=seq_length,
                    avg_time_ms=avg_time_ms,
                    min_time_ms=np.min(cuda_times),
                    max_time_ms=np.max(cuda_times),
                    std_time_ms=np.std(cuda_times),
                    num_iterations=config.num_iterations,
                    flops=flops,
                    tflops_per_sec=tflops_per_sec
                )
                results.append(result)
                
                print(f"  Prefill attention time: {result.avg_time_ms:.3f} ms "
                        f"(min: {result.min_time_ms:.3f}, max: {result.max_time_ms:.3f}, std: {result.std_time_ms:.3f})")
                print(f"  FLOPs: {flops/1e9:.2f} GFLOPs, Performance: {tflops_per_sec:.2f} TFLOPS/s")
                
                # Clean up
                model_runner.req_to_token_pool.clear()
                model_runner.token_to_kv_pool_allocator.clear()
                del hidden_states, positions, forward_batch, batch
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"  Prefill test failed: {str(e)}")
                print(f"  Skipping this configuration...")
                continue
    
    # Test Decode Phase
    print("\n" + "="*50)
    print("DECODE PHASE BENCHMARKING")
    print("="*50)
    
    # For decode phase, always use CUDA graph for optimal performance
    print("Using CUDA graph for decode testing...")
    
    # Decode phase will use CUDA graph capture and replay
    
    # Use the same attention module
    attention_module_decode = attention_module
    
    for batch_size in config.decode_batch_sizes:
        for kv_length in config.decode_kv_lengths:
            print(f"\nDecode: batch_size={batch_size}, kv_cache_length={kv_length}")
            # Calculate FLOPs for decode
            flops = calculate_decode(batch_size, kv_length)
            
            try:
                    # First run a full forward pass to initialize KV cache properly
                    # Create requests with the KV cache length
                reqs = []
                for i in range(batch_size):
                    req = Req(
                        rid=str(i),
                        origin_input_text="",
                        origin_input_ids=list(torch.randint(0, 10000, (kv_length,)).tolist()),
                        sampling_params=SamplingParams(temperature=0, max_new_tokens=1),
                    )
                    req.prefix_indices = []
                    req.fill_ids = req.origin_input_ids
                    req.extend_input_len = len(req.fill_ids)
                    req.logprob_start_len = 0
                    req.cached_tokens = 0  
                    req.already_computed = 0  
                    reqs.append(req)
                
                # Initialize batch and run through model to populate KV cache
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
                batch.output_ids = kv_length
                batch.prepare_for_decode()
                model_worker_batch_decode = batch.get_model_worker_batch()
                forward_batch_decode = ForwardBatch.init_new(model_worker_batch_decode, model_runner)
                model_runner.attn_backend.init_forward_metadata(forward_batch_decode)
                
                # Create decode inputs - single token
                decode_hidden = torch.randn(
                    batch_size, model_runner.model.config.hidden_size,
                    dtype=torch.bfloat16, device="cuda"
                )
                decode_positions = torch.full((batch_size,), kv_length, device="cuda")
                
                # Create BumpAllocator
                zero_allocator = BumpAllocator(
                    buffer_size=2048,
                    dtype=torch.float32,
                    device="cuda"
                )
                # CUDA Graph capture for decode
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    with torch.no_grad():
                        _ = attention_module_decode(
                            positions=decode_positions,
                            hidden_states=decode_hidden,
                            forward_batch=forward_batch_decode,
                            zero_allocator=zero_allocator
                        )
                
                # Warmup with CUDA graph
                for _ in range(config.num_warmup):
                    g.replay()
                
                # Timing with CUDA graph
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                for _ in range(config.num_iterations):
                    g.replay()
                end_event.record()
                torch.cuda.synchronize()  
                
                # Calculate timing
                avg_cuda_time = start_event.elapsed_time(end_event) / config.num_iterations
                cuda_times = [avg_cuda_time] * config.num_iterations  # For compatibility with existing code
                
                
                # Profiler for detailed performance analysis (optional)
                if config.enable_profiler:
                    profiler_output_dir = "/root/fac/llm-pet/profiler_output"
                    try:
                        os.makedirs(profiler_output_dir, exist_ok=True)
                        profiler_trace_path = os.path.join(
                            profiler_output_dir, 
                            f"decode_attention_b{batch_size}_kv{kv_length}_layer{config.test_layer}"
                        )
                        
                        with profile(
                            activities=[ProfilerActivity.CUDA],
                            record_shapes=False,
                            profile_memory=False,
                            with_stack=False,
                            schedule=torch.profiler.schedule(wait=1, warmup=1, active=10, repeat=1)
                        ) as prof:
                            for iter_idx in range(config.num_iterations):
                                with record_function("attention_decode"):
                                    g.replay()  # Use CUDA graph replay instead of direct call
                                torch.cuda.synchronize()
                                prof.step()
                            
                        prof.export_chrome_trace(f"{profiler_trace_path}.json")
                        print(f"  Profiler trace saved: {profiler_trace_path}.json")
                        
                    except Exception as e:
                        print(f"  Warning: Profiler failed: {str(e)}")
                else:
                    print("  Profiler disabled")
                
                torch.cuda.empty_cache()
                # Calculate performance metrics using CUDA graph timing
                avg_time_ms = avg_cuda_time
                tflops_per_sec = (flops / 1e12) / (avg_time_ms / 1000)  # TFLOPS/s
                
                # Record results
                result = BenchResult(
                    phase="decode",
                    batch_size=batch_size,
                    seq_length=kv_length,  # For decode, seq_length represents KV cache length
                    avg_time_ms=avg_time_ms,
                    min_time_ms=avg_cuda_time,  # Single value from CUDA graph
                    max_time_ms=avg_cuda_time,  # Single value from CUDA graph
                    std_time_ms=0.0,           # No variation in CUDA graph timing
                    num_iterations=config.num_iterations,
                    flops=flops,
                    tflops_per_sec=tflops_per_sec
                )
                results.append(result)
                
                print(f"  Decode attention time: {result.avg_time_ms:.3f} ms "
                        f"(min: {result.min_time_ms:.3f}, max: {result.max_time_ms:.3f}, std: {result.std_time_ms:.3f})")
                print(f"  FLOPs: {flops/1e9:.2f} GFLOPs, Performance: {tflops_per_sec:.2f} TFLOPS/s")
                
                # Clean up
                model_runner.req_to_token_pool.clear()
                model_runner.token_to_kv_pool_allocator.clear()
                del decode_hidden, decode_positions, forward_batch_decode, batch
                torch.cuda.empty_cache()

                
            except Exception as e:
                print(f"  Decode test failed: {str(e)}")
                print(f"  Skipping this configuration...")
                continue
    
    return results


def save_results(results: List[BenchResult], output_path: str, attention_backend: str = "auto"):
    """Save results to JSON file in TRTLLM format"""
    
    # Group results by phase
    prefill_results = [r for r in results if r.phase == "prefill"]
    decode_results = [r for r in results if r.phase == "decode"]
    
    # Format for TRTLLM
    output_data = {
        "model": "deepseek_v3",
        "module": "attention",
        "attention_backend": attention_backend,
        "results": {
            "prefill": {},
            "decode": {}
        }
    }
    
    # Process prefill results
    for result in prefill_results:
        key = f"batch{result.batch_size}_seq{result.seq_length}"
        result_dict = {
            "avg_ms": result.avg_time_ms,
            "min_ms": result.min_time_ms,
            "max_ms": result.max_time_ms,
            "std_ms": result.std_time_ms
        }
        
        # Add FLOP metrics if available
        if result.flops is not None:
            result_dict.update({
                "flops": result.flops,
                "gflops": result.flops / 1e9,
                "tflops_per_sec": result.tflops_per_sec
            })
        
        output_data["results"]["prefill"][key] = result_dict
    
    # Process decode results  
    for result in decode_results:
        key = f"batch{result.batch_size}_seq{result.seq_length}"
        result_dict = {
            "avg_ms": result.avg_time_ms,
            "min_ms": result.min_time_ms,
            "max_ms": result.max_time_ms,
            "std_ms": result.std_time_ms
        }
        
        # Add FLOP metrics if available
        if result.flops is not None:
            result_dict.update({
                "flops": result.flops,
                "gflops": result.flops / 1e9,
                "tflops_per_sec": result.tflops_per_sec
            })
        
        output_data["results"]["decode"][key] = result_dict
    
    # Save to file
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nResults saved to {output_path}")
    
    # Print summary statistics
    print("\n" + "="*50)
    print("PERFORMANCE SUMMARY")
    print("="*50)
    
    if prefill_results:
        print("\nPrefill Performance:")
        for result in prefill_results:
            if result.flops is not None and result.tflops_per_sec is not None:
                print(f"  Batch {result.batch_size}, Seq {result.seq_length}: "
                      f"{result.tflops_per_sec:.2f} TFLOPS/s "
                      f"({result.flops/1e9:.2f} GFLOPs in {result.avg_time_ms:.3f} ms)")
            else:
                print(f"  Batch {result.batch_size}, Seq {result.seq_length}: "
                      f"{result.avg_time_ms:.3f} ms (FLOPs not calculated)")
    
    if decode_results:
        print("\nDecode Performance:")
        for result in decode_results:
            if result.flops is not None and result.tflops_per_sec is not None:
                print(f"  Batch {result.batch_size}, KV Cache {result.seq_length}: "
                      f"{result.tflops_per_sec:.2f} TFLOPS/s "
                      f"({result.flops/1e9:.2f} GFLOPs in {result.avg_time_ms:.3f} ms)")
            else:
                print(f"  Batch {result.batch_size}, KV Cache {result.seq_length}: "
                      f"{result.avg_time_ms:.3f} ms (FLOPs not calculated)")


def main():
    parser = argparse.ArgumentParser(description="Benchmark DeepSeek V3 attention module")
    
    # Common parameters
    parser.add_argument("--test-layer", type=int, 
                        default=int(os.environ.get("SGLANG_TEST_LAYER", "0")),
                        help="Which layer's attention module to test")
    parser.add_argument("--num-warmup", type=int, default=3,
                        help="Number of warmup iterations")
    parser.add_argument("--num-iterations", type=int, default=10,
                        help="Number of benchmark iterations")
    parser.add_argument("--output", type=str, default="/root/fac/llm-pet/attention_benchmark_results.json",
                        help="Output file for results")
    parser.add_argument("--model-path", type=str, default=DEEPSEEK_MODEL_PATH,
                        help="Model path or HuggingFace model ID")
    parser.add_argument("--dtype", type=str, default="auto",
                        help="Model dtype (auto, float16, bfloat16)")
    parser.add_argument("--attention-backend", type=str, default="flashinfer",
                        choices=["auto", "fa3", "flashinfer", "aiter", "triton", "torch_native", "flashmla", "cutlass_mla", "intel_amx"],
                        help="Attention backend to use")
    parser.add_argument("--quick-test", action="store_true",
                        help="Run quick test with small configurations")
    parser.add_argument("--prefill-only", action="store_true",
                        help="Only test prefill phase")
    parser.add_argument("--decode-only", action="store_true",
                        help="Only test decode phase")
    parser.add_argument("--list-backends", action="store_true",
                        help="List available attention backends and exit")
    
    args = parser.parse_args()
    
    # Show attention backend information if requested
    if args.list_backends:
        print_attention_backend_info()
        return
    
    # Validate attention backend compatibility
    if not validate_attention_backend(args.attention_backend, args.model_path):
        print("Continuing with the specified attention backend...")
    
    # Generate output filename with attention backend info
    if args.attention_backend != "auto":
        output_filename = f"/root/fac/llm-pet/attention_benchmark_results_bf8b_tp1tt_{args.attention_backend}.json"
    else:
        output_filename = args.output
    
    print(f"Loading model from {args.model_path}...")
    print(f"Using attention backend: {args.attention_backend}")
    print("\nTip: To test with dummy weights and limited layers, use:")
    print("  SGLANG_LOAD_FORMAT=dummy SGLANG_TEST_NUM_LAYERS=2 SGLANG_TEST_LAYER=0 python collect_attn.py")
    
    # Load model runner once
    base_config = BenchConfig(
        batch_sizes=[],  # Will be set per test case
        seq_lengths=[],  # Will be set per test case
        decode_batch_sizes=[],  # Will be set per test case
        decode_kv_lengths=[],  # Will be set per test case
        test_layer=args.test_layer,
        num_warmup=args.num_warmup,
        num_iterations=args.num_iterations,
        model_path=args.model_path,
        dtype=args.dtype,
        attention_backend=args.attention_backend,
        enable_profiler=False
    )
    
    try:
        # Load model runner once
        model_runner, tokenizer, server_args = load_model_runner(base_config)
        
        # Display attention backend information
        backend_info = get_attention_backend_info(model_runner)
        print(f"\nAttention Backend Information:")
        print(f"  Configured: {args.attention_backend}")
        print(f"  Actual: {backend_info}")
        
        all_results = []
        
        # Run context attention tests
        print("\n" + "="*50)
        print("CONTEXT ATTENTION TESTS")
        print("="*50)
        test_cases = get_context_attention_test_cases()
        print(f"Running {len(test_cases)} context attention test cases...")
        
        for test_case in test_cases:
            try:
                # Create config for this test case
                config = BenchConfig(
                    batch_sizes=[test_case[0]],
                    seq_lengths=[test_case[1]],
                    decode_batch_sizes=[],  # No decode for context tests
                    decode_kv_lengths=[],   # No decode for context tests
                    test_layer=args.test_layer,
                    num_warmup=args.num_warmup,
                    num_iterations=args.num_iterations,
                    model_path=args.model_path,
                    dtype=args.dtype,
                    attention_backend=args.attention_backend,
                    enable_profiler=True
                )
                
                # Benchmark the attention module
                results = benchmark_attention_module(model_runner, config)
                all_results.extend(results)
                
            except Exception as e:
                print(f"Error running context attention test case {test_case}: {e}")
                continue
        
        # Run generation attention tests
        print("\n" + "="*50)
        print("GENERATION ATTENTION TESTS")
        print("="*50)
        test_cases = get_generation_attention_test_cases()
        print(f"Running {len(test_cases)} generation attention test cases...")
        
        for test_case in test_cases:
            try:
                # Create config for this test case
                config = BenchConfig(
                    batch_sizes=[],  # No prefill for generation tests
                    seq_lengths=[],  # No prefill for generation tests
                    decode_batch_sizes=[test_case[0]],
                    decode_kv_lengths=[test_case[1]],
                    test_layer=args.test_layer,
                    num_warmup=args.num_warmup,
                    num_iterations=args.num_iterations,
                    model_path=args.model_path,
                    dtype=args.dtype,
                    attention_backend=args.attention_backend,
                    enable_profiler=True
                )
                
                # Benchmark the attention module
                results = benchmark_attention_module(model_runner, config)
                all_results.extend(results)
                
            except Exception as e:
                print(f"Error running generation attention test case {test_case}: {e}")
                continue
            
            # Save all results
        save_results(all_results, output_filename, args.attention_backend)
    
    finally:
        # Cleanup
        if dist.is_initialized():
            destroy_distributed_environment()
    
    print("\n" + "="*50)
    print("ALL TESTS COMPLETED")
    print("="*50)


if __name__ == "__main__":
    main() 