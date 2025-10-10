#!/usr/bin/env python3
import os
import time
import json
import torch
import numpy as np
from typing import List, Tuple
from dataclasses import dataclass, field
import logging

# SGLang imports
from sglang.srt.server_args import ServerArgs, PortArgs
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.utils import suppress_other_loggers, BumpAllocator
from sglang.srt.entrypoints.engine import _set_envs_and_config
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

from torch.profiler import profile, ProfilerActivity, record_function


# Constants
DEEPSEEK_MODEL_PATH = os.environ.get("DEEPSEEK_MODEL_PATH", "/deepseek-v3")
logger = logging.getLogger(__name__)

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

@dataclass
class BenchConfig:
    # Prefill parameters
    batch_sizes: List[int] = field(default_factory=lambda: [1, 4])
    seq_lengths: List[int] = field(default_factory=lambda: [128, 512])
    
    # Decode parameters
    decode_batch_sizes: List[int] = field(default_factory=lambda: [1, 4])
    decode_kv_lengths: List[int] = field(default_factory=lambda: [128, 512])
    
    # Common parameters
    test_layer: int = 0
    num_warmup: int = 3
    num_iterations: int = 10
    model_path: str = DEEPSEEK_MODEL_PATH
    dtype: str = "auto"
    device: str = "cuda"
    enable_profiler: bool = False
    attention_backend: str = "auto"  # Available options: auto, fa3, flashinfer, aiter, triton, torch_native, flashmla, cutlass_mla, intel_amx
    head_num: int = 128  

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
    attention_backend: str = "unknown"
    head_num: int = 128

def get_attention_test_cases():
    """Get test cases for attention benchmarking with batch_size, seq_length, attention_backend, and head_num"""
    test_cases = []
    
    context_batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    context_seq_lengths = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    
    generation_batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    generation_seq_lengths = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    
    attention_backends = ["flashinfer", "fa3"]
    head_nums = [128, 64, 32, 16]
    
    # Generate test cases
    for attention_backend in attention_backends:
        for head_num in head_nums:
            # Context (prefill) test cases
            for batch_size in sorted(context_batch_sizes):
                for seq_length in sorted(context_seq_lengths):
                    # Memory limit checks for context
                    if batch_size * seq_length > 1024 * 2048:
                        continue
                    # Add prefill test case
                    test_cases.append([batch_size, seq_length, attention_backend, head_num, True])
            
            # Generation (decode) test cases
            for batch_size in sorted(generation_batch_sizes):
                for seq_length in sorted(generation_seq_lengths):
                    # Memory limit checks for generation
                    if batch_size * seq_length > 1024 * 2048:  # More lenient for decode
                        continue
                    
                    # Add decode test case (using seq_length as kv_length)
                    test_cases.append([batch_size, seq_length, attention_backend, head_num, False])
    
    return test_cases

def load_model_runner(config: BenchConfig, tp_rank: int = 0) -> Tuple[ModelRunner, object, ServerArgs]:
    """Load model runner similar to bench_one_batch.py
    You can also set environment variables to control the number of layers:
    - SGLANG_TEST_NUM_LAYERS=2  # Load only 2 layers
    - SGLANG_TEST_LAYER=0       # Test layer 0's attention module
    """
    suppress_other_loggers()
    
    num_layers = int(os.environ.get("SGLANG_TEST_NUM_LAYERS", "2"))
    load_format = os.environ.get("SGLANG_LOAD_FORMAT", "dummy")
    
    server_args = ServerArgs(
        model_path=config.model_path,
        dtype=config.dtype,
        device=config.device,
        load_format=load_format,
        tp_size=1,
        trust_remote_code=True,
        mem_fraction_static=0.5,
        disable_radix_cache=True,
    )

    server_args.attention_backend = config.attention_backend
    print(f"Using attention backend: {config.attention_backend}")
    
    if num_layers > 0 and load_format == "dummy":
        override_args = {
            "num_hidden_layers": num_layers
        }
        if hasattr(config, 'head_num') and config.head_num != 128:
            override_args["num_attention_heads"] = config.head_num
        server_args.json_model_override_args = json.dumps(override_args)
    
    _set_envs_and_config(server_args)
    
    port_args = PortArgs.init_new(server_args)
    model_config = ModelConfig.from_server_args(server_args)
    model_runner = ModelRunner(
        model_config=model_config,
        mem_fraction_static=0.5,
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

    return model_runner

def run_attention_torch(
    model_runner: ModelRunner,
    cases: List,
    backend_config: BenchConfig
) -> List[BenchResult]:
    """Run prefill benchmark for attention module"""
    
    results = []
    attention_module = model_runner.model.model.layers[backend_config.test_layer].self_attn
    model_runner.req_to_token_pool.clear()
    model_runner.token_to_kv_pool_allocator.clear()

    for test_case in cases:
        batch_size, seq_length, attention_backend, head_num, is_prefill = test_case
        
        if is_prefill:
            print(f"\nPrefill: batch_size={batch_size}, seq_length={seq_length}")
            
            try:
                model_runner.req_to_token_pool.clear()
                model_runner.token_to_kv_pool_allocator.clear()
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
                
                batch = ScheduleBatch.init_new(
                    reqs=reqs,
                    req_to_token_pool=model_runner.req_to_token_pool,
                    token_to_kv_pool_allocator=model_runner.token_to_kv_pool_allocator,
                    tree_cache=None,
                    model_config=model_runner.model_config,
                    enable_overlap=False,
                    spec_algorithm=SpeculativeAlgorithm.NONE,
                    enable_custom_logit_processor=False  
                )
                batch.prepare_for_extend()
                model_worker_batch = batch.get_model_worker_batch()
                forward_batch = ForwardBatch.init_new(model_worker_batch, model_runner)
                
                model_runner.attn_backend.init_forward_metadata(forward_batch)
                
                hidden_states = torch.randn(
                    batch_size * seq_length, model_runner.model.config.hidden_size,
                    dtype=torch.bfloat16, device="cuda"
                )
                positions = torch.arange(seq_length, device="cuda").unsqueeze(0).expand(batch_size, -1).flatten()
                zero_allocator = BumpAllocator(
                    buffer_size=256,
                    dtype=torch.float32,
                    device="cuda"
                )
                
                use_cuda_graph = False
                
                for _ in range(backend_config.num_warmup):
                    with torch.no_grad():
                        _ = attention_module(
                            positions=positions,
                            hidden_states=hidden_states,
                            forward_batch=forward_batch,
                            zero_allocator=zero_allocator
                        )
                
                cuda_times = []
                for i in range(backend_config.num_iterations):
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
                    if i > 1:
                        cuda_times.append(start_event.elapsed_time(end_event))
                
                avg_cuda_time = np.mean(cuda_times)
                
                # Profiler for detailed performance analysis (optional)
                if backend_config.enable_profiler:
                    profiler_output_dir = "/aiconfigurator/profiler_output"
                    try:
                        os.makedirs(profiler_output_dir, exist_ok=True)
                        profiler_trace_path = os.path.join(
                            profiler_output_dir, 
                            f"prefill_attention_b{batch_size}_s{seq_length}_layer{backend_config.test_layer}"
                        )
                        
                        with profile(
                            activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU],  
                            record_shapes=True,  
                            profile_memory=True,  
                            with_stack=True,     
                            schedule=torch.profiler.schedule(wait=1, warmup=1, active=10, repeat=1)
                        ) as prof:
                            for iter_idx in range(backend_config.num_iterations):
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
                        
                        
                        prof.export_chrome_trace(f"{profiler_trace_path}.json")
                        print(f"  Profiler trace saved: {profiler_trace_path}.json")
                        
                    except Exception as e:
                        print(f"  Warning: Profiler failed: {str(e)}")
                
                avg_time_ms = np.mean(cuda_times)
                
                result = BenchResult(
                    phase="prefill",
                    batch_size=batch_size,
                    seq_length=seq_length,
                    avg_time_ms=avg_time_ms,
                    min_time_ms=np.min(cuda_times),
                    max_time_ms=np.max(cuda_times),
                    std_time_ms=np.std(cuda_times),
                    num_iterations=backend_config.num_iterations,
                    attention_backend=backend_config.attention_backend,
                    head_num=backend_config.head_num
                )
                results.append(result)
                
                print(f"  Prefill attention time: {result.avg_time_ms:.3f} ms "
                        f"(min: {result.min_time_ms:.3f}, max: {result.max_time_ms:.3f}, std: {result.std_time_ms:.3f})")
                
                
                model_runner.req_to_token_pool.clear()
                model_runner.token_to_kv_pool_allocator.clear()
                del hidden_states, positions, forward_batch, batch
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"  Prefill test failed: {str(e)}")
                print(f"  Skipping this configuration...")
                continue
        
        else:  # decode phase
            print(f"\nDecode: batch_size={batch_size}, kv_cache_length={seq_length}")
            
            try:
                model_runner.req_to_token_pool.clear()
                model_runner.token_to_kv_pool_allocator.clear()
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
                    req.cached_tokens = 0  
                    req.already_computed = 0  
                    reqs.append(req)
                batch = ScheduleBatch.init_new(
                    reqs=reqs,
                    req_to_token_pool=model_runner.req_to_token_pool,
                    token_to_kv_pool_allocator=model_runner.token_to_kv_pool_allocator,
                    tree_cache=None,
                    model_config=model_runner.model_config,
                    enable_overlap=False,
                    spec_algorithm=SpeculativeAlgorithm.NONE,
                    enable_custom_logit_processor=False  
                )
                batch.prepare_for_extend()
                batch.output_ids = seq_length
                batch.prepare_for_decode()
                model_worker_batch_decode = batch.get_model_worker_batch()
                forward_batch_decode = ForwardBatch.init_new(model_worker_batch_decode, model_runner)
                model_runner.attn_backend.init_forward_metadata(forward_batch_decode)
                decode_hidden = torch.randn(
                    batch_size, model_runner.model.config.hidden_size,
                    dtype=torch.bfloat16, device="cuda"
                )
                decode_positions = torch.full((batch_size,), seq_length, device="cuda")
                zero_allocator = BumpAllocator(
                    buffer_size=2048,
                    dtype=torch.float32,
                    device="cuda"
                )
                
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    with torch.no_grad():
                        _ = attention_module(
                            positions=decode_positions,
                            hidden_states=decode_hidden,
                            forward_batch=forward_batch_decode,
                            zero_allocator=zero_allocator
                        )
    
                for _ in range(backend_config.num_warmup):
                    g.replay()
                
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                for _ in range(backend_config.num_iterations):
                    g.replay()
                end_event.record()
                torch.cuda.synchronize()  
                
                avg_cuda_time = start_event.elapsed_time(end_event) / backend_config.num_iterations
                cuda_times = [avg_cuda_time] * backend_config.num_iterations  # For compatibility with existing code
 
                if backend_config.enable_profiler:
                    profiler_output_dir = "/aiconfigurator/profiler_output"
                    try:
                        os.makedirs(profiler_output_dir, exist_ok=True)
                        profiler_trace_path = os.path.join(
                            profiler_output_dir, 
                            f"decode_attention_b{batch_size}_kv{seq_length}_layer{backend_config.test_layer}"
                        )
                        
                        with profile(
                            activities=[ProfilerActivity.CUDA],
                            record_shapes=False,
                            profile_memory=False,
                            with_stack=False,
                            schedule=torch.profiler.schedule(wait=1, warmup=1, active=10, repeat=1)
                        ) as prof:
                            for iter_idx in range(backend_config.num_iterations):
                                with record_function("attention_decode"):
                                    g.replay()  
                                torch.cuda.synchronize()
                                prof.step()
                            
                        prof.export_chrome_trace(f"{profiler_trace_path}.json")
                        print(f"  Profiler trace saved: {profiler_trace_path}.json")
                        
                    except Exception as e:
                        print(f"  Warning: Profiler failed: {str(e)}")

                torch.cuda.empty_cache()
                avg_time_ms = avg_cuda_time
                
                result = BenchResult(
                    phase="decode",
                    batch_size=batch_size,
                    seq_length=seq_length,  
                    avg_time_ms=avg_time_ms,
                    min_time_ms=avg_cuda_time,  
                    max_time_ms=avg_cuda_time,  
                    std_time_ms=0.0,           
                    num_iterations=backend_config.num_iterations,
                    attention_backend=backend_config.attention_backend,
                    head_num=backend_config.head_num
                )
                results.append(result)
                
                print(f"  Decode attention time: {result.avg_time_ms:.3f} ms "
                        f"(min: {result.min_time_ms:.3f}, max: {result.max_time_ms:.3f}, std: {result.std_time_ms:.3f})")
                
                model_runner.req_to_token_pool.clear()
                model_runner.token_to_kv_pool_allocator.clear()
                del decode_hidden, decode_positions, forward_batch_decode, batch
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"  Decode test failed: {str(e)}")
                print(f"  Skipping this configuration...")
                continue
    
    return results

def output_results(results: List[BenchResult], output_path: str):
    """Save results to separate CSV files for prefill and decode phases"""
    
    prefill_results = [r for r in results if r.phase == "prefill"]
    decode_results = [r for r in results if r.phase == "decode"]
    os.makedirs(output_path, exist_ok=True)
    context_output_path = os.path.join(output_path, "context_mla_perf.txt")
    generation_output_path = os.path.join(output_path, "generation_mla_perf.txt")
    
    if prefill_results:
        file_exists = os.path.exists(context_output_path)
        with open(context_output_path, 'a' if file_exists else 'w') as f:
            if not file_exists:
                f.write("framework,version,device,op_name,kernel_source,mla_dtype,kv_cache_dtype,num_heads,batch_size,isl,tp_size,step,latency\n")
    
            for result in prefill_results:
                attention_backend = getattr(result, 'attention_backend', 'unknown')
                num_heads = getattr(result, 'head_num', 128)
                op_name = "mla_context"
                isl = result.seq_length
                step = 0
                mla_dtype = "fp8_block"
                kv_cache_dtype = "fp8"
                tp_size = 1 
                
                f.write(f"SGLang,1.0.0,NVIDIA H20-3e,{op_name},{attention_backend},{mla_dtype},{kv_cache_dtype},{num_heads},{result.batch_size},{isl},{tp_size},{step},{result.avg_time_ms}\n")
        
        if file_exists:
            print(f"\nPrefill results appended to {context_output_path}")
        else:
            print(f"\nPrefill results saved to {context_output_path}")
    
    if decode_results:
        file_exists = os.path.exists(generation_output_path)
        
        with open(generation_output_path, 'a' if file_exists else 'w') as f:
            if not file_exists:
                f.write("framework,version,device,op_name,kernel_source,mla_dtype,kv_cache_dtype,num_heads,batch_size,isl,tp_size,step,latency\n")

            for result in decode_results:
                attention_backend = getattr(result, 'attention_backend', 'unknown')
                num_heads = getattr(result, 'head_num', 128)
                op_name = "mla_generation" 
                isl = result.seq_length
                step = 0
                mla_dtype = "fp8_block"
                kv_cache_dtype = "fp8"
                tp_size = 1  
                
                f.write(f"SGLang,1.0.0,NVIDIA H20-3e,{op_name},{attention_backend},{mla_dtype},{kv_cache_dtype},{num_heads},{result.batch_size},{isl},{tp_size},{step},{result.avg_time_ms}\n")
        
        if file_exists:
            print(f"\nDecode results appended to {generation_output_path}")
        else:
            print(f"\nDecode results saved to {generation_output_path}")
    
    
    if prefill_results:
        print("\nPrefill Performance Summary:")
        for result in prefill_results:
            attention_backend = getattr(result, 'attention_backend', 'unknown')
            head_num = getattr(result, 'head_num', 128)
            print(f"  Backend: {attention_backend}, Heads: {head_num}, Batch: {result.batch_size}, Seq: {result.seq_length}: "
                  f"{result.avg_time_ms:.3f} ms")
    
    if decode_results:
        print("\nDecode Performance Summary:")
        for result in decode_results:
            attention_backend = getattr(result, 'attention_backend', 'unknown')
            head_num = getattr(result, 'head_num', 128)
            print(f"  Backend: {attention_backend}, Heads: {head_num}, Batch: {result.batch_size}, KV Cache: {result.seq_length}: "
                  f"{result.avg_time_ms:.3f} ms")

def main():
    output_path = "/aiconfigurator/src/aiconfigurator/systems/data/h100_sxm/sglang/0.5.0/"
    model_path = DEEPSEEK_MODEL_PATH
    
    cleanup_distributed()
    
    print(f"Loading model from {model_path}...")
    print("\nTip: To test with dummy weights and limited layers, use:")
    print("  SGLANG_LOAD_FORMAT=dummy SGLANG_TEST_NUM_LAYERS=2 SGLANG_TEST_LAYER=0 python collect_attn.py")
    
    base_config = BenchConfig(
        batch_sizes=[],  
        seq_lengths=[],  
        decode_batch_sizes=[],  
        decode_kv_lengths=[],  
        test_layer=0,
        num_warmup=3,
        num_iterations=10,
        model_path=model_path,
        dtype="auto",
        attention_backend="flashinfer",
        enable_profiler=False,
        head_num=128  # Default head num, will be overridden in benchmark function
    )

    all_results = []

    test_cases = get_attention_test_cases()
    print(f"Running {len(test_cases)} test cases...")

    grouped_cases = {}
    for test_case in test_cases:
        batch_size, seq_length, attention_backend, head_num, is_prefill = test_case
        key = (attention_backend, head_num)
        if key not in grouped_cases:
            grouped_cases[key] = []
        grouped_cases[key].append(test_case)
    
    for (attention_backend, head_num), cases in grouped_cases.items():
        print(f"\n{'='*60}")
        print(f"TESTING: Attention Backend={attention_backend}, Head Num={head_num}")
        print(f"Test cases: {len(cases)}")
        print(f"{'='*60}")
        cleanup_distributed()

        backend_config = BenchConfig(
            batch_sizes=[],  
            seq_lengths=[],  
            decode_batch_sizes=[],  
            decode_kv_lengths=[],  
            test_layer=base_config.test_layer,
            num_warmup=base_config.num_warmup,
            num_iterations=base_config.num_iterations,
            model_path=base_config.model_path,
            dtype=base_config.dtype,
            attention_backend=attention_backend,
            enable_profiler=base_config.enable_profiler,
            head_num=head_num
        )
        torch.cuda.empty_cache()
        model_runner = load_model_runner(backend_config)

        results = run_attention_torch(model_runner, cases, backend_config)
        all_results.extend(results)

        del model_runner
        cleanup_distributed()
        torch.cuda.empty_cache()
    
    output_results(all_results, output_path)
    
    print("\n" + "="*50)
    print("ALL TESTS COMPLETED")
    print("="*50)

if __name__ == "__main__":
    main()